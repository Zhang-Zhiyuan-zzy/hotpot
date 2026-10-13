"""Small public facade for governed dataset assessment."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

from ..database import DatasetStore
from ..levels import EvidenceLevel, QualityLevel
from ..models import DatasetBundle, DatasetStandardError, canonical_json
from ..quality import (
    QualityCertificate,
    QualityCheck,
    QualityGateError,
    assess_bundle,
)
from .inputs import Submission, SubmissionFenceError, load_submission


CONFIGURE_VERSION = "hotpot-ds-certificate-config/current"
CERTIFICATE_ENVELOPE_VERSION = "hotpot-ds-public-certificate/current"


class _Unset:
    def __repr__(self) -> str:
        return "UNSET"


UNSET = _Unset()


DEFAULT_GENERIC_RULES: Dict[str, Any] = {
    "min_records": 1,
    "require_structure_or_label": True,
    "max_primary_structures": 1,
}

# These checks are implemented by the normalized model, assessor, and store.
# A dataset rule may add stricter requirements, but it may not turn them off.
IMMUTABLE_GATES = frozenset(
    {
        "record_standard",
        "identity_uniqueness",
        "relation_uniqueness",
        "referential_integrity",
        "record_entity_ownership",
        "asset_integrity",
        "certificate_integrity",
    }
)


def _copy_json(value: Mapping[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(value))


def _validate_rule_map(rules: Mapping[str, Any], context: str) -> Dict[str, Any]:
    if not isinstance(rules, Mapping):
        raise TypeError(f"{context} must be a mapping")
    copied = _copy_json(rules)
    disabled = sorted(
        gate for gate in IMMUTABLE_GATES if copied.get(gate) is False
    )
    if disabled:
        raise ValueError(
            f"{context} cannot disable immutable quality gates: {disabled}"
        )
    return copied


@dataclass(frozen=True)
class Configure:
    """Composable general and dataset-specific certification requirements.

    Dataset-specific keys replace general keys.  The immutable integrity gates
    remain mandatory regardless of configuration.
    """

    generic: Mapping[str, Any] = field(default_factory=dict)
    datasets: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        generic = _validate_rule_map(self.generic, "generic rules")
        if not isinstance(self.datasets, Mapping):
            raise TypeError("datasets rules must be a mapping")
        datasets = {
            str(name): _validate_rule_map(rules, f"dataset rules for {name!r}")
            for name, rules in self.datasets.items()
        }
        object.__setattr__(self, "generic", generic)
        object.__setattr__(self, "datasets", datasets)

    def effective(self, ds_name: str) -> Dict[str, Any]:
        rules = {**DEFAULT_GENERIC_RULES, **self.generic}
        rules.update(self.datasets.get(ds_name, {}))
        return rules

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": CONFIGURE_VERSION,
            "generic": _copy_json(self.generic),
            "datasets": _copy_json(self.datasets),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Configure":
        if not isinstance(value, Mapping):
            raise TypeError("certificate configuration must be a mapping")
        expected = {"schema_version", "generic", "datasets"}
        if set(value) != expected:
            raise ValueError(
                "certificate configuration fields must be exactly "
                f"{sorted(expected)}"
            )
        if value["schema_version"] != CONFIGURE_VERSION:
            raise ValueError(
                f"unsupported certificate configuration: {value['schema_version']!r}"
            )
        return cls(generic=value["generic"], datasets=value["datasets"])

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "Configure":
        with Path(path).open(encoding="utf-8") as handle:
            value = json.load(handle)
        return cls.from_dict(value)


def _normalise_quality_level(value: Any) -> str:
    if isinstance(value, QualityLevel):
        return value.value
    text = str(value)
    short = {level.name: level.value for level in QualityLevel}
    if text in short:
        return short[text]
    return QualityLevel(text).value


def _sequence(value: Any, name: str) -> Tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a list or tuple; use [] to declare none")
    return tuple(str(item) for item in value)


def _apply_declarations(
    submission: Submission,
    ds_name: str,
    quality_level: Any = UNSET,
    core_labels: Any = UNSET,
    core_info: Any = UNSET,
) -> Submission:
    bundle = submission.bundle
    if len(bundle.datasets) != 1 or bundle.datasets[0].dataset_id != ds_name:
        names = [dataset.dataset_id for dataset in bundle.datasets]
        raise SubmissionFenceError(
            f"submission dataset identity {names!r} does not match ds_name={ds_name!r}"
        )
    if any(release.dataset_id != ds_name for release in bundle.releases):
        raise SubmissionFenceError("all submitted releases must belong to ds_name")

    level = UNSET if quality_level is UNSET else _normalise_quality_level(quality_level)
    labels = UNSET if core_labels is UNSET else _sequence(core_labels, "core_labels")
    info = UNSET if core_info is UNSET else _sequence(core_info, "core_info")
    releases = []
    for release in bundle.releases:
        quality = dict(release.quality)
        if labels is not UNSET:
            quality["core_labels"] = list(labels)
        if info is not UNSET:
            quality["core_info"] = list(info)
        releases.append(
            replace(
                release,
                quality_level=release.quality_level if level is UNSET else level,
                quality=quality,
            )
        )
    updated = replace(bundle, releases=tuple(releases))
    return replace(submission, bundle=updated)


def _record_facts(bundle: DatasetBundle) -> Dict[str, Dict[str, Any]]:
    facts = {
        record.record_id: {
            "local_id": record.local_id,
            "metadata": record.metadata,
            "labels": set(),
            "identifiers": {},
            "structures": set(),
        }
        for record in bundle.records
    }
    record_entities: Dict[str, set] = {}
    entity_records: Dict[str, set] = {}
    for link in bundle.record_entities:
        record_entities.setdefault(link.record_id, set()).add(link.entity_id)
        entity_records.setdefault(link.entity_id, set()).add(link.record_id)
    for observation in bundle.observations:
        if observation.record_id in facts:
            facts[observation.record_id]["labels"].add(observation.property_type_id)
    for identifier in bundle.external_identifiers:
        for record_id in entity_records.get(identifier.entity_id, ()):
            facts[record_id]["identifiers"].setdefault(identifier.namespace, []).append(
                identifier.value
            )
    structure_entities = {structure.entity_id: structure.structure_id for structure in bundle.structures}
    for record_id, entity_ids in record_entities.items():
        for entity_id in entity_ids:
            if entity_id in structure_entities:
                facts[record_id]["structures"].add(structure_entities[entity_id])
    return facts


def _path_present(fact: Mapping[str, Any], path: str) -> bool:
    parts = path.split(".")
    current: Any = fact
    for part in parts:
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            return False
    return current is not None and current != "" and current != [] and current != {} and current != set()


def _configured_issues(bundle: DatasetBundle, rules: Mapping[str, Any]) -> list:
    issues = []
    facts = _record_facts(bundle)
    min_records = int(rules.get("min_records", 0))
    if len(bundle.records) < min_records:
        issues.append(
            f"record count {len(bundle.records)} is below configured minimum {min_records}"
        )

    core_labels = tuple(str(item) for item in rules.get("core_labels", ()))
    core_info = tuple(str(item) for item in rules.get("core_info", ()))
    for record_id, fact in facts.items():
        for label in core_labels:
            if label not in fact["labels"]:
                issues.append(f"{record_id}: missing core label {label}")
        for path in core_info:
            if not _path_present(fact, path):
                issues.append(f"{record_id}: missing core info {path}")
        if rules.get("require_structure_or_label", False):
            if not fact["structures"] and not fact["labels"]:
                issues.append(f"{record_id}: has neither a structure nor a label")

    allowed_formats = rules.get("allowed_structure_formats")
    if allowed_formats is not None:
        allowed = {str(item).lower() for item in allowed_formats}
        for link in bundle.structure_assets:
            if link.role.lower() not in allowed:
                issues.append(
                    f"structure {link.structure_id}: format {link.role!r} is not allowed"
                )

    minimum_evidence = rules.get("minimum_evidence", {})
    if minimum_evidence:
        ranks = {level.value: index for index, level in enumerate(EvidenceLevel)}
        for observation in bundle.observations:
            required = minimum_evidence.get(observation.property_type_id)
            if required is None:
                continue
            required_value = EvidenceLevel(required).value
            if ranks[observation.evidence_level] < ranks[required_value]:
                issues.append(
                    f"{observation.observation_id}: {observation.evidence_level} is below "
                    f"configured minimum {required_value}"
                )
    maximum_primary = rules.get("max_primary_structures")
    if maximum_primary is not None:
        primary_by_record: Dict[str, int] = {}
        for link in bundle.record_entities:
            if link.role.startswith("primary."):
                primary_by_record[link.record_id] = primary_by_record.get(link.record_id, 0) + 1
        for record_id, count in primary_by_record.items():
            if count > int(maximum_primary):
                issues.append(
                    f"{record_id}: has {count} primary structures; configured maximum is {maximum_primary}"
                )
    return issues


def _issue_with_configured_failure(
    base: QualityCertificate, issues: Sequence[str]
) -> QualityCertificate:
    if not issues:
        return base
    check = QualityCheck(
        "configured_requirements",
        False,
        len(issues),
        len(issues),
        tuple(issues[:25]),
    )
    return QualityCertificate.issue(
        profile=base.profile,
        subject_digest=base.subject_digest,
        checks=(*base.checks, check),
        assessed_levels=base.assessed_levels,
        counts=base.counts,
    )


def _source_digest(source: Any) -> str:
    digest = hashlib.sha256()
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_file():
            digest.update(path.read_bytes())
        elif path.is_dir():
            for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
                digest.update(item.relative_to(path).as_posix().encode())
                digest.update(item.read_bytes())
        else:
            digest.update(f"missing:{path}".encode())
    else:
        digest.update(repr(source).encode())
    return digest.hexdigest()


def _failed_fence_certificate(source: Any, error: Exception) -> QualityCertificate:
    return QualityCertificate.issue(
        profile="admission",
        subject_digest=_source_digest(source),
        checks=(
            QualityCheck(
                "submission_fence",
                False,
                1,
                1,
                (f"{type(error).__name__}: {error}",),
            ),
        ),
        assessed_levels={},
        counts={},
    )


def _source_kind(source: Any) -> str:
    if isinstance(source, DatasetBundle):
        return "bundle"
    if isinstance(source, Mapping):
        return "manifest"
    path = Path(source)
    if path.is_dir():
        return "folder"
    if path.suffix.lower() in {".sqlite", ".sqlite3", ".db"}:
        return "sqlite"
    return "json"


def _public_established_spec(spec: Mapping[str, Any]) -> Dict[str, Any]:
    policy = spec.get("policy")
    return {
        "title": spec["title"],
        "description": spec["description"],
        "versions": sorted(spec["versions"]),
        "core_labels": list(spec["core_labels"]),
        "core_info": list(spec["core_info"]),
        "quality_level": spec["quality_level"],
        "release_id": spec["release_id"],
        "policy_revision": policy["revision"] if policy is not None else None,
        "policy_sha256": policy["config_sha256"] if policy is not None else None,
    }


def _store_path(store: Any) -> Path:
    if isinstance(store, (str, Path)):
        return Path(store)
    if hasattr(store, "path"):
        return Path(store.path)
    if hasattr(store, "backend") and hasattr(store.backend, "path"):
        return Path(store.backend.path)
    if hasattr(store, "_backend") and hasattr(store._backend, "path"):
        return Path(store._backend.path)
    raise TypeError("store must be a database path, Store, or DatasetStore")


def _store_backend(store: Any) -> DatasetStore:
    if isinstance(store, DatasetStore):
        return store
    if hasattr(store, "backend") and isinstance(store.backend, DatasetStore):
        return store.backend
    if hasattr(store, "_backend") and isinstance(store._backend, DatasetStore):
        return store._backend
    return DatasetStore(_store_path(store))


def _established_spec(store: Any, ds_name: str) -> Dict[str, Any]:
    path = _store_path(store)
    if not path.is_file():
        raise ValueError(f"dataset store does not exist: {path}")
    connection = sqlite3.connect(str(path))
    connection.row_factory = sqlite3.Row
    try:
        dataset = connection.execute(
            "SELECT dataset_id,title,description FROM datasets WHERE dataset_id=?",
            (ds_name,),
        ).fetchone()
        if dataset is None:
            raise ValueError(f"established dataset does not exist: {ds_name!r}")
        release = connection.execute(
            "SELECT release_id,version,quality_level,core_labels_json,quality_json "
            "FROM dataset_releases "
            "WHERE dataset_id=? ORDER BY release_serial DESC LIMIT 1",
            (ds_name,),
        ).fetchone()
        quality = json.loads(release["quality_json"]) if release else {}
        core_labels = json.loads(release["core_labels_json"]) if release else []
        if not isinstance(core_labels, list):
            raise ValueError("stored core_labels_json must be a JSON array")
        policy = _store_backend(store).get_dataset_policy(ds_name)
        return {
            "title": dataset["title"],
            "description": dataset["description"],
            "versions": {
                row[0]
                for row in connection.execute(
                    "SELECT version FROM dataset_releases WHERE dataset_id=?", (ds_name,)
                )
            },
            "core_labels": list(core_labels or []),
            "core_info": list(quality.get("core_info", [])),
            "quality_level": release["quality_level"] if release else None,
            "release_id": release["release_id"] if release else None,
            "property_types": {
                row["property_type_id"]: tuple(row)
                for row in connection.execute(
                    "SELECT property_type_id,name,value_kind,canonical_unit,schema_version,metadata_json "
                    "FROM property_types"
                )
            },
            "policy": policy,
        }
    except sqlite3.DatabaseError as error:
        raise ValueError(f"cannot read established dataset specification: {error}") from error
    finally:
        connection.close()


def _existing_issues(bundle: DatasetBundle, ds_name: str, spec: Mapping[str, Any]) -> list:
    issues = []
    if len(bundle.datasets) != 1 or bundle.datasets[0].dataset_id != ds_name:
        issues.append(f"submission must contain only established dataset {ds_name!r}")
        return issues
    dataset = bundle.datasets[0]
    if dataset.title != spec["title"] or dataset.description != spec["description"]:
        issues.append("dataset title and description differ from the immutable catalog entry")
    for release in bundle.releases:
        if release.version in spec["versions"]:
            issues.append(
                f"release version {release.version!r} already exists; integration requires a new immutable release"
            )
    for definition in bundle.property_types:
        existing = spec["property_types"].get(definition.property_type_id)
        if existing is None:
            continue
        incoming = (
            definition.property_type_id,
            definition.name,
            definition.value_kind,
            definition.canonical_unit,
            definition.schema_version,
            json.dumps(definition.metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        )
        if incoming != existing:
            issues.append(
                f"property type {definition.property_type_id!r} conflicts with its immutable stored definition"
            )
    return issues


def _assessment_configuration(
    configuration: Configure,
    ds_name: str,
    core_labels: Any,
    core_info: Any,
) -> Configure:
    dataset_rules = dict(configuration.datasets.get(ds_name, {}))
    if core_labels is not UNSET:
        dataset_rules["core_labels"] = list(_sequence(core_labels, "core_labels"))
    if core_info is not UNSET:
        dataset_rules["core_info"] = list(_sequence(core_info, "core_info"))
    return Configure(
        generic=configuration.generic,
        datasets={ds_name: dataset_rules},
    )


def _bundle_core_declaration(bundle: DatasetBundle, field: str) -> Any:
    declared = {
        tuple(release.quality[field])
        for release in bundle.releases
        if field in release.quality
    }
    if len(declared) > 1:
        raise SubmissionFenceError(
            f"all releases in one assessment must declare the same {field}"
        )
    return next(iter(declared)) if declared else UNSET


class Certificate:
    """A public certificate plus its prepared submission when assessed in memory."""

    Configure = Configure

    def __init__(
        self,
        certificate: QualityCertificate,
        submission: Optional[Submission] = None,
        configure: Optional[Configure] = None,
        *,
        ds_name: Optional[str] = None,
        mode: Optional[str] = None,
        declarations: Optional[Mapping[str, Any]] = None,
        established_spec: Optional[Mapping[str, Any]] = None,
        source_kind: Optional[str] = None,
        configuration_sha256: Optional[str] = None,
        envelope_sha256: Optional[str] = None,
    ):
        self._certificate = certificate
        self.submission = submission
        self.configure = configure
        self.ds_name = ds_name
        self.mode = mode
        self.declarations = dict(declarations or {})
        self.established_spec = dict(established_spec or {})
        self.source_kind = source_kind
        expected_digest = self._configuration_digest() if self.facade_attested else None
        self.configuration_sha256 = configuration_sha256 or expected_digest
        if configuration_sha256 is not None and configuration_sha256 != expected_digest:
            raise QualityGateError("certificate configuration integrity check failed")
        expected_envelope_digest = self._envelope_digest() if self.facade_attested else None
        self.envelope_sha256 = envelope_sha256 or expected_envelope_digest
        if envelope_sha256 is not None and envelope_sha256 != expected_envelope_digest:
            raise QualityGateError("public certificate envelope integrity check failed")

    @property
    def facade_attested(self) -> bool:
        return (
            self.configure is not None
            and self.ds_name is not None
            and self.mode in {"new", "existing"}
        )

    def _configuration_payload(self) -> Dict[str, Any]:
        return {
            "configuration": self.configure.to_dict() if self.configure is not None else None,
            "dataset": self.ds_name,
            "mode": self.mode,
            "declarations": self.declarations,
            "established_spec": self.established_spec,
            "source_kind": self.source_kind,
        }

    def _configuration_digest(self) -> Optional[str]:
        if not self.facade_attested:
            return None
        return hashlib.sha256(
            canonical_json(self.configure.to_dict()).encode()
        ).hexdigest()

    def _unsigned_envelope(self) -> Dict[str, Any]:
        return {
            "schema_version": CERTIFICATE_ENVELOPE_VERSION,
            "certificate": self._certificate.to_dict(),
            **self._configuration_payload(),
            "configuration_sha256": self.configuration_sha256,
        }

    def _envelope_digest(self) -> Optional[str]:
        if not self.facade_attested:
            return None
        return hashlib.sha256(canonical_json(self._unsigned_envelope()).encode()).hexdigest()

    @property
    def raw(self) -> QualityCertificate:
        return self._certificate

    def __getattr__(self, name: str) -> Any:
        return getattr(self._certificate, name)

    def to_dict(self) -> Dict[str, Any]:
        if not self.facade_attested:
            return self._certificate.to_dict()
        return {**self._unsigned_envelope(), "envelope_sha256": self.envelope_sha256}

    def verify_integrity(self) -> None:
        self._certificate.verify_integrity()
        if self.facade_attested and self.configuration_sha256 != self._configuration_digest():
            raise QualityGateError("certificate configuration integrity check failed")
        if self.facade_attested and self.envelope_sha256 != self._envelope_digest():
            raise QualityGateError("public certificate envelope integrity check failed")

    @classmethod
    def assess_new(
        cls,
        source: Any,
        *,
        ds_name: str,
        quality_level: Any = UNSET,
        core_labels: Any = UNSET,
        core_info: Any = UNSET,
        configure: Optional[Configure] = None,
        asset_root: Optional[Union[str, Path]] = None,
    ) -> "Certificate":
        requested_configuration = configure or Configure()
        rules = requested_configuration.effective(ds_name)
        declared_quality = (
            rules.get("quality_level", UNSET) if quality_level is UNSET else quality_level
        )
        declared_labels = rules.get("core_labels", UNSET) if core_labels is UNSET else core_labels
        declared_info = rules.get("core_info", UNSET) if core_info is UNSET else core_info
        if (
            declared_quality is UNSET
            and declared_labels is UNSET
            and declared_info is UNSET
        ):
            raise ValueError(
                "new dataset assessment requires quality_level, core_labels, or core_info; "
                "pass [] to explicitly declare that no core fields exist"
            )
        configuration = _assessment_configuration(
            requested_configuration, ds_name, declared_labels, declared_info
        )
        rules = configuration.effective(ds_name)
        try:
            submission = load_submission(source, asset_root=asset_root)
            submission = _apply_declarations(
                submission,
                ds_name,
                declared_quality,
                declared_labels,
                declared_info,
            )
            policy_labels = (
                _bundle_core_declaration(submission.bundle, "core_labels")
                if declared_labels is UNSET
                else declared_labels
            )
            policy_info = (
                _bundle_core_declaration(submission.bundle, "core_info")
                if declared_info is UNSET
                else declared_info
            )
            configuration = _assessment_configuration(
                requested_configuration, ds_name, policy_labels, policy_info
            )
            rules = configuration.effective(ds_name)
        except (SubmissionFenceError, DatasetStandardError, ValueError) as error:
            return cls(
                _failed_fence_certificate(source, error),
                None,
                configuration,
                ds_name=ds_name,
                mode="new",
                declarations={
                    "quality_level": None if declared_quality is UNSET else declared_quality,
                    "core_labels": None if declared_labels is UNSET else list(declared_labels),
                    "core_info": None if declared_info is UNSET else list(declared_info),
                },
                source_kind=_source_kind(source),
            )

        effective = dict(rules)
        if declared_labels is not UNSET:
            effective["core_labels"] = list(_sequence(declared_labels, "core_labels"))
        if declared_info is not UNSET:
            effective["core_info"] = list(_sequence(declared_info, "core_info"))
        base = assess_bundle(submission.bundle, submission.asset_root, profile="admission")
        certificate = _issue_with_configured_failure(
            base, _configured_issues(submission.bundle, effective)
        )
        return cls(
            certificate,
            submission,
            configuration,
            ds_name=ds_name,
            mode="new",
            declarations={
                "quality_level": None
                if declared_quality is UNSET
                else _normalise_quality_level(declared_quality),
                "core_labels": None
                if declared_labels is UNSET
                else list(_sequence(declared_labels, "core_labels")),
                "core_info": None
                if declared_info is UNSET
                else list(_sequence(declared_info, "core_info")),
            },
            source_kind=_source_kind(source),
        )

    @classmethod
    def assess_existing(
        cls,
        source: Any,
        *,
        ds_name: str,
        store: Any,
        configure: Optional[Configure] = None,
        asset_root: Optional[Union[str, Path]] = None,
    ) -> "Certificate":
        if configure is not None:
            raise ValueError(
                "established dataset rules are loaded from the store and cannot be overridden"
            )
        spec = _established_spec(store, ds_name)
        persisted_policy = spec["policy"]
        if persisted_policy is None:
            configuration = Configure(
                datasets={
                    ds_name: {
                        "core_labels": list(spec["core_labels"]),
                        "core_info": list(spec["core_info"]),
                    }
                }
            )
        else:
            configuration = Configure.from_dict(persisted_policy["config"])
            digest = hashlib.sha256(
                canonical_json(configuration.to_dict()).encode()
            ).hexdigest()
            if digest != persisted_policy["config_sha256"]:
                raise QualityGateError("stored dataset policy integrity check failed")
        rules = configuration.effective(ds_name)
        rules["core_labels"] = sorted(
            set(rules.get("core_labels", ())) | set(spec["core_labels"])
        )
        rules["core_info"] = sorted(
            set(rules.get("core_info", ())) | set(spec["core_info"])
        )
        try:
            submission = load_submission(source, asset_root=asset_root)
            submission = _apply_declarations(
                submission,
                ds_name,
                UNSET,
                rules["core_labels"],
                rules["core_info"],
            )
        except (SubmissionFenceError, DatasetStandardError, ValueError) as error:
            return cls(
                _failed_fence_certificate(source, error),
                None,
                configuration,
                ds_name=ds_name,
                mode="existing",
                declarations={
                    "quality_level": None,
                    "core_labels": list(rules["core_labels"]),
                    "core_info": list(rules["core_info"]),
                },
                established_spec=_public_established_spec(spec),
                source_kind=_source_kind(source),
            )

        base = assess_bundle(submission.bundle, submission.asset_root, profile="admission")
        issues = _existing_issues(submission.bundle, ds_name, spec)
        issues.extend(_configured_issues(submission.bundle, rules))
        certificate = _issue_with_configured_failure(base, issues)
        return cls(
            certificate,
            submission,
            configuration,
            ds_name=ds_name,
            mode="existing",
            declarations={
                "quality_level": None,
                "core_labels": list(rules["core_labels"]),
                "core_info": list(rules["core_info"]),
            },
            established_spec=_public_established_spec(spec),
            source_kind=_source_kind(source),
        )

    @classmethod
    def assess(
        cls,
        source: Any,
        *,
        ds_name: str,
        quality_level: Any = UNSET,
        core_labels: Any = UNSET,
        core_info: Any = UNSET,
        configure: Optional[Configure] = None,
        store: Any = None,
        asset_root: Optional[Union[str, Path]] = None,
    ) -> "Certificate":
        if store is not None:
            if any(value is not UNSET for value in (quality_level, core_labels, core_info)):
                raise ValueError(
                    "established dataset assessment loads its core specification from store"
                )
            return cls.assess_existing(
                source,
                ds_name=ds_name,
                store=store,
                configure=configure,
                asset_root=asset_root,
            )
        return cls.assess_new(
            source,
            ds_name=ds_name,
            quality_level=quality_level,
            core_labels=core_labels,
            core_info=core_info,
            configure=configure,
            asset_root=asset_root,
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Certificate":
        with Path(path).open(encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise QualityGateError("certificate file root must be an object")
        if value.get("schema_version") != CERTIFICATE_ENVELOPE_VERSION:
            raise QualityGateError("certificate does not use the current public envelope")
        expected = {
            "schema_version",
            "certificate",
            "configuration",
            "dataset",
            "mode",
            "declarations",
            "established_spec",
            "source_kind",
            "configuration_sha256",
            "envelope_sha256",
        }
        if set(value) != expected:
            raise QualityGateError("public certificate envelope fields do not match the standard")
        certificate = cls(
            QualityCertificate.from_dict(value["certificate"]),
            configure=Configure.from_dict(value["configuration"]),
            ds_name=value["dataset"],
            mode=value["mode"],
            declarations=value["declarations"],
            established_spec=value["established_spec"],
            source_kind=value["source_kind"],
            configuration_sha256=value["configuration_sha256"],
            envelope_sha256=value["envelope_sha256"],
        )
        certificate.verify_integrity()
        return certificate

    def write(self, path: Union[str, Path]) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


__all__ = [
    "CONFIGURE_VERSION",
    "CERTIFICATE_ENVELOPE_VERSION",
    "DEFAULT_GENERIC_RULES",
    "IMMUTABLE_GATES",
    "UNSET",
    "Certificate",
    "Configure",
]
