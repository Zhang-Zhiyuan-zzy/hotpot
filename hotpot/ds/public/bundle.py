"""Folder-backed Bundle lifecycle for governed Hotpot datasets."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Optional, Union

from .._paths import DEFAULT_STORE_PATH
from ..gates import BenchmarkGate, EvidenceGate, QualityGate
from ..gates._persistence import (
    default_gates,
    deserialize_gate_configs,
    materialize_checker_sources,
    serialize_gate_configs,
)
from ..gates._checker import OSIsolatedCheckerRunner
from ..storage import (
    TARGET_SCHEMA_ID,
    TargetStore,
    TargetSchemaError,
    asset_inventory_digest,
    case_content_digest,
    case_set_digest,
)
from ._assessment import canonical_digest, canonical_json
from ._bundle_state import (
    BUNDLE_SCHEMA,
    BundleState,
    BundleStateError,
    _validate_manifest,
)
from ._implementation import require_current_runtime_policy, runtime_policy
from .case import Case, CaseCollection, CaseView, UnsynchronizedBundleError


class ExistingDatasetPolicyError(RuntimeError):
    """An existing dataset policy was supplied or differs from the store."""


class MissingCertificateError(RuntimeError):
    """Ingestion was requested without a current root certificate."""


class SealedReleaseError(RuntimeError):
    """A mutating operation targeted an already ingested release."""


class SealedIntegrityError(RuntimeError):
    """A sealed release no longer matches its authoritative receipt."""


def _read_manifest(root: Path, ds_name: str) -> Dict[str, Any]:
    path = root / "bundle.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BundleStateError(f"cannot read current Bundle manifest: {path}") from error
    return _validate_manifest(root, value, ds_name)


def _resolved_database(db: Optional[Union[str, Path]]) -> Path:
    return Path(DEFAULT_STORE_PATH if db is None else db).expanduser().resolve()


def _target_snapshot(path: Path, ds_name: str) -> Dict[str, Any]:
    if not path.exists():
        return {
            "canonical_path": str(path),
            "store_state": "absent",
            "store_uuid": None,
            "schema_id": TARGET_SCHEMA_ID,
            "dataset_state": "absent",
            "dataset_policy_revision": None,
            "dataset_policy_digest": None,
        }
    if not path.is_file():
        raise TargetSchemaError(f"target store path is not a file: {path}")
    store = TargetStore(path)
    metadata = store.metadata()
    policy = store.read_policy(ds_name)
    return {
        "canonical_path": str(path),
        "store_state": "existing",
        "store_uuid": metadata["store_uuid"],
        "schema_id": metadata["schema_id"],
        "dataset_state": "present" if policy is not None else "absent",
        "dataset_policy_revision": None if policy is None else policy["revision"],
        "dataset_policy_digest": None if policy is None else policy["digest"],
    }


def _gate_set(gates: Optional[Iterable[Any]]) -> tuple:
    defaults = {type(gate): gate for gate in default_gates()}
    supplied = () if gates is None else tuple(gates)
    supplied_types = [type(gate) for gate in supplied]
    allowed = {EvidenceGate, QualityGate, BenchmarkGate}
    if any(gate_type not in allowed for gate_type in supplied_types):
        raise TypeError(
            "gates accepts only EvidenceGate, QualityGate, and BenchmarkGate"
        )
    if len(supplied_types) != len(set(supplied_types)):
        raise ValueError("gates cannot contain duplicate Gate types")
    for gate in supplied:
        defaults[type(gate)] = gate
    return tuple(
        defaults[gate_type]
        for gate_type in (EvidenceGate, QualityGate, BenchmarkGate)
    )


def _policy_value(manifest: Mapping[str, Any], gates: Iterable[Any]) -> Dict[str, Any]:
    return {
        "schema": "hotpot-ds-policy/current",
        "gates": serialize_gate_configs(gates),
        "identity_schema": deepcopy(manifest.get("identity_schema", {})),
        "required_structure_roles": list(
            manifest.get("required_structure_roles", [])
        ),
        "property_definitions": deepcopy(
            manifest.get("property_definitions", {})
        ),
        "property_comparators": deepcopy(
            manifest.get("property_comparators", {})
        ),
        "runtime_policy": runtime_policy(),
    }


def _policy_has_custom_checkers(value: Any) -> bool:
    if isinstance(value, Mapping):
        if value.get("kind") == "custom":
            return True
        return any(_policy_has_custom_checkers(item) for item in value.values())
    if isinstance(value, list):
        return any(_policy_has_custom_checkers(item) for item in value)
    return False


class Bundle:
    """One editable, folder-backed dataset release and its assessments."""

    def __init__(
        self,
        root: Union[str, Path],
        ds_name: str,
        *,
        db: Optional[Union[str, Path]] = None,
        gates: Optional[Iterable[Any]] = None,
    ) -> None:
        bundle_root = Path(root).expanduser().resolve()
        if not bundle_root.is_dir():
            raise NotADirectoryError(bundle_root)
        manifest = _read_manifest(bundle_root, ds_name)
        database = _resolved_database(db)
        target = _target_snapshot(database, ds_name)
        if target["dataset_state"] == "present":
            if gates is not None:
                raise ExistingDatasetPolicyError(
                    "existing datasets use their persisted Gates; gates must be omitted"
                )
            stored_policy = TargetStore(database).read_policy(ds_name)
            if stored_policy is None:
                raise ExistingDatasetPolicyError(
                    "existing dataset has no persisted policy"
                )
            policy = stored_policy["value"]
            _assert_manifest_policy(manifest, policy)
            # A successor Bundle is a new folder even though its dataset policy
            # already exists.  Recreate the content-addressed checker files from
            # that authoritative persisted policy before the canonical state is
            # built; callers must not have to resupply executable policy.
            materialize_checker_sources(
                deserialize_gate_configs(policy["gates"]), bundle_root
            )
        else:
            normalized_gates = _gate_set(gates)
            materialize_checker_sources(normalized_gates, bundle_root)
            policy = _policy_value(manifest, normalized_gates)
        state = BundleState.build(bundle_root, ds_name, policy, target)
        self._initialize(state, database)

    @classmethod
    def _from_state(cls, state: BundleState, database: Path) -> "Bundle":
        instance = cls.__new__(cls)
        instance._initialize(state, database)
        return instance

    def _initialize(self, state: BundleState, database: Path) -> None:
        self._state = state
        self._database = database
        self._case_cache: Dict[str, Case] = {}
        self._validate_target_policy()

    @classmethod
    def from_certificate(
        cls,
        path: Union[str, Path],
        *,
        db: Optional[Union[str, Path]] = None,
    ) -> "Bundle":
        from .certificate import load_certificate_envelope, verify_bundle_certificate

        certificate_path = Path(path).expanduser().resolve()
        envelope = load_certificate_envelope(certificate_path)
        certificate = envelope["certificate"]
        root = certificate_path.parent
        if certificate_path != root / "certificate.json":
            raise ValueError(
                "a Bundle certificate must be bundle-root/certificate.json"
            )
        certified_database = Path(
            certificate["target_context"]["canonical_path"]
        ).resolve()
        database = certified_database if db is None else _resolved_database(db)
        if database != certified_database:
            raise ValueError("db does not match the certificate-bound target")
        bundle = load_bundle(root, certificate["dataset"], db=database)
        verify_bundle_certificate(bundle, certificate_path)
        return bundle

    @property
    def root(self) -> Path:
        return self._state.root

    @property
    def name(self) -> str:
        return self._state.ds_name

    @property
    def mode(self) -> str:
        # Dataset policy authority is fixed when the Bundle is built.  Deriving
        # this from the live target would incorrectly turn an existing-dataset
        # successor into a policy-editable "new" Bundle if that target vanished.
        return (
            "existing"
            if self._state.target_context["dataset_state"] == "present"
            else "new"
        )

    @property
    def info(self) -> Dict[str, Any]:
        manifest = self._state.manifest
        return {
            "name": self.name,
            "dataset_id": manifest["dataset_id"],
            "release_id": manifest["release_id"],
            "version": manifest["version"],
            "title": manifest["title"],
            "description": manifest["description"],
            "license": manifest["license"],
            "access_policy": manifest["access_policy"],
            "core_labels": list(manifest["core_labels"]),
            "core_info": list(manifest["core_info"]),
            "size": len(self),
            "mode": self.mode,
            "database": str(self._database),
        }

    @property
    def status(self) -> Dict[str, Any]:
        state = self._state.status
        generation = None
        generation_valid = False
        if state.get("current_generation_id"):
            try:
                generation = self._current_generation
                generation_valid = generation is not None
            except Exception:
                generation_valid = False
        receipt = None
        receipt_integrity = "valid"
        try:
            receipt = self._authoritative_receipt()
        except SealedIntegrityError:
            receipt_integrity = "invalid"
        sealed = receipt is not None or receipt_integrity == "invalid"
        synchronized = bool(
            receipt is not None
            or (not sealed and state["is_synchronized"] and generation_valid)
        )
        certificate_state = "absent"
        if self.certificate_path.is_file():
            try:
                from .certificate import verify_bundle_certificate

                verify_bundle_certificate(self)
                certificate_state = "current"
            except Exception:
                certificate_state = "stale"
        target_ready = self._target_precondition_satisfied()
        ingest_ready = (
            synchronized
            and certificate_state == "current"
            and target_ready
            and receipt_integrity == "valid"
            and state.get("runtime_policy_current", False)
        )
        assessment_governance = (
            None if generation is None else generation.manifest["governance"]
        )
        governance_update_available = bool(
            sealed
            and assessment_governance is not None
            and state["governance_registry_revision"]
            > assessment_governance["revision"]
        )
        if not state.get("runtime_policy_current", False):
            next_action = (
                "create_successor_release" if sealed else "owner_policy_migration"
            )
        elif receipt_integrity == "invalid":
            next_action = "repair_sealed_integrity"
        elif sealed:
            next_action = "none"
        elif not synchronized:
            next_action = "synchronize"
        elif certificate_state != "current":
            next_action = "certify"
        elif not target_ready:
            next_action = "resolve_target"
        else:
            next_action = "ingest"
        return {
            **state,
            "is_synchronized": synchronized,
            "generation_id": None if generation is None else generation.id,
            "generation_digest": None if generation is None else generation.digest,
            "assessment_governance_registry_revision": (
                None
                if assessment_governance is None
                else assessment_governance["revision"]
            ),
            "assessment_governance_registry_digest": (
                None
                if assessment_governance is None
                else assessment_governance["digest"]
            ),
            "governance_update_available": governance_update_available,
            "certificate_state": certificate_state,
            "target_precondition_satisfied": target_ready,
            "ingest_ready": bool(ingest_ready),
            "release_state": (
                "sealed_integrity_error"
                if receipt_integrity == "invalid"
                else "sealed" if sealed else "draft"
            ),
            "receipt_integrity": receipt_integrity,
            "next_action": next_action,
        }

    @property
    def is_synchronized(self) -> bool:
        try:
            if self._authoritative_receipt() is not None:
                return True
        except SealedIntegrityError:
            return False
        state = self._state.status
        if not state["is_synchronized"] or not state.get("current_generation_id"):
            return False
        try:
            return self._current_generation is not None
        except Exception:
            return False

    @property
    def summary(self) -> Dict[str, Any]:
        return deepcopy(self._require_generation().summary)

    @property
    def gates(self) -> Mapping[str, Any]:
        gates = deserialize_gate_configs(self._state.policy["gates"])
        return MappingProxyType(
            {
                "evidence": gates[0],
                "quality": gates[1],
                "benchmark": gates[2],
            }
        )

    @property
    def certificate_path(self) -> Path:
        return self.root / "certificate.json"

    @property
    def certificate_stale(self) -> bool:
        if not self.certificate_path.is_file():
            return True
        try:
            from .certificate import verify_bundle_certificate

            verify_bundle_certificate(self)
            return False
        except Exception:
            return True

    @property
    def cases(self) -> CaseCollection:
        return CaseCollection(self)

    @property
    def incomplete(self) -> CaseView:
        return self.cases.select(archive_class="incomplete")

    @property
    def ordinary(self) -> CaseView:
        return self.cases.select(archive_class="ordinary")

    @property
    def benchmark(self) -> CaseView:
        return self.cases.select(archive_class="benchmark")

    @property
    def _current_generation(self):
        status = self._state.status
        generation_id = status.get("current_generation_id")
        if not generation_id:
            return None
        from ._generation import AssessmentGeneration

        return AssessmentGeneration.open(
            self.root,
            generation_id,
            expected_digest=status.get("current_generation_digest"),
        )

    def __len__(self) -> int:
        return len(self._state)

    def __repr__(self) -> str:
        return (
            f"Bundle(name={self.name!r}, release={self._state.manifest['release_id']!r}, "
            f"size={len(self)}, mode={self.mode!r})"
        )

    def use_gate(self, gate: Any) -> None:
        with self._locked():
            self._assert_editable()
            if self.mode == "existing":
                raise ExistingDatasetPolicyError(
                    "an existing dataset's persisted Gates cannot be changed by Bundle"
                )
            if type(gate) not in {EvidenceGate, QualityGate, BenchmarkGate}:
                raise TypeError(
                    "gate must be EvidenceGate, QualityGate, or BenchmarkGate"
                )
            by_type = {type(item): item for item in self.gates.values()}
            by_type[type(gate)] = gate
            normalized = tuple(
                by_type[kind]
                for kind in (EvidenceGate, QualityGate, BenchmarkGate)
            )
            materialize_checker_sources(normalized, self.root)
            policy = self._state.policy
            policy["gates"] = serialize_gate_configs(normalized)
            self._state.mutate_policy(
                policy,
                action="gate_policy_changed",
                details={"gate": type(gate).__name__, "digest": gate.digest},
            )

    def synchronize(self) -> Dict[str, Any]:
        with self._locked():
            self._assert_editable()
            self._state.refresh_source_state()
            self._state.refresh_governance_state()
            status = self._state.status
            from ._generation import AssessmentGeneration

            generation = AssessmentGeneration.build(
                self.root,
                self._state,
                self._state.policy,
                self._governance_registry_snapshot(),
                isolated_runner=(
                    OSIsolatedCheckerRunner((self.root, self._database.parent))
                    if _policy_has_custom_checkers(self._state.policy)
                    else None
                ),
            )
            self._state.publish_generation(
                generation.id,
                generation.digest,
                expected_source_revision=status["source_revision"],
                expected_source_digest=status["source_digest"],
                expected_policy_revision=status["policy_revision"],
                expected_policy_digest=status["policy_digest"],
                expected_governance_registry_revision=status[
                    "governance_registry_revision"
                ],
                expected_governance_registry_digest=status[
                    "governance_registry_digest"
                ],
            )
            result = {
                "generation_id": generation.id,
                "generation_digest": generation.digest,
                "archive_counts": deepcopy(
                    generation.summary.get("archive_class", {})
                ),
                "summary": deepcopy(generation.summary),
                "is_synchronized": True,
            }
            self._collect_generation_garbage()
            return result

    def write_certificate(self) -> Path:
        if self._authoritative_receipt() is not None:
            from .certificate import verify_bundle_certificate

            try:
                verify_bundle_certificate(self)
            except Exception as error:
                raise SealedIntegrityError(
                    "sealed release certificate or bound data is no longer intact"
                ) from error
            return self.certificate_path
        from .certificate import write_bundle_certificate

        _target_control_asset_descriptors(
            self.root,
            self._require_generation(),
            self._state.manifest,
            include_certificate=False,
        )
        path = write_bundle_certificate(self)
        with self._locked():
            self._collect_generation_garbage()
        return path

    def ingest(
        self,
        *,
        include_incomplete: bool = False,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self._locked():
            require_current_runtime_policy(self._state.policy["runtime_policy"])
            if not self.is_synchronized:
                raise UnsynchronizedBundleError(
                    "Bundle is not synchronized; run bundle.synchronize() first"
                )
            if not self.certificate_path.is_file():
                raise MissingCertificateError(
                    "No current Bundle certificate exists; run bundle.write_certificate() "
                    "or hotpot ds certify first."
                )
            from .certificate import certificate_digest, verify_bundle_certificate

            generation = self._require_generation()
            with generation.lease():
                verify_bundle_certificate(self)
                snapshot = self._ingestion_snapshot(
                    certificate_digest(self.certificate_path),
                    include_incomplete,
                    generation=generation,
                )
                verify_bundle_certificate(self)
                result = TargetStore(self._database).ingest(
                    snapshot,
                    include_incomplete=include_incomplete,
                    reason=reason,
                    _precommit_check=lambda: self._assert_ingestion_snapshot_current(
                        snapshot,
                        include_incomplete=include_incomplete,
                    ),
                )
            projection = result.to_dict()
            projection["receipt_digest"] = hashlib.sha256(
                canonical_json(projection).encode("utf-8")
            ).hexdigest()
            _atomic_json(self.root / "ingestion-receipt.json", projection)
            self._collect_generation_garbage()
            return projection

    def _case_ids(self) -> tuple[str, ...]:
        with self._state.connect() as connection:
            return tuple(
                row[0]
                for row in connection.execute(
                    "SELECT case_id FROM cases ORDER BY public_index,case_id"
                )
            )

    def _case_object(self, case_id: str) -> Case:
        if case_id not in self._case_cache:
            self._state.get_case(case_id)
            self._case_cache[case_id] = Case(self, case_id)
        return self._case_cache[case_id]

    def _case_payload(self, case_id: str) -> Dict[str, Any]:
        return self._state.get_case(case_id)

    def _case_assessment(self, case_id: str) -> Dict[str, Any]:
        return deepcopy(self._require_generation().get_case(case_id))

    def _mutate_case(
        self,
        case_id: str,
        action: str,
        mutation: Callable[
            [Dict[str, Any]], tuple[Mapping[str, Any], Any]
        ],
    ) -> Any:
        with self._locked():
            self._assert_editable()
            return self._state.mutate_case(case_id, action, mutation)["result"]

    @contextmanager
    def _locked(self):
        lock_path = self.root / ".bundle.lock"
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _new_fact_identifier(self, case_id: str, kind: str) -> str:
        return f"{case_id}:{kind}:{uuid.uuid4().hex}"

    def _require_generation(self):
        if not self.is_synchronized:
            raise UnsynchronizedBundleError(
                "derived results are stale; run bundle.synchronize() first"
            )
        generation = self._current_generation
        if generation is None:
            raise UnsynchronizedBundleError(
                "no current assessment generation; run bundle.synchronize() first"
            )
        return generation

    def _collect_generation_garbage(self) -> Dict[str, Any]:
        """Run private reference-safe generation maintenance under the Bundle lock."""

        from ._generation import _collect_obsolete_generations

        status = self._state.status
        current_id = status.get("current_generation_id")
        current_digest = status.get("current_generation_digest")
        current_reference = (
            (current_id, current_digest)
            if isinstance(current_id, str) and isinstance(current_digest, str)
            else None
        )
        receipt_digests = []
        receipt_reference_unknown = False
        local_receipt = self.root / "ingestion-receipt.json"
        if self._database.is_file():
            try:
                release = TargetStore(self._database).load_release_metadata(
                    self.name,
                    self._state.manifest["release_id"],
                )
                receipt = None if release is None else release.get("receipt")
                if receipt is not None:
                    receipt_digests.append(receipt["generation_digest"])
                elif local_receipt.exists():
                    receipt_reference_unknown = True
            except Exception:
                receipt_reference_unknown = True
        elif local_receipt.exists():
            receipt_reference_unknown = True
        return _collect_obsolete_generations(
            self.root,
            current_reference=current_reference,
            receipt_generation_digests=receipt_digests,
            receipt_reference_unknown=receipt_reference_unknown,
        )

    def _assert_editable(self) -> None:
        if self._authoritative_receipt() is not None:
            raise SealedReleaseError(
                "this release is sealed; create a successor release for corrections"
            )

    def _current_target(self) -> Dict[str, Any]:
        return _target_snapshot(self._database, self.name)

    def _validate_target_policy(self) -> None:
        current = self._current_target()
        if current["dataset_state"] != "present":
            return
        stored = TargetStore(self._database).read_policy(self.name)
        if stored is None or stored["digest"] != canonical_digest(self._state.policy):
            raise ExistingDatasetPolicyError(
                "Bundle policy differs from the established dataset policy"
            )

    def _target_precondition_satisfied(self) -> bool:
        try:
            current = self._current_target()
        except Exception:
            return False
        original = self._state.target_context
        if current["schema_id"] != TARGET_SCHEMA_ID:
            return False
        if (
            original["store_state"] == "existing"
            and current["store_uuid"] != original["store_uuid"]
        ):
            return False
        if current["dataset_state"] == "present":
            return current["dataset_policy_digest"] == canonical_digest(
                self._state.policy
            )
        return True

    def _authoritative_receipt(self) -> Optional[Dict[str, Any]]:
        local_receipt_exists = (self.root / "ingestion-receipt.json").is_file()
        if not self._database.is_file():
            if local_receipt_exists:
                raise SealedIntegrityError(
                    "the authoritative target for a locally recorded sealed release is missing"
                )
            return None
        try:
            release = TargetStore(self._database).load_release_metadata(
                self.name, self._state.manifest["release_id"]
            )
        except Exception as error:
            raise SealedIntegrityError(
                "the authoritative target store cannot be verified"
            ) from error
        receipt = None if release is None else release.get("receipt")
        if receipt is None:
            if local_receipt_exists:
                raise SealedIntegrityError(
                    "the authoritative target has lost the locally recorded sealed release"
                )
            return None
        try:
            if not self.certificate_path.is_file():
                raise SealedIntegrityError(
                    "sealed release root certificate is missing"
                )
            certificate_sha256, _ = _hash_file(self.certificate_path)
            generation = self._frozen_generation()
            snapshot = self._ingestion_snapshot(
                certificate_sha256,
                bool(receipt["include_incomplete"]),
                generation=generation,
            )
            expected = snapshot["digests"]
            actual = {
                "bundle": receipt["bundle_digest"],
                "certificate": receipt["certificate_digest"],
                "generation": receipt["generation_digest"],
                "case_content": receipt["case_content_digest"],
                "included_cases": receipt["included_cases_digest"],
                "excluded_cases": receipt["excluded_cases_digest"],
                "asset_inventory": receipt["asset_inventory_digest"],
            }
            if actual != expected:
                raise SealedIntegrityError(
                    "sealed release receipt does not match current certified content"
                )
            if receipt["policy_digest"] != snapshot["policy"]["digest"]:
                raise SealedIntegrityError(
                    "sealed release receipt policy does not match current policy"
                )
            binding = receipt.get("binding")
            if not isinstance(binding, Mapping) or binding.get("target_context") != self._state.target_context:
                raise SealedIntegrityError(
                    "sealed release receipt target context does not match Bundle"
                )
            frozen_runtime = self._state.policy.get("runtime_policy")
            if (
                binding.get("runtime_policy") != frozen_runtime
                or generation.manifest.get("runtime_policy") != frozen_runtime
            ):
                raise SealedIntegrityError(
                    "sealed release runtime policy binding is inconsistent"
                )
        except SealedIntegrityError:
            raise
        except Exception as error:
            raise SealedIntegrityError(
                "sealed release receipt binding cannot be verified"
            ) from error
        return receipt

    def _frozen_generation(self):
        """Return the receipt-bound generation without consulting live governance."""

        with self._state.connect() as connection:
            control = dict(
                connection.execute(
                    "SELECT * FROM control WHERE singleton=1"
                ).fetchone()
            )
            source_digest = self._state._compute_source_digest(
                connection, self._state._stored_manifest(connection)
            )
            policy_digest = canonical_digest(
                json.loads(
                    connection.execute(
                        "SELECT policy_json FROM policy WHERE singleton=1"
                    ).fetchone()[0]
                )
            )
        generation_id = control.get("current_generation_id")
        if not generation_id:
            raise SealedIntegrityError(
                "sealed release has no complete assessment generation"
            )
        from ._generation import AssessmentGeneration

        generation = AssessmentGeneration.open(
            self.root,
            generation_id,
            expected_digest=control.get("current_generation_digest"),
        )
        manifest = generation.manifest
        source_matches = (
            control["source_revision"] == control["synchronized_source_revision"]
            and source_digest == control["source_digest"]
            and source_digest == control["synchronized_source_digest"]
            and manifest["source"]
            == {
                "revision": control["source_revision"],
                "digest": source_digest,
            }
        )
        policy_matches = (
            control["policy_revision"] == control["synchronized_policy_revision"]
            and policy_digest == control["policy_digest"]
            and policy_digest == control["synchronized_policy_digest"]
            and manifest["policy"]
            == {
                "revision": control["policy_revision"],
                "digest": policy_digest,
            }
        )
        frozen_governance = manifest["governance"]
        governance_matches = (
            control["governance_registry_revision"]
            == control["synchronized_governance_registry_revision"]
            == frozen_governance["revision"]
            and control["governance_registry_digest"]
            == control["synchronized_governance_registry_digest"]
            == frozen_governance["digest"]
        )
        if not (source_matches and policy_matches and governance_matches):
            raise SealedIntegrityError(
                "sealed release state does not match its frozen assessment generation"
            )
        return generation

    def _governance_registry_snapshot(self) -> Dict[str, Any]:
        from ..governance import default_registry_snapshot, verify_registry

        snapshot = default_registry_snapshot()
        verify_registry(snapshot)
        return snapshot

    def _hotpot_governance_context(self) -> Dict[str, Any]:
        return {
            "dataset": self.name,
            "release": self._state.manifest["release_id"],
            "registry": self._governance_registry_snapshot(),
        }

    def _scientific_asset_inventory(self) -> list[Dict[str, Any]]:
        inventory = []
        for case in self._state.iter_cases():
            for collection in ("structures", "evidence"):
                for record in case.get(collection, []):
                    for descriptor in _asset_descriptors(record):
                        path_text = descriptor.get("path")
                        if not path_text:
                            inventory.append(
                                {
                                    "case_id": case["id"],
                                    "declaration_id": _record_id(record, collection),
                                    "path": None,
                                    "state": "remote_not_materialized",
                                    "sha256": descriptor.get("expected_sha256"),
                                    "size": None,
                                }
                            )
                            continue
                        path = (self.root / path_text).resolve()
                        if not path.is_file():
                            inventory.append(
                                {
                                    "case_id": case["id"],
                                    "declaration_id": _record_id(record, collection),
                                    "path": path_text,
                                    "state": "missing",
                                    "sha256": None,
                                    "size": None,
                                }
                            )
                            continue
                        digest, size = _hash_file(path)
                        inventory.append(
                            {
                                "case_id": case["id"],
                                "declaration_id": _record_id(record, collection),
                                "path": path_text,
                                "state": "present",
                                "sha256": digest,
                                "size": size,
                            }
                        )
        return sorted(
            inventory,
            key=lambda item: (
                item["case_id"],
                item["declaration_id"],
                str(item["path"]),
            ),
        )

    def _ingestion_snapshot(
        self,
        certificate_sha256: str,
        include_incomplete: bool,
        *,
        generation: Optional[Any] = None,
    ) -> Dict[str, Any]:
        generation = self._require_generation() if generation is None else generation
        included_ids = []
        excluded_ids = []
        assets = []
        manifest = self._state.manifest
        control_assets = _target_control_asset_descriptors(
            self.root, generation, manifest
        )
        assets.extend(control_assets)
        for case in self._state.iter_cases():
            assessment = generation.get_case(case["id"])
            selected = (
                assessment["archive_class"] != "incomplete" or include_incomplete
            )
            (included_ids if selected else excluded_ids).append(case["id"])
            if selected:
                assets.extend(_target_scientific_asset_descriptors(self.root, case))
        bundle_digest = canonical_digest(
            {
                "manifest": manifest,
                "source_digest": self._state.status["source_digest"],
                "policy_digest": self._state.status["policy_digest"],
                "generation_digest": generation.digest,
            }
        )
        policy = self._state.policy
        current_policy = None
        if self._database.is_file():
            current_policy = TargetStore(self._database).read_policy(self.name)
        policy_revision = (
            current_policy["revision"]
            if current_policy is not None
            else self._state.status["policy_revision"] + 1
        )
        content_digest = case_content_digest(
            self._iter_ingestion_content(generation)
        )
        return {
            "target_context": self._state.target_context,
            "bundle": {
                "dataset": self.name,
                "release": manifest["release_id"],
                "version": manifest["version"],
                "release_serial": manifest.get("release_serial"),
                "title": manifest["title"],
                "description": manifest["description"],
                "metadata": {
                    "license": manifest["license"],
                    "access_policy": manifest["access_policy"],
                    "provenance": manifest["provenance"],
                    "core_labels": manifest["core_labels"],
                    "core_info": manifest["core_info"],
                },
            },
            "policy": {
                "revision": policy_revision,
                "digest": canonical_digest(policy),
                "value": policy,
            },
            "digests": {
                "bundle": bundle_digest,
                "certificate": certificate_sha256,
                "generation": generation.digest,
                "case_content": content_digest,
                "included_cases": case_set_digest(included_ids),
                "excluded_cases": case_set_digest(excluded_ids),
                "asset_inventory": asset_inventory_digest(assets),
            },
            "control_assets": control_assets,
            "cases": self._iter_ingestion_cases(generation),
        }

    def _assert_ingestion_snapshot_current(
        self,
        snapshot: Mapping[str, Any],
        *,
        include_incomplete: bool,
    ) -> None:
        """Compare-and-swap guard executed inside the target transaction.

        The Bundle lock prevents mutations through the supported API while the
        target spools assets and Cases.  This final check also detects external
        changes to facts, certificate, generation, policy, Case selection, or
        bound assets before the target transaction publishes the release.
        """

        from .certificate import certificate_digest, verify_bundle_certificate

        generation = self._require_generation()
        verify_bundle_certificate(self)
        current_certificate_digest = certificate_digest(self.certificate_path)
        current = self._ingestion_snapshot(
            current_certificate_digest,
            include_incomplete,
            generation=generation,
        )
        for section in ("target_context", "bundle", "policy", "digests"):
            if current[section] != snapshot[section]:
                raise SealedIntegrityError(
                    "Bundle changed after ingestion snapshot creation; retry ingest"
                )

    def _iter_ingestion_cases(self, generation: Any) -> Iterator[Dict[str, Any]]:
        for case in self._state.iter_cases():
            assessment = generation.get_case(case["id"])
            yield {
                "id": case["id"],
                "public_index": case["public_index"],
                "local_id": case["local_id"],
                "facts": case,
                "assessment": assessment,
                "identifiers": [
                    {"namespace": str(namespace), "value": str(value)}
                    for namespace, raw in case.get("identifiers", {}).items()
                    for value in (raw if isinstance(raw, list) else [raw])
                ],
                "structures": _target_structure_descriptors(self.root, case),
                "assets": _target_scientific_asset_descriptors(self.root, case),
            }

    def _iter_ingestion_content(
        self, generation: Any
    ) -> Iterator[Dict[str, Any]]:
        """Project receipt-bound Case content without rehashing asset files."""

        for case in self._state.iter_cases():
            yield {
                "id": case["id"],
                "public_index": case["public_index"],
                "local_id": case["local_id"],
                "facts": case,
                "assessment": generation.get_case(case["id"]),
                "identifiers": [
                    {"namespace": str(namespace), "value": str(value)}
                    for namespace, raw in case.get("identifiers", {}).items()
                    for value in (raw if isinstance(raw, list) else [raw])
                ],
            }


def _assert_manifest_policy(
    manifest: Mapping[str, Any], policy: Mapping[str, Any]
) -> None:
    if manifest.get("identity_schema", {}) != policy.get("identity_schema", {}):
        raise ExistingDatasetPolicyError(
            "bundle.json identity schema differs from the established policy"
        )
    if list(manifest.get("required_structure_roles", [])) != list(
        policy.get("required_structure_roles", [])
    ):
        raise ExistingDatasetPolicyError(
            "bundle.json structure requirements differ from the established policy"
        )
    if manifest.get("property_definitions", {}) != policy.get(
        "property_definitions", {}
    ):
        raise ExistingDatasetPolicyError(
            "bundle.json property definitions differ from the established policy"
        )
    if manifest.get("property_comparators", {}) != policy.get(
        "property_comparators", {}
    ):
        raise ExistingDatasetPolicyError(
            "bundle.json property comparators differ from the established policy"
        )


def _record_id(record: Mapping[str, Any], collection: str) -> str:
    identity = {"structures": "structure_id", "evidence": "evidence_id"}[
        collection
    ]
    return str(record.get(identity, ""))


def _asset_descriptors(record: Mapping[str, Any]) -> Iterator[Dict[str, Any]]:
    if "path" in record or record.get("remote_not_materialized"):
        yield dict(record)
    assets = record.get("assets", [])
    if isinstance(assets, list):
        for asset in assets:
            if isinstance(asset, Mapping):
                yield dict(asset)
    representations = record.get("representations", {})
    if isinstance(representations, Mapping):
        for fmt, raw in representations.items():
            if isinstance(raw, str):
                yield {"format": fmt, "path": raw}
            elif isinstance(raw, Mapping):
                yield {"format": fmt, **dict(raw)}
    elif isinstance(representations, list):
        for raw in representations:
            if isinstance(raw, Mapping):
                yield dict(raw)


def _target_structure_descriptors(
    root: Path, case: Mapping[str, Any]
) -> list[Dict[str, Any]]:
    result = []
    for structure in case.get("structures", []):
        if structure.get("status", "active") != "active":
            continue
        for asset in _asset_descriptors(structure):
            path_text = asset.get("path")
            if not path_text:
                continue
            path = (root / path_text).resolve()
            if not path.is_file():
                continue
            digest, size = _hash_file(path)
            expected = asset.get("expected_sha256") or asset.get("sha256")
            if expected is not None and expected != digest:
                continue
            result.append(
                {
                    "structure_id": str(structure["structure_id"]),
                    "format": str(
                        asset.get("format")
                        or structure.get("format")
                        or path.suffix.lstrip(".")
                    ),
                    "role": str(structure.get("role", "structure")),
                    "primary": bool(structure.get("primary", False)),
                    "source_path": path,
                    "sha256": digest,
                    "size": size,
                }
            )
    return result


def _target_scientific_asset_descriptors(
    root: Path, case: Mapping[str, Any]
) -> list[Dict[str, Any]]:
    result = []
    for collection in ("structures", "evidence"):
        for record in case.get(collection, []):
            declaration_id = _record_id(record, collection)
            for asset_index, asset in enumerate(_asset_descriptors(record)):
                path_text = asset.get("path")
                if not path_text:
                    continue
                path = (root / path_text).resolve()
                if not path.is_file():
                    continue
                digest, size = _hash_file(path)
                result.append(
                    {
                        "collection": collection,
                        "declaration_id": declaration_id,
                        "asset_index": asset_index,
                        "role": str(
                            asset.get("role")
                            or asset.get("format")
                            or record.get("role")
                            or "asset"
                        ),
                        "status": str(record.get("status", "active")),
                        "source_path": path,
                        "sha256": digest,
                        "size": size,
                    }
                )
    return result


def _target_control_asset_descriptors(
    root: Path,
    generation: Any,
    manifest: Mapping[str, Any],
    *,
    include_certificate: bool = True,
) -> list[Dict[str, Any]]:
    paths = [
        ("bundle", "bundle.json", root / "bundle.json", None),
        ("facts", "facts.sqlite3", root / "cases" / "facts.sqlite3", None),
        (
            "governance",
            "governance-registry.json",
            generation.manifest_path.parent / "governance-registry.json",
            None,
        ),
    ]
    if include_certificate:
        paths.append(
            ("certificate", "certificate.json", root / "certificate.json", None)
        )
    generation_root = generation.manifest_path.parent
    paths.extend(
        (
            "generation",
            path.relative_to(generation_root).as_posix(),
            path,
            None,
        )
        for path in sorted(generation_root.rglob("*"))
        if path.is_file()
        and path != generation_root / "governance-registry.json"
    )
    checker_root = root / "policy" / "checkers"
    if checker_root.is_dir():
        paths.extend(
            ("checker", path.relative_to(checker_root).as_posix(), path, None)
            for path in sorted(checker_root.rglob("*.py"))
            if path.is_file()
        )
    paths.extend(
        (
            str(item["category"]),
            str(item["name"]),
            root / item["path"],
            item,
        )
        for item in manifest.get("control_assets", [])
    )
    result = []
    identities = set()
    for category, name, path, declaration in paths:
        identity = (category, name)
        if identity in identities:
            raise ValueError(
                f"duplicate release control asset identity: {category}/{name}"
            )
        identities.add(identity)
        digest, size = _hash_file(path)
        if declaration is not None and (
            digest != declaration["sha256"] or size != declaration["size"]
        ):
            raise ValueError(
                f"release control asset changed: {declaration['path']}"
            )
        result.append(
            {
                "category": category,
                "name": name,
                "source_path": path,
                "sha256": digest,
                "size": size,
            }
        )
    return sorted(result, key=lambda item: (item["category"], item["name"]))


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (canonical_json(dict(value)) + "\n").encode("utf-8")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_bundle(
    root: Union[str, Path],
    ds_name: str,
    *,
    db: Optional[Union[str, Path]] = None,
) -> Bundle:
    """Validation-only reopen of one already canonical Bundle root."""

    state = BundleState.open(root, ds_name)
    stored_database = Path(state.target_context["canonical_path"]).resolve()
    database = stored_database if db is None else _resolved_database(db)
    if database != stored_database:
        raise ValueError("db does not match the Bundle's canonical target")
    return Bundle._from_state(state, database)


def write_certificate(bundle: Bundle) -> Path:
    """Thin public wrapper over :meth:`Bundle.write_certificate`."""

    return bundle.write_certificate()


__all__ = ["Bundle", "load_bundle", "write_certificate"]
