"""Certificate-gated public storage facade."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from ..database import DatasetStore, IngestResult
from ..models import DatasetBundle
from ..quality import QualityGateError
from .certificate import (
    Certificate,
    UNSET,
    _apply_declarations,
    _configured_issues,
    _established_spec,
    _existing_issues,
    _public_established_spec,
)
from .inputs import Submission, load_submission
from .catalog import DEFAULT_DATABASE


def _same_source(source: Any, submission: Submission) -> bool:
    if isinstance(source, DatasetBundle):
        return source.digest == submission.bundle.digest
    if submission.manifest is not None and isinstance(source, Mapping):
        return source == submission.manifest
    return False


class Store:
    """Write datasets only after a passing admission assessment.

    Existing datasets are extended with new immutable releases.  An existing
    release is never updated in place.
    """

    def __init__(self, path: Union[str, Path] = DEFAULT_DATABASE):
        self.path = Path(path)
        self._backend = DatasetStore(self.path)

    def ingest(
        self,
        source: Any,
        *,
        certificate: Certificate,
        asset_root: Optional[Union[str, Path]] = None,
    ) -> IngestResult:
        if not isinstance(certificate, Certificate) or not certificate.facade_attested:
            raise QualityGateError(
                "store.ingest requires the certificate path produced by write_certificate"
            )
        if certificate.mode == "new" and all(
            certificate.declarations.get(field) is None
            for field in ("quality_level", "core_labels", "core_info")
        ):
            raise QualityGateError(
                "a new-dataset certificate must explicitly declare quality_level, "
                "core_labels, or core_info; empty core field lists are valid declarations"
            )
        raw_certificate = certificate.raw
        prepared = certificate.submission

        certificate.verify_integrity()
        if raw_certificate.profile != "admission" or not raw_certificate.passed:
            raise QualityGateError("a passing admission certificate is required")
        if prepared is not None and _same_source(source, prepared):
            submission = prepared
            if asset_root is not None:
                submission = Submission(
                    prepared.bundle,
                    Path(asset_root).resolve(),
                    prepared.source,
                    prepared.manifest,
                )
        else:
            submission = load_submission(source, asset_root=asset_root)
            declarations = certificate.declarations
            submission = _apply_declarations(
                submission,
                certificate.ds_name,
                declarations.get("quality_level")
                if declarations.get("quality_level") is not None
                else UNSET,
                declarations.get("core_labels")
                if declarations.get("core_labels") is not None
                else UNSET,
                declarations.get("core_info")
                if declarations.get("core_info") is not None
                else UNSET,
            )

        if raw_certificate.subject_digest != submission.bundle.digest:
            raise QualityGateError(
                "certificate does not match the normalized submission; assess this exact "
                "source (and keep core declarations in dataset.json when loading a saved certificate)"
            )

        self._backend.initialize()
        submitted_names = {dataset.dataset_id for dataset in submission.bundle.datasets}
        if submitted_names != {certificate.ds_name}:
            raise QualityGateError(
                f"certificate dataset {certificate.ds_name!r} does not match {sorted(submitted_names)!r}"
            )
        with self._backend._connect() as connection:
            already_stored = connection.execute(
                "SELECT 1 FROM quality_certificates WHERE certificate_id=?",
                (raw_certificate.certificate_id,),
            ).fetchone()
        policy_config = certificate.configure.to_dict()
        if already_stored is not None:
            return self._backend.ingest(
                submission.bundle,
                raw_certificate,
                submission.asset_root,
                dataset_policies={certificate.ds_name: policy_config},
            )
        for dataset in submission.bundle.datasets:
            existing = self._backend.get_dataset(dataset.dataset_id)
            if existing is None:
                if certificate.mode != "new":
                    raise QualityGateError(
                        "an established-dataset certificate cannot create a new dataset"
                    )
                issues = _configured_issues(
                    submission.bundle,
                    certificate.configure.effective(dataset.dataset_id),
                )
                if issues:
                    raise QualityGateError(
                        "new dataset no longer satisfies its certified configuration: "
                        + "; ".join(issues[:10])
                    )
                continue
            if certificate.mode != "existing":
                raise QualityGateError(
                    "an existing dataset must be assessed with "
                    "write_certificate(..., dataset=<existing dataset>)"
                )
            spec = _established_spec(self, dataset.dataset_id)
            if certificate.established_spec != _public_established_spec(spec):
                raise QualityGateError(
                    "the established dataset specification changed after assessment; "
                    "obtain a new certificate"
                )
            stored_policy = spec["policy"]
            if stored_policy is not None and stored_policy["config"] != policy_config:
                raise QualityGateError(
                    "certificate configuration differs from the persisted dataset policy"
                )
            if (
                stored_policy is not None
                and stored_policy["config_sha256"] != certificate.configuration_sha256
            ):
                raise QualityGateError(
                    "certificate configuration digest differs from the persisted dataset policy"
                )
            issues = _existing_issues(submission.bundle, dataset.dataset_id, spec)
            effective_rules = certificate.configure.effective(dataset.dataset_id)
            effective_rules["core_labels"] = sorted(
                set(effective_rules.get("core_labels", ())) | set(spec["core_labels"])
            )
            effective_rules["core_info"] = sorted(
                set(effective_rules.get("core_info", ())) | set(spec["core_info"])
            )
            issues.extend(
                _configured_issues(submission.bundle, effective_rules)
            )
            for release in submission.bundle.releases:
                if release.dataset_id != dataset.dataset_id:
                    continue
                if "core_labels" not in release.quality or "core_info" not in release.quality:
                    issues.append(
                        f"release {release.release_id}: established-dataset assessment must "
                        "record core_labels and core_info"
                    )
            if issues:
                raise QualityGateError(
                    "existing dataset integration failed: " + "; ".join(issues[:10])
                )

        return self._backend.ingest(
            submission.bundle,
            raw_certificate,
            submission.asset_root,
            dataset_policies={certificate.ds_name: policy_config},
        )


class _StoreFacade:
    """Opaque public write gate; use :attr:`hotpot.ds.store`."""

    def ingest(
        self,
        bundle,
        *,
        certificate: Union[str, Path],
        db: Union[str, Path],
    ) -> dict:
        from .bundle import Bundle

        if not isinstance(bundle, Bundle):
            raise TypeError("bundle must be created by load_bundle")
        if not isinstance(certificate, (str, Path)):
            raise TypeError("certificate must be the path returned by write_certificate")
        target = Store(db)
        result = target.ingest(
            bundle._submission.bundle,
            certificate=Certificate.load(certificate),
            asset_root=bundle._submission.asset_root,
        )
        counts = asdict(result)
        counts.pop("ingestion_id")
        counts.pop("certificate_id")
        return {
            "dataset": bundle.name,
            "versions": bundle.versions,
            **counts,
        }

    def __repr__(self) -> str:
        return "<hotpot.ds.store: certificate-gated ingestion>"


store = _StoreFacade()


__all__ = ["store"]
