"""Minimal public facade for dataset submissions and certification."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union

from .certificate import Certificate, Configure, UNSET
from .catalog import DEFAULT_DATABASE
from .dataset import Dataset
from .inputs import Submission, load_submission


class Bundle:
    """A submission prepared for certification without exposing backend rows."""

    def __init__(self, source: Any, *, asset_root: Optional[Union[str, Path]] = None):
        submission = load_submission(source, asset_root=asset_root)
        if len(submission.bundle.datasets) != 1:
            raise ValueError("a public Bundle must contain exactly one dataset")
        self._submission = submission

    @property
    def name(self) -> str:
        return self._submission.bundle.datasets[0].dataset_id

    @property
    def versions(self) -> list[str]:
        return [
            release.version
            for release in self._submission.bundle.releases
            if release.dataset_id == self.name
        ]

    @property
    def version(self) -> Optional[str]:
        versions = self.versions
        return versions[0] if len(versions) == 1 else None

    @property
    def info(self) -> Dict[str, Any]:
        dataset = self._submission.bundle.datasets[0]
        return {
            "name": self.name,
            "title": dataset.title,
            "description": dataset.description,
            "versions": self.versions,
            "size": len(self),
        }

    def __len__(self) -> int:
        release_ids = {
            release.release_id
            for release in self._submission.bundle.releases
            if release.dataset_id == self.name
        }
        return sum(
            record.release_id in release_ids
            for record in self._submission.bundle.records
        )

    def __repr__(self) -> str:
        return (
            f"Bundle(name={self.name!r}, versions={self.versions!r}, "
            f"size={len(self)})"
        )


def load_bundle(
    source: Any, *, asset_root: Optional[Union[str, Path]] = None
) -> Bundle:
    """Load a folder, compact SQLite file, mapping, or backend bundle."""

    if isinstance(source, Bundle):
        if asset_root is not None:
            raise ValueError("asset_root cannot replace the root of a loaded Bundle")
        return source
    return Bundle(source, asset_root=asset_root)


def _load_configure(
    configure: Optional[Union[Configure, Mapping[str, Any], str, Path]],
) -> Optional[Configure]:
    if configure is None or isinstance(configure, Configure):
        return configure
    if isinstance(configure, Mapping):
        return Configure.from_dict(configure)
    return Configure.from_file(configure)


def write_certificate(
    bundle: Bundle,
    path: Union[str, Path],
    *,
    quality_level: Optional[str] = None,
    core_labels: Optional[Sequence[str]] = None,
    core_info: Optional[Sequence[str]] = None,
    configure: Optional[Union[Mapping[str, Any], str, Path]] = None,
    dataset: Optional[Union[Dataset, str]] = None,
    db: Optional[Union[str, Path]] = None,
) -> Path:
    """Assess a bundle, write its integrity envelope, and return its path."""

    if not isinstance(bundle, Bundle):
        raise TypeError("bundle must be created by load_bundle")

    configuration = _load_configure(configure)
    submission: Submission = bundle._submission
    source = submission.bundle

    if dataset is not None:
        if any(value is not None for value in (quality_level, core_labels, core_info)):
            raise ValueError(
                "established dataset certification cannot override quality_level, "
                "core_labels, or core_info"
            )
        if configuration is not None:
            raise ValueError(
                "established dataset certification cannot override its stored rules"
            )
        if isinstance(dataset, Dataset):
            ds_name = dataset.name
            store = dataset._store if db is None else db
        elif isinstance(dataset, str):
            ds_name = dataset
            store = DEFAULT_DATABASE if db is None else db
        else:
            raise TypeError("dataset must be a Dataset or dataset-name string")
        certificate = Certificate.assess_existing(
            source,
            ds_name=ds_name,
            store=store,
            asset_root=submission.asset_root,
        )
    else:
        if db is not None:
            raise ValueError("db is only used with dataset=<existing dataset>")
        certificate = Certificate.assess_new(
            source,
            ds_name=bundle.name,
            quality_level=UNSET if quality_level is None else quality_level,
            core_labels=UNSET if core_labels is None else core_labels,
            core_info=UNSET if core_info is None else core_info,
            configure=configuration,
            asset_root=submission.asset_root,
        )

    certificate.write(path)
    return Path(path)


__all__ = ["Bundle", "load_bundle", "write_certificate"]
