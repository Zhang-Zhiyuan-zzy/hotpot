"""Dataset catalog and the concise :func:`load` entry point."""

from __future__ import annotations

import json
from typing import Iterator, List, Optional, Sequence, Union

from .._paths import DEFAULT_STORE_PATH
from ..database import SCHEMA_VERSION, DatasetSchemaError, DatasetStore
from ..models import STANDARD_VERSION
from .dataset import Dataset


DEFAULT_DATABASE = DEFAULT_STORE_PATH


def _coerce_store(store=None, db=None) -> DatasetStore:
    if store is not None and db is not None:
        raise TypeError("store and db are mutually exclusive")
    if db is not None:
        return DatasetStore(db)
    if store is None:
        return DatasetStore(DEFAULT_DATABASE)
    if isinstance(store, DatasetStore):
        return store
    for attribute in ("backend", "_backend", "dataset_store", "_store"):
        candidate = getattr(store, attribute, None)
        if isinstance(candidate, DatasetStore):
            return candidate
    return DatasetStore(store)


def _check_store(store: DatasetStore) -> None:
    if not store.path.is_file():
        raise FileNotFoundError(store.path)
    try:
        with store._connect() as connection:
            metadata = dict(connection.execute("SELECT key,value FROM schema_metadata"))
    except Exception as error:
        raise DatasetSchemaError("database is not a Hotpot dataset store") from error
    expected = {
        "schema_version": str(SCHEMA_VERSION),
        "record_standard": STANDARD_VERSION,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise DatasetSchemaError(
            "dataset store schema is incompatible; rebuild it with the current Hotpot revision"
        )


class DatasetCatalog(Sequence[str]):
    """A printable, iterable view of dataset names in one store."""

    def __init__(self, store=None) -> None:
        self._store_spec = store

    def _backend(self) -> DatasetStore:
        return _coerce_store(self._store_spec)

    def _names(self) -> List[str]:
        store = self._backend()
        _check_store(store)
        with store._connect() as connection:
            return [
                row["dataset_id"]
                for row in connection.execute(
                    """SELECT DISTINCT d.dataset_id FROM datasets d
                    JOIN dataset_releases r USING(dataset_id)
                    ORDER BY d.dataset_id"""
                )
            ]

    def __call__(self, db=None) -> List[str]:
        if db is None:
            return self._names()
        return DatasetCatalog(db)._names()

    def __len__(self) -> int:
        return len(self._names())

    def __iter__(self) -> Iterator[str]:
        return iter(self._names())

    def __getitem__(self, index: Union[int, slice]):
        return self._names()[index]

    def __repr__(self) -> str:
        return repr(self._names())

    __str__ = __repr__

    def _dataset_objects(self) -> List[Dataset]:
        return [
            load(name, db=self._backend().path)
            for name in self._names()
        ]


def load(
    ds_name: str,
    version: Optional[Union[str, int]] = None,
    db=None,
) -> Dataset:
    """Load one release by dataset name and optional source version."""
    backend = _coerce_store(db=db)
    _check_store(backend)
    with backend._connect() as connection:
        dataset_row = connection.execute(
            "SELECT dataset_id,title,description FROM datasets WHERE dataset_id=?",
            (ds_name,),
        ).fetchone()
        if dataset_row is None:
            raise KeyError(f"unknown dataset {ds_name!r}")
        releases = connection.execute(
            """SELECT release_id,dataset_id,version,release_serial,quality_level,
            core_labels_json,license,access_policy,provenance_json,quality_json,ingestion_id
            FROM dataset_releases WHERE dataset_id=? ORDER BY release_serial""",
            (ds_name,),
        ).fetchall()
        if not releases:
            raise KeyError(f"dataset {ds_name!r} has no releases")
        if version is None:
            raw_release = releases[-1]
        else:
            raw_release = None
            if isinstance(version, str):
                raw_release = next(
                    (
                        release
                        for release in releases
                        if release["version"] == version
                        or release["release_id"] == version
                    ),
                    None,
                )
            if raw_release is None:
                serial = (
                    version
                    if isinstance(version, int)
                    else int(version)
                    if isinstance(version, str) and version.isdigit()
                    else None
                )
                raw_release = next(
                    (
                        release
                        for release in releases
                        if serial is not None and release["release_serial"] == serial
                    ),
                    None,
                )
            if raw_release is None:
                raise KeyError(f"dataset {ds_name!r} has no version {version!r}")
        release = dict(raw_release)
        release["provenance"] = json.loads(release.pop("provenance_json"))
        release["quality"] = json.loads(release.pop("quality_json"))
        release["core_labels"] = json.loads(release.pop("core_labels_json"))
        records = [
            dict(row)
            for row in connection.execute(
                """SELECT record_id,release_id,public_index,local_id,metadata_json
                FROM records WHERE release_id=? ORDER BY public_index""",
                (release["release_id"],),
            )
        ]
    for record in records:
        record["metadata"] = json.loads(record.pop("metadata_json"))
    indexed_records = [(record["public_index"], record) for record in records]
    return Dataset(
        backend,
        dict(dataset_row),
        release,
        int(release["release_serial"]),
        indexed_records,
    )


datasets_list = DatasetCatalog()


__all__ = ["DEFAULT_DATABASE", "DatasetCatalog", "datasets_list", "load"]
