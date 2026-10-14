"""Dataset catalog and concise current-store loader."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Optional, Union

from .._paths import DEFAULT_STORE_PATH
from ..storage import TargetStore
from .dataset import Dataset


DEFAULT_DATABASE = DEFAULT_STORE_PATH
_ARCHIVE_CLASSES = frozenset({"incomplete", "ordinary", "benchmark"})


def _store(db: Optional[Union[str, Path]] = None) -> TargetStore:
    path = Path(DEFAULT_DATABASE if db is None else db).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    store = TargetStore(path)
    store.metadata()
    return store


class DatasetCatalog(Sequence[str]):
    """Printable catalog and callable ``datasets_list`` entry point."""

    def __init__(self, db: Optional[Union[str, Path]] = None) -> None:
        self._db = db

    def _names(self) -> list[str]:
        store = _store(self._db)
        with store._connect() as connection:
            return [
                row[0]
                for row in connection.execute(
                    "SELECT dataset_id FROM datasets ORDER BY dataset_id"
                )
            ]

    def __call__(
        self, *, db: Optional[Union[str, Path]] = None
    ) -> list[str]:
        return DatasetCatalog(self._db if db is None else db)._names()

    def __len__(self) -> int:
        return len(self._names())

    def __iter__(self) -> Iterator[str]:
        return iter(self._names())

    def __getitem__(self, index):
        return self._names()[index]

    def __repr__(self) -> str:
        return repr(self._names())

    __str__ = __repr__

    def _dataset_objects(self):
        return [load(name, db=self._db) for name in self._names()]


def _classes(value) -> tuple[str, ...]:
    if value is None:
        return ("ordinary", "benchmark")
    values = (value,) if isinstance(value, str) else tuple(value)
    if not values or any(item not in _ARCHIVE_CLASSES for item in values):
        raise ValueError("archive_class must contain incomplete, ordinary, or benchmark")
    return tuple(dict.fromkeys(values))


def load(
    ds_name: str,
    version: Optional[Union[str, int]] = None,
    *,
    db: Optional[Union[str, Path]] = None,
    archive_class: Optional[Union[str, Iterable[str]]] = None,
) -> Dataset:
    """Load one current release; incomplete Cases are excluded by default."""

    store = _store(db)
    dataset = store.inspect_dataset(ds_name)
    if dataset is None:
        raise KeyError(f"unknown dataset {ds_name!r}")
    releases = dataset["releases"]
    if not releases:
        raise KeyError(f"dataset {ds_name!r} has no releases")
    if version is None:
        release = releases[-1]
    else:
        release = next(
            (
                item
                for item in releases
                if item["version"] == str(version)
                or item["release"] == str(version)
                or item["release_serial"] == version
                or (
                    isinstance(version, str)
                    and version.isdigit()
                    and item["release_serial"] == int(version)
                )
            ),
            None,
        )
        if release is None:
            raise KeyError(f"dataset {ds_name!r} has no version {version!r}")
    archive_classes = _classes(archive_class)
    headers = store.iter_case_headers(
        ds_name, release["release"], archive_classes=archive_classes
    )
    return Dataset(
        store,
        dataset,
        release,
        headers,
        archive_classes=archive_classes,
    )


datasets_list = DatasetCatalog()


__all__ = ["DEFAULT_DATABASE", "DatasetCatalog", "datasets_list", "load"]
