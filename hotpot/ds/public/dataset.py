"""Concise read-only Dataset and Data views over the current target store."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from copy import deepcopy
from typing import Any, Dict, Optional, Union

from .structures import Structures


_MULTI = (list, tuple, set, frozenset)
_MISSING = object()


def _value_at(value: Any, path: str) -> Any:
    current = value
    for part in path.replace("__", ".").split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            return _MISSING
    return current


def _matches(actual: Any, expected: Any) -> bool:
    if actual is _MISSING:
        return False
    if callable(expected):
        return bool(expected(actual))
    if isinstance(expected, _MULTI):
        options = tuple(expected)
        if isinstance(actual, _MULTI):
            return any(item in options for item in actual)
        return actual in options
    if isinstance(actual, _MULTI):
        return expected in actual
    return actual == expected


class IdentifierIndex:
    """Dataset-specific identifier lookup, e.g. ``dataset.CSD_identifier[x]``."""

    def __init__(self, dataset: "Dataset", namespace: str) -> None:
        self._dataset = dataset
        self._namespace = namespace

    def __getitem__(self, value: str) -> Union[Data, Dataset]:
        matched_ids = set(
            self._dataset._store.find_case_ids_by_identifier(
                self._dataset.name,
                self._dataset.release_id,
                self._namespace,
                value,
                archive_classes=self._dataset._archive_classes,
            )
        )
        matches = [
            item for item in self._dataset._headers if item["id"] in matched_ids
        ]
        if not matches:
            raise KeyError(value)
        if len(matches) == 1:
            return self._dataset._data(matches[0])
        return self._dataset._with_headers(matches)


class Dataset(Sequence["Data"]):
    """One immutable stored release or filtered view of that release."""

    def __init__(
        self,
        store: Any,
        dataset_info: Mapping[str, Any],
        release: Mapping[str, Any],
        headers: Iterable[Mapping[str, Any]],
        *,
        archive_classes: Iterable[str],
    ) -> None:
        self._store = store
        self._dataset_info = deepcopy(dict(dataset_info))
        self._release = deepcopy(dict(release))
        self._headers = tuple(deepcopy(dict(item)) for item in headers)
        self._archive_classes = frozenset(archive_classes)
        self._identifier_namespaces = None

    @property
    def name(self) -> str:
        return str(self._dataset_info["dataset"])

    @property
    def dataset_id(self) -> str:
        return self.name

    @property
    def release_id(self) -> str:
        return str(self._release["release"])

    @property
    def version(self) -> str:
        return str(self._release["version"])

    @property
    def version_serial(self) -> int:
        return int(self._release["release_serial"])

    @property
    def info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dataset_id": self.dataset_id,
            "release_id": self.release_id,
            "version": self.version,
            "version_serial": self.version_serial,
            "title": self._dataset_info["title"],
            "description": self._dataset_info["description"],
            "metadata": deepcopy(self._dataset_info.get("metadata", {})),
            "release_metadata": deepcopy(self._release.get("metadata", {})),
            "size": len(self),
            "archive_classes": sorted(self._archive_classes),
            "handle": self.handle,
            "labels": self.labels,
            "core_labels": self.core_labels,
            "quality_levels": self.quality_levels,
            "evidence_levels": self.evidence_levels,
            "identifier_namespaces": list(self.identifier_namespaces),
        }

    @property
    def handle(self) -> list[str]:
        metadata = self._release.get("metadata", {})
        provenance = metadata.get("provenance", {})
        values = []
        for key in ("doi", "url", "citation", "repository"):
            raw = provenance.get(key)
            if raw:
                values.extend(raw if isinstance(raw, list) else [raw])
        return [str(item) for item in values]

    @property
    def labels(self) -> Dict[str, int]:
        counts: Counter[str] = Counter()
        for data in self:
            counts.update(data.label.keys())
        return dict(sorted(counts.items()))

    @property
    def core_labels(self) -> list[str]:
        return list(self._release.get("metadata", {}).get("core_labels", []))

    @property
    def quality_levels(self) -> Dict[str, int]:
        return dict(
            sorted(Counter(item.get("quality_level") for item in self._headers).items())
        )

    @property
    def evidence_levels(self) -> Dict[str, int]:
        return dict(
            sorted(
                Counter(
                    item.get("evidence_level") or "not_applicable"
                    for item in self._headers
                ).items()
            )
        )

    @property
    def identifier_namespaces(self) -> tuple[str, ...]:
        if self._identifier_namespaces is None:
            expected_size = sum(
                int(self._release["counts"].get(item, 0))
                for item in self._archive_classes
            )
            visible_case_ids = (
                None
                if len(self._headers) == expected_size
                else (str(item["id"]) for item in self._headers)
            )
            self._identifier_namespaces = self._store.identifier_namespaces(
                self.name,
                self.release_id,
                archive_classes=self._archive_classes,
                visible_case_ids=visible_case_ids,
            )
        return self._identifier_namespaces

    def __len__(self) -> int:
        return len(self._headers)

    def __iter__(self) -> Iterator[Data]:
        return (self._data(item) for item in self._headers)

    def __getitem__(self, index: Union[int, slice]) -> Union[Data, Dataset]:
        if isinstance(index, slice):
            return self._with_headers(self._headers[index])
        return self._data(self._headers[index])

    def __getattr__(self, name: str) -> IdentifierIndex:
        if name.startswith("_"):
            raise AttributeError(name)
        suffix = "_identifier"
        if name.endswith(suffix):
            namespace = name[: -len(suffix)]
            if namespace in self.identifier_namespaces:
                return IdentifierIndex(self, namespace)
        raise AttributeError(name)

    def subset(self, **criteria: Any) -> Dataset:
        selected = []
        for header in self._headers:
            data = self._data(header)
            if all(
                _matches(_criterion_value(data, field), expected)
                for field, expected in criteria.items()
            ):
                selected.append(header)
        return self._with_headers(selected)

    def _data(self, header: Mapping[str, Any]) -> Data:
        return Data(
            self,
            self._store.get_case(self.name, self.release_id, str(header["id"])),
        )

    def _with_headers(self, headers: Iterable[Mapping[str, Any]]) -> Dataset:
        return Dataset(
            self._store,
            self._dataset_info,
            self._release,
            headers,
            archive_classes=self._archive_classes,
        )

    def _select_ids(self, case_ids: Iterable[str]) -> Dataset:
        wanted = set(case_ids)
        return self._with_headers(
            item for item in self._headers if item["id"] in wanted
        )

    def _select(self, data: Iterable[Data]) -> Dataset:
        return self._select_ids(item.record_id for item in data)

    def _filtered(self, predicate: Any) -> Dataset:
        return self._select(item for item in self if predicate(item))

    def __repr__(self) -> str:
        return (
            f"Dataset(name={self.name!r}, version={self.version!r}, "
            f"size={len(self)})"
        )


class Data:
    """One immutable stored Case projected into the concise read API."""

    def __init__(self, dataset: Dataset, value: Mapping[str, Any]) -> None:
        self.dataset = dataset
        self._value = deepcopy(dict(value))
        self._structures = Structures(self)

    @property
    def id(self) -> str:
        return (
            f"hp-{self.dataset.name}[{self.dataset.version_serial:03d}]-"
            f"{self._value['public_index']}"
        )

    @property
    def record_id(self) -> str:
        return str(self._value["id"])

    @property
    def local_id(self) -> str:
        return str(self._value["local_id"])

    @property
    def metadata(self) -> Dict[str, Any]:
        return deepcopy(self._value["facts"].get("metadata", {}))

    @property
    def identifiers(self) -> Dict[str, Any]:
        grouped: Dict[str, list[str]] = {}
        for item in self._value.get("identifiers", []):
            grouped.setdefault(str(item["namespace"]), []).append(str(item["value"]))
        return {
            key: values[0] if len(values) == 1 else values
            for key, values in grouped.items()
        }

    @property
    def entities(self) -> list[Dict[str, Any]]:
        return deepcopy(self._value["facts"].get("entities", []))

    @property
    def label(self) -> Dict[str, Any]:
        labels = self._value["facts"].get("labels")
        if isinstance(labels, dict):
            return deepcopy(labels)
        result: Dict[str, Any] = {}
        for item in self._value["facts"].get("observations", []):
            if item.get("status", "active") != "active":
                continue
            name = item.get("property") or item.get("property_type_id")
            if name is None:
                continue
            value = deepcopy(item.get("value"))
            if name in result:
                result[name] = (
                    result[name] + [value]
                    if isinstance(result[name], list)
                    else [result[name], value]
                )
            else:
                result[name] = value
        return result

    @property
    def quality_level(self) -> str:
        return str(self._value["assessment"]["quality_level"])

    @property
    def evidence_level(self) -> Optional[str]:
        return self._value["assessment"].get("evidence_level")

    @property
    def archive_class(self) -> str:
        return str(self._value["assessment"]["archive_class"])

    @property
    def info(self) -> Dict[str, Any]:
        target_structures = []
        for item in self._value.get("structures", []):
            structure = deepcopy(dict(item))
            structure["path"] = str(structure["path"])
            target_structures.append(structure)
        evidence_assets = []
        for item in self._value.get("evidence_assets", []):
            asset = deepcopy(dict(item))
            asset["path"] = str(asset["path"])
            evidence_assets.append(asset)
        return {
            "id": self.id,
            "record_id": self.record_id,
            "local_id": self.local_id,
            "dataset": self.dataset.name,
            "release_id": self.dataset.release_id,
            "metadata": self.metadata,
            "facts": deepcopy(self._value["facts"]),
            "labels": self.label,
            "identifiers": self.identifiers,
            "entities": self.entities,
            "structures": target_structures,
            "evidence_assets": evidence_assets,
            "quality_level": self.quality_level,
            "evidence_level": self.evidence_level,
            "archive_class": self.archive_class,
            "assessment": deepcopy(self._value["assessment"]),
        }

    @property
    def structures(self) -> Structures:
        return self._structures

    @property
    def molecule(self):
        molecule = self.structures.molecule()
        attachment = "ds_data" if hasattr(molecule, "data") else "data"
        setattr(molecule, attachment, self)
        return molecule

    @property
    def crystal(self):
        molecule = self.molecule
        crystal = getattr(molecule, "crystal", None)
        if crystal is None:
            from .structures import CrystalStructureUnavailableError

            raise CrystalStructureUnavailableError(
                f"data {self.id} has no loadable crystal information"
            )
        attachment = "ds_data" if hasattr(crystal, "data") else "data"
        setattr(crystal, attachment, self)
        return crystal

    def __repr__(self) -> str:
        return f"Data(id={self.id!r}, local_id={self.local_id!r})"


def _criterion_value(data: Data, field: str) -> Any:
    value = _value_at(data.info, field)
    if value is not _MISSING or "." in field or "__" in field:
        return value
    for mapping in (
        data.label,
        data.metadata,
        data.identifiers,
        data._value["facts"].get("core_info", {}),
    ):
        if field in mapping:
            return mapping[field]
    return _MISSING


__all__ = ["Data", "Dataset"]
