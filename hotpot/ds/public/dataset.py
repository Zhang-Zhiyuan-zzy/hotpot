"""Small, user-facing dataset and data objects backed by the governed store."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from functools import cached_property
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

from ..levels import EvidenceLevel, QualityLevel
from .structures import Structures


_MISSING = object()


def _json(value: str) -> Any:
    return json.loads(value)


def _level_rank(value: str) -> int:
    try:
        return tuple(level.value for level in QualityLevel).index(value)
    except ValueError:
        return -1


def _evidence_rank(value: str) -> int:
    try:
        return tuple(level.value for level in EvidenceLevel).index(value)
    except ValueError:
        return -1


def _criterion_matches(actual: Any, expected: Any) -> bool:
    expected_values = (
        tuple(expected)
        if isinstance(expected, (list, tuple, set, frozenset))
        else (expected,)
    )
    actual_values = (
        tuple(actual)
        if isinstance(actual, (list, tuple, set, frozenset))
        else (actual,)
    )
    return any(actual_value == expected_value for actual_value in actual_values for expected_value in expected_values)


class DatasetInvariantError(RuntimeError):
    """Stored public dataset metadata contradicts its declared invariant."""


class IdentifierIndex:
    """A dynamic record index for one external-identifier namespace."""

    def __init__(self, dataset: "Dataset", namespace: str, public_name: str) -> None:
        self.dataset = dataset
        self.namespace = namespace
        self.public_name = public_name

    def getall(self, value: str) -> "Dataset":
        connection = self.dataset._store._connect()
        try:
            rows = connection.execute(
                """SELECT DISTINCT re.record_id FROM external_identifiers x
                JOIN record_entities re ON re.entity_id=x.entity_id
                JOIN records rec ON rec.record_id=re.record_id
                WHERE rec.release_id=? AND x.namespace=? AND x.value=?
                ORDER BY re.record_id""",
                (self.dataset.release_id, self.namespace, value),
            ).fetchall()
        finally:
            connection.close()
        wanted = {row["record_id"] for row in rows}
        return self.dataset._select(
            data for data in self.dataset if data.record_id in wanted
        )

    def __getitem__(self, value: str) -> Union["Data", "Dataset"]:
        matches = self.getall(value)
        if not matches:
            raise KeyError(value)
        return matches[0] if len(matches) == 1 else matches

    def __iter__(self) -> Iterator[str]:
        visible = {data.record_id for data in self.dataset}
        connection = self.dataset._store._connect()
        try:
            rows = connection.execute(
                """SELECT DISTINCT x.value,re.record_id FROM external_identifiers x
                JOIN record_entities re ON re.entity_id=x.entity_id
                JOIN records rec ON rec.record_id=re.record_id
                WHERE rec.release_id=? AND x.namespace=? ORDER BY x.value""",
                (self.dataset.release_id, self.namespace),
            ).fetchall()
        finally:
            connection.close()
        yield from sorted({row["value"] for row in rows if row["record_id"] in visible})

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __repr__(self) -> str:
        return f"IdentifierIndex({self.public_name!r}, count={len(self)})"


class Dataset(Sequence["Data"]):
    """One immutable dataset release, optionally narrowed to a record view."""

    def __init__(
        self,
        store,
        dataset_row: Mapping[str, Any],
        release_row: Mapping[str, Any],
        version_serial: int,
        records: Sequence[Tuple[int, Mapping[str, Any]]],
    ) -> None:
        self._store = store
        self._dataset_row = dict(dataset_row)
        self._release_row = dict(release_row)
        self._version_serial = version_serial
        self._records = tuple((index, dict(row)) for index, row in records)

    @property
    def name(self) -> str:
        return self._dataset_row["dataset_id"]

    @property
    def dataset_id(self) -> str:
        return self.name

    @property
    def release_id(self) -> str:
        return self._release_row["release_id"]

    @property
    def version(self) -> str:
        return self._release_row["version"]

    @property
    def version_serial(self) -> int:
        return self._version_serial

    @property
    def quality_level(self) -> str:
        return self._release_row["quality_level"]

    def __len__(self) -> int:
        return len(self._records)

    @cached_property
    def _items(self) -> Tuple["Data", ...]:
        return tuple(Data(self, index, row) for index, row in self._records)

    def __iter__(self) -> Iterator["Data"]:
        return iter(self._items)

    def __getitem__(self, index: Union[int, slice]) -> Union["Data", "Dataset"]:
        if isinstance(index, slice):
            return Dataset(
                self._store,
                self._dataset_row,
                self._release_row,
                self._version_serial,
                self._records[index],
            )
        return self._items[index]

    def __repr__(self) -> str:
        return f"Dataset(name={self.name!r}, version={self.version!r}, size={len(self)})"

    def _select(self, items: Iterable[Union["Data", str, int]]) -> "Dataset":
        selected: List[Tuple[int, Mapping[str, Any]]] = []
        by_record = {row["record_id"]: (index, row) for index, row in self._records}
        by_index = {index: (index, row) for index, row in self._records}
        seen = set()
        for item in items:
            if isinstance(item, Data):
                key = item.record_id
                candidate = by_record.get(key)
            elif isinstance(item, int):
                candidate = by_index.get(item)
            else:
                candidate = by_record.get(item)
            if candidate is not None and candidate[0] not in seen:
                seen.add(candidate[0])
                selected.append(candidate)
        return Dataset(
            self._store,
            self._dataset_row,
            self._release_row,
            self._version_serial,
            selected,
        )

    def _filtered(self, predicate: Callable[["Data"], bool]) -> "Dataset":
        return self._select(data for data in self if predicate(data))

    @cached_property
    def _labels_by_record(self) -> Dict[str, Dict[str, Any]]:
        connection = self._store._connect()
        try:
            rows = connection.execute(
                """SELECT o.record_id,o.property_type_id,o.value_json FROM observations o
                JOIN records rec ON rec.record_id=o.record_id
                WHERE rec.release_id=? AND o.status='active'
                ORDER BY o.record_id,o.observation_id""",
                (self.release_id,),
            ).fetchall()
        finally:
            connection.close()
        grouped: Dict[str, Dict[str, List[Any]]] = defaultdict(lambda: defaultdict(list))
        for row in rows:
            grouped[row["record_id"]][row["property_type_id"]].append(
                _json(row["value_json"])
            )
        return {
            record_id: {
                name: values[0] if len(values) == 1 else values
                for name, values in labels.items()
            }
            for record_id, labels in grouped.items()
        }

    @cached_property
    def labels(self) -> Dict[str, int]:
        visible = {row["record_id"] for _, row in self._records}
        counts: Counter = Counter()
        for record_id, labels in self._labels_by_record.items():
            if record_id in visible:
                counts.update(labels.keys())
        return {
            label: count for label, count in sorted(counts.items())
        }

    @cached_property
    def _identifiers_by_record(self) -> Dict[str, Dict[str, Any]]:
        connection = self._store._connect()
        try:
            rows = connection.execute(
                """SELECT re.record_id,x.namespace,x.value
                FROM external_identifiers x JOIN record_entities re USING(entity_id)
                JOIN records rec USING(record_id) WHERE rec.release_id=?
                ORDER BY re.record_id,x.namespace,x.value""",
                (self.release_id,),
            ).fetchall()
        finally:
            connection.close()
        grouped: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        for row in rows:
            values = grouped[row["record_id"]][row["namespace"]]
            if row["value"] not in values:
                values.append(row["value"])
        return {
            record_id: {
                namespace: values[0] if len(values) == 1 else values
                for namespace, values in identifiers.items()
            }
            for record_id, identifiers in grouped.items()
        }

    @cached_property
    def _quality_by_record(self) -> Dict[str, str]:
        levels: Dict[str, List[str]] = {
            row["record_id"]: [self.quality_level] for _, row in self._records
        }
        connection = self._store._connect()
        try:
            rows = connection.execute(
                """SELECT m.record_id,s.quality_level FROM subset_memberships m
                JOIN subsets s USING(subset_id) WHERE s.release_id=?""",
                (self.release_id,),
            ).fetchall()
        finally:
            connection.close()
        for row in rows:
            if row["record_id"] in levels:
                levels[row["record_id"]].append(row["quality_level"])
        return {
            record_id: max(values, key=_level_rank)
            for record_id, values in levels.items()
        }

    @cached_property
    def _evidence_by_record(self) -> Dict[str, str]:
        levels: Dict[str, List[str]] = {
            row["record_id"]: [] for _, row in self._records
        }
        connection = self._store._connect()
        try:
            rows = connection.execute(
                """SELECT o.record_id,o.evidence_level FROM observations o
                JOIN records rec USING(record_id)
                WHERE rec.release_id=? AND o.status='active'
                UNION ALL
                SELECT c.record_id,c.evidence_level FROM claims c
                JOIN records rec USING(record_id)
                WHERE rec.release_id=? AND c.status!='rejected'""",
                (self.release_id, self.release_id),
            ).fetchall()
        finally:
            connection.close()
        for row in rows:
            if row["record_id"] in levels:
                levels[row["record_id"]].append(row["evidence_level"])
        return {
            record_id: min(values, key=_evidence_rank)
            if values
            else EvidenceLevel.E0.value
            for record_id, values in levels.items()
        }

    @cached_property
    def core_labels(self) -> List[str]:
        core = list(self._release_row["core_labels"])
        incomplete = [name for name in core if self.labels.get(name, 0) != len(self)]
        if incomplete:
            raise DatasetInvariantError(
                f"release {self.release_id!r} declares incomplete core labels: {incomplete}"
            )
        return core

    @cached_property
    def handle(self) -> List[str]:
        visible = {row["record_id"] for _, row in self._records}
        connection = self._store._connect()
        try:
            rows = connection.execute(
                """SELECT o.record_id,s.source_id,s.citation,s.doi
                FROM observations o JOIN observation_evidence oe USING(observation_id)
                JOIN evidence_items ev USING(evidence_id) JOIN sources s USING(source_id)
                JOIN records rec ON rec.record_id=o.record_id WHERE rec.release_id=?
                UNION ALL
                SELECT c.record_id,s.source_id,s.citation,s.doi
                FROM claims c JOIN claim_evidence ce USING(claim_id)
                JOIN evidence_items ev USING(evidence_id) JOIN sources s USING(source_id)
                JOIN records rec ON rec.record_id=c.record_id WHERE rec.release_id=?""",
                (self.release_id, self.release_id),
            ).fetchall()
        finally:
            connection.close()
        return sorted(
            {
                row["doi"] or row["citation"]
                for row in rows
                if row["record_id"] in visible
            }
        )

    @cached_property
    def identifier_namespaces(self) -> Tuple[str, ...]:
        connection = self._store._connect()
        try:
            rows = connection.execute(
                """SELECT DISTINCT x.namespace FROM external_identifiers x
                JOIN record_entities re ON re.entity_id=x.entity_id
                JOIN records rec ON rec.record_id=re.record_id
                WHERE rec.release_id=? ORDER BY x.namespace""",
                (self.release_id,),
            ).fetchall()
        finally:
            connection.close()
        return tuple(row["namespace"] for row in rows)

    def _identifier_namespace(self, public_prefix: str) -> Optional[str]:
        public_name = f"{public_prefix}_identifier"
        canonical = [
            namespace
            for namespace in self.identifier_namespaces
            if namespace.lower() == public_name.lower()
        ]
        if canonical:
            return canonical[0]
        exact = [
            namespace
            for namespace in self.identifier_namespaces
            if namespace.lower() == public_prefix.lower()
        ]
        if exact:
            return exact[0]
        matches = [
            namespace
            for namespace in self.identifier_namespaces
            if namespace.lower().startswith(public_prefix.lower() + "_")
        ]
        return matches[0] if len(matches) == 1 else None

    def __getattr__(self, name: str):
        if name.endswith("_identifier"):
            prefix = name[: -len("_identifier")]
            namespace = self._identifier_namespace(prefix)
            if namespace is not None:
                return IdentifierIndex(self, namespace, name)
        raise AttributeError(name)

    def __dir__(self) -> List[str]:
        dynamic = {
            f"{namespace.split('_', 1)[0]}_identifier"
            for namespace in self.identifier_namespaces
        }
        return sorted(set(super().__dir__()).union(dynamic))

    def subset(self, **criteria: Any) -> "Dataset":
        """Return records matching every criterion; collection values mean ``IN``."""
        def matches(data: "Data") -> bool:
            for field, expected in criteria.items():
                actual = data._lookup(field)
                if actual is _MISSING or not _criterion_matches(actual, expected):
                    return False
            return True

        return self._filtered(matches)

    @cached_property
    def info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "title": self._dataset_row["title"],
            "description": self._dataset_row["description"],
            "release_id": self.release_id,
            "version": self.version,
            "version_serial": f"{self.version_serial:03d}",
            "size": len(self),
            "quality_level": self.quality_level,
            "license": self._release_row["license"],
            "access_policy": self._release_row["access_policy"],
            "core_labels": self.core_labels,
            "labels": self.labels,
            "handle": self.handle,
            "identifier_namespaces": list(self.identifier_namespaces),
            "provenance": self._release_row["provenance"],
            "quality": self._release_row["quality"],
        }


class Data:
    """One record in a :class:`Dataset` release."""

    def __init__(self, dataset: Dataset, index: int, row: Mapping[str, Any]) -> None:
        self.dataset = dataset
        self._row = dict(row)
        self.index = int(self._row.get("public_index", index))

    @property
    def id(self) -> str:
        return f"hp-{self.dataset.name}[{self.dataset.version_serial:03d}]-{self.index}"

    @property
    def record_id(self) -> str:
        return self._row["record_id"]

    @property
    def local_id(self) -> str:
        return self._row["local_id"]

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._row["metadata"]

    def __repr__(self) -> str:
        return f"Data(id={self.id!r}, local_id={self.local_id!r})"

    @cached_property
    def label(self) -> Dict[str, Any]:
        return dict(self.dataset._labels_by_record.get(self.record_id, {}))

    @cached_property
    def identifiers(self) -> Dict[str, Any]:
        return dict(self.dataset._identifiers_by_record.get(self.record_id, {}))

    @cached_property
    def entities(self) -> List[Dict[str, Any]]:
        connection = self.dataset._store._connect()
        try:
            rows = connection.execute(
                """SELECT e.entity_id,e.entity_kind,e.parent_entity_id,e.metadata_json,re.role
                FROM record_entities re JOIN entities e USING(entity_id)
                WHERE re.record_id=? ORDER BY re.role,e.entity_id""",
                (self.record_id,),
            ).fetchall()
        finally:
            connection.close()
        return [
            {
                "entity_id": row["entity_id"],
                "entity_kind": row["entity_kind"],
                "parent_entity_id": row["parent_entity_id"],
                "role": row["role"],
                "metadata": _json(row["metadata_json"]),
            }
            for row in rows
        ]

    @cached_property
    def quality_level(self) -> str:
        return self.dataset._quality_by_record[self.record_id]

    @cached_property
    def evidence_level(self) -> str:
        return self.dataset._evidence_by_record[self.record_id]

    @cached_property
    def structures(self) -> Structures:
        return Structures(self)

    @property
    def molecule(self):
        return self.structures.load_molecule()

    @property
    def crystal(self):
        return self.structures.load_crystal()

    def _lookup(self, field: str) -> Any:
        if field == "id":
            return self.id
        if field == "record_id":
            return self.record_id
        if field == "local_id":
            return self.local_id
        if field == "quality_level":
            return self.quality_level
        if field == "evidence_level":
            return self.evidence_level
        if field in self.dataset.labels:
            return self.label.get(field, _MISSING)
        if field in self.metadata:
            return self.metadata[field]
        if field in self.dataset.identifier_namespaces:
            return self.identifiers.get(field, _MISSING)
        if field.endswith("_identifier"):
            namespace = self.dataset._identifier_namespace(field[: -len("_identifier")])
            if namespace is not None:
                return self.identifiers.get(namespace, _MISSING)
        current: Any = self.info
        for part in field.replace("__", ".").split("."):
            if not isinstance(current, Mapping) or part not in current:
                return _MISSING
            current = current[part]
        return current

    @cached_property
    def info(self) -> Dict[str, Any]:
        connection = self.dataset._store._connect()
        try:
            observation_ids = [
                row["observation_id"]
                for row in connection.execute(
                    "SELECT observation_id FROM observations WHERE record_id=? ORDER BY observation_id",
                    (self.record_id,),
                )
            ]
            claim_ids = [
                row["claim_id"]
                for row in connection.execute(
                    "SELECT claim_id FROM claims WHERE record_id=? ORDER BY claim_id",
                    (self.record_id,),
                )
            ]
        finally:
            connection.close()
        observations = [
            self.dataset._store.get_observation(identity) for identity in observation_ids
        ]
        claims = [self.dataset._store.get_claim(identity) for identity in claim_ids]
        return {
            "id": self.id,
            "dataset": self.dataset.name,
            "dataset_version": self.dataset.version,
            "record_id": self.record_id,
            "local_id": self.local_id,
            "quality_level": self.quality_level,
            "evidence_level": self.evidence_level,
            "label": self.label,
            "labels": self.label,
            "identifiers": self.identifiers,
            "metadata": self.metadata,
            "entities": self.entities,
            "structures": self.structures.info,
            "observations": observations,
            "claims": claims,
        }


__all__ = [
    "Data",
    "Dataset",
    "DatasetInvariantError",
    "IdentifierIndex",
]
