"""Python/native translation and ownership for the topology-search executor."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from types import ModuleType
from typing import Iterable

__all__ = ()


def _native_module() -> ModuleType:
    try:
        return import_module("hotpot.cheminfo.adapters.native.substructure._search_native")
    except ImportError as exc:
        raise ImportError(
            "the hotpot.cheminfo.adapters.native.substructure native extension is unavailable; "
            "install a compatible Hotpot wheel or rebuild Hotpot from source"
        ) from exc


@dataclass(frozen=True)
class _NativeAutomorphismTask:
    _native: object
    keys: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    node_labels: tuple[int, ...]
    edge_labels: tuple[int, ...]

    def equivalent(self, first: int, second: int) -> bool:
        return self._native.equivalent(first, second)


class _NativeMappingSource:
    """Expose native embeddings as query-to-molecule mappings."""

    def __init__(self, problem):
        self._problem = problem

    def iter_mappings(self) -> Iterable[dict[int, int]]:
        query_keys = self._problem.query_keys
        for target_values in self._problem.cursor():
            yield dict(zip(query_keys, target_values))

    def has_match(self) -> bool:
        """Probe one native embedding without materializing a mapping."""
        return self._problem.has_match()

    def mapped_target_indices(self, query_key: int) -> Iterable[int]:
        """Project one query key directly from each raw native embedding."""
        position = self._problem.query_keys.index(query_key)
        for target_values in self._problem.cursor():
            yield target_values[position]


class _NativeTarget:
    """Own one packed native target shared by all task-specific queries."""

    def __init__(self, target_spec: dict) -> None:
        self._native_module = _native_module()
        self._target = self._native_module.NativeTarget(target_spec)

    def problem(self, query_spec: dict):
        return self._native_module.NativeProblem(query_spec, self._target)

    def mapping_source(self, query_spec: dict) -> _NativeMappingSource:
        return _NativeMappingSource(self.problem(query_spec))

    def iter_mappings(self, query_spec: dict) -> Iterable[dict[int, int]]:
        return self.mapping_source(query_spec).iter_mappings()

    def automorphism(self, node_labels, edge_labels):
        return self._native_module._NativeAutomorphism(
            self._target,
            node_labels,
            edge_labels,
        )
