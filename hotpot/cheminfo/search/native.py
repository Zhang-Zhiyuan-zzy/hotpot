"""Private adapters for Hotpot's native topology-search executor."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Iterable, Sequence


__all__ = ()


def _native_module() -> ModuleType:
    try:
        return import_module("hotpot.cheminfo.search._search_native")
    except ImportError as exc:
        raise ImportError(
            "the hotpot.cheminfo.search native extension is unavailable; "
            "install a compatible Hotpot wheel or rebuild Hotpot from source"
        ) from exc


def _synthetic_problem(
    query_keys: Sequence[int],
    query_edges: Sequence[tuple[int, int]],
    target_keys: Sequence[int],
    target_edges: Sequence[tuple[int, int]],
    node_compatible: Sequence[Sequence[bool]],
    edge_compatible: Sequence[Sequence[bool]],
):
    return _native_module().SyntheticProblem(
        query_keys,
        query_edges,
        target_keys,
        target_edges,
        node_compatible,
        edge_compatible,
    )


class _NativeMappingSource:
    """Adapt query-aligned native tuples to the legacy matcher direction."""

    def __init__(self, problem):
        self._problem = problem

    def subgraph_monomorphisms_iter(self) -> Iterable[dict[int, int]]:
        query_keys = self._problem.query_keys
        for target_values in self._problem.cursor():
            yield {
                target: query
                for query, target in zip(query_keys, target_values)
            }
