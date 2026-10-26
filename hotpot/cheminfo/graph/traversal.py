"""Deterministic traversal helpers over explicit integer topology."""

from collections.abc import Container, Iterable
from typing import Optional

from .paths import depth_first_path


__all__ = ("graph_dfs_path", "graph_dfs_paths")


def graph_dfs_path(
        node_count: int,
        edges: Iterable[tuple[int, int]],
        start_node: int = None,
        scope_nodes: Container = None,
        min_deep: int = None,
        max_deep: int = None
) -> Optional[list[int]]:
    """Return the legacy one-branch DFS event over an explicit edge range."""
    return depth_first_path(
        node_count,
        edges,
        start_node=0 if start_node is None else start_node,
        scope_nodes=scope_nodes,
        min_depth=min_deep,
        max_depth=max_deep,
    )


def graph_dfs_paths(
        node_count: int,
        edges: Iterable[tuple[int, int]],
        start_node: int,
        scope_nodes: Container = None,
        min_deep: int = None,
        max_deep: int = None
) -> list[list[int]]:
    """Retain the historical empty-result contract without constructing Graph."""
    return []
