"""NetworkX-backed graph traversal helpers."""

from typing import Container, Optional

import networkx as nx


__all__ = ("graph_dfs_path", "graph_dfs_paths")


def graph_dfs_path(
        graph: nx.Graph,
        start_node: int = None,
        scope_nodes: Container = None,
        min_deep: int = None,
        max_deep: int = None
) -> Optional[list[int]]:
    """"""
    def _dfs(_node: int, visited: list[int]):
        visited.append(_node)
        if max_deep and len(visited) >= max_deep:
            return visited

        for child in nx.neighbors(graph, _node):
            if (child not in visited) and (scope_nodes and child in scope_nodes):
                return _dfs(child, visited)

        if min_deep and len(visited) >= min_deep:
            return visited

    if start_node is None:
        start_node = 0

    return _dfs(start_node, [])


def graph_dfs_paths(
        graph: nx.Graph,
        start_node: int,
        scope_nodes: Container = None,
        min_deep: int = None,
        max_deep: int = None
) -> list[list[int]]:
    paths = []

    def _dfs(node: int, visited: set[int], path: list[int]) -> None:
        path.append(node)
        visited.add(node)

        if max_deep and len(visited) >= max_deep:
            paths.append(path)
            return

        for child in nx.neighbors(graph, node):
            if (child not in visited) and (scope_nodes and child in scope_nodes):
                _dfs(child, visited, path)

        if min_deep and len(visited) >= min_deep:
            paths.append(path)

    if start_node is None:
        start_node = 0

    return paths
