"""Existing deterministic graph query composition over explicit numeric topology."""

from itertools import combinations
from typing import Iterable

from .graph.components import connected_components
from .graph.paths import all_simple_paths, longest_simple_path


__all__ = ()


def _simple_paths(node_count, edges, source=None, target=None, cutoff=None):
    nodes = range(node_count)
    if source is None and target is None:
        st = combinations(nodes, 2)
    elif isinstance(source, int) and isinstance(target, int):
        st = [(source, target)]
    elif isinstance(source, int):
        st = [(source, t) for t in nodes if t != source]
    else:
        st = [(s, target) for s in nodes if s != target]
    return [path for start, end in st for path in all_simple_paths(node_count, edges, start, end, cutoff)]


def _longest_path(node_count: int, edges: Iterable[tuple[int, int]], _start_node: int=None, _end_node: int=None, _scope_nodes: Iterable[int]=None):
    scope = tuple(sorted(range(node_count) if _scope_nodes is None else _scope_nodes))
    if not scope:
        return []
    local_rows = {node: row for row, node in enumerate(scope)}
    local_edges = tuple(((local_rows[first], local_rows[second]) for first, second in edges if first in local_rows and second in local_rows))
    path = longest_simple_path(len(scope), local_edges, None if _start_node is None else local_rows[_start_node], None if _end_node is None else local_rows[_end_node])
    return [scope[row] for row in path]


def _canonical_tree(node_count, edges):

    def _dfs(scope_nodes: tuple[int, ...], scope_edges: tuple[tuple[int, int], ...], path_nodes: tuple[int, ...], _nested: dict):
        adjacency = {node: [] for node in scope_nodes}
        for first, second in scope_edges:
            adjacency[first].append(second)
            adjacency[second].append(first)
        branch_start = [(neighbor, parent) for parent in path_nodes for neighbor in sorted(adjacency[parent]) if neighbor not in path_nodes]
        if branch_start:
            removed = {(min(first, second), max(first, second)) for first, second in branch_start}
            retained_edges = tuple(((first, second) for first, second in scope_edges if (min(first, second), max(first, second)) not in removed))
        else:
            return
        local_rows = {node: row for row, node in enumerate(scope_nodes)}
        local_components = connected_components(len(scope_nodes), ((local_rows[first], local_rows[second]) for first, second in retained_edges))
        for local_component in local_components:
            connected_nodes = tuple((scope_nodes[row] for row in local_component))
            if not set(connected_nodes).difference(path_nodes):
                continue
            start_node, parent_node = next(((node, parent) for node, parent in branch_start if node in connected_nodes))
            sub_edges = tuple(((first, second) for first, second in retained_edges if first in connected_nodes and second in connected_nodes))
            longest_path = _longest_path(node_count, sub_edges, start_node, _scope_nodes=connected_nodes)
            list_nested = _nested.setdefault(parent_node, [])
            _sub_nested = {}
            list_nested.append((longest_path, _sub_nested))
            _dfs(connected_nodes, sub_edges, longest_path, _sub_nested)
    nested_tree = (_longest_path(node_count, edges), {})
    _dfs(tuple(range(node_count)), edges, nested_tree[0], nested_tree[1])
    return nested_tree
