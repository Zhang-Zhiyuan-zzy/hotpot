"""Differential validation against a fixed RingDecomposerLib oracle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

import pytest

from hotpot.cheminfo.graph import relevant_cycles

from .rdl_oracle import RDLOracle, load_oracle_from_environment
from .rdl_oracle.corpus import (
    deterministic_random_graph_cases,
    golden_graph_cases,
    graph_atlas_cases,
    handcrafted_graph_cases,
    named_graph_cases,
    pubchem_graph_cases,
)

RDL_COMMIT = "3a7ff93de0d9c4f6a5661508549c6063573f39c7"
GOLDEN_PATH = Path(__file__).with_name("rdl_oracle") / "golden_relevant_cycles.json"


def _edge_digest(edges: Iterable[Sequence[int]]) -> str:
    normalized = sorted(
        (min(node1, node2), max(node1, node2)) for node1, node2 in edges
    )
    payload = ";".join(f"{node1},{node2}" for node1, node2 in normalized)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _live_oracle() -> RDLOracle:
    oracle = load_oracle_from_environment()
    if oracle is None:
        pytest.skip("set HOTPOT_RDL_ORACLE_LIBRARY to run live RDL comparisons")
    return oracle


def test_relevant_cycles_match_pinned_rdl_golden_corpus():
    payload = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    assert payload["oracle"]["commit"] == RDL_COMMIT

    graph_cases = golden_graph_cases()
    for record in payload["cases"]:
        edges = graph_cases[record["name"]]
        assert _edge_digest(edges) == record["edge_sha256"]
        expected = tuple(tuple(cycle) for cycle in record["cycles"])
        assert relevant_cycles(edges, max_cycles=None) == expected, record["name"]


@pytest.mark.slow
def test_live_rdl_matches_named_and_pubchem_graphs():
    oracle = _live_oracle()
    cases = handcrafted_graph_cases() + named_graph_cases() + pubchem_graph_cases()
    for name, edges in cases:
        expected = oracle.relevant_cycles(edges, max_cycles=None)
        assert relevant_cycles(edges, max_cycles=None) == expected, name


@pytest.mark.slow
def test_live_rdl_matches_size_bounded_results():
    oracle = _live_oracle()
    cases = handcrafted_graph_cases() + named_graph_cases() + pubchem_graph_cases()
    for name, edges in cases:
        for max_size in (3, 4, 5, 6, 8, 12):
            expected = oracle.relevant_cycles(
                edges,
                max_size=max_size,
                max_cycles=None,
            )
            actual = relevant_cycles(
                edges,
                max_size=max_size,
                max_cycles=None,
            )
            assert actual == expected, (name, max_size)


@pytest.mark.slow
def test_live_rdl_matches_every_cyclic_networkx_graph_atlas_graph():
    oracle = _live_oracle()
    compared = 0
    for name, edges in graph_atlas_cases():
        expected = oracle.relevant_cycles(edges, max_cycles=None)
        assert relevant_cycles(edges, max_cycles=None) == expected, name
        compared += 1
    assert compared == 1_173


@pytest.mark.slow
def test_live_rdl_matches_deterministic_random_graph_stress_corpus():
    oracle = _live_oracle()
    cases = tuple(deterministic_random_graph_cases())
    assert len(cases) == 1_000
    reference_cycle_count = 0
    for name, edges in cases:
        expected = oracle.relevant_cycles(edges, max_cycles=None)
        assert relevant_cycles(edges, max_cycles=None) == expected, name
        reference_cycle_count += len(expected)
    assert reference_cycle_count == 4_114
