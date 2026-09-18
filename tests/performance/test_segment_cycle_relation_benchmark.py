"""Benchmark the geometry work performed by force-field frame checks.

This module intentionally contains no pytest tests.  Run it explicitly to
record timing and tri-state distribution evidence without creating a brittle
wall-clock CI threshold.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass
from math import cos, sin, tau
from pathlib import Path
from typing import Callable, Sequence, Tuple, TypeVar

import numpy as np

from hotpot.cheminfo.geometry import (
    Cycle,
    PiercingState,
    Segment,
    determine_bond_ring_piercing_state,
    determine_segment_cycle_relation,
    scan_bond_ring_relations,
)


ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class _Atom:
    idx: int
    coordinates: Tuple[float, float, float]


@dataclass(frozen=True)
class _Bond:
    atom1: _Atom
    atom2: _Atom


@dataclass(frozen=True)
class _Ring:
    atoms: Tuple[_Atom, ...]


@dataclass(frozen=True)
class _Molecule:
    atoms: Tuple[_Atom, ...]
    bonds: Tuple[_Bond, ...]
    rings: Tuple[_Ring, ...]

    def rings_for_scope(self, ring_scope: str) -> Sequence[_Ring]:
        return self.rings


def _percentile(samples: Sequence[float], fraction: float) -> float:
    ordered = sorted(samples)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _measure(function: Callable[[], ResultT], repeats: int) -> tuple[dict, ResultT]:
    samples = []
    result = function()
    for _ in range(repeats):
        started = time.perf_counter_ns()
        result = function()
        samples.append((time.perf_counter_ns() - started) / 1.0e6)
    return {
        "repeats": repeats,
        "median_ms": statistics.median(samples),
        "p95_ms": _percentile(samples, 0.95),
        "minimum_ms": min(samples),
        "maximum_ms": max(samples),
    }, result


def _regular_cycle(vertex_count: int, *, nonplanar: bool) -> Cycle:
    vertices = tuple(
        (
            2.0 * cos(tau * index / vertex_count),
            2.0 * sin(tau * index / vertex_count),
            0.2 * (1.0 if index % 2 else -1.0) if nonplanar else 0.0,
        )
        for index in range(vertex_count)
    )
    return Cycle(vertices)


def _scan_fixture() -> _Molecule:
    atoms = []
    bonds = []
    rings = []
    for ring_number, offset in enumerate((-3.0, 3.0)):
        ring_atoms = tuple(
            _Atom(
                idx=ring_number * 10 + index,
                coordinates=(
                    offset + 1.4 * cos(tau * index / 6),
                    1.4 * sin(tau * index / 6),
                    0.0,
                ),
            )
            for index in range(6)
        )
        atoms.extend(ring_atoms)
        bonds.extend(
            _Bond(ring_atoms[index], ring_atoms[(index + 1) % 6])
            for index in range(6)
        )
        rings.append(_Ring(ring_atoms))

    crossing_atoms = (
        _Atom(20, (-3.0, 0.0, -2.0)),
        _Atom(21, (-3.0, 0.0, 2.0)),
    )
    atoms.extend(crossing_atoms)
    bonds.insert(0, _Bond(*crossing_atoms))
    bonds.append(_Bond(rings[0].atoms[0], rings[1].atoms[3]))
    return _Molecule(tuple(atoms), tuple(bonds), tuple(rings))


def _relation_distribution() -> dict:
    planar = _regular_cycle(6, nonplanar=False)
    nonplanar = _regular_cycle(6, nonplanar=True)
    cases = (
        (Segment((0.0, 0.0, -2.0), (0.0, 0.0, 2.0)), planar),
        (Segment((3.0, 0.0, -2.0), (3.0, 0.0, 2.0)), planar),
        (Segment((0.0, 0.0, 0.0), (3.0, 0.0, 0.0)), planar),
        (Segment((0.0, 0.0, -2.0), (0.0, 0.0, 2.0)), nonplanar),
    )
    relations = tuple(
        determine_segment_cycle_relation(segment, cycle)
        for segment, cycle in cases
    )
    states = Counter(relation.state.value for relation in relations)
    causes = Counter(
        cause.value
        for relation in relations
        for cause in relation.indeterminacy_causes
    )
    return {
        "case_count": len(cases),
        "states": dict(sorted(states.items())),
        "indeterminacy_causes": dict(sorted(causes.items())),
    }


def run_benchmark(repeats: int) -> dict:
    pair_timings = {}
    pair_evidence = {}
    crossing = Segment((0.0, 0.0, -2.0), (0.0, 0.0, 2.0))
    for vertex_count in (6, 8):
        for geometry_kind, nonplanar in (("planar", False), ("nonplanar", True)):
            cycle = _regular_cycle(vertex_count, nonplanar=nonplanar)
            name = f"{geometry_kind}_{vertex_count}_member_ring"
            timing, relation = _measure(
                lambda cycle=cycle: determine_segment_cycle_relation(
                    crossing,
                    cycle,
                ),
                repeats,
            )
            pair_timings[name] = timing
            pair_evidence[name] = {
                "state": relation.state.value,
                "surface_model": (
                    None
                    if relation.surface_model is None
                    else relation.surface_model.value
                ),
                "surface_evidence": asdict(relation.surface_evidence),
            }

    mol = _scan_fixture()
    lazy_timing, lazy_state = _measure(
        lambda: determine_bond_ring_piercing_state(
            mol,
            ring_scope="full_graph",
            max_ring_size=8,
        ),
        repeats,
    )
    dense_timing, dense_report = _measure(
        lambda: scan_bond_ring_relations(
            mol,
            ring_scope="full_graph",
            max_ring_size=8,
        ),
        repeats,
    )

    return {
        "schema_version": 1,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "scope": {
            "coordinate_unit": "angstrom",
            "pair_ring_sizes": [6, 8],
            "frame_fixture_atom_count": len(mol.atoms),
            "frame_fixture_bond_count": len(mol.bonds),
            "frame_fixture_ring_count": len(mol.rings),
        },
        "single_pair": pair_timings,
        "single_pair_evidence": pair_evidence,
        "frame_relation_gate": {
            "lazy": {**lazy_timing, "state": lazy_state.value},
            "dense": {
                **dense_timing,
                "state_counts": {
                    PiercingState.PIERCES.value: dense_report.piercing_pair_count,
                    PiercingState.DOES_NOT_PIERCE.value: (
                        dense_report.does_not_pierce_pair_count
                    ),
                    PiercingState.UNDETERMINED.value: (
                        dense_report.undetermined_pair_count
                    ),
                },
                "candidate_pair_count": dense_report.candidate_pair_count,
                "scan_complete": dense_report.scan_complete,
            },
        },
        "curated_relation_distribution": _relation_distribution(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record Segment-Cycle and frame-scan performance evidence.",
    )
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = run_benchmark(arguments.repeats)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
