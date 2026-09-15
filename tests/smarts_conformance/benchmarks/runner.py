"""Trend-oriented SMARTS parser, matcher, batch, and concurrency benchmark."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import networkx

from hotpot.cheminfo.core_utils import read_mol
from hotpot.cheminfo.search.search import Searcher, Substructure
from hotpot.cheminfo.search.smarts import tokenize


def _percentile(samples: Sequence[int], fraction: float) -> int:
    ordered = sorted(samples)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def measure(
    operation: Callable[[], object],
    *,
    repeats: int,
    warmups: int = 1,
) -> Dict[str, object]:
    for _ in range(warmups):
        operation()
    gc.collect()
    tracemalloc.start()
    samples = []
    checksums = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        result = operation()
        samples.append(time.perf_counter_ns() - started)
        checksums.append(_stable_digest(result))
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "repeats": repeats,
        "median_ms": statistics.median(samples) / 1_000_000,
        "p90_ms": _percentile(samples, 0.90) / 1_000_000,
        "min_ms": min(samples) / 1_000_000,
        "max_ms": max(samples) / 1_000_000,
        "peak_traced_bytes": peak_bytes,
        "result_deterministic": len(set(checksums)) == 1,
        "samples_ms": [sample / 1_000_000 for sample in samples],
    }


def measure_fresh_match(
    setup: Callable[[], Tuple[Searcher, object]],
    *,
    repeats: int,
) -> Dict[str, object]:
    samples = []
    checksums = []
    peaks = []
    for _ in range(repeats):
        searcher, molecule = setup()
        gc.collect()
        tracemalloc.start()
        started = time.perf_counter_ns()
        result = _hit_summary(searcher.search(molecule))
        samples.append(time.perf_counter_ns() - started)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peaks.append(peak)
        checksums.append(_stable_digest(result))
    return {
        "repeats": repeats,
        "median_ms": statistics.median(samples) / 1_000_000,
        "p90_ms": _percentile(samples, 0.90) / 1_000_000,
        "min_ms": min(samples) / 1_000_000,
        "max_ms": max(samples) / 1_000_000,
        "peak_traced_bytes": max(peaks),
        "result_deterministic": len(set(checksums)) == 1,
        "samples_ms": [sample / 1_000_000 for sample in samples],
    }


def _stable_result(result: object) -> object:
    if isinstance(result, list):
        return tuple(_stable_result(item) for item in result)
    if isinstance(result, tuple):
        return tuple(_stable_result(item) for item in result)
    if isinstance(result, dict):
        return tuple(
            sorted((key, _stable_result(value)) for key, value in result.items())
        )
    if hasattr(result, "query_atoms") and hasattr(result, "query_bonds"):
        return len(result.query_atoms), len(result.query_bonds)
    return result


def _stable_digest(result: object) -> str:
    payload = json.dumps(
        _stable_result(result), sort_keys=True, separators=(",", ":"), default=repr
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _linear_query(atom_count: int) -> str:
    return "-".join("C" for _ in range(atom_count))


def _alkane(atom_count: int) -> str:
    return "C" * atom_count


def _disconnected_atoms(atom: str, atom_count: int) -> str:
    return ".".join(atom for _ in range(atom_count))


def _nested_branch(depth: int) -> str:
    return "C(" * depth + "C" + ")" * depth


def _nested_recursive(depth: int) -> str:
    query = "C"
    for _ in range(depth):
        query = f"[C;$({query})]"
    return query


def _logic_chain(width: int) -> str:
    return "[" + ";".join("#6,#7" for _ in range(width)) + "]"


def _multi_ring(ring_count: int) -> str:
    return ".".join("C1CCCCC1" for _ in range(ring_count))


def _hit_summary(hits: object) -> Tuple[int, int]:
    return len(hits), sum(len(hit.mappings) for hit in hits)


def _batch_summary(
    searcher: Searcher, molecules: Iterable[object]
) -> Tuple[Tuple[int, int], ...]:
    return tuple(_hit_summary(searcher.search(molecule)) for molecule in molecules)


def _concurrent_summary(
    searcher: Searcher,
    molecules: Sequence[object],
    workers: int,
) -> Tuple[Tuple[int, int], ...]:
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return tuple(
            executor.map(
                lambda molecule: _hit_summary(searcher.search(molecule)), molecules
            )
        )


def _concurrent_compile_summary(
    queries: Sequence[str], workers: int
) -> Tuple[Tuple[int, int], ...]:
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return tuple(executor.map(_compile_counts, queries))


def _compile_counts(query_text: str) -> Tuple[int, int]:
    query = Substructure.from_smarts(query_text)
    return len(query.query_atoms), len(query.query_bonds)


def _serial_compile_summary(queries: Sequence[str]) -> Tuple[Tuple[int, int], ...]:
    return tuple(map(_compile_counts, queries))


def _trend_rows(
    sizes: Sequence[int],
    operation: Callable[[int], Callable[[], object]],
    *,
    repeats: int,
    warmups: int = 1,
) -> List[Dict[str, object]]:
    rows = []
    for size in sizes:
        stats = measure(operation(size), repeats=repeats, warmups=warmups)
        rows.append({"size": size, **stats})
    for previous, current in zip(rows, rows[1:]):
        previous_time = float(previous["median_ms"])
        current["median_growth_from_previous"] = (
            None
            if previous_time == 0.0
            else float(current["median_ms"]) / previous_time
        )
    return rows


def run_benchmarks(profile: str, workers: int) -> Dict[str, object]:
    heavy = profile == "heavy"
    repeats = 15 if heavy else 5
    query_sizes = (8, 16, 32, 64) if heavy else (2, 4, 8, 16)
    target_sizes = (32, 64, 128, 256) if heavy else (8, 16, 32, 64)
    batch_sizes = (64, 256, 1_024) if heavy else (8, 32, 128)
    disconnected_sizes = (5, 6, 7, 8) if heavy else (4, 5, 6, 7)

    parse = _trend_rows(
        query_sizes,
        lambda size: lambda: len(tokenize(_linear_query(size))),
        repeats=repeats,
    )
    compile_query = _trend_rows(
        query_sizes,
        lambda size: lambda: Substructure.from_smarts(_linear_query(size)),
        repeats=repeats,
    )

    match_query_text = "[C]-[C]"

    def fresh_setup() -> Tuple[Searcher, object]:
        query = Substructure.from_smarts(match_query_text)
        molecule = read_mol(_alkane(max(target_sizes)), fmt="smi")
        return Searcher(query), molecule

    first_match = measure_fresh_match(fresh_setup, repeats=repeats)

    repeated_query = Substructure.from_smarts(match_query_text)
    repeated_searcher = Searcher(repeated_query)
    repeated_target = read_mol(_alkane(max(target_sizes)), fmt="smi")
    repeated_match = measure(
        lambda: _hit_summary(repeated_searcher.search(repeated_target)),
        repeats=repeats,
        warmups=2,
    )

    enumeration_targets = {
        size: read_mol(_alkane(size), fmt="smi") for size in target_sizes
    }
    enumeration_query = Substructure.from_smarts(match_query_text)
    enumeration_searcher = Searcher(enumeration_query)
    enumeration = _trend_rows(
        target_sizes,
        lambda size: (
            lambda: _hit_summary(enumeration_searcher.search(enumeration_targets[size]))
        ),
        repeats=repeats,
    )

    disconnected_enumeration = []
    for size in disconnected_sizes:
        query = Substructure.from_smarts(_disconnected_atoms("*", size))
        searcher = Searcher(query)
        target = read_mol(_disconnected_atoms("C", size), fmt="smi")
        observed = _hit_summary(searcher.search(target))
        stats = measure(
            lambda searcher=searcher, target=target: _hit_summary(
                searcher.search(target)
            ),
            repeats=3 if heavy else repeats,
        )
        disconnected_enumeration.append(
            {
                "size": size,
                "expected_raw_embeddings": math.factorial(size),
                "observed_hit_count": observed[0],
                "observed_raw_embeddings": observed[1],
                **stats,
            }
        )
    for previous, current in zip(
        disconnected_enumeration, disconnected_enumeration[1:]
    ):
        current["median_growth_from_previous"] = float(current["median_ms"]) / float(
            previous["median_ms"]
        )

    structure_sizes = (2, 4, 8, 12) if heavy else (2, 4, 6, 8)
    deep_branch_compile = _trend_rows(
        structure_sizes,
        lambda size: lambda: Substructure.from_smarts(_nested_branch(size)),
        repeats=repeats,
    )
    recursive_compile = _trend_rows(
        structure_sizes,
        lambda size: lambda: Substructure.from_smarts(_nested_recursive(size)),
        repeats=repeats,
    )
    logic_compile = _trend_rows(
        structure_sizes,
        lambda size: lambda: Substructure.from_smarts(_logic_chain(size)),
        repeats=repeats,
    )
    multi_ring_compile = _trend_rows(
        structure_sizes,
        lambda size: lambda: Substructure.from_smarts(_multi_ring(size)),
        repeats=repeats,
    )

    near_match_sizes = (12, 24, 48, 96) if heavy else (8, 16, 32, 48)
    near_match_pairs = {
        size: (
            Searcher(Substructure.from_smarts("C" * (size - 1) + "N")),
            read_mol("C" * size, fmt="smi"),
        )
        for size in near_match_sizes
    }
    near_match_failure = _trend_rows(
        near_match_sizes,
        lambda size: (
            lambda: _hit_summary(
                near_match_pairs[size][0].search(near_match_pairs[size][1])
            )
        ),
        repeats=3 if heavy else repeats,
    )

    molecule_pool = tuple(
        read_mol(smiles, fmt="smi")
        for smiles in ("C", "CC", "CCO", "CCN", "CCC", "C=O", "c1ccccc1", "C1CCCCC1")
    )
    batch_query = Substructure.from_smarts("[C,N]-[C,O]")
    batch_searcher = Searcher(batch_query)
    batch_molecules = {
        size: tuple(molecule_pool[index % len(molecule_pool)] for index in range(size))
        for size in batch_sizes
    }
    linear_batch = _trend_rows(
        batch_sizes,
        lambda size: lambda: _batch_summary(batch_searcher, batch_molecules[size]),
        repeats=repeats,
    )
    concurrent_batch = _trend_rows(
        batch_sizes,
        lambda size: (
            lambda: _concurrent_summary(batch_searcher, batch_molecules[size], workers)
        ),
        repeats=repeats,
    )

    concurrency_consistency = []
    for size in batch_sizes:
        linear = _batch_summary(batch_searcher, batch_molecules[size])
        concurrent = _concurrent_summary(batch_searcher, batch_molecules[size], workers)
        concurrency_consistency.append(
            {
                "size": size,
                "same_results": concurrent == linear,
                "linear_checksum": _stable_digest(linear),
                "concurrent_checksum": _stable_digest(concurrent),
            }
        )

    compile_queries = tuple(
        (
            _nested_branch(index % 6 + 1),
            _nested_recursive(index % 5 + 1),
            _logic_chain(index % 7 + 2),
        )[index % 3]
        for index in range(max(batch_sizes))
    )
    concurrent_compile = measure(
        lambda: _concurrent_compile_summary(compile_queries, workers),
        repeats=3 if heavy else repeats,
    )
    serial_compile_result = _serial_compile_summary(compile_queries)
    concurrent_compile_result = _concurrent_compile_summary(compile_queries, workers)
    compile_consistency = {
        "same_results": concurrent_compile_result == serial_compile_result,
        "serial_checksum": _stable_digest(serial_compile_result),
        "concurrent_checksum": _stable_digest(concurrent_compile_result),
    }

    sections = {
        "parse_tokenize": parse,
        "compile_query_graph": compile_query,
        "first_match_fresh_objects": first_match,
        "repeated_match_same_objects": repeated_match,
        "enumerate_all_embeddings": enumeration,
        "factorial_disconnected_enumeration": disconnected_enumeration,
        "deep_branch_compile": deep_branch_compile,
        "nested_recursive_compile": recursive_compile,
        "wide_logic_compile": logic_compile,
        "multi_ring_compile": multi_ring_compile,
        "near_match_final_failure": near_match_failure,
        "linear_batch": linear_batch,
        "concurrent_batch": concurrent_batch,
        "concurrency_consistency": concurrency_consistency,
        "concurrent_compile": concurrent_compile,
        "concurrent_compile_consistency": compile_consistency,
    }
    deterministic = (
        all(
            row.get("result_deterministic", True)
            for section in (
                parse,
                compile_query,
                enumeration,
                disconnected_enumeration,
                deep_branch_compile,
                recursive_compile,
                logic_compile,
                multi_ring_compile,
                near_match_failure,
                linear_batch,
                concurrent_batch,
            )
            for row in section
        )
        and first_match["result_deterministic"]
        and repeated_match["result_deterministic"]
        and concurrent_compile["result_deterministic"]
    )
    consistent = all(row["same_results"] for row in concurrency_consistency)
    return {
        "schema_version": 1,
        "profile": profile,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
            "networkx": networkx.__version__,
            "workers": workers,
        },
        "method": {
            "clock": "time.perf_counter_ns",
            "summary": "median and p90 after warmup; no wall-clock pass threshold",
            "memory": "peak Python allocations measured by tracemalloc",
            "parse_scope": "tokenization only",
            "compile_scope": "tokenization, parsing, and Hotpot query-graph construction",
            "first_match_scope": "matching with fresh pre-built query and target objects",
            "repeated_match_scope": "same Searcher and Molecule; Hotpot has no result cache",
            "concurrency_scope": "shared Searcher, read-only target Molecules, ThreadPoolExecutor",
        },
        "summary": {
            "all_repeated_results_deterministic": deterministic,
            "concurrent_equals_linear": consistent,
            "concurrent_compile_equals_serial": compile_consistency["same_results"],
        },
        "benchmarks": sections,
    }


def _parse_args(argv: object = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("quick", "heavy"), default="quick")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: object = None) -> int:
    args = _parse_args(argv)
    report = run_benchmarks(args.profile, args.workers)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    if not report["summary"]["all_repeated_results_deterministic"]:
        print("benchmark detected non-deterministic repeated results", file=sys.stderr)
        return 1
    if not report["summary"]["concurrent_equals_linear"]:
        print("concurrent search differs from linear search", file=sys.stderr)
        return 1
    if not report["summary"]["concurrent_compile_equals_serial"]:
        print("concurrent compilation differs from serial compilation", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
