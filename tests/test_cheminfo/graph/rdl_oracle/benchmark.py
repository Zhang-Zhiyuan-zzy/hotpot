"""Measure Hotpot Relevant Cycles against the pinned live RDL oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from time import perf_counter_ns
from typing import Iterable, Optional, Protocol, Sequence, Tuple

from corpus import (
    deterministic_random_graph_cases,
    graph_atlas_cases,
    handcrafted_graph_cases,
    named_graph_cases,
    pubchem_graph_cases,
)
from oracle import RDLOracle

RDL_COMMIT = "3a7ff93de0d9c4f6a5661508549c6063573f39c7"
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]

NodeCycle = Tuple[int, ...]
Cycles = Tuple[NodeCycle, ...]


class CycleFunction(Protocol):
    def __call__(
        self,
        edges: Iterable[Sequence[int]],
        *,
        max_cycles: Optional[int],
    ) -> Cycles:
        ...


def _timed_call(
    cycle_function: CycleFunction,
    edges: Iterable[Sequence[int]],
) -> int:
    started = perf_counter_ns()
    cycle_function(edges, max_cycles=None)
    return perf_counter_ns() - started


def _percentile(samples: Sequence[int], fraction: float) -> int:
    ordered = sorted(samples)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _native_source_paths() -> Tuple[Path, ...]:
    native_dir = REPOSITORY_ROOT / "hotpot/cheminfo/graph/_native"
    return (
        native_dir / "bindings.cpp",
        native_dir / "relevant_cycles.cpp",
        native_dir / "relevant_cycles.hpp",
    )


def _git_state(native_sources: Sequence[Path]) -> Tuple[str, bool]:
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    relative_sources = tuple(
        str(source.relative_to(REPOSITORY_ROOT)) for source in native_sources
    )
    clean_result = subprocess.run(
        ("git", "diff", "--quiet", "HEAD", "--", *relative_sources),
        cwd=REPOSITORY_ROOT,
        check=False,
    )
    return commit, clean_result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--include-atlas", action="store_true")
    parser.add_argument("--include-random", action="store_true")
    arguments = parser.parse_args()

    sys.path.insert(0, str(REPOSITORY_ROOT))
    from hotpot.cheminfo.graph import _relevant_cycles, relevant_cycles

    extension_path = Path(_relevant_cycles.__file__).resolve()
    native_sources = _native_source_paths()
    stale_sources = tuple(
        source for source in native_sources
        if source.stat().st_mtime_ns > extension_path.stat().st_mtime_ns
    )
    if stale_sources:
        names = ", ".join(source.name for source in stale_sources)
        raise SystemExit(f"native extension is older than source files: {names}")
    git_commit, native_sources_match_commit = _git_state(native_sources)

    oracle = RDLOracle(arguments.library)
    handcrafted_cases = handcrafted_graph_cases()
    named_cases = named_graph_cases()
    pubchem_cases = pubchem_graph_cases()
    atlas_cases = tuple(graph_atlas_cases()) if arguments.include_atlas else ()
    random_cases = (
        tuple(deterministic_random_graph_cases()) if arguments.include_random else ()
    )
    cases = list(
        handcrafted_cases
        + named_cases
        + pubchem_cases
        + atlas_cases
        + random_cases
    )
    hotpot_samples = []
    rdl_samples = []
    mismatches = []
    cycle_count = 0
    for name, edges in cases:
        hotpot_result = relevant_cycles(edges, max_cycles=None)
        rdl_result = oracle.relevant_cycles(edges, max_cycles=None)
        cycle_count += len(rdl_result)
        if hotpot_result != rdl_result:
            mismatches.append(name)
        for repeat in range(arguments.repeats):
            if repeat % 2 == 0:
                hotpot_samples.append(_timed_call(relevant_cycles, edges))
                rdl_samples.append(_timed_call(oracle.relevant_cycles, edges))
            else:
                rdl_samples.append(_timed_call(oracle.relevant_cycles, edges))
                hotpot_samples.append(_timed_call(relevant_cycles, edges))

    hotpot_median = median(hotpot_samples) / 1_000
    rdl_median = median(rdl_samples) / 1_000
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "oracle": {
            "name": "RingDecomposerLib",
            "commit": RDL_COMMIT,
            "library_file": arguments.library.name,
            "library_sha256": _file_sha256(arguments.library.resolve()),
        },
        "hotpot": {
            "git_commit": git_commit,
            "native_sources_match_git_commit": native_sources_match_commit,
            "extension_file": extension_path.name,
            "extension_sha256": _file_sha256(extension_path),
            "source_files_sha256": {
                source.name: _file_sha256(source) for source in native_sources
            },
        },
        "corpus": {
            "handcrafted": len(handcrafted_cases),
            "named_networkx": len(named_cases),
            "pubchem": len(pubchem_cases),
            "networkx_graph_atlas": len(atlas_cases),
            "deterministic_random": len(random_cases),
        },
        "graph_count": len(cases),
        "cycle_count": cycle_count,
        "matched_graphs": len(cases) - len(mismatches),
        "exact_match_rate": (len(cases) - len(mismatches)) / len(cases),
        "mismatches": mismatches,
        "repeats": arguments.repeats,
        "timing_microseconds_per_call": {
            "hotpot_median": hotpot_median,
            "hotpot_p95": _percentile(hotpot_samples, 0.95) / 1_000,
            "rdl_median": rdl_median,
            "rdl_p95": _percentile(rdl_samples, 0.95) / 1_000,
            "hotpot_over_rdl_median_ratio": hotpot_median / rdl_median,
        },
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
