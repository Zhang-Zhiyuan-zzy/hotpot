"""Reproducible Plan 16 search performance and resource gate runner.

Run this file unchanged on a baseline and a candidate, then compare the two
JSON reports.  The runner deliberately contains no performance thresholds and
does not infer speedups.  It does enforce semantic invariants needed for a
meaningful comparison: bounded-search behavior, complete mapping content,
deterministic order, recursion, and mixed native/Python predicates.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import hotpot as hp
import networkx

try:
    from hotpot.cheminfo.search import native
except ImportError:
    native = None


DISCONNECTED_SIZE = 7
DISCONNECTED_SMARTS = ".".join("*" for _ in range(DISCONNECTED_SIZE))
DISCONNECTED_SMILES = ".".join("C" for _ in range(DISCONNECTED_SIZE))
RECURSIVE_SMARTS = "[C;$([C]=O)]"
RECURSIVE_SMILES = "CC=O.CCO.CC=O.C=N"
MIXED_CALLABLE_SMILES = "CCCCO"


def _stable_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(
            (key, _stable_value(item))
            for key, item in sorted(value.items(), key=lambda pair: repr(pair[0]))
        )
    if isinstance(value, (tuple, list)):
        return tuple(_stable_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_stable_value(item) for item in value), key=repr))
    return value


def _stable_digest(value: Any) -> str:
    payload = json.dumps(
        _stable_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _percentile(samples: Sequence[int], fraction: float) -> int:
    ordered = sorted(samples)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def _measure(
    operation: Callable[[], Any],
    *,
    repeats: int,
    warmups: int,
    transform: Callable[[Any], Any] = lambda value: value,
) -> Dict[str, Any]:
    for _ in range(warmups):
        operation()
    gc.collect()

    samples = []
    digests = []
    result_summary = None
    for _ in range(repeats):
        started = time.perf_counter_ns()
        result = operation()
        samples.append(time.perf_counter_ns() - started)
        transformed = transform(result)
        digests.append(_stable_digest(transformed))
        result_summary = _compact_summary(transformed)
        del result, transformed
        gc.collect()

    return {
        "repeats": repeats,
        "warmups": warmups,
        "samples_ms": [sample / 1_000_000 for sample in samples],
        "median_ms": statistics.median(samples) / 1_000_000,
        "p90_ms": _percentile(samples, 0.90) / 1_000_000,
        "min_ms": min(samples) / 1_000_000,
        "max_ms": max(samples) / 1_000_000,
        "result_sha256": digests[0],
        "result_deterministic": len(set(digests)) == 1,
        "result_summary": result_summary,
    }


def _compact_summary(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        summary = {
            key: item
            for key, item in value.items()
            if key in {"count", "hit_count", "raw_mapping_count", "truncated"}
        }
        return summary or {"type": type(value).__name__}
    if isinstance(value, (tuple, list)):
        return {"type": type(value).__name__, "count": len(value)}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    return {"type": type(value).__name__}


def _canonical_mapping(mapping: Mapping[int, int]) -> Tuple[Tuple[int, int], ...]:
    return tuple(sorted((int(query), int(target)) for query, target in mapping.items()))


def _consume_public_mappings(searcher, molecule, max_matches):
    mappings = searcher.iter_mappings(molecule, max_matches=max_matches)
    sequence = tuple(_canonical_mapping(mapping) for mapping in mappings)
    return {
        "count": len(sequence),
        "truncated": bool(mappings.truncated),
        "mappings": sequence,
    }


def _hits_snapshot(hits) -> Dict[str, Any]:
    groups = tuple(
        (
            tuple(sorted(int(index) for index in hit.atom_indices)),
            tuple(_canonical_mapping(mapping) for mapping in hit.mappings),
        )
        for hit in hits
    )
    return {
        "hit_count": len(groups),
        "raw_mapping_count": sum(len(mappings) for _, mappings in groups),
        "truncated": bool(hits.truncated),
        "groups": groups,
    }


def _force_hit_objects(searcher, molecule, max_matches=None):
    hits = searcher.search(molecule, max_matches=max_matches)
    tuple(hits)
    return hits


def _query_summary(substructure) -> Dict[str, int]:
    return {
        "atom_count": len(substructure.query_atoms),
        "bond_count": len(substructure.query_bonds),
    }


def _prepared_query_summary(prepared) -> Dict[str, Any]:
    return {
        "query_key_count": len(prepared.query_spec["keys"]),
        "query_edge_count": len(prepared.query_spec["edges"]),
        "needs_atom_objects": prepared.needs_atom_objects,
        "needs_bond_objects": prepared.needs_bond_objects,
        "derived_requirements": prepared.derived_requirements,
    }


def _target_spec_summary(target_spec) -> Dict[str, int]:
    return {
        "target_key_count": len(target_spec["keys"]),
        "target_edge_count": len(target_spec["edges"]),
        "atom_record_count": len(target_spec["atoms"]),
        "bond_record_count": len(target_spec["bonds"]),
    }


def _native_target_summary(target) -> Dict[str, str]:
    return {"type": type(target).__name__}


def _native_problem_summary(problem) -> Dict[str, Any]:
    return {
        "type": type(problem).__name__,
        "query_keys": tuple(int(key) for key in problem.query_keys),
    }


def _raw_cursor_snapshot(problem):
    return tuple(tuple(int(value) for value in mapping) for mapping in problem.cursor())


def _adapter_snapshot(source):
    return tuple(
        tuple(sorted((int(target), int(query)) for target, query in mapping.items()))
        for mapping in source.subgraph_monomorphisms_iter()
    )


def _make_mixed_callable_query():
    substructure = hp.Substructure()
    substructure.add_atom(
        hp.QueryAtom(
            atomic_number={6},
            opaque_atom=lambda atom: atom.formal_charge == 0,
        )
    )
    substructure.add_atom(
        hp.QueryAtom(
            atomic_number={6},
            opaque_atom=lambda atom: not atom.is_aromatic,
        )
    )
    substructure.add_bond(
        0,
        1,
        bond_order={1.0},
        opaque_bond=lambda bond: bond.bond_order == 1,
    )
    return substructure


def _bounded_semantics(searcher, molecule) -> Tuple[Dict[str, Any], Dict[str, bool]]:
    limits = (0, 1, 64, None)
    observed = {}
    sequences = {}
    repeat_sequences = {}

    for limit in limits:
        first = _consume_public_mappings(searcher, molecule, limit)
        second = _consume_public_mappings(searcher, molecule, limit)
        name = "none" if limit is None else str(limit)
        sequences[limit] = first["mappings"]
        repeat_sequences[limit] = second["mappings"]
        observed[name] = {
            "max_matches": limit,
            "count": first["count"],
            "truncated": first["truncated"],
            "ordered_mapping_sha256": _stable_digest(first["mappings"]),
            "repeat_ordered_mapping_sha256": _stable_digest(second["mappings"]),
            "first_mappings": first["mappings"][:3],
            "last_mappings": first["mappings"][-3:],
        }

    complete = sequences[None]
    expected_count = math.factorial(DISCONNECTED_SIZE)
    target_keys = tuple(sorted(int(atom.idx) for atom in molecule.atoms))
    query_keys = tuple(range(DISCONNECTED_SIZE))
    every_mapping_valid = all(
        tuple(query for query, _ in mapping) == query_keys
        and tuple(sorted(target for _, target in mapping)) == target_keys
        for mapping in complete
    )
    checks = {
        "bounded_0_count": len(sequences[0]) == 0,
        "bounded_0_truncated": observed["0"]["truncated"] is True,
        "bounded_1_count": len(sequences[1]) == 1,
        "bounded_1_truncated": observed["1"]["truncated"] is True,
        "bounded_64_count": len(sequences[64]) == 64,
        "bounded_64_truncated": observed["64"]["truncated"] is True,
        "complete_count": len(complete) == expected_count,
        "complete_not_truncated": observed["none"]["truncated"] is False,
        "complete_mappings_unique": len(set(complete)) == expected_count,
        "complete_mappings_valid": every_mapping_valid,
        "bounded_1_is_complete_prefix": sequences[1] == complete[:1],
        "bounded_64_is_complete_prefix": sequences[64] == complete[:64],
        "bounded_order_repeats": all(
            sequences[limit] == repeat_sequences[limit] for limit in limits
        ),
    }
    return observed, checks


def _phase_breakdown(
    query,
    prepared,
    molecule,
    target_spec,
    native_target,
    problem,
    searcher,
    *,
    repeats,
    complete_repeats,
    warmups,
):
    requirements = {
        "needs_atom_objects": prepared.needs_atom_objects,
        "needs_bond_objects": prepared.needs_bond_objects,
        "derived_requirements": prepared.derived_requirements,
    }
    native_module = native._native_module()
    source = native._NativeMappingSource(problem)

    return {
        "public_query_parse": _measure(
            lambda: hp.Substructure.from_smarts(DISCONNECTED_SMARTS),
            repeats=repeats,
            warmups=warmups,
            transform=_query_summary,
        ),
        "native_query_ir_compile": _measure(
            lambda: native._prepare_query(query),
            repeats=repeats,
            warmups=warmups,
            transform=_prepared_query_summary,
        ),
        "python_target_pack": _measure(
            lambda: native._pack_target_snapshot(molecule, **requirements),
            repeats=repeats,
            warmups=warmups,
            transform=_target_spec_summary,
        ),
        "native_target_construction": _measure(
            lambda: native_module.NativeTarget(target_spec),
            repeats=repeats,
            warmups=warmups,
            transform=_native_target_summary,
        ),
        "native_problem_construction": _measure(
            lambda: native_module.NativeProblem(prepared.query_spec, native_target),
            repeats=repeats,
            warmups=warmups,
            transform=_native_problem_summary,
        ),
        "raw_native_cursor_complete": _measure(
            lambda: _raw_cursor_snapshot(problem),
            repeats=complete_repeats,
            warmups=warmups,
        ),
        "legacy_direction_adapter_complete": _measure(
            lambda: _adapter_snapshot(source),
            repeats=complete_repeats,
            warmups=warmups,
        ),
        "public_mappingproxy_complete": _measure(
            lambda: _consume_public_mappings(searcher, molecule, None),
            repeats=complete_repeats,
            warmups=warmups,
        ),
        "public_hits_group_materialization_complete": _measure(
            lambda: searcher.search(molecule),
            repeats=complete_repeats,
            warmups=warmups,
            transform=_hits_snapshot,
        ),
        "public_hit_objects_eager_complete": _measure(
            lambda: _force_hit_objects(searcher, molecule),
            repeats=complete_repeats,
            warmups=warmups,
            transform=_hits_snapshot,
        ),
    }


def _workload_benchmarks(
    disconnected_searcher,
    disconnected_molecule,
    recursive_searcher,
    recursive_molecule,
    mixed_searcher,
    mixed_molecule,
    *,
    repeats,
    complete_repeats,
    warmups,
):
    no_match_searcher = hp.Searcher(hp.Substructure.from_smarts("N"))
    return {
        "has_match_disconnected_7_hit": _measure(
            lambda: disconnected_searcher.has_match(disconnected_molecule),
            repeats=repeats,
            warmups=warmups,
        ),
        "has_match_disconnected_7_miss": _measure(
            lambda: no_match_searcher.has_match(disconnected_molecule),
            repeats=repeats,
            warmups=warmups,
        ),
        "bounded_disconnected_7": {
            "0": _measure(
                lambda: _consume_public_mappings(
                    disconnected_searcher, disconnected_molecule, 0
                ),
                repeats=repeats,
                warmups=warmups,
            ),
            "1": _measure(
                lambda: _consume_public_mappings(
                    disconnected_searcher, disconnected_molecule, 1
                ),
                repeats=repeats,
                warmups=warmups,
            ),
            "64": _measure(
                lambda: _consume_public_mappings(
                    disconnected_searcher, disconnected_molecule, 64
                ),
                repeats=repeats,
                warmups=warmups,
            ),
            "none": _measure(
                lambda: _consume_public_mappings(
                    disconnected_searcher, disconnected_molecule, None
                ),
                repeats=complete_repeats,
                warmups=warmups,
            ),
        },
        "complete_public_hits_disconnected_7": _measure(
            lambda: disconnected_searcher.search(disconnected_molecule),
            repeats=complete_repeats,
            warmups=warmups,
            transform=_hits_snapshot,
        ),
        "recursive_complete": _measure(
            lambda: _consume_public_mappings(
                recursive_searcher, recursive_molecule, None
            ),
            repeats=repeats,
            warmups=warmups,
        ),
        "recursive_has_match": _measure(
            lambda: recursive_searcher.has_match(recursive_molecule),
            repeats=repeats,
            warmups=warmups,
        ),
        "mixed_callable_complete": _measure(
            lambda: _consume_public_mappings(mixed_searcher, mixed_molecule, None),
            repeats=repeats,
            warmups=warmups,
        ),
        "mixed_callable_has_match": _measure(
            lambda: mixed_searcher.has_match(mixed_molecule),
            repeats=repeats,
            warmups=warmups,
        ),
        "repeated_same_searcher_target_bounded_64": _measure(
            lambda: _consume_public_mappings(
                disconnected_searcher, disconnected_molecule, 64
            ),
            repeats=repeats * 2,
            warmups=warmups,
        ),
    }


def _current_rss_bytes() -> Optional[int]:
    statm = Path("/proc/self/statm")
    if not statm.exists():
        return None
    fields = statm.read_text().split()
    return int(fields[1]) * int(os.sysconf("SC_PAGE_SIZE"))


def _max_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _resource_growth(
    operation: Callable[[], Any], *, cycles: int, checkpoint_count: int
) -> Dict[str, Any]:
    operation()
    operation()
    gc.collect()
    rss_before = _current_rss_bytes()
    max_rss_before = _max_rss_bytes()

    tracemalloc.start()
    gc.collect()
    traced_before, _ = tracemalloc.get_traced_memory()
    checkpoint_stride = max(1, math.ceil(cycles / checkpoint_count))
    checkpoints = []
    digests = set()
    samples = []
    for cycle in range(1, cycles + 1):
        started = time.perf_counter_ns()
        result = operation()
        samples.append(time.perf_counter_ns() - started)
        digests.add(_stable_digest(result))
        del result
        if cycle % checkpoint_stride == 0 or cycle == cycles:
            gc.collect()
            current, peak = tracemalloc.get_traced_memory()
            checkpoints.append(
                {
                    "completed_cycles": cycle,
                    "traced_current_bytes": current,
                    "traced_peak_bytes": peak,
                    "current_rss_bytes": _current_rss_bytes(),
                    "max_rss_bytes": _max_rss_bytes(),
                }
            )

    gc.collect()
    traced_after, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = _current_rss_bytes()
    max_rss_after = _max_rss_bytes()
    return {
        "cycles": cycles,
        "samples_ms": [sample / 1_000_000 for sample in samples],
        "median_ms_per_cycle": statistics.median(samples) / 1_000_000,
        "result_deterministic": len(digests) == 1,
        "result_sha256": next(iter(digests)),
        "traced_current_before_bytes": traced_before,
        "traced_current_after_gc_bytes": traced_after,
        "traced_current_growth_bytes": traced_after - traced_before,
        "traced_peak_bytes": traced_peak,
        "current_rss_before_bytes": rss_before,
        "current_rss_after_bytes": rss_after,
        "current_rss_growth_bytes": (
            None
            if rss_before is None or rss_after is None
            else rss_after - rss_before
        ),
        "max_rss_before_bytes": max_rss_before,
        "max_rss_after_bytes": max_rss_after,
        "max_rss_growth_bytes": max_rss_after - max_rss_before,
        "checkpoints": checkpoints,
    }


def _git_metadata() -> Dict[str, Any]:
    def run(*args):
        completed = subprocess.run(
            ("git",) + args,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    status = run("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "tracked_worktree_dirty": None if status is None else bool(status),
    }


def _file_sha256(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _environment(label: Optional[str]) -> Dict[str, Any]:
    native_module = None
    if native is not None:
        try:
            native_module = native._native_module()
        except ImportError:
            pass
    affinity = (
        len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    )
    return {
        "label": label,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hostname": platform.node(),
        "logical_cpu_count": os.cpu_count(),
        "cpu_affinity_count": affinity,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "hotpot_version": hp.version(),
        "hotpot_path": hp.__file__,
        "networkx_version": networkx.__version__,
        "native_private_api_available": native_module is not None,
        "native_extension_path": (
            None if native_module is None else native_module.__file__
        ),
        "native_extension_sha256": _file_sha256(
            None if native_module is None else native_module.__file__
        ),
        "git": _git_metadata(),
    }


def _measurement_determinism(value: Any) -> bool:
    if isinstance(value, Mapping):
        own = value.get("result_deterministic", True)
        return bool(own) and all(
            _measurement_determinism(item)
            for key, item in value.items()
            if key != "result_deterministic"
        )
    if isinstance(value, (tuple, list)):
        return all(_measurement_determinism(item) for item in value)
    return True


def run_gate(
    profile: str,
    *,
    label: Optional[str] = None,
    repeats_override: Optional[int] = None,
    resource_cycles_override: Optional[int] = None,
) -> Dict[str, Any]:
    settings = {
        "quick": {
            "repeats": 5,
            "complete_repeats": 3,
            "warmups": 1,
            "resource_cycles": 12,
            "resource_checkpoints": 4,
        },
        "heavy": {
            "repeats": 15,
            "complete_repeats": 7,
            "warmups": 2,
            "resource_cycles": 50,
            "resource_checkpoints": 5,
        },
    }[profile]
    if repeats_override is not None:
        settings["repeats"] = repeats_override
        settings["complete_repeats"] = repeats_override
    if resource_cycles_override is not None:
        settings["resource_cycles"] = resource_cycles_override

    disconnected_query = hp.Substructure.from_smarts(DISCONNECTED_SMARTS)
    disconnected_molecule = hp.read_mol(DISCONNECTED_SMILES, "smi")
    disconnected_searcher = hp.Searcher(disconnected_query)
    recursive_query = hp.Substructure.from_smarts(RECURSIVE_SMARTS)
    recursive_molecule = hp.read_mol(RECURSIVE_SMILES, "smi")
    recursive_searcher = hp.Searcher(recursive_query)
    mixed_query = _make_mixed_callable_query()
    mixed_molecule = hp.read_mol(MIXED_CALLABLE_SMILES, "smi")
    mixed_searcher = hp.Searcher(mixed_query)

    bounded, checks = _bounded_semantics(
        disconnected_searcher, disconnected_molecule
    )
    recursive_first = _consume_public_mappings(
        recursive_searcher, recursive_molecule, None
    )
    recursive_second = _consume_public_mappings(
        recursive_searcher, recursive_molecule, None
    )
    mixed_first = _consume_public_mappings(mixed_searcher, mixed_molecule, None)
    mixed_second = _consume_public_mappings(mixed_searcher, mixed_molecule, None)
    checks.update(
        {
            "has_match_true_uses_public_api": disconnected_searcher.has_match(
                disconnected_molecule
            )
            is True,
            "has_match_false_uses_public_api": hp.Searcher(
                hp.Substructure.from_smarts("N")
            ).has_match(disconnected_molecule)
            is False,
            "recursive_nonempty": recursive_first["count"] > 0,
            "recursive_order_repeats": recursive_first == recursive_second,
            "recursive_has_match_agrees": recursive_searcher.has_match(
                recursive_molecule
            )
            is bool(recursive_first["count"]),
            "mixed_callable_nonempty": mixed_first["count"] > 0,
            "mixed_callable_order_repeats": mixed_first == mixed_second,
            "mixed_callable_has_match_agrees": mixed_searcher.has_match(
                mixed_molecule
            )
            is bool(mixed_first["count"]),
        }
    )

    native_module = None
    native_unavailable_reason = None
    if native is None:
        native_unavailable_reason = (
            "hotpot.cheminfo.search.native is absent on this implementation"
        )
    else:
        try:
            native_module = native._native_module()
        except ImportError as exc:
            native_unavailable_reason = str(exc)

    if native_module is None:
        phase_breakdown = {
            "available": False,
            "reason": native_unavailable_reason,
            "measurements": {},
        }
    else:
        prepared = native._prepare_query(disconnected_query)
        requirements = {
            "needs_atom_objects": prepared.needs_atom_objects,
            "needs_bond_objects": prepared.needs_bond_objects,
            "derived_requirements": prepared.derived_requirements,
        }
        target_spec = native._pack_target_snapshot(
            disconnected_molecule, **requirements
        )
        native_target = native_module.NativeTarget(target_spec)
        problem = native_module.NativeProblem(prepared.query_spec, native_target)
        phase_breakdown = {
            "available": True,
            "reason": None,
            "measurements": _phase_breakdown(
                disconnected_query,
                prepared,
                disconnected_molecule,
                target_spec,
                native_target,
                problem,
                disconnected_searcher,
                repeats=settings["repeats"],
                complete_repeats=settings["complete_repeats"],
                warmups=settings["warmups"],
            ),
        }
    workloads = _workload_benchmarks(
        disconnected_searcher,
        disconnected_molecule,
        recursive_searcher,
        recursive_molecule,
        mixed_searcher,
        mixed_molecule,
        repeats=settings["repeats"],
        complete_repeats=settings["complete_repeats"],
        warmups=settings["warmups"],
    )

    def resource_operation():
        bounded_result = _consume_public_mappings(
            disconnected_searcher, disconnected_molecule, 64
        )
        recursive_result = _consume_public_mappings(
            recursive_searcher, recursive_molecule, None
        )
        mixed_result = _consume_public_mappings(
            mixed_searcher, mixed_molecule, None
        )
        return {
            "has_match": disconnected_searcher.has_match(disconnected_molecule),
            "bounded_64_sha256": _stable_digest(bounded_result),
            "recursive_sha256": _stable_digest(recursive_result),
            "mixed_callable_sha256": _stable_digest(mixed_result),
        }

    resource_growth = _resource_growth(
        resource_operation,
        cycles=settings["resource_cycles"],
        checkpoint_count=settings["resource_checkpoints"],
    )
    all_measurements_deterministic = (
        _measurement_determinism(phase_breakdown)
        and _measurement_determinism(workloads)
        and resource_growth["result_deterministic"]
    )
    checks["all_measurements_deterministic"] = all_measurements_deterministic

    semantic_payload = {
        "bounded": bounded,
        "recursive": recursive_first,
        "mixed_callable": mixed_first,
    }
    substantive_payload = {
        "disconnected_complete_mapping_set": tuple(
            sorted(
                _consume_public_mappings(
                    disconnected_searcher, disconnected_molecule, None
                )["mappings"]
            )
        ),
        "recursive_mapping_set": tuple(sorted(recursive_first["mappings"])),
        "mixed_callable_mapping_set": tuple(sorted(mixed_first["mappings"])),
    }
    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    return {
        "schema_version": 1,
        "profile": profile,
        "environment": _environment(label),
        "settings": settings,
        "method": {
            "clock": "time.perf_counter_ns",
            "timing": "warmup followed by per-sample wall time; GC occurs outside timed regions",
            "comparison": "run this unchanged for baseline and candidate; no threshold is embedded",
            "mapping_order": "outer tuple preserves executor enumeration order; each mapping is sorted by query key",
            "query_parse_scope": "Substructure.from_smarts only",
            "query_ir_scope": "compile an existing Substructure into native query IR",
            "target_pack_scope": "Python Molecule snapshot and requested derived facts",
            "native_target_scope": "parse/copy one already-packed target into NativeTarget",
            "native_problem_scope": "parse query predicates, plan RI order, and build domains against an existing NativeTarget",
            "raw_cursor_scope": "consume a NativeProblem cursor into query-aligned tuples",
            "adapter_scope": "consume the private legacy-direction adapter into dictionaries",
            "public_mapping_scope": "compile/pack/construct, consume iter_mappings, and snapshot MappingProxy results",
            "hits_group_scope": "Searcher.search eagerly consumes and groups mappings; Hit objects remain lazy until result digesting after timing",
            "hits_eager_scope": "Searcher.search plus explicit construction of all public Hit objects",
            "resource_scope": "repeated public has_match, bounded-64, recursive, and mixed-callable calls; snapshots/digests included",
            "rss": "current RSS from /proc/self/statm when available; max RSS normalized from getrusage to bytes",
        },
        "fixtures": {
            "disconnected_7": {
                "smarts": DISCONNECTED_SMARTS,
                "smiles": DISCONNECTED_SMILES,
                "expected_complete_raw_mappings": math.factorial(
                    DISCONNECTED_SIZE
                ),
            },
            "recursive": {
                "smarts": RECURSIVE_SMARTS,
                "smiles": RECURSIVE_SMILES,
            },
            "mixed_callable": {
                "description": "two native carbon sets plus atom and bond Python callable leaves",
                "smiles": MIXED_CALLABLE_SMILES,
            },
        },
        "semantic_observations": {
            "bounded_disconnected_7": bounded,
            "recursive": {
                "count": recursive_first["count"],
                "truncated": recursive_first["truncated"],
                "ordered_mapping_sha256": _stable_digest(
                    recursive_first["mappings"]
                ),
            },
            "mixed_callable": {
                "count": mixed_first["count"],
                "truncated": mixed_first["truncated"],
                "ordered_mapping_sha256": _stable_digest(mixed_first["mappings"]),
            },
        },
        "phase_breakdown": phase_breakdown,
        "workloads": workloads,
        "resource_growth": resource_growth,
        "summary": {
            "all_checks_passed": not failed_checks,
            "failed_checks": failed_checks,
            "checks": checks,
            "all_measurements_deterministic": all_measurements_deterministic,
            "semantic_sha256": _stable_digest(semantic_payload),
            "substantive_mapping_sets_sha256": _stable_digest(
                substantive_payload
            ),
        },
    }


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("quick", "heavy"), default="quick")
    parser.add_argument("--label", help="free-form baseline/candidate label")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--repeats",
        type=int,
        help="override both ordinary and complete-enumeration repeat counts",
    )
    parser.add_argument(
        "--resource-cycles",
        type=int,
        help="override repeated-call resource cycles",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    report = run_gate(
        args.profile,
        label=args.label,
        repeats_override=args.repeats,
        resource_cycles_override=args.resource_cycles,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0 if report["summary"]["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
