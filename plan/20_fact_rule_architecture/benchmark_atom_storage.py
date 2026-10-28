"""Measure Atom storage candidates without importing the Hotpot facade.

The common workload retains all 19 current numeric columns to isolate layout
cost. It does not decide their final fact/result ownership or measure the new
identity, presence, source, dependency, transaction, or public projection code.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from types import ModuleType

THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
)
for _variable in THREAD_VARIABLES:
    os.environ[_variable] = "1"

import numpy as np


FIELDS = (
    "atomic_number", "n", "s", "p", "d", "f", "g", "formal_charge",
    "partial_charge", "is_aromatic", "x", "y", "z", "valence",
    "implicit_hydrogens", "id", "x_constraint", "y_constraint", "z_constraint",
)
INDEX = {name: index for index, name in enumerate(FIELDS)}
COORDINATES = ("x", "y", "z")
BOOLEAN_FIELDS = ("is_aromatic", "x_constraint", "y_constraint", "z_constraint")
FLOAT_FIELDS = ("partial_charge", *COORDINATES)
CAST = {
    name: bool if name in BOOLEAN_FIELDS else float if name in FLOAT_FIELDS else int
    for name in FIELDS
}
# Keep the current float64 storage domain for charge, H, source IDs and the
# still-unmigrated integer projections. Narrowing them needs its own evidence.
DTYPES = {
    name: np.dtype("bool" if name in BOOLEAN_FIELDS else
                   "uint8" if name == "atomic_number" else "float64")
    for name in FIELDS
}
ROW_DTYPE = np.dtype(
    [(name, DTYPES[name]) for name in FIELDS[:10]]
    + [("coordinates", np.float64, (3,))]
    + [(name, DTYPES[name]) for name in FIELDS[13:]],
    align=True,
)
SCALAR_FIELDS = ("formal_charge", "implicit_hydrogens", "id", "x")
EXPORT_FIELDS = (
    "atomic_number", "formal_charge", "partial_charge", "x", "y", "z",
    "implicit_hydrogens",
)
ROTATION = np.array(((0., -1., 0.), (1., 0., 0.), (0., 0., 1.)))
TRANSLATION = np.array((0.125, -0.25, 0.5))
SIZES = (16, 256, 4096)
SEED = 20261028
WARMUPS = 3
SAMPLES = 10


class _CurrentRows:
    """One independent 19-float array per Atom, without the Atom object cost."""

    def __init__(self, matrix):
        self.rows = [row.copy() for row in matrix]

    def get(self, row, field):
        return CAST[field](self.rows[row][INDEX[field]])

    def set(self, row, field, value):
        self.rows[row][INDEX[field]] = value

    def column(self, field):
        index = INDEX[field]
        return np.fromiter((row[index] for row in self.rows), dtype=DTYPES[field])

    def set_column(self, field, values):
        index = INDEX[field]
        for row, value in zip(self.rows, values):
            row[index] = value

    def coordinates(self):
        return np.array([row[10:13] for row in self.rows])

    def set_coordinates(self, values):
        for row, coordinates in zip(self.rows, values):
            row[10:13] = coordinates

    def export(self, fields):
        indices = [INDEX[field] for field in fields]
        return np.array([row[indices] for row in self.rows])

    def matrix(self):
        return np.array(self.rows)

    def remap(self, keep, added, order):
        combined = [self.rows[index] for index in keep] + [row.copy() for row in added]
        self.rows = [combined[index] for index in order]


class _ColumnStorage:
    def get(self, row, field):
        return CAST[field](self.columns[field][row])

    def set(self, row, field, value):
        self.columns[field][row] = value

    def column(self, field):
        return self.columns[field].copy()

    def set_column(self, field, values):
        self.columns[field][...] = values

    def coordinates(self):
        return self.coordinate_block.copy()

    def set_coordinates(self, values):
        self.coordinate_block[...] = values

    def export(self, fields):
        return np.column_stack([self.columns[field] for field in fields])

    def matrix(self):
        return self.export(FIELDS)


class _TypedSoA(_ColumnStorage):
    def __init__(self, matrix):
        self.columns = {
            field: matrix[:, INDEX[field]].astype(DTYPES[field], copy=True)
            for field in FIELDS if field not in COORDINATES
        }
        self.coordinate_block = matrix[:, 10:13].copy()
        self._bind_coordinates()

    def _bind_coordinates(self):
        self.columns.update({
            field: self.coordinate_block[:, index]
            for index, field in enumerate(COORDINATES)
        })

    def remap(self, keep, added, order):
        self.columns = {
            field: np.concatenate((self.columns[field][keep],
                                   added[:, INDEX[field]].astype(DTYPES[field])))[order]
            for field in FIELDS if field not in COORDINATES
        }
        self.coordinate_block = np.concatenate(
            (self.coordinate_block[keep], added[:, 10:13])
        )[order]
        self._bind_coordinates()


class _StructuredAoS(_ColumnStorage):
    def __init__(self, matrix):
        self.records = np.empty(len(matrix), dtype=ROW_DTYPE)
        self._bind_columns()
        for field in FIELDS:
            self.columns[field][...] = matrix[:, INDEX[field]]

    def _bind_columns(self):
        self.coordinate_block = self.records["coordinates"]
        self.columns = {
            field: self.records[field] for field in FIELDS if field not in COORDINATES
        }
        self.columns.update({
            field: self.coordinate_block[:, index]
            for index, field in enumerate(COORDINATES)
        })

    def remap(self, keep, added, order):
        new_records = _StructuredAoS(added).records
        self.records = np.concatenate((self.records[keep], new_records))[order]
        self._bind_columns()


class _BlockedColumns:
    """Two homogeneous blocks preserve the old numeric domain without row objects.

    This candidate measures grouping overhead separately from dtype narrowing.
    Coordinates have one contiguous N×3 owner. The remaining columns share one
    float64 block; logical scalar codecs still use the frozen CAST declarations.
    """

    def __init__(self, matrix):
        self.scalars = np.concatenate((matrix[:, :10], matrix[:, 13:]), axis=1)
        self.coordinate_block = matrix[:, 10:13].copy()

    def get(self, row, field):
        index = INDEX[field]
        value = (self.coordinate_block[row, index - 10] if 10 <= index < 13
                 else self.scalars[row, index if index < 10 else index - 3])
        return CAST[field](value)

    def set(self, row, field, value):
        index = INDEX[field]
        if 10 <= index < 13:
            self.coordinate_block[row, index - 10] = value
        else:
            self.scalars[row, index if index < 10 else index - 3] = value

    def _column(self, field):
        index = INDEX[field]
        return (self.coordinate_block[:, index - 10] if 10 <= index < 13
                else self.scalars[:, index if index < 10 else index - 3])

    def column(self, field):
        return self._column(field).astype(DTYPES[field], copy=True)

    def set_column(self, field, values):
        self._column(field)[...] = values

    def coordinates(self):
        return self.coordinate_block.copy()

    def set_coordinates(self, values):
        self.coordinate_block[...] = values

    def export(self, fields):
        return np.column_stack([self._column(field) for field in fields])

    def matrix(self):
        return np.concatenate((self.scalars[:, :10], self.coordinate_block,
                               self.scalars[:, 10:]), axis=1)

    def remap(self, keep, added, order):
        added_scalars = np.concatenate((added[:, :10], added[:, 13:]), axis=1)
        self.scalars = np.concatenate((self.scalars[keep], added_scalars))[order]
        self.coordinate_block = np.concatenate(
            (self.coordinate_block[keep], added[:, 10:13])
        )[order]


LAYOUTS = {
    "current_atom_arrays": _CurrentRows,
    "typed_soa": _TypedSoA,
    "structured_aos": _StructuredAoS,
    "blocked_float_columns": _BlockedColumns,
}


class _CapacityTable:
    """Exercise the actual private table through the common storage workload."""

    def __init__(self, matrix):
        self.table = _FACT_TABLE(matrix)

    def get(self, row, field):
        return self.table.read(row, field)

    def set(self, row, field, value):
        self.table.update(row, {field: value})

    def column(self, field):
        return self.table.column(field).astype(DTYPES[field], copy=False)

    def set_column(self, field, values):
        self.table.update(None, {field: values})

    def coordinates(self):
        return self.table.coordinates()

    def set_coordinates(self, values):
        self.table.update(None, dict(zip(COORDINATES, values.T)))

    def export(self, fields):
        return self.table.select(fields)

    def matrix(self):
        return self.table.matrix()

    def remap(self, keep, added, order):
        self.table.remap(keep)
        self.table.append(added)
        self.table.remap(order)


def _load_private_table(repository, output):
    """Load only the two numeric leaf modules and retain their measured bytes."""
    global _FACT_TABLE
    source = repository / "hotpot" / "cheminfo" / "structure"
    package = ModuleType("_plan20_atom_storage")
    package.__path__ = [str(source)]
    sys.modules[package.__name__] = package
    fields = importlib.import_module(f"{package.__name__}._fields")
    table = importlib.import_module(f"{package.__name__}._atom_table")
    assert fields.ATOM_FIELDS == FIELDS
    _FACT_TABLE = table._AtomFactTable
    LAYOUTS["capacity_fact_table"] = _CapacityTable
    files = []
    for name in ("_fields.py", "_atom_table.py"):
        data = (source / name).read_bytes()
        destination = output / name
        destination.write_bytes(data)
        files.append({"source": str(source / name), "copy": str(destination),
                      "sha256": hashlib.sha256(data).hexdigest()})
    return files


def _fixture(count):
    rng = np.random.default_rng(SEED + count)
    matrix = rng.integers(0, 8, size=(count, len(FIELDS))).astype(np.float64)
    matrix[:, 0] = rng.integers(1, 119, size=count)
    for field in BOOLEAN_FIELDS:
        matrix[:, INDEX[field]] %= 2
    matrix[:, 8] = rng.normal(size=count)
    matrix[:, 10:13] = rng.normal(size=(count, 3))
    matrix[:, 15] = 2**40 + np.arange(count)
    matrix[:2, 7] = (2**20 + 17, -(2**20 + 17))
    matrix[:2, 14] = (2**20 + 33, 2**20 + 34)
    # Existing float storage also accepts integers beyond int64; do not make
    # an int16/int64 conversion look lossless by testing only small numbers.
    matrix[2, 15] = float(2**70)
    matrix[2, 7] = float(-(2**65))
    matrix[2, 14] = float(2**65)
    inserted_count = max(1, count // 8)
    return {
        "matrix": matrix,
        "values": matrix[:, 7].copy(),
        "coordinates": matrix[:, 10:13].copy() + TRANSLATION,
        "keep": np.arange(inserted_count, count),
        "added": matrix[:inserted_count].copy(),
        "order": rng.permutation(count),
    }


def _scalar_read(store, fixture, iterations):
    total = 0.
    for _ in range(iterations):
        for row in range(len(fixture["matrix"])):
            for field in SCALAR_FIELDS:
                total += store.get(row, field)
    return total


def _scalar_write(store, fixture, iterations):
    values = fixture["values"]
    for _ in range(iterations):
        for row in range(len(values)):
            store.set(row, "formal_charge", values[row])
            store.set(row, "implicit_hydrogens", fixture["matrix"][row, 14])
            store.set(row, "x", fixture["coordinates"][row, 0])


def _column_read(store, fixture, iterations):
    for _ in range(iterations):
        result = store.column("formal_charge")
    return result


def _column_write(store, fixture, iterations):
    for _ in range(iterations):
        store.set_column("formal_charge", fixture["values"])


def _coordinate_transform(store, fixture, iterations):
    for _ in range(iterations):
        store.set_coordinates(store.coordinates() @ ROTATION + TRANSLATION)


def _selected_export(store, fixture, iterations):
    for _ in range(iterations):
        result = store.export(EXPORT_FIELDS)
    return result


def _full_atom_matrix_export(store, fixture, iterations):
    for _ in range(iterations):
        result = store.matrix()
    return result


def _insert_delete_remap(store, fixture, iterations):
    for _ in range(iterations):
        store.remap(fixture["keep"], fixture["added"], fixture["order"])


WORKLOADS = {
    "scalar_read": _scalar_read,
    "scalar_write": _scalar_write,
    "column_read": _column_read,
    "column_write": _column_write,
    "coordinate_transform": _coordinate_transform,
    "selected_export": _selected_export,
    "full_atom_matrix_export": _full_atom_matrix_export,
    "insert_delete_remap": _insert_delete_remap,
}


def _owned_memory(store):
    seen = set()
    numeric_bytes = 0

    def visit(value):
        nonlocal numeric_bytes
        if id(value) in seen:
            return 0
        seen.add(id(value))
        size = sys.getsizeof(value)
        if isinstance(value, np.ndarray):
            if value.base is None:
                numeric_bytes += value.nbytes
            else:
                size += visit(value.base)
        elif isinstance(value, dict):
            size += sum(visit(item) for item in value.values())
        elif isinstance(value, (list, tuple)):
            size += sum(visit(item) for item in value)
        elif hasattr(value, "__dict__"):
            size += visit(value.__dict__)
        elif hasattr(type(value), "__slots__"):
            size += sum(visit(getattr(value, name)) for name in type(value).__slots__)
        return size

    retained = visit(store)
    return {"retained_bytes": retained, "numeric_buffer_bytes": numeric_bytes}


def _validate(fixtures):
    checks = []
    for count, fixture in fixtures.items():
        for name, factory in LAYOUTS.items():
            input_matrix = fixture["matrix"].copy()
            store = factory(input_matrix)
            input_matrix[...] = 0
            np.testing.assert_array_equal(store.export(FIELDS), fixture["matrix"])
            matrix = store.matrix()
            assert matrix.dtype == np.float64
            np.testing.assert_array_equal(matrix, fixture["matrix"])
            for output in (store.column("formal_charge"), store.coordinates(),
                           store.export(EXPORT_FIELDS), matrix):
                output[...] = -999
            np.testing.assert_array_equal(store.export(FIELDS), fixture["matrix"])
            column = fixture["values"].copy()
            store.set_column("formal_charge", column)
            column[...] = 0
            np.testing.assert_array_equal(store.column("formal_charge"), fixture["values"])
            coordinates = fixture["coordinates"].copy()
            store.set_coordinates(coordinates)
            coordinates[...] = 0
            np.testing.assert_array_equal(store.coordinates(), fixture["coordinates"])
            added = fixture["added"].copy()
            expected = np.concatenate((store.export(FIELDS)[fixture["keep"]], added))[
                fixture["order"]
            ]
            store.remap(fixture["keep"], added, fixture["order"])
            added[...] = 0
            np.testing.assert_array_equal(store.export(FIELDS), expected)
            checks.append({"size": count, "layout": name,
                           "input_and_output_alias_isolation": "passed"})
        for workload, operation in WORKLOADS.items():
            stores = [factory(fixture["matrix"]) for factory in LAYOUTS.values()]
            results = [operation(store, fixture, 3) for store in stores]
            for store, result in zip(stores[1:], results[1:]):
                np.testing.assert_array_equal(store.export(FIELDS), stores[0].export(FIELDS))
                np.testing.assert_array_equal(result, results[0])
            checks.append({"size": count, "workload": workload,
                           "state_and_output_equivalence": "passed"})
    return checks


def _sample(fixtures, layout, count, workload, iterations):
    fixture = fixtures[count]
    store = LAYOUTS[layout](fixture["matrix"])
    start = time.perf_counter_ns()
    result = WORKLOADS[workload](store, fixture, iterations)
    elapsed = time.perf_counter_ns() - start
    del result
    return elapsed


def _summarize(raw):
    rng = np.random.default_rng(SEED)
    rows = []
    for count in SIZES:
        for workload in WORKLOADS:
            values = {
                layout: np.array([
                    entry["ns_per_operation"] for entry in raw
                    if entry["phase"] == "measured" and entry["layout"] == layout
                    and entry["size"] == count and entry["workload"] == workload
                ]) for layout in LAYOUTS
            }
            baseline = values["current_atom_arrays"]
            for layout, times in values.items():
                median = float(np.median(times))
                indices = rng.integers(0, SAMPLES, size=(2000, SAMPLES))
                ratios = np.median(times[indices], axis=1) / np.median(baseline[indices], axis=1)
                rows.append({
                    "size": count, "workload": workload, "layout": layout,
                    "samples": len(times), "median_ns": median,
                    "mad_ns": float(np.median(np.abs(times - median))),
                    "p95_ns": float(np.percentile(times, 95)),
                    "median_ratio_to_current": median / float(np.median(baseline)),
                    "paired_bootstrap_ratio_ci95": np.percentile(ratios, (2.5, 97.5)).tolist(),
                })
    return rows


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--include-private-table", action="store_true")
    parser.add_argument("--shared-load-note", default="Shared-host activity was not controlled.")
    args = parser.parse_args()
    available_cpus = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {args.cpu})
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise FileExistsError("Choose an empty output directory; preserve earlier evidence.")
    repository = Path(__file__).resolve().parents[2]
    private_sources = (_load_private_table(repository, args.output)
                       if args.include_private_table else [])
    (args.output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    fixtures = {count: _fixture(count) for count in SIZES}
    validation = _validate(fixtures)
    _write_json(args.output / "validation.json", validation)
    if args.validate_only:
        print(json.dumps({"validation": "passed", "checks": len(validation),
                          "output": str(args.output)}))
        return
    manifest = {
        "kind": "storage-only prototype, not production performance acceptance",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv, "python": sys.version, "executable": sys.executable,
        "numpy": np.__version__, "platform": platform.platform(), "host": platform.node(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "private_source_files": private_sources,
        "available_cpu_affinity": available_cpus, "measured_cpu_affinity": sorted(os.sched_getaffinity(0)),
        "threads": {name: os.environ[name] for name in THREAD_VARIABLES},
        "shared_load_note": args.shared_load_note, "load_average_before": os.getloadavg(),
        "sizes": SIZES, "warmup_rounds": WARMUPS, "measured_rounds": SAMPLES,
        "order": "all layout/size/workload entries shuffled independently in every round",
        "seed": SEED, "fields": FIELDS, "selected_export_fields": EXPORT_FIELDS,
        "full_export_scope": "requested atom_matrix property: independent complete 19-column float64 matrix",
        "selected_export_scope": "private/native/model task; not an additional public matrix API",
        "dtypes": {field: str(dtype) for field, dtype in DTYPES.items()},
        "layout_dtype_override": {
            "blocked_float_columns": "two float64 blocks; scalar/column export uses the same logical codecs",
            "capacity_fact_table": "actual private two-block float64 table; precomputed column views, validated writes and capacity growth",
        },
        "aos_aligned_record_bytes": ROW_DTYPE.itemsize,
        "input_sha256": {str(count): hashlib.sha256(fixture["matrix"].tobytes()).hexdigest()
                         for count, fixture in fixtures.items()},
        "scope": {
            "included": "numeric storage, scalar conversion, independent reads/exports, writes and row remap",
            "excluded": "Atom handles, chemistry, UID lookup, presence/source/status, commits, native ABI packing and full public API dispatch",
            "dtype_policy": "charge/H/id and other existing non-boolean projections retain float64 storage; no narrow-integer range is approved",
            "memory": "owned NumPy buffers plus Python containers/array headers, deduplicating shared bases; excludes shared field-name strings and caller data",
            "timing": "one complete workload pass; fresh layout construction excluded; repeated operations use one private mutable store",
            "new_only_costs": "presence/source/status and other absent old mechanisms need separate production measurements",
        },
    }
    memory = [{"size": count, "layout": name, **_owned_memory(factory(fixture["matrix"]))}
              for count, fixture in fixtures.items() for name, factory in LAYOUTS.items()]
    order_rng = random.Random(SEED)
    entries = [(layout, count, workload) for layout in LAYOUTS
               for count in SIZES for workload in WORKLOADS]
    raw = []
    failures = []
    _write_json(args.output / "manifest.json", manifest)
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        for phase, rounds in (("warmup", WARMUPS), ("measured", SAMPLES)):
            for round_index in range(rounds):
                order_rng.shuffle(entries)
                for sequence, (layout, count, workload) in enumerate(entries):
                    iterations = max(4, 4096 // count)
                    attempt = {"phase": phase, "round": round_index, "sequence": sequence,
                               "layout": layout, "size": count, "workload": workload,
                               "iterations": iterations}
                    started = time.perf_counter_ns()
                    try:
                        elapsed = _sample(fixtures, layout, count, workload, iterations)
                    except Exception as error:
                        attempt.update(status="failed", error=f"{type(error).__name__}: {error}",
                                       total_attempt_ns=time.perf_counter_ns() - started)
                        failures.append(attempt)
                    else:
                        attempt.update(status="completed", elapsed_ns=elapsed,
                                       ns_per_operation=elapsed / iterations)
                    raw.append(attempt)
                _write_json(args.output / "raw_samples.json", raw)
                print(f"{phase} round {round_index + 1}/{rounds}", flush=True)
    finally:
        _write_json(args.output / "raw_samples.json", raw)
        if gc_enabled:
            gc.enable()
    manifest["load_average_after"] = os.getloadavg()
    manifest["attempts"] = {"requested": len(entries) * (WARMUPS + SAMPLES),
                            "completed": len(raw) - len(failures), "failed": len(failures),
                            "not_run": len(entries) * (WARMUPS + SAMPLES) - len(raw)}
    _write_json(args.output / "manifest.json", manifest)
    if failures:
        raise RuntimeError("Incomplete prototype measurement; all attempts retained, no parity comparison produced.")
    summary = _summarize(raw)
    _write_json(args.output / "manifest.json", manifest)
    _write_json(args.output / "raw_samples.json", raw)
    _write_json(args.output / "summary.json", {"timings": summary, "retained_memory": memory})
    lines = [
        "# Atom storage layout prototype", "",
        "These measurements isolate the same 19-column storage work. They do not",
        "establish production parity or include the new identity/status/commit costs.", "",
        "The full matrix export models the requested atom_matrix property: all 19",
        "columns in float64. Selected-column export is a private/native/model task.", "",
        f"CPU affinity: {args.cpu}; NumPy threads: 1; warmups: {WARMUPS}; samples: {SAMPLES}.",
        f"Shared load: {args.shared_load_note}", "",
        "Charge, H and ID storage retains the current float64 domain; no narrower",
        "integer input contract is inferred from this benchmark.", "",
        "| N | Workload | Layout | Median µs | MAD µs | Ratio to current | Paired 95% CI |",
        "|---:|---|---|---:|---:|---:|---|",
    ]
    for row in summary:
        low, high = row["paired_bootstrap_ratio_ci95"]
        lines.append(f"| {row['size']} | {row['workload']} | {row['layout']} | "
                     f"{row['median_ns'] / 1000:.3f} | {row['mad_ns'] / 1000:.3f} | "
                     f"{row['median_ratio_to_current']:.3f} | {low:.3f}–{high:.3f} |")
    lines += ["", "| N | Layout | Retained bytes | Numeric buffer bytes |",
              "|---:|---|---:|---:|"]
    lines += [f"| {row['size']} | {row['layout']} | {row['retained_bytes']} | "
              f"{row['numeric_buffer_bytes']} |" for row in memory]
    (args.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "measured_entries": len(summary),
                      "raw_samples": len(raw), "validation_checks": len(validation)}))


if __name__ == "__main__":
    main()
