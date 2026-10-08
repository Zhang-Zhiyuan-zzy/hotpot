"""Experiment orchestration for the coordination-complex benchmark."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import subprocess
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Optional, Sequence

from .configuration import (
    BUILTIN_SUITES,
    REPOSITORY_ROOT,
    BenchmarkSettings,
    BenchmarkSuite,
    RunProfile,
    settings_for_profile,
)
from .io import load_case_reports, load_smiles, sha256_file, write_json
from .pipeline import CASE_RUNNERS
from .rendering import render_experiment
from .reporting import aggregate_run

DEFAULT_OUTPUT = REPOSITORY_ROOT / "movie/benchmarks/extractants_eu_187"


def _git_state() -> dict[str, object]:
    revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    status = subprocess.run(
        ("git", "status", "--porcelain"),
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    return {
        "commit": revision.stdout.strip() if revision.returncode == 0 else None,
        "dirty": status.returncode != 0 or bool(status.stdout.strip()),
    }


def _scientific_configuration(
    suite: BenchmarkSuite,
    settings: BenchmarkSettings,
    profile: RunProfile,
    input_sha256: str,
    selected_indices: Sequence[int],
    backend: str,
) -> dict[str, object]:
    return {
        "backend": backend,
        "workflow": "cbond-complexes-build",
        "suite": suite.to_manifest(),
        "profile": profile.value,
        "input_sha256": input_sha256,
        "settings": settings.to_manifest(),
        "selected_indices": list(selected_indices),
    }


def _write_or_check_manifest(
    output_root: Path,
    scientific_configuration: dict[str, object],
    *,
    workers: int,
    resume: bool,
) -> None:
    manifest_path = output_root / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not resume:
            raise FileExistsError(
                f"{manifest_path} already exists; use --resume or a new output path"
            )
        if existing["scientific_configuration"] != scientific_configuration:
            raise ValueError(
                "--resume configuration differs from the existing manifest"
            )
        return
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git": _git_state(),
        "scientific_configuration": scientific_configuration,
        "execution": {"workers": workers, "multiprocessing_start_method": "spawn"},
    }
    write_json(manifest_path, manifest)


def _failed_worker_record(
    index: int,
    smiles: str,
    error: BaseException,
) -> dict[str, object]:
    return {
        "index": index,
        "smiles": smiles,
        "backend": "hotpot",
        "workflow": "cbond-complexes-build",
        "status": "failed_worker",
        "phase": "worker",
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
        "total_seconds": 0.0,
    }


def run_benchmark(
    suite: BenchmarkSuite,
    output_root: Path,
    *,
    profile: RunProfile = RunProfile.STANDARD,
    settings: Optional[BenchmarkSettings] = None,
    workers: int = 16,
    limit: Optional[int] = None,
    indices: Optional[Sequence[int]] = None,
    resume: bool = False,
    aggregate_only: bool = False,
    render_mode: str = "off",
    render_workers: Optional[int] = None,
    backend: str = "hotpot",
) -> dict[str, object]:
    """Execute or re-aggregate one reproducible benchmark selection."""
    resolved_settings = settings or settings_for_profile(profile)
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if aggregate_only and not (output_root / "manifest.json").is_file():
        raise FileNotFoundError(
            "--aggregate-only requires an existing benchmark manifest"
        )
    all_records = load_smiles(suite.input_path)
    if suite.expected_count and len(all_records) != suite.expected_count:
        raise ValueError(
            f"suite {suite.name!r} expected {suite.expected_count} records, "
            f"found {len(all_records)}"
        )
    if indices is not None:
        available_indices = {index for index, _ in all_records}
        missing_indices = sorted(set(indices) - available_indices)
        if missing_indices:
            raise ValueError(f"corpus indices do not exist: {missing_indices}")
    selected = load_smiles(suite.input_path, limit=limit, indices=indices)
    if not selected:
        raise ValueError("benchmark selection contains no molecules")
    expected_indices = tuple(index for index, _ in selected)
    scientific_configuration = _scientific_configuration(
        suite,
        resolved_settings,
        profile,
        sha256_file(suite.input_path),
        expected_indices,
        backend,
    )
    _write_or_check_manifest(
        output_root,
        scientific_configuration,
        workers=workers,
        resume=resume or aggregate_only,
    )

    os.environ["HOTPOT_CBOND_DEVICE"] = "cpu"
    wall_seconds = None
    wall_seconds_scope = "unavailable"
    if aggregate_only:
        summary_path = output_root / "summary.json"
        if summary_path.is_file():
            previous_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            wall_seconds = previous_summary.get("wall_seconds")
            wall_seconds_scope = str(
                previous_summary.get("wall_seconds_scope", "unavailable")
            )
        records = [
            record
            for record in load_case_reports(output_root)
            if int(record["index"]) in set(expected_indices)
        ]
    else:
        tasks = [
            (
                index,
                smiles,
                str(output_root),
                suite.metal,
                resolved_settings,
                resume,
            )
            for index, smiles in selected
        ]
        records = []
        preexisting = len(load_case_reports(output_root)) if resume else 0
        started = perf_counter()
        case_runner = CASE_RUNNERS[backend]
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            futures = {
                pool.submit(case_runner, *task): (task[0], task[1]) for task in tasks
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                index, smiles = futures[future]
                try:
                    record = future.result()
                except Exception as error:
                    record = _failed_worker_record(index, smiles, error)
                    case_dir = output_root / "cases" / f"{index:04d}"
                    case_dir.mkdir(parents=True, exist_ok=True)
                    write_json(case_dir / "report.json", record)
                records.append(record)
                print(
                    f"[{completed}/{len(tasks)}] case={index:04d} "
                    f"status={record['status']} phase={record['phase']} "
                    f"seconds={float(record['total_seconds']):.1f}",
                    flush=True,
                )
        if preexisting == 0:
            wall_seconds = perf_counter() - started
            wall_seconds_scope = "complete_validation_invocation"
        else:
            wall_seconds_scope = "unavailable_after_resumed_partial_run"

    summary = aggregate_run(
        output_root,
        records,
        suite,
        resolved_settings,
        profile,
        expected_indices,
        wall_seconds=wall_seconds,
        wall_seconds_scope=wall_seconds_scope,
    )
    if render_mode != "off" or not (output_root / "render_report.json").is_file():
        render_experiment(
            output_root,
            title=suite.title,
            workers=render_workers or workers,
            mode=render_mode,
        )
    return summary


def settings_with_overrides(
    profile: RunProfile,
    *,
    epochs: Optional[int] = None,
    steps_per_epoch: Optional[int] = None,
    timeout: Optional[float] = None,
) -> BenchmarkSettings:
    """Apply explicit CLI overrides while retaining manifest visibility."""
    settings = settings_for_profile(profile)
    changes = {
        name: value
        for name, value in {
            "epochs": epochs,
            "steps_per_epoch": steps_per_epoch,
            "timeout": timeout,
        }.items()
        if value is not None
    }
    return replace(settings, **changes)


def resolve_suite(name: str, input_path: Optional[Path], metal: str) -> BenchmarkSuite:
    """Resolve a built-in suite or an explicitly named custom corpus."""
    if input_path is None:
        return BUILTIN_SUITES[name]
    records = load_smiles(input_path)
    return BenchmarkSuite(
        name="custom",
        input_path=input_path.resolve(),
        expected_count=len(records),
        metal=metal,
        title=f"{metal} coordination complexes from {input_path.name}",
    )
