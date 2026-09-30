"""Measure force-field trajectory retention cost in fresh processes.

Each case/repeat/retention combination runs in a new Python process.  The
reported ``peak_rss_kib`` is Linux ``ru_maxrss`` for the complete worker
process, including imports, CBond inference, and force-field execution.  Wall
time covers only ``ff.complexes_build``.  No trajectory is written to disk.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Mapping, Optional, Sequence

from tests.benchmarks.coordination_complexes.configuration import (
    BUILTIN_SUITES,
    RunProfile,
    settings_for_profile,
)
from tests.benchmarks.coordination_complexes.io import load_smiles


DEFAULT_CASES = (1, 54, 61, 109)
_WORKER_PREFIX = "HOTPOT_RETENTION_SAMPLE="


def parse_cases(value: str) -> tuple[int, ...]:
    """Parse one-based comma-separated corpus indices."""
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Profile in-memory force-field trajectory retention with fresh "
            "worker processes."
        ),
    )
    parser.add_argument("--cases", type=parse_cases, default=DEFAULT_CASES)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--profile",
        choices=tuple(profile.value for profile in RunProfile),
        default=RunProfile.STANDARD.value,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--repeat", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument(
        "--save-movie",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def _trajectory_metrics(archive: object) -> dict[str, int]:
    if archive is None:
        return {
            "frame_count": 0,
            "main_frame_count": 0,
            "coordinate_revision_count": 0,
            "trajectory_coordinate_bytes": 0,
        }
    trajectories = (archive.main, *archive.ligand_build_attempts)
    return {
        "frame_count": sum(len(trajectory) for trajectory in trajectories),
        "main_frame_count": len(archive.main),
        "coordinate_revision_count": sum(
            trajectory.coordinate_revision_count for trajectory in trajectories
        ),
        "trajectory_coordinate_bytes": sum(
            trajectory.coordinate_revision_count * len(trajectory.atoms) * 3 * 8
            for trajectory in trajectories
        ),
    }


def _conformer_coordinate_bytes(mol: object) -> int:
    if mol.conformers_number == 0:
        return 0
    return sum(
        int(mol.conformer_get(index)["coordinates"].nbytes)
        for index in range(mol.conformers_number)
    )


def run_worker(
    case_index: int,
    *,
    repeat: int,
    save_movie: bool,
    profile: RunProfile,
) -> dict[str, object]:
    """Run one complete CBond/setup/force-field sample in this process."""
    from hotpot import read_mol
    from hotpot.cheminfo import forcefields as ff
    from hotpot.cheminfo.AImodels.cbond.apply import (
        auto_build_cbond,
        get_cbond_runtime,
    )

    suite = BUILTIN_SUITES["extractants-eu-187"]
    records = dict(load_smiles(suite.input_path, indices=(case_index,)))
    smiles = records[case_index]
    settings = settings_for_profile(profile)
    os.environ["HOTPOT_CBOND_DEVICE"] = "cpu"

    ligand = read_mol(smiles, fmt="smi")
    try:
        cbond_result = auto_build_cbond(
            ligand,
            suite.metal,
            threshold=settings.cbond_threshold,
            runtime=get_cbond_runtime("cpu"),
            return_details=True,
        )
    except Exception as error:
        return _sample_payload(
            case_index,
            repeat,
            save_movie,
            status="failed_cbond",
            wall_seconds=0.0,
            archive=None,
            mol=None,
            error=error,
        )

    complex_mol = cbond_result.molecule
    archive = None
    started = perf_counter()
    try:
        report = ff.complexes_build(
            complex_mol,
            epochs=settings.epochs,
            steps_per_epoch=settings.steps_per_epoch,
            max_attempts=settings.max_attempts,
            candidate_warmup_steps=settings.candidate_warmup_steps,
            candidate_score_steps=settings.candidate_score_steps,
            best_candidate_refine_steps=settings.best_candidate_refine_steps,
            ligand_untangling_attempts=settings.ligand_untangling_attempts,
            coordination_restoration_attempts=(
                settings.coordination_restoration_attempts
            ),
            coordination_relaxation_steps=settings.coordination_relaxation_steps,
            complex_untangling_attempts=settings.complex_untangling_attempts,
            timeout=settings.timeout,
            quality_level=settings.quality_level,
            seed=settings.seed + case_index,
            perturb_sigma=settings.perturb_sigma,
            save_movie=save_movie,
            trajectory_start=ff.TrajectoryStart(settings.trajectory_start),
            trajectory_path=None,
        )
        archive = report.trajectory
        status = "passed" if report.quality_report.passed else "failed_quality"
        error = None
    except ff.ForceFieldError as caught:
        archive = caught.trajectory
        status = "failed_forcefield"
        error = caught
    wall_seconds = perf_counter() - started
    return _sample_payload(
        case_index,
        repeat,
        save_movie,
        status=status,
        wall_seconds=wall_seconds,
        archive=archive,
        mol=complex_mol,
        error=error,
    )


def _sample_payload(
    case_index: int,
    repeat: int,
    save_movie: bool,
    *,
    status: str,
    wall_seconds: float,
    archive: object,
    mol: object,
    error: Optional[BaseException],
) -> dict[str, object]:
    trajectory = _trajectory_metrics(archive)
    conformer_bytes = 0 if mol is None else _conformer_coordinate_bytes(mol)
    payload: dict[str, object] = {
        "case": case_index,
        "repeat": repeat,
        "save_movie": save_movie,
        "status": status,
        "wall_seconds": wall_seconds,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        **trajectory,
        "materialized_conformer_coordinate_bytes": conformer_bytes,
        "retained_coordinate_bytes": (
            trajectory["trajectory_coordinate_bytes"] + conformer_bytes
        ),
    }
    if error is not None:
        payload["error_type"] = type(error).__name__
        payload["error_message"] = str(error)
    return payload


def _median(samples: Sequence[Mapping[str, object]], field: str) -> float:
    return statistics.median(float(sample[field]) for sample in samples)


def aggregate_samples(samples: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Aggregate samples while preserving paired retention comparisons."""
    by_retention = {
        save_movie: [
            sample for sample in samples if bool(sample["save_movie"]) is save_movie
        ]
        for save_movie in (False, True)
    }
    modes = {
        str(save_movie).lower(): {
            "sample_count": len(mode_samples),
            "status_counts": dict(
                sorted(
                    Counter(
                        str(sample["status"]) for sample in mode_samples
                    ).items()
                )
            ),
            "median_peak_rss_kib": _median(mode_samples, "peak_rss_kib"),
            "median_wall_seconds": _median(mode_samples, "wall_seconds"),
            "median_frame_count": _median(mode_samples, "frame_count"),
            "median_retained_coordinate_bytes": _median(
                mode_samples,
                "retained_coordinate_bytes",
            ),
        }
        for save_movie, mode_samples in by_retention.items()
        if mode_samples
    }

    indexed = {
        (int(sample["case"]), int(sample["repeat"]), bool(sample["save_movie"])): sample
        for sample in samples
    }
    paired = []
    for case, repeat, save_movie in sorted(indexed):
        if save_movie:
            continue
        without_movie = indexed[(case, repeat, False)]
        with_movie = indexed.get((case, repeat, True))
        if with_movie is None:
            continue
        rss_delta = int(with_movie["peak_rss_kib"]) - int(
            without_movie["peak_rss_kib"]
        )
        wall_delta = float(with_movie["wall_seconds"]) - float(
            without_movie["wall_seconds"]
        )
        paired.append(
            {
                "case": case,
                "repeat": repeat,
                "peak_rss_delta_kib": rss_delta,
                "peak_rss_delta_percent": (
                    100.0 * rss_delta / int(without_movie["peak_rss_kib"])
                ),
                "wall_seconds_delta": wall_delta,
                "wall_seconds_delta_percent": (
                    100.0 * wall_delta / float(without_movie["wall_seconds"])
                    if float(without_movie["wall_seconds"])
                    else 0.0
                ),
                "retained_coordinate_bytes_delta": (
                    int(with_movie["retained_coordinate_bytes"])
                    - int(without_movie["retained_coordinate_bytes"])
                ),
            }
        )
    return {"modes": modes, "paired_deltas": paired}


def _run_fresh_worker(
    case_index: int,
    repeat: int,
    save_movie: bool,
    profile: RunProfile,
) -> dict[str, object]:
    command = [
        sys.executable,
        "-m",
        "tests.performance.profile_forcefield_retention",
        "--worker",
        str(case_index),
        "--repeat",
        str(repeat),
        "--profile",
        profile.value,
    ]
    if save_movie:
        command.append("--save-movie")
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    worker_lines = [
        line
        for line in completed.stdout.splitlines()
        if line.startswith(_WORKER_PREFIX)
    ]
    return json.loads(worker_lines[-1][len(_WORKER_PREFIX) :])


def run_profile(
    cases: Sequence[int],
    repeats: int,
    profile: RunProfile,
) -> dict[str, object]:
    samples = [
        _run_fresh_worker(case, repeat, save_movie, profile)
        for case in cases
        for repeat in range(repeats)
        for save_movie in (False, True)
    ]
    samples.sort(
        key=lambda sample: (
            sample["case"],
            sample["repeat"],
            sample["save_movie"],
        )
    )
    return {
        "schema_version": 1,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "peak_rss_unit": "KiB (Linux ru_maxrss)",
            "peak_rss_scope": "complete fresh worker process",
            "wall_time_scope": "ff.complexes_build only",
            "coordinate_byte_accounting": {
                "trajectory_coordinate_bytes": (
                    "unique pooled float64 coordinate revisions"
                ),
                "materialized_conformer_coordinate_bytes": (
                    "float64 coordinates retained by Molecule.conformers"
                ),
                "retained_coordinate_bytes": (
                    "trajectory plus materialized conformer coordinates; "
                    "excludes Python object and topology overhead"
                ),
            },
        },
        "configuration": {
            "suite": "extractants-eu-187",
            "cases": list(cases),
            "repeats": repeats,
            "profile": profile.value,
            "retention_modes": [False, True],
            "trajectory_path": None,
        },
        "samples": samples,
        "summary": aggregate_samples(samples),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = build_parser().parse_args(argv)
    profile = RunProfile(arguments.profile)
    if arguments.worker is not None:
        sample = run_worker(
            arguments.worker,
            repeat=arguments.repeat,
            save_movie=arguments.save_movie,
            profile=profile,
        )
        print(_WORKER_PREFIX + json.dumps(sample, sort_keys=True), flush=True)
        return 0

    report = run_profile(arguments.cases, arguments.repeats, profile)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
