"""Command-line interface for the optional coordination benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from .configuration import BUILTIN_SUITES, RunProfile
from .pipeline import CASE_RUNNERS
from .runner import (
    DEFAULT_OUTPUT,
    resolve_suite,
    run_benchmark,
    settings_with_overrides,
)


def _indices(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run an opt-in coordination-complex benchmark: Hotpot CBond, "
            "force-field optimization, geometry validation, full trajectory "
            "persistence, reporting, and optional PyMOL rendering."
        )
    )
    parser.add_argument(
        "--suite",
        choices=tuple(BUILTIN_SUITES),
        default="extractants-eu-187",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="custom SMILES file; uses the same Hotpot end-to-end backend",
    )
    parser.add_argument("--metal", default="Eu", help="metal for a custom suite")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--backend",
        choices=tuple(CASE_RUNNERS),
        default="hotpot",
    )
    parser.add_argument(
        "--profile",
        choices=tuple(profile.value for profile in RunProfile),
        default=RunProfile.STANDARD.value,
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="use the smoke settings and, unless --limit is given, one case",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--cases",
        type=_indices,
        help="comma-separated one-based corpus indices, for example 54,61,109",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument(
        "--render",
        choices=("off", "auto", "required"),
        default="off",
        help=(
            "off: scientific artifacts only; auto: render when PyMOL exists; "
            "required: fail when PyMOL is unavailable"
        ),
    )
    parser.add_argument("--render-workers", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--steps-per-epoch", type=int)
    parser.add_argument("--timeout", type=float)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    profile = RunProfile.SMOKE if args.smoke else RunProfile(args.profile)
    limit = 1 if args.smoke and args.limit is None else args.limit
    suite = resolve_suite(args.suite, args.input, args.metal)
    settings = settings_with_overrides(
        profile,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        timeout=args.timeout,
    )
    summary = run_benchmark(
        suite,
        args.output,
        profile=profile,
        settings=settings,
        workers=args.workers,
        limit=limit,
        indices=args.cases,
        resume=args.resume,
        aggregate_only=args.aggregate_only,
        render_mode=args.render,
        render_workers=args.render_workers,
        backend=args.backend,
    )
    print(
        f"completed={summary['sample_count']} "
        f"passed={summary['quality_passed']} "
        f"success_rate={summary['overall_success_rate']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
