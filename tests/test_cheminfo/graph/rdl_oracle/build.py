"""Clone, verify, and build the pinned RingDecomposerLib test oracle."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Sequence

RDL_REPOSITORY = "https://github.com/rareylab/RingDecomposerLib.git"
RDL_COMMIT = "3a7ff93de0d9c4f6a5661508549c6063573f39c7"


def _run(command: Sequence[str], *, cwd: Optional[Path] = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _verified_source(source_dir: Path, *, clone: bool) -> Path:
    if clone:
        _run(("git", "clone", RDL_REPOSITORY, str(source_dir)))
        _run(("git", "checkout", "--detach", RDL_COMMIT), cwd=source_dir)

    actual_commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=source_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_commit != RDL_COMMIT:
        raise SystemExit(
            "RingDecomposerLib source is not the pinned commit: "
            f"expected {RDL_COMMIT}, got {actual_commit}"
        )
    tracked_changes = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=no"),
        cwd=source_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if tracked_changes:
        raise SystemExit("RingDecomposerLib source contains tracked modifications")
    return source_dir


def _find_library(build_dir: Path) -> Path:
    candidates = tuple(
        path
        for pattern in (
            "**/libRingDecomposerLib.so",
            "**/libRingDecomposerLib.dylib",
            "**/RingDecomposerLib.dll",
        )
        for path in build_dir.glob(pattern)
    )
    if len(candidates) != 1:
        raise SystemExit(
            f"Expected one RDL shared library under {build_dir}, found {candidates}"
        )
    return candidates[0].resolve()


def build_oracle(source_dir: Path, build_dir: Path, jobs: int) -> Path:
    """Build the fixed upstream oracle and return its shared-library path."""
    _run(
        (
            "cmake",
            "-S",
            str(source_dir),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_MINIMAL_EXAMPLE=OFF",
            "-DBUILD_PYTHON_WRAPPER=OFF",
            "-DBUILD_RDKIT_BENCHMARK=OFF",
        )
    )
    _run(
        (
            "cmake",
            "--build",
            str(build_dir),
            "--target",
            "RingDecomposerLib",
            "--parallel",
            str(jobs),
        )
    )
    return _find_library(build_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument(
        "--clone",
        action="store_true",
        help="clone the pinned upstream repository into --source",
    )
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    arguments = parser.parse_args()

    if shutil.which("cmake") is None:
        raise SystemExit("cmake is required to build the RDL oracle")
    source_dir = _verified_source(arguments.source.resolve(), clone=arguments.clone)
    library_path = build_oracle(source_dir, arguments.build.resolve(), arguments.jobs)
    print(f'export HOTPOT_RDL_ORACLE_LIBRARY="{library_path}"')


if __name__ == "__main__":
    main()
