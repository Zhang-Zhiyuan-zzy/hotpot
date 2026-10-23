"""Thin entry point for Plan-17 failure and efficiency analysis."""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

main = import_module(
    "tests.benchmarks.coordination_complexes.validation_failure_analysis"
).main


if __name__ == "__main__":
    raise SystemExit(main())
