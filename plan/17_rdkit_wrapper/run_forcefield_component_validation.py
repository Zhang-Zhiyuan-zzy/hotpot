#!/usr/bin/env python3
"""Thin entry point for the Plan-17 force-field component validation."""

from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tests.benchmarks.coordination_complexes.validation_runner import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
