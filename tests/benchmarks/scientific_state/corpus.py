"""Immutable corpus protocol for scientific-state dry resolution."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

__all__ = ["DryStateCase", "DryStateCorpus", "load_corpus"]


@dataclass(frozen=True)
class DryStateCase:
    """One ordered molecular record in a dry-resolution corpus."""

    index: int
    case_id: str
    smiles: str


@dataclass(frozen=True)
class DryStateCorpus:
    """Source-bound collection of dry-resolution records."""

    source: Path
    source_sha256: str
    cases: tuple[DryStateCase, ...]


def _case_from_line(index: int, line: str) -> DryStateCase:
    fields = line.split(maxsplit=1)
    smiles = fields[0]
    case_id = fields[1].strip() if len(fields) == 2 else f"case-{index:06d}"
    if not case_id:
        raise ValueError(f"Corpus line {index} has an empty case ID")
    return DryStateCase(index, case_id, smiles)


def load_corpus(path: Path) -> DryStateCorpus:
    """Load non-empty, non-commented SMILES records from ``path``."""

    encoded = path.read_bytes()
    lines = encoded.decode("utf-8").splitlines()
    cases = tuple(
        _case_from_line(index, stripped)
        for index, line in enumerate(lines, start=1)
        if (stripped := line.strip()) and not stripped.startswith("#")
    )
    if not cases:
        raise ValueError("Dry-resolution corpus contains no molecular records")
    case_ids = tuple(case.case_id for case in cases)
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Dry-resolution corpus contains duplicate case IDs")
    return DryStateCorpus(
        source=path.resolve(),
        source_sha256=hashlib.sha256(encoded).hexdigest(),
        cases=cases,
    )
