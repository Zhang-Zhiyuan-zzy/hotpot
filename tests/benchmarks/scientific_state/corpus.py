"""Immutable corpus protocol for scientific-state dry resolution."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

__all__ = [
    "DryStateCase",
    "DryStateCorpus",
    "DryStateSourceKind",
    "load_corpus",
    "load_structure_registry",
    "load_topology_manifest",
]


class DryStateSourceKind(str, Enum):
    """Representation used to reconstruct one benchmark molecule."""

    SMILES = "smiles"
    FILE = "file"


@dataclass(frozen=True)
class DryStateCase:
    """One ordered molecular record in a dry-resolution corpus."""

    index: int
    case_id: str
    source_kind: DryStateSourceKind
    source: str
    format: str
    source_sha256: str
    coordination_bonds: tuple[tuple[int, int], ...] = ()


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
    return DryStateCase(
        index=index,
        case_id=case_id,
        source_kind=DryStateSourceKind.SMILES,
        source=smiles,
        format="smi",
        source_sha256=hashlib.sha256(smiles.encode("utf-8")).hexdigest(),
    )


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


def load_structure_registry(
    path: Path,
    structure_root: Path,
) -> DryStateCorpus:
    """Load the canonical structure registry without rewriting its assets."""

    encoded = path.read_bytes()
    root = structure_root.resolve()
    rows = csv.DictReader(encoded.decode("utf-8").splitlines())
    required = {
        "structure_id",
        "format",
        "store_path",
        "canonical_file_sha256",
    }
    if rows.fieldnames is None or not required.issubset(rows.fieldnames):
        raise ValueError("Structure registry lacks required identity fields")
    cases = []
    for index, row in enumerate(rows, start=2):
        source = (root / row["store_path"]).resolve()
        if not source.is_relative_to(root):
            raise ValueError(f"Structure row {index} escapes its asset root")
        cases.append(
            DryStateCase(
                index=index,
                case_id=row["structure_id"],
                source_kind=DryStateSourceKind.FILE,
                source=str(source),
                format=row["format"],
                source_sha256=row["canonical_file_sha256"],
            )
        )
    if not cases:
        raise ValueError("Structure registry contains no molecular records")
    case_ids = tuple(case.case_id for case in cases)
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Structure registry contains duplicate structure IDs")
    return DryStateCorpus(
        source=path.resolve(),
        source_sha256=hashlib.sha256(encoded).hexdigest(),
        cases=tuple(cases),
    )


def _coordination_pairs(value: object, line_number: int) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, list):
        raise ValueError(
            f"Topology manifest line {line_number} requires coordination_bonds"
        )
    pairs = []
    for item in value:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or any(type(index) is not int or index < 0 for index in item)
            or item[0] == item[1]
        ):
            raise ValueError(
                f"Topology manifest line {line_number} has an invalid atom pair"
            )
        pairs.append(tuple(sorted((item[0], item[1]))))
    result = tuple(sorted(pairs))
    if len(set(result)) != len(result):
        raise ValueError(
            f"Topology manifest line {line_number} repeats a coordination bond"
        )
    return result


def load_topology_manifest(
    path: Path,
    structure_root: Path,
) -> DryStateCorpus:
    """Load explicitly curated coordination bonds from a JSON Lines manifest."""

    encoded = path.read_bytes()
    root = structure_root.resolve()
    cases = []
    for line_number, line in enumerate(
        encoded.decode("utf-8").splitlines(),
        start=1,
    ):
        stripped = line.strip()
        if not stripped:
            continue
        payload: object = json.loads(stripped)
        if not isinstance(payload, dict):
            raise ValueError(
                f"Topology manifest line {line_number} must be an object"
            )
        required = {
            "case_id",
            "format",
            "store_path",
            "source_sha256",
            "coordination_bonds",
        }
        if set(payload) != required:
            raise ValueError(
                f"Topology manifest line {line_number} fields do not match"
            )
        case_id = payload["case_id"]
        format_name = payload["format"]
        store_path = payload["store_path"]
        source_sha256 = payload["source_sha256"]
        if any(
            not isinstance(value, str) or not value
            for value in (case_id, format_name, store_path, source_sha256)
        ):
            raise ValueError(
                f"Topology manifest line {line_number} has invalid identity fields"
            )
        source = (root / store_path).resolve()
        if not source.is_relative_to(root):
            raise ValueError(
                f"Topology manifest line {line_number} escapes its asset root"
            )
        cases.append(
            DryStateCase(
                index=line_number,
                case_id=case_id,
                source_kind=DryStateSourceKind.FILE,
                source=str(source),
                format=format_name,
                source_sha256=source_sha256,
                coordination_bonds=_coordination_pairs(
                    payload["coordination_bonds"],
                    line_number,
                ),
            )
        )
    if not cases:
        raise ValueError("Topology manifest contains no molecular records")
    case_ids = tuple(case.case_id for case in cases)
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Topology manifest contains duplicate case IDs")
    return DryStateCorpus(
        source=path.resolve(),
        source_sha256=hashlib.sha256(encoded).hexdigest(),
        cases=tuple(cases),
    )
