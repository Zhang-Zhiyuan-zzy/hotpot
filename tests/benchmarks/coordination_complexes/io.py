"""Dataset loading and serialization helpers for the benchmark."""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np


def json_value(value: object) -> object:
    """Convert scientific report objects without expanding trajectories."""
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: json_value(getattr(value, field.name))
            for field in fields(value)
            if field.name not in {"trajectory", "ligand_build_attempts"}
        }
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically write one JSON evidence file."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(json_value(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def load_smiles(
    path: Path,
    *,
    limit: Optional[int] = None,
    indices: Optional[Sequence[int]] = None,
) -> list[tuple[int, str]]:
    """Load one-indexed SMILES records while preserving corpus indices."""
    records = [
        (index, line.split(maxsplit=1)[0])
        for index, line in enumerate(
            (
                line
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ),
            start=1,
        )
    ]
    if indices is not None:
        selected = set(indices)
        records = [record for record in records if record[0] in selected]
    return records if limit is None else records[:limit]


def sha256_file(path: Path) -> str:
    """Return the content identity stored in the run manifest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_case_reports(output_root: Path) -> list[dict[str, object]]:
    """Read every completed case report in stable case order."""
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((output_root / "cases").glob("*/report.json"))
    ]
