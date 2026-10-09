"""Frozen input contracts shared by the four force-field benchmarks."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from .backend_comparison import (
    CanonicalCase,
    export_canonical_manifest,
    load_canonical_cases,
)
from .io import load_smiles, sha256_file

SCHEMA_VERSION = 1
EXPECTED_INPUT_COUNT = 187
EXPECTED_COMPLEX_COUNT = 182


@dataclass(frozen=True)
class LigandCase:
    """One indexed ligand from the immutable input corpus."""

    index: int
    smiles: str
    seed: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkCohort:
    """The complete ligand corpus and its CBond-eligible complex subset."""

    input_path: Path
    input_sha256: str
    cohort_path: Path
    cohort_sha256: str
    ligand_cases: tuple[LigandCase, ...]
    complex_cases: Mapping[int, CanonicalCase]

    @property
    def complex_indices(self) -> tuple[int, ...]:
        return tuple(sorted(self.complex_cases))


def _resolve_manifest(
    output_root: Path,
    *,
    reference_root: Optional[Path],
    cohort_path: Optional[Path],
) -> Path:
    if (reference_root is None) == (cohort_path is None):
        raise ValueError("provide exactly one of reference_root or cohort_path")
    if cohort_path is not None:
        return cohort_path.resolve()

    manifest_path = output_root / "canonical_cases.json"
    if not manifest_path.is_file():
        export_canonical_manifest(
            reference_root.resolve(),
            manifest_path,
            expected_count=EXPECTED_COMPLEX_COUNT,
        )
    return manifest_path


def resolve_cohort(
    input_path: Path,
    output_root: Path,
    *,
    reference_root: Optional[Path] = None,
    cohort_path: Optional[Path] = None,
    seed: int = 20260921,
) -> BenchmarkCohort:
    """Load and cross-check the 187 ligands and frozen 182-complex cohort."""
    input_path = input_path.resolve()
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    records = load_smiles(input_path)
    if len(records) != EXPECTED_INPUT_COUNT:
        raise ValueError(
            f"expected {EXPECTED_INPUT_COUNT} input ligands, found {len(records)}"
        )

    resolved_cohort_path = _resolve_manifest(
        output_root,
        reference_root=reference_root,
        cohort_path=cohort_path,
    )
    manifest = json.loads(resolved_cohort_path.read_text(encoding="utf-8"))
    source_input_sha256 = manifest.get("source", {}).get("input_sha256")
    input_sha256 = sha256_file(input_path)
    if source_input_sha256 is not None and source_input_sha256 != input_sha256:
        raise ValueError("canonical cohort and input corpus SHA-256 differ")
    canonical_cases = load_canonical_cases(resolved_cohort_path)
    if len(canonical_cases) != EXPECTED_COMPLEX_COUNT:
        raise ValueError(
            f"expected {EXPECTED_COMPLEX_COUNT} CBond-eligible complexes, "
            f"found {len(canonical_cases)}"
        )

    smiles_by_index = dict(records)
    complex_by_index: dict[int, CanonicalCase] = {}
    for case in canonical_cases:
        if case.index not in smiles_by_index:
            raise ValueError(
                f"complex case {case.index} is absent from the input corpus"
            )
        if case.smiles != smiles_by_index[case.index]:
            raise ValueError(
                f"complex case {case.index} SMILES differs from the input corpus"
            )
        if case.index in complex_by_index:
            raise ValueError(f"duplicate complex case index {case.index}")
        complex_by_index[case.index] = case

    ligand_cases = tuple(
        LigandCase(index=index, smiles=smiles, seed=seed + index)
        for index, smiles in records
    )
    return BenchmarkCohort(
        input_path=input_path,
        input_sha256=input_sha256,
        cohort_path=resolved_cohort_path,
        cohort_sha256=sha256_file(resolved_cohort_path),
        ligand_cases=ligand_cases,
        complex_cases=complex_by_index,
    )


__all__ = (
    "EXPECTED_COMPLEX_COUNT",
    "EXPECTED_INPUT_COUNT",
    "BenchmarkCohort",
    "LigandCase",
    "resolve_cohort",
)
