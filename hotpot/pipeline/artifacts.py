"""Deterministic molecular payload identity and pipeline artifact helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Mapping, Optional, Tuple

from .contracts import (
    Artifact,
    JSONValue,
    MolecularPayload,
    StageExecutionError,
)

if TYPE_CHECKING:
    from hotpot.cheminfo.calculator.electronic_state import ElectronicState
    from hotpot.cheminfo.core import Atom, Bond


__all__ = [
    "artifact_from_file",
    "payload_sha256",
]


# Canonical molecular lineage.


def _float_token(value: float) -> str:
    """Represent one floating-point fact without decimal-format ambiguity."""

    return float(value).hex()


def _state_document(
    state: Optional["ElectronicState"],
) -> Optional[Mapping[str, JSONValue]]:
    if state is None:
        return None
    return {
        "assumptions": tuple(state.assumptions),
        "charge": state.charge,
        "charge_source": state.charge_source.value,
        "fragment_charges": tuple(state.fragment_charges),
        "multiplicity": state.multiplicity,
        "spin_source": state.spin_source.value,
        "unpaired_electrons": state.unpaired_electrons,
    }


def _atom_document(atom: "Atom") -> Mapping[str, JSONValue]:
    return {
        "aromatic": bool(atom.is_aromatic),
        "atomic_number": int(atom.atomic_number),
        "coordinates": tuple(_float_token(value) for value in atom.coordinates),
        "formal_charge": int(atom.formal_charge),
        "implicit_hydrogens": int(atom.implicit_hydrogens),
    }


def _bond_document(bond: "Bond") -> Mapping[str, JSONValue]:
    first_index, second_index = sorted((bond.a1idx, bond.a2idx))
    return {
        "atom_indices": (first_index, second_index),
        "bond_direction": bond.bond_direction,
        "bond_kind": bond.bond_kind.value,
        "bond_order": _float_token(bond.bond_order),
    }


def _payload_document(payload: MolecularPayload) -> Mapping[str, JSONValue]:
    records: List[JSONValue] = []
    for record in payload.records:
        mol = record.molecule
        bonds = sorted(
            (_bond_document(bond) for bond in mol.bonds),
            key=lambda item: (
                item["atom_indices"],
                item["bond_order"],
                item["bond_kind"],
            ),
        )
        records.append(
            {
                "atoms": tuple(_atom_document(atom) for atom in mol.atoms),
                "bonds": tuple(bonds),
                "electronic_state": _state_document(record.electronic_state),
                "molecular_charge": int(mol.charge),
            }
        )
    return {"records": tuple(records), "schema_version": 1}


# Molecular and JSON persistence.


def _payload_sdf(payload: MolecularPayload) -> str:
    return "".join(
        record.molecule.copy().write(fmt="sdf", write_single=True)
        for record in payload.records
    )


def _write_text_atomic(path: Path, content: str) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(content, encoding="utf-8")
    os.replace(temporary_path, path)


def _write_json_atomic(path: Path, document: Mapping[str, JSONValue]) -> None:
    _write_text_atomic(
        path,
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


def _write_payload(path: Path, payload: MolecularPayload) -> str:
    content = _payload_sdf(payload)
    _write_text_atomic(path, content)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _copy_file_atomic(source: Path, destination: Path) -> str:
    content = source.read_bytes()
    temporary_path = destination.with_name(f".{destination.name}.tmp")
    temporary_path.write_bytes(content)
    os.replace(temporary_path, destination)
    return hashlib.sha256(content).hexdigest()


# Declared stage-artifact verification.


def _artifact_path(stage_directory: Path, relative_path: Path) -> Path:
    if relative_path.is_absolute() or not relative_path.parts:
        raise StageExecutionError(
            f"stage artifact path must be relative: {relative_path}"
        )
    if any(part in {"", ".", ".."} for part in relative_path.parts):
        raise StageExecutionError(
            f"stage artifact path is unsafe: {relative_path}"
        )

    artifact_path = stage_directory
    for part in relative_path.parts:
        artifact_path = artifact_path / part
        if artifact_path.is_symlink():
            raise StageExecutionError(
                f"stage artifact cannot be a symlink: {relative_path}"
            )
    if not artifact_path.is_file():
        raise StageExecutionError(f"stage artifact does not exist: {relative_path}")
    return artifact_path


def _verify_artifacts(
    stage_directory: Path,
    artifacts: Iterable[Artifact],
) -> Tuple[Mapping[str, JSONValue], ...]:
    documents = []
    for artifact in artifacts:
        artifact_path = _artifact_path(stage_directory, artifact.relative_path)
        content = artifact_path.read_bytes()
        actual_sha256 = hashlib.sha256(content).hexdigest()
        if actual_sha256 != artifact.sha256:
            raise StageExecutionError(
                "stage artifact SHA-256 does not match its declaration: "
                f"{artifact.relative_path}"
            )
        if len(content) != artifact.size_bytes:
            raise StageExecutionError(
                "stage artifact size does not match its declaration: "
                f"{artifact.relative_path}"
            )
        documents.append(
            {
                "relative_path": artifact.relative_path.as_posix(),
                "sha256": actual_sha256,
                "size_bytes": len(content),
            }
        )
    return tuple(documents)


# Public operation.


def artifact_from_file(stage_directory: Path, path: Path) -> Artifact:
    """Create a verified declaration for one regular stage-local file."""

    stage_root = stage_directory.absolute()
    artifact_path = path if path.is_absolute() else stage_directory / path
    try:
        relative_path = artifact_path.absolute().relative_to(stage_root)
    except ValueError as error:
        raise StageExecutionError(
            f"stage artifact is outside its stage directory: {path}"
        ) from error

    verified_path = _artifact_path(stage_directory, relative_path)
    content = verified_path.read_bytes()
    return Artifact(
        relative_path=relative_path,
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
    )


def payload_sha256(payload: MolecularPayload) -> str:
    """Return the deterministic identity of molecular topology, geometry and state."""

    canonical = json.dumps(
        _payload_document(payload),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()

