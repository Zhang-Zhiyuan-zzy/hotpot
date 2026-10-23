"""Validation-only coordinate providers for the Plan-17 route matrix."""

from __future__ import annotations

import hashlib
import json
from copy import copy
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from typing import Mapping, Sequence, TYPE_CHECKING

import numpy as np

from hotpot.cheminfo.conformers import (
    ConformerBatch,
    ConformerSampler,
    ConformerSamplingError,
    ConformerSamplingRequest,
    RDKitConformerAdapter,
)
from hotpot.cheminfo.forcefields.working_copy import (
    _copy_molecule_metadata,
    _hydrogenated_working_copy,
    _make_worker_mol,
)
from hotpot.cheminfo.stereo import get_stereo_payload


if TYPE_CHECKING:
    from hotpot.cheminfo.core import Molecule


def _json_value(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        _json_value(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def coordinate_sha256(coordinates: np.ndarray) -> str:
    """Hash exact little-endian float64 coordinate rows."""
    rows = np.ascontiguousarray(coordinates, dtype="<f8")
    return hashlib.sha256(memoryview(rows)).hexdigest()


def component_sha256(molecule: "Molecule") -> str:
    """Identify the exact worker component addressed by a frozen proposal."""
    atom_rows = {
        id(atom): row for row, atom in enumerate(molecule.atoms)
    }
    payload = {
        "atoms": [
            {
                "row": row,
                "atom_id": int(atom.id),
                "atomic_number": int(atom.atomic_number),
                "formal_charge": int(atom.formal_charge),
                "is_aromatic": bool(atom.is_aromatic),
            }
            for row, atom in enumerate(molecule.atoms)
        ],
        "bonds": sorted(
            (
                min(atom_rows[id(bond.atom1)], atom_rows[id(bond.atom2)]),
                max(atom_rows[id(bond.atom1)], atom_rows[id(bond.atom2)]),
                float(bond.bond_order),
                bond.bond_kind.value,
                bond.bond_direction,
            )
            for bond in molecule.bonds
        ),
        "stereo": get_stereo_payload(molecule),
    }
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclass(frozen=True)
class FrozenCoordinateProposal:
    """One precomputed component proposal stored independently of a workflow."""

    component_sha256: str
    coordinate_sha256: str
    coordinates: tuple[tuple[float, float, float], ...]
    ordinal: int

    @classmethod
    def from_array(
        cls,
        molecule: "Molecule",
        coordinates: np.ndarray,
        *,
        ordinal: int,
    ) -> "FrozenCoordinateProposal":
        rows = np.ascontiguousarray(coordinates, dtype=np.float64)
        expected_shape = (len(molecule.atoms), 3)
        if rows.shape != expected_shape or not np.all(np.isfinite(rows)):
            raise ConformerSamplingError(
                "A frozen proposal requires finite coordinates with shape "
                f"{expected_shape}, got {rows.shape}"
            )
        return cls(
            component_sha256=component_sha256(molecule),
            coordinate_sha256=coordinate_sha256(rows),
            coordinates=tuple(tuple(float(value) for value in row) for row in rows),
            ordinal=int(ordinal),
        )

    def coordinate_array(self) -> np.ndarray:
        return np.asarray(self.coordinates, dtype=np.float64)

    def to_manifest(self) -> dict[str, object]:
        return {
            "component_sha256": self.component_sha256,
            "coordinate_sha256": self.coordinate_sha256,
            "atom_count": len(self.coordinates),
            "ordinal": self.ordinal,
        }


@dataclass(frozen=True)
class FrozenCoordinateReplayAdapter:
    """Replay precomputed rows and perform no toolkit coordinate generation."""

    proposals: tuple[FrozenCoordinateProposal, ...]

    def sample(
        self,
        molecule: "Molecule",
        request: ConformerSamplingRequest,
    ) -> ConformerBatch:
        del request
        key = component_sha256(molecule)
        matches = tuple(
            proposal
            for proposal in self.proposals
            if proposal.component_sha256 == key
        )
        if len(matches) != 1:
            raise ConformerSamplingError(
                "Frozen coordinate replay requires exactly one proposal for "
                f"component {key}, found {len(matches)}"
            )
        proposal = matches[0]
        coordinates = proposal.coordinate_array()
        if coordinate_sha256(coordinates) != proposal.coordinate_sha256:
            raise ConformerSamplingError("Frozen coordinate proposal bytes changed")
        return ConformerBatch(coordinates[None, :, :].copy(), (proposal.ordinal,))


def _worker_components(
    molecule: "Molecule",
    *,
    add_hydrogens: bool,
    seed: int,
) -> tuple["Molecule", ...]:
    working = _hydrogenated_working_copy(
        molecule,
        add_hydrogens=add_hydrogens,
        seed=seed,
    )
    worker = _make_worker_mol(working)
    detached = copy(worker)
    _copy_molecule_metadata(worker, detached)
    detached.hide_metal_ligand_bonds(clear_conformers=False)
    return tuple(
        component
        for component in detached.components
        if not component.has_metal
    )


def freeze_rdkit_proposals(
    molecule: "Molecule",
    *,
    seed: int,
    add_hydrogens: bool = True,
    sampler: ConformerSampler | None = None,
) -> FrozenCoordinateReplayAdapter:
    """Precompute F2 on the exact component graph used at worker entry."""
    selected_sampler = sampler if sampler is not None else RDKitConformerAdapter()
    request = ConformerSamplingRequest(conformer_count=1, seed=seed)
    proposals = []
    for component in _worker_components(
        molecule,
        add_hydrogens=add_hydrogens,
        seed=seed,
    ):
        batch = selected_sampler.sample(component, request)
        coordinates = np.asarray(batch.coordinates, dtype=np.float64)
        if coordinates.shape != (1, len(component.atoms), 3):
            raise ConformerSamplingError(
                "F2 freezing requires exactly one mapped conformer per component"
            )
        ordinal = int(batch.ordinals[0]) if batch.ordinals else 0
        proposals.append(
            FrozenCoordinateProposal.from_array(
                component,
                coordinates[0],
                ordinal=ordinal,
            )
        )
    return FrozenCoordinateReplayAdapter(tuple(proposals))


def frozen_proposal_manifest(
    adapter: FrozenCoordinateReplayAdapter,
) -> tuple[dict[str, object], ...]:
    return tuple(proposal.to_manifest() for proposal in adapter.proposals)


__all__ = (
    "FrozenCoordinateProposal",
    "FrozenCoordinateReplayAdapter",
    "component_sha256",
    "coordinate_sha256",
    "freeze_rdkit_proposals",
    "frozen_proposal_manifest",
)
