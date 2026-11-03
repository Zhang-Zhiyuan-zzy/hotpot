"""Stable public result types."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AtomPrediction:
    atom_index: int
    element: str
    mca_kj_mol: float

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class SitePrediction:
    atom_index: int
    element: str
    site_type: str
    mca_kj_mol: float

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class MoleculePrediction:
    smiles: str
    formal_charge: int
    sites: tuple[SitePrediction, ...]
    model_variant: str
    atom_predictions: tuple[AtomPrediction, ...] = ()

    def to_dict(self):
        value = asdict(self)
        value["sites"] = [site.to_dict() for site in self.sites]
        value["atom_predictions"] = [
            atom.to_dict() for atom in self.atom_predictions
        ]
        return value


MCA_MOLECULE_DTO_VERSION = 1


@dataclass(frozen=True, slots=True)
class AtomDTO:
    """The Hotpot atom state consumed by MCA site detection."""

    atomic_number: int
    formal_charge: int
    partial_charge: float | None
    is_aromatic: bool
    implicit_hydrogens: int
    valence: int
    isotope: int
    coordinates: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class BondDTO:
    """A graph edge including Hotpot's bond semantic category."""

    begin_atom_index: int
    end_atom_index: int
    bond_order: float
    bond_kind: str
    bond_direction: str | None


@dataclass(frozen=True, slots=True)
class MoleculeDTO:
    """Versioned immutable payload shared by the P1 and P2 worker pools."""

    version: int
    smiles: str
    formal_charge: int
    atom_symbols: tuple[str, ...]
    atom_count: int
    atoms: tuple[AtomDTO, ...]
    bonds: tuple[BondDTO, ...]
    rdkit_binary: bytes
