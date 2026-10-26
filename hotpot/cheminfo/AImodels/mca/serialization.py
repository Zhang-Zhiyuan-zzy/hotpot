"""Spawn-safe molecular transfer objects for the MCA CPU stages.

The site detector consumes Hotpot's coordination-aware graph semantics while
the featurizer consumes RDKit stereochemistry.  ``MoleculeDTO`` therefore
stores both representations produced by the single public-input
normalization step instead of relying on either toolkit object's pickle
implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem

from hotpot.cheminfo.convert import to_hotpot_mol
from hotpot.cheminfo.core import Molecule as HotpotMolecule


MCA_MOLECULE_DTO_VERSION = 1


@dataclass(frozen=True, slots=True)
class AtomDTO:
    """The Hotpot atom state consumed by MCA site detection."""

    atomic_number: int
    formal_charge: int
    partial_charge: float
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


def _feature_molecule(value, native_mol: HotpotMolecule) -> Chem.Mol:
    if isinstance(value, Chem.Mol):
        return Chem.Mol(value)
    mol = Chem.Mol(native_mol.to_rdmol())
    Chem.SanitizeMol(mol)
    return mol


def normalize_molecule(value) -> MoleculeDTO:
    """Normalize one public MCA input exactly once into a process-safe DTO."""

    native_mol = to_hotpot_mol(value)
    feature_mol = _feature_molecule(value, native_mol)
    atoms = tuple(
        AtomDTO(
            atomic_number=int(atom.atomic_number),
            formal_charge=int(atom.formal_charge),
            partial_charge=float(atom.partial_charge),
            is_aromatic=bool(atom.is_aromatic),
            implicit_hydrogens=int(atom.implicit_hydrogens),
            valence=int(atom.valence),
            isotope=int(getattr(atom, "isotope", 0)),
            coordinates=tuple(float(coordinate) for coordinate in atom.coordinates),
        )
        for atom in native_mol.atoms
    )
    bonds = tuple(
        BondDTO(
            begin_atom_index=int(bond.a1idx),
            end_atom_index=int(bond.a2idx),
            bond_order=float(bond.bond_order),
            bond_kind=bond.bond_kind.value,
            bond_direction=bond.bond_direction,
        )
        for bond in native_mol.bonds
    )
    atom_symbols = tuple(atom.symbol for atom in native_mol.atoms)
    return MoleculeDTO(
        version=MCA_MOLECULE_DTO_VERSION,
        smiles=native_mol.smiles,
        formal_charge=int(native_mol.charge),
        atom_symbols=atom_symbols,
        atom_count=len(atom_symbols),
        atoms=atoms,
        bonds=bonds,
        rdkit_binary=bytes(
            feature_mol.ToBinary(Chem.PropertyPickleOptions.AllProps)
        ),
    )


def validate_dto(
    molecule: MoleculeDTO,
    position: int,
    allow_charged: bool = False,
) -> None:
    """Apply the existing MCA target-domain checks to one normalized input."""

    if any(atom.atomic_number == 1 for atom in molecule.atoms):
        raise ValueError(
            f"Explicit hydrogen atoms at molecule position {position} cannot "
            "be MCA targets; remove explicit hydrogens before prediction"
        )
    if molecule.formal_charge != sum(atom.formal_charge for atom in molecule.atoms):
        raise ValueError(
            "Molecular total charge does not match the sum of atom formal "
            f"charges at molecule position {position}"
        )
    if not allow_charged and molecule.formal_charge != 0:
        raise ValueError(
            f"Charged molecule at position {position} is outside the validated "
            "domain; set allow_charged=True to opt in"
        )


def hotpot_mol_from_dto(molecule: MoleculeDTO) -> HotpotMolecule:
    """Rebuild the Hotpot graph used by the P1 site detector."""

    if molecule.version != MCA_MOLECULE_DTO_VERSION:
        raise ValueError(
            f"Unsupported MCA molecule DTO version {molecule.version}; "
            f"expected {MCA_MOLECULE_DTO_VERSION}"
        )
    native_mol = HotpotMolecule()
    for atom in molecule.atoms:
        native_mol._create_atom(
            atomic_number=atom.atomic_number,
            formal_charge=atom.formal_charge,
            partial_charge=atom.partial_charge,
            is_aromatic=atom.is_aromatic,
            implicit_hydrogens=atom.implicit_hydrogens,
            valence=atom.valence,
            isotope=atom.isotope,
            coordinates=atom.coordinates,
        )
    for bond in molecule.bonds:
        native_mol._add_bond(
            bond.begin_atom_index,
            bond.end_atom_index,
            bond_order=bond.bond_order,
            bond_kind=bond.bond_kind,
            bond_direction=bond.bond_direction,
        )
    native_mol._update_topology()
    native_mol.charge = molecule.formal_charge
    return native_mol


def rdmol_from_dto(molecule: MoleculeDTO) -> Chem.Mol:
    """Rebuild the RDKit molecule used by conformer generation and P2."""

    if molecule.version != MCA_MOLECULE_DTO_VERSION:
        raise ValueError(
            f"Unsupported MCA molecule DTO version {molecule.version}; "
            f"expected {MCA_MOLECULE_DTO_VERSION}"
        )
    return Chem.Mol(molecule.rdkit_binary)


__all__ = [
    "MCA_MOLECULE_DTO_VERSION",
    "AtomDTO",
    "BondDTO",
    "MoleculeDTO",
    "hotpot_mol_from_dto",
    "normalize_molecule",
    "rdmol_from_dto",
    "validate_dto",
]
