"""MCA input preparation, transfer construction and fresh carrier allocation."""
from __future__ import annotations
from rdkit import Chem
from hotpot.cheminfo.core import Molecule as HotpotMolecule
from hotpot.cheminfo.operations.conversion import to_hotpot_mol
from hotpot.cheminfo.structure._results import _partial_charge_projection
from hotpot.cheminfo.adapters.structure.inference import mca_molecule_record
from hotpot.cheminfo.adapters.inference.serialization import copy_feature_molecule, encode_molecule, decode_molecule
from hotpot.cheminfo.rules.chemistry.mca_sites import validate_target_domain
from .results import MCA_MOLECULE_DTO_VERSION, AtomDTO, BondDTO, MoleculeDTO


def _feature_molecule(value, native_mol: HotpotMolecule) -> Chem.Mol:
    if isinstance(value, Chem.Mol):
        return copy_feature_molecule(value, sanitize=False)
    return copy_feature_molecule(native_mol.to_rdmol(), sanitize=True)


def normalize_molecule(value) -> MoleculeDTO:
    """Normalize one public MCA input exactly once into a process-safe DTO."""

    if isinstance(value, MoleculeDTO):
        return value
    native_mol = to_hotpot_mol(value)
    with native_mol._mutation_lock:
        from hotpot.cheminfo.operations.chemistry import _require_implicit_hydrogens
        hydrogens = _require_implicit_hydrogens(native_mol)
        feature_mol = _feature_molecule(value, native_mol)
        charges = _partial_charge_projection(native_mol)
        atoms = tuple(
            AtomDTO(
                atomic_number=int(atom.atomic_number),
                formal_charge=int(atom.formal_charge),
                partial_charge=charge,
                is_aromatic=bool(atom.is_aromatic),
                implicit_hydrogens=hydrogen,
                valence=int(atom.valence),
                isotope=int(getattr(atom, "isotope", None) or 0),
                coordinates=tuple(float(coordinate) for coordinate in atom.coordinates),
            )
            for atom, hydrogen, charge in zip(native_mol.atoms, hydrogens, charges)
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
            rdkit_binary=encode_molecule(feature_mol),
        )


def validate_dto(molecule: MoleculeDTO, position: int, allow_charged: bool = False) -> None:
    """Apply the existing MCA target-domain checks to one normalized input."""
    validate_target_domain(
        tuple(atom.atomic_number for atom in molecule.atoms),
        tuple(atom.formal_charge for atom in molecule.atoms),
        molecule.formal_charge, position, allow_charged,
    )


def hotpot_mol_from_dto(molecule: MoleculeDTO) -> HotpotMolecule:
    """Rebuild the Hotpot graph used by the P1 site detector."""

    if molecule.version != MCA_MOLECULE_DTO_VERSION:
        raise ValueError(
            f"Unsupported MCA molecule DTO version {molecule.version}; "
            f"expected {MCA_MOLECULE_DTO_VERSION}"
        )
    record = mca_molecule_record(molecule.atoms, molecule.bonds, molecule.formal_charge)
    return HotpotMolecule._build_molecule(record)


def rdmol_from_dto(molecule: MoleculeDTO) -> Chem.Mol:
    """Rebuild the RDKit molecule used by conformer generation and P2."""

    if molecule.version != MCA_MOLECULE_DTO_VERSION:
        raise ValueError(
            f"Unsupported MCA molecule DTO version {molecule.version}; "
            f"expected {MCA_MOLECULE_DTO_VERSION}"
        )
    return decode_molecule(molecule.rdkit_binary)


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
