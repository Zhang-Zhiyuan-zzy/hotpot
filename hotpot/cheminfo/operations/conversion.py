"""Carrier preparation, factual import, and fresh toolkit conversion operations."""

from dataclasses import replace
import numpy as np

from ..adapters import conversion as _input
from ..adapters.openbabel import conversion as _openbabel
from ..adapters.rdkit import conversion as _rdkit
from ..adapters.rdkit import stereo_transport as _stereo
from ..structure._commit import _FactTransaction
from ..structure._fields import ATOM_SOURCE_CODES
from ..structure._stereo import _get_stereo_payload
from . import chemistry as _chemistry

__all__ = ["is_molecule_input", "to_hotpot_mol", "mol2obmol", "obmol2mol", "to_rdmol", "from_rdmol", "write_by_pybel"]


def _molecule_type():
    from ..core import Molecule
    return Molecule


def is_molecule_input(value) -> "bool":
    """Return whether ``value`` is one supported single-molecule input."""
    return _input._classify_input(value, _molecule_type()) != "unsupported"


def to_hotpot_mol(value, *, fmt=None, **kwargs) -> "Molecule":
    """Return ``value`` as a Hotpot molecule, preserving supported dispatch order."""
    molecule_type = _molecule_type()
    kind = _input._classify_input(value, molecule_type)
    if kind == "hotpot":
        return value
    if kind == "rdkit":
        return from_rdmol(value, molecule_type())
    if kind == "openbabel":
        return obmol2mol(value, molecule_type())
    if kind == "pybel":
        return obmol2mol(value.OBMol, molecule_type())
    if kind == "source":
        from .io import read_mol
        return read_mol(value, fmt=fmt, **kwargs)
    if kind == "protocol":
        rdmol = value.to_rdmol()
        if not _input._is_rdmol(rdmol):
            raise TypeError("to_rdmol() must return an RDKit Chem.Mol")
        return from_rdmol(rdmol, molecule_type())
    raise TypeError(f"unsupported molecule type: {type(value)!r}")


def _capture_ob_values(mol, bonds):
    table = mol._atom_table
    topology = mol._topology_index
    atom_rows = tuple((table.read(row, 'atomic_number'), table.read(row, 'formal_charge'),
                       table.read(row, 'partial_charge'), tuple(atom.coordinates),
                       table.raw(row, 'is_aromatic'))
                      for row, atom in enumerate(mol.atoms))
    bond_rows = []
    for bond in bonds:
        kind = bond._bond_kind.value
        if kind == "dative":
            bond_rows.append((kind,))
            return atom_rows, tuple(bond_rows), None
        first, second = bond._atoms
        bond_rows.append((kind, topology[id(first)][0], topology[id(second)][0],
                          float(bond._attrs[0]), bond._is_aromatic))
    cell = None
    if mol.crystal:
        crystal = mol.crystal
        cell = (crystal.a, crystal.b, crystal.c, crystal.alpha, crystal.beta, crystal.gamma)
        from ..structure.crystal import Crystal
        cell = Crystal(*cell)
    return atom_rows, tuple(bond_rows), cell


def mol2obmol(mol):
    with mol._mutation_lock:
        _chemistry._require_aromatic_representation(mol)
        bonds = tuple(mol.bonds)
        obmol = _openbabel._new_obmol()
        atom_rows, bond_rows, cell = _capture_ob_values(mol, bonds)
        return _openbabel._factual_obmol(obmol, atom_rows, bond_rows, cell)


def obmol2mol(obmol, mol):
    with _FactTransaction(mol, producer=ATOM_SOURCE_CODES["openbabel"]):
        if mol._atoms or mol._bonds or mol._hided_metal_bonds or mol._hided_covalent_bonds:
            raise ValueError("Factual import requires an empty Atom and complete-Bond target")
        initial = mol._capture_factual_record()[0]
        record = _openbabel._obmol_record(obmol, initial, mol._stereo_payload)
        return mol._install_molecule(record)


def _capture_rd_bonds(bonds):
    rows = []
    for bond in bonds:
        first, second = bond.atom1.idx, bond.atom2.idx
        kind = bond.bond_kind.value
        direction = None
        raw_type_name = None
        aromatic = None
        order = None
        if kind == "dative":
            direction = bond.bond_direction
            if direction not in ("atom1_to_atom2", "atom2_to_atom1"):
                rows.append((first, second, kind, direction))
                break
            raw_type_name = bond.bond_source_metadata.get("raw_bond_type")
        elif kind != "aromatic":
            aromatic = bond.is_aromatic
            if not aromatic and kind not in ("zero", "single", "double", "triple"):
                order = bond.bond_order
        rows.append((first, second, kind, direction, raw_type_name, aromatic, order))
    return tuple(rows)


def to_rdmol(mol, kekulize: bool = True, sanitize: bool = False):
    with mol._mutation_lock:
        hydrogens = _chemistry._require_implicit_hydrogens(mol)
        _chemistry._require_aromatic_representation(mol)
        return _to_rdmol_prepared(mol, kekulize, sanitize, hydrogens)


def _to_rdmol_prepared(mol, kekulize, sanitize, hydrogens):
    rdmol = _rdkit._new_rdmol()
    payload = _get_stereo_payload(mol)
    atoms = mol.atoms
    bonds = mol.bonds
    _stereo._validate_payload_references(payload, len(atoms),
        tuple((bond.atom1.idx, bond.atom2.idx) for bond in bonds))
    atom_facts = _stereo._payload_atom_facts(payload, len(atoms))
    atom_rows = tuple((atom.atomic_number, atom.formal_charge, atom.partial_charge,
                       atom.is_aromatic, atom.hyb) for atom in atoms)
    coordinates = tuple(tuple(row) for row in mol.coordinates)
    bond_rows = _capture_rd_bonds(bonds)
    rdmol, row_to_idx = _rdkit._pack_rdmol(rdmol, atom_rows, bond_rows, coordinates,
        hydrogens, atom_facts, kekulize, sanitize)
    export_payload = _chemistry._remap_stereo_transport(payload, row_to_idx)
    _stereo._apply_stereo_payload(rdmol, export_payload)
    _stereo._verify_current_implicit_hydrogens(rdmol,
        _stereo._payload_atom_facts(export_payload, len(atoms)))
    return _rdkit._finish_rdmol(rdmol)


def from_rdmol(rdmol, mol):
    """Populate the supplied empty target from an RDKit molecule."""
    with _FactTransaction(mol, producer=ATOM_SOURCE_CODES["rdkit"]):
        if mol._atoms or mol._bonds or mol._hided_metal_bonds or mol._hided_covalent_bonds:
            raise ValueError("Factual import requires an empty Atom and complete-Bond target")
        initial = mol._capture_factual_record()[0]
        record, payload, idx_to_row = _rdkit._rdmol_record(rdmol, initial)
        record = replace(record, stereo_payload=_chemistry._remap_stereo_transport(payload, idx_to_row))
        return mol._install_molecule(record)


def write_by_pybel(mol, fmt='smi', filename=None, overwrite=False, opt=None):
    _openbabel._write_by_pybel(mol2obmol(mol)[0], fmt, filename, overwrite, opt)


def _to_pybel_mol(mol):
    return _openbabel._to_pybel(mol2obmol(mol)[0])


def _inchi(mol):
    return _openbabel._inchi(mol.to_obmol())


def _smiles(mol, kekulize=False):
    return _openbabel._smiles(mol.to_obmol(), kekulize)


def _get_partial_charge(mol, model, model_args, known_models):
    ob_charge_model = _openbabel._charge_model(model)
    if ob_charge_model is None:
        raise ValueError(
            f"OpenBabel charge model {model!r} is unavailable; "
            f"known models are {', '.join(known_models)}"
        )
    projected = mol.to_obmol()
    _copied, succeeded = _openbabel._compute_charges(ob_charge_model, projected, model_args)
    if not succeeded:
        raise RuntimeError(f"OpenBabel charge model {model!r} could not calculate charges")
    charges = _openbabel._partial_charges(ob_charge_model)
    if len(charges) != len(mol.atoms):
        raise RuntimeError(
            f"OpenBabel charge model {model!r} returned {len(charges)} "
            f"charges for {len(mol.atoms)} atoms"
        )
    if not np.all(np.isfinite(charges)):
        raise RuntimeError(f"OpenBabel charge model {model!r} returned non-finite charges")
    return charges


def _assign_partial_charge(mol, model, model_args):
    charges = mol.get_partial_charge(model=model, model_args=model_args)
    for atom, charge in zip(mol.atoms, charges):
        atom.partial_charge = charge
    return charges
