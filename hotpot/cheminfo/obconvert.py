"""
python v3.9.0
@Project: hotpot
@File   : obconvert
@Auther : Zhiyuan Zhang
@Data   : 2024/12/5
@Time   : 21:28
"""
from dataclasses import replace
from typing import Any
import numpy as np
from openbabel import openbabel as ob, pybel as pb

from hotpot.utils.chem import atom as chem_atom
from hotpot.cheminfo.elements import elements
from .structure._builder import _freeze_value
from .structure._commit import _FactTransaction
from .structure._fields import ATOM_SOURCE_CODES, FIELD_INDEX, FIELD_PRESENCE_BITS

def write_by_pybel(mol, fmt='smi', filename=None, overwrite=False, opt=None):
    pmol = pb.Molecule(mol2obmol(mol)[0])
    pmol.write(fmt, filename=filename, overwrite=overwrite, opt=opt)


def get_ob_conversion(fmt='smi', **kwargs):
    conv = ob.OBConversion()
    conv.SetOutFormat(fmt)

    for k, v in kwargs.items():
        if v is None:
            conv.AddOption(k, conv.OUTOPTIONS)
        else:
            conv.AddOption(k, conv.OUTOPTIONS, str(v))

    return conv


def write_obmol_to_string(obmol: ob.OBMol, fmt='smi', **kwargs):
    return get_ob_conversion(fmt, **kwargs).WriteString(obmol)


def _retrieve_bonds_attrs_from_obmol(obmol: ob.OBMol, i2r: dict[int, int]) -> dict[tuple[int, int], dict[str, Any]]:
    bond_attrs = {}
    for obb in ob.OBMolBondIter(obmol):
        bond_order = obb.GetBondOrder()
        if bond_order == 0:
            bond_kind = "unknown"
        elif obb.IsAromatic():
            bond_kind = "aromatic"
        else:
            bond_kind = {
                1: "single",
                2: "double",
                3: "triple",
            }.get(bond_order, "unknown")

        bond_attrs[(i2r[obb.GetBeginAtomIdx()], i2r[obb.GetEndAtomIdx()])] = {
            "bond_order": bond_order,
            "bond_kind": bond_kind,
            "bond_direction": None,
            "bond_source": "openbabel",
            "bond_source_metadata": {
                "raw_bond_order": bond_order,
                "raw_flags": int(obb.GetFlags()),
                "is_aromatic": bool(obb.IsAromatic()),
                "is_amide": bool(obb.IsAmide()),
                "is_in_ring": bool(obb.IsInRing()),
                "is_wedge": bool(obb.IsWedge()),
                "is_hash": bool(obb.IsHash()),
            },
        }
    return bond_attrs


def _add_mol_bonds_from_obmol(mol, obmol, idx_to_row):
    # for obb in ob.OBMolBondIter(obmol):
    #     begin_atom_idx = idx_to_row[obb.GetBeginAtomIdx()]  # Map to reordered index
    #     end_atom_idx = idx_to_row[obb.GetEndAtomIdx()]  # Map to reordered index
    #     bond_order = obb.GetBondOrder()
    #     # is_aromatic = obb.IsAromatic()
    _bond_attrs: dict[tuple[int, int], dict[str, Any]] = _retrieve_bonds_attrs_from_obmol(obmol, idx_to_row)

    for (a1idx, a2idx), attrs in _bond_attrs.items():
        mol._add_bond(a1idx, a2idx, **attrs)


def obmol2mol(obmol, mol):
    with _FactTransaction(mol, producer=ATOM_SOURCE_CODES["openbabel"]):
        if mol._atoms or mol._bonds or mol._hided_metal_bonds or mol._hided_covalent_bonds:
            raise ValueError("Factual import requires an empty Atom and complete-Bond target")
        initial = mol._capture_factual_record()[0]
        atoms = tuple(ob.OBMolAtomIter(obmol))
        idx_to_row = {atom.GetIdx(): row for row, atom in enumerate(atoms)}
        values = np.zeros((len(atoms), 19), dtype="<f8")
        for row, atom in enumerate(atoms):
            values[row] = (
                atom.GetAtomicNum(), *elements.electron_configs[atom.GetAtomicNum()],
                atom.GetFormalCharge(), atom.GetPartialCharge(), float(atom.IsAromatic()),
                atom.GetX(), atom.GetY(), atom.GetZ(), atom.GetTotalValence(),
                atom.GetImplicitHCount(), 0, 0, 0, 0,
            )
        supplied = tuple(range(15))
        presence = np.full(len(atoms), sum(FIELD_PRESENCE_BITS[index] for index in supplied), dtype="<u4")
        sources = np.zeros((len(atoms), 19), dtype="u1")
        sources[:, supplied] = ATOM_SOURCE_CODES["openbabel"]
        sources[:, 1:7] = ATOM_SOURCE_CODES["electron_configuration"]
        entries = tuple(_retrieve_bonds_attrs_from_obmol(obmol, idx_to_row).items())
        raw_bonds = np.zeros((len(entries), 3), dtype="<f8")
        raw_bonds[:, 0] = [entry[1]["bond_order"] for entry in entries]
        cell_data = obmol.GetData(ob.UnitCell)
        cell = initial.cell
        if cell_data:
            unit_cell = ob.toUnitCell(cell_data)
            cell = tuple(getattr(unit_cell, name)() for name in (
                "GetA", "GetB", "GetC", "GetAlpha", "GetBeta", "GetGamma"))
        record = replace(
            initial, atom_count=len(atoms), atom_values=values.tobytes(),
            atom_presence=presence.tobytes(), atom_sources=sources.tobytes(),
            bond_endpoints=tuple(endpoints for endpoints, _ in entries),
            bond_values=raw_bonds.tobytes(), bond_partitions=(0,) * len(entries),
            bond_kinds=tuple(attrs["bond_kind"] for _, attrs in entries),
            bond_directions=tuple(attrs["bond_direction"] for _, attrs in entries),
            bond_sources=("openbabel",) * len(entries),
            bond_metadata=tuple(_freeze_value(attrs["bond_source_metadata"]) for _, attrs in entries),
            charge=(True, obmol.GetTotalCharge(), "openbabel"), cell=cell,
            stereo_payload=mol._stereo_payload,
            legacy_frame_values=(("none",),) * 9,
        )
        return mol._install_molecule(record)


def mol2obmol(mol):
    obmol = ob.OBMol()
    obmol.BeginModify()

    row_to_idx = {}
    for i, atom in enumerate(mol.atoms):
        oba = obmol.NewAtom()
        oba.SetAtomicNum(atom.atomic_number)
        oba.SetFormalCharge(atom.formal_charge)
        oba.SetPartialCharge(atom.partial_charge)
        oba.SetVector(*atom.coordinates)

        # Store mapping from label to atom index (1-based indexing in OBMol)
        row_to_idx[i] = oba.GetIdx()

    for bond in mol.bonds:
        if bond.bond_kind.value == "dative":
            raise ValueError(
                "Open Babel cannot represent dative bonds losslessly"
            )
        begin_atom_idx, end_atom_idx, bond_order = bond.atom1.idx, bond.atom2.idx, bond.bond_order
        obmol.AddBond(
            row_to_idx[begin_atom_idx],
            row_to_idx[end_atom_idx],
            int(bond_order)
        )

    obmol.EndModify()
    for row, atom in enumerate(mol.atoms):
        obmol.GetAtom(row_to_idx[row]).SetAromatic(atom.is_aromatic)
    for bond in mol.bonds:
        obb = obmol.GetBond(
            row_to_idx[bond.atom1.idx],
            row_to_idx[bond.atom2.idx],
        )
        obb.SetAromatic(
            bond.bond_kind.value == "aromatic" or bond.is_aromatic
        )
    obmol.SetAromaticPerceived(True)

    # Add UnitCell
    if mol.crystal:
        obmol.CloneData(mol.crystal.obcell)

    return obmol, row_to_idx


def link_atoms(mol):
    obmol, row_to_idx = mol2obmol(mol)
    idx_to_row = {i: r for r, i in row_to_idx.items()}

    obmol.ConnectTheDots()

    _add_mol_bonds_from_obmol(mol, obmol, idx_to_row)


def assign_bond_order(mol):
    obmol, row_to_idx = mol2obmol(mol)
    idx_to_row = {i: r for r, i in row_to_idx.items()}

    obmol.PerceiveBondOrders()
    _bond_attrs: dict[tuple[int, int], dict[str, Any]] = _retrieve_bonds_attrs_from_obmol(obmol, idx_to_row)
    for (a1idx, a2idx), attrs in _bond_attrs.items():
        bond = mol.bond(a1idx, a2idx)
        bond.bond_order = attrs['bond_order']
        bond._set_bond_metadata(**{
            key: value for key, value in attrs.items()
            if key != 'bond_order'
        })


def extract_obatom_coordinate(obatom: ob.OBMol):
    return np.array((obatom.GetX(), obatom.GetY(), obatom.GetZ()))


def extract_obmol_coordinates(obmol: ob.OBMol) -> np.ndarray:
    return np.array([(oba.GetX(), oba.GetY(), oba.GetZ()) for oba in ob.OBMolAtomIter(obmol)])


def set_obmol_coordinates(obmol: ob.OBMol, coords):
    coords = np.array(coords)
    assert coords.ndim == 2
    assert coords.shape[1] == 3
    assert obmol.NumAtoms() == coords.shape[0]

    for oba, coord in zip(ob.OBMolAtomIter(obmol), coords):
        oba.SetVector(*coord)


def to_arrays(obmol):
    # Populate the idx_to_row dictionary to map OBMol atom indices to the reordered indices
    idx_to_row = {oba.GetIdx():i for i, oba in enumerate(ob.OBMolAtomIter(obmol))}

    # Iterate over atoms in the molecule
    atoms_array = np.array([
        [
            atom.GetAtomicNum(),
            atom.GetFormalCharge(),
            atom.GetPartialCharge(),
            float(atom.IsAromatic()),
            atom.GetX(),
            atom.GetY(),
            atom.GetZ(),
            atom.GetTotalValence(),
            atom.GetImplicitHCount()
        ]
        for atom in ob.OBMolAtomIter(obmol)
    ])

    bonds_array = np.array([
        [
            idx_to_row[bond.GetBeginAtomIdx()],
            idx_to_row[bond.GetEndAtomIdx()],
            bond.GetBondOrder()
        ]
        for bond in ob.OBMolBondIter(obmol)
    ])

    return atoms_array, bonds_array, idx_to_row
