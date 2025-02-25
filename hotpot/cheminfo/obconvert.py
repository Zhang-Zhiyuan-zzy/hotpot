"""
python v3.9.0
@Project: hotpot
@File   : obconvert
@Auther : Zhiyuan Zhang
@Data   : 2024/12/5
@Time   : 21:28
"""
from typing import Any
import numpy as np
from openbabel import openbabel as ob, pybel as pb

from hotpot.utils.chem import atom as chem_atom
from hotpot.cheminfo.elements import elements

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
    return {
        (i2r[obb.GetBeginAtomIdx()], i2r[obb.GetEndAtomIdx()]): {"bond_order": obb.GetBondOrder()}
        for obb in ob.OBMolBondIter(obmol)
    }


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
    # Populate the idx_to_row dictionary to map OBMol atom indices to the reordered indices
    idx_to_row = {oba.GetIdx():i for i, oba in enumerate(ob.OBMolAtomIter(obmol))}

    for oba in ob.OBMolAtomIter(obmol):
        n, s, p, d, f, g = elements.electron_configs[oba.GetAtomicNum()]
        mol._create_atom_from_array(
            attrs_array=np.array([
                oba.GetAtomicNum(),
                n, s, p, d, f, g,  # electron configure
                oba.GetFormalCharge(),
                oba.GetPartialCharge(),
                float(oba.IsAromatic()),
                oba.GetX(), oba.GetY(), oba.GetZ(),
                oba.GetTotalValence(),
                oba.GetImplicitHCount(),
                0,
                0, 0, 0,
                ], dtype=np.float64)
            )

    _add_mol_bonds_from_obmol(mol, obmol, idx_to_row)
    #
    mol._update_graph()
    # mol.calc_atom_valence()

    # add Crystal
    cell_index = ob.UnitCell  # Get the index the UnitCell data save
    cell_data = obmol.GetData(cell_index)
    if cell_data:
        c = ob.toUnitCell(cell_data)
        mol.create_crystal(c.GetA(), c.GetB(), c.GetC(), c.GetAlpha(), c.GetBeta(), c.GetGamma())

    return mol


def mol2obmol(mol):
    obmol = ob.OBMol()

    row_to_idx = {}
    for i, atom in enumerate(mol.atoms):
        oba = obmol.NewAtom()
        oba.SetAtomicNum(atom.atomic_number)
        oba.SetFormalCharge(atom.formal_charge)
        oba.SetPartialCharge(atom.partial_charge)
        oba.SetVector(*atom.coordinates)
        # oba.IsAromatic()
        oba.SetAromatic(atom.is_aromatic)  # Convert to bool
        oba.IsAromatic()

        # Store mapping from label to atom index (1-based indexing in OBMol)
        row_to_idx[i] = oba.GetIdx()

    for bond in mol.bonds:
        begin_atom_idx, end_atom_idx, bond_order = bond.atom1.idx, bond.atom2.idx, bond.bond_order
        obmol.AddBond(
            row_to_idx[begin_atom_idx],
            row_to_idx[end_atom_idx],
            int(bond_order)
        )
        obb = obmol.GetBond(row_to_idx[begin_atom_idx], row_to_idx[end_atom_idx])
        obb.IsAromatic()
        obb.SetAromatic(bool(bond.is_aromatic))  # Convert to bool
        obb.IsAromatic()

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
        mol.bond(a1idx, a2idx).bond_order = attrs['bond_order']


def add_hydrogens(mol):
    obmol, row_to_idx = mol2obmol(mol)
    obmol.AddHydrogens(False, False, 1.0)

    modified = False
    for a in mol.atoms:
        if not (a.is_hydrogen or a.is_metal):
            oba = obmol.GetAtom(row_to_idx[a.idx])
            assert oba.GetAtomicNum() == a.atomic_number
            assert (oba.GetX(), oba.GetY(), oba.GetZ()) == a.coordinates

            neigh_hydrogens = [na for na in a.neighbours if na.is_hydrogen]
            a_hnum = len(neigh_hydrogens)

            oba_hnum = len([neigh_oba for neigh_oba in ob.OBAtomAtomIter(oba) if neigh_oba.GetAtomicNum() == 1])
            delta = oba_hnum - a_hnum

            if delta > 0:
                for i in range(delta):
                    a._add_atom()

                modified = True

            elif delta < 0:
                mol._rm_atoms(neigh_hydrogens[:abs(delta)])

                modified = True

        return modified


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

