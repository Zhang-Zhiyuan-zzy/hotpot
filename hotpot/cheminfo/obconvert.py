"""
python v3.9.0
@Project: hotpot
@File   : obconvert
@Auther : Zhiyuan Zhang
@Data   : 2024/12/5
@Time   : 21:28
"""
from openbabel import openbabel as ob, pybel as pb
# from core__ import Molecule, Atom

def write_by_pybel(mol, fmt='smi', filename=None, overwrite=False, opt=None):
    pmol = pb.Molecule(mol2obmol(mol)[0])
    pmol.write(fmt, filename=filename, overwrite=overwrite, opt=opt)

def obmol2mol(obmol, mol):
    # mol = Molecule()

    # Populate the idx_to_row dictionary to map OBMol atom indices to the reordered indices
    idx_to_row = {oba.GetIdx():i for i, oba in enumerate(ob.OBMolAtomIter(obmol))}

    for oba in ob.OBMolAtomIter(obmol):
        mol._create_atom(
            atomic_number=oba.GetAtomicNum(),
            formal_charge=oba.GetFormalCharge(),
            partial_charge=oba.GetPartialCharge(),
            is_aromatic=oba.IsAromatic(),
            coordinates=(oba.GetX(), oba.GetY(), oba.GetZ()),
            valence=oba.GetTotalValence(),
            implicit_hydrogens=oba.GetImplicitHCount()
        )

    # Iterate over bonds in the molecule
    for obb in ob.OBMolBondIter(obmol):
        begin_atom_idx = idx_to_row[obb.GetBeginAtomIdx()]  # Map to reordered index
        end_atom_idx = idx_to_row[obb.GetEndAtomIdx()]  # Map to reordered index
        bond_order = obb.GetBondOrder()
        # is_aromatic = obb.IsAromatic()

        mol._add_bond(begin_atom_idx, end_atom_idx, bond_order)

    mol._update_graph()
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

    return obmol, row_to_idx
