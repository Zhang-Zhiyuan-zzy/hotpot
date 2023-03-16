"""
python v3.7.9
@Project: hotpot
@File   : test_chemif.py
@Author : Zhiyuan Zhang
@Date   : 2023/3/15
@Time   : 3:47
"""
import src.cheminfo as ci
from openbabel import pybel as pb


if __name__ == '__main__':

    # Read molecule from mol2 file
    path_mol2 = 'examples/struct/mol.mol2'
    mol = ci.Molecule.readfile(path_mol2)
    print(len(mol.atoms))

    # Add a new atom
    new_atom = ci.Atom(symbol='U')
    mol.add_atom(new_atom)
    print(len(mol.atoms))
    print(mol.atoms[-1].symbol, mol.atoms[-1]._OBAtom.GetIdx())

    # Normalise its labels
    print(mol.atoms)
    mol.normalize_labels()
    print(mol.atoms)

    # get the molecular coordinate matrix
    print(mol.coord_matrix[10])

    clone = mol.copy()

    mol_generator = mol.perturb_mol_lattice()
    new_mol = next(mol_generator)
    print(new_mol.coord_matrix[10])


