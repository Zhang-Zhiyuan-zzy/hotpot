"""
python v3.7.9
@Project: hotpot
@File   : test_chemif.py
@Author : Zhiyuan Zhang
@Date   : 2023/3/15
@Time   : 3:47
"""
import io
from pathlib import Path
from src.cheminfo import Molecule as Mol, Atom, ob
import os
import cclib
import time


def discarded():
    # Read molecule from mol2 file
    path_mol2 = 'examples/struct/mol.mol2'
    mol = Mol.readfile(path_mol2)
    print(len(mol.atoms))

    # Add a new atom
    new_atom = Atom(symbol='U')
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


def test_io():
    """   Test Input from and Output to various of format """
    dir_struct = Path('examples/struct')
    list_mol = []
    for path_file in dir_struct.glob("*"):
        mol = Mol.readfile(path_file)
        list_mol.append(mol)

    return list_mol


def dump_mol_to_gjf(mol: Mol):
    return mol.dump(
        fmt='gjf',
        link0='nproc=48',
        route='M062X/Def2SVP ADMP(MaxPoints=1000) SCRF Temperature=325'
    )


def write_mol_to_file(mol):
    dir_file_save = Path('output/mol_write').absolute()
    list_fmt = ['gjf', 'mol2']

    gjf_dict = {
        'link0': 'nproc=48',
        'route': 'M062X/Def2SVP ADMP(MaxPoints=1000) SCRF Temperature=325'
    }

    for fmt in list_fmt:
        path_file = dir_file_save.joinpath(f'mol.{fmt}')
        mol.writefile(fmt, path_file, **gjf_dict)


if __name__ == '__main__':
    mols = test_io()
    script = dump_mol_to_gjf(mols[1])
    write_mol_to_file(mols[1])

    mol = mols[1]

    mols_gen = mol.perturb_mol_lattice()

    g16root = '/home/pub'

    out, err = mol.gaussian(
        g16root,
        link0='CPU=0-48',
        route='M062X/Def2SVP SCRF',
        inplace_attrs=True
    )

    string_buffer = io.StringIO(out)
    data = cclib.ccopen(string_buffer)

    mol = mols[0]
    mol.normalize_labels()
    b = mol.bonds[20]
    gb = mol.bond(b.atom2.label, b.atom1.label)
    # mol.clean_bonds()

