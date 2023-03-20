"""
python v3.7.9
@Project: hotpot
@File   : test_chemif.py
@Author : Zhiyuan Zhang
@Date   : 2023/3/15
@Time   : 3:47
"""
from pathlib import Path
from src.cheminfo import Molecule as Mol, Atom
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

    mol = mols[0]

    mols_gen = mol.perturb_mol_lattice()

    path_gen_mol2 = Path('D:/hotpot/test/output/gen_mol2')

    t1 = time.time()

    for a in mol.atoms:
        a.neighbours

    t2 = time.time()

    print(t2 - t1)

    # for i, mol in enumerate(mols_gen):
    #     mol.writefile('mol2', path_gen_mol2.joinpath(f"{i}.mol2"))
    # result = mol.gaussian(
    #     link0='nproc=32',
    #     route='M062X/Def2SVP ADMP(MaxPoints=10) SCRF Temperature=325',
    #     g16root='/home/pub',
    #     gauss_scrdir='/home/zzy/M062X/g16_scrat'
    # )
