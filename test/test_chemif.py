"""
python v3.7.9
@Project: hotpot
@File   : test_chemif.py
@Author : Zhiyuan Zhang
@Date   : 2023/3/15
@Time   : 3:47
"""
import src.cheminfo as ci
from src._io import Parser


def mol_io():
    """ Test the whether IO classes and io function work """
    pass


def mol2_read():
    path_mol2 = 'examples/struct/mol.mol2'
    mol = ci.Molecule.readfile(path_mol2)

    return mol


def dump_gjf():
    path_mol2 = 'examples/struct/mol.mol2'
    mol = ci.Molecule.readfile(path_mol2)

    script = mol.dump('gjf',
                      link0='CPU=0-48',
                      route='M062X/Def2SVP opt(MaxCyc=10)/freq SCRF pop(Always)',
                      )

    print(script)


def parse_g16log():
    path_mol1 = 'examples/struct/0.log'
    path_mol2 = 'examples/struct/222.log'
    path_mol3 = 'examples/struct/Cs-VOHSAM-5.mol2'

    mol1 = ci.Molecule.read_from(path_mol1, 'g16log')
    mol2 = ci.Molecule.read_from(path_mol2, 'g16log')
    mol3 = ci.Molecule.read_from(path_mol3)

    return mol1, mol2, mol3



def perturb_cif():
    path_cif = 'examples/struct/aCarbon.cif'
    mol = ci.Molecule.read_from(path_cif, 'cif')
    gen = mol.perturb_mol_lattice(mol_distance=0.05, max_generate_num=2)

    for i, gen_mol in enumerate(gen):
        gen_mol.writefile('cif', f'output/gen_cif/aCarbon_{i}.cif')


if __name__ == '__main__':
    m1, m2, m3 = parse_g16log()
    # data = mol.dump('dpmd_sys', path_save='output/dpmd_sys')
    m = m1 + m3
    # for a in mol.atoms:
    #     print(a.partial_charge, a.spin_density, a.force_vector.shape)
    # perturb_cif()
