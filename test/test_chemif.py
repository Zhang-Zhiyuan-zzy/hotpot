"""
python v3.7.9
@Project: hotpot
@File   : test_chemif.py
@Author : Zhiyuan Zhang
@Date   : 2023/3/15
@Time   : 3:47
"""
import src.cheminfo as ci


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


if __name__ == '__main__':
    dump_gjf()
