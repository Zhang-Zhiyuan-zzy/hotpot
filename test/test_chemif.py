"""
python v3.7.9
@Project: hotpot
@File   : test_chemif.py
@Author : Zhiyuan Zhang
@Date   : 2023/3/15
@Time   : 3:47
"""
import hotpot.cheminfo as ci

def mol2_read():
    path_mol2 = 'examples/struct/mol.mol2'
    mol = ci.Molecule.read_from(path_mol2)

    return mol


def dump_gjf():
    path_mol2 = 'examples/struct/mol.mol2'
    mol = ci.Molecule.read_from(path_mol2)

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
    from openbabel import openbabel as ob
    mol = ci.Molecule.read_from('c1ccc(C(=O)O)cc1', 'smi')
    print(
        [a.GetAtomicNum() for a in ob.OBMolAtomIter(mol.ob_mol)], '\n',
        [a.GetId() for a in ob.OBMolAtomIter(mol.ob_mol)], '\n',
        mol.atoms
    )
    mol.build_conformer()
    print(mol.atoms)
    mol.remove_hydrogens()
    sr = mol.add_atom('Sr')
    mol.normalize_labels()
    # mol.add_bond(sr, 'O1', 1)
    b = mol.add_bond(sr, 'O2', 1)
    mol.build_conformer()
    mol.writefile('mol2', '/home/zz1/coor.mol2')
    print([a.valence for a in mol.atoms])
    print([a.link_degree for a in mol.atoms])
