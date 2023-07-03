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
    gen = mol.perturb_atoms_coordinates(mol_distance=0.05, max_generate_num=2)

    for i, gen_mol in enumerate(gen):
        gen_mol.writefile('cif', f'output/gen_cif/aCarbon_{i}.cif')


def gen_pairs():
    from openbabel import openbabel as ob
    mol = ci.Molecule.read_from('c1ccc(C(=O)O)c(O)c1CCCC', 'smi')
    mol2 = ci.Molecule.read_from('c1ccc(C(=O)O)c(O)c1CCCC', 'smi')
    mol.build_3d('UFF')
    mol2.build_3d()
    mol.normalize_labels()
    mol2.normalize_labels()
    g = mol.generate_metal_ligand_pair('Sr')

    # for a1, a2 in zip(mol.atoms, mol2.atoms):
    #     print(f'{a1.label}: {a1.partial_charge}---{a2.label}: {a2.partial_charge}')

    ps = [p for i, p in enumerate(g)]
    ps[1].remove_atoms('C1', 'C2')
    ps[1].writefile('mol2', f'/home/zz1/g01.mol2')


# TODO: debug, the following code will encounter with error, abnormal exit with 139
# TODO: check the coordinates and bond property
def have_bug():
    mol = ci.Molecule.read_from('C=CC(=O)O', 'smi')
    mol.build_3d()
    mol.normalize_labels()

    g = mol.generate_metal_ligand_pair('Sr')

    for p in g:
        p.assign_atoms_formal_charge()
        c = p.copy()  # TODO: Guess that the bonds was not assigned correctly during the copy process
        c.build_3d()


def test_valence_charge():
    """ Test whether could the molecule to assign atoms formal charge and valence correctly """
    mol = ci.Molecule.read_from('c1nc(CS(=O)(=O)O)c(OCCOP(O)(=O)OC)cc1', 'smi')

    pairs = list(mol.generate_metal_ligand_pair('Cs'))

    for i, pair in enumerate(pairs):

        print(pair.dump('gjf', link0='sldf', route='hfka'))
        print(pair.smiles, pair.charge)


if __name__ == '__main__':
    # from hotpot.utils.load_chem_lib import library as lib
    #
    # periodic_table = lib.get('PeriodicTable')
    #
    # for element in periodic_table:
    #     print(element)

    # mol = ci.Molecule.read_from('/home/zz1/proj/be/output.log', 'g16log'
    import hotpot as hp
    mol = hp.Molecule.read_from('c1ccccc1', 'smi')
    mol.gaussian('/home/zzy/sw', ['nproc=8'], 'opt')
