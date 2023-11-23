"""
python v3.9.0
@Project: hotpot
@File   : tutorial4
@Auther : Zhiyuan Zhang
@Data   : 2023/11/20
@Time   : 10:18
"""
import hotpot as hp


if __name__ == '__main__':
    mol = hp.Molecule.read_from('O=C(C1=CC=CC=C1)C1=NC2=C(C=C1)C=CC1=C2N=C(C(=O)C2=CC=CC=C2)C=C1', 'smi')
    mol.build_3d()

    coordinating_atoms = [a for a in mol.atoms if a.symbol == 'N' or (a.symbol == 'O' and len(a.neighbours) == 1)]
    Eu = mol.add_atom('Eu')

    for ca in coordinating_atoms:
        mol.add_bond(Eu, ca, 1)

    mol.localed_optimize()

    mol.writefile(
        'gjf', '../output_files/mol1.gjf',
        link0=['nproc=8', 'Mem=64GB'],
        route=['opt', 'B3YLP/6-31G']
    )
    # or get the gjf script with str format directly
    script = mol.dump('gjf', link0=['nproc=8', 'Mem=64GB'], route=['opt', 'B3LYP/LC/6-31+G(d)'])

    # mol.gaussian(link0=['nproc=8', 'Mem=64GB'],route=['opt', 'B3LYP/LC/6-31+G(d)'])
