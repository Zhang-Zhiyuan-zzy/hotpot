"""
python v3.9.0
@Project: hotpot
@File   : metal_ligand_bond_energy.py
@Author : Zhiyuan Zhang
@Date   : 2023/6/14
@Time   : 10:30

Note: This script to high-throughput determine the bind energy when a metal cation coordinate to a ligand with
specified coordination pattern
"""
import random
from pathlib import Path
import pandas as pd
from tqdm import tqdm

import hotpot as hp


def mol_gp(mol: hp.Molecule):
    """ Specify the molecule probability according to the number of atoms """
    p = 0.04 / mol.atom_num
    return random.choices([0, 1], [1-p, p])[0]


def select_mol_more_than_100atom():
    # The path to a ligands datasets
    path_to_smiles_dataset = Path('/home/zz1/proj/be/struct/ligands_smi.csv')
    path_to_selected_smiles = path_to_smiles_dataset.parent.joinpath('selected_smi.csv')

    series = pd.read_csv(path_to_smiles_dataset)

    selected_smi = []
    for s in tqdm(series.values.flatten()):
        m = hp.Molecule.read_from(s, 'smi')
        if m.atom_num < 50:
            selected_smi.append(s)

    series = pd.Series(selected_smi)
    series.to_csv(path_to_selected_smiles)


def generate_metal_ligand_pairs():
    path_to_smiles_dataset = Path('/home/zz1/proj/be/struct/ligands_smi.csv')

    series = pd.read_csv(path_to_smiles_dataset)

    bundle = hp.MolBundle.read_from('smi', series.values.flatten().tolist(), generate=True, num_proc=16)

    return bundle


def test_retrieve_mol_attrs(mole):
    print(mole.bonds)
    print(mole.angles)
    print(mole.labels)
    print(mole.inchi)
    print(mole.is_labels_unique)
    print(mole.is_organic)
    print(mole.ob_mol)
    print(mole.unique_atoms)
    print(mole.unique_all_atoms)
    print(mole.unique_bonds)
    print(mole.unique_bond_pairs)


if __name__ == '__main__':
    path_smiles = Path('/home/zzy/proj/be/struct/choice_ligand')
    g16root = '/home/zzy/sw'
    work_dir = Path('/home/zzy/proj/be/g16')

    smiles = open(path_smiles).readlines()

    mb = hp.MolBundle.read_from('smi', smiles)

    for i, mol in enumerate(mb):
        print(f'{i}/{len(mb)}: {mol}')
        pair_bundle = mol.generate_pairs_bundle('Sr', ('O', 'N'))
        pair_bundle.determine_metal_ligand_bind_energy(
            g16root, work_dir.joinpath(str(i)), 'M062X', 'Def2SVP', 'SCRF pop(Always)'
        )
