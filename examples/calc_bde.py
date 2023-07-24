"""
python v3.9.0
@Project: hotpot
@File   : calc_bde.py
@Author : Zhiyuan Zhang
@Date   : 2023/6/14
@Time   : 10:30

Note:
    This script to high-throughput determine the bind energy when a metal cation coordinate to a ligand with
    specified coordination pattern
"""
import os
from pathlib import Path
import hotpot as hp


if __name__ == '__main__':

    START_NUM = 8

    path_smiles = Path('/home/zz1/proj/be/struct/choice_ligand')
    g16root = '/home/pub'
    work_dir = Path('/home/zz1/proj/be/g161')
    os.chdir(work_dir)

    smiles = open(path_smiles).readlines()

    for i, s in enumerate(smiles[START_NUM:], START_NUM):

        mol = hp.Molecule.read_from(s, 'smi')
        pair_bundle = mol.generate_pairs_bundle('Sr')

        if len(pair_bundle) == 0:
            continue

        pair_bundle.determine_metal_ligand_bind_energy(
            g16root, work_dir.joinpath(str(i)), 'M062X', 'Def2SVP', 'SCRF pop(Always)', cpu_uti=0.75,
            skip_complete=True
        )
