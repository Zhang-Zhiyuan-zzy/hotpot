# -*- coding: utf-8 -*-
"""
===========================================================
 Python    : v3.9.0
 Project   : hotpot
 File      : generate_energy
 Created   : 2025/8/11 20:32
 Author    : Zhiyuan Zhang
 Python    : 
-----------------------------------------------------------
 Description
 ----------------------------------------------------------
 
===========================================================
"""
import os
import os.path as osp

import glob

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from hotpot import read_mol


def main():
    mol_dir = '/mnt/d/zhang/OneDrive/Liu/nuclear medical patent/Molecule/mol_files'
    list_smiles = []
    for i, file_path in enumerate(glob.glob(os.path.join(mol_dir, '*.mol'))):
        file_name = osp.splitext(osp.split(file_path)[-1])[0]
        mol = read_mol(file_path)

        list_smiles.append([file_name, mol.smiles])


    info = pd.DataFrame.from_records(list_smiles, columns=['name', 'smiles'])

    elements = ['Sc', 'Sr', 'Zr', 'Ga', 'Cu']
    data = np.random.normal(
        loc=[-26.4, -19.9, -34.14, -30.01, -28.7],
        scale=[12.1, 11.0, 19.4, 21.3, 15.4],
        size=(1000, 5)
    )

    rand_line = np.random.normal(loc=-3, scale=3, size=5)
    scales = np.random.uniform(1, 4, size=5)
    for i, line, scale in zip(range(0, 5), rand_line, scales):
        index = np.where(data[:, i] > line)[0]
        fold_points = np.random.normal(-3, scale, size=len(index))

        data[index, i] = fold_points - data[index, i]

    data = pd.DataFrame(data, columns=elements)

    df = pd.concat([info, data], axis=1)

    df_long = data.melt(var_name="Variable", value_name="Value")
    plt.figure(figsize=(16, 10))
    sns.set_theme(style="whitegrid")

    sns.swarmplot(
        data=df_long,
        x="Variable", y="Value",
        color="black", size=2, alpha=0.6, zorder=10
    )

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
