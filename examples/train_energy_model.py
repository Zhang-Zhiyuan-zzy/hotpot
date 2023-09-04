"""
python v3.9.0
@Project: hotpot
@File   : train_energy_model
@Auther : Zhiyuan Zhang
@Data   : 2023/8/22
@Time   : 20:48
"""
from pathlib import Path

import pandas as pd
from sklearn.linear_model import Ridge

from openbabel import openbabel as ob

import torch
import torch_geometric as pg
from torch_geometric.loader import DataLoader

import hotpot as hp
from hotpot.tasks.ml.graph.data import MolBundleGraph
from hotpot.tasks.ml.graph.module import get_atom_energy_tensor, MolNet


def kernel_ridge():
    data = pd.read_csv('/home/zz1/proj/be/calculated/atoms1.csv', index_col=0)

    x, y = data.iloc[:, :-1], data.iloc[:, -1]

    ridge = Ridge(alpha=0.0001)

    ridge.fit(x.iloc[:1800, :], y.iloc[:1800])

    p = ridge.predict(x)

    print(ridge.score(x.iloc[1800:, :], y.iloc[1800:]))
    print(ridge.coef_)

    return x, p, y, ridge.coef_


def direct_add(x):
    ee = get_atom_energy_tensor('M062X', "Def2SVP", "water").numpy()

    atom_nums = [ob.GetAtomicNum(s) for s in x.columns]

    energy = []
    for row in x.values:
        e = 0.0
        for j, v in enumerate(row):
            e += v * ee[ob.GetAtomicNum(x.columns[j])]

        energy.append(e)

    return energy


def load_dataset(dir_struct, path_energy_sheet, ds_root):
    """"""
    energy = pd.read_csv(path_energy_sheet, index_col=0)["energy[eV]"]

    mols = []
    for path_mol in Path(dir_struct).glob("*.mol2"):
        mol = hp.Molecule.read_from(path_mol)
        mol.set(energy=energy[path_mol.stem])
        mols.append(mol)

    return MolBundleGraph(ds_root, mols)


def pred_by_ridge():
    X, P, Y, coef = kernel_ridge()
    ele_energy = get_atom_energy_tensor('M062X', "Def2SVP", "water").numpy()

    items = ["coef", 'actual', 'diff']
    data = []
    symbols = []
    for i, sym in enumerate(X.columns):
        an = ob.GetAtomicNum(sym)

        if an < len(ele_energy):
            symbols.append(sym)
            data.append([coef[i], ele_energy[an], coef[i] - ele_energy[an]])

    coef_energy_diff = pd.DataFrame(data, index=symbols, columns=items)

    ener = pd.Series(direct_add(X), index=X.index)


if __name__ == "__main__":
    ds = load_dataset(
        dir_struct='/home/zz1/proj/be/calculated/gen',
        path_energy_sheet='/home/zz1/proj/be/calculated/gen.csv',
        ds_root='/home/zz1/proj/be/dataset/gen'
    )

    ele_energy = get_atom_energy_tensor('M062X', "Def2SVP", "water").numpy()
    model = MolNet(ele_energy)

    for batch in DataLoader(ds, batch_size=64):
        print(batch.edge_attr[batch.edge_attr > 1].shape)
        break
