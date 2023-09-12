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
import torch.nn.functional as F
import torch_geometric as pg
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import QM9

import hotpot as hp
from hotpot.tasks.ml.graph.data import MolGraph
from hotpot.tasks.ml.graph.module import get_atom_energy_tensor, MolNet, SampleGAT, CoordNet


def pre_filter(data: Data):
    return torch.all(data.x[:, 0] < 58)


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


def load_dataset(dir_struct, path_energy_sheet, ds_root, integral_coords=False):
    """"""
    mols = []

    if not Path(ds_root).joinpath("processed").exists():
        energy = pd.read_csv(path_energy_sheet, index_col=0)["energy[eV]"]

        for path_mol in Path(dir_struct).glob("*.mol2"):
            mol = hp.Molecule.read_from(path_mol)
            mol.set(energy=energy[path_mol.stem])
            mols.append(mol)

    return MolGraph(ds_root, mols, pre_filter=pre_filter)


def load_temporal_dataset(dir_log):
    samples = 0
    for path_mol in Path(dir_log).glob("**/log/*.log"):

        try:
            mol = hp.Molecule.read_from(path_mol, 'g16log')
        except StopIteration:
            continue

        if not mol:
            continue

        print(mol, mol.conformer_counts)
        samples += mol.conformer_counts

    print(samples)


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


def training():
    device = "cuda"
    gcn_layers = 6
    emb_size = 8
    integral_coords = True
    lr = 0.01
    weight_decay = 4e-5

    ds = load_dataset(
        dir_struct='/home/zz1/proj/be/calculated/gen',
        path_energy_sheet='/home/zz1/proj/be/calculated/gen.csv',
        ds_root='/home/zz1/proj/be/dataset/gen',
        integral_coords=True
    )

    ele_energy = get_atom_energy_tensor('M062X', "Def2SVP", "water")

    # model = CoordNet(emb_size, gcn_layers).to(device)
    model = MolNet(ele_energy.to(device), emb_size, gcn_layers).to(device)
    # model = SampleGAT(emb_size, gcn_layers, integral_coords=integral_coords).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    test_ds = ds[2048:]
    train_ds = ds[:2048]

    loader = DataLoader(train_ds, batch_size=512, shuffle=True)

    train_loss = []
    epochs = 3000
    for e in range(epochs):
        model.train()
        mae_train = []
        for batch in loader:
            batch = batch.to(device)

            p, Ea, Eb = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

            y = batch.y

            mae_train.append(torch.mean(abs(p - y)).detach().cpu().numpy().tolist())

            loss = F.mse_loss(p, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        mae_train = sum(mae_train) / len(mae_train)

        # Evaluation
        batch = next(iter(DataLoader(test_ds, len(test_ds))))
        batch = batch.to(device)

        p, Ea, Eb = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y = batch.y

        mae_test = torch.mean(abs(p - y)).detach().cpu().numpy().tolist()
        print(mae_train, mae_test, Eb.mean().item())

        train_loss.append([mae_train, mae_test, Eb.mean().item()])


if __name__ == "__main__":
    device = "cuda"
    gcn_layers = 6
    emb_size = 8
    integral_coords = True
    lr = 0.01
    weight_decay = 4e-5

    ds = load_dataset(
        dir_struct='/home/zz1/proj/be/calculated/gen',
        path_energy_sheet='/home/zz1/proj/be/calculated/gen.csv',
        ds_root='/home/zz1/proj/be/dataset/gen',
        integral_coords=True
    )

    ele_energy = get_atom_energy_tensor('M062X', "Def2SVP", "water")

    # model = CoordNet(emb_size, gcn_layers).to(device)
    model = MolNet(ele_energy.to(device), emb_size, gcn_layers).to(device)
    # model = SampleGAT(emb_size, gcn_layers, integral_coords=integral_coords).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    test_ds = ds[2048:]
    train_ds = ds[:2048]

    loader = DataLoader(train_ds, batch_size=512, shuffle=True)

    train_loss = []
    epochs = 3000
    for e in range(epochs):
        model.train()
        mae_train = []
        for batch in loader:
            batch = batch.to(device)

            p, Ea, Eb = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.c)

            y = batch.y

            mae_train.append(torch.mean(abs(p - y)).detach().cpu().numpy().tolist())

            loss = F.mse_loss(p, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        mae_train = sum(mae_train) / len(mae_train)

        # Evaluation
        batch = next(iter(DataLoader(test_ds, len(test_ds))))
        batch = batch.to(device)

        p, Ea, Eb = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.c)

        y = batch.y

        mae_test = torch.mean(abs(p - y)).detach().cpu().numpy().tolist()
        print(mae_train, mae_test, Eb.mean().item())

        train_loss.append([mae_train, mae_test, Eb.mean().item()])





