import os.path as osp
from glob import glob
import socket
from typing import Callable
from tqdm import tqdm

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import hotpot as hp
from hotpot.plugins.PyG import data as pyg_data
from hotpot.plugins.ccdc_api import statistics as cc_stat
from hotpot.plugins.xtb import xtb_batch_run

import torch
from torch_geometric.loader import DataLoader

machine_name = socket.gethostname()
if machine_name == '4090':
    project_root = '/home/zzy/proj/bayes'
elif machine_name == 'DESKTOP-G9D9UUB':
    project_root = '/mnt/d/zhang/OneDrive/Papers/BayesDesign/results'
else:
    raise ValueError


ccdc_files_dir = osp.join(project_root, 'ccdc')
ccdc_all_file_dir = osp.join(ccdc_files_dir, 'all_mol2')
ccdc_mono_file_dir = osp.join(ccdc_files_dir, 'mono')
ccdc_xtb_file_dir = osp.join(ccdc_files_dir, 'xtb')


def statistic_node_type_weight():
    data_dir = osp.join(project_root, 'datasets', 'tmqm_data0207')
    dataset = pyg_data.tmQmDataset(data_dir)
    loader = DataLoader(dataset, batch_size=1024)

    list_node_type = []
    for batch in tqdm(loader):
        list_node_type.append(batch.x[:, 0].flatten())

    return list_node_type

def statistic_ccdc_complexes():
    for file in glob(osp.join(ccdc_files_dir, 'all', '*.mol2')):
        mol = next(hp.MolReader(file))


def extract_pyg_data_from_tmQm():
    data_dir = osp.join(project_root, 'datasets', 'tmqm_data_')
    dataset = pyg_data.tmQmDataset(data_dir)

    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    for batch in loader:
        print(batch)


def extract_pyg_data_from_ccdc_mono():
    dir_row_mol2 = osp.join(project_root, 'ccdc', 'mono')
    data_dir = osp.join(project_root, 'ccdc', 'pygData', 'mono1')

    ds = pyg_data.ComplexDataset(data_dir, dir_row_mol2)

    loader = DataLoader(ds, batch_size=2, shuffle=True)

    for batch in loader:
        print(batch)


def ccdc_statistics():
    dir_stat = osp.join(project_root, 'ccdc', 'stat')
    statistic = cc_stat.ComplexStatistics(ccdc_files_dir)
    statistic.statistic()

    with pd.ExcelWriter(osp.join(dir_stat, 'atom_counts.xlsx')) as writer:
        total = {c: len(v) for c, v in  statistic.atom_counts.items()}
        total = pd.Series(total)
        total.to_excel(writer, sheet_name='total')
        fig, ax = plt.subplots()
        ax.bar(total.index.values, total.values)
        fig.savefig(osp.join(dir_stat, 'picture', 'atom_counts.png'))

        for c, v in total.items():
            idt = pd.Series(v)
            idt.to_excel(writer, sheet_name=str(c))

    with pd.ExcelWriter(osp.join(dir_stat, 'metal_counts.xlsx')) as writer:
        total = {c: len(v) for c, v in statistic.metal_counts.items()}
        total = pd.Series(total)
        total.to_excel(writer, sheet_name='total')
        fig, ax = plt.subplots()
        ax.bar(total.index.values, total.values)
        fig.savefig(osp.join(dir_stat, 'picture', 'metal_counts.png'))

        for c, v in total.items():
            idt = pd.Series(v)
            idt.to_excel(writer, sheet_name=str(c))

    fig, ax = plt.subplots()
    ax.hist(statistic.mol_weights, bins=40)
    fig.savefig(osp.join(dir_stat, 'picture', 'mol_weights.png'))

    with pd.ExcelWriter(osp.join(dir_stat, 'metal_types.xlsx')) as writer:
        total = {c: len(v) for c, v in statistic.metal_types.items()}
        total = pd.Series(total)
        total.to_excel(writer, sheet_name='total')


def _filter_mol(mol):
    if not (10 < len(mol.atoms) < 150):
        return False
    if any(a.atomic_number > 86 for a in mol.atoms):
        return False

    return True


def export_xtb_files():
    print('Export XTB files')
    xtb_batch_run(
        mol_file_dir=ccdc_all_file_dir,
        res_file_dir=ccdc_xtb_file_dir,
        perform=False,
        charge=0
    )


def draw_periodic_table():
    pos = {
        1: {1:'H', 18:'H'},
        2: {1:'Li',2:'Be',13:'B',14:'C',15:'N',16:'O',17:'F',18:'Ne'},
        3: {1:'Na',2:'Mg',13:'Al',14:'Si',15:'P',16:'S',17:'Cl',18:'Ar'},
        4: {1:'K',2:'Ca',3:'Sc',4:'Ti',5:'V',6:'Cr',7:'Mn'}
    }

    pos = {
        'H': (1, 1), 'He': (1, 18),
        'Li': (2, 1), 'Be': (2, 2), 'B': (2, 13), 'C': (2, 14), 'N': (2, 15), 'O': (2, 16), 'F': (2, 17), 'Ne': (2, 18),
        'Na': (3, 1), 'Mg': (3, 2), 'Al': (3, 13), 'Si': (3, 14), 'P': (3, 15), 'S': (3, 16), 'Cl': (3, 17), 'Ar': (3, 18),
        'K': (4, 1), 'Ca': (4, 2), 'Sc': (4, 3), 'Ti': (4, 4), 'V': (4, 5), 'Cr': (4, 6), 'Mn': (4, 7), 'Fe': (4, 8), 'Co': (4, 9), 'Ni': (4, 10), 'Cu': (4, 11), 'Zn': (4, 12), 'Ga': (4, 13), 'Ge': (4, 14), 'As': (4, 15), 'Se': (4, 16), 'Br': (4, 17), 'Kr': (4, 18), 'Pd': (5, 10), 'Ag': (5, 11), 'Cd': (5, 12), 'In': (5, 13), 'Sn': (5, 14), 'Sb': (5, 15), 'Te': (5, 16), 'I': (5, 17), 'Xe': (5, 18),
        'Cs': (6, 1), 'Ba': (6, 2), 'Ln': (7, 3),
        'Fr': (7, 1), 'Ra': (7, 2), 'An': (7, 3), 'Rf': (7, 4), 'Db': (7, 5), 'Sg': (7, 6), 'Bh': (7, 7), 'Hs': (7, 8), 'Mt': (7, 9), 'Ds': (7, 10), 'Rg': (7, 11), 'Cn': (7, 12), 'Nh': (7, 13), 'Fl': (7, 14), 'Mc': (7, 15), 'Lv': (7, 16), 'Ts': (7, 17), 'Og': (7, 18)
    }

    fig, ax = plt.subplots()

    # # 根据 value 来获取颜色
    # cmap = cm.get_cmap('viridis')
    # color = cmap(norm(value))
    #
    # # 在指定坐标处画一个矩形
    # ax.add_patch(
    #     plt.Rectangle(
    #         (group, -period),  # 左下角坐标 (x, y)，此处 y 取负是为了让 period 从上往下
    #         width=1,
    #         height=1,
    #         linewidth=1,
    #         edgecolor="black",
    #         facecolor=color
    #     )
    # )
    # # 在矩形中央标注元素符号
    # ax.text(
    #     group + 0.5,
    #     -period + 0.5,
    #     symbol,
    #     ha="center",
    #     va="center",
    #     fontsize=8,
    #     color="black"
    # )


def test_():
    # for file in glob('/home/zzy/hotpot/test/outdir/tmqm/mol/*.mol2'):
    #     stem = osp.splitext(osp.basename(file))[0]
    #     mol = next(hp.MolReader(file))
    #     charge = mol.calc_mol_default_charge()
    #
    #     if charge not in [-1, 0, 1]:
    #         print(f"{stem}: {charge}")

    mol = next(hp.MolReader('/home/zzy/hotpot/test/outdir/tmqm/mol/IWAGOE.mol2'))
    charge = mol.calc_mol_default_charge()


if __name__ == '__main__':
    # extract_pyg_data_from_tmQm()
    # export_xtb_files()
    # test_()
    # nt = statistic_node_type_weight()
    extract_pyg_data_from_ccdc_mono()

