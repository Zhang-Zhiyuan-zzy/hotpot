import os
import torch
import hotpot as hp
import random
from tqdm import tqdm

import pebble
import concurrent
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from hotpot.plugins.plots import SciPlotter

def extract_metal_symbol(smiles: str):
    s1, s2 = smiles.split('.')
    if len(s2) == 1:
        metal = s1[1:-1].split('+')[0]
    else:
        metal = s2[1:-1].split('+')[0]
    return metal


def loading_data():
    ds_root = '/home/zzy/proj/datasets/SclogK_with_cb'
    list_data = []
    for name in tqdm(os.listdir(ds_root), 'Loading data'):
        f = os.path.join(ds_root, name)
        data = torch.load(f, weights_only=False)
        list_data.append(data)
    return list_data


def get_broken_complexes():
    list_data = loading_data()
    broken_data, linked_data = [], []

    for data in tqdm(list_data, "Searching for broken complexes"):
        if len(data.pair_smiles.split('.')) > 1:
            broken_data.append(data)
        else:
            linked_data.append(data)

    return broken_data, linked_data


def hist_of_broken_linked_complexes():
    break_data, linked_data = get_broken_complexes()

    break_y = np.array([data.y.flatten().item() for data in break_data])
    link_y = np.array([data.y.flatten().item() for data in linked_data])

    plt.hist(break_y, bins=100, alpha=0.5, label='Broken complexes')
    plt.hist(link_y, bins=100, alpha=0.5, label='Link complexes')

    plt.legend()
    plt.show()

    print(f'Broken complexes: {np.mean(break_y)}')
    print(f'Link complexes: {np.mean(link_y)}')


def get_cbond_ys(return_df: bool = False):
    break_data, linked_data = get_broken_complexes()
    cbond_ys = {0: [d.y.flatten().item() for d in break_data]}
    for data in tqdm(linked_data, "Statistic cbond number"):
        mol = hp.read_mol(data.pair_smiles)
        ys = cbond_ys.setdefault(len(mol.c_bonds), [])
        ys.append(data.y.flatten().item())

    if not return_df:
        return {i: np.array(ys) for i, ys in sorted(cbond_ys.items(), key=lambda it: it[0])}
    else:
        list_ys = []
        list_cb = []
        for i, ys in cbond_ys.items():
            list_ys.extend(ys)
            list_cb.extend([i] * len(ys))
        data = np.array([list_ys, list_cb]).T
        return pd.DataFrame(data, columns=['logK', 'CB Numbers'])


def statistic_cbond_number():
    xlabel = "CB Numbers"
    ylabel = "$logK_{1}$"
    def violin_maker(cbond_ys):
        def maker(ax, sciplot):
            sns.violinplot(cbond_ys, ax=ax, x='CB Numbers', y='logK')
            # ax.set_xlabel(xlabel)
            # ax.set_ylabel(ylabel)
        return maker

    def stat_bars(cbond_ys):
        def maker(ax, sciplot):
            sns.barplot(cbond_ys, ax=ax, x='CB Numbers', errorbar='cd')
        return maker

    # Plotters
    cys = get_cbond_ys(True)

    plotter = SciPlotter(violin_maker(cys))
    fig, axs = plotter()
    fig.show()



if __name__ == '__main__':
    miss_smi = '/home/zzy/proj/datasets/miss_smi.txt'
    statistic_cbond_number()
    # break_data, linked_data = get_broken_complexes()
    # cbond_ys = {0: [d.y.flatten().item() for d in break_data]}
    # for data in tqdm(linked_data, "Statistic cbond number"):
    #     mol = hp.read_mol(data.pair_smiles)
    #     ys = cbond_ys.setdefault(len(mol.c_bonds), [])
    #     ys.append(data.y.flatten().item())
    #
    # # cbond_ys = {i: np.array(ys) for i, ys in cbond_ys.items()}
    #
    # fig, ax = plt.subplots()
    # # sns.violinplot(list(cbond_ys.values()), x=list(cbond_ys.keys()), ax=ax)
    # sns.violinplot(cbond_ys, ax=ax)
    # stat_results = []
    # cols = ['mean', 'std', 'median', 'SampleNums']
    # index = []
    # for i, ys in sorted(cbond_ys.items(), key=lambda x: x[0]):
    #     index.append(i)
    #     stat_results.append([round(np.mean(ys),3), round(np.std(ys),3), round(np.median(ys),3), len(ys)])
    #     print(f'CBondNum{i}:\tmean={round(np.mean(ys),3)},\t\tstd={round(np.std(ys),3)},\tmedian={round(np.median(ys),3)}\tlen={len(ys)}')
    #
    # df = pd.DataFrame(stat_results, columns=cols, index=index,)
    #
    # fig.show()



    # change = []
    # still = []
    # with pebble.ProcessPool(2) as pool:
    #     for data in tqdm(list_data):
    #         mol = hp.read_mol(data.smiles)
    #         mol.add_hydrogens()
    #         mol.force_remove_polar_hydrogens()
    #         try:
    #             mol = pool.schedule(mol.auto_pair_metal, (extract_metal_symbol(data.pair_smiles),), timeout=30.).result()
    #         except AssertionError:
    #             continue
    #         except concurrent.futures.TimeoutError:
    #             continue
    #
    #         if '.' in mol.smiles:
    #             still.append(mol.smiles)
    #         else:
    #             change.append(mol.smiles)
    #
    # print(len(still))
    # print(len(change))
    # with open(miss_smi, 'w') as f:
    #     f.write('\n'.join(still))