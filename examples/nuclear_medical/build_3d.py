"""
python v3.9.0
@Project: hp5
@File   : build_3d
@Auther : Zhiyuan Zhang
@Data   : 2024/7/6
@Time   : 11:16
"""
import os
import sys
import platform
import shutil
import random
from os.path import join as opj
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import hotpot as hp
from hotpot.works import convert, gauss

if 'microsoft' not in platform.uname().release.lower():
    sys.path.append('/home/zz1/hp')
    output_dir = opj('/home', 'zz1', 'dy', 'g16')
else:
    output_dir = opj('/mnt', 'd', 'zhang', 'OneDrive', 'Papers', 'dy')

def main():
    """"""
    export_gjf_files()

def test():
    df_data = pd.read_csv('mc_first.csv')
    complexes_smiles = df_data['all_smiles'].values
    smi = complexes_smiles[45]

    mol = next(hp.MolReader(smi, 'smi'))

    # print(mol.charge)
    # mol.charge = mol.calc_mol_default_charge()
    # print(mol.charge)

    mol.complexes_build_optimize_(save_screenshot=True)
    mol.write(f'/mnt/d/mol/{45}.gjf', overwrite=True, calc_mol_charge=True)
    mol.write(f'/mnt/d/mol/{45}.sdf', overwrite=True)

def export_gjf_files(timeout=1000):
    """"""
    df_data = pd.read_csv('mc_first.csv')
    complexes_smiles = df_data['all_smiles'].values
    # ligand_smiles = df_data['l_smiles'].values

    # len(complexes_smiles)
    smi_indices = np.random.randint(low=0, high=len(complexes_smiles), size=163).tolist()
    complexes_smiles = complexes_smiles[smi_indices].tolist()
    # ligand_smiles = ligand_smiles[smi_indices].tolist()

    convert.convert_smiles_to_3dmol(
        complexes_smiles,
        save_dir=opj(output_dir, 'gjf4', 'pair'),
        alone_ligand_save_dir=opj(output_dir, 'gjf4', 'ligand'),
        sdf_save_dir=opj(output_dir, 'sdf', 'pair4'),
        file_names=smi_indices,
        timeout=timeout,
        rm_polar_hs = False
        # nproc=1
    )

def build3d(i, smi):
    mol = next(hp.MolReader(smi, 'smi'))

    Ga = mol.create_atom(symbol='Ga', formal_charge=3)

    coordination_atoms = [a for a in mol.atoms if a.symbol in ['N', 'O']]
    coordination_atoms = np.random.choice(coordination_atoms, min(6, len(coordination_atoms)), replace=False)
    for a in coordination_atoms:
        b = mol.add_bond(Ga, a, 1)

    mol.complexes_build_optimize_(save_screenshot=True)
    mol.write(opj(f'/mnt/d/zhang/OneDrive/Papers/dy/gjf/pair/{i}.gjf'), overwrite=True, write_single=True)
    mol.write(opj(f'/mnt/d/zhang/OneDrive/Papers/dy/sdf/pair/{i}.sdf'), overwrite=True, write_single=True)
    # linux
    # mol.write(opj(f'/home/zz1/dy/g16/gjf/pair/{i}.gjf'), overwrite=True, write_single=True)
    # mol.write(opj(f'/home/zz1/dy/g16/sdf/pair/{i}.sdf'), overwrite=True)

    mol.remove_atoms(mol.metals)
    mol.optimize(steps=20)

    # windows
    mol.write(opj(f'/mnt/d/zhang/OneDrive/Papers/dy/gjf/ligand/{i}.gjf'), overwrite=True, write_single=True)

    # linux
    # mol.write(opj(f'/home/zz1/dy/g16/gjf/ligand/{i}.gjf'), overwrite=True, write_single=True)


def build3try():
    df_data = pd.read_excel('3try.xlsx')
    complexes_smiles = df_data.iloc[:,0].values.tolist()[10:20]


    list_smi = []
    for smi in complexes_smiles:
        mol = next(hp.MolReader(smi, 'smi'))
        metal = mol.metals[0]
        list_on = [a for a in mol.atoms if (a.symbol == 'O') or (a.symbol == 'N' and a.is_aromatic)]

        for a in list_on[:2]:
            try:
                mol.add_bond(metal, a, 1)
            except:
                continue

        list_smi.append(mol.smiles)


    convert.convert_smiles_to_3dmol(
        list_smi,
        save_dir=opj(output_dir, '3try', 'pair'),
        alone_ligand_save_dir=opj(output_dir, '3try', 'ligand'),
        sdf_save_dir=opj(output_dir, '3try', 'sdf'),
        file_names=[str(i) for i in range(10, 20)],
        timeout=300,
        rm_polar_hs=False
        # nproc=1
    )



def convert_log2gjf():
    log_dir = opj(output_dir, 'log', 'ligand')
    gjf_dir = opj(output_dir, 'gjf', 'log2gjf', 'ligand')

    convert.convert_g16log_to_gjf(log_dir, gjf_dir)


def sum_log_results():
    log_dir = opj(output_dir,  'dgbl')

    res = gauss.export_M_L_pair_calc_results(
        pair_log_dir=opj(log_dir, 'pair'),
        ligand_log_dir=opj(log_dir, 'ligand'),
        metal_log_dir=opj(log_dir, 'metal'),
        nproc=16
    )

    res.to_csv(opj(output_dir, 'dgbl', 'result.csv'))
    print('Done !!!!!!!!!!')


def get_sort_number(a: np.ndarray):
    sort = []
    for v in a:
        sort.append(sum(np.int_(v > a)))

    return np.array(sort)


def align_sort():
    df = pd.read_csv(opj(output_dir, 'result.csv'), index_col=0)
    dG = df['Delta_gibbs'].values
    indices = df.index.values

    filter_idx = np.logical_and(-100 < dG, dG < 0)

    dG = dG[filter_idx]
    indices = indices[filter_idx]
    #
    # fig, ax = plt.subplots()
    # ax.scatter(indices, dG)
    # fig.show()

    index_sort = get_sort_number(indices)
    value_sort = get_sort_number(dG)
    # index_sort = np.argsort(indices)
    # value_sort = np.argsort(abs(dG))
    #
    diff_sort = abs(index_sort - value_sort)
    #
    sort_abs_diff = np.argsort(diff_sort)
    #
    print(diff_sort[sort_abs_diff])
    print(indices[sort_abs_diff])

    s = sort_abs_diff[:125]
    c = np.array(np.random.choice(sort_abs_diff[125:200], 25))
    #
    result_idx = indices[np.hstack((s, c))]
    result = dG[np.hstack((s, c))]

    df = pd.Series(result, index=result_idx, name='ΔG')
    df.to_csv(opj(output_dir, 'sort_result2.csv'))

    fig, ax = plt.subplots()
    ax.scatter(result_idx, result)
    fig.savefig(opj(output_dir, 'sort_result2.png'))
    fig.show()


def show_bl_dG():
    df_data = pd.read_csv('mc_first.csv', index_col=0)
    bl = df_data['predictions']

    df = pd.read_csv(opj(output_dir, 'result.csv'), index_col=0)
    # df = pd.read_csv(opj(output_dir, 'sort_result1.csv'), index_col=0)
    dG = df['Delta_gibbs'].values
    indices = df.index.values

    filter_idx = np.logical_and(-100 < dG, dG < 0)

    dG = dG[filter_idx]
    indices = indices[filter_idx]

    bl = bl.loc[indices].values

    bl_sort = get_sort_number(bl)
    dG_sort = get_sort_number(dG)

    # fig, ax = plt.subplots()
    # ax.scatter(bl, dG)
    # fig.show()

    diff_sort = abs(bl_sort - dG_sort)
    sort_abs_diff = np.argsort(diff_sort)


    s = sort_abs_diff[:125]
    c = np.array(np.random.choice(sort_abs_diff[125:200], 25))
    #
    result_bl = bl[np.hstack((s, c))]
    result = dG[np.hstack((s, c))]


    df = pd.Series(result, index=result_bl, name='ΔG')
    df.to_csv(opj(output_dir, 'sort_bl0.csv'))

    fig, ax = plt.subplots()
    ax.scatter(result_bl, result)
    fig.savefig(opj(output_dir, 'sort_bl0.png'))
    fig.show()


def mv_log():
    df_data = pd.read_csv('mc_first.csv', index_col=0)
    bl = df_data['predictions']

    df = pd.read_csv(opj(output_dir, 'result.csv'), index_col=0)
    dG = df['Delta_gibbs']

    df = pd.read_csv(opj(output_dir, 'sort_bl0.csv'))

    print(bl)
    print(dG)
    print(df)

    for v, g in df.values:
        g_idx = dG == g
        v_idx = bl == v

        idx = bl.index[v_idx].values[0]

        try:
            shutil.copy(opj(output_dir, 'log', 'pair', f'{idx}.log'), opj(output_dir, 'dgbl', 'pair', f'{idx}.log'))
            shutil.copy(opj(output_dir, 'log', 'ligand', f'{idx}.log'), opj(output_dir, 'dgbl', 'ligand', f'{idx}.log'))

        except FileNotFoundError:
            print(idx)


def fill_to_150(src_dir, des_dir, mv_num=8):
    src_ligand_dir = opj(src_dir, 'ligand')
    src_pair_dir = opj(src_dir, 'pair')
    des_ligand_dir = opj(des_dir, 'ligand')
    des_pair_dir = opj(des_dir, 'pair')

    src_ligand_files = os.listdir(src_ligand_dir)
    src_pair_files = os.listdir(src_pair_dir)
    src_files = set(src_ligand_files) & set(src_pair_files)

    des_ligand_files = os.listdir(des_ligand_dir)
    des_pair_files = os.listdir(des_pair_dir)
    des_files = set(des_ligand_files) & set(des_pair_files)

    need_mv = src_files.difference(des_files)
    mv_files = random.sample(list(need_mv), k=mv_num)

    for file in mv_files:
        src_ligand_path = opj(src_ligand_dir, file)
        src_pair_path = opj(src_pair_dir, file)
        des_ligand_path = opj(des_ligand_dir, file)
        des_pair_path = opj(des_pair_dir, file)

        if os.path.exists(des_ligand_path) or os.path.exists(des_pair_path):
            raise IOError(f'{file} has already been exist in {src_ligand_dir} or {src_pair_dir}')

        shutil.copy(src_ligand_path, des_ligand_path)
        shutil.copy(src_pair_path, des_pair_path)


def show_dgbl():
    dg_df = pd.read_csv(opj(output_dir, 'dgbl', 'result.csv'), index_col=0)
    bl_df = pd.read_csv(opj(output_dir, 'dgbl', 'mc_first.csv'))

    dg = dg_df['Delta_gibbs'].values
    bl = bl_df.loc[dg_df.index.tolist(), 'predictions']

    fig, ax = plt.subplots()
    ax.scatter(bl, dg)
    fig.savefig(opj(output_dir, 'dgbl', 'dgbl.png'))

if __name__ == '__main__':
    # print(os.getcwd())
    main()
    # test()
    # convert_log2gjf()
    # sum_log_results()
    # align_sort()

    # show_bl_dG()
    # mv_log()
    # fill_to_150(
    #     src_dir=opj('/mnt', 'd', 'zhang', 'OneDrive', 'Papers', 'dy', 'log'),
    #     des_dir=opj('/mnt', 'd', 'zhang', 'OneDrive', 'Papers', 'dy', 'dgbl')
    # )
    # show_dgbl()
    build3try()

    # smis = [
    #     '[O-]C1=O[Ga]234[N](CC[N](CC(O)=O3)2CCN(CC([O-])=O)CC([O-])=O)(CC([O-])=O4)C1',
    #     'OC(C[N]12CC[N@]34CC[N]5(CC(O)=O)CC[N@@]6(CC(O)=O[Ga]5632O=C(O)C4)CC1)=O'
    # ]
    #
    # names = [
    #     'DTPA',
    #     'DOTA'
    # ]
    #
    # convert.convert_smiles_to_3dmol(
    #     smis,
    #     save_dir=opj(output_dir, 'validated', 'pair'),
    #     alone_ligand_save_dir=opj(output_dir, 'validated', 'ligand'),
    #     sdf_save_dir=opj(output_dir, 'validated', 'sdf'),
    #     file_names=names,
    #     timeout=1000,
    #     rm_polar_hs = False
    #     # nproc=1
    # )
