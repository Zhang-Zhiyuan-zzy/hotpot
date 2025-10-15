import sys
import os.path as osp
import socket
import cProfile
import logging
import pstats
from copy import copy
from tqdm import tqdm
import random
import multiprocessing as mp


logging.basicConfig(level=logging.CRITICAL)

# Initialize paths.
machine_name = socket.gethostname()
print(machine_name)
if machine_name == '4090':
    project_root = '/home/zzy/proj'
    sys.path.append(osp.join(project_root, 'hotpot'))
elif machine_name == 'DESKTOP-G9D9UUB':  # 221 PC
    project_root = '/mnt/d/zhang/OneDrive/Papers/BayesDesign/results'
    sys.path.append(osp.join(project_root, 'hotpot'))
elif machine_name == 'LAPTOP-K2H04HI4':
    project_root = '/mnt/d/zhang/OneDrive/Papers/BayesDesign/results'
    sys.path.append(osp.join(project_root, 'hotpot'))
elif machine_name == 'docker':
    project_root = '/app/proj'
    sys.path.append(osp.join(project_root, 'hotpot'))
elif machine_name == '3090':
    project_root = '/home/zz1/docker/proj'
    sys.path.append(osp.join(project_root, 'hotpot'))

# Running in Super
elif str.split(__file__, '/')[1:4] == ['data', 'run01', 'scz0s3z']:
    print('In Super')
    project_root = '/HOME/scz0s3z/run/proj/'
    sys.path.append(osp.join(project_root, 'hotpot-zzy'))
elif str.split(__file__, '/')[1:4] == ['data', 'user', 'hd54396']:
    print('In zksl Super')
    project_root = '/data/user/hd54396/proj'
    sys.path.append(osp.join(project_root, 'hotpot'))
else:
    raise ValueError(__file__)

import hotpot as hp


def _link_cb(list_smi):
    Eu_pairs = []
    Am_pairs = []
    for smi in list_smi:
        mol = hp.read_mol(smi.strip())
        clone = copy(mol)

        mol.add_hydrogens()
        clone.add_hydrogens()

        Eu_pair = mol.auto_pair_metal('Eu')
        Am_pair = clone.auto_pair_metal('Am')

        Eu_pairs.append(Eu_pair.smiles)
        Am_pairs.append(Am_pair.smiles)

    return Eu_pairs, Am_pairs

def lcb(smi):
    mol = hp.read_mol(smi.strip())
    clone = copy(mol)

    mol.add_hydrogens()
    clone.add_hydrogens()

    Eu_pair = mol.auto_pair_metal('Eu')
    Am_pair = clone.auto_pair_metal('Am')
    return Eu_pair.smiles, Am_pair.smiles


def link_cb_mp():
    smi_file = '/mnt/d/zhang/OneDrive/Papers/BayesDesign/results/raw_ds/DaPhen/smi.txt'
    Am_smi_file = '/mnt/d/zhang/OneDrive/Papers/BayesDesign/results/raw_ds/DaPhen/Am_pair.txt'
    Eu_smi_file = '/mnt/d/zhang/OneDrive/Papers/BayesDesign/results/raw_ds/DaPhen/Eu_pair.txt'

    nproc = 1
    print(f'CPU counts: {nproc}')

    with open(smi_file, 'r') as f:
        lines = f.readlines()
        lines = random.choices(lines, k=1000)

    with mp.Pool(processes=nproc) as pool, open(Am_smi_file, 'a') as f_Am, open(Eu_smi_file, 'a') as f_Eu:
        results_iter = pool.imap(lcb, lines, 200)
        for i, result in enumerate(tqdm(results_iter)):
            Eu_result, Am_result = zip(result)
            f_Eu.write('\n'.join(Eu_result) + '\n')
            f_Am.write('\n'.join(Am_result) + '\n')



def link_cb():
    smi_file = osp.join(project_root, 'raws_ds/DaPhen/smi.txt')
    Am_smi_file = osp.join(project_root, 'raws_ds/DaPhen/Am_pair.txt')
    Eu_smi_file = osp.join('raws_ds/DaPhen/Eu_pair.txt')
    pair_dir = osp.join(project_root, 'raws_ds/DaPhen/pair')

    Eu_pairs = []
    Am_pairs = []
    with open(smi_file, 'r') as f:
        for i, smi in enumerate(tqdm(f), 1):
            mol = hp.read_mol(smi.strip())
            clone = copy(mol)

            mol.add_hydrogens()
            clone.add_hydrogens()

            Eu_pair = mol.auto_pair_metal('Eu')
            Am_pair = clone.auto_pair_metal('Am')

            Eu_pairs.append(Eu_pair.smiles)
            Am_pairs.append(Am_pair.smiles)

            if i % 10000 == 0:
                with open(Eu_smi_file, 'a') as Euf:
                    Euf.write('\n'.join(Eu_pairs) + '\n')
                with open(Am_smi_file, 'a') as Amf:
                    Amf.write('\n'.join(Am_pairs) + '\n')


def link_optimize():
    smi_file = '/mnt/d/zhang/OneDrive/Papers/BayesDesign/results/raw_ds/DaPhen/smi.txt'
    pair_dir = '/mnt/d/zhang/OneDrive/Papers/BayesDesign/results/raw_ds/DaPhen/pair'

    with open(smi_file, 'r') as f:
        for i, smi in enumerate(tqdm(f), 1):
            mol = hp.read_mol(smi.strip())
            clone = copy(mol)

            mol.add_hydrogens()
            clone.add_hydrogens()

            Eu_pair = mol.auto_pair_metal('Eu')
            Am_pair = clone.auto_pair_metal('Am')

            Eu_pair.complexes_build_optimize_(steps=100, step_size=50, save_screenshot=False)
            Am_pair.complexes_build_optimize_(steps=150, step_size=50, save_screenshot=False)


            intersect_mol = []
            if Eu_pair.has_bond_ring_intersection:
                intersect_mol.append(f'Eu_{i}pair')
            if Am_pair.has_bond_ring_intersection:
                intersect_mol.append(f'Am_{i}pair')

            if intersect_mol:
                print(RuntimeWarning(','.join(intersect_mol) + ' is intersected Molecule'))
                continue

            Eu_pair.write(osp.join(pair_dir, f'Eu_{i}pair.mol'))
            Am_pair.write(osp.join(pair_dir, f'Am_{i}pair.mol'))

            if i == 100:
                break


def statistics_rings_size():
    smi_file = '/mnt/d/zhang/OneDrive/Papers/BayesDesign/results/raw_ds/DaPhen/smi.txt'

    max_rings_nums = 0
    max_rings_size = 0
    with open(smi_file, 'r') as f:
        lines = f.readlines()
        lines = random.choices(lines, k=100000)

        for i, smi in enumerate(tqdm(lines), 1):
            mol = hp.read_mol(smi.strip())
            rings = mol.ligand_rings
            max_rings_nums = max(max_rings_nums, len(rings))
            max_rings_size = max(max_rings_size, max([len(r) for r in rings]))

            if i % 100000 == 0:
                print(f'Current max_rings_nums: {max_rings_nums}, max_rings_size: {max_rings_size}')


def main():
    link_cb()
    # link_optimize()
    # link_cb_mp()

if __name__ == '__main__':
    main()
    # with cProfile.Profile() as pr:
    #     main()
    # ps = pstats.Stats(pr)
    # ps.sort_stats(pstats.SortKey.CUMULATIVE).print_stats('hotpot')
    # ps.sort_stats(pstats.SortKey.CUMULATIVE).print_stats('/mnt/d/hotpot/hotpot/cheminfo/AImodels/cbond/apply.py')
    # # statistics_rings_size()
    # for item, count in apply.cbond_session_stat.items():
    #     print(item, count)
