import os.path as osp
from glob import glob
import shutil
import socket
from tqdm import tqdm

import hotpot as hp


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


def fileter_xtb_mol():
    filter_file_dir = osp.join(ccdc_files_dir, 'An')
    for xtb_dir in tqdm(glob(osp.join(ccdc_xtb_file_dir, '*'))):
        mol = next(hp.MolReader(osp.join(xtb_dir, 'struct.mol')))
        if _filter_an(mol):
            shutil.copytree(xtb_dir, osp.join(filter_file_dir, osp.split(xtb_dir)[-1]))


def _filter_mol(mol):
    if not (10 < len(mol.atoms) < 150):
        return False
    if any(a.atomic_number > 86 for a in mol.atoms):
        return False

    return True


def _filter_ln(mol):
    an = [a.atomic_number for a in mol.atoms]
    if (10 < len(an) < 150) and any(57 <= n <= 71 for n in an) and all(n < 86 for n in an):
        return True

    return False


def _filter_an(mol):
    an = [a.atomic_number for a in mol.atoms]
    if (10 < len(an) < 300) and any(86 <= n for n in an):
        return True

    return False


if __name__ == '__main__':
    fileter_xtb_mol()
