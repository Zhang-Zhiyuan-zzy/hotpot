"""
python v3.7.9
@Project: hotpot
@File   : run_gaussian_bundle.py
@Author : Zhiyuan Zhang
@Date   : 2023/3/17
@Time   : 8:39
"""
import sys
sys.path.append('/home/zz1/hotpot')
from src.bundle import MolBundle
from pathlib import Path


def mol_filter(path_mol, mol):
    if len(mol.atoms) < 5:
        return True
    return False


dir_pair = '/home/zz1/proj/bond_length/results/multi_bond/struct/pair'
g16root = '/home/pub'
dir_guass = Path('/home/zz1/proj/gauss')
dir_log = dir_guass.joinpath('log')
dir_chk = dir_guass.joinpath('chk')
dir_err = dir_guass.joinpath('err')

ptb_kwargs = [
    {'mol_distance': 0.05, 'max_generate_num': 50},
    {'mol_distance': 0.1, 'max_generate_num': 100},
    {'mol_distance': 0.15, 'max_generate_num': 150},
    {'mol_distance': 0.20, 'max_generate_num': 200},
]

mb = MolBundle.read_from_dir('mol2', read_dir=dir_pair, generate=True, condition=mol_filter)

mb.gaussian(
    g16root=g16root,
    dir_out=dir_log,
    link0='CPU=0-48',
    route='M062X/Def2SVP SCRF',
    dir_chk=dir_chk,
    dir_err=dir_err,
    perturb_kwargs=ptb_kwargs
)
