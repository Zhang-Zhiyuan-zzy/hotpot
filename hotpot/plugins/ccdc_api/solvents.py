import os
import os.path as osp
from collections import OrderedDict

import ccdc.io as cio

import hotpot as hp

__all__ = ['solvents']


solvents_dir = osp.join(hp.package_root, 'cheminfo', 'ChemData', 'solvent')

solvents = OrderedDict()
for sol_filename in os.listdir(solvents_dir):
    sol_path = osp.join(solvents_dir, sol_filename)
    with cio.MoleculeReader(sol_path) as reader:
        mol = reader[0]
        solvents[sol_filename] = mol


if __name__ == '__main__':
    for sol_name, solvent in solvents.items():
        print(f'{sol_name}: {solvent.name.smiles}')
