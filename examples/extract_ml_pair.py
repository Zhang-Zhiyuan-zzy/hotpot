"""
python v3.9.0
@Project: hotpot
@File   : extract_ml_pair
@Auther : Zhiyuan Zhang
@Data   : 2023/7/3
@Time   : 2:34

Note:
    This python code to perform extraction of metal-ligand pairs from cif file
"""
import tqdm
from pathlib import Path
import hotpot as hp


if __name__ == '__main__':
    path_cif = Path('/home/zz1/database/CSD')
    bundle = hp.MolBundle.read_from('cif', path_cif, ranges=range(1000))
    for i, mol in enumerate(tqdm.tqdm(bundle)):
        mol.remove_solvents()
