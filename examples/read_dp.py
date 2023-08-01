"""
python v3.9.0
@Project: hotpot
@File   : read_dp
@Auther : Zhiyuan Zhang
@Data   : 2023/7/27
@Time   : 12:15
"""
from pathlib import Path
from hotpot.tanks.deepmd import read_system
import hotpot as hp
from tqdm import tqdm

if __name__ == "__main__":
    path_log = Path('/home/zz1/proj/gauss/new/log')
    ms = read_system(path_log, "**/*.log", "gaussian/md", ranges=range(10000))
    mols = []
    for ls in tqdm(ms):
        mols.append(hp.Molecule.build_from_dpdata_system(ls))

    mol = mols[0]
    sys = mol.to_dp_system()
