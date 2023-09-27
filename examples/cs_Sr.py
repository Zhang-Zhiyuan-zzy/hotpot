"""
python v3.9.0
@Project: hotpot
@File   : cs_Sr
@Auther : Zhiyuan Zhang
@Data   : 2023/9/1
@Time   : 9:05
"""
from pathlib import Path
import pandas as pd

import hotpot as hp

if __name__ == "__main__":
    path_csv = Path("/home/zz1/temp/d_rbl.csv")
    dir_mol2 = Path("/home/zz1/temp/v_pair")

    series = pd.read_csv(path_csv, index_col=0)

    smiles = []
    for idt, rbl in zip(series.index, series.values.flatten()):
        path_mol2 = dir_mol2.joinpath(f"{idt}-Sr.mol2")
        try:
            mol = hp.Molecule.read_from(path_mol2)
        except OSError:
            continue

        print(mol)
        smiles.append([mol.smiles.replace('[Sr]', '[M]'), rbl])

    df = pd.DataFrame(smiles)
    df.to_csv("/home/zz1/temp/rank_smi.csv")
