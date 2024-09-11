"""
python v3.9.0
@Project: hotpot
@File   : format_convert
@Auther : Zhiyuan Zhang
@Data   : 2024/8/3
@Time   : 15:48
"""
import os
from typing import *
from pathlib import Path
from tqdm import tqdm

from openbabel import openbabel as ob, pybel as pb
import hotpot as hp



def convert_sdf_to_smiles(
        sdf_dir: [str, Path],
        smiles_dir: Union[str, Path],
        split_number: int = 1000000
) -> None:
    sdf_dir = Path(sdf_dir)
    smiles_dir = Path(smiles_dir)
    if not smiles_dir.is_dir():
        raise NotADirectoryError('smiles_dir is not a directory')

    file_num = len(os.listdir(sdf_dir))

    lst_smiles = []
    file_count = 0
    for path_sdf in tqdm(sdf_dir.glob('*.sdf'), total=file_num):
        mol_reader = pb.readfile('sdf', str(path_sdf))
        for mol in tqdm(mol_reader):
            lst_smiles.append(mol.write('smi').strip())
            if len(lst_smiles) % split_number == 0:
                with open(smiles_dir.joinpath(f'smi_{file_count}.csv'), 'w') as writer:
                    for smiles in lst_smiles:
                        writer.write(f"{smiles},\n")

                lst_smiles = []
                file_count += 1

    if lst_smiles:
        with open(smiles_dir.joinpath(f'smi_{file_count}.csv'), 'w') as writer:
            for smiles in lst_smiles:
                writer.write(f"{smiles},\n")

if __name__ == '__main__':
    sdf_dir_ = Path('/home/zz1/data')
    smiles_dir_ = Path('/home/zz1/data_smi')
    convert_sdf_to_smiles(sdf_dir_, smiles_dir_)
