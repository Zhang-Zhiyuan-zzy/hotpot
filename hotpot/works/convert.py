"""
python v3.9.0
@Project: hotpot
@File   : format_convert
@Auther : Zhiyuan Zhang
@Data   : 2024/8/3
@Time   : 15:48
"""
import os
import time
from copy import copy
from os.path import join as opj
from typing import *
from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm
import numpy as np

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


def _build3d(
        mol,
        save_path: str,
        fmt,
        ligand_save_path,
        screenshot_save_path,
        rm_polar_hs: bool = True,
        **kwargs
) -> None:
    if mol.has_metal:
        mol.complexes_build_optimize_(rm_polar_hs = rm_polar_hs, **kwargs)
    else:
        mol.build3d(**kwargs)
        mol.optimize(**kwargs)

    if ligand_save_path and mol.has_metal:
        ligand = copy(mol)
        ligand.remove_metals()
        ligand.optimize(**kwargs)

        ligand.write(
            ligand_save_path,
            fmt,
            write_single=True,
            overwrite=True,
            calc_mol_charge=True,
            **kwargs
        )

    mol.write(
        save_path,
        fmt,
        write_single=True,
        overwrite=True,
        calc_mol_charge=True,
        **kwargs
    )

    if screenshot_save_path:
        mol.write(screenshot_save_path, fmt='sdf')


def convert_smiles_to_3dmol(
        list_smi: list[str],
        save_dir: str,
        alone_ligand_save_dir: str = None,
        sdf_save_dir: str = None,
        file_names: list[str] = None,
        fmt: str = 'gjf',
        nproc: Optional[int] = None,
        timeout: int = 1000,
        **kwargs
):
    """
    Converts a list of SMILES to be a 3D molecule files
    Args:
        list_smi (list[str]):
        save_dir (str):
        alone_ligand_save_dir (str, os.Pathlike, optional):
        file_names (list[str], optional):
        sdf_save_dir (str, os.Pathlike, optional):
        fmt (str): the save file format, default is 'gjf'
        nproc (int, optional):
        timeout (int, optional):

    Keyword Args:
    Molecule 3d builder kwargs:
        build_times:
        init_opt_steps:
        second_opt_steps:
        min_energy_opt_steps:
        correct_hydrogens:
        timeout:

    Molecule 3d optimizer kwargs:
        ff: Optional[Literal['UFF', 'MMFF94', 'MMFF94s', 'GAFF', 'Ghemical']],
        algorithm: Literal["steepest", "conjugate"]
        steps: Optional[int] = 100,
        step_size: int = 100,
        equilibrium: bool = False,
        equi_check_steps: int = 5,
        equi_max_displace: float = 1e-4,
        equi_max_energy: float = 1e-4,
        perturb_steps: Optional[int] = None,
        perturb_sigma: float = 0.5,
        save_screenshot: bool = False,
        increasing_Vdw: bool = False,
        Vdw_cutoff_start: float = 0.0,
        Vdw_cutoff_end: float = 12.5,
        print_energy: Optional[int] = None,
    """
    if file_names is None:
        name_smiles = dict(enumerate(list_smi))
    else:
        if len(file_names) != len(list_smi):
            raise ValueError('The given file_names should have the same length as the given list_smi.')
        name_smiles = dict(zip(file_names, list_smi))

    if not os.path.exists(save_dir):
        raise ValueError(f'The dir {save_dir} does not exist.')

    if alone_ligand_save_dir and str(alone_ligand_save_dir) == str(save_dir):
        raise ValueError('The pairs and ligands should save in different directories.')

    if nproc is None:
        nproc = mp.cpu_count()

    processes = {}
    while name_smiles or processes:
        if name_smiles and len(processes) < nproc:
            name, smiles = name_smiles.popitem()

            mol = next(hp.MolReader(smiles, 'smi'))
            save_path = opj(save_dir, f'{name}.{fmt}')

            if alone_ligand_save_dir:
                ligand_save_path = opj(alone_ligand_save_dir, f'{name}.{fmt}')
            else:
                ligand_save_path = None

            if sdf_save_dir:
                sdf_save_path = opj(sdf_save_dir, f'{name}.sdf')
            else:
                sdf_save_path = None

            p = mp.Process(
                target=_build3d,
                args=(mol, save_path, fmt, ligand_save_path, sdf_save_path),
                kwargs=kwargs
            )
            p.start()
            processes[p] = (time.time(), name)

        to_remove = []
        for p, (t, name) in processes.items():
            if not p.is_alive() or time.time() - t > timeout:
                p.terminate()
                to_remove.append(p)
                print(f"Stop process {name}!!!!")

        for p in to_remove:
            processes.pop(p)


def _convert_g16log_to_gjf(
        g16log_path: str,
        g16gjf_path: str,
        link0=None,
        route=None
):
    try:
        mol = next(hp.MolReader(g16log_path, 'g16log'))
        mol.write(
            g16gjf_path,
            'gjf',
            write_single=True,
            link0=link0,
            route=route,
            overwrite=True,
            ob_opt={'b': None},
            miss_charge_calc=True
        )

    except ValueError:
        print(f'Error convert for {g16log_path}')


def convert_g16log_to_gjf(
     g16log_dir: str,
     g16gjf_dir: str,
     link0=None,
     route=None
):
    for p in Path(g16log_dir).glob("*.log"):
        name = p.stem
        g16gjf_path = opj(g16gjf_dir, f"{name}.gjf")
        _convert_g16log_to_gjf(str(p), g16gjf_path, link0, route)
