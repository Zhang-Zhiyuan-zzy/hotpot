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
from typing import Optional, Union
from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm

from openbabel import pybel as pb
import hotpot as hp


_PROCESS_POLL_INTERVAL = 0.01
_PROCESS_SHUTDOWN_TIMEOUT = 5.0
_BUILD_TIMEOUT_CLEANUP_GRACE = 2.0 * _PROCESS_SHUTDOWN_TIMEOUT


def _terminate_process(process: mp.Process) -> None:
    """Stop a conversion worker and wait until its process resources are reaped."""
    process.terminate()
    process.join(timeout=_PROCESS_SHUTDOWN_TIMEOUT)
    if process.is_alive():
        process.kill()
        process.join(timeout=_PROCESS_SHUTDOWN_TIMEOUT)
    if process.is_alive():
        raise RuntimeError("Conversion worker did not stop after kill")



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
        **kwargs
) -> None:
    save_movie = kwargs.get('save_movie', False)
    mol.build3d(**kwargs)

    if ligand_save_path and mol.has_metal:
        ligand = copy(mol)
        ligand.remove_metals()
        optimize_options = {
            name: value
            for name, value in kwargs.items()
            if name in {
                'forcefield',
                'algorithm',
                'epochs',
                'steps_per_epoch',
                'add_hydrogens',
                'quality_level',
                'quality_thresholds',
                'seed',
                'perturb_interval',
                'perturb_sigma',
                'save_movie',
                'increasing_vdw',
                'vdw_cutoff_start',
                'vdw_cutoff_end',
            }
        }
        ligand.optimize(**optimize_options)

        ligand.write(
            ligand_save_path,
            fmt,
            write_single=not save_movie,
            overwrite=True,
            calc_mol_charge=True,
        )

    mol.write(
        save_path,
        fmt,
        write_single=not save_movie,
        overwrite=True,
        calc_mol_charge=True,
    )

    if screenshot_save_path:
        mol.write(
            screenshot_save_path,
            fmt='sdf',
            write_single=not save_movie,
            overwrite=True,
        )


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
        timeout (int, optional): Timeout passed to ``Molecule.build3d``. The
            outer conversion worker receives a bounded cleanup grace period
            before it is forcibly stopped.

    Keyword Args:
        Keyword arguments accepted by :meth:`Molecule.build3d`, including
        ``forcefield``, ``epochs``, ``steps_per_epoch``, ``add_hydrogens``,
        ``quality_level``, ``seed``, and complex-candidate options.
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

    build_options = dict(kwargs)
    build_options['timeout'] = timeout
    process_timeout = timeout + _BUILD_TIMEOUT_CLEANUP_GRACE
    processes = {}
    try:
        while name_smiles or processes:
            while name_smiles and len(processes) < nproc:
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
                    kwargs=build_options,
                )
                p.start()
                processes[p] = (time.monotonic(), name)

            to_remove = []
            for p, (started_at, name) in processes.items():
                if not p.is_alive():
                    p.join()
                    to_remove.append(p)
                    if p.exitcode != 0:
                        print(f"Process {name} exited with code {p.exitcode}.")
                elif time.monotonic() - started_at > process_timeout:
                    _terminate_process(p)
                    to_remove.append(p)
                    print(
                        f"Process {name} exceeded the {timeout:g}-second build "
                        "timeout and was stopped."
                    )

            for p in to_remove:
                processes.pop(p)

            if processes:
                time.sleep(_PROCESS_POLL_INTERVAL)
    finally:
        for process in processes:
            if process.is_alive():
                _terminate_process(process)
            else:
                process.join()


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
