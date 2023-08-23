"""
python v3.9.0
@Project: hotpot
@File   : calc_bde.py
@Author : Zhiyuan Zhang
@Date   : 2023/6/14
@Time   : 10:30

Note:
    This script to high-throughput determine the bind energy when a metal cation coordinate to a ligand with
    specified coordination pattern
"""
import os
from pathlib import Path
from typing import Union

import hotpot as hp

from hotpot.cheminfo import Atom


g16root = '/home/pub'


def calc_element_energy():
    """ Calculate the electron energies for all """
    for i in range(1, 58):
        a = Atom(atomic_number=i)
        e, c = a.calc_qm_energy(g16root, "M062X", "Def2SVP", _record=True)
        print(a.symbol, c, e)


def calc_cc_bde():
    """ Calculate the BDE of the coordinating bonds in Metal-Ligand pair """
    START_NUM = 29

    path_smiles = Path('/home/zz1/proj/be/struct/choice_ligand')
    work_dir = Path('/home/zz1/proj/be/g161')
    os.chdir(work_dir)

    smiles = open(path_smiles).readlines()

    for i, s in enumerate(smiles[START_NUM:], START_NUM):

        mol = hp.Molecule.read_from(s, 'smi')
        pair_bundle = mol.generate_pairs_bundle('Sr')

        if len(pair_bundle) == 0:
            continue

        pair_bundle.determine_metal_ligand_bind_energy(
            g16root, work_dir.joinpath(str(i)), 'M062X', 'Def2SVP', 'SCRF pop(Always)', cpu_uti=0.5,
            skip_complete=True
        )


def read_calc_data(root_dir: Union[Path, str]):
    """ collect the collected data from log file """
    bundle = hp.MolBundle.read_from("g16log", root_dir, "*/log/*.log", nproc=2)

    print(f"pair number: {len([m for m in bundle if m.is_pair])}")
    for mol in bundle:
        if mol.metals:
            metal = mol.metals[0]
            n_atom, dist = metal.nearest_atom

            if n_atom.symbol == "O" and dist < 3.0 and not mol.bond(metal.ob_id, n_atom.ob_id):
                mol.add_bond(metal, n_atom)
                print(f"{mol} add bond between {metal} and {n_atom} with length {dist}")

    print(f"pair number: {len([m for m in bundle if m.is_pair])}")

    return bundle


def calc_csd_pair(pair_dir: Union[Path, str], result_dir: Union[Path, str]):
    pair_dir = Path(pair_dir)
    result_dir = Path(result_dir)

    for path_mol in pair_dir.glob("*.mol2"):
        mol = hp.Molecule.read_from(path_mol)

        mol.gaussian(
            g16root,
            link0=['nproc=32', "Mem=128GB"],
            route=["M062X", "Def2SVP"],
            path_log_file=result_dir.joinpath("log", f"{path_mol.stem}.log"),
            path_err_file=result_dir.joinpath("err", f"{path_mol.stem}.err"),
            path_chk_file=result_dir.joinpath("chk", f"{path_mol.stem}.chk"),
            path_rwf_file=result_dir.joinpath("rwf", f"{path_mol.stem}.rwf"),
            output_in_running=False,
            path_gjf=result_dir.joinpath("gjf", f"{path_mol.stem}.gjf")
        )


if __name__ == '__main__':
    b = read_calc_data("/home/zz1/proj/be/g16")
    # calc_csd_pair("/home/zz1/proj/be/csd/sc", "/home/zz1/proj/be/csd/result")
