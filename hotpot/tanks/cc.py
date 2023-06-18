"""
python v3.9.0
@Project: hotpot
@File   : cc
@Auther : Zhiyuan Zhang
@Data   : 2023/6/4
@Time   : 21:05

This module is to perform `Task` about Coordination Chemistry (cc)
"""
import os
from pathlib import Path
from typing import *
import json

import numpy as np
import pandas as pd

from hotpot import data_root
from hotpot.cheminfo import Molecule, Atom
from hotpot.bundle import MolBundle

# the atomic single point energies determined Gaussian with various methods and basis sets
_atom_single_point: dict = json.load(open(Path(data_root).joinpath('atom_single_point.json')))


class PairBundle(MolBundle):
    """The MolBundle which contains a ligands, a metal as well as their pairs assembled by them"""
    def __init__(self, metal: Atom, ligand: Molecule, pairs: Sequence[Molecule]):
        self.metal = Molecule()
        self.metal.add_atom(metal)
        self.metal.identifier = metal.symbol

        self.ligand = ligand

        super(PairBundle, self).__init__(pairs)

    @property
    def pairs(self):
        return self.mols

    def determine_metal_ligand_bind_energy(
            self, g16root: Union[str, os.PathLike], work_dir: Union[str, os.PathLike],
            method: str = 'B3LYP', basis_set: str = '6-311', route: str = ''
    ) -> pd.DataFrame:
        # Specify directories
        if isinstance(work_dir, str):
            work_dir = Path(work_dir)

        if not work_dir.exists():
            work_dir.mkdir()

        log_dir = work_dir.joinpath('log')
        err_dir = work_dir.joinpath('err')
        struct_dir = work_dir.joinpath()

        if not log_dir.exists():
            log_dir.mkdir()
        if not err_dir.exists():
            err_dir.mkdir()
        if not struct_dir.exists():
            struct_dir.mkdir()

        pairs_log_dir = log_dir.joinpath('pairs')
        pairs_err_dir = err_dir.joinpath('pairs')

        if not pairs_log_dir.exists():
            pairs_log_dir.mkdir()
        if not pairs_err_dir.exists():
            pairs_err_dir.mkdir()

        e_sheet = []  # Energy sheet

        # optimize the configure of ligand and calculate their total energy after optimization
        self.ligand.gaussian(
            g16root,
            link0=f'CPU=0-{os.cpu_count()-1}',
            route=f'opt {method}/{basis_set}' + route,
            path_log_file=log_dir.joinpath('ligand.log'),
            path_err_file=err_dir.joinpath('ligand.err'),
            inplace_attrs=True
        )

        ligand_energy = self.ligand.energy  # Retrieve the energy after optimizing the conformer
        e_sheet.append(['ligand', self.ligand.smiles, ligand_energy])

        # save the optimized structures
        self.ligand.writefile('mol2', struct_dir.joinpath('ligand.mol2'))

        # Calculate the single point (sp) energy for metal
        try:
            metal_sp = _atom_single_point[self.metal.atoms[0].symbol][method][basis_set]
        except KeyError:
            self.metal.gaussian(
                g16root,
                link0=f'CPU=0-{os.cpu_count()-1}',
                route=f'{method}/{basis_set}' + route,
                path_log_file=log_dir.joinpath('metal.log'),
                path_err_file=err_dir.joinpath('metal.err'),
                inplace_attrs=True
            )

            ele_dict = _atom_single_point.setdefault(self.metal.atoms[0].symbol, {})
            ele_method_dict = ele_dict.setdefault(method, {})

            metal_sp = ele_method_dict[basis_set] = self.metal.energy  # Recording the calculate SCF energy to the dict

            # Save the single point as package data
            json.dump(_atom_single_point, open(Path(data_root).joinpath('atom_single_point.json'), 'w'), indent=True)

        # Append the metal energy values to energy sheet
        e_sheet.append(['metal', self.metal.smiles, metal_sp])
        # Save metal structure
        self.metal.writefile('mol2', struct_dir.joinpath(f'{self.metal.identifier}.mol2'))

        # Optimizing the conformer of metal-ligands pairs and Retrieve the energies in the last stable conformer
        for i, pair in enumerate(self.pairs):
            pair.gaussian(
                g16root,
                link0=f'CPU=0-{os.cpu_count() - 1}',
                route=f'opt {method}/{basis_set}' + route,
                path_log_file=pairs_log_dir.joinpath(f'pair_{i}.log'),
                path_err_file=pairs_err_dir.joinpath(f'pair_{i}.err'),
                inplace_attrs=True
            )

            # Append the pairs energy values to energy sheet
            e_sheet.append([f'pair_{i}', pair.smiles, pair.energy])
            # Save Metal-ligand pair struct
            pair.writefile('mol2', struct_dir.joinpath(f'pair_{i}.mol2'))

        # Save the energy sheet to csv
        e_sheet = np.array(e_sheet)
        df = pd.DataFrame(e_sheet[:, 1:], index=e_sheet[:, 0], columns=['smiles', 'Energy(eV)'])
        df.to_csv(work_dir.joinpath('energies.csv'))

        return df


