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
from openbabel import openbabel as ob

from hotpot import data_root
from hotpot.cheminfo import Molecule, Atom
from hotpot.bundle import MolBundle
from hotpot.utils.manage_machine import machine

# the atomic single point energies determined Gaussian with various methods and basis sets
_atom_single_point: dict = json.load(open(Path(data_root).joinpath('atom_single_point.json')))


class MetalLigandPair(Molecule):
    """ The Molecule to represent a metal-ligand pair """
    def _set_bond_dissociation_energy(self, bde: float):
        """"""
        bde_store = ob.OBCommentData()
        bde_store.SetData(str(bde))
        bde_store.SetAttribute('BDE')
        self.ob_mol.CloneData(bde_store)

    @property
    def bond_dissociation_energy(self):
        bde_store = self.ob_mol.GetData('BDE')
        if bde_store:
            bde = ob.toCommentData(bde_store).GetValue()
            return float(bde)

        return 0.0

    @bond_dissociation_energy.setter
    def bond_dissociation_energy(self, bde: float):
        print(bde)
        self._set_bond_dissociation_energy(bde)


class PairBundle(MolBundle):
    """The MolBundle which contains a ligands, a metal as well as their pairs assembled by them"""

    class DirsFiles:
        """
        Specify directory and files path for utils in Gaussian calculation
        Attributes:
            work_dir: the root dir for all results af Gaussian calculation
            log_dir: the dir to store Gaussian log files
            err_dir: the dir to store Gaussian error message
            struct_dir: the structure after optimizing
            energy_path: path of the csv file for storing energy of each structures (metal, ligand, pairs)
            bde_path: path of the csv file for storing bond dissociation energy of each pairs
        """

        def __init__(self, work_dir: Union[str, os.PathLike]):
            if isinstance(work_dir, str):
                work_dir = Path(work_dir)

            self.work_dir = work_dir
            self.chk_dir = work_dir.joinpath('chk')
            self.log_dir = work_dir.joinpath('log')
            self.err_dir = work_dir.joinpath('err')
            self.struct_dir = work_dir.joinpath('struct')
            self.energy_path = work_dir.joinpath('energy.csv')
            self.bde_path = work_dir.joinpath('bde.csv')

        @property
        def ligand_chk_path(self):
            return self.chk_dir.joinpath('ligand.chk')

        @property
        def ligand_log_path(self):
            return self.log_dir.joinpath('ligand.log')

        @property
        def ligand_err_path(self):
            return self.err_dir.joinpath('ligand.err')

        @property
        def ligand_struct_path(self):
            return self.struct_dir.joinpath('ligand.mol2')

        def make_dirs(self):
            """ Check and make dirs """
            if not self.work_dir.exists():
                self.work_dir.mkdir()
            if not self.log_dir.exists():
                self.log_dir.mkdir()
            if not self.err_dir.exists():
                self.err_dir.mkdir()
            if not self.struct_dir.exists():
                self.struct_dir.mkdir()

        @property
        def metal_chk_path(self):
            return self.chk_dir.joinpath('metal.log')

        @property
        def metal_log_path(self):
            return self.log_dir.joinpath('metal.log')

        @property
        def metal_err_path(self):
            return self.err_dir.joinpath('metal.err')

        @property
        def metal_struct_path(self):
            return self.struct_dir.joinpath('metal.mol2')

        def pair_chk_path(self, idx: int):
            return self.chk_dir.joinpath(f'pair_{idx}.chk')

        def pair_log_path(self, idx: int):
            return self.log_dir.joinpath(f'pair_{idx}.log')

        def pair_err_path(self, idx: int):
            return self.err_dir.joinpath(f'pair_{idx}.err')

        def pair_struct_path(self, idx: int):
            return self.struct_dir.joinpath(f'pair_{idx}.mol2')

    def __init__(self, metal: Atom, ligand: Molecule, pairs: Sequence[Molecule]):
        self.metal = Molecule()
        self.metal.add_atom(metal)
        self.metal.identifier = metal.symbol

        self.ligand = ligand

        super(PairBundle, self).__init__(pairs)

    @classmethod
    def _specify_dir_files(cls, work_dir: Union[str, os.PathLike]):
        return cls.DirsFiles(work_dir)

    @property
    def pairs(self):
        return self.mols

    def determine_metal_ligand_bind_energy(
            self, g16root: Union[str, os.PathLike], work_dir: Union[str, os.PathLike],
            method: str = 'B3LYP', basis_set: str = '6-311', route: str = ''
    ) -> pd.DataFrame:
        # Specify directories and make these dirs
        dirs_files = self._specify_dir_files(work_dir)
        dirs_files.make_dirs()

        # Energy sheet, bond dissociation energy (BDE) sheet
        e_sheet, bde_sheet = [], []

        # save the initialized ligand structure
        self.ligand.writefile('mol2', dirs_files.ligand_struct_path)

        # optimize the configure of ligand and calculate their total energy after optimization
        self.ligand.gaussian(
            g16root,
            link0=[
                f'CPU=0-{machine.take_CPUs()-1}',
                f'Mem={machine.take_memory(0.25)}GB'
            ],
            route=f'opt=recalc=5 {method}/{basis_set} ' + route,
            path_log_file=dirs_files.ligand_log_path,
            path_err_file=dirs_files.ligand_err_path,
            inplace_attrs=True
        )

        ligand_energy = self.ligand.energy  # Retrieve the energy after optimizing the conformer
        e_sheet.append(['ligand', self.ligand.smiles, ligand_energy])

        # save the optimized structures
        self.ligand.writefile('mol2', dirs_files.ligand_struct_path)

        # Calculate the single point (sp) energy for metal
        try:
            metal_sp = _atom_single_point[self.metal.atoms[0].symbol][method][basis_set]
        except KeyError:
            self.metal.gaussian(
                g16root,
                link0=[
                    f'CPU=0-{machine.take_CPUs()-1}',
                    f'Mem={machine.take_memory(0.25)}GB'
                ],
                route=f'{method}/{basis_set} ' + route,
                path_log_file=dirs_files.metal_log_path,
                path_err_file=dirs_files.ligand_err_path,
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
        self.metal.writefile('mol2', dirs_files.metal_struct_path)

        # Optimizing the conformer of metal-ligands pairs and Retrieve the energies in the last stable conformer
        for i, pair in enumerate(self.pairs):
            # Save initialized Metal-ligand pair struct
            pair.writefile('mol2', dirs_files.pair_struct_path(i))

            # Performing calculation
            pair.gaussian(
                g16root,
                link0=[
                    f'CPU=0-{machine.take_CPUs()-1}',
                    f'Mem={machine.take_memory(0.25)}GB'
                ],
                route=f'opt=recalc=5 {method}/{basis_set} ' + route,
                path_log_file=dirs_files.pair_log_path(i),
                path_err_file=dirs_files.pair_err_path(i),
                inplace_attrs=True
            )

            # Append the pairs energy values to energy sheet
            e_sheet.append([f'pair_{i}', pair.smiles, pair.energy])
            bde_sheet.append([f'pair_{i}', pair.smiles, pair.energy - ligand_energy - metal_sp])
            # Save refined Metal-ligand pair struct
            pair.writefile('mol2', dirs_files.pair_struct_path(i))

        # Save the energy sheet to csv
        e_sheet = np.array(e_sheet)
        df = pd.DataFrame(e_sheet[:, 1:], index=e_sheet[:, 0], columns=['smiles', 'Energy(eV)'])
        df.to_csv(dirs_files.energy_path)

        # Save the BDE sheet to csv
        bde_sheet = np.array(bde_sheet)
        df = pd.DataFrame(bde_sheet[:, 1:], index=bde_sheet[:, 0], columns=['smiles', 'BDE(eV)'])
        df.to_csv(dirs_files.bde_path)

        return df

    @classmethod
    def read_2d_bde_results(cls, work_dir: Union[str, os.PathLike]):
        """
        Read Calculated BDE results with the 2d molecular results, restore from SMILES str
        Args:
            work_dir: work dir to calculate the results

        Returns:
            PairBundle
        """
        dirs_files = cls._specify_dir_files(work_dir)

        bde_sheet = pd.read_csv(dirs_files.bde_path, index_col=0)

        mols = []
        for i, row in bde_sheet.iterrows():
            mol = Molecule.read_from(row['smiles'], 'smi')
            mol.bde = row['energy']

        return mols

    @classmethod
    def read_3d_bde_results(cls, work_dir: Union[str, os.PathLike]):
        """
        TODO
        Read calculated BDE results with the 3d molecular results, the 3d structure stored in mol2 file
        Args:
            work_dir:  work dir to calculate the results

        Returns:

        """
        dirs_files = cls._specify_dir_files(work_dir)

    @classmethod
    def read_all_results(cls, work_dir):
        """
        Read all direct calculated result in gaussian log file, include ligand, metal, pair, energy, force, charge,et al
        Args:
            work_dir:  work dir to calculate the results
        Returns:

        """
