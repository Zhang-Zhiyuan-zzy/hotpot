"""
python v3.9.0
@Project: hotpot
@File   : test_deepmd
@Auther : Zhiyuan Zhang
@Data   : 2023/7/25
@Time   : 18:47

Test the workflows correlating to DeepModeling
"""
import os
import unittest as ut

import numpy as np

from test import test_root
import hotpot as hp
import hotpot.cheminfo as ci
import hotpot.tanks.deepmd as dpmd
from hotpot.tanks.deepmd import read_system


class TestDeepModeling(ut.TestCase):
    """"""
    @classmethod
    def setUpClass(cls) -> None:
        print('Test', cls.__class__)

    def setUp(self) -> None:
        print('running test:', self._testMethodName)

    def test_system_convert(self):
        dir_log = test_root.joinpath('inputs', 'glog')
        dir_sys = test_root.joinpath('output', 'dpmd_sys')

        print(f"read g16 log from {dir_log}")

        _ = hp.MolBundle.read_from('g16log', dir_log)  # read by single processor
        bundle = hp.MolBundle.read_from('g16log', dir_log, nproc=4)  # read by multiply processor

        print(dir_log.absolute())
        print(os.listdir(dir_log))
        self.assertEqual(len(bundle), 15)

        bundle = bundle.to('DeepModelBundle')

        # To get standard DeePMD-kit system
        bundle_m = bundle.merge_conformers()
        self.assertEqual(len(bundle_m), 11)
        for mol in bundle_m:
            self.assertIsInstance(mol, ci.Molecule)
            system = mol.dump('dpmd_sys')
            self.assertIsInstance(system, dpmd.DeepSystem)

            coords = system.coord
            types = system.type

            self.assertTrue(np.all(mol.atomic_numbers_array == types), "all types is not matched!")
            self.assertTrue(np.all(mol.all_energy == system.energy), "all energy is not matched!")

            for i, c in enumerate(coords):
                mol.conformer_select(i)
                self.assertTrue(np.all(mol.coordinates.flatten() == c), "coordinates is not matched!")

        dir_sys.mkdir(parents=True, exist_ok=True)

        # Save as standard system
        dir_std_sys = dir_sys.joinpath('std')
        bundle.to_dpmd_sys(dir_std_sys, 0.5, mode='std')

        # Save as attention system
        dir_std_sys = dir_sys.joinpath('att')
        bundle.to_dpmd_sys(dir_std_sys, 0.5, mode='att')

    def test_system_load(self):
        """ test load the saved system from 'test_system_convert''"""
        log_dir = test_root.joinpath('inputs', 'glog')
        struct_dir = test_root.joinpath('output', 'mol2_struct')
        img_dir = test_root.joinpath('output', 'img')
        print(img_dir.absolute())

        if not struct_dir.exists():
            struct_dir.mkdir(parents=True)
        if not img_dir.exists():
            img_dir.mkdir(parents=True)

        # Read MultiSystem object
        ms = read_system(log_dir, file_pattern='*.log', fmt="gaussian/md")

        mols = []
        for ls in ms:
            mol = hp.Molecule.build_from_dpdata_system(ls)
            mols.append(mol)

        # Supposed that I want to know the process of breaking and generating of bonds of the first Molecule
        mol = mols[0]
        # Iterating each conformer in the quantum chemistry calculation
        for i in range(mol.conformer_counts):
            mol.conformer_select(i)
            mol.remove_bonds(*mol.bonds)  # Clear all pre-build bonds
            mol.build_bonds()  # rebuild bonds according to the point cloud of atoms
            mol.assign_bond_types()

            # Save the 3D mol structure with built bonds to mol2 file
            mol.writefile('mol2', struct_dir.joinpath(f"{i}.mol2"))
            mol.save_2d_img(img_dir.joinpath(f'{i}.png'))  # Save the 2d img structure to png file




