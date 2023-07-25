"""
python v3.9.0
@Project: hotpot
@File   : test_deepmd
@Auther : Zhiyuan Zhang
@Data   : 2023/7/25
@Time   : 18:47

Test the workflows correlating to DeepModeling
"""
import unittest as ut
from pathlib import Path

import numpy as np

import hotpot as hp
import hotpot.cheminfo as ci
import hotpot.tanks.deepmd as dpmd


class TestDeepModeling(ut.TestCase):
    """"""
    @classmethod
    def setUpClass(cls) -> None:
        print('Test', cls.__class__)

    def setUp(self) -> None:
        print('running test:', self._testMethodName)

    def test_system_convert(self):
        dir_log = Path(hp.hp_root).joinpath('..', 'test', 'inputs', 'glog')
        dir_sys = Path(hp.hp_root).joinpath('..', 'test', 'output', 'dpmd_sys')

        _ = hp.MolBundle.read_from('g16log', dir_log)  # read by single processor
        bundle = hp.MolBundle.read_from('g16log', dir_log, nproc=4)  # read by multiply processor
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

        dir_sys.mkdir()

        # Save as standard system
        dir_std_sys = dir_sys.joinpath('std')
        bundle.to_dpmd_sys(dir_std_sys, 0.5, mode='std')

        # Save as attention system
        dir_std_sys = dir_sys.joinpath('att')
        bundle.to_dpmd_sys(dir_std_sys, 0.5, mode='att')

    def test_system_load(self):
        """ test load the saved system from 'test_system_convert''"""





