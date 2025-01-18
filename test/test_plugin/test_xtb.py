import os.path as osp
import unittest
import unittest as ut

import hotpot as hp
from hotpot.plugins.xtb import core
from test import output_dir, input_dir

class TestXTB(unittest.TestCase):
    def test_xtb(self):
        xtb_calculator = core.XtbCalculator(osp.join(output_dir, 'xtb'))
        mol = next(hp.MolReader('c1ccccc1C(=O)O[Sr]'))
        mol.build3d()

        xtb_calculator.mol = mol
        xtb_calculator.set_mol_charge_unpairEs()

        print(xtb_calculator.charge, xtb_calculator.unpair)

        xtb_calculator.set_opt()

        stdout= xtb_calculator.run()

    def test_batch_run(self):
        mol_file_dir = osp.join(input_dir, 'xtb_mol2')
        res_file_dir = osp.join(output_dir, 'xtb')

        core.xtb_batch_run(
            mol_file_dir=mol_file_dir,
            res_file_dir=res_file_dir,
        )

