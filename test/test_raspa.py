"""
python v3.9.0
@Project: hotpot
@File   : test_raspa
@Auther : Zhiyuan Zhang
@Data   : 2023/9/25
@Time   : 15:29
"""
import os
import unittest as ut

import test

import hotpot as hp
from hotpot.plugins.raspa import RASPA

raspa_root = hp.settings.get("paths", {}).get("raspa_root") or os.environ.get('RASPA_DIR')


class TestRaspa(ut.TestCase):
    """ Test the hotpot.plugins.raspa subpackage """
    @classmethod
    def setUpClass(cls) -> None:
        print('Test', cls.__class__)

    def setUp(self) -> None:
        print('running test:', self._testMethodName)

    def tearDown(self) -> None:
        print('Normalize terminate test!', self._testMethodName)

    @ut.skipIf(not raspa_root, "the test need the Raspa software")
    def test_run(self):
        """"""
        mof_name = "IRMOF-1"
        path_mof = test.test_root.joinpath("inputs", "struct", f"{mof_name}.cif")
        work_dir = test.test_root.joinpath("output", 'raspa', "IRMOF-1")

        mof = hp.Molecule.read_from(path_mof)

        raspa = RASPA()

        result = raspa.run(mof, "CO2", "O2", "N2", cycles=10000)

        work_dir.mkdir(parents=True, exist_ok=True)
        with open(work_dir.joinpath("pure"), 'w') as writer:
            writer.write(result.output)

