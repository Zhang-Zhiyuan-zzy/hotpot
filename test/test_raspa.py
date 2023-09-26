"""
python v3.9.0
@Project: hotpot
@File   : test_raspa
@Auther : Zhiyuan Zhang
@Data   : 2023/9/25
@Time   : 15:29
"""
import json
import unittest as ut

import test

import hotpot as hp
from hotpot.tasks.raspa import RASPA


class TestRaspa(ut.TestCase):
    """ Test the hotpot.tasks.raspa subpackage """
    @classmethod
    def setUpClass(cls) -> None:
        print('Test', cls.__class__)

    def setUp(self) -> None:
        print('running test:', self._testMethodName)

    def tearDown(self) -> None:
        print('Normalize terminate test!', self._testMethodName)

    def test_run(self):
        """"""
        mof_name = "IRMOF-1"
        path_mof = test.test_root.joinpath("inputs", "struct", "IRMOF-1.cif")
        work_dir = test.test_root.joinpath("output", 'raspa', "IRMOF-1")

        mof = hp.Molecule.read_from(path_mof)

        raspa = RASPA(work_dir, parsed_output=False)

        script = raspa.run(mof, "CO2", cycles=100000)
        # json.dump(script, open(work_dir.joinpath("output.json"), 'w'), indent=True)

        # work_dir.mkdir(parents=True, exist_ok=True)
        with open(work_dir.joinpath("pure"), 'w') as writer:
            writer.write(script)

