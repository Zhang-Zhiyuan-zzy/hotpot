"""
python v3.9.0
@Project: hotpot
@File   : test_cheminfo
@Auther : Zhiyuan Zhang
@Data   : 2023/7/16
@Time   : 22:21
Notes:
    Test `hotpot/cheminfo` module
"""
from pathlib import Path
import unittest as ut
import hotpot as hp


class TestMolecule(ut.TestCase):
    """ Test `hotpot/cheminfo/Molecule` class """
    def test_read_from(self):
        """ test the `read_from` method """
        mol_path = Path(hp.hp_root).joinpath('..', 'test', 'inputs', 'struct', 'abnormal_output.log')
        mol_ab16log = hp.Molecule.read_from(mol_path, 'g16log', force=True)

        self.assertIsInstance(mol_ab16log, hp.Molecule)
        self.assertTrue(mol_ab16log.has_3d)
        self.assertGreater(mol_ab16log.conformer_counts, 1)
