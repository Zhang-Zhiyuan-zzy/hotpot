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
import hotpot.cheminfo as ci


class TestMolecule(ut.TestCase):
    """ Test `hotpot/cheminfo/Molecule` class """

    @classmethod
    def setUpClass(cls) -> None:
        print('Test', cls.__class__)

    def setUp(self) -> None:
        print('running test:', self._testMethodName)

    def test_read_from(self):
        """ test the `read_from` method """
        mol_path = Path(hp.hp_root).joinpath('..', 'test', 'inputs', 'struct', 'abnormal_output.log')
        mol = hp.Molecule.read_from(mol_path, 'g16log', force=True)

        self.assertIsInstance(mol, hp.Molecule)
        self.assertTrue(mol.has_3d)
        self.assertGreater(mol.conformer_counts, 1)  # the read molecule should have multiply conformers

        # Test the accessibility of Molecule attributes
        self.assertIsInstance(mol.atoms[0], ci.Atom)
        self.assertIsInstance(mol.bonds[0], ci.Bond)
