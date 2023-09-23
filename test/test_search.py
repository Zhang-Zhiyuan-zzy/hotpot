"""
python v3.9.0
@Project: hotpot
@File   : test_search
@Auther : Zhiyuan Zhang
@Data   : 2023/9/13
@Time   : 10:12
"""
import unittest as ut

import hotpot as hp
from hotpot.search import SubstructureSearcher


class TestSearch(ut.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        print('Test', cls.__class__)

    def setUp(self) -> None:
        print('running test:', self._testMethodName)

    def tearDown(self) -> None:
        print('Normalize terminate test!', self._testMethodName)

    def test_substructure_search(self):
        """"""
        mol1 = hp.Molecule.read_from('CC(=O)O', 'smi')
        mol2 = hp.Molecule.read_from('OC(=O)c1ccc(cc1)C(=O)O', 'smi')
        mol3 = hp.Molecule.read_from('OC(=O)CCC(C(=O)O)CC(=O)O', 'smi')

        searcher = SubstructureSearcher('OC=O')

        matched_mol = searcher.search(mol1, mol2, mol3)

        self.assertEqual(len(matched_mol[0]), 1)
        self.assertEqual(len(matched_mol[1]), 2)
        self.assertEqual(len(matched_mol[2]), 3)

        self.assertFalse(
            matched_mol[0].mol is mol1,
            "the Molecule in the MatchedMol should be a copy of the searched Molecule, instead of its self"
        )
