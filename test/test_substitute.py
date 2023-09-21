"""
python v3.9.0
@Project: hotpot
@File   : test_substitute
@Auther : Zhiyuan Zhang
@Data   : 2023/9/20
@Time   : 11:17
"""
import logging
import unittest as ut

import hotpot as hp
from hotpot import substitute


logging.basicConfig(level=logging.INFO)


class TestSubstitute(ut.TestCase):
    """"""
    @classmethod
    def setUpClass(cls) -> None:
        print('Test', cls.__class__)

    def setUp(self) -> None:
        print('running test:', self._testMethodName)

    def tearDown(self) -> None:
        print('Normalize terminate test!', self._testMethodName)

    def test_substitute(self):
        """"""
        subst = substitute.EdgeJoinSubstituent(
            "pyrrol_0_4",
            hp.Molecule.read_from('C1=CNC=C1', 'smi'),
            [0, 4], "cc"
        )

        benzene = hp.Molecule.read_from("c1cnccc1", 'smi')

        generate_mol = subst(benzene)
        print(generate_mol)

        for i, mol in enumerate(generate_mol):
            mol.add_hydrogens()
            mol.build_3d()

            if not mol.has_nan_coordinates:
                print(mol.smiles)

        subst.writefile(substitute.substituent_root)

        # Reload the saved substitute



