"""
python v3.9.0
@Project: hotpot
@File   : test_materials_make
@Auther : Zhiyuan Zhang
@Data   : 2023/7/25
@Time   : 13:59
"""
import unittest as ut
import hotpot as hp


class TestMaterialsMaker(ut.TestCase):
    """ Test hotpot/plugins/lmp/materials """

    @classmethod
    def setUpClass(cls) -> None:
        print('Test', cls.__class__)

    def setUp(self) -> None:
        print('running test:', self._testMethodName)

    def test_make_amorphous_crystal(self):
        """ Test hotpot/plugins/lmp/materials/AmorphousMaker class """
        frame = hp.Molecule.create_aCryst_by_mq({"C": 1.0}, "aMaterials/SiC.tersoff", ff_args=("C",), density=0.1)
        self.assertIsInstance(frame, hp.Molecule)
