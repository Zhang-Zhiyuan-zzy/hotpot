"""
python v3.9.0
@Project: hotpot
@File   : test_quantum
@Auther : Zhiyuan Zhang
@Data   : 2023/7/19
@Time   : 22:08
"""
import os
import unittest as ut
import hotpot as hp


class TestGaussian(ut.TestCase):
    """"""
    @classmethod
    def setUpClass(cls) -> None:
        print('Test', cls.__class__)

    def setUp(self) -> None:
        print('running test:', self._testMethodName)

    @ut.skipIf(not os.environ.get('g16root'), "the g16root env is not found")
    def test_run_gaussian(self):
        test_dir = os.path.join(hp.hp_root, '..', 'test', 'output', 'gaussrun')
        if not os.path.exists(test_dir):
            os.mkdir(test_dir)
        os.chdir(test_dir)

        g16root = os.environ.get('g16root')

        mol = hp.Molecule.read_from('c1cc2(O[Fe+3]O2)(N)ccc1', 'smi')
        mol.build_3d()

        mol.gaussian(
            g16root=g16root,
            link0=["nproc=16", "mem=64GB"],
            route="opt M062X/6-311",
            inplace_attrs=True
        )
