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

from hotpot.plugins.qm.gaussian import Gaussian

g16root = hp.settings.get("paths", {}).get("g16root") or os.environ.get('g16root')


class TestGaussian(ut.TestCase):
    """"""
    @classmethod
    def setUpClass(cls) -> None:
        print('Test', cls.__class__)

    def setUp(self) -> None:
        print('running test:', self._testMethodName)

    def tearDown(self) -> None:
        print('Normalize terminate test!')

    @ut.skipIf(not g16root, "the g16root env is not found")
    def test_run_gaussian(self):
        test_dir = os.path.join(hp.hp_root, '..', 'test', 'output', 'gaussrun')
        if not os.path.exists(test_dir):
            os.mkdir(test_dir)
        os.chdir(test_dir)

        mol = hp.Molecule.read_from('C', 'smi')

        gjf = mol.dump('gjf', link0=['Mem=128GB', 'nproc=16'], route='opt/freq B3LYP/Def2SVP')

        gauss = Gaussian()
        gauss.run(gjf)

        print([b.length for b in mol.bonds])
        gauss.set_molecule_attrs(mol)
        print([b.length for b in mol.bonds])
