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
    def test_run_gaussian(self):
        if os.environ.get('g16root'):
            test_dir = os.path.join(hp.hp_root, '..', 'test', 'output', 'gaussrun')
            if not os.path.exists(test_dir):
                os.mkdir(test_dir)
            os.chdir(test_dir)

            g16root = "/home/pub"

            mol = hp.Molecule.read_from('c1cc2(O[Fe+3]O2)(N)ccc1', 'smi')
            mol.build_3d()

            mol.gaussian(
                g16root=g16root,
                link0=["nproc=16", "mem=64GB"],
                route="opt M062X/6-311",
                inplace_attrs=True
            )
