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
from hotpot.tanks.qm.gaussian import Gaussian


class TestGaussian(ut.TestCase):
    """"""
    @classmethod
    def setUpClass(cls) -> None:
        print('Test', cls.__class__)

    def setUp(self) -> None:
        print('running test:', self._testMethodName)

    def tearDown(self) -> None:
        print('Normalize terminate test!')

    @ut.skipIf(not os.environ.get('g16root'), "the g16root env is not found")
    def test_run_gaussian(self):
        test_dir = os.path.join(hp.hp_root, '..', 'test', 'output', 'gaussrun')
        if not os.path.exists(test_dir):
            os.mkdir(test_dir)
        os.chdir(test_dir)

        g16root = os.environ.get('g16root')
        if not g16root:
            g16root = os.getcwd()

        mol = hp.Molecule.read_from('CO', 'smi')

        mol.gaussian(
            g16root=g16root,
            link0=["nproc=16", "mem=64GB"],
            route="opt M062X/6-311",
            inplace_attrs=True
        )

    @ut.skipIf(not os.environ.get('g16root'), "the g16root env is not found")
    def test_config_gaussian_from_scratch(self):
        """ test running a gaussian work by call the Gaussian class directly """
        g16root = '/home/pub'
        gauss = Gaussian(g16root)

    def test_options(self):
        g16root = '/home/pub'
        gauss = Gaussian(g16root)

        self.assertTrue(gauss.op.path.is_root, "The option in the Gaussian must be root option!")
        self.assertTrue(not gauss.op, "The empty root option should be False!")

        gauss.op.link0.nproc(48)
        self.assertFalse(not gauss.op, "The root option with item should be True!")
        gauss.op.link0.Mem("256GB")
        gauss.op.link0.rwf('/home/zz1/proj/be/readwrite.rwf')
        gauss.op.link0.NoSave()
        gauss.op.link0.chk('/home/zz1/proj/be/checkpoint.chk')

        gauss.op.route.opt()
        gauss.op.route.method.B3LYP()
        gauss.op.route.basis._6_31G()
        gauss.op.route.method.M062X()
        gauss.op.route.opt.restart()
        gauss.op.route.opt.algorithm.GEDIIS()

        in_dict = gauss.op.get_option_dict()
        route = in_dict['route']
        with self.assertRaises(KeyError):
            v = route['B3LYP']

        self.assertFalse(not gauss.op, "the options should contain many options!!!")
        gauss.op.clear()
        self.assertTrue(not gauss.op, "the options should be empty after calling clear()!!!")

        mol = hp.Molecule.read_from('OC(=O)c1cc(O)ccc1', 'smi')
        script = mol.dump(
            'gjf', link0='nproc=50', route="opt(restart,Cartesian,MaxStep=2) M062X/Def2SVP CBSExtrap=NMin=6"
        )
        gauss.parsed_input = gauss._parse_input_script(script)
        gauss.op.parsed_input_to_options(gauss.parsed_input)
        gauss.op.update_parsed_input(gauss.parsed_input)
        print(gauss.parsed_input)
        self.assertFalse(not gauss.op, "the options should contain many options, after convert the parsed_input into")

        gauss.op.clear()
        self.assertFalse(gauss.op, "There is none option after the Options.clear() is called!")

        route, opt, coordinate, cartesian = gauss.op.path.get_normalize_path('route.opt.coor.Car').split('.')
        self.assertEqual(route, "route")
        self.assertEqual(opt, 'optimization')
        self.assertEqual(coordinate, "Coordinate")
        self.assertEqual(cartesian, "Cartesian")
