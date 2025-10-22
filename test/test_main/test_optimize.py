import os
import os.path as osp
import unittest as ut
from unittest.mock import patch
from hotpot.__main__ import build_parser, run, main


_parent_dir = osp.dirname(__file__)
class TestCliOptimize(ut.TestCase):
    """ This test block to test `CLI hotpot optimize ...`"""
    def test_COF_example(self):
        # cmd = [
        #     'optimize',
        #     'somewhere',
        #     osp.join(_parent_dir, 'output', 'optimize'),
        #     '--examples', 'COF'
        # ]
        # self.assertEqual(main(cmd), 0, "Error terminated for hotpot optimize")
        #
        # cmd = [
        #     'optimize',
        #     'somewhere',
        #     osp.join(_parent_dir, 'output', 'optimize'),
        #     '--examples', 'COF',
        #     '--scatter-map'
        # ]
        # self.assertEqual(main(cmd), 0, "Error terminated for hotpot optimize with `--scatter-map`")

        cmd = [
            'optimize',
            'somewhere',
            osp.join(_parent_dir, 'output', 'optimize'),
            '--examples', 'COF',
            '--scatter-map',
            '--cmap', 'BluesSat'
        ]
        self.assertEqual(main(cmd), 0, "Error terminated for hotpot optimize with `--scatter-map`, `--cmap BluesSat`")