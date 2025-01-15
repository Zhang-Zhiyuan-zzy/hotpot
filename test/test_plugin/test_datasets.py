"""
python v3.9.0
@Project: hotpot
@File   : test_datasets
@Auther : Zhiyuan Zhang
@Data   : 2024/12/27
@Time   : 11:16
"""
import unittest as ut
from hotpot.dataset.tmqm import TmQmDataset


class TestDataset(ut.TestCase):
    def test_tmqm(self):
        dataset = TmQmDataset()
        print(dataset)
