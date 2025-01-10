"""
python v3.9.0
@Project: hotpot
@File   : test_datasets
@Auther : Zhiyuan Zhang
@Data   : 2025/1/3
@Time   : 10:07
"""
from tqdm import tqdm
import unittest as ut
from hotpot.dataset.tmqm import TmQmDataset


class TestDatasets(ut.TestCase):
    def test_tmqm(self):
        dataset = TmQmDataset()
        for mol in tqdm(dataset):
            pass


