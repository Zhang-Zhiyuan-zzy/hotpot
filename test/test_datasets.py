"""
python v3.9.0
@Project: hotpot
@File   : test_datasets
@Auther : Zhiyuan Zhang
@Data   : 2025/1/3
@Time   : 10:07
"""
import os
import shutil
import os.path as osp
import unittest as ut

from torch_geometric.loader import DataLoader

from hotpot.dataset.tmqm import TmQmDataset
from hotpot.plugins.complex_model import data as pyg_data
from test import test_dir
test_out_dir = osp.join(test_dir, 'outdir')


class TestDatasets(ut.TestCase):
    def test_tmqm_mol(self):
        dataset_dir = osp.join(test_out_dir, 'tmqm', 'mol')
        if not osp.exists(dataset_dir):
            # shutil.rmtree(dataset_dir)
            os.mkdir(dataset_dir)

        tmD = TmQmDataset()
        for i, mol in enumerate(tmD):
            print(mol.identifier, sum(a.partial_charge for a in mol.atoms))
            mol.write(osp.join(test_out_dir, 'tmqm', 'mol', f'{mol.identifier}.mol2'), overwrite=True)
            if i > 100:
                break

    def test_tmqm_data(self):
        dataset_dir = osp.join(test_out_dir, 'tmqm', 'data')
        if not osp.exists(dataset_dir):
            shutil.rmtree(dataset_dir)
            os.mkdir(dataset_dir)

        dataset = pyg_data.tmQmDataset(osp.join(test_out_dir, 'tmqm', 'data'), test_num=1024)
        loader = iter(DataLoader(dataset, batch_size=256))
        for batch in loader:
            print(batch.y)
