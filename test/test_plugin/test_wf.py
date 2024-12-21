"""
python v3.9.0
@Project: hotpot
@File   : test_wf
@Auther : Zhiyuan Zhang
@Data   : 2024/9/12
@Time   : 20:55
"""
import logging
from pathlib import Path
import unittest as ut
import shutil

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from hotpot.plugins.ml.wf import MachineLearning_
from hotpot.dataset import load_dataset

logging.basicConfig(level=logging.DEBUG)
class TestWf(ut.TestCase):
    def test_ml_(self):
        from hotpot.plugins.ml.wf import MachineLearning_
        from hotpot.dataset import load_dataset

        work_dir = Path(__file__).parents[1].joinpath('outdir', 'wf')
        if work_dir.exists():
            shutil.rmtree(work_dir)

        data = load_dataset('logβ1')

        wf = MachineLearning_(
            work_dir=work_dir,
            data=data,
            features=data.columns.tolist()[:-1],
            target=data.columns.tolist()[-1],
            xscaler=MinMaxScaler(),
        )

        wf.work()
        # wf.partial_dependence(recover_to_original_scale=True)
        wf.train_surrogate_tree()

    def test_wf_skip_some_feature_engineering_procedures(self):
        data_path = Path(__file__).parents[2].joinpath('hotpot/dataset/ChemData/examples/logβ.xlsx')

        with pd.ExcelFile(data_path) as excel:
            data = excel.parse(sheet_name='ChemData', index_col=0)
            test_indices = excel.parse(sheet_name='test_indices', index_col=0)

        essential_features = ['dG', 'groups', 'c1', 'SMR_VSA1', 'Med(MP)', 'Med(c)']
        # essential_features = ['SMR_VSA1', 'BCUT2D_MRHI', 'groups', 'dG', 'Med(MP)', 'BCUT2D_LOGPHI']
        work_dir = Path(__file__).parents[1].joinpath('outdir', 'wf_miss_fe')
        if work_dir.exists():
            shutil.rmtree(work_dir)

        wf = MachineLearning_(
            work_dir=work_dir,
            data=data,
            features=data.columns.tolist()[:-1],
            target=data.columns.tolist()[-1],
            # xscaler=MinMaxScaler(),
            essential_features=essential_features,
            test_indices=test_indices
        )

        wf.work()
        wf.partial_dependence(recover_to_original_scale=True)

