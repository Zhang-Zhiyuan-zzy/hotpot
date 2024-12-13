"""
python v3.9.0
@Project: hotpot
@File   : __init__.py
@Auther : Zhiyuan Zhang
@Data   : 2024/8/10
@Time   : 15:46
"""
import os
from copy import copy
import os.path as osp
import pandas as pd


dir_root = osp.join(osp.dirname(__file__))
_accessible_dataset = [name.split('.')[0] for name in os.listdir(osp.join(dir_root, 'data'))]


def get_dataset_name():
    return copy(_accessible_dataset)


def load_dataset(name: str):
    if name not in _accessible_dataset:
        raise ValueError('The dataset {self.name} does not exist, '
                         'call get_dataset_names() to get all available datasets')

    # Find the file name
    suffix = None
    for file in os.listdir(osp.join(dir_root, 'data')):
        if file.split('.')[0] == name:
            suffix = file.split('.')[1]

    if not suffix:
        raise RuntimeError('the suffix of file not was not captured!')

    if suffix == 'xlsx':
        print('pandas.DataFrame')
        return pd.read_excel(osp.join(dir_root, 'data', f'{name}.xlsx'))
    elif suffix == 'txt':
        def _loader():
            with open(osp.join(dir_root, 'data', f'{name}.txt'), 'r') as f:
                while line := f.readline():
                    yield line
        print('txt lines loader')
        return _loader()
