"""
python v3.9.0
@Project: hotpot
@File   : __init__.py
@Auther : Zhiyuan Zhang
@Data   : 2024/12/24
@Time   : 15:33
"""
import os
from os import path
from os.path import join as opj
from typing import Union
from . import download


_module_root = path.dirname(__file__)
_data_dir = opj(_module_root, 'data')


# 读取tmQm的分子信息，输出Molecule.
class TmQmDataset:
    _dataset_files = ('tmQm.xyz', 'tmQm.q', 'tmQm.csv', 'tmQm.BO')
    def __init__(self, dataset_file: Union[str, os.PathLike] = None):
        """"""
        if not dataset_file:
            dataset_file = _data_dir
        else:
            dataset_file = str(dataset_file)



        if not path.exists(_data_dir):
            os.mkdir(_data_dir)
            download.download_files()

        else:
            list_data = os.listdir(_data_dir)
            for ext, url in download.urls.items():
                if f'tmQm.{ext}' not in list_data:
                    file_path = opj(_data_dir, f'tmQm.{ext}.gz')
                    download.download_file(file_path, url)
                    download.uncompress_all_gz(file_path)
                    os.remove(file_path)  # Remove the gz file

                else:
                    print(f"tmQm.{ext} already exist !!")

        self.xyz = ...
        self.q = ...
        self.csv = ...
        self.bo = ...

    def __iter__(self):
        """"""

    def __next__(self):
        """"""

