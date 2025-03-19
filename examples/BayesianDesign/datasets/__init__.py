import glob
import os
import os.path as osp
import shutil
import random
from typing import Iterable, Optional, Protocol, Type, Union
import socket

import torch
from torch_geometric.data import Data


def torch_load_data(p: str) -> Data:
    if torch.__version__ >= '2.6':
        return torch.load(p, weights_only=False)
    else:
        return torch.load(p)


class DatasetProtocol(Protocol):
    def __init__(self, root: str):
        ...
    def __getitem__(self, idx: int) -> Data:
        ...
    def __len__(self) -> int:
        ...
    def __iter__(self) -> Iterable[Data]:
        ...
    def get(self, idx: int) -> Data:
        ...


# Initialize paths.
machine_name = socket.gethostname()
if machine_name == '4090':
    project_root = '/home/zzy/proj/bayes'
elif machine_name == 'DESKTOP-TZ001':
    project_root = '/mnt/d/1-hnh/Data'
elif machine_name == 'docker':
    project_root = '/app/proj'
elif machine_name == '3090':
    project_root = '/home/hnh/proj'
else:
    raise ValueError


class PretrainDataset:
    def __init__(self, root: str, in_mem_size: int = 20000) -> None:
        self.root = root
        self.files = glob.glob(osp.join(root, '*.pt'))
        self.len = len(self.files)
        self.in_mem_size = in_mem_size

        if self.len < self.in_mem_size:
            self.InMemory = True
        else:
            self.InMemory = False

        self.list_data = []

    def __len__(self):
        return self.len

    def __getitem__(self, idx: int):
        if self.InMemory:
            if self.len > idx >= len(self.list_data):
                for i in range(len(self.list_data), idx+1):
                    self.list_data.append(torch_load_data(self.files[i]))

            return self.list_data[idx]
        else:
            return torch_load_data(self.files[idx])

    def __iter__(self):
        for i in range(self.len):
            yield self[i]



class DatasetGetter:
    __items__ = {
        'x_input': ('atomic_number', 'n', 's', 'p', 'd', 'f', 'g', 'x', 'y', 'z'),
        'x': ('atomic_number', 'partial_charge'),
        'pair_attr': ('length_shortest_path', 'wiberg_bond_order'),
        'ring_attr': ('is_aromatic',)
    }

    def __init__(
            self, proj_root,
            ds_name: str,
            items: dict[str, Iterable[str]] = None,
            test_ratio: float = 0.1,
    ):
        self.items = items if isinstance(items, dict) else self.__items__
        assert 0 < test_ratio < 1
        self.test_ratio = test_ratio

        self.proj_root = proj_root
        self.ds_root = osp.join(proj_root, "datasets", ds_name)
        self.train_dir = osp.join(proj_root, "datasets", ds_name, "train")
        self.test_dir = osp.join(proj_root, "datasets", ds_name, "test")

        self.split_train_test()

        # self.dir_InMemory = osp.join(self.proj_root, DatasetGetter.__dataset_name__, 'InMemory')

        self.train_dataset = PretrainDataset(self.train_dir)
        self.test_dataset = PretrainDataset(self.test_dir)

        self.first_data: Data = self.train_dataset[0]

    def split_train_test(self):
        if not osp.exists(self.train_dir):
            os.mkdir(self.train_dir)
        if not osp.exists(self.test_dir):
            os.mkdir(self.test_dir)

        for p in glob.glob(osp.join(self.ds_root, '*.pt')):
            if random.uniform(0, 1) < self.test_ratio:
                shutil.move(p, osp.join(self.test_dir, osp.basename(p)))
            else:
                shutil.move(p, osp.join(self.train_dir, osp.basename(p)))

    def get_index(self, data_item: str, attrs: Union[str, Iterable[str]] = None) -> Union[int, list[int]]:
        item_names = self.first_data[f"{data_item}_names"]
        if attrs is None:
            return list(range(len(item_names)))
        elif isinstance(attrs, str):
            return item_names.index(attrs)
        elif isinstance(attrs, Iterable):
            return [item_names.index(a) for a in attrs]

    def get_y_attrs(self):
        return self.first_data.y_names

    def get_datasets(self):
        return self.train_dataset, self.test_dataset
