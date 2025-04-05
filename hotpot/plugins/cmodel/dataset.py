# Author: Zhiyuan Zhang
import glob
import os
import os.path as osp
import shutil
import random
from typing import Iterable, Optional, Protocol, Type, Union

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader.dataloader import Collater, DataLoader

import lightning as L


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

class PretrainDataset(Dataset):
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

    def load_all(self, sample_num: Optional[int] = None) -> Iterable[Data]:
        if sample_num:
            return [self[i] for i in tqdm(range(min(len(self), sample_num)))]
        else:
            return [self[i] for i in tqdm(range(len(self)), desc="Loading data")]



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

        if not osp.exists(self.train_dir):
            raise FileNotFoundError(f'Dataset dir {self.train_dir} does not exist')

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
        print(f"Train dir: {self.train_dir}")
        print(f"Test dir: {self.test_dir}")
        return self.train_dataset, self.test_dataset


class MultiDataset(Dataset):
    def __init__(
            self,
            *datasets,
            ds_names: Optional[Iterable[str]] = None,
            batch_size: Optional[int] = None,
            shuffle: bool = True,
    ) -> None:
        self.datasets = datasets
        self.ds_names = ds_names

        self.batch_size = batch_size
        self.shuffle = shuffle

        self._init_datasets()

    def _get_data(self) -> Data:
        return self.datasets[self.dataset_index][self._dataset_counts[self.dataset_index]]

    def _get_random_data(self) -> Data:
        return self.datasets[self.dataset_index][random.randint(0, self._datasets_len[self.dataset_index])]

    def __len__(self) -> int:
        return self._len

    def __iter__(self) -> Iterable[Data]:
        while self._total_idx < self._len:
            if self.shuffle:
                if self._batch_counts >= self.batch_size:
                    if self._dataset_counts[self.dataset_index] >= self._datasets_len[self.dataset_index]:
                        self.active_datasets.remove(self.dataset_index)

                    self.dataset_index = random.choice(self.active_datasets)
                    self._batch_counts = 0

                if self._dataset_counts[self.dataset_index] < self._datasets_len[self.dataset_index]:
                    yield self.dataset_index, self._get_data()
                    self._dataset_counts[self.dataset_index] += 1
                    self._total_idx += 1
                else:
                    # Randomly yield one of data in the dataset as supplement
                    yield self.dataset_index, self._get_random_data()

            else:
                if self._dataset_counts[self.dataset_index] >= self._datasets_len[self.dataset_index]:
                    self.active_datasets.remove(self.dataset_index)
                    self.dataset_index += 1

                yield self.dataset_index, self._get_data()
                self._dataset_counts[self.dataset_index] += 1
                self._total_idx += 1

        self._init_datasets()

    def __next__(self):
        return next(iter(self))

    def __getitem__(self, idx: int) -> Data:
        return next(self)

    def _init_datasets(self):
        self.active_datasets = list(range(len(self.datasets)))
        self.dataset_index = random.choice(self.active_datasets) if self.shuffle else 0
        self._batch_counts = 0

        self._datasets_len = [len(ds) for ds in self.datasets]
        self._dataset_counts = [0] * len(self.datasets)

        self._len = sum(map(len, self.datasets))
        self._total_idx = 0

    @property
    def num_datasets(self) -> int:
        return len(self.datasets)

class MDataset:
    def __init__(
            self,
            *datasets,
            ds_names: Optional[Iterable[str]] = None,
            batch_size: Optional[int] = None,
            shuffle: bool = True,
    ) -> None:
        self.datasets = datasets
        self.ds_names = ds_names

        self.batch_size = batch_size
        self.shuffle = shuffle

        self._len = None
        self.ds_indices = None

    def refresh_datasets(self):
        self.ds_indices = [
            np.concatenate(
                [np.arange(len(ds)), np.random.randint(len(ds), size=len(ds) % self.batch_size)],
                axis=0)
            for ds in self.datasets]

        if self.shuffle:
            for ds_idx in range(len(self.ds_indices)):
                np.random.shuffle(self.ds_indices[ds_idx])

        _indices = []
        for i, ds_idx in enumerate(self.ds_indices):
            np.random.shuffle(ds_idx)
            for batch_idx in np.split(ds_idx, len(ds_idx) // self.batch_size):
                _indices.append((i, batch_idx))

        self.ds_indices = _indices

        if self.shuffle:
            np.random.shuffle(self.ds_indices)

    def __len__(self) -> int:
        if self.ds_indices is not None:
            return sum(map(len, self.ds_indices))
        else:
            return sum(map(len, self.datasets))

    def __iter__(self):
        self.refresh_datasets()

        for dsi, sample_idx in self.ds_indices:
            for si in sample_idx:
                yield dsi, self.datasets[dsi][si]

    def __getitem__(self, idx: int) -> Data:
        if not self.ds_indices:
            self.refresh_datasets()

        quo_idx, re_idx = divmod(idx, self.batch_size)
        ds_idx, batch_idx = self.ds_indices[quo_idx]
        data = self.datasets[ds_idx][batch_idx[re_idx]]

        return ds_idx, data

class CDataModule(L.LightningDataModule):
    def __init__(self, *datasets, ds_names: Union[str, Iterable[str]] = None, **kwargs):
        super().__init__()
        self.datasets = list(datasets)
        if isinstance(ds_names, str):
            self.ds_names = [ds_names]
        else:
            self.ds_names = list(ds_names)

        if ds_names:
            assert len(ds_names) == len(self.datasets), f"The length of datasets should equal to ds_name, but {len(ds_names)} != {len(self.datasets)}"
