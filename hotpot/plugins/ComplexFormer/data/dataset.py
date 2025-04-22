# Author: Zhiyuan Zhang
import glob
import os
import os.path as osp
import shutil
import random
import bisect

from typing import Iterable, Optional, Protocol, Union, Sequence
from typing_extensions import override

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, ConcatDataset
from torch_geometric.data import Data
from torch_geometric.loader.dataloader import Collater, DataLoader

import lightning as L


def _is_int_like(a):
    return isinstance(a, (int, np.integer)) or (
        torch.is_tensor(a) and a.numel() == 1 and a.dtype in (torch.int8, torch.int16, torch.int32, torch.int64))

def torch_load_data(p: str) -> Data:
    if torch.__version__ >= '2.6':
        return torch.load(p, weights_only=False)
    else:
        return torch.load(p)

def load_data(dataset: Union[Iterable[Data], Sequence[Data]], sample_num: int = None):
    lst_data = []
    total = len(dataset) if sample_num is None else min(sample_num, len(dataset))
    if isinstance(dataset, Iterable):
        for i, data in enumerate(tqdm(dataset, 'Loading data', total=total)):
            if isinstance(sample_num, int) and i >= sample_num:
                break
            lst_data.append(data)
    elif isinstance(dataset, Sequence):
        for i in tqdm(range(len(dataset)), 'Loading data', total=total):
            if isinstance(sample_num, int) and i >= sample_num:
                break
            lst_data.append(dataset[i])
    else:
        raise TypeError("dataset must be an instance of Iterable[Data] or Sequence[Data]")

    return DataWrapper(lst_data)


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


class DataWrapper(Dataset):
    def __init__(self, data_iter: Iterable[Data]) -> object:
        self.data_iter = data_iter

    def __getitem__(self, idx: Union[int, slice]) -> Data:
        if not isinstance(self.data_iter, list):
            self.data_iter = list(self.data_iter)
        return self.data_iter[idx]

    def __len__(self) -> int:
        try:
            return len(self.data_iter)
        except TypeError:
            data_iter = list(self.data_iter)
            return len(data_iter)

    def __iter__(self) -> Iterable[Data]:
        return iter(self.data_iter)


class PretrainDataset(Dataset):
    def __init__(
            self,
            root: str,
            InMenory: bool = None,
            in_mem_size: int = 2000000,
            autoload: bool = False
    ) -> None:
        self.root = root
        self.files = glob.glob(osp.join(root, '*.pt'))
        self.len = len(self.files)
        self.in_mem_size = in_mem_size

        if InMenory is None:
            if self.len < self.in_mem_size:
                self.InMemory = True
            else:
                self.InMemory = False
        elif isinstance(InMenory, bool):
            self.InMemory = InMenory
        else:
            raise TypeError('InMenory should be bool or None')

        # self.iter_bar = tqdm(total=self.len, desc='Loading data', leave=False)
        self.list_data = []

        if self.InMemory and autoload:
            self.load_data()

    def __repr__(self):
        return f'{self.__class__.__name__}({self.len})'

    def __len__(self):
        return self.len

    def __getitem__(self, idx: int):
        if self.InMemory:
            if self.len > idx >= len(self.list_data):
                self.load_data(idx+1)

            return self.list_data[idx]
        else:
            return torch_load_data(self.files[idx])

    def __iter__(self):
        for i in range(self.len):
            yield self[i]

    def load_data(self, sample_num: Optional[int] = None) -> Iterable[Data]:
        if sample_num is None:
            sample_num = self.len
        else:
            sample_num = min(sample_num, self.len)

        if len(self.list_data) < sample_num:
            for i in range(len(self.list_data), sample_num):
                self.list_data.append(torch_load_data(self.files[i]))
                # self.iter_bar.update()

        return self.list_data


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

        self.train_dataset = None
        self.test_dataset = None

        # self.first_data: Data = self.train_dataset[0]

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

    def get_datasets(self, InMenory: bool = None, in_mem_size: int = 2000000):
        print(f"Train dir: {self.train_dir}")
        print(f"Test dir: {self.test_dir}")
        if self.train_dataset is None:
            self.train_dataset = PretrainDataset(self.train_dir, InMenory=InMenory, in_mem_size=in_mem_size)
        if self.test_dataset is None:
            self.test_dataset = PretrainDataset(self.test_dir, InMenory=InMenory, in_mem_size=in_mem_size)

        return self.train_dataset, self.test_dataset

    @property
    def first_data(self) -> Data:
        if isinstance(self.train_dataset, PretrainDataset):
            return self.train_dataset[0]
        elif isinstance(self.test_dataset, PretrainDataset):
            return self.test_dataset[0]
        else:
            train_data_files = glob.glob((osp.join(self.train_dir, '*.pt')))
            return torch_load_data(train_data_files[0])


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
    ) -> None:
        self.datasets = datasets
        self.ds_names = ds_names

    # def _refresh_datasets(self):
    #     self.ds_indices = [
    #         np.concatenate(
    #             [np.arange(len(ds)), np.random.randint(len(ds), size=len(ds) % self.batch_size)],
    #             axis=0)
    #         for ds in self.datasets]
    #
    #     if self.shuffle:
    #         for ds_idx in range(len(self.ds_indices)):
    #             np.random.shuffle(self.ds_indices[ds_idx])
    #
    #     _indices = []
    #     for i, ds_idx in enumerate(self.ds_indices):
    #         np.random.shuffle(ds_idx)
    #         for batch_idx in np.split(ds_idx, len(ds_idx) // self.batch_size):
    #             _indices.append((i, batch_idx))
    #
    #     self.ds_indices = _indices
    #
    #     if self.shuffle:
    #         np.random.shuffle(self.ds_indices)

    def __len__(self) -> int:
        return sum(map(len, self.datasets))

    def __getitem__(self, dataset_idx, sample_idx) -> Data:
        return self.datasets[dataset_idx][sample_idx]

class MConcatDataset(ConcatDataset):
    def __repr__(self):
        return f'{self.__class__.__name__}(dataset={len(self.datasets)}, data={len(self)})'

    # def get_dataset_index(self, sample_idx: int) -> int:
    #     return

    def get_with_ds_idx(self, idx):
        data = super().__getitem__(idx)
        data.dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        return data

    def load_data(self, sample_num: Optional[int] = None):
        self.datasets = [load_data(dataset, sample_num) for dataset in self.datasets]
        self.cumulative_sizes = self.cumsum(self.datasets)
        return self

    def __getitem__(self, idx: Union[int, slice]) -> Data:
        if _is_int_like(idx):
            return self.get_with_ds_idx(idx)
        elif isinstance(idx, slice):
            return [self.get_with_ds_idx(i) for i in range(*idx.indices(idx.stop))]
        else:
            raise TypeError('idx to subscript the MConcatDataset must be int or slice')


class OnFlyLoadingDataset(Dataset):
    def __init__(self, data_files: list['str']):
        self.data_files = data_files
        self.data = {}

        # self.p_bat = tqdm(total=len(self.data_files), desc='loading data')

    def __len__(self) -> int:
        return len(self.data_files)

    def __getitem__(self, idx: int) -> Data:
        try:
            return self.data[idx]
        except KeyError:
            self.data[idx] = torch_load_data(self.data_files[idx])
            # self.p_bat.update(1)
            return self[idx]

    @property
    def data_number(self) -> int:
        return len(self.data)
