# Author: Zhiyuan Zhang
import bisect
from typing import Iterable, Optional,Union, Sequence

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, ConcatDataset
from torch_geometric.data import Data


def _is_int_like(a):
    return isinstance(a, (int, np.integer)) or (
        torch.is_tensor(a) and a.numel() == 1 and a.dtype in (torch.int8, torch.int16, torch.int32, torch.int64))

def torch_load_data(p: str) -> Data:
    try:
        if torch.__version__ >= '2.6':
            return torch.load(p, weights_only=False)
        else:
            return torch.load(p)
    except Exception as e:
        print(p)
        raise e

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

class MConcatDataset(ConcatDataset):
    def __repr__(self):
        return f'{self.__class__.__name__}(dataset={len(self.datasets)}, data={len(self)})'

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


class PathStoredDataset(Dataset):
    def __init__(self, data_files: list['str']):
        self.data_files = data_files

    def __len__(self) -> int:
        return len(self.data_files)

    def __getitem__(self, idx: int) -> Data:
        return torch_load_data(self.data_files[idx])
