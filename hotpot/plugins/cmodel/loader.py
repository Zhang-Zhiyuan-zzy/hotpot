import logging

import math
from typing import Optional, Union, List, Sequence, Iterable
import warnings
import bisect

import numpy as np

import torch
from torch.utils.data import Dataset, BatchSampler, Sampler
import torch.distributed as dist

from torch_geometric.data import Data
from torch_geometric.data.data import BaseData
from torch_geometric.loader import DataLoader
from torch_geometric.data.datapipes import DatasetAdapter

from .dataset import MConcatDataset


class MDatasetBatchSampler:
    def __init__(self, *datasets, batch_size: int = 1, shuffle: bool = False):
        self.datasets = list(datasets)
        self.batch_size = batch_size
        self.shuffle = shuffle

        self._batch_nums = [math.ceil(len(ds)/self.batch_size) for ds in datasets]

    def __len__(self):
        return sum(len(ds)*bn for ds, bn in zip(self.datasets, self._batch_nums))

    def __iter__(self):
        ds_indices = [
            np.concatenate(
                [np.arange(len(ds)), np.random.randint(len(ds), size=len(ds) % self.batch_size)],
                axis=0)
            for ds in self.datasets]

        if self.shuffle:
            for ds_idx in range(len(ds_indices)):
                np.random.shuffle(ds_indices[ds_idx])

        batches = []
        for i, ds_idx in enumerate(ds_indices):
            np.random.shuffle(ds_idx)
            for batch_idx in np.split(ds_idx, len(ds_idx) // self.batch_size):
                batches.append((i, batch_idx))

        if self.shuffle:
            np.random.shuffle(batches)

        for indices in batches:
            yield indices


def _concat_batch_sampler_creator(
        dataset: Union[MConcatDataset, Iterable[Union[Dataset, Iterable[BaseData]]]],
        _batch_size: int = 1,
        shuffle: bool = False,
        _drop_last: bool = False
):
    if not (isinstance(dataset, MConcatDataset) or isinstance(dataset, Iterable)):
        raise TypeError('datasets should be either a MConcatDataset or Iterable[Dataset]')

    if not isinstance(dataset, MConcatDataset) and isinstance(dataset, Iterable):
        dataset = MConcatDataset(dataset)

    class CDBatchSampler(BatchSampler):
        def __init__(
            self,
            sampler: Union[Sampler[int], Iterable[int]],
            batch_size: int,
            drop_last: bool,
        ):
            super().__init__(sampler, batch_size, drop_last)
            self.dataset = dataset
            self.shuffle = shuffle
            self.consum_size = self.cumsum(self.dataset)

            if drop_last:
                self._batch_nums = sum(len(ds) // self.batch_size for ds in self.datasets)
            else:
                self._batch_nums = sum(len(ds) // self.batch_size + 1 for ds in self.datasets)

            logging.debug(f'BatchSampler batch_size{self.batch_size}')

        def __repr__(self):
            return (f'{self.__class__.__name__}(' +
                    ', '.join([f'{k}={v}' for k, v in vars(self).items() if not k.startswith("_")]) + ')')

        @property
        def datasets(self):
            return self.dataset.datasets

        @staticmethod
        def cumsum(mc_dataset: MConcatDataset):
            r, s = [0], 0
            for e in mc_dataset.datasets:
                l = len(e)
                r.append(l + s)
                s += l
            return r

        def __len__(self):
            return self._batch_nums

        def __iter__(self):
            if self.drop_last:
                datasets_indices = [self.consum_size[i] + np.arange(len(ds)) for i, ds in enumerate(self.dataset)]
            else:
                datasets_indices = [
                    self.consum_size[i] + np.concatenate([
                        np.arange(len(ds)),
                        np.random.randint(len(ds), size=(self.batch_size - len(ds) % self.batch_size))
                    ], axis=0) for i, ds in enumerate(self.datasets)]

            if self.shuffle:
                for dataset_index in datasets_indices:
                    np.random.shuffle(dataset_index)

            if self.drop_last:
                datasets_indices = [ds_idx[:(len(ds_idx) // self.batch_size) * self.batch_size] for ds_idx in
                                    datasets_indices]

            batches = []
            for dataset_index in datasets_indices:
                batch_num, rest = divmod(len(dataset_index), self.batch_size)
                assert rest == 0
                batches.extend(np.split(dataset_index, batch_num))

            if self.shuffle:
                np.random.shuffle(batches)

            return iter(batches)

    return CDBatchSampler(range(len(dataset)), _batch_size, drop_last=_drop_last)

class ConcatDatasetBatchSampler:
    def __init__(
            self,
            dataset: Union[MConcatDataset, Iterable[Union[Dataset, Iterable[BaseData]]]],
            batch_size: int = 1,
            shuffle: bool = False,
            drop_last: bool = False
    ):
        if isinstance(dataset, MConcatDataset):
            self.dataset = dataset
        elif isinstance(dataset, Iterable):
            self.dataset = MConcatDataset(dataset)
        else:
            raise TypeError('datasets should be either a MConcatDataset or Iterable[Dataset]')

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.consum_size = self.cumsum(self.dataset)

        if drop_last:
            self._batch_nums = sum(len(ds)//self.batch_size for ds in self.datasets)
        else:
            self._batch_nums = sum(len(ds)//self.batch_size + 1 for ds in self.datasets)

    def __repr__(self):
        return f"{self.__class__.__name__}(batch_size={self.batch_size}, shuffle={self.shuffle})"

    @property
    def datasets(self):
        return self.dataset.datasets

    @staticmethod
    def cumsum(mc_dataset: MConcatDataset):
        r, s = [0], 0
        for e in mc_dataset.datasets:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __len__(self):
        return self._batch_nums

    def __iter__(self):
        if self.drop_last:
            datasets_indices = [self.consum_size[i] + np.arange(len(ds)) for i, ds in enumerate(self.dataset)]
        else:
            datasets_indices = [
                self.consum_size[i] + np.concatenate([
                    np.arange(len(ds)),
                    np.random.randint(len(ds), size=(self.batch_size - len(ds) % self.batch_size))
                ], axis=0) for i, ds in enumerate(self.datasets)]

        if self.shuffle:
            for dataset_index in datasets_indices:
                np.random.shuffle(dataset_index)

        if self.drop_last:
            datasets_indices = [ds_idx[:(len(ds_idx)//self.batch_size)*self.batch_size] for ds_idx in datasets_indices]

        batches = []
        for dataset_index in datasets_indices:
            batch_num, rest = divmod(len(dataset_index), self.batch_size)
            assert rest == 0
            batches.extend(np.split(dataset_index, batch_num))

        if self.shuffle:
            np.random.shuffle(batches)

        return iter(batches)


class CDataLoader(DataLoader):
    def __init__(
        self,
        dataset: Union[MConcatDataset, Iterable[Iterable[BaseData]]],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        if not isinstance(dataset, MConcatDataset):
            if isinstance(dataset, Iterable):
                dataset = MConcatDataset(dataset)
            else:
                raise TypeError("dataset must be an instance of MConcatDataset or Iterable of Dataset[PyG.Data]")

        # Remove for pytorch lightning reconstruction
        if not kwargs.get('batch_sampler', None):
            kwargs['batch_sampler'] = _concat_batch_sampler_creator(
                    dataset,
                    _batch_size=batch_size,
                    shuffle=shuffle,
                    _drop_last=kwargs.pop('drop_last', False))

        super().__init__(
            dataset,
            1,
            None,
            follow_batch,
            exclude_keys,
            # batch_sampler=ConcatDatasetBatchSampler(
            #     dataset,
            #     batch_size=batch_size,
            #     shuffle=shuffle,
            #     drop_last=kwargs.pop('drop_last', None)),
            **kwargs
        )
