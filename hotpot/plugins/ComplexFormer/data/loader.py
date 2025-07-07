import math
import logging
import random

from typing import Optional, Union, List, Iterable, Mapping, Literal

import numpy as np

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, BatchSampler, Sampler, DistributedSampler

from torch_geometric.data import Data
from torch_geometric.data.data import BaseData
from torch_geometric.loader import DataLoader

from hotpot.utils import fmt_print
from hotpot.plugins.ComplexFormer.data import (
    dataset as D,
    collate
)


def _slice_iter_dataset(dataset: Iterable[Data], stop: int) -> list[Data]:
    list_data = []
    for i, data in enumerate(dataset):
        list_data.append(data)
        if i >= stop:
            break

    return list_data

def _slice_mapping_dataset(dataset: Mapping, stop: int) -> list[Data]:
    return [dataset[i] for i in range(min(len(dataset), stop))]

def _slice_dataset(ds: Union[Iterable[Data], Mapping], stop: int) -> D.DataWrapper:
    if isinstance(ds, Mapping):
        return D.DataWrapper(_slice_mapping_dataset(ds, stop))
    elif isinstance(ds, Iterable):
        return D.DataWrapper(_slice_iter_dataset(ds, stop))
    else:
        raise TypeError(f'The dataset in the collection should be Iterable or Mapping')


def _check_concat_dataset(
        dataset: Union[D.MConcatDataset, Iterable[Union[Dataset, Iterable[BaseData]]]]
) -> D.MConcatDataset:
    """
    Check whether the given dataset is an MConcatDataset or an Iterable of Dataset.

    If Neither, raise a TypeError.

    If the dataset is an Iterable of Dataset, convert it to MConcatDataset.
    """
    if not (isinstance(dataset, D.MConcatDataset) or isinstance(dataset, Iterable)):
        raise TypeError('datasets should be either a MConcatDataset or Iterable[Dataset]')

    if isinstance(dataset, Iterable):
        first_dataset = next(iter(dataset))
        if isinstance(first_dataset, (Data, torch.Tensor)):
            raise TypeError(f'Expecting a Iterable of Datasets, but got a Iterable of {type(first_dataset)}')

    if not isinstance(dataset, D.MConcatDataset) and isinstance(dataset, Iterable):
        dataset = D.MConcatDataset(dataset)
    return dataset


def _cumsum_datasets(mc_dataset: D.MConcatDataset):
    assert isinstance(mc_dataset, D.MConcatDataset)
    r, s = [0], 0
    for e in mc_dataset.datasets:
        l = len(e)
        r.append(l + s)
        s += l
    return r


class DistConcatBatchSampler(Sampler):
    def __init__(
            self,
            dataset: Dataset,
            batch_size: Optional[int] = None,
            *,
            num_replicas: Optional[int] = None,
            rank: Optional[int] = None,
            shuffle: bool = True,
            seed: int = 0,
            drop_last: bool = False,
    ):
        super().__init__()
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )
        self.dataset = _check_concat_dataset(dataset)
        self.cunsum_size = _cumsum_datasets(self.dataset)
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.split_size = self.batch_size * self.num_replicas

        logging.debug(f'DistConcatBatchSampler in Rank {rank}')

        # If the dataset length is evenly divisible by num_replicas * batch_size, the there
        # is no need to drop or supply any data.
        if any(len(ds) % self.split_size != 0 for ds in self.datasets):
            # If drop_last was specified, the sample number is equal to nearest available length
            # that is evenly divisible.
            self.num_samples = 0
            if self.drop_last:
                for ds in self.datasets:
                    num, _rest = divmod(len(ds) // self.split_size * self.split_size, self.num_replicas)
                    assert _rest == 0, ('The sample numbers for every dataset should be '
                                        'evenly divisible by the number of replicas')
                    self.num_samples += num

            else:
                for ds in self.datasets:
                    num, _rest = divmod(math.ceil(len(ds) / self.split_size) * self.split_size, self.num_replicas)
                    assert _rest == 0, ('The sample numbers for every dataset should be '
                                        'evenly divisible by the number of replicas')
                    self.num_samples += num

        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)

        assert isinstance(self.num_samples, int), f'Error num_samples {self.num_samples} or Error type {type(self.num_samples)}'

        self.batch_nums, _rest = divmod(self.num_samples, self.batch_size)
        assert _rest == 0

        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def _get_datasets_indices(self) -> list[list[int]]:
        if self.drop_last:
            return [self.cunsum_size[i] + np.arange(len(ds)) for i, ds in enumerate(self.datasets)]

        indices = []
        for i, ds in enumerate(self.datasets):
            index = np.arange(len(ds))
            if len(ds) % self.split_size != 0:
                randidx = np.random.randint(len(ds), size=(self.split_size - len(ds) % self.split_size))
                index = np.concatenate([index, randidx], axis=0)

            indices.append(index + self.cunsum_size[i])

        return indices

    @property
    def datasets(self) -> list[Dataset]:
        return self.dataset.datasets

    def __len__(self) -> int:
        return self.batch_nums

    def __iter__(self) -> Iterable[list[int]]:
        datasets_indices = self._get_datasets_indices()
        np.random.seed(self.seed + self.epoch)

        if self.shuffle:
            for dataset_index in datasets_indices:
                np.random.shuffle(dataset_index)

        if self.drop_last:
            datasets_indices = [ds_idx[:(len(ds_idx) // self.split_size) * self.split_size] for ds_idx in
                                datasets_indices]

        batches = []
        for dataset_index in datasets_indices:
            batch_num, rest = divmod(len(dataset_index), self.split_size)
            assert rest == 0
            batches.extend(np.split(dataset_index, batch_num))

        if self.shuffle:
            np.random.shuffle(batches)

        batches = [batch[self.rank::self.num_replicas] for batch in batches]

        return iter(batches)


def _create_dist_concat_batch_sampler(
        _dataset: Union[D.MConcatDataset, Iterable[Union[Dataset, Iterable[BaseData]]]],
        _batch_size: int = 1,
        _shuffle: bool = False,
        _drop_last: bool = False,
        _num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
):
    if not isinstance(_num_replicas, int):
        _num_replicas = torch.cuda.device_count()

    _dataset = _check_concat_dataset(_dataset)
    batch_sampler = _create_concat_batch_sampler(
        _dataset,
        _batch_size * _num_replicas,
        shuffle=_shuffle,
        _drop_last=_drop_last
    )
    class DistConcatBatchSampler:
        def __init__(self, **kwargs):
            self.batch_sampler = batch_sampler
            self.kwargs = kwargs
            # logging.debug(f'Batch sampler initialized with kwargs: {kwargs}')

        def __len__(self):
            return len(self.batch_sampler)

        def __iter__(self):
            logging.debug(f"Rank {self.kwargs.get('rank')} in DistributedSampler")
            for batch in self.batch_sampler:
                dist_batch = list(DistributedSampler(batch, **self.kwargs))
                # logging.debug(f'DistributedSampler batches: {dist_batch}')
                yield dist_batch

    return DistConcatBatchSampler(
        shuffle=_shuffle,
        drop_last=_drop_last,
        num_replicas=_num_replicas,
        rank=rank
    )


def _create_concat_batch_sampler(
        dataset: Union[D.MConcatDataset, Iterable[Union[Dataset, Iterable[BaseData]]]],
        _batch_size: int = 1,
        shuffle: bool = False,
        _drop_last: bool = False,
        **kwargs
):
    """
    The implementation of Pytorch Lightning will reinitialize the BatchSampler, which leads to
    wrong arguments passed into the reinitialized instance, say the `batch_size` will be set to `1`,
    no matter which values are specified by user. Through defining the `BatchSampler` class in a
    closure, this mistake can be avoided.
    """
    dataset = _check_concat_dataset(dataset)
    class CDBatchSampler(BatchSampler):
        def __init__(
                self,
                sampler: Union[Sampler[int], Iterable[int]],
                batch_size: int,
                drop_last: bool,
                total_nums: Union[int, Literal['shortest', 'longest', 'total', 'log-mean']] = 'total',
                split_ratio: Optional[Union[Iterable[float], Literal['mean', 'as-ratio', 'log2']]] = None,
        ):
            super().__init__(sampler, batch_size, drop_last)
            self.dataset: D.MConcatDataset = dataset
            self.shuffle = shuffle
            self.dataset_sizes: List[int] = [len(ds) for ds in self.datasets]
            self.cumsum_size = _cumsum_datasets(self.dataset)

            self.total_nums, self.total_num_mode = self._set_total_nums(total_nums)
            self.split_ratio, self.split_mode =  self._set_sample_ratio(split_ratio)

            self.sample_nums, self._batch_nums = self._set_batch_nums()

            fmt_print.bold_magenta(f"Sampling numbers in BatchSampler: {self.sample_nums}")
            logging.debug(f'BatchSampler batch_size{self.batch_size}, sampler_size{len(self.sampler)}, dataset_size{len(self.dataset)}')

        def _set_total_nums(self, total_nums):
            if isinstance(total_nums, int):
                if total_nums <= 0:
                    raise ValueError('total_nums must be a positive integer')
                return total_nums, 'user_specify'

            dataset_sizes = [len(ds) for ds in self.datasets]
            if total_nums == 'total':
                return sum(dataset_sizes), 'total'
            elif total_nums == 'shortest':
                return min(dataset_sizes) * len(dataset_sizes), 'shortest'
            elif total_nums == 'longest':
                return max(dataset_sizes) * len(dataset_sizes), 'longest'
            elif total_nums == 'log-mean':
                log_mean = np.mean(np.log10(dataset_sizes))
                return (10 ** log_mean) * len(dataset_sizes), 'log-mean'
            else:
                raise NotImplementedError(f'Unsupported total_nums: {total_nums}')

        def _set_sample_ratio(
                self,
                split_ratio: Optional[Union[Iterable[float], Literal['mean', 'as-ratio', 'log2']]]
        ) -> list[int]:
            if not isinstance(split_ratio, str) and isinstance(split_ratio, Iterable):
                assert len(split_ratio) == self.dataset_sizes, (
                    f'The length of split_ratio must be equal to the number of '
                    f'datasets, but {len(split_ratio)} != {self.dataset_sizes}')
                assert all(isinstance(r, float) for r in split_ratio), f"all split_ratio values must be floats, but {split_ratio}"
                assert all(r > 0 for r in split_ratio), f"all split_ratio values must be positive, but {split_ratio}"

                ratio = np.array(split_ratio) / sum(split_ratio)
                return ratio, 'user-specified'

            elif isinstance(split_ratio, str):
                if split_ratio in ['mean', 'as-ratio', 'log2']:
                    split_mode = split_ratio
                else:
                    raise ValueError(f'Unsupported split_ratio: {split_ratio}')

            elif split_ratio is None:  # Auto mode
                if self.total_num_mode == 'total':
                    split_mode = 'as-ratio'
                elif self.total_num_mode in ('shortest', 'longest'):
                    split_mode = 'mean'
                elif self.total_num_mode == 'log-mean':
                    split_mode = 'log2'
                else:
                    ds_min, ds_max = min(self.dataset_sizes), max(self.dataset_sizes)
                    if ds_max / ds_min > 10:
                        split_mode = 'log2'
                    else:
                        split_mode = 'as-ratio'

            else:
                raise TypeError(f'Unsupported split_ratio: {split_ratio}')

            if split_mode == 'mean':
                return np.ones(len(self.dataset_sizes)) / len(self.dataset_sizes), 'mean'
            elif split_mode == 'as-ratio':
                return np.array(self.dataset_sizes) / sum(self.dataset_sizes), 'as-ratio'
            elif split_mode == 'log2':
                ratio = 1 + np.log2(np.array(self.dataset_sizes)/min(self.dataset_sizes))
                ratio = ratio / sum(ratio)
                return ratio, 'log2'
            else:
                raise ValueError(f'Unsupported split_mode: {split_mode}')

        def _set_batch_nums(self) -> (np.ndarray, int):
            # New version
            _temp_sample_nums = np.long(self.total_nums * self.split_ratio)
            batch_nums, rest = np.divmod(_temp_sample_nums, self.batch_size)

            if not self.drop_last:
                batch_nums += np.bool(rest)

            # sample_nums, batch_nums
            return batch_nums * self.batch_size, sum(batch_nums)

        def __repr__(self):
            return (f'{self.__class__.__name__}(' +
                    ', '.join([f'{k}={v}' for k, v in vars(self).items() if not k.startswith("_")]) + ')')

        @property
        def datasets(self):
            return self.dataset.datasets

        def __len__(self):
            return self._batch_nums

        def _iter(self):
            rep_factor, residual = np.divmod(self.sample_nums, np.array(self.dataset_sizes))

            indices = []
            for i, (rf, res) in enumerate(zip(rep_factor, residual)):
                idx_start, idx_end = self.cumsum_size[i], self.cumsum_size[i+1]
                dataset_idx_range = list(range(idx_start, idx_end))

                index = dataset_idx_range * rf + random.sample(dataset_idx_range, res)
                indices.append(index)

            # Check whether the number of indices for each dataset equals to integer_times of batch_size
            indices_length = np.array([len(idx) for idx in indices])
            batch_num, rest = divmod(indices_length, self.batch_size)
            assert not np.any(rest)

            if self.shuffle:
                for idx in indices:
                    np.random.shuffle(idx)

            batches = []
            for ds_index, b_num in zip(indices, batch_num):
                batches.extend(np.split(np.array(ds_index), b_num))

            if self.shuffle:
                np.random.shuffle(batches)

            return iter(batches)

        def __iter__(self):
            return self._iter()

    # End of CDBatchSampler

    # Initialization of the BatchSampler
    return CDBatchSampler(
        range(len(dataset)),
        _batch_size,
        drop_last=_drop_last,
        total_nums=kwargs.pop('total_nums', 'total'),
        split_ratio=kwargs.pop('split_ratio', 'log2'),
    )

class CDataLoader(DataLoader):
    def __init__(
        self,
        dataset: Union[D.MConcatDataset, Iterable[Iterable[BaseData]]],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        if not isinstance(dataset, D.MConcatDataset):
            if isinstance(dataset, Iterable):
                dataset = D.MConcatDataset(dataset)
            else:
                raise TypeError("dataset must be an instance of MConcatDataset or Iterable of Dataset[PyG.Data]")

        # Remove for pytorch lightning reconstruction
        if not kwargs.get('batch_sampler', None):
            kwargs['batch_sampler'] = _create_concat_batch_sampler(
                    dataset,
                    _batch_size=batch_size,
                    shuffle=shuffle,
                    _drop_last=kwargs.pop('drop_last', False)
            )

        super().__init__(
            dataset,
            1,
            None,
            follow_batch,
            exclude_keys,
            **kwargs
        )
        self.collate_fn = collate.Collater(dataset, follow_batch, exclude_keys)

class DistConcatLoader(DataLoader):
    def __init__(
            self,
            dataset: Union[D.MConcatDataset, Iterable[Iterable[BaseData]]],
            batch_size: int = 1,
            shuffle: bool = False,
            follow_batch: Optional[List[str]] = None,
            exclude_keys: Optional[List[str]] = None,
            **kwargs,
    ):
        if not isinstance(dataset, D.MConcatDataset):
            if isinstance(dataset, Iterable):
                dataset = D.MConcatDataset(dataset)
            else:
                raise TypeError("dataset must be an instance of MConcatDataset or Iterable of Dataset[PyG.Data]")

        # Remove for pytorch lightning reconstruction
        if not kwargs.get('batch_sampler', None):
            kwargs['batch_sampler'] = DistConcatBatchSampler(
                    dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    drop_last=kwargs.pop('drop_last', False),
                    num_replicas=kwargs.pop('num_replicas', 1),
                    rank=kwargs.pop('rank', None),
            )

        super().__init__(
            dataset,
            1,
            None,
            follow_batch,
            exclude_keys,
            **kwargs
        )
        self.collate_fn = collate.Collater(dataset, follow_batch, exclude_keys)