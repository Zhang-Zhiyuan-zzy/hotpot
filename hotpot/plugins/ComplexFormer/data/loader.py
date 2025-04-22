import logging

import math
from typing import Optional, Union, List, Iterable, Mapping

import numpy as np

from torch.utils.data import Dataset, BatchSampler, Sampler, IterableDataset

from torch_geometric.data import Data
from torch_geometric.data.data import BaseData
from torch_geometric.loader import DataLoader

from hotpot.plugins.ComplexFormer.data import dataset as D


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
        dataset: Union[D.MConcatDataset, Iterable[Union[Dataset, Iterable[BaseData]]]],
        _batch_size: int = 1,
        shuffle: bool = False,
        _drop_last: bool = False
):
    """
    The implementation of Pytorch Lightning will reinitialize the BatchSampler, which leads to
    wrong arguments passed into the reinitialized instance, say the `batch_size` will be set to `1`,
    no matter which values are specified by user. Through defining the `BatchSampler` class in a
    closure, this mistake can avoid.
    """
    if not (isinstance(dataset, D.MConcatDataset) or isinstance(dataset, Iterable)):
        raise TypeError('datasets should be either a MConcatDataset or Iterable[Dataset]')

    if not isinstance(dataset, D.MConcatDataset) and isinstance(dataset, Iterable):
        dataset = D.MConcatDataset(dataset)

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
        def cumsum(mc_dataset: D.MConcatDataset):
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


def prepare_dataloader(
        train_dataset: Union[D.MConcatDataset, Iterable[D.PretrainDataset], D.PretrainDataset],
        test_dataset: Union[D.MConcatDataset, Iterable[D.PretrainDataset], D.PretrainDataset],
        load_all_data: bool = True,
        sample_num: int = None,
        batch_size: Optional[int] = None,
        train_shuffle: bool = True,
        test_shuffle: bool = False,
        debug: bool = False,
        # **kwargs,
):
    if batch_size is None:
        batch_size = 256

    if debug:
        train_shuffle = test_shuffle = False

    if (sample_num is None) and debug:
        sample_num = 8 * batch_size

    datasets = [train_dataset, test_dataset]
    dataset_names = ['train', 'test']

    list_data = []
    for i, dataset in enumerate(datasets):
        if isinstance(dataset, D.PretrainDataset):
            if load_all_data or debug:
                list_data.append(D.load_data(dataset, sample_num=sample_num))
            else:
                list_data.append(D.load_data(dataset))

        # When dataset is a true dataset
        elif isinstance(dataset, D.MConcatDataset):
            if debug or load_all_data:
                list_data.append(dataset.load_data(sample_num=sample_num))
            else:
                list_data.append(dataset)

        elif isinstance(dataset, Dataset):
            if debug:
                list_data.append([dataset[i] for i in range(sample_num)])
            else:
                list_data.append(dataset)

        elif isinstance(dataset, IterableDataset):
            if debug:
                list_data.append([dataset[i] for i in range(sample_num)])
            elif load_all_data:
                list_data.append([d for d in dataset])
            else:
                list_data.append(dataset)

        elif isinstance(dataset, Iterable):
            dataset = list(dataset)

            # When the dataset is a collection of datasets
            if isinstance(dataset[0], (Dataset, IterableDataset)):
                # Check which type of dataset is given
                assert all(isinstance(ds, (Dataset, IterableDataset, Iterable)) for ds in dataset), (
                    'Multi dataset should make sure all items is a Dataset, IterableDataset or Iterable')

                # Check iterable-formatted dataset
                for k, ds in enumerate(dataset):
                    if isinstance(ds, Iterable):
                        assert all(isinstance(d, Data) for d in ds), (
                            'When `dataset` given by Iterable, all items should be Data'
                        )
                ############## Check End ###############

                # Convert the datasets collection to MConcatDataset
                if debug:
                    datasets.append(D.MConcatDataset([_slice_dataset(ds, sample_num) for ds in dataset]))
                else:
                    list_data.append(D.MConcatDataset([D.DataWrapper(ds) for ds in dataset]))

            # Really dataset
            elif isinstance(dataset[0], Data):
                list_data.append(dataset)

            else:
                raise TypeError('When `dataset` given by Iterable, all items should be Data or Dataset')

        else:
            raise TypeError(
                f'{dataset_names[i]} dataset should be a Data, Iterable[Data], IterableDataset, Dataset,'
                f'MConcatDataset or Iterable[Dataset]'
            )

    train_dataset, test_dataset = list_data
    if isinstance(train_dataset, D.MConcatDataset):
        assert isinstance(test_dataset, D.MConcatDataset)
        assert len(train_dataset.datasets) == len(test_dataset.datasets)

    if isinstance(train_dataset, D.MConcatDataset):
        train_loader = CDataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle)
        test_loader = CDataLoader(test_dataset, batch_size=batch_size, shuffle=test_shuffle)
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=train_shuffle,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=test_shuffle,
        )

    return train_loader, test_loader