import os
import os.path as osp

import math
import glob
import warnings
import itertools

from typing import Optional, Sequence, Union
from collections import OrderedDict
from tqdm import tqdm

import torch
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import lightning as L


from .dataset import MConcatDataset, torch_load_data, DataWrapper, OnFlyLoadingDataset
from .loader import CDataLoader, DistConcatLoader


def rand_split_on_fly_dataset(
        dataset: OnFlyLoadingDataset,
        lengths: tuple,
        generator: torch.Generator = None,
):
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: list[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(dataset) * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(
                    f"Length of split at index {i} is 0. "
                    f"This might result in an empty dataset."
                )

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):  # type: ignore[arg-type]
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!"
        )

    data_files = dataset.data_files
    indices = torch.randperm(sum(lengths), generator=generator).tolist()  # type: ignore[arg-type, call-overload]
    subset_indices = [
        indices[offset - length: offset]
        for offset, length in zip(itertools.accumulate(lengths), lengths)
    ]
    return [OnFlyLoadingDataset([data_files[i] for i in subset_index]) for subset_index in subset_indices]


def get_first_data(dir_datasets: str) -> dict[str, Data]:
    list_datasets = os.listdir(dir_datasets)
    return {
        ds_name: torch_load_data(next(glob.iglob(osp.join(dir_datasets, ds_name, '*.pt'))))
        for ds_name in list_datasets
    }

class DataModule(L.LightningDataModule):
    def __init__(
            self,
            dir_datasets: str,
            dataset_names: Union[str, Sequence[str]] = None,
            exclude_datasets: Union[str, Sequence[str]] = None,
            *,
            seed: int = 315,
            debug: bool = False,
            ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
            batch_size: int = 1,
            shuffle: bool = True,
            devices: Optional[int] = None,
            num_replicas: Optional[int] = None,
    ):
        super().__init__()
        self.dir_datasets = dir_datasets

        datasets_subdir = os.listdir(dir_datasets)
        if not dataset_names:
            self.list_datasets = sorted(os.listdir(dir_datasets))
        elif isinstance(dataset_names, str):
            self.list_datasets = [dataset_names]
        elif isinstance(dataset_names, Sequence):
            self.list_datasets = list(dataset_names)
        else:
            raise TypeError(f"dataset_names must be a string or a sequence of strings")

        for ds_name in self.list_datasets:
            if ds_name not in datasets_subdir:
                raise ValueError(f'Unknown dataset "{ds_name}", select from {datasets_subdir}')

        if isinstance(exclude_datasets, str):
            self.list_datasets.remove(exclude_datasets)
        elif isinstance(exclude_datasets, Sequence):
            for ds_name in exclude_datasets:
                self.list_datasets.remove(ds_name)

        if len(self.list_datasets) == 0:
            raise AttributeError(f"No datasets found in list_datasets: {self.list_datasets}")

        self.debug = debug

        self._datasets = OrderedDict()

        self.seed = seed
        self.ratios = [ r /sum(ratios) for r in ratios]

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_replicas = num_replicas

        if isinstance(devices, int):
            self.devices = devices
        else:
            self.devices = torch.cuda.device_count()

        self._loading_data_to_memory()
        # self._loading_data_path()


    @property
    def dataset_counts(self) -> int:
        return len(self.list_datasets)

    @property
    def first_data(self):
        if len(self.list_datasets) == 1:
            return torch_load_data(
                next(glob.iglob(osp.join(self.dir_datasets, self.list_datasets[0], '*.pt')))
            )
        else:
            return [
                torch_load_data(next(glob.iglob(osp.join(self.dir_datasets, ds_name, '*.pt'))))
                for ds_name in self.list_datasets
            ]

    @property
    def is_multi_datasets(self) -> bool:
        return len(self.list_datasets) > 1

    def _loading_data_to_memory(self):
        for ds_name in self.list_datasets:
            dir_dataset = osp.join(self.dir_datasets, ds_name)
            if self.debug:
                path_generator = glob.iglob(osp.join(dir_dataset, '*.pt'))
                list_data = []
                for _ in tqdm(range(10*self.devices*self.batch_size), 'loading data'):
                    try:
                        list_data.append(torch_load_data(next(path_generator)))
                    except StopIteration:
                        break

            else:
                list_data = [torch_load_data(p) for p in tqdm(glob.glob(osp.join(dir_dataset, '*.pt')), 'loading data')]

            self._datasets[ds_name] = DataWrapper(list_data)

    def _loading_data_path(self):
        for ds_name in self.list_datasets:
            dir_dataset = osp.join(self.dir_datasets, ds_name)
            if self.debug:
                path_generator = glob.iglob(osp.join(dir_dataset, '*.pt'))
                list_path = []
                for _ in tqdm(range(40*self.devices*self.batch_size), 'loading data'):
                    try:
                        list_path.append(next(path_generator))
                    except StopIteration:
                        break

            else:
                list_path = [p for p in tqdm(glob.glob(osp.join(dir_dataset, '*.pt')), 'loading data')]

            self._datasets[ds_name] = OnFlyLoadingDataset(list_path)

    def setup(self, stage: Optional[str] = None):
        ratios = [0.8, 0.1, 0.1] if self.debug else self.ratios

        generator = torch.Generator().manual_seed(self.seed)
        _train_datasets = []
        _val_datasets = []
        _test_datasets = []
        for ds_name, dataset in self._datasets.items():
            train, val, test = random_split(dataset, ratios, generator)
            _train_datasets.append(train)
            _val_datasets.append(val)
            _test_datasets.append(test)

        if len(self._datasets) > 1:
            self.train_dataset = MConcatDataset(_train_datasets)
            self.val_dataset = MConcatDataset(_val_datasets)
            self.test_dataset = MConcatDataset(_test_datasets)
        else:
            self.train_dataset = _train_datasets[0]
            self.val_dataset = _val_datasets[0]
            self.test_dataset = _test_datasets[0]

    def _setup(self, stage: Optional[str] = None):
        generator = torch.Generator().manual_seed(self.seed)
        _train_datasets = []
        _val_datasets = []
        _test_datasets = []
        for ds_name, dataset in self._datasets.items():
            assert isinstance(dataset, OnFlyLoadingDataset)
            train, val, test = rand_split_on_fly_dataset(dataset, self.ratios, generator)
            _train_datasets.append(train)
            _val_datasets.append(val)
            _test_datasets.append(test)

        if len(self._datasets) > 1:
            self.train_dataset = MConcatDataset(_train_datasets)
            self.val_dataset = MConcatDataset(_val_datasets)
            self.test_dataset = MConcatDataset(_test_datasets)
        else:
            self.train_dataset = _train_datasets[0]
            self.val_dataset = _val_datasets[0]
            self.test_dataset = _test_datasets[0]

    def _get_loader(self, dataset, batch_size: int = 1, shuffle: bool = False):
        if isinstance(self.num_replicas, int) and self.num_replicas > 1 and isinstance(dataset, MConcatDataset):
            return DistConcatLoader(dataset, batch_size, shuffle, num_workers=6, num_replicas=self.num_replicas, pin_memory=True)
        elif isinstance(dataset, MConcatDataset):
            return CDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=6, pin_memory=True)
        else:
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=6, pin_memory=True)

    def train_dataloader(self) -> DataLoader:
        return self._get_loader(self.train_dataset, self.batch_size, self.shuffle)

    def val_dataloader(self) -> DataLoader:
        return self._get_loader(self.val_dataset, self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return self._get_loader(self.test_dataset, self.batch_size)