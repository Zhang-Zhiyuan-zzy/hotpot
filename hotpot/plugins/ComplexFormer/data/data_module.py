"""
data_module.py
==============

High-level utilities for turning a folder full of torch-saved ``*.pt`` files
into ready-to-use PyTorch Lightning dataloaders.

The file contains two public call-sites:

1. **DataModule**

   A `pytorch_lightning.LightningDataModule` implementation that

   • Discovers one or more dataset sub-directories inside a *root* folder.
   • Loads the underlying ``torch_geometric.data.Data`` tensors either eagerly
     (RAM) or lazily (disk paths).
   • Splits every dataset into train/validation/test subsets with deterministic
     randomness.
   • Concatenates multiple datasets into meta-datasets while preserving
     per-dataset identity (supports distributed training via replica-aware
     loaders).
   • Provides standard Lightning hooks: ``setup``, ``train_dataloader``,
     ``val_dataloader`` and ``test_dataloader``.

   The expected on-disk layout is::

       <dir_datasets>/
           dataset_A/
               0001.pt
               0002.pt
               ...
           dataset_B/
               0001.pt
               0002.pt
               ...

   Every ``.pt`` file must resolve to a single
   ``torch_geometric.data.Data`` object via the project-specific helper
   ``torch_load_data``.

2. **get_first_data**

   Convenience helper that returns the *first* sample it finds in **every**
   dataset sub-directory—handy for quickly inspecting shapes and attributes
   without instantiating the full `DataModule`.

Typical usage
-------------

>>> from data_module import DataModule
>>> dm = DataModule(
...     dir_datasets="~/data/graphs",
...     batch_size=32,
...     devices=2,              # falls back to cuda.device_count() if None
...     load_data_memory=False  # keep only file paths; load lazily
... )
>>> dm.setup()                 # create train/val/test splits
>>> next(iter(dm.train_dataloader())).x.shape
torch.Size([32, 5])

Dependencies
------------

• PyTorch ≥ 1.13
• PyTorch Geometric for the ``Data`` class (imported indirectly)
• PyTorch Lightning
• tqdm, glob, os, torch, collections.OrderedDict

The module purposefully avoids heavyweight data-processing libraries so it can
be dropped into most Lightning/Geometric projects with minimal friction.
"""

import os
import os.path as osp

import glob
import random

from typing import Optional, Sequence, Union
from collections import OrderedDict
from tqdm import tqdm

import torch
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import lightning as L


from .dataset import MConcatDataset, torch_load_data, DataWrapper, PathStoredDataset
from .loader import CDataLoader, DistConcatLoader


def get_first_data(dir_datasets: str) -> dict[str, Data]:
    """Load the first ``*.pt`` sample found in every dataset sub-directory.

    Parameters
    ----------
    dir_datasets : str
        Path to the parent directory whose immediate sub-folders each
        represent a dataset (the same layout expected by :class:`DataModule`).

    Returns
    -------
    dict[str, Data]
        Mapping ``{dataset_name: first_sample_tensor}``.

    Raises
    ------
    FileNotFoundError
        If a dataset directory contains no ``*.pt`` files.
    """
    list_datasets = os.listdir(dir_datasets)
    return {
        ds_name: torch_load_data(next(glob.iglob(osp.join(dir_datasets, ds_name, '*.pt'))))
        for ds_name in list_datasets
    }

class DataModule(L.LightningDataModule):
    """
    Lightning data module that aggregates one or more torch-saved datasets (.pt files)
    located in sub-directories of a root folder and yields train/val/test dataloaders.

    The module supports the following features:
      • Loading the actual tensors into RAM or deferring to on-demand disk access
        (``load_data_memory``).
      • Working with a single dataset folder or a list of folders, with optional
        inclusion/exclusion filtering.
      • Deterministic random splits according to user-defined ratios and seed.
      • Concatenation of multiple datasets into *meta* datasets that preserve each
        child dataset’s identity.
      • Distributed (replicated) training via custom loaders when ``num_replicas`` > 1.
      • A *debug* mode that restricts each dataset to *≈ 10 × devices × batch_size*
        samples for quicker iteration.

    Parameters
    ----------
    dir_datasets : str
        Path to the parent folder whose immediate sub-directories each represent
        a dataset (e.g.
        ``dir_datasets/
            dataset_A/
                0001.pt
                0002.pt
                ...
            dataset_B/
                0001.pt
                ...``).
    dataset_names : str | Sequence[str], optional
        Dataset sub-folder(s) to load.
        *None* (default) means "use every sub-directory found in *dir_datasets*".
    exclude_datasets : str | Sequence[str], optional
        Dataset(s) to remove from the final selection (evaluated **after** the
        ``dataset_names`` filter).
    seed : int, default 315
        Random seed used when splitting data into train/val/test sets.
    debug : bool, default False
        If *True*, only a small subset of each dataset is loaded for rapid
        prototyping and the split ratios default to (0.8, 0.1, 0.1) regardless of
        ``ratios``.
    ratios : tuple[float, float, float], default (0.8, 0.1, 0.1)
        Fractions for train, validation and test splits. They are normalised
        internally so the exact magnitudes are unimportant (e.g. (8,1,1) is valid).
    batch_size : int, default 1
        Batch size returned by the dataloaders.
    shuffle : bool, default True
        Whether to shuffle batches in the training dataloader.
    devices : int | None, default None
        Number of CUDA devices. If *None*, ``torch.cuda.device_count()`` is used.
        Only relevant for *debug* sample cap and replica-aware loaders.
    num_replicas : int | None, default None
        When set to an integer greater than one, distributed concatenated loaders
        (`DistConcatLoader`) are used.
    load_data_memory : bool, default True
        If *True*, every ``.pt`` file is loaded into RAM at construction time.
        If *False*, only the file paths are stored and tensors are loaded lazily
        when sampled.

    Attributes
    ----------
    list_datasets : list[str]
        Final ordered list of dataset names being used.
    dataset_counts : int
        Number of datasets in ``list_datasets``.
    first_data : Tensor | list[Tensor]
        Convenience property returning the *first* tensor of the first dataset
        (or a list of first tensors when multiple datasets are present)—useful
        for inspecting shapes/dtypes.
    is_multi_datasets : bool
        *True* when more than one dataset is selected.
    train_dataset / val_dataset / test_dataset : torch.utils.data.Dataset
        Split datasets produced during :py:meth:`setup`.
    train_dataloader / val_dataloader / test_dataloader : torch.utils.data.DataLoader
        Lightning dataloaders built on top of the split datasets.

    Raises
    ------
    ValueError
        If a requested dataset is not found inside *dir_datasets*.
    AttributeError
        If, after filtering, no datasets remain.
    TypeError
        If ``dataset_names`` is neither *None*, *str* nor sequence of *str*.

    Notes
    -----
    • All file discovery relies on ``glob`` with the pattern ``*.pt``.
    • The actual tensor loading is delegated to a project-specific utility
      ``torch_load_data``; replace/monkey-patch this for custom logic.
    • For lazy loading, each dataset is wrapped in ``PathStoredDataset`` which
      yields file paths that are internally turned into tensors at access time.

    Examples
    --------
    >>> dm = DataModule(
    ...     dir_datasets="~/data/my_experiments",
    ...     exclude_datasets="bad_run",
    ...     batch_size=8,
    ...     num_replicas=4,
    ... )
    >>> dm.setup()
    >>> len(dm.train_dataloader())  # number of batches
    """
    _DEBUG_BATCHES = 40
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
            load_data_memory: bool = True,
            test_only: bool = False,
    ):
        super().__init__()
        self.dir_datasets = dir_datasets
        self.test_only = test_only

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

        if load_data_memory and not test_only:
            self._loading_data_to_memory()
        else:
            self._loading_data_path()


    @property
    def dataset_counts(self) -> int:
        """int: Number of datasets selected."""
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
        """bool: ``True`` when more than one dataset is in use."""
        return len(self.list_datasets) > 1

    def _loading_data_to_memory(self):
        """Load every *.pt* file into RAM and wrap in a `DataWrapper`."""
        for ds_name in self.list_datasets:
            dir_dataset = osp.join(self.dir_datasets, ds_name)
            if self.debug:

                debug_sample_nums = self._DEBUG_BATCHES * self.devices * self.batch_size \
                                    + random.randint(0, self.batch_size)  # and a random residual

                path_generator = glob.iglob(osp.join(dir_dataset, '*.pt'))
                list_data = []
                for _ in tqdm(range(debug_sample_nums), 'loading data'):
                    try:
                        list_data.append(torch_load_data(next(path_generator)))
                    except StopIteration:
                        break

            else:
                list_data = [torch_load_data(p) for p in tqdm(glob.glob(osp.join(dir_dataset, '*.pt')), 'loading data')]

            self._datasets[ds_name] = DataWrapper(list_data)

    def _loading_data_path(self):
        """Store only file paths on disk and wrap in `PathStoredDataset`."""
        for ds_name in self.list_datasets:
            dir_dataset = osp.join(self.dir_datasets, ds_name)
            if self.debug:
                path_generator = glob.iglob(osp.join(dir_dataset, '*.pt'))
                list_path = []
                for _ in tqdm(range(self._DEBUG_BATCHES*self.devices*self.batch_size), 'loading data'):
                    try:
                        list_path.append(next(path_generator))
                    except StopIteration:
                        break

            else:
                list_path = list(glob.glob(osp.join(dir_dataset, '*.pt')))

            self._datasets[ds_name] = PathStoredDataset(list_path)

    def setup(self, stage: Optional[str] = None):
        """Create train/val/test splits (eager mode).

        Parameters
        ----------
        stage : str | None, optional
            Lightning stage (*fit*, *validate*, *test*, or *predict*).
            It is ignored here but required by the API.
        """
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

    def _get_loader(self, dataset, batch_size: int = 1, shuffle: bool = False):
        """Return the correct DataLoader/ConcatLoader for the given dataset.

        Chooses between standard, concatenated, and distributed loaders based on
        ``num_replicas`` and dataset type.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Dataset to wrap.
        batch_size : int, default 1
            Mini-batch size.
        shuffle : bool, default False
            Whether to shuffle each epoch.

        Returns
        -------
        torch.utils.data.DataLoader
            Ready-to-use dataloader.
        """
        if isinstance(self.num_replicas, int) and self.num_replicas > 1 and isinstance(dataset, MConcatDataset):
            return DistConcatLoader(dataset, batch_size, shuffle, num_workers=6, num_replicas=self.num_replicas, pin_memory=True)
            # return CDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=6, pin_memory=True)
        elif isinstance(dataset, MConcatDataset):
            return CDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=6, pin_memory=True)
        else:
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=6, pin_memory=True)

    def train_dataloader(self) -> DataLoader:
        """DataLoader: Training dataloader created in :py:meth:`setup`."""
        return self._get_loader(self.train_dataset, self.batch_size, self.shuffle)

    def val_dataloader(self) -> DataLoader:
        """DataLoader: Validation dataloader created in :py:meth:`setup`."""
        return self._get_loader(self.val_dataset, self.batch_size)

    def test_dataloader(self) -> DataLoader:
        """DataLoader: Test dataloader created in :py:meth:`setup`."""
        return self._get_loader(self.test_dataset, self.batch_size)