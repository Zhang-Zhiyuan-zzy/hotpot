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

from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from sklearn.model_selection import KFold
from tqdm import tqdm

import torch
from torch.utils.data import random_split, Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import lightning as L


from .dataset import MConcatDataset, torch_load_data, DataWrapper, PathStoredDataset
from .loader import CDataLoader, DistConcatLoader
from ..run import datacls
from ....utils import fmt_print


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
    batch_num : bool, default False
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
            args_cfg: datacls.DataModuleArgs,
    ):
        super().__init__()
        self.dir_datasets = args_cfg.dir_datasets
        self.test_only = args_cfg.test_only

        datasets_subdir = os.listdir(args_cfg.dir_datasets)
        if not args_cfg.dataset_names:
            self.list_dataset_names = sorted(os.listdir(args_cfg.dir_datasets))
        elif isinstance(args_cfg.dataset_names, str):
            self.list_dataset_names = [args_cfg.dataset_names]
        elif isinstance(args_cfg.dataset_names, Sequence):
            self.list_dataset_names = list(args_cfg.dataset_names)
        else:
            raise TypeError(f"dataset_names must be a string or a sequence of strings")

        for ds_name in self.list_dataset_names:
            if ds_name not in datasets_subdir:
                raise ValueError(f'Unknown dataset "{ds_name}", select from {datasets_subdir}')

        if isinstance(args_cfg.exclude_datasets, str):
            self.list_dataset_names.remove(args_cfg.exclude_datasets)
        elif isinstance(args_cfg.exclude_datasets, Sequence):
            for ds_name in args_cfg.exclude_datasets:
                self.list_dataset_names.remove(ds_name)

        if len(self.list_dataset_names) == 0:
            raise AttributeError(f"No datasets found in list_datasets: {self.list_dataset_names}")

        self.batch_num = args_cfg.batch_num

        self._datasets = OrderedDict()

        self.seed = args_cfg.seed
        self.ratios = [ r /sum(args_cfg.ratios) for r in args_cfg.ratios]

        self.batch_size = args_cfg.batch_size
        self.shuffle = args_cfg.shuffle
        self.num_replicas = args_cfg.num_replicas

        if args_cfg.devices is None:
            self.device_count = torch.cuda.device_count()
        elif isinstance(args_cfg.devices, int):
            self.device_count = args_cfg.devices
        elif isinstance(args_cfg.devices, (tuple, list)):
            self.device_count = len(args_cfg.devices)
        else:
            raise TypeError(f"devices must be a int or a sequence of ints")

        if args_cfg.load_data_memory and not args_cfg.test_only:
            self._loading_data_to_memory()
        else:
            self._loading_data_path()


    @property
    def dataset_counts(self) -> int:
        """int: Number of datasets selected."""
        return len(self.list_dataset_names)

    @property
    def first_data(self):
        if len(self.list_dataset_names) == 1:
            first_data = torch_load_data(
                next(glob.iglob(osp.join(self.dir_datasets, self.list_dataset_names[0], '*.pt')))
            )
            first_data.dir_datasets = self.dir_datasets
            first_data.dataset_name = self.list_dataset_names[0]
            return first_data
        else:
            list_first_data = [
                torch_load_data(next(glob.iglob(osp.join(self.dir_datasets, ds_name, '*.pt'))))
                for ds_name in self.list_dataset_names
            ]
            for i, first_data in enumerate(list_first_data):
                first_data.dir_datasets = self.dir_datasets
                first_data.dataset_name = self.list_dataset_names[i]
            return list_first_data

    @property
    def is_multi_datasets(self) -> bool:
        """bool: ``True`` when more than one dataset is in use."""
        return len(self.list_dataset_names) > 1

    def _loading_data_to_memory(self):
        """Load every *.pt* file into RAM and wrap in a `DataWrapper`."""
        for ds_name in self.list_dataset_names:
            dir_dataset = osp.join(self.dir_datasets, ds_name)
            if self.batch_num:

                debug_sample_nums = self.batch_num * self.device_count * self.batch_size \
                                    + random.randint(0, self.batch_size)  # and a random residual

                path_generator = glob.iglob(osp.join(dir_dataset, '*.pt'))
                list_data = []
                for _ in tqdm(range(debug_sample_nums), 'loading data to Memory in Debug'):
                    try:
                        list_data.append(torch_load_data(next(path_generator)))
                    except StopIteration:
                        break

            else:
                list_data = [torch_load_data(p) for p in tqdm(glob.glob(osp.join(dir_dataset, '*.pt')), 'loading data to Memory')]

            self._datasets[ds_name] = DataWrapper(list_data)

    def _loading_data_path(self):
        """Store only file paths on disk and wrap in `PathStoredDataset`."""
        for ds_name in self.list_dataset_names:
            dir_dataset = osp.join(self.dir_datasets, ds_name)
            if self.batch_num:
                path_generator = glob.iglob(osp.join(dir_dataset, '*.pt'))
                list_path = []
                for _ in tqdm(range(self.batch_num * self.device_count * self.batch_size), 'loading data path in Debug'):
                    try:
                        list_path.append(next(path_generator))
                    except StopIteration:
                        break

            else:
                list_path = list(glob.glob(osp.join(dir_dataset, '*.pt')))

            fmt_print.dark_green('Initialize PathStoredDataset')
            self._datasets[ds_name] = PathStoredDataset(list_path)

    def train_val_test_split(self):
        generator = torch.Generator().manual_seed(self.seed)
        _train_datasets = []
        _val_datasets = []
        _test_datasets = []
        for ds_name, dataset in self._datasets.items():
            train, val, test = random_split(dataset, self.ratios, generator)
            _train_datasets.append(train)
            _val_datasets.append(val)
            _test_datasets.append(test)

        return _train_datasets, _val_datasets, _test_datasets

    def cross_val_split(self, cv: int = 5):
        kf = KFold(n_splits=cv, shuffle=True, random_state=self.seed)
        if len(self._datasets) > 1:
            cv_datasets = [[[], []] for _ in range(cv)]
            for ds_name, dataset in self._datasets.items():
                for i, (train_idx, val_idx) in enumerate(kf.split(dataset)):
                    cv_datasets[i][0].append(Subset(dataset, train_idx))
                    cv_datasets[i][1].append(Subset(dataset, val_idx))

            return [(MConcatDataset(train_ds), MConcatDataset(val_ds)) for train_ds, val_ds in cv_datasets]

        else:
            dataset = list(self._datasets.values())[0]
            return [(Subset(dataset, train_idx), Subset(dataset, train_idx)) for train_idx, val_idx in kf.split(dataset)]

    def setup(self, stage: Optional[str] = None):
        """Create train/val/test splits (eager mode).

        Parameters
        ----------
        stage : str | None, optional
            Lightning stage (*fit*, *validate*, *test*, or *predict*).
            It is ignored here but required by the API.
        """
        _train_datasets, _val_datasets, _test_datasets = self.train_val_test_split()

        if len(self._datasets) > 1:
            self.train_dataset = MConcatDataset(_train_datasets)
            self.val_dataset = MConcatDataset(_val_datasets)
            self.test_dataset = MConcatDataset(_test_datasets)
        else:
            self.train_dataset = _train_datasets[0]
            self.val_dataset = _val_datasets[0]
            self.test_dataset = _test_datasets[0]

    def get_loader(self, dataset, batch_size: int = 1, shuffle: bool = False):
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
        return self.get_loader(self.train_dataset, self.batch_size, self.shuffle)

    def val_dataloader(self) -> DataLoader:
        """DataLoader: Validation dataloader created in :py:meth:`setup`."""
        return self.get_loader(self.val_dataset, self.batch_size)

    def test_dataloader(self) -> DataLoader:
        """DataLoader: Test dataloader created in :py:meth:`setup`."""
        return self.get_loader(self.test_dataset, self.batch_size)
