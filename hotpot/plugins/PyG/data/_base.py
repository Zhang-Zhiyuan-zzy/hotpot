import os
import time
import os.path as osp
from glob import glob
from typing import Iterable, Callable
import multiprocessing as mp
from abc import ABC, abstractmethod

import torch
from torch_geometric.data import Data


__all__ = ["BaseDataset"]


class BaseDataset(ABC):
    __attr_names__: dict[str, tuple[str]] = None

    def __init__(
            self,
            data_dir: str,
            test_num = None,
            nproc=None,
            timeout=None,
            prefilter: Callable = None,
            transform: Callable = None,
    ):
        self.data_dir = data_dir
        self.prefilter = prefilter
        self.transform = transform
        self.test_num = test_num
        self.timeout = timeout

        if nproc is None:
            self.nproc = os.cpu_count() // 2
        else:
            self.nproc = nproc

        if not osp.exists(self.data_dir):
            os.mkdir(self.data_dir)

        self.datalist = [p for p in glob(osp.join(self.data_dir, '*.pt'))]
        if not self.datalist:
            self.process()
            self.datalist = [p for p in glob(osp.join(self.data_dir, '*.pt'))]

        self.len = len(self.datalist)
        self._check_data_integrity()

    def __getitem__(self, idx: int) -> Data:
        ...
    def __len__(self) -> int:
        ...
    def __iter__(self) -> Iterable[Data]:
        ...
    def get(self, idx: int) -> Data:
        ...

    def _check_data_integrity(self):
        if self.__attr_names__ is None:
            raise AttributeError('the data items should be specified')

        # Check
        first_data = self[0]
        for item, attr_names in self.__attr_names__.items():
            assert getattr(first_data, f"{item}_names") == attr_names

    @abstractmethod
    def to_data(self, mol, *args, **kwargs):
        pass

    def _get_data(self, mol, data_dir, *args, **kwargs):
        data = self.to_data(mol, *args, **kwargs)
        torch.save(data, osp.join(data_dir, f"{data.identifier}.pt"))

    def process(self):
        raise NotImplementedError

    def mp_process(self, mol, processes):
        while len(processes) >= self.nproc:
            t0 = time.time()
            to_remove = []
            for p in processes:
                if not p.is_alive():
                    to_remove.append(p)

            for p in to_remove:
                processes.remove(p)

            if self.timeout and time.time() - t0 > self.timeout:
                raise TimeoutError("In exporting molecule PyG data object")

        p = mp.Process(
            target=self._get_data,
            args=(mol, self.data_dir)
        )
        p.start()
        processes.append(p)

    @property
    def attr_names(self):
        return self.__attr_names__
