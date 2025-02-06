from collections import defaultdict
from pathlib import Path
from typing import Union

import h5py
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset


class LnQM_Dataset(InMemoryDataset):
    """Dataset holding LnQM data.

    Usage:
    >>> dataset = LnQM_Dataset(path_to_hdf5="lnqm.h5")
    """

    def __init__(self, path_to_hdf5: Union[str, Path], transform=None):
        super().__init__("./", transform, pre_transform=None, pre_filter=None)
        self.path_to_hdf5 = path_to_hdf5

        if self.path_to_hdf5:
            self.data, self.slices = LnQM_Dataset.from_hdf5(self.path_to_hdf5)
        else:
            # empty dataset
            self.data, self.slices = Data(), defaultdict(dict, {})

    @staticmethod
    def from_hdf5(fp: Union[str, Path]) -> tuple[Data, defaultdict]:
        """Load data and slices from HDF5 file."""
        data = {}
        slices = {}
        with h5py.File(fp, "r") as f:
            for key in f["data"].keys():
                np_arrays = {"data": f["data"][key][:], "slices": f["slices"][key][:]}
                # some casting
                for prop, val in np_arrays.items():
                    if val.dtype == np.uint64:
                        np_arrays[prop] = val.astype(np.int64)
                # uid is of dtype string, so we got to handle it seperately
                if key == "uid":
                    uids = [s.decode("utf-8") for s in np_arrays["data"].tolist()]
                    data[key] = uids
                    slices[key] = torch.from_numpy(np_arrays["slices"])
                else:
                    data[key] = torch.from_numpy(np_arrays["data"])
                    slices[key] = torch.from_numpy(np_arrays["slices"])
        return Data.from_dict(data), defaultdict(dict, slices)

    def to_hdf5(self, fp: Union[str, Path]):
        """Save the data and slices of the dataset to an HDF5 file."""
        with h5py.File(fp, "w") as f:
            data_group = f.create_group("data")
            slices_group = f.create_group("slices")

            for key, value in self._data.items():
                if not isinstance(value, list):  # strings such as uid
                    if isinstance(value, torch.Tensor):
                        value = value.numpy()
                    if value.dtype == np.int64:
                        value = value.astype(np.uint64)

                data_group.create_dataset(key, data=value)

                # save slices
                slice_value = self.slices[key].numpy()
                if slice_value.dtype == np.int64:
                    slice_value = slice_value.astype(np.uint64)
                slices_group.create_dataset(key, data=slice_value)