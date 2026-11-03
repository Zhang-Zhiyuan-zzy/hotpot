"""Ordered molecule collections with shared member references."""

import os
import typing as tp

from .molecule import Molecule


__all__ = ("MolBundle",)


class MolBundle:
    def __init__(self, list_mol: tp.Iterable[Molecule]):
        self.list_mol = list(list_mol)

    def to_pyg_dataset(self, dataset_root: os.PathLike, prefix: str = ''):
        from ..operations.collections import _export_pyg_dataset
        _export_pyg_dataset(self.list_mol, dataset_root, prefix=prefix)
