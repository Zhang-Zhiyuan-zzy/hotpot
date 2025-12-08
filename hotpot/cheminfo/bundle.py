"""
@File Name:        bundle
@Project:          
@Author:           Zhiyuan Zhang
@Created On:       2025/12/5 14:00
@Project:          Hotpot

Bundle operation for Molecule
"""
import os
import typing as tp
from pathlib import Path
from .core import Molecule

class MolBundle:
    def __init__(self, list_mol: tp.Iterable[Molecule]):
        self.list_mol = list(list_mol)

    def to_pyg_dataset(self, dataset_root: os.PathLike, prefix: str = ''):
        dataset_root = Path(dataset_root)
        if not dataset_root.exists():
            dataset_root.mkdir(parents=False)

        import torch
        for i, mol in enumerate(self.list_mol):
            filename = f"{mol.identifier}.pt" or f"{i}.pt"
            torch.save(mol.to_pyg_data(prefix=prefix), dataset_root / filename)


def to_pyg_dataset(
        list_mol: tp.Iterable[Molecule],
        dataset_root: os.PathLike,
        prefix: str = ''
):
    MolBundle(list_mol).to_pyg_dataset(dataset_root, prefix=prefix)
