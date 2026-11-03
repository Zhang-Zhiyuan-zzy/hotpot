"""Export ordered molecule collections through external graph serialization."""

import os
import typing as tp
from pathlib import Path

from ..structure.molecule import Molecule


__all__ = ("to_pyg_dataset",)


def to_pyg_dataset(
        list_mol: tp.Iterable[Molecule],
        dataset_root: os.PathLike,
        prefix: str = ''
):
    _export_pyg_dataset(list(list_mol), dataset_root, prefix=prefix)


def _export_pyg_dataset(list_mol, dataset_root, prefix):
    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        dataset_root.mkdir(parents=False)

    from ..adapters.pyg import _torch_save
    save = _torch_save()
    for i, mol in enumerate(list_mol):
        filename = f"{mol.identifier}.pt" or f"{i}.pt"
        save(mol.to_pyg_data(prefix=prefix), dataset_root / filename)
