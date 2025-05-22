
import time
import os.path as osp
from typing import Callable
from operator import attrgetter
import multiprocessing as mp

from tqdm import tqdm
import torch
from torch_geometric.data import Data

from hotpot.cheminfo.core import Molecule
from hotpot.dataset import tmqm
from ._base import BaseDataset
from .utils import *


class tmQmDataset(BaseDataset):
    __attr_names__ = {
        'x': ('atomic_number','n', 's', 'p', 'd', 'f', 'g',
              'formal_charge','partial_charge', 'is_aromatic',
              'x', 'y', 'z',
              'valence', 'implicit_hydrogens', 'is_metal'),
        'edge_attr': ('bond_order', 'is_aromatic', 'is_metal_ligand_bond'),
        'pair_attr': ('wiberg_bond_order', 'length_shortest_path'),
        'ring_attr': ('is_aromatic', 'has_metal'),
        'y': ('energy', 'dispersion', 'dipole', 'metal_q', 'Hl', 'HOMO', 'LUMO', 'polarizability')
    }

    def __init__(
            self,
            data_dir: str,
            test_num=None,
            nproc=None,
            timeout=None,
            prefilter: Callable = None,
            transform: Callable = None,
    ):
        super().__init__(data_dir, test_num, nproc, timeout, prefilter, transform)

    @staticmethod
    def to_data(mol: Molecule, *args, **kwargs) -> Data:
        """ Convert hotpot.Molecule to PyG Data object """
        x, x_names = extract_atom_attrs(mol)
        y_names = args[0]

        edge_attr_names = ('bond_order', 'is_aromatic', 'is_metal_ligand_bond')
        edge_index, edge_attr = extract_bond_attrs(mol, edge_attr_names)

        # Organize pair data
        pair_index, pair_attr, pair_attr_names = extract_atom_pairs(mol)

        y_getter = attrgetter(*y_names[1:])
        y = torch.tensor([y_getter(mol)]).float()

        # Process mol Ring attribute
        ring_attr_names = ('is_aromatic', 'has_metal')
        mol_ring_nums, ring_node_index, ring_node_nums, mol_ring_node_nums, ring_attr = extract_ring_attrs(mol, ring_attr_names)
        return Data(
            x=x,
            x_names=x_names,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_attr_names=edge_attr_names,
            pair_index=pair_index,
            pair_attr=pair_attr,
            pair_attr_names=pair_attr_names,
            y=y,
            y_names=y_names[1:],
            identifier=mol.identifier,
            mol_rings_nums=mol_ring_nums,
            rings_node_index=ring_node_index,
            rings_node_nums=ring_node_nums,
            mol_rings_node_nums=mol_ring_node_nums,
            rings_attr=ring_attr,
            rings_attr_names=ring_attr_names
        )

    def _get_data(self, mol, data_dir, *args, **kwargs):
        data = self.to_data(mol, *args, **kwargs)
        torch.save(data, osp.join(data_dir, f"{data.identifier}.pt"))

    def process(self):
        # Load raw data
        raw_data = tmqm.TmQmDataset(nproc=self.nproc)

        if self.nproc == 1:
            for i, mol in enumerate(tqdm(raw_data, "Processing tmQm dataset")):

                if self.test_num and self.test_num <= i:
                    break

                if self.prefilter and not self.prefilter(mol):
                    continue

                self._get_data(mol, tmqm.TmQmDataset.mol_attrs, self.data_dir)

        else:
            processes = []
            for i, mol in enumerate(tqdm(raw_data, "Processing tmQm dataset")):

                if self.test_num and self.test_num <= i:
                    break

                if self.prefilter and not self.prefilter(mol):
                    continue

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
                    args=(mol, tmqm.TmQmDataset.mol_attrs, self.data_dir)
                )
                p.start()
                processes.append(p)

            for p in processes:
                p.join()
                p.terminate()

        print('Process Done!!!')

    def __iter__(self):
        for idx in range(self.len):
            yield self.get(idx)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.get(idx)

    def get(self, idx):
        data = torch.load(self.datalist[idx], weights_only=False)
        return data

