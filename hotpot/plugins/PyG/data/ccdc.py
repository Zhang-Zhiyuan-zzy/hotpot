import os.path as osp
from glob import glob
from typing import Callable

from tqdm import tqdm
import torch
from torch_geometric.data import Data

import hotpot as hp
from hotpot.cheminfo.core import Molecule

from ._base import *
from .utils import *


class ComplexDataset(BaseDataset):
    __attr_names__ = {
        'x': ('atomic_number','n', 's', 'p', 'd', 'f', 'g',
              'formal_charge','partial_charge', 'is_aromatic',
              'x', 'y', 'z',
              'valence', 'implicit_hydrogens', 'is_metal'),

        'edge_attr': ('bond_order', 'is_aromatic', 'is_metal_ligand_bond'),
        'pair_attr': ('wiberg_bond_order', 'length_shortest_path'),
        'ring_attr': ('is_aromatic', 'has_metal'),
    }

    def __init__(
            self,
            data_dir: str,
            raw_mol2_dir: str = None,
            test_num=None,
            nproc=None,
            timeout=None,
            prefilter: Callable = None,
            transform: Callable = None,
    ):
        self.raw_mol2_dir = raw_mol2_dir
        super().__init__(data_dir, test_num, nproc, timeout, prefilter, transform)

    def _get_data(self, mol, data_dir, *args, **kwargs):
        data = self.to_data(mol)
        torch.save(data, osp.join(data_dir, f"{data.identifier}.pt"))

    @staticmethod
    def to_data(mol: Molecule, *args, **kwargs) -> Data:
        x, x_names = extract_atom_attrs(mol)

        edge_attr_names = ('bond_order', 'is_aromatic', 'is_metal_ligand_bond')
        edge_index, edge_attr = extract_bond_attrs(mol, edge_attr_names)

        # Organize pair data
        pair_index, pair_attr, pair_attr_names = extract_atom_pairs(mol)

        # Process mol Ring attribute
        ring_attr_names = ('is_aromatic', 'has_metal')
        mol_ring_nums, ring_node_index, ring_node_nums, mol_ring_node_nums, ring_attr = extract_ring_attrs(mol, ring_attr_names)
        y = None
        y_names = None

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
            y_names=y_names,
            identifier=mol.identifier,
            mol_rings_nums=mol_ring_nums,
            rings_node_index=ring_node_index,
            rings_node_nums=ring_node_nums,
            mol_rings_node_nums=mol_ring_node_nums,
            rings_attr=ring_attr,
            rings_attr_names=ring_attr_names
        )

    def process(self):
        if self.nproc == 1:
            for i, mol_path in enumerate(tqdm(self.raw_mol2_dir, "Processing CCDC raw data")):
                try:
                    mol = next(hp.MolReader(mol_path, 'mol2'))
                    mol.identifier = osp.splitext(osp.basename(mol_path))[0]
                except StopIteration:
                    continue

                if self.test_num and self.test_num <= i:
                    break

                if self.prefilter and not self.prefilter(mol):
                    continue

                self._get_data(mol, self.data_dir)

        else:
            processes = []
            for i, mol_path in enumerate(tqdm(glob(osp.join(self.raw_mol2_dir, '*.mol2')), "Processing CCDC raw data")):
                try:
                    mol = next(hp.MolReader(mol_path, 'mol2'))
                    mol.identifier = osp.splitext(osp.basename(mol_path))[0]
                except StopIteration:
                    continue

                if self.test_num and self.test_num <= i:
                    break

                if self.prefilter and not self.prefilter(mol):
                    continue

                self.mp_process(mol, processes)

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
        data = torch.load(self.datalist[idx])
        return data

