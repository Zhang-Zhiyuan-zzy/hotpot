"""
python v3.9.0
@Project: hotpot
@File   : data
@Auther : Zhiyuan Zhang
@Data   : 2025/1/13
@Time   : 15:42
"""
import os
import time
import os.path as osp
from glob import glob
from typing import Iterable, Callable
from operator import attrgetter
import multiprocessing as mp
from abc import ABC, abstractmethod

from tqdm import tqdm
import numpy as np
import torch
from torch_geometric.data import Data
from triton.language import dtype

import hotpot as hp
from hotpot.cheminfo.core import Molecule, Atom, Bond, AtomPair
from hotpot.dataset import tmqm
from hotpot.utils import tools


def direct_edge_to_indirect(attr_or_index: torch.Tensor, is_index=True) -> torch.Tensor:
    """"""
    # assert attr_or_index.shape[0] == 2
    if is_index:
        return torch.cat([attr_or_index, attr_or_index.flip(0)], dim=1)
    else:
        return torch.cat([attr_or_index, attr_or_index.flip(0)], dim=0)


def _extract_atom_attrs(mol: Molecule) -> (torch.Tensor, list):
    x_names = Atom._attrs_enumerator[:15]
    additional_attr_names = ('is_metal',)
    x_names = x_names + additional_attr_names
    additional_attr_getter = attrgetter(*additional_attr_names)
    x = torch.from_numpy(np.array([a.attrs[:15].tolist() + [additional_attr_getter(a)] for a in mol.atoms])).float()

    return x, x_names

def _extract_bond_attrs(mol: Molecule, edge_attr_names: Iterable[str]) -> (torch.Tensor, torch.Tensor):
    bond_attr_getter = attrgetter(*edge_attr_names)
    edge_index = direct_edge_to_indirect(torch.tensor(mol.link_matrix).T).long()
    edge_attr = direct_edge_to_indirect(torch.from_numpy(np.array([(bond_attr_getter(b)) for b in mol.bonds])), is_index=False).float()

    return edge_index, edge_attr

def _extract_atom_pairs(mol: Molecule) -> (torch.Tensor, torch.Tensor, list):
    atom_pairs = mol.atom_pairs
    atom_pairs.update_pairs()
    pair_index = torch.tensor(atom_pairs.idx_matrix).T.long()
    pair_attr = torch.tensor([p.attrs for k, p in atom_pairs.items()]).float()
    pair_attr_names = AtomPair.attr_names

    return pair_index, pair_attr, pair_attr_names

def _extract_ring_attrs(mol: Molecule, ring_attr_names: Iterable[str]) -> (torch.Tensor, torch.Tensor):
    rings = mol.ligand_rings
    ring_attr_getter = attrgetter(*ring_attr_names)
    rings_node_index = [r.atoms_indices for r in rings]
    rings_node_nums = [len(rni) for rni in rings_node_index]
    if rings_node_index:
        mol_rings_nums = torch.tensor([len(rings_node_nums)], dtype=torch.long)
        rings_node_index = torch.tensor(sum(rings_node_index, start=[]), dtype=torch.long)
        rings_node_nums = torch.tensor(rings_node_nums, dtype=torch.int)
        mol_rings_node_nums = torch.tensor([rings_node_nums.sum()], dtype=torch.int)
        rings_attr = torch.from_numpy(np.array([ring_attr_getter(r) for r in rings])).float()
    else:
        mol_rings_nums = torch.tensor([0])
        rings_node_index = torch.tensor([])
        rings_node_nums = torch.tensor([])
        mol_rings_node_nums = torch.tensor([])
        rings_attr = torch.tensor([])

    return mol_rings_nums, rings_node_index, rings_node_nums, mol_rings_node_nums, rings_attr


def merge_individual_data_to_block(indiv_data_dir, merged_data_dir, bundle_size: int = 200000):
    list_data = []
    total = 0
    for i, p in enumerate(tqdm(glob(osp.join(indiv_data_dir, "*.pt")), 'Merging data'), 1):
        if torch.__version__ >= '2.6':
            list_data.append(torch.load(p, weights_only=False))
        else:
            list_data.append(torch.load(p))

        if i % bundle_size == 0:
            torch.save(list_data, osp.join(merged_data_dir, f"{i}.pt"))
            list_data = []
            total += len(list_data)

    if list_data:
        total += len(list_data)
        torch.save(list_data, osp.join(merged_data_dir, f"{total}.pt"))


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


class InMemoryDataset:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_files = os.listdir(data_dir)
        self.slice = tuple(sorted([int(d.split('.')[0]) for d in self.data_files]))
        self.list_data = torch.load(osp.join(data_dir, f'{self.slice[0]}.pt'))
        self.start = 0
        self.end = self.slice[0]
        self.len = self.slice[-1]

    @staticmethod
    def merge_individual_data_to_block(indiv_data_dir, merged_data_dir, bundle_size: int = 200000):
        merge_individual_data_to_block(indiv_data_dir, merged_data_dir, bundle_size)

    def get(self, idx):
        if idx < 0:
            idx += self.slice[-1]
            if idx < 0:
                raise IndexError(f"Index out of range, dataset hasonly {self.slice[-1]} item")

        if self.start <= idx < self.end:
            return self.list_data[idx-self.start]
        elif idx >= self.slice[-1]:
            raise IndexError(f"Index out of range, {idx} with only {self.slice[-1]} item")
        else:
            slice_i = next((i for i in range(len(self.slice)) if self.slice[i] > idx))
            self.start = self.slice[slice_i-1] if slice_i > 0 else 0
            self.end = self.slice[slice_i]
            self.list_data = torch.load(osp.join(self.data_dir, f'{self.end}.pt'))
            return self.list_data[idx-self.start]

    def __iter__(self):
        for idx in range(self.len):
            yield self.get(idx)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.get(idx)


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
        x, x_names = _extract_atom_attrs(mol)
        y_names = args[0]

        edge_attr_names = ('bond_order', 'is_aromatic', 'is_metal_ligand_bond')
        edge_index, edge_attr = _extract_bond_attrs(mol, edge_attr_names)

        # Organize pair data
        pair_index, pair_attr, pair_attr_names = _extract_atom_pairs(mol)

        y_getter = attrgetter(*y_names[1:])
        y = torch.tensor([y_getter(mol)]).float()

        # Process mol Ring attribute
        ring_attr_names = ('is_aromatic', 'has_metal')
        mol_ring_nums, ring_node_index, ring_node_nums, mol_ring_node_nums, ring_attr = _extract_ring_attrs(mol,
                                                                                                                 ring_attr_names)
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
            mol_ring_nums=mol_ring_nums,
            ring_node_index=ring_node_index,
            ring_node_nums=ring_node_nums,
            mol_ring_node_nums=mol_ring_node_nums,
            ring_attr=ring_attr,
            ring_attr_names=ring_attr_names
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
        x, x_names = _extract_atom_attrs(mol)

        edge_attr_names = ('bond_order', 'is_aromatic', 'is_metal_ligand_bond')
        edge_index, edge_attr = _extract_bond_attrs(mol, edge_attr_names)

        # Organize pair data
        pair_index, pair_attr, pair_attr_names = _extract_atom_pairs(mol)

        # Process mol Ring attribute
        ring_attr_names = ('is_aromatic', 'has_metal')
        mol_ring_nums, ring_node_index, ring_node_nums, mol_ring_node_nums, ring_attr = _extract_ring_attrs(mol,
                                                                                                                 ring_attr_names)
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
            mol_ring_nums=mol_ring_nums,
            ring_node_index=ring_node_index,
            ring_node_nums=ring_node_nums,
            mol_ring_node_nums=mol_ring_node_nums,
            ring_attr=ring_attr,
            ring_attr_names=ring_attr_names
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
