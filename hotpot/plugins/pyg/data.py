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
from typing import Iterable
from operator import attrgetter
import multiprocessing as mp

from tqdm import tqdm
import numpy as np
import torch
from torch_geometric.data import Data
from triton.language import dtype

from hotpot.cheminfo.core import Molecule, Atom, Bond, AtomPair
from hotpot.dataset import tmqm
from hotpot.utils import tools


def direct_edge_to_indirect(edge_index: torch.Tensor) -> torch.Tensor:
    """"""
    assert edge_index.shape[0] == 2
    return torch.cat([edge_index, edge_index.flip(0)], dim=1)

def to_pyg_data(mol: Molecule, y_names: Iterable[str]) -> Data:
    """ Convert hotpot.Molecule to PyG Data object """
    x_names = Atom._attrs_enumerator[:15]
    additional_attr_names = ('is_metal',)
    x_names = x_names + additional_attr_names
    additional_attr_getter = attrgetter(*additional_attr_names)
    x = torch.from_numpy(np.array([a.attrs[:15].tolist() + [additional_attr_getter(a)] for a in mol.atoms]))

    edge_attr_names = ('bond_order', 'is_aromatic', 'is_metal_ligand_bond')
    bond_attr_getter = attrgetter(*edge_attr_names)
    edge_index = direct_edge_to_indirect(torch.tensor(mol.link_matrix).T)
    edge_attr = torch.from_numpy(np.array([bond_attr_getter(b) for b in mol.bonds]))

    pair_index = direct_edge_to_indirect(torch.tensor(mol.atom_pairs.idx_matrix).T)
    pair_attr = torch.tensor([p.attrs for k, p in mol.atom_pairs.items()])
    pair_attr_name = AtomPair.attr_names

    y_getter = attrgetter(*y_names[1:])
    y = torch.tensor([y_getter(mol)])

    # Process mol Ring attribute
    rings = mol.ligand_rings
    ring_attr_names = ('is_aromatic', 'has_metal')
    ring_attr_getter = attrgetter(*ring_attr_names)
    rings_node_index = [r.atoms_indices for r in rings]
    rings_node_nums = [len(rni) for rni in rings_node_index]
    if rings_node_index:
        mol_rings_nums = torch.tensor([len(rings_node_nums)])
        rings_node_index = torch.tensor(sum(rings_node_index, start=[]))
        rings_node_nums = torch.tensor(rings_node_nums)
        mol_rings_node_nums = torch.tensor([rings_node_nums.sum()])
        rings_attr = torch.from_numpy(np.array([ring_attr_getter(r) for r in rings])).float()
    else:
        mol_rings_nums = torch.tensor([0])
        rings_node_index = torch.tensor([])
        rings_node_nums = torch.tensor([])
        mol_rings_node_nums = torch.tensor([])
        rings_attr = torch.tensor([])

    return Data(
        x=x,
        x_names=x_names,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_attr_names=edge_attr_names,
        pair_index=pair_index,
        pair_attr=pair_attr,
        pair_attr_name=pair_attr_name,
        y=y,
        y_names=y_names[1:],
        identifier=mol.identifier,
        mol_rings_nums=mol_rings_nums,
        rings_node_index=rings_node_index,
        rings_node_nums=rings_node_nums,
        mol_rings_node_nums=mol_rings_node_nums,
        rings_attr=rings_attr,
    )


class tmQmDataset:
    def __init__(self, root, test_num=None, nproc=None, timeout=None):
        self.data_dir = root

        if nproc is None:
            self.nproc = os.cpu_count() // 2
        else:
            self.nproc = nproc

        self.tmqm = tmqm.TmQmDataset(nproc=self.nproc)
        self.test_num = test_num
        self.timeout = timeout

        if not osp.exists(root):
            os.mkdir(root)

        self.datalist = [p for p in glob(osp.join(self.data_dir, '*.pt'))]
        if not self.datalist:
            self.process()
            self.datalist = [p for p in glob(osp.join(self.data_dir, '*.pt'))]

        self.len = len(self.datalist)

    @staticmethod
    def _get_data(mol, y_attrs, data_dir):
        data = to_pyg_data(mol, y_attrs)
        torch.save(data, osp.join(data_dir, f"{data.identifier}.pt"))

    def process(self):
        if self.nproc == 1:
            for i, mol in enumerate(tqdm(self.tmqm, "Processing tmQm dataset")):

                if self.test_num and self.test_num <= i:
                    break

                self._get_data(mol, tmqm.TmQmDataset.mol_attrs, self.data_dir)

        else:
            processes = []
            for i, mol in enumerate(tqdm(self.tmqm, "Processing tmQm dataset")):

                if self.test_num and self.test_num <= i:
                    break

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
        data = torch.load(self.datalist[idx])
        return data
