"""
python v3.9.0
@Project: hotpot
@File   : data
@Auther : Zhiyuan Zhang
@Data   : 2023/8/8
@Time   : 13:43
"""
from os import PathLike
from pathlib import Path
from typing import *

import torch
import torch_geometric as pyg

from hotpot.cheminfo import Molecule
from hotpot.bundle import MolBundle


def mols2graphs(mols: Union[Sequence[Molecule], MolBundle], *feature_names: str):
    """"""
    data = []
    for mol in mols:
        idt, feat, adj = mol.graph_representation(*feature_names)
        data.append(pyg.data.Data(x=feat, edge_index=adj, idt=idt))

    return data


class MolGraph(pyg.data.InMemoryDataset):
    """"""
    def __init__(
            self, root: Union[str, Path], mols: Union[list[Molecule], Generator],
            transform=None, pre_transform=None, pre_filter=None, integral_coords=False
    ):
        """"""
        self.mols = mols
        self.integral_coords = integral_coords
        pyg.data.InMemoryDataset.__init__(self, root, transform, pre_transform, pre_filter)
        self._data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return ['data.pt']

    def process(self):
        """"""
        data_list = []
        for mol in self.mols:
            data = pyg.data.Data(
                x=torch.tensor(mol.feature_matrix()).contiguous(),
                c=torch.tensor(mol.coordinates).contiguous(),
                edge_index=torch.tensor(mol.link_matrix, dtype=torch.long).contiguous(),
                edge_attr=torch.tensor(mol.link_order, dtype=torch.long).contiguous(),
                idt=mol.identifier.split('/')[-1].split('.')[0],
                y=torch.tensor(mol.energy, dtype=torch.float)
            )
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class TemporalMolGraph(MolGraph):
    """"""
