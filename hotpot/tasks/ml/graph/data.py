"""
python v3.9.0
@Project: hotpot
@File   : data
@Auther : Zhiyuan Zhang
@Data   : 2023/8/8
@Time   : 13:43
"""
from typing import Union, Sequence

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

