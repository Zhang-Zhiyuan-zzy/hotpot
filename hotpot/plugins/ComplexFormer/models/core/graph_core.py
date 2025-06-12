# -*- coding: utf-8 -*-
"""
===========================================================
 Project   : hotpot
 File      : graph_core
 Created   : 2025/6/10 16:45
 Author    : zhang
 Python    : 
-----------------------------------------------------------
 Description
 ----------------------------------------------------------
 
===========================================================
"""
from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from .base import CoreBase
from .node_processor import NodeProcessor
from .graph import CompleteGraph


__all__ = [
    'CompleteGraphExtractor',
    'CompleteGraphCore'
]

class CompleteGraphExtractor:
    @staticmethod
    def extract_atom_vec(mol_vec: Tensor, node_vec: list[Tensor], ring_vec: list[Tensor], batch, batch_getter=None) -> Tensor:
        return torch.cat(node_vec)

    @staticmethod
    def extract_cbond_pair(mol_vec: Tensor, node_vec: list[Tensor], ring_vec: list[Tensor], batch, batch_getter=None) -> Tensor:
        Znode = CompleteGraphCore.extract_atom_vec(mol_vec, node_vec, ring_vec, batch, batch_getter)
        cbond_index = batch.cbond_index

        upper_Znode = Znode[cbond_index[0]]
        lower_Znode = Znode[cbond_index[1]]

        # cbond_feature = torch.cat([upper_Znode, lower_Znode], dim=1)
        cbond_feature = upper_Znode + lower_Znode

        return cbond_feature

    @staticmethod
    def extract_pair_vec(mol_vec: Tensor, node_vec: list[Tensor], ring_vec: list[Tensor], batch, batch_getter=None) -> Tensor:
        Znode = CompleteGraphCore.extract_atom_vec(mol_vec, node_vec, ring_vec, batch, batch_getter)

        pair_index = batch_getter.pair_index

        upper_Znode = Znode[pair_index[0]]
        lower_Znode = Znode[pair_index[1]]

        return (upper_Znode + lower_Znode) / 2

    @staticmethod
    def extract_ring_vec(mol_vec: Tensor, node_vec: list[Tensor], ring_vec: list[Tensor], batch, batch_getter=None):
        return torch.cat(ring_vec)

    @staticmethod
    def extract_mol_vec(mol_vec: Tensor, node_vec: list[Tensor], ring_vec: list[Tensor], batch, batch_getter=None):
        return mol_vec


class CompleteGraphCore(CoreBase):
    extractor_class = CompleteGraphExtractor
    def __init__(
            self,
            x_dim: int,
            vec_dim: int = 512,
            x_label_nums: Optional[int] = None,
            graph_model: nn.Module = None,

            # Rings Transformer arguments
            ring_layers: int = 1,
            ring_nheads: int = 2,
            ring_encoder_kw: dict = None,
            ring_encoder_block_kw: dict = None,

            # Molecular Transformer arguments
            mol_layers: int = 1,
            mol_nheads: int = 4,
            mol_encoder_kw: dict = None,
            mol_encoder_block_kw: dict = None,
            **kwargs,
    ):
        super(CompleteGraphCore, self).__init__(vec_dim, x_label_nums)
        self.node_processor = NodeProcessor(x_dim, vec_dim, x_label_nums=x_label_nums, graph_model=graph_model)

        self.ring_encoder = CompleteGraph(vec_dim, ring_layers, ring_nheads)
        self.mol_encoder = CompleteGraph(vec_dim, mol_layers, mol_nheads)

    def forward(
            self,
            x, edge_index, edge_attr, rings_node_index, rings_node_nums, mol_rings_nums, batch, ptr,
            *,
            xyz: Optional[Union[torch.Tensor, torch.nested.nested_tensor]] = None,
    ) -> (torch.Tensor, list[torch.Tensor], list[torch.Tensor]):
        x = self.node_processor(x, edge_index, ptr, xyz=xyz)

        xr = self.ring_encoder(x, rings_node_index, rings_node_nums)
        mol_vec, atom_vec, ring_vec = self.mol_encoder(x, xr, mol_rings_nums, ptr, batch)
        return mol_vec, atom_vec, ring_vec

    @property
    def x_mask_vec(self):
        return self.node_processor.x_mask_vec
