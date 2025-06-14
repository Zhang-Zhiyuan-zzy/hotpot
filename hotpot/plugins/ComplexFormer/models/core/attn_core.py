# -*- coding: utf-8 -*-
"""
===========================================================
 Project   : hotpot
 File      : attn_core
 Created   : 2025/6/10 15:33
 Author    : zhang
 Python    : 
-----------------------------------------------------------
 Description
 ----------------------------------------------------------
 
===========================================================
"""
from typing import Optional, Union, Literal, Sequence

import torch
import torch.nn as nn

import torch_geometric.nn as pygnn

from .. import utils
from .base import CoreBase
from .node_processor import NodeProcessor
from ._utils import split_padding, seq_absmax_pooling


__all__ = [
    'AttnExtractor',
    'AttnCore'
]

class AttnExtractor:
    @staticmethod
    def extract_atom_vec(seq, X_mask, R_mask, batch, batch_getter=None):
        Znode = []
        node_seq = seq[:, 1:X_mask.shape[-1]+1]
        for s, m in zip(node_seq, X_mask.sum(dim=-1)):
            Znode.append(s[:m])

        return torch.cat(Znode, dim=0)

    @staticmethod
    def extract_bond_vec(seq, X_mask, R_mask, batch, batch_getter=None):
        Znode = AttnExtractor.extract_atom_vec(seq, X_mask, R_mask, batch, batch_getter)
        edge_index = batch.edge_index

        upper_Znode = Znode[edge_index[0]]
        lower_Znode = Znode[edge_index[1]]

        return (upper_Znode + lower_Znode) / 2

    @staticmethod
    def extract_metal_vec(seq, X_mask, R_mask, batch, batch_getter=None):
        metal_idx = utils.where_metal(batch.x[:, 0])
        Znode = AttnExtractor.extract_atom_vec(seq, X_mask, R_mask, batch, batch_getter)
        return Znode[metal_idx]

    @staticmethod
    def extract_cbond_pair(seq, X_mask, R_mask, batch, batch_getter=None):
        Znode = AttnExtractor.extract_atom_vec(seq, X_mask, R_mask, batch, batch_getter)
        cbond_index = batch.cbond_index

        upper_Znode = Znode[cbond_index[0]]
        lower_Znode = Znode[cbond_index[1]]

        # cbond_feature = torch.cat([upper_Znode, lower_Znode], dim=1)
        #
        # assert cbond_feature.shape == (upper_Znode.shape[0], upper_Znode.shape[1] * 2)

        # return cbond_feature
        return (upper_Znode + lower_Znode) / 2

    @staticmethod
    def extract_pair_vec(seq, X_mask, R_mask, batch, batch_getter=None):
        Znode = AttnExtractor.extract_atom_vec(seq, X_mask, R_mask, batch, batch_getter)

        pair_index = batch.pair_index

        upper_Znode = Znode[pair_index[0]]
        lower_Znode = Znode[pair_index[1]]

        return (upper_Znode + lower_Znode) / 2

    @staticmethod
    def extract_ring_vec(seq, X_mask, R_mask, batch, batch_getter=None):
        Zring = []

        ring_seq = seq[:, -R_mask.shape[-1]-1:-1]
        assert ring_seq.shape[:2] == R_mask.shape

        for s, m in zip(ring_seq, R_mask.sum(dim=-1)):
            Zring.append(s[:m])

        return torch.cat(Zring, dim=0)

    @staticmethod
    def extract_mol_vec(seq, X_mask, R_mask, batch, batch_getter=None):
        return seq[:, 1]

NodeProcessorType = Literal['graph', 'se3']
class AttnCore(CoreBase):
    # Feature Extractors
    extractor_class = AttnExtractor
    def __init__(
            self,
            x_dim: int,
            vec_dim: int = 512,
            node_processor: NodeProcessorType = 'graph',
            x_label_nums: Optional[int] = None,
            graph_model: nn.Module = None,
            cloud_model: nn.Module = None,

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

            # Mol level info MLP
            mol_level_net: Optional[Union[nn.Module, Sequence[int]]] = None,
            **kwargs,
    ):
        super(AttnCore, self).__init__(vec_dim, x_label_nums)
        self.node_processor = NodeProcessor(
            x_dim, vec_dim,
            x_label_nums=x_label_nums,
            graph_model=graph_model,
            cloud_model=cloud_model,
        )

        self.ring_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=vec_dim,
                nhead=ring_nheads,
                dim_feedforward=1024,
                batch_first=True,
            ), num_layers=ring_layers,
        )
        self.mol_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=vec_dim,
                nhead=mol_nheads,
                dim_feedforward=1024,
                batch_first=True,
            ), num_layers=mol_layers,
        )

        self.CLS = nn.Parameter(torch.randn(1, vec_dim))
        self.RING = nn.Parameter(torch.randn(1, vec_dim))
        self.END = nn.Parameter(torch.randn(1, vec_dim))

        # Definition of MolInfo Net
        if isinstance(mol_level_net, (list, tuple)):
            mol_level_net = list(mol_level_net)
            if mol_level_net[-1] != vec_dim:
                mol_level_net = mol_level_net + [vec_dim]

            self.mol_info_net = pygnn.MLP(mol_level_net)

        elif isinstance(mol_level_net, nn.Module):
            self.mol_info_net = mol_level_net
        else:
            self.mol_info_net = None

    @property
    def x_mask_vec(self):
        return self.node_processor.x_mask_vec

    def _assemble_sequence(self, X, Xr, X_mask, Xr_mask):
        CLS = torch.tile(self.CLS, (X.shape[0], 1, 1))
        RING = torch.tile(self.RING, (X.shape[0], 1, 1))
        END = torch.tile(self.END, (X.shape[0], 1, 1))

        seq = torch.cat((CLS, X, RING, Xr, END), dim=-2)
        seq_padding_mask = torch.cat([
            torch.zeros((X.shape[0], 1), dtype=torch.bool, device=seq.device),
            X_mask,
            torch.zeros((X.shape[0], 1), dtype=torch.bool, device=seq.device),
            Xr_mask,
            torch.zeros((X.shape[0], 1), dtype=torch.bool, device=seq.device),
        ], dim=1)

        return seq, seq_padding_mask

    def _rings_attention(self, x, rings_node_index, rings_node_nums):
        x = x[rings_node_index]
        X, padding_mask = split_padding(x, rings_node_nums)
        X = self.ring_encoder(X, src_key_padding_mask=padding_mask)
        return seq_absmax_pooling(X)

    def _mol_attention(self, x, xr, mol_rings_nums, ptr, batch):
        X, X_mask = split_padding(x, ptr[1:] - ptr[:-1])
        Xr, Xr_mask = split_padding(xr, mol_rings_nums)
        seq, seq_padding_mask = self._assemble_sequence(X, Xr, X_mask, Xr_mask)
        seq = self.mol_encoder(seq, src_key_padding_mask=seq_padding_mask)
        return seq, torch.logical_not(X_mask), torch.logical_not(Xr_mask)

    def forward(
            self,
            x, edge_index, edge_attr, rings_node_index, rings_node_nums, mol_rings_nums, batch, ptr,
            *,
            xyz: Optional[Union[torch.Tensor, torch.nested.nested_tensor]] = None,
            mol_level_info: Optional[Union[torch.Tensor, torch.nested.nested_tensor]] = None,
    ):
        x = self.node_processor(x, edge_index, batch, xyz=xyz)
        xr = self._rings_attention(x, rings_node_index, rings_node_nums)
        seq, X_not_mask, Xr_not_mask = self._mol_attention(x, xr, mol_rings_nums, ptr, batch)

        # Add mol level info
        if self.mol_info_net is not None and isinstance(mol_level_info, torch.Tensor):
            assert seq.shape[0] == mol_level_info.shape[0]
            mol_info_vec = self.mol_info_net(mol_level_info)
            seq[:, 0, :] = seq[:, 0, :] + mol_info_vec

        return seq, X_not_mask, Xr_not_mask
