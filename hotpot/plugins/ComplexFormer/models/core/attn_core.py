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
from collections.abc import Iterable
from typing import Optional, Union, Literal, Sequence, overload

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

import torch_geometric.nn as pygnn

from .. import utils
from .base import CoreBase
from .node_processor import NodeProcessor
from .envs_encoder import SolventNet
from ._utils import split_padding, seq_absmax_pooling


__all__ = [
    'AttnExtractor',
    'AttnCore'
]

class AttnExtractor:
    @staticmethod
    def extract_atom_vec(seq, X_mask, R_mask, batch):
        Znode = []
        node_seq = seq[:, 1:X_mask.shape[-1]+1]
        for s, msk in zip(node_seq, X_mask):
            Znode.append(s[msk])

        return torch.cat(Znode, dim=0)

    @staticmethod
    def extract_bond_vec(seq, X_mask, R_mask, batch):
        Znode = AttnExtractor.extract_atom_vec(seq, X_mask, R_mask, batch)
        edge_index = batch.edge_index

        upper_Znode = Znode[edge_index[0]]
        lower_Znode = Znode[edge_index[1]]

        return (upper_Znode + lower_Znode) / 2

    @staticmethod
    def extract_metal_vec(seq, X_mask, R_mask, batch):
        metal_idx = utils.where_metal(batch.x[:, 0])
        Znode = AttnExtractor.extract_atom_vec(seq, X_mask, R_mask, batch)
        return Znode[metal_idx]

    @staticmethod
    def extract_cbond_pair(seq, X_mask, R_mask, batch):
        Znode = AttnExtractor.extract_atom_vec(seq, X_mask, R_mask, batch)
        cbond_index = batch.cbond_index

        upper_Znode = Znode[cbond_index[0]]
        lower_Znode = Znode[cbond_index[1]]

        # return cbond_feature
        return (upper_Znode + lower_Znode) / 2

    @staticmethod
    def extract_pair_vec(seq, X_mask, R_mask, batch):
        Znode = AttnExtractor.extract_atom_vec(seq, X_mask, R_mask, batch)

        pair_index = batch.pair_index

        upper_Znode = Znode[pair_index[0]]
        lower_Znode = Znode[pair_index[1]]

        return (upper_Znode + lower_Znode) / 2

    @staticmethod
    def extract_ring_vec(seq, X_mask, R_mask, batch):
        Zring = []

        ring_seq = seq[:, -R_mask.shape[-1]-1:-1]
        assert ring_seq.shape[:2] == R_mask.shape

        for s, m in zip(ring_seq, R_mask.sum(dim=-1)):
            Zring.append(s[:m])

        return torch.cat(Zring, dim=0)

    @staticmethod
    def extract_mol_vec(seq, X_mask, R_mask, batch):
        return seq[:, 1]

NodeProcessorType = Literal['graph', 'se3']
class AttnCore(CoreBase):
    # Feature Extractors
    extractor_class = AttnExtractor
    def __init__(
            self,
            vec_dim: int = 512,
            emb_type: Literal['proj', 'atom', 'orbital'] = 'proj',
            x_label_nums: Optional[int] = None,
            graph_layer: int = 6,

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

            # Solvent encoder args
            with_sol_encoder: bool = False,
            sol_props_nums: Optional[int] = None,
            sol_props_net_layers: int = 2,
            sol_gnn_layers: int = 3,
            sol_gnn_kw: Optional[dict] = None,

            # Media encoder args
            with_med_encoder: bool = False,
            med_props_nums: Optional[int] = None,
            med_props_net_layers: int = 2,
            med_gnn_layers: int = 3,
            med_gnn_kw: Optional[dict] = None,

            # Mol level info MLP
            mol_level_net: Optional[Union[nn.Module, Sequence[int]]] = None,
            **kwargs,
    ):
        self.emb_type = emb_type
        super(AttnCore, self).__init__(vec_dim, x_label_nums)
        self.node_processor = NodeProcessor(
            vec_dim,
            emb_type=emb_type,
            graph_layer=graph_layer
        )

        self.ring_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=vec_dim,
                nhead=ring_nheads,
                dim_feedforward=ring_encoder_kw.get('dim_feedforward', 1024) if isinstance(ring_encoder_kw, dict) else 1024,
                batch_first=True,
            ), num_layers=ring_layers, **(ring_encoder_block_kw or {})
        )
        self.mol_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=vec_dim,
                nhead=mol_nheads,
                dim_feedforward=mol_encoder_kw.get('dim_feedforward', 1024) if isinstance(mol_encoder_kw, dict) else 1024,
                batch_first=True,
            ), num_layers=mol_layers, **(ring_encoder_block_kw or {})
        )

        self.CLS = nn.Parameter(torch.randn(1, vec_dim))
        self.RING = nn.Parameter(torch.randn(1, vec_dim))
        self.END = nn.Parameter(torch.randn(1, vec_dim))

        # Definition of a Solvent net
        if with_sol_encoder:
            self.sol_encoder = SolventNet(
                vec_dim=vec_dim,
                node_embedder=self.node_processor.node_embedder,
                props_nums=sol_props_nums,
                props_net_layers=sol_props_net_layers,
                gnn_layers=sol_gnn_layers,
                gnn_kw=sol_gnn_kw,
            )
        else:
            self.sol_encoder = None

        # Definition of a Media net
        if with_med_encoder:
            self.med_encoder = SolventNet(
                vec_dim=vec_dim,
                node_embedder=self.node_processor.node_embedder,
                props_nums=med_props_nums,
                props_net_layers=med_props_net_layers,
                gnn_layers=med_gnn_layers,
                gnn_kw=med_gnn_kw,
            )
        else:
            self.med_encoder = None

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
    def labeled_elements(self) -> Optional[bool]:
        if getattr(self, 'node_processor', None) is None:
            return None

        if isinstance(self.elem_emb_vec, torch.Tensor):
            return True
        return False

    @property
    def x_mask_vec(self):
        return self.node_processor.x_mask_vec

    @property
    def node_mask(self):
        return self.node_processor.node_mask

    @property
    def elem_emb_vec(self) -> Optional[torch.Tensor]:
        if isinstance(emb_net := getattr(self.node_processor, 'x_emb'), nn.Embedding):
            return emb_net.weight
        return None

    def _assemble_sequence(
            self, X, Xr, X_mask, Xr_mask,
            sol_vec: torch.Tensor = None,
            med_vec: torch.Tensor = None,
            env_vec: torch.Tensor = None,
    ):
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

        if isinstance(sol_vec, torch.Tensor):
            seq = torch.cat([seq, sol_vec.unsqueeze(-2)], dim=-2)

        if isinstance(med_vec, torch.Tensor):
            seq = torch.cat([seq, med_vec.unsqueeze(-2)], dim=-2)

        if isinstance(env_vec, torch.Tensor):
            seq = torch.cat([seq, env_vec.unsqueeze(-2)], dim=-2)

        padding_cols = seq.size(-2) - seq_padding_mask.size(-1)
        if padding_cols > 0:
            seq_padding_mask = torch.cat(
                [
                    seq_padding_mask,
                    torch.zeros([X.size(0), padding_cols], dtype=torch.bool, device=seq.device)
                ], dim=-1)

        return seq, seq_padding_mask

    def _rings_attention(self, x, rings_node_index, rings_node_nums):
        x = x[rings_node_index]
        X, padding_mask = split_padding(x, rings_node_nums)
        torch._check(1 != X.shape[1])
        X = self.ring_encoder(X, src_key_padding_mask=padding_mask)
        X = X.masked_fill_(padding_mask.unsqueeze(-1), 0.)
        return seq_absmax_pooling(X)

    def _mol_attention(
            self, x, xr, mol_rings_nums, ptr,
            sol_vec: torch.Tensor = None,
            med_vec: torch.Tensor = None,
            env_vec: torch.Tensor = None,
    ):
        X, X_mask = split_padding(x, ptr[1:] - ptr[:-1])
        Xr, Xr_mask = split_padding(xr, mol_rings_nums)
        seq, seq_padding_mask = self._assemble_sequence(
            X, Xr, X_mask, Xr_mask, sol_vec, med_vec, env_vec)
        seq = self.mol_encoder(seq, src_key_padding_mask=seq_padding_mask).masked_fill_(seq_padding_mask.unsqueeze(-1), 0.)
        return seq, torch.logical_not(X_mask), torch.logical_not(Xr_mask)

    def env_encoder(
            self,
            net_name: str,
            batch_size: int,
            graph_inputs: Optional[Union[dict[str, torch.Tensor], Iterable[dict[str, torch.Tensor]]]] = None,
            prop_inputs: Optional[Union[torch.Tensor, Iterable[torch.Tensor]]] = None,
            sol_ratios: torch.Tensor = None
    ):
        """ Encoder for general environmental information, such as solvents, media, and T, P, pH and so on. """
        if graph_inputs is None and prop_inputs is None:
            return None

        net = getattr(self, net_name)
        assert isinstance(net, nn.Module), f"The {net_name} expects a nn.Module, got {type(net)}"

        if sol_ratios is not None:  # Expected shape of sol_ratio is (batch_size[B], sol_num[N])
            assert sol_ratios.ndim == 2
            cat_batch_size = batch_size * sol_ratios.size(1)

            if graph_inputs is not None:
                assert isinstance(graph_inputs, (tuple, list))
                assert len(graph_inputs) == sol_ratios.size(1)
                assert all(isinstance(g, dict) for g in graph_inputs)

                graph_repr = {'batch_size': cat_batch_size}
                for arg in graph_inputs[0].keys():
                    if 'batch' in arg:
                        graph_repr[arg] = torch.cat(
                            [graph_inputs[i][arg] + i * batch_size for i in range(len(graph_inputs))],
                            dim=-1
                        )
                    elif "index" in arg:
                        graph_repr[arg] = torch.cat(
                            [graph_inputs[i][arg] + (i and len(graph_inputs[i - 1]['x'])) for i in range(len(graph_inputs))],
                            dim=-1
                        )
                    else:
                        graph_repr[arg] = torch.cat([graph_inputs[i][arg] for i in range(len(graph_inputs))], dim=0)

            else:
                graph_repr = None

            if prop_inputs is not None:
                assert isinstance(prop_inputs, (tuple, list))
                assert len(prop_inputs) == sol_ratios.size(1)
                assert all(p.size(0) == batch_size for p in prop_inputs)

                prop_repr = torch.cat(prop_inputs, dim=0)

            else:
                prop_repr = None

            sol_vec = net(graph_repr, prop_repr)
            sol_vec = sol_vec.view((sol_ratios.size(1), batch_size, -1))

            return torch.sum(sol_ratios.permute(1, 0).unsqueeze(-1) * sol_vec, dim=0)

        else:
            assert isinstance(graph_inputs, dict)
            assert isinstance(prop_inputs, torch.Tensor)
            assert prop_inputs.size(0) == batch_size

            graph_repr = graph_inputs.copy()
            graph_repr['batch_size'] = batch_size

            return net(graph_repr, prop_inputs)

    def mol_level_encoder(self, mol_level_info: Optional[torch.Tensor] = None):
        if mol_level_info is None:
            return None
        return self.mol_info_net(mol_level_info)

    @overload
    def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_attr: torch.Tensor,
            rings_node_index: torch.Tensor,
            rings_node_nums: torch.Tensor,
            mol_rings_nums: torch.Tensor,
            batch: torch.Tensor,
            ptr: torch.Tensor,
            *,
            xyz: Optional[torch.Tensor] = None,
            sol_graph: Optional[Iterable[torch.Tensor]] = None,
            sol_props: Optional[torch.Tensor] = None,
            med_graph: Optional[Iterable[torch.Tensor]] = None,
            med_props: Optional[torch.Tensor] = None,
            mol_level_info: Optional[torch.Tensor] = None,
    ):
        """ Single solvent mode """

    @overload
    def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_attr: torch.Tensor,
            rings_node_index: torch.Tensor,
            rings_node_nums: torch.Tensor,
            mol_rings_nums: torch.Tensor,
            batch: torch.Tensor,
            ptr: torch.Tensor,
            *,
            xyz: Optional[torch.Tensor] = None,
            sol_ratios: torch.Tensor = None,
            sol_graph: Optional[Iterable[Iterable[torch.Tensor]]] = None,
            sol_props: Optional[Iterable[torch.Tensor]] = None,
            med_graph: Optional[Iterable[torch.Tensor]] = None,
            med_props: Optional[torch.Tensor] = None,
            mol_level_info: Optional[torch.Tensor] = None,
    ):
        """ Multiply solvent mode """

    def forward(
            self,
            x, edge_index, edge_attr, rings_node_index, rings_node_nums, mol_rings_nums, batch, ptr,
            *,
            xyz: Optional[Union[torch.Tensor, torch.nested.nested_tensor]] = None,
            sol_graph: Optional[Union[dict[str, torch.Tensor], Iterable[dict[str, torch.Tensor]]]] = None,
            sol_props: Optional[Union[torch.Tensor, Iterable[torch.Tensor]]] = None,
            sol_ratios: Optional[torch.Tensor] = None,
            med_graph: Optional[Iterable[torch.Tensor]] = None,
            med_props: Optional[torch.Tensor] = None,
            med_ratios: Optional[torch.Tensor] = None,
            mol_level_info: Optional[Union[torch.Tensor, torch.nested.nested_tensor]] = None,
            **kwargs
    ):
        batch_size = batch.max().to(torch.int) + 1

        x = self.node_processor(x, edge_index, batch, xyz=xyz)
        xr = self._rings_attention(x, rings_node_index, rings_node_nums)

        x_sol = self.env_encoder('sol_encoder', batch_size, sol_graph, sol_props, sol_ratios)
        x_med = self.env_encoder('med_encoder', batch_size, med_graph, med_props, med_ratios)
        x_mol = self.mol_level_encoder(mol_level_info)

        seq, X_not_mask, Xr_not_mask = self._mol_attention(x, xr, mol_rings_nums, ptr, x_sol, x_med, x_mol)

        return seq, X_not_mask, Xr_not_mask
