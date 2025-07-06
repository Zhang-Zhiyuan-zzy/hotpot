# -*- coding: utf-8 -*-
"""
===========================================================
 Project   : hotpot
 File      : envs_encoder
 Created   : 2025/6/19 9:12
 Author    : zhang
 Python    : 
-----------------------------------------------------------
 Description
 ----------------------------------------------------------
 Definition of Network to compile environments information, such as solvents, media, Temp., Pressure, and so on.
===========================================================
"""
from typing import Optional, Union, Type

import torch
import torch.nn as nn

import torch_geometric.nn as pyg_nn


class SolventNet(nn.Module):
    def __init__(
            self,
            vec_dim: int,
            props_nums: Optional[int] = None,
            labeled_node: bool = False,
            node_dim: Optional[int] = None,
            embedding_weights: Optional[Union[int, tuple[int, int], torch.Tensor]] = None,
            props_net_layers: int = 2,
            gnn_layers: int = 3,
            gnn: Optional[Union[nn.Module, str]] = None,
            gnn_kw: Optional[dict] = None,
    ):
        if not isinstance(props_nums, int) and not isinstance(node_dim, int):
            raise ValueError("'props_nums' and 'node_dim' must be given as int at least one'")

        self.vec_dim = vec_dim

        # Configure properties compiling Module
        super(SolventNet, self).__init__()
        if isinstance(props_nums, int) and isinstance(props_net_layers, int):
            mlp_nums = [props_nums] + props_net_layers * [vec_dim]
            self.props_net = pyg_nn.MLP(mlp_nums)
        else:
            self.props_net = None

        # Add node_embedding layers
        if labeled_node:
            if node_dim is None:
                node_dim = vec_dim

            if embedding_weights is None:
                self.emb = nn.Embedding(119, node_dim)
            elif isinstance(embedding_weights, int):
                self.emb = nn.Embedding(embedding_weights, node_dim)
            elif isinstance(embedding_weights, tuple):
                num_emb, node_dim = embedding_weights
                self.emb = nn.Embedding(num_emb, node_dim)
            elif isinstance(embedding_weights, torch.Tensor):
                # This configuration can make sure all elements with identical embedding vector,
                # on matter in ligands, solvents or media.
                assert embedding_weights.dim() == 2
                num_emb, node_dim = embedding_weights.shape
                self.emb = nn.Embedding(num_emb, node_dim)
                self.emb.weight = embedding_weights
            else:
                raise TypeError(f'embedding_weights must be None, int, tuple of int, or a Tensor with dim=2')
        else:
            self.emb = None

        # Configure the GNN modules
        if isinstance(gnn, nn.Module):
            self.gnn = gnn
        elif isinstance(gnn, str):
            gnn_type: Type[nn.Module] = getattr(pyg_nn, gnn)
            gnn_kw = gnn_kw or {}
            self.gnn = gnn_type(node_dim, vec_dim, gnn_layers, **gnn_kw)
        elif isinstance(node_dim, int) and isinstance(gnn_layers, int):
            gnn_kw = gnn_kw or {}
            self.gnn = pyg_nn.GIN(node_dim, vec_dim, gnn_layers, **gnn_kw)
        else:
            self.gnn = None

    def forward(self, graph_repr: Optional[dict[str, torch.Tensor]] = None, props_vec: torch.Tensor = None) -> torch.Tensor:
        """"""
        if not graph_repr and not props_vec:
            return torch.zeros(self.vec_dim)

        if isinstance(graph_repr, dict):
            if self.gnn is None:
                raise AttributeError('The graph encoder is not defined, cannot to compile sol_graph info')

            if isinstance(self.emb, nn.Module):
                graph_repr['x'] = self.emb(graph_repr['x'])
            xg = pyg_nn.global_max_pool(self.gnn(**graph_repr), batch=graph_repr['batch'], size=graph_repr['batch_size'])

        else:
            xg = 0

        if isinstance(props_vec, torch.Tensor):
            if self.props_net is None:
                raise AttributeError('The Properties encoder is not defined, cannot to compile props info')

            prop_mask = (props_vec.abs().max(dim=-1)[0] > 1e-8).unsqueeze(1)
            xp = self.props_net(props_vec) * prop_mask

        else:
            xp = 0

        return xg + xp
