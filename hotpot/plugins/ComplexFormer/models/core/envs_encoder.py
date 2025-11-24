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
            node_embedder: nn.Module,
            props_nums: Optional[int] = None,
            props_net_layers: int = 2,
            gnn_layers: int = 3,
            gnn_kw: Optional[dict] = None,
    ):
        self.vec_dim = vec_dim

        # Configure properties compiling Module
        super(SolventNet, self).__init__()
        if isinstance(props_nums, int) and isinstance(props_net_layers, int):
            mlp_nums = [props_nums] + props_net_layers * [vec_dim]
            self.props_net = pyg_nn.MLP(mlp_nums)
        else:
            self.props_net = None

        # Add node_embedding layers
        self.node_embedder = node_embedder

        # Configure the GNN modules
        gnn_kw = gnn_kw or {}
        self.gnn = pyg_nn.GIN(vec_dim, vec_dim, gnn_layers, **gnn_kw)

    def forward(self, graph_repr: Optional[dict[str, torch.Tensor]] = None, props_vec: torch.Tensor = None) -> torch.Tensor:
        """"""
        if not graph_repr and not props_vec:
            return torch.zeros(self.vec_dim)

        if isinstance(graph_repr, dict):
            if self.gnn is None:
                raise AttributeError('The graph encoder is not defined, cannot to compile sol_graph info')

            graph_repr['x'] = self.node_embedder(graph_repr['x'])
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
