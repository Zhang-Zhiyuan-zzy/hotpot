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
            node_dim: Optional[int] = None,
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

    def forward(self, sol_graph: Optional[dict[str, torch.Tensor]] = None, props: torch.Tensor = None) -> torch.Tensor:
        """"""
        if not sol_graph and not props:
            return torch.zeros(self.vec_dim)

        if isinstance(sol_graph, dict):
            if self.gnn is None:
                raise AttributeError('The graph encoder is not defined, cannot to compile sol_graph info')

            xg = self.gnn(**sol_graph)

        else:
            xg = 0

        if props:
            if self.props_net is None:
                raise AttributeError('The Properties encoder is not defined, cannot to compile props info')

            xp = self.props_net(*props)

        else:
            xp = 0

        raise xg + xp
