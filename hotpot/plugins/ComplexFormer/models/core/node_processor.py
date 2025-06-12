# -*- coding: utf-8 -*-
"""
===========================================================
 Project   : hotpot
 File      : node_processor
 Created   : 2025/6/10 15:50
 Author    : zhang
 Python    : 
-----------------------------------------------------------
 Description
 ----------------------------------------------------------
 
===========================================================
"""
from typing import Optional

import torch
import torch.nn as nn
import torch_geometric.nn as pygnn

from .cloud import CloudGraph
from .se3net import SE3Net


__all__ = ['NodeProcessor']

class NodeProcessor(nn.Module):
    def __init__(
            self,
            x_dim: int,
            vec_dim: int = 512,
            x_label_nums: Optional[int] = None,
            graph_model: nn.Module = None,
            cloud_model: nn.Module = None,
    ):
        super(NodeProcessor, self).__init__()
        self.vec_size = vec_dim
        self.x_label_nums = x_label_nums

        if isinstance(x_label_nums, int):
            self.x_emb = nn.Embedding(x_label_nums+1, vec_dim)
            self.x_mask_vec = self.x_emb.weight[0]
        else:
            self.x_proj = nn.Linear(x_dim, vec_dim)
            self.x_mask_vec = nn.Parameter(torch.randn(x_dim))

        # self.cloud_net = CloudGraph(vec_dim)
        self.lin = nn.Linear(vec_dim, vec_dim)
        self.norm = nn.BatchNorm1d(vec_dim)

        if graph_model:
            self.graph = graph_model
        else:
            self.graph = pygnn.GAT(
                vec_dim, vec_dim, 6,
                vec_dim, 0.1, norm=pygnn.LayerNorm(vec_dim),
                edge_dim=vec_dim, v2=True
            )

        if cloud_model:
            self.cloud = cloud_model
        else:
            self.cloud = SE3Net(
                vec_dim, max_l=1, parity=-1,
                num_layers=4, max_radius=4.5, num_radius_basis=10
            )

    def forward(self, x, edge_index=None, batch=None, xyz=None):

        # Node Embedding
        if isinstance(self.x_label_nums, int):
            x = self.x_emb(x.long())
        else:
            x = self.x_proj(x)

        # Graph side
        if edge_index is not None:
            xg = self.graph(x, edge_index)  # x from GNN
        else:
            xg = None

        # Coordinates side
        if xyz is not None:
            xc = self.cloud(x, xyz, batch)  # x from cloud net

            if xg is not None:
                x = self.norm(self.lin(x + xg + xc))
            else:
                x = self.norm(self.lin(x + xc))
        else:
            if xg is None:
                raise ValueError('the graph `edge_index` and the `xyz` should be given at least one')

            x = self.norm(self.lin(x + xg))

        return x
