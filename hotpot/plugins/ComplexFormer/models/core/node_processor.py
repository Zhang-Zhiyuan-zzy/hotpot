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
from typing import Optional, Sequence, Literal

import torch
import torch.nn as nn
import torch_geometric.nn as pygnn

from .se3net import SE3Net, Parity


__all__ = ['NodeProcessor']



class GraphCloud(nn.Module):
    def __init__(
            self,
            vec_dim: int = 256,
            graph_layer: int = 6,
            cloud_layer: int = 4,
            max_l: int = 1,
            parity: Parity = -1,
            weights_mlp: Sequence[int] = (64,),
            max_radius: float = 4.5,
            num_radius_basis: int = 10,
    ):
        super(GraphCloud, self).__init__()
        self.graph = pygnn.GAT(
            vec_dim, vec_dim, graph_layer,
            vec_dim, 0.1, norm=pygnn.LayerNorm(vec_dim),
            edge_dim=vec_dim, v2=True
        )
        self.cloud = SE3Net(
            vec_dim, max_l=max_l, parity=parity,
            num_layers=cloud_layer, max_radius=max_radius,
            num_radius_basis=num_radius_basis
        )

    def forward(self, x, edge_index=None, batch=None, xyz=None):
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


class NodeEmb(nn.Module):
    def __init__(
            self,
            vec_dim: int = 256,
            emb_type: Literal['proj', 'atom', 'orbital'] = 'atom'
    ):
        super(NodeEmb, self).__init__()
        self.emb_type = emb_type
        if emb_type == 'proj':
            self.x_proj = pygnn.MLP([7, 1024, vec_dim])
            self.x_mask_vec = nn.Parameter(torch.randn(7))
        elif emb_type == 'atom':
            self.x_emb = nn.Embedding(120, vec_dim)
            self.x_mask_vec = self.x_emb.weight[0]
        elif emb_type == 'orbital':
            self.emb_dim = vec_dim // 4
            # The start positional of [atomic_number, n, s, p, d, f, g]
            self.emb_segment = [0, 120, 8, 3, 7, 11, 15, 19]
            self.x_emb = nn.Embedding(sum(self.emb_segment), self.emb_dim)

            self.register_buffer('offset', torch.cumsum(torch.tensor(self.emb_segment), dim=0)[:-1].int())
            self.x_mask_vec = self.x_emb(self.offset).flatten()

            self.emb_proj = pygnn.MLP([self.emb_dim * 7, vec_dim])

        else:
            raise ValueError('emb_type must be "atom", "proj", "orbital"')

    def forward(self, x):
        if self.emb_type == 'proj':
            return self.x_proj(x)
        elif self.emb_type == 'atom':
            return self.x_emb(x.int())
        elif self.emb_type == 'orbital':
            x = x + self.offset
            emb_x = self.x_emb(x.int()).view(x.size(0), -1)
            return self.emb_proj(emb_x)
        else:
            raise ValueError('emb_type must be "atom", "proj", "orbital"')

    @property
    def node_mask(self):
        if self.emb_type == 'proj':
            return self.x_mask_vec
        elif self.emb_type == 'atom':
            return 0
        elif self.emb_type == 'orbital':
            return torch.zeros(7, dtype=torch.int)
        else:
            raise NotImplementedError('Unknown emb_type `{}`'.format(self.emb_type))


class NodeProcessor(nn.Module):
    def __init__(
            self,
            vec_dim: int = 512,
            emb_type: Literal['proj', 'atom', 'orbital'] = 'atom',
            graph_layer: int = 6,
    ):
        super(NodeProcessor, self).__init__()
        self.vec_size = vec_dim
        self.node_embedder = NodeEmb(vec_dim, emb_type)

        self.graph = pygnn.GAT(
            vec_dim, vec_dim, graph_layer,
            vec_dim, 0.1, norm=pygnn.LayerNorm(vec_dim),
            edge_dim=vec_dim, v2=True
        )

        self.cloud = SE3Net(
            vec_dim, max_l=1, parity=-1,
            num_layers=4, max_radius=4.5, num_radius_basis=10
        )

        self.lin = nn.Linear(vec_dim, vec_dim)
        self.norm = nn.BatchNorm1d(vec_dim)

    def forward(self, x, edge_index=None, batch=None, xyz=None):

        # Node Embedding
        x = self.node_embedder(x)

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

    @property
    def node_mask(self):
        return self.node_embedder.node_mask
