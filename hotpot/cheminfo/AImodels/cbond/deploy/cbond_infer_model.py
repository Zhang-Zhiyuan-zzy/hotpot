# -*- coding: utf-8 -*-
"""
===========================================================
 Python    : v3.9.0
 Project   : hotpot
 File      : cbond_infer_model
 Created   : 2025/10/7 20:52
 Author    : Zhiyuan Zhang
 Python    : 
-----------------------------------------------------------
 Description
 ----------------------------------------------------------
 
===========================================================
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from hotpot.plugins.ComplexFormer import models as M
import torch_geometric.nn as pygnn


# New
class CBondInfer(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = InferCore_()
        self.predictors = M.Predictor(128, 'binary')

    def forward(self, x, padded_Xr, rings_mask, cbond_index) -> torch.Tensor:
        xr = self.core.f_rings(padded_Xr, rings_mask)
        xr_mask = torch.all(rings_mask, dim=1)
        seq = self.core._mol_attention(x, xr, xr_mask)

        zx = seq[0, 1:x.shape[0]+1]

        upper_idx, lower_idx = cbond_index

        uzx = zx[upper_idx]
        lzx = zx[lower_idx]

        vec_cb = (uzx + lzx) / 2  # cbond vector

        return self.predictors(vec_cb)

    @staticmethod
    def extract_X_rings(xg, rings_node_index, rings_node_nums, max_rings_nums, max_rings_size):
        xg = xg[rings_node_index]
        indices = torch.arange(max_rings_size)
        padded_rings_num = F.pad(rings_node_nums, (0, max_rings_nums - len(rings_node_nums))).unsqueeze(-1)
        rings_mask = indices >= padded_rings_num
        return CBondInfer.split_padding_deploy(xg, rings_mask)

    @staticmethod
    def split_padding_deploy(x: torch.Tensor, padding_mask: torch.Tensor):
        batch_size, length = padding_mask.shape

        padded_Xr = torch.zeros(
            (batch_size, length, x.shape[-1]),
            device=x.device,
            dtype=x.dtype
        )

        padded_Xr[~padding_mask] = x
        return padded_Xr, padding_mask


class InferGraph(nn.Module):
    def __init__(self):
        super().__init__()
        self.node_processor = NodeProcessor()

    def forward(self, x, edge_index):
        x = self.node_processor(x, edge_index)
        return x

# New 2025.10.08
class Infer_(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = InferCore_()
        self.predictors = M.Predictor(128, 'binary')

    def forward(self, x, edge_index, rings_node_index, cbond_index, rings_mask) -> torch.Tensor:
        seq = self.core(x, edge_index, rings_node_index, rings_mask)
        zx = seq[0, 1:x.shape[0]+1]

        upper_idx, lower_idx = cbond_index

        uzx = zx[upper_idx]
        lzx = zx[lower_idx]

        vec_cb = (uzx + lzx) / 2  # cbond vector

        return self.predictors(vec_cb)

class InferCore_(nn.Module):
    def __init__(self, vec_dim=128, ring_layers=1, ring_nheads=2, mol_layers=4, mol_nheads=4):
        super(InferCore_, self).__init__()
        self.node_processor = NodeProcessor(vec_dim)

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

    def f_rings(self, X, padding_mask):
        X = self.ring_encoder(X, src_key_padding_mask=padding_mask)
        X = X.masked_fill_(padding_mask.unsqueeze(-1), 0.)
        return M.seq_absmax_pooling(X)

    def forward(self, x, edge_index, rings_node_index, rings_mask):
        x = self.node_processor(x, edge_index)
        xr = self._rings_attention(x, rings_node_index, rings_mask)
        xr_mask = torch.all(rings_mask, dim=1)
        assert xr.shape[:-1] == xr_mask.shape
        return self._mol_attention(x, xr, xr_mask)

    def _rings_attention(self, x, rings_node_index, rings_mask):
        x = x[rings_node_index]
        X, padding_mask = self.split_padding_deploy(x, rings_mask)
        logging.info(f"Xr={X.shape}, msk={padding_mask.shape}")
        # torch._check(X.shape[1] != 1)
        # torch._check(X.shape[1] != 0)
        print(X.shape, padding_mask.shape)
        X = self.ring_encoder(X, src_key_padding_mask=padding_mask)
        X = X.masked_fill_(padding_mask.unsqueeze(-1), 0.)
        return M.seq_absmax_pooling(X)

    def _mol_attention(self, x, xr, xr_mask):
        seq, seq_padding_mask = self._assemble_sequence(x, xr, xr_mask)
        assert seq.shape[:-1] == seq_padding_mask.shape
        print(seq.shape, seq_padding_mask.shape)
        seq = self.mol_encoder(seq, src_key_padding_mask=seq_padding_mask)
        seq = seq.masked_fill_(seq_padding_mask.unsqueeze(-1), 0.)
        return seq

    def _assemble_sequence(self, x, xr, xr_mask):
        seq = torch.cat((self.CLS, x, self.RING, xr, self.END), dim=0).unsqueeze(0)

        seq_padding_mask = torch.cat(
            (
                torch.zeros(2 + x.shape[0], dtype=torch.bool, device=x.device),
                xr_mask,
                torch.zeros(1, dtype=torch.bool, device=x.device)
            ), dim=0
        ).unsqueeze(0)
        return seq, seq_padding_mask

    @staticmethod
    def extract_X_rings(x, rings_node_index, rings_node_nums):
        x = x[rings_node_index]
        indices = torch.arange(64)
        padded_rings_num = F.pad(rings_node_nums, (0, 128 - len(rings_node_nums))).unsqueeze(-1)
        rings_mask = indices >= padded_rings_num
        return InferCore_.split_padding_deploy(x, rings_mask)

    @staticmethod
    def split_padding_deploy(x: torch.Tensor, padding_mask: torch.Tensor):
        batch_size, length = padding_mask.shape

        padded_X = torch.zeros(
            (batch_size, length, x.shape[-1]),
            device=x.device,
            dtype=x.dtype
        )

        padded_X[~padding_mask] = x
        return padded_X, padding_mask


# New independent module
class Infer(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = InferCore()
        self.predictors = M.Predictor(128, 'binary')

    def forward(self, x, edge_index, rings_node_index, rings_node_nums, cbond_index) -> torch.Tensor:
        assert rings_node_nums.shape[0] >= 0
        seq = self.core(x, edge_index, rings_node_index, rings_node_nums)
        zx = seq[0, 1:x.shape[0]+1]

        upper_idx, lower_idx = cbond_index

        uzx = zx[upper_idx]
        lzx = zx[lower_idx]

        vec_cb = (uzx + lzx) / 2  # cbond vector

        return self.predictors(vec_cb)


class InferCore(nn.Module):
    def __init__(self, vec_dim=128, ring_layers=1, ring_nheads=2, mol_layers=4, mol_nheads=4):
        super(InferCore, self).__init__()
        self.node_processor = NodeProcessor(vec_dim)

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

    def forward(self, x, edge_index, rings_node_index, rings_node_nums):
        x = self.node_processor(x, edge_index)
        xr = self._rings_attention(x, rings_node_index, rings_node_nums)
        return self._mol_attention(x, xr)

    def _rings_attention(self, x, rings_node_index, rings_node_nums):
        x = x[rings_node_index]
        X, padding_mask = M.split_padding_deploy(x, rings_node_nums)
        logging.info(f"Xr={X.shape}, msk={padding_mask.shape}")
        # torch._check(X.shape[1] != 1)
        # torch._check(X.shape[1] != 0)
        X = self.ring_encoder(X, src_key_padding_mask=padding_mask)
        return M.seq_absmax_pooling(X)

    def _mol_attention(self, x, xr):
        seq, seq_padding_mask = self._assemble_sequence(x, xr)
        seq = self.mol_encoder(seq, src_key_padding_mask=seq_padding_mask)
        return seq

    def _assemble_sequence(self, X, Xr):
        seq = torch.cat((self.CLS, X, self.RING, Xr, self.END), dim=0).unsqueeze(0)
        seq_padding_mask = torch.zeros((1, 3+X.shape[0]+Xr.shape[0]), dtype=torch.bool, device=X.device)

        return seq, seq_padding_mask


class NodeProcessor(nn.Module):
    def __init__(self,vec_dim: int = 128):
        super(NodeProcessor, self).__init__()
        self.vec_size = vec_dim

        self.x_emb = nn.Embedding(120, vec_dim)
        self.x_mask_vec = self.x_emb.weight[0]

        # self.cloud_net = CloudGraph(vec_dim)
        self.lin = nn.Linear(vec_dim, vec_dim)
        self.norm = nn.BatchNorm1d(vec_dim)

        self.graph = pygnn.GAT(
            vec_dim, vec_dim, 6,
            vec_dim, 0.1, norm=pygnn.LayerNorm(vec_dim),
            edge_dim=vec_dim, v2=True
        )

    def forward(self, x, edge_index):
        # Node Embedding
        x = self.x_emb(x.long())
        xg = self.graph(x, edge_index)  # x from GNN
        x = self.norm(self.lin(x + xg))
        return x
