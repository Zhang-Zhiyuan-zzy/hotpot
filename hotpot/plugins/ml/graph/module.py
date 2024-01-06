"""
python v3.9.0
@Project: hotpot
@File   : graph
@Auther : Zhiyuan Zhang
@Data   : 2023/8/8
@Time   : 13:28
"""
from pathlib import Path
import json

import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric as pg
from openbabel import openbabel as ob

import hotpot as hp
from hotpot import data_root
from hotpot.cheminfo import Atom


def get_atom_energy_tensor(
        method: str, basis: str,
        solvent: str = None,
        charges: list[int] = None,
        end_element: int = 58,
        padding_miss: bool = False
) -> torch.Tensor:
    """"""
    path_atom_single_point = Path(data_root).joinpath('atom_single_point.json')
    _atom_single_point = json.load(open(path_atom_single_point))

    if solvent is None:
        solvent = "null"

    if isinstance(charges, list):
        charges = [0] + charges

    energies = [0.0]
    for i in range(1, end_element):
        atom = Atom(atomic_number=i)

        if charges:
            c = charges[i]
        elif atom.is_metal:
            c = atom.stable_charge
        else:
            c = 0

        try:
            energy = _atom_single_point[atom.symbol][method][basis][solvent][str(c)]
        except KeyError as e:
            if padding_miss:
                energy = 0.0
            else:
                print(KeyError(atom.symbol))
                raise e

        energies.append(energy)

    return torch.tensor(energies)


class MolNet(nn.Module):
    """"""
    def __init__(
            self,
            element_energy: torch.Tensor,
            emb_size: int = 6,
            gcn_layers: int = 6,
    ):
        """

        Args:
            element_energy: elemental energies
        """
        super().__init__()
        self.element_energy = nn.Parameter(element_energy)

        self.node_mlp = pg.nn.MLP([6, emb_size])
        self.node_emb = nn.Embedding(59, emb_size)

        self.gcns = pg.nn.GAT(-1, emb_size, gcn_layers, dropout=0.1, norm="layer_norm", heads=2, edge_dim=emb_size)
        self.atts = NodeAttention(emb_size)
        self.coord_net = CoordNet(emb_size)

        # Encode the edge info
        self.edge_emb = torch.nn.Embedding(4, emb_size)
        self.edge_mlp = pg.nn.MLP([emb_size, emb_size, emb_size])

        self.lin = nn.Linear(2*emb_size, 2*emb_size)

        self.edge2feature = pg.nn.MLP([2*emb_size, 2*emb_size, 2*emb_size], dropout=0.1)

        self.nodef2E = pg.nn.MLP([emb_size, emb_size, 1])
        self.edgef2E = pg.nn.MLP([2*emb_size, 2*emb_size, 1])

    def forward_(self, x, edges, edge_attr, batch_idx):
        """"""
        x = x.float()
        row, col = edges
        edges_batch_idx = batch_idx[row]

        Ea = self.element_energy[x[:, 0].long()]  # atom-wise Energies.
        Ea_graph = pg.nn.global_add_pool(Ea, batch_idx)  # sum of atom-wise Energies in whole graph.

        # x = self.node_mlp(x.float())
        x = self.node_emb(x[:, 0].long())
        edge_attr = self.edge_mlp(self.edge_emb(edge_attr))

        x = self.gcns(x, edges, edge_attr=edge_attr)
        x = self.atts(x, batch_idx)

        x_edge_max, max_idx = x[edges].permute([1, 0, 2]).max(-2)

        xv = torch.concat([x_edge_max, edge_attr], dim=-1)

        xv = self.lin(xv) + self.edge2feature(xv)

        Eb = F.leaky_relu(self.edgef2E(xv))  # bond-wise Energies
        Eb_graph = pg.nn.global_add_pool(Eb, edges_batch_idx)  # sum of bond-wise Energies in whole graph.

        E = Ea_graph + Eb_graph.flatten()

        return E, Ea_graph, Eb_graph

    def forward(self, x, edges, edge_attr, batch_idx, c):
        Ec, _, _ = self.coord_net(x[:, 0], c, edges, batch_idx)

        x = x.float()
        row, col = edges
        edges_batch_idx = batch_idx[row]

        Ea = self.element_energy[x[:, 0].long()]  # atom-wise Energies.
        Ea_graph = pg.nn.global_add_pool(Ea, batch_idx)  # sum of atom-wise Energies in whole graph.

        x = self.node_mlp(x.float())
        edge_attr = self.edge_mlp(self.edge_emb(edge_attr))

        x = self.gcns(x, edges, edge_attr=edge_attr)
        x = self.atts(x, batch_idx)

        x_edge_max, max_idx = x[edges].permute([1, 0, 2]).max(-2)

        xv = torch.concat([x_edge_max, edge_attr], dim=-1)

        xv = self.lin(xv) + self.edge2feature(xv)

        Eb = F.leaky_relu(self.edgef2E(xv))  # bond-wise Energies
        Eb_graph = pg.nn.global_add_pool(Eb, edges_batch_idx)  # sum of bond-wise Energies in whole graph.

        E = Ea_graph + Eb_graph.flatten() + Ec.flatten()
        # E = Ea_graph + Ec.flatten()

        return E, None, Ec.flatten()


class CoordNet(nn.Module):
    def __init__(
            self,
            emb_size,
    ):
        super().__init__()
        self.type_emb = nn.Embedding(59, emb_size)
        self.posi_mlp = pg.nn.MLP([3, emb_size])
        self.npn_mlp = pg.nn.MLP([3*emb_size, emb_size])

        self.predictor = pg.nn.MLP([emb_size * emb_size, emb_size, emb_size, 1], act="leaky_relu")

    def forward(self, t, c, edges, batch_idx):
        c = c.float()
        v_t = self.type_emb(t)

        src_idx, tgt_idx = edges
        edge_index = batch_idx[src_idx]

        split_edge_idx = edge_index.bincount().tolist()

        p_s2t = c[tgt_idx] - c[src_idx]  # the relative position from source node to target node

        v_p = self.posi_mlp(p_s2t)

        v_npn = torch.concat([v_t[src_idx], v_p, v_t[tgt_idx]], dim=-1)
        v_npn = self.npn_mlp(v_npn)

        features = []
        for i, (pos_i, v_npn_i) in enumerate(zip(p_s2t.split(split_edge_idx), v_npn.split(split_edge_idx))):

            p_norm2 = torch.norm(pos_i, dim=-1)
            cos = torch.matmul(pos_i, pos_i.T) / torch.matmul(p_norm2, p_norm2.T)

            vR_npn_i = torch.matmul(torch.matmul(v_npn_i.T, cos), v_npn_i)

            features.append(vR_npn_i.flatten())

        features = torch.stack(features)

        return self.predictor(features), None, None


class BDENet(nn.Module):
    """"""
    def __init__(self):
        """"""

    def forward(self, x, e):
        """"""


class NodeAttention(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.key_encoder = nn.Linear(input_size, input_size)
        self.query_encoder = nn.Linear(input_size, input_size)
        self.value_encoder = nn.Linear(input_size, input_size)
        self.layer_norm = nn.LayerNorm(input_size)

    def forward(self, x, batch):
        """"""
        keys, query, values = self.key_encoder(x), self.query_encoder(x), self.value_encoder(x)
        bins = batch.bincount().tolist()
        sqrt_dk = pow(keys.shape[-1], 0.5)

        zs = []
        for k, q, v in zip(keys.split(bins), query.split(bins), values.split(bins)):
            kq = F.softmax(torch.matmul(k, q.T) / sqrt_dk, dim=-1)
            zs.append(torch.matmul(kq, v))

        return self.layer_norm(x + torch.concat(zs, -2))


class SampleGAT(nn.Module):
    """"""
    def __init__(
            self,
            emb_size: int = 6,
            gcn_layers: int = 6,
            integral_coords: bool = False
    ):
        super().__init__()
        if integral_coords:
            self.node_mlp = pg.nn.MLP([9, emb_size])
        else:
            self.node_mlp = pg.nn.MLP([6, emb_size])

        self.gcns = pg.nn.GAT(-1, emb_size, gcn_layers, dropout=0.1, norm="layer_norm", heads=2, edge_dim=emb_size)
        self.atts = NodeAttention(emb_size)

        # Encode the edge info
        self.edge_emb = torch.nn.Embedding(4, emb_size)
        self.edge_mlp = pg.nn.MLP([emb_size, emb_size, emb_size])

        self.lin = nn.Linear(2*emb_size, 2*emb_size)

        self.node2Ef = pg.nn.MLP([emb_size, 2*emb_size, 4*emb_size])

        self.predictor = pg.nn.MLP([4*emb_size, 2*emb_size, emb_size, 1], act="leaky_relu")

    def forward(self, x, edges, edge_attr, batch_idx, coords=None):
        """"""
        x = x.float()

        if coords is not None:
            x = torch.concat((x, coords), -1)

        x = self.node_mlp(x.float())
        edge_attr = self.edge_mlp(self.edge_emb(edge_attr))

        x = self.gcns(x, edges, edge_attr=edge_attr)
        x = self.atts(x, batch_idx)
        x = self.node2Ef(x)

        graph_feature = pg.nn.global_max_pool(x, batch_idx)
        return self.predictor(graph_feature), None, None


