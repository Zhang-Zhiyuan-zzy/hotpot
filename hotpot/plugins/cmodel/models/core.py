from abc import abstractmethod
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import torch_geometric.nn as pygnn

from .graph import CompleteGraph
from . import utils

################################ Utils functions ########################################
def complete_graph_generator(ptr):
    return torch.cat(
        [torch.combinations(torch.arange(ptr[i], ptr[i+1]), with_replacement=True) for i in range(len(ptr)-1)]
    ).T.to(ptr.device)

def _split_padding(x: torch.Tensor, nums: torch.Tensor):
    """
    Split X in PyG-style batch and padding.
    :param x: PyG-style batch node vectors
    :param nums: node numbers in each sample
    :return:
    """
    B = nums.shape[0]
    L = max(nums).item()
    D = x.shape[-1]

    padded_X = torch.zeros((B, L, D)).to(x.device)
    padding_mask = torch.ones((B, L), dtype=torch.bool, device=x.device)

    start = 0
    for i, size in enumerate(nums.long()):
        size: int
        padded_X[i, :size] = x[start:start + size]
        padding_mask[i, :size] = 0
        start += size

    return padded_X, padding_mask

def _seq_absmax_pooling(seq):
    """ Pooling each seq[L, E] to a vec[1, E] """
    pooled_vec = torch.zeros((len(seq), seq.shape[-1])).to(seq.device)
    for i, t in enumerate(seq):
        pooled_vec[i, :] = t.gather(-2, torch.argmax(torch.abs(t), dim=-2).unsqueeze(-2))
    return pooled_vec

################################################################################################


class CloudGraph(nn.Module):
    """ Graph network to perform complete graph operation """
    def __init__(self, vec_dim: int):
        super(CloudGraph, self).__init__()
        self.xyz_proj = nn.Linear(3, vec_dim, bias=False)
        self.xyz_proj_norm = nn.BatchNorm1d(vec_dim)

        self.lin1 = nn.Linear(vec_dim, vec_dim)
        self.norm1 = nn.LayerNorm(vec_dim)

        self.lin2 = nn.Linear(vec_dim, vec_dim)
        self.norm2 = nn.LayerNorm(vec_dim)

    def forward(self, x, xyz, ptr):
        # TODO: add vector cross-multiply module
        # TODO: Add CNN module

        cloud_edge_index = complete_graph_generator(ptr)
        rela_xyz = xyz[cloud_edge_index[0]] - xyz[cloud_edge_index[1]]
        rela_x = x[cloud_edge_index[0]] - x[cloud_edge_index[1]]

        dist_xyz = torch.norm(rela_xyz, p=2, dim=-1)
        weight = torch.exp(-dist_xyz).unsqueeze(-1)

        x = x + self.norm1(pygnn.global_add_pool(
            F.relu(self.lin1(weight * rela_x)),
            cloud_edge_index[0]
        ))

        x = x + self.xyz_proj_norm(
            pygnn.global_add_pool(
                F.relu(self.xyz_proj(rela_xyz)),
                cloud_edge_index[0]
            )
        )

        return x


class NodeProcessor(nn.Module):
    def __init__(
            self,
            x_dim: int,
            vec_dim: int = 512,
            x_label_nums: Optional[int] = None,
            graph_model: nn.Module = None,
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

        self.cloud_graph = CloudGraph(vec_dim)
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

    def forward(self, x, edge_index, ptr, xyz=None):
        if isinstance(self.x_label_nums, int):
            x = self.x_emb(x.long())
        else:
            x = self.x_proj(x)

        xg = self.graph(x, edge_index)

        # coordinates side
        if xyz is not None:
            x_cloud = self.cloud_graph(x, xyz, ptr)
            x = self.norm(self.lin(x + xg + x_cloud))
        else:
            x = self.norm(self.lin(x + xg))

        return x


class CoreBase(nn.Module):
    """"""
    extractor_class = None
    extractor_keys = {
        'atom': 'extract_atom_vec',
        'pair': 'extract_atom_vec',
        'ring': 'extract_ring_vec',
        'mol': 'extract_mol_vec',
        'cbond': 'extract_cbond_pair',
        'metal': 'extract_metal_vec'
    }
    def __init__(self, vec_dim: int, x_label_nums: Optional[int] = None):
        super(CoreBase, self).__init__()
        self.vec_size = vec_dim
        self.x_label_nums = x_label_nums

        self.feature_extractor = {
            key:getattr(self.extractor_class, method_name)
            for key, method_name in self.extractor_keys.items()
            if hasattr(self.extractor_class, method_name)
        }

    @property
    @abstractmethod
    def x_mask_vec(self):
        raise NotImplementedError('the property `x_mask_vec` is not implemented.')

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

        cbond_feature = torch.cat([upper_Znode, lower_Znode], dim=1)

        assert cbond_feature.shape == (upper_Znode.shape[0], upper_Znode.shape[1] * 2)

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


class AttnExtractor:
    @staticmethod
    def extract_atom_vec(seq, X_mask, R_mask, batch, batch_getter=None):
        Znode = []
        node_seq = seq[:, 1:X_mask.shape[-1]+1]
        for s, m in zip(node_seq, X_mask.sum(dim=-1)):
            Znode.append(s[:m])

        return torch.cat(Znode, dim=0)

    @staticmethod
    def extract_metal_vec(seq, X_mask, R_mask, batch, batch_getter=None):
        metal_idx = utils.where_metal(batch.x[:, 0])
        Znode = AttnExtractor.extract_atom_vec(seq, X_mask, R_mask, batch, batch_getter)
        return Znode[metal_idx]

    @staticmethod
    def extract_cbond_pair(seq, X_mask, R_mask, batch, batch_getter=None):
        Znode = AttnCore.extract_atom_vec(seq, X_mask, R_mask, batch, batch_getter)
        cbond_index = batch.cbond_index

        upper_Znode = Znode[cbond_index[0]]
        lower_Znode = Znode[cbond_index[1]]

        cbond_feature = torch.cat([upper_Znode, lower_Znode], dim=1)

        assert cbond_feature.shape == (upper_Znode.shape[0], upper_Znode.shape[1] * 2)

        return cbond_feature

    @staticmethod
    def extract_pair_vec(seq, X_mask, R_mask, batch, batch_getter=None):
        Znode = AttnCore.extract_atom_vec(seq, X_mask, R_mask, batch, batch_getter)

        pair_index = batch_getter(batch)

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


class AttnCore(CoreBase):
    # Feature Extractors
    extractor_class = AttnExtractor
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
        super(AttnCore, self).__init__(vec_dim, x_label_nums)
        self.node_processor = NodeProcessor(x_dim, vec_dim, x_label_nums=x_label_nums, graph_model=graph_model)
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
        X, padding_mask = _split_padding(x, rings_node_nums)
        X = self.ring_encoder(X, src_key_padding_mask=padding_mask)
        return _seq_absmax_pooling(X)

    def _mol_attention(self, x, xr, mol_rings_nums, ptr, batch):
        X, X_mask = _split_padding(x, ptr[1:] - ptr[:-1])
        Xr, Xr_mask = _split_padding(xr, mol_rings_nums)
        seq, seq_padding_mask = self._assemble_sequence(X, Xr, X_mask, Xr_mask)
        seq = self.mol_encoder(seq, src_key_padding_mask=seq_padding_mask)
        return seq, torch.logical_not(X_mask), torch.logical_not(Xr_mask)

    def forward(
            self,
            x, edge_index, edge_attr, rings_node_index, rings_node_nums, mol_rings_nums, batch, ptr,
            *,
            xyz: Optional[Union[torch.Tensor, torch.nested.nested_tensor]] = None,
    ):
        x = self.node_processor(x, edge_index, ptr, xyz=xyz)
        xr = self._rings_attention(x, rings_node_index, rings_node_nums)
        return self._mol_attention(x, xr, mol_rings_nums, ptr, batch)


Core = AttnCore
