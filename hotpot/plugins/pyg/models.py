import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
import torch.optim as optim
import torch_geometric.nn as pygnn
from sympy.physics.units import moles
from torch.xpu import device


def get_masked_input_and_labels(
        inp_vec: torch.Tensor,
        mask_vec: torch.Tensor,
        inp_atom_labels: torch.Tensor,
):
    # 15% BERT masking
    masked_idx = torch.from_numpy(np.random.uniform(inp_vec.shape[0]) < 0.15).to(inp_vec.device)
    # Set targets to -1 by default, it means ignore
    atom_labels = -1 * torch.ones(inp_vec.shape[0], dtype=torch.int).to(inp_vec.device)
    # Set labels for masked tokens
    atom_labels[masked_idx] = inp_atom_labels[masked_idx]

    # Prepare masked input
    masked_vec = torch.copy(inp_vec)
    # Set input to [MASK] which is the last token for the 90% of tokens
    # This means leaving 10% unchanged
    mask2mask_idx = masked_idx & (torch.rand(inp_vec.shape[0]) < 0.90).to(inp_vec.device)
    masked_vec[
        mask2mask_idx
    ] = mask_vec  # mask token is the last in the dict

    # Set 10% to a random token
    mask2rand_idx = mask2mask_idx & (torch.rand(inp_vec.shape[0]) < 1 / 9).to(inp_vec.device)
    masked_vec[mask2rand_idx] = inp_vec[torch.randint(0, len(inp_vec), (len(masked_idx),))]

    # Prepare sample_weights to pass to .fit() method
    sample_weights = torch.ones(atom_labels.shape)
    sample_weights[atom_labels == -1] = 0

    return masked_vec, atom_labels, sample_weights


class ComplexFormer(nn.Module):
    def __init__(
            self,
            x_dim: int,
            edge_dim: int,
            vec_dim: int = 512,
            graph_model: nn.Module = None,
            rings_kwargs: dict = None,
            transformer_kwargs: dict = None,

            # Rings Transformer arguments
            ring_layers: int = 1,
            ring_nheads: int = 2,
            ring_encoder_kw: dict = None,
            ring_encoder_block_kw: dict = None,

            # Molecular Transformer arguments
            mol_layers: int = 3,
            mol_nheads: int = 8,
            mol_encoder_kw: dict = None,
            mol_encoder_block_kw: dict = None,
    ):
        super(ComplexFormer, self).__init__()
        self.vec_size = vec_dim
        self.rings_kwargs = rings_kwargs if rings_kwargs else {}
        self.transformer_kwargs = transformer_kwargs if transformer_kwargs else {}

        self.x_project = nn.Linear(x_dim, vec_dim)
        self.e_project = nn.Linear(edge_dim, vec_dim)

        if graph_model:
            self.graph = graph_model
        else:
            self.graph = pygnn.GAT(
                vec_dim, vec_dim, 6,
                vec_dim, 0.1, norm=pygnn.LayerNorm(vec_dim),
                edge_dim=vec_dim, v2=True
            )

        self.ring_encoder_kw = ring_encoder_kw if ring_encoder_kw else {}
        self.ring_encoder_block_kw = ring_encoder_block_kw if ring_encoder_block_kw else {}
        self.ring_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=vec_dim,
                nhead=ring_nheads,
                batch_first=True,
                **self.ring_encoder_kw
            ),
            num_layers=ring_layers, **self.ring_encoder_block_kw
        )

        self.mol_encoder_kw = mol_encoder_kw if mol_encoder_kw else {}
        self.mol_encoder_block_kw = mol_encoder_block_kw if mol_encoder_block_kw else {}
        self.mol_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=vec_dim,
                nhead=mol_nheads, batch_first=True,
                **self.mol_encoder_kw
            ),
            num_layers=mol_layers, **self.mol_encoder_block_kw
        )

    def forward(self, x, edge_index, edge_attr, rings_node_index, rings_node_nums, mol_rings_nums, batch, ptr):
        x = self.x_project(x.float())
        e = self.e_project(edge_attr.float())

        x = self.graph(x, edge_index, edge_attr=e)

        x_r = self._rings_attention(x, rings_node_index, rings_node_nums)

    def _rings_attention(self, x, rings_node_index, rings_node_nums):
        x = x[rings_node_index.long()]

        B = rings_node_nums.shape[0]
        L = max(rings_node_nums).int().item()
        D = x.shape[-1]

        padded_X = torch.zeros((B, L, D)).to(x.device)
        mask = torch.zeros((B, L), dtype=torch.bool, device=x.device)

        start = 0
        for i, size in enumerate(rings_node_nums.long()):
            padded_X[i, :size] = x[start:start + size]
            mask[i, :size] = 1
            start += size

        x = self.ring_encoder(padded_X, mask=mask)
        return torch.max(x, dim=-1)


    @staticmethod
    def get_graph_model(graph_model: str = 'GCN', **kwargs):
        if graph_model == 'GCN':
            return pygnn.GCN(**kwargs)
        elif graph_model == 'GAT':
            return pygnn.GAT(**kwargs)

    @staticmethod
    def get_ringsformer(**kwargs):
        return nn.TransformerEncoder(**kwargs)
