"""
python v3.9.0
@Project: hotpot
@File   : vae
@Auther : Zhiyuan Zhang
@Data   : 2024/8/16
@Time   : 9:24
"""
from typing import Callable, Union, Literal, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric as pyg
import torch_geometric.nn as pyg_nn

from ..function.graph import graph_spectrum_similarity, batch_spectrum
from ..function.base import get_positional_encoding, kl_div_with_multivariate_normal

class VAE(nn.Module):
    def __init__(self, encoder, decoder, latent_dim=1024):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_norm = nn.BatchNorm1d(latent_dim)

    def forward(self, *args, **kwargs):
        mu, logvar = self.encoder(*args, **kwargs)

        # Sampling from latent space
        z = self.reparameterize(mu, logvar)

        # Reconstruction
        return self.decoder(z), mu, logvar, z

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        Args:
            mu: (Tensor) Mean of the latent Gaussian [B x D]
            logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        Return:
            (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        # return mu
        return eps * std + mu

    @staticmethod
    def kld_loss(mu, logvar):
        return torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)


class GraphEncoder(nn.Module):
    """"""
    def __init__(
            self,
            in_channels: int,
            hidden_channels: int = 16,
            gnn_layers: int = 8,
            gnn_pool_ratio: Union[int, float, Sequence] = None,
            latent_size: int = 512,
            num_layers: int = 10,
            n_head: int = 4,
            square_sigma: bool = True,
            batch_first: bool = False
        ):
        super(GraphEncoder, self).__init__()
        # storing basic arguments
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.n_head = n_head
        self.square_sigma = square_sigma
        self.batch_first = batch_first
        self.gnn_layers = gnn_layers
        self.gnn_dim_add = (latent_size - hidden_channels) // gnn_layers

        # Initialize node feature preprocess network
        self.encoder0 = pyg_nn.MLP([in_channels, hidden_channels])

        # Initialize GraphConv
        self.gnn_channels = (
            [hidden_channels] +
            [hidden_channels + i * self.gnn_dim_add for i in range(1, num_layers)] +
            [latent_size]
        )
        self.gin = nn.ModuleList([
            pyg_nn.GINConv(nn.Linear(in_dim, out_dim), eps=10-7)
            for in_dim, out_dim in zip(self.gnn_channels[:-1], self.gnn_channels[1:])
        ])

        # Initialize Pool
        self.pool_ratio = (
            1 - (1 / num_layers)
            if not gnn_pool_ratio
            else
            gnn_pool_ratio
            if not isinstance(gnn_pool_ratio, Sequence)
            else
            list(gnn_pool_ratio)
        )
        self.SAGPools = nn.ModuleList([
            pyg_nn.SAGPooling(dim, ratio=self.pool_ratio) for dim in self.gnn_channels[1:]
        ])
        self.gnn_norm = nn.ModuleList([
            nn.BatchNorm1d(dim, eps=1e-4) for dim in self.gnn_channels[1:]
        ])

        self.global_pool = pyg_nn.SAGPooling(latent_size, ratio=1)
        self.batch_norm = nn.BatchNorm1d(latent_size, eps=1e-4)

        # self.transformer = nn.Transformer(
        #     d_model=hidden_channels,
        #     nhead=n_head,
        #     num_encoder_layers=2,
        #     num_decoder_layers=2,
        #     dim_feedforward=1024,
        # )

        self.num_query = latent_size // hidden_channels + 1
        self.dim_after_transformer = hidden_channels * self.num_query

        # self.dim_align_layers = pyg_nn.MLP([32, latent_size])
        # self.dim_align_layers = pyg_nn.MLP([self.dim_after_transformer, latent_size])

        # self.mu_norm = nn.LayerNorm(latent_size)
        self.to_mu = pyg_nn.MLP(2 * [latent_size])
        self.to_logvar = pyg_nn.MLP(2 * [latent_size])

    def pool_graph(self, x, delta_ptr):
        query = get_positional_encoding(self.num_query, self.hidden_channels, len(delta_ptr), device=x.device)
        batch_padded_x, padded_mask = self._graph_unsqueeze_with_cpu_style(x, delta_ptr)
        # query = torch.triu(torch.ones(x.shape[0], self.num_query, self.num_query), diagonal=1).to(x.device)
        if not self.batch_first:
            query = query.transpose(0, 1)[1:]
            batch_padded_x = batch_padded_x.transpose(0, 1)

        return self.transformer(
            batch_padded_x,
            query,
            src_key_padding_mask=padded_mask
        )

    def forward(self, x, edge_index, batch, ptr):
        # x, edge_index, batch, ptr = inputs.x.float(), inputs.edge_index, inputs.batch, inputs.ptr
        delta_ptr = ptr[1:] - ptr[:-1]
        # batch_max_len = delta_ptr.max()
        # src_padding_mask = torch.zeros((len(delta_ptr), batch_max_len))
        # src_padding_mask[:, :delta_ptr] = 1
        # src_padding_mask = self._get_sequence_padding_mask(delta_ptr)

        x = self.encoder0(x.float())

        for conv, pool, norm in zip(self.gin, self.SAGPools, self.gnn_norm):
            x = conv(x, edge_index)
            x, edge_index, edge_attr, batch, perm, score = pool(x, edge_index, batch=batch)
            x = norm(x)

        prev_z, edge_index, edge_attr, batch, perm, score = self.global_pool(x, edge_index, batch=batch)
        # prev_z = self.pool_graph(x, delta_ptr)
        prev_z = self.batch_norm(prev_z)

        # prev_z = self.dim_align_layers(prev_z)
        # prev_z = self.dim_align_layers(prev_z.transpose(0, 1).reshape((len(delta_ptr), -1)))

        mu = self.to_mu(prev_z)
        logvar = self.to_logvar(prev_z)

        return mu, logvar

    @staticmethod
    def _get_sequence_padding_mask(seq_lengths: torch.Tensor) -> torch.Tensor:
        assert seq_lengths.dim() == 1

        max_len = int(torch.max(seq_lengths))
        num_seq = seq_lengths.size(0)

        indices_arange = torch.arange(max_len).expand(num_seq, max_len).to(seq_lengths.device)

        return indices_arange >= seq_lengths.unsqueeze(1)

    @staticmethod
    def _graph_unsqueeze_with_cpu_style(x: torch.Tensor, num_nodes_per_graph) -> (torch.Tensor, torch.Tensor):
        max_graph_nodes = torch.max(num_nodes_per_graph)
        batch_size = num_nodes_per_graph.size(0)

        batched_padded_x = torch.zeros((batch_size, max_graph_nodes, x.size(-1)), device=x.device)
        padded_mask = torch.ones((batch_size, max_graph_nodes), device=x.device)

        # padding sequence vectors
        start_idx = 0
        for i in range(batch_size):
            num_nodes = num_nodes_per_graph[i]
            end_idx = start_idx + num_nodes
            batched_padded_x[i, :num_nodes, :] = x[start_idx:end_idx, :]
            padded_mask[i, :num_nodes] = 0

            start_idx = end_idx

        return batched_padded_x, padded_mask

    @staticmethod
    def _graph_unsqueeze_with_gpu_style(x: torch.Tensor, num_nodes_per_graph, ptr: torch.Tensor) -> torch.Tensor:
        max_graph_nodes = torch.max(num_nodes_per_graph)
        batch_size = num_nodes_per_graph.size(0)
        batched_padded_x = torch.zeros((batch_size, max_graph_nodes, x.size(-1)), device=x.device)

        batched_indices = torch.arange(int(max_graph_nodes), device=x.device).expand((batch_size, max_graph_nodes))
        mask_indices = batched_indices < num_nodes_per_graph.unsqueeze(-1)

        block_indices = torch.arange(int(ptr[-1]), device=x.device).expand((batch_size, ptr[-1]))
        block_indices = (ptr[:-1].unsqueeze(-1) <= block_indices) & (block_indices < ptr[1:].unsqueeze(-1))

        batched_padded_x[mask_indices] = x[block_indices]

        return batched_padded_x


class NodeNumDecoder(nn.Module):
    """ Decode the latent vector to number of nodes """
    def __init__(self, latent_dim: int):
        super(NodeNumDecoder, self).__init__()
        self.node_decoder = pyg_nn.MLP([latent_dim, latent_dim//2, latent_dim//8, latent_dim//16, 1])

    def forward(self, z: torch.Tensor):
        return self.node_decoder(z)


class MatrixDecoder(nn.Module):
    """"""
    def __init__(self, latent_dim: int, n_heads: int = 4, num_layers: int = 3):
        super(MatrixDecoder, self).__init__()

        self.decoder0 = self.encoder_start = pyg_nn.MLP([latent_dim, latent_dim])
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=latent_dim, nhead=n_heads),
            num_layers=num_layers
        )


class GraphDecoder(nn.Module):
    """"""
    def __init__(
            self,
            latent_size: int,
            kernel_size: int = 16,
            mean_node_num: int = 32,
            mat_act: Callable = None,
            batch_first: bool = False
    ):
        """"""
        super(GraphDecoder, self).__init__()
        self.z_size = latent_size
        self.batch_first = batch_first
        self.mean_node_num = mean_node_num

        # self.latent_norm = nn.BatchNorm1d(z_size)

        self.mlp = pyg_nn.MLP([latent_size, latent_size, latent_size, latent_size], act='LeakyReLU')
        self.pred_node_num = nn.Linear(latent_size, 1)

        self.latent_proliferation = KernelLinear(latent_size, latent_size, kernel_size)

        self.encoder_start = pyg_nn.MLP([latent_size, latent_size])

        self.transformer = nn.Transformer(
            d_model=latent_size,
            nhead=2,
            num_encoder_layers=4,
            num_decoder_layers=2
        )

        self.one_to_six = KernelLinear(latent_size, latent_size, 6)
        self.to_seq_vector = pyg_nn.MLP([latent_size] * 3, norm='layer_norm')

    def forward(self, z: torch.Tensor):
        """"""
        # z = self.latent_norm(z)
        z_to_num = self.mlp(z) + z
        node_num = F.leaky_relu(self.pred_node_num(z_to_num)) \
                   # + self.mean_node_num

        max_node_num = int(node_num.max())
        z = self.latent_proliferation(z)

        batch_size = z.shape[1]
        query = get_positional_encoding(max_node_num, z.shape[-1], batch_size, device=z.device)
        if not self.batch_first:
            query = query.transpose(0, 1)[1:]

        seq_vector = self.transformer(z, query)
        seq_vector = self.one_to_six(seq_vector)

        seq_vector = self.to_seq_vector(seq_vector) + seq_vector

        if not self.batch_first:
            seq_vector = seq_vector.transpose(0, 2)

        matrix = seq_vector @ seq_vector.transpose(-1, -2) / seq_vector.size(-1)

        return node_num, self._to_spectrum(matrix, node_num)
        # return self._to_spectrum(matrix, node_num), node_num

    @staticmethod
    def _to_spectrum(matrix: torch.Tensor, node_num: torch.Tensor) -> torch.Tensor:
        return batch_spectrum(matrix, node_num)


class KernelLinear(nn.Module):
    """"""
    def __init__(self, in_features: int, out_features: int, kernel_num: int = 1, bias=True) -> None:
        super(KernelLinear, self).__init__()
        self.kernel_num = kernel_num
        self.weight = nn.Parameter(torch.randn(kernel_num, out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(kernel_num, 1, out_features))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.matmul(x.unsqueeze(-3), self.weight.transpose(-1, -2))
        if self.bias is not None:
            return x + self.bias
        else:
            return x


class MultiKernelMLP(nn.Module):
    """"""
    def __init__(
            self,
            in_feature: int = None,
            out_feature: int = None,
            hidden_feature: int = None,
            filter_number: int = None,
            layer_number: int = None
    ):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.hidden_feature = hidden_feature
        self.filter_number = filter_number
        self.layer_number = layer_number

        self.kernel_linear = KernelLinear(in_feature, hidden_feature, filter_number)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        x = self.kernel_linear(x)

class SpectrumLoss(nn.Module):
    def __init__(self):
        super(SpectrumLoss, self).__init__()


def spectrum_loss(
        pred_spectrum: torch.Tensor,
        target_spectrum: torch.Tensor,
        metric: Union[Literal['min', 'mean'], Callable] = 'min',
        to_distance: bool = False,
        eps: float = 1e-6
) -> torch.Tensor:
    """"""
    assert len(pred_spectrum.shape) == len(target_spectrum.shape) in [2, 3]
    if len(pred_spectrum.shape) == 2:
        pred_spectrum = pred_spectrum.unsqueeze(-2)
        target_spectrum = target_spectrum.unsqueeze(-2)

    assert pred_spectrum.shape[:2] == target_spectrum.shape[:2]

    # padding spectrum
    if (p_len := pred_spectrum.shape[-1]) > (t_len := target_spectrum.shape[-1]):
        target_spectrum = F.pad(target_spectrum, (0, p_len - t_len))
    elif p_len < t_len:
        pred_spectrum = F.pad(pred_spectrum, (0, t_len - p_len))

    batch_similarity = torch.cosine_similarity(target_spectrum.transpose(-1, -2), pred_spectrum.transpose(-1, -2))

    if metric is None:
        similarity = batch_similarity
    elif isinstance(metric, Callable):
        similarity = metric(batch_similarity)
    elif metric == 'min':
        similarity = batch_similarity.min(dim=-1)[0]
    elif metric == 'mean':
        similarity = batch_similarity.mean(dim=-1)
    else:
        raise NotImplementedError(f'The metric {metric} is not recognized !!!')

    if to_distance:
        return 1 / (similarity + eps)
    else:
        raise similarity


def graph_vae_loss(
        true_atom_num: torch.Tensor,
        pred_atom_num: torch.Tensor,
        true_spectrum: torch.Tensor,
        pred_spectrum: torch.Tensor,
        latent_vector: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor
):
    """"""
    atom_number_loss = F.mse_loss(pred_atom_num.flatten(), true_atom_num.flatten().float())
    graph_distance = spectrum_loss(pred_spectrum, true_spectrum, to_distance=True)

    kld_loss = kl_div_with_multivariate_normal(latent_vector) * 1e-8
    # kld_loss = torch.log(torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0))

    # return kld_loss
    # return torch.mean(graph_distance)
    # return atom_number_loss + torch.mean(graph_distance)
    return torch.mean(graph_distance) + kld_loss
    # return atom_number_loss + torch.mean(graph_distance) + kld_loss
