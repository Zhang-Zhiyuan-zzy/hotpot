"""
python v3.9.0
@Project: hotpot
@File   : base
@Auther : Zhiyuan Zhang
@Data   : 2024/8/23
@Time   : 9:41
"""
from typing import Union

import numpy as np
import torch
from torch import distributions as dist
from torch.distributions import MultivariateNormal as MNormal

def eigen_abs(x: torch.Tensor) -> torch.Tensor:
    """ Convert all eigenvalues to their absolute values """
    if torch.det(x) == 0:
        raise AttributeError("The input matrix is singular.")

    eigval, eigvec = torch.linalg.eigh(x)

    abs_eigval = torch.diag(torch.abs(eigval))

    return eigvec @ abs_eigval @ torch.inverse(eigvec)


def is_symmetric(x, tol: float = 5e-7) -> bool:
    """ Check if A is a symmetric matrix """
    # return torch.equal(x, x.T)
    x = x / torch.abs(x).max()
    return torch.allclose(x, x.transpose(-1, -2), rtol=tol, atol=tol)


def matmul_power(x: torch.Tensor, exp: Union[int, float, torch.Tensor], abs_val: bool = False) -> torch.Tensor:
    """"""
    if not torch.det(x):
        raise AttributeError('The input matrix is singular.')

    symmetric = is_symmetric(x)

    eigval, eigvec = torch.linalg.eigh(x)

    # Compute the powered eigenvalues
    if abs_val:
        powered_diag = torch.diag(torch.abs(eigval) ** exp)
    else:
        powered_diag = torch.diag(eigval ** exp)

    # Reconstruct the matrix A^x
    if symmetric:
        return eigvec @ powered_diag @ eigvec.T
    else:
        return eigvec @ powered_diag @ torch.inverse(eigvec)


def kl_div_with_multivariate_normal(z: torch.Tensor, eps: float = 1e-2) -> torch.Tensor:
    mu = z.mean(dim=0)
    cov = torch.cov(z.T) + eps * torch.eye(z.shape[1], device=z.device)

    mnd = MNormal(mu, cov)
    norm = MNormal(torch.zeros_like(mu, device=z.device), (1. + eps) * torch.eye(mu.size(-1), device=z.device))

    return dist.kl.kl_divergence(norm, mnd)


def get_positional_encoding(
        max_seq_len: int,
        embed_dim: int,
        batch_size: int = None,
        base: int = None,
        device: torch.device = None
):
    """
    Generate positional encoding.
    Args:
        max_seq_len: maximum sequence length
        embed_dim: embedding dimension
        batch_size: Size of min-batch
        base: base number in positional encoding
        device: which device to use for computing
    """
    if base is None:
        base = int(100 * max_seq_len / embed_dim)

    dim_arange = torch.arange(embed_dim)
    seq_arange = torch.arange(max_seq_len + 1).unsqueeze(1)

    # zero_dim = torch.zeros(embed_dim).unsqueeze(0)

    positional_encoding = torch.nan_to_num(seq_arange / torch.pow(base, 2 * dim_arange / seq_arange)).to(device)
    # positional_encoding = torch.cat((zero_dim, positional_encoding), dim=0)

    # positional_encoding = np.array([
    #     [pos / np.power(10000, 2 * i / embed_dim) for i in range(embed_dim)]
    #     if pos != 0 else np.zeros(embed_dim) for pos in range(max_seq_len)])

    positional_encoding[1:, 0::2] = torch.sin(positional_encoding[1:, 0::2])  # dim 2i even
    positional_encoding[1:, 1::2] = torch.cos(positional_encoding[1:, 1::2])  # dim 2i+1 odd

    if batch_size is None:
        return positional_encoding
    else:
        return positional_encoding.expand((batch_size, max_seq_len+1, embed_dim))
