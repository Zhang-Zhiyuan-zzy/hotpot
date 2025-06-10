# -*- coding: utf-8 -*-
"""
===========================================================
 Project   : hotpot
 File      : se3graph
 Created   : 2025/5/26 9:15
 Author    : zhang
 Python    : 
-----------------------------------------------------------
 Description
 ----------------------------------------------------------
 SE(3)-Equivariant Layer Implementation with Auto-Channel Splitting

===============================================================

This module provides a flexible SE(3)-equivariant neural network layer
using PyTorch and e3nn. It supports:

- **max_l parameter:** Easily control the maximum order l of SO(3) features
  (e.g., mix scalars, vectors, tensors up to l=2 or higher).
- **Automatic channel allocation:**
  Set in_irreps/out_irreps to 'auto' or None, and the layer will
  automatically split the embedding dim across all allowed l, with
  as even a distribution as possible.
- **Explicit irreps mode:**
  You may also specify custom irreps strings for full control over representation layout.
- **Residual connection and normalization:**
  If input and output channels match, a skip/residual connection is used.
- **e3nn-powered tensor product:**
  Ensures strict SE(3)/SO(3) equivariance of all message-passing and update operations.

**Typical usage:**

1. Instantiate `SE3EquivariantLayer` in "auto" mode (recommended for most users)
2. Integrate as a building block in your molecule, material, or general geometric graph networks
3. Adjust max_l and embedding dimension to trade off modeling power and computational cost

This implementation can serve as a template for advanced equivariant GNNs in
physics, chemistry, materials, or molecular science.
===========================================================
"""
from typing import Optional, Sequence
from enum import Enum

import math, torch
from torch import nn

from torch_cluster import radius_graph
from torch_scatter import scatter

from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.o3 import Irreps, spherical_harmonics, FullyConnectedTensorProduct
from e3nn.nn import FullyConnectedNet


Parity = Enum('Parity', [('Even', 1), ('Odd', -1)])

# -----------------------------------------------------------
# helper: automatically generate Irreps string like "10x0e + 5x1o + 2x2e"
# -----------------------------------------------------------
def _get_eo(l, parity: Parity):
    if parity == 1:
        return 'e'
    elif parity == -1:
        return 'e' if l % 2 == 0 else 'o'
    else:
        raise ValueError('parity must be either 1 (even) or -1 (odd)')


def auto_irreps(emb_dim: int, max_l: int, parity: Parity = -1) -> Irreps:
    """
    Evenly distribute each order's "total dimension," meaning multiplicity*(2l+1),
    so that Σ multiplicity*(2l+1) == emb_dim
    """
    dims = [2*l + 1 for l in range(max_l + 1)]           # [1,3,5,...]
    total_unit = sum(dims)                                # 1+3+5+...
    base, extra = divmod(emb_dim, total_unit)             # average share
    mults = [base] * len(dims)

    # Distribute the remainder round-robin from low l to high l
    idx = 0
    while extra > 0:
        mults[idx] += 1
        extra -= dims[idx]
        idx = (idx + 1) % len(dims)

    parts = [
        f"{m}x{l}{_get_eo(l, parity)}"
        for m, l in zip(mults, range(max_l + 1)) if m > 0
    ]
    return Irreps(" + ".join(parts))


# -----------------------------------------------------------
# Core equivariant layer
# -----------------------------------------------------------
def _get_irreps(irreps: str, emb_dim: int, max_l: int, parity: Parity) -> str:
    return irreps or auto_irreps(emb_dim, max_l, parity)



class SE3EquivLayer(nn.Module):
    def __init__(
            self,
            emb_dim: int,
            in_irreps: Optional[str] = None,
            out_irreps: Optional[str] = None,
            max_l: Optional[int] = 1,
            parity: Parity = -1,
            weights_mlp: Sequence[int] = (64,),
            max_radius: float = 4.5,
            num_radius_basis: int = 10,
    ):
        super(SE3EquivLayer, self).__init__()

        self.in_irreps = Irreps(_get_irreps(in_irreps, emb_dim, max_l, parity))
        self.out_irreps = Irreps(_get_irreps(out_irreps, emb_dim, max_l, parity))

        self.lmax = self.in_irreps.lmax * self.out_irreps.lmax
        self.sh_irreps = Irreps.spherical_harmonics(lmax=self.lmax, p=parity)

        self.tp = FullyConnectedTensorProduct(self.in_irreps, self.sh_irreps, self.out_irreps, shared_weights=False)

        self.radius_mlp_channels = [num_radius_basis] + list(weights_mlp) + [self.tp.weight_numel]
        self.radius_mlp = FullyConnectedNet(self.radius_mlp_channels, torch.relu)
        self.max_radius = max_radius
        self.num_radius_basis = num_radius_basis

    def forward(self, x, src, dst, edge_vec):
        """"""
        sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        edge_length_embedding = soft_one_hot_linspace(
            edge_vec.norm(dim=1),
            start=0.0,
            end=self.max_radius,
            number=self.num_radius_basis,
            basis='smooth_finite',
            cutoff=True,
        )
        edge_length_embedding = edge_length_embedding.mul(self.num_radius_basis ** 0.5)
        weights = self.radius_mlp(edge_length_embedding)

        summand = self.tp(x[src], sh, weights)
        return scatter(summand, dst, dim=0, dim_size=len(x)).div(len(src) / len(x))


class SE3Net(nn.Module):
    def __init__(
            self,
            emb_dim: int = 64,
            in_irreps: Optional[Irreps] = None,
            out_irreps: Optional[Irreps] = None,
            max_l: int = 1,
            parity: Parity = -1,
            weights_mlp: Sequence[int] = (16, 64),
            num_layers: int = 2,
            max_radius: float = 4.5,
            num_radius_basis: int = 10,
    ):
        super().__init__()
        self.max_radius = max_radius

        in_irreps = _get_irreps(in_irreps, emb_dim, max_l, parity)
        out_irreps = _get_irreps(out_irreps, emb_dim, max_l, parity)

        self.layers = nn.ModuleList([
            SE3EquivLayer(emb_dim, in_irreps, out_irreps, max_l, parity, weights_mlp, max_radius, num_radius_basis)
            for _ in range(num_layers)
        ])

    def D_in(self, rot: torch.Tensor) -> torch.Tensor:
        """ Get the input group representation under the SO(3) rotation """
        return self.layers[0].in_irreps.D_from_matrix(rot)

    def D_out(self, rot: torch.Tensor) -> torch.Tensor:
        """ Get the output group representation under the SO(3) rotation """
        return self.layers[-1].out_irreps.D_from_matrix(rot)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        src, dst = radius_graph(pos, self.max_radius)
        edge_vec = pos[dst] - pos[src]

        # Performing convolution
        out = x
        for layer in self.layers:
            out = layer(out, src, dst, edge_vec) + out

        return out


# -----------------------------------------------------------
# Quick demo
# -----------------------------------------------------------
if __name__ == "__main__":
    from scipy.spatial.transform import Rotation as Rot
    import matplotlib.pyplot as plt


    N, emb_dim, max_l = 16, 64, 2
    xyz = torch.randn(N, 3)
    h = torch.ones(N, emb_dim)

    rot_mat = o3.rand_matrix()
    euler_angles = Rot.from_matrix(rot_mat).as_euler('zyx', degrees=True)
    print(rot_mat)
    print(euler_angles)

    # Build fully-connected edges
    net = SE3Net(emb_dim=emb_dim, max_l=max_l)

    D_in = net.D_in(rot_mat).numpy()
    D_out = net.D_out(rot_mat).numpy()
    y_rot_before = net(h @ net.D_out(rot_mat).T, xyz @ rot_mat.T)
    y_rot_after = net(h, xyz) @ net.D_out(rot_mat).T

    res = torch.allclose(y_rot_after, y_rot_before, rtol=1e-4, atol=1e-4)

    print(res)

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(D_in, cmap='coolwarm')
    axs[0, 1].imshow(D_out, cmap='coolwarm')
    axs[1, 0].imshow(h.detach().numpy())
    axs[1, 1].imshow(y_rot_after.detach().numpy())

    fig.show()
