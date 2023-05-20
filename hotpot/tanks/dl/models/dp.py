"""
python v3.7.9
@Project: hotpot
@File   : dp.py
@Author : Zhiyuan Zhang
@Date   : 2023/3/29
@Time   : 5:27
"""
import torch
import torch.nn as nn
import torch.nn.functional as fn
from torch import Tensor


class AttQeqPot(torch.nn.Module):
    """ Attention Charge equilibration Potential """

    _atom_elec_struct: Tensor = torch.tensor()
    _atom_cov_radii = None

    def __init__(
            self, atom_emb_size: int, atom_hides: int,
            att_heads: int, att_layers: int, desc_dims: int
    ):
        """"""
        super(AttQeqPot, self).__init__()
        self.atom_emb = AtomEmb(atom_emb_size, atom_hides)

    def forward(self):
        """"""


class AtomEmb(torch.nn.Module):
    """Given the atomic symbol to embedding the atom and represent atoms to D-dim vectors"""
    # TODO: the atom, from 0 to 118, properties represent to vector of real or int number
    # TODO: the vector can be get by atomic number, in which the 0 is refer to the zero vector.
    # TODO: the Tensor shouldn't be allowed to grad
    _atom_properties: Tensor = None

    def __init__(self, emb_dim: int, hides: int = 2):
        super(AtomEmb, self).__init__()
        self.p_vec_dim = self._atom_properties.size[1]  # the atomic property vector dimension
        self.emb_dim = emb_dim
        self.hides = hides

        self.dim_convert = nn.Linear(self.p_vec_dim, self.emb_dim)

        fcn = nn.ModuleList()
        for _ in range(hides):
            fcn.extend([nn.Linear(emb_dim, emb_dim), nn.ReLU()])
        self.emb_layers = Residue(fcn)

    def forward(self, x: Tensor):
        x = self.dim_convert(x)
        output = self.emb_layers(x)

        return output


class RelativeCoord(nn.Module):
    """
    Calculate the relative coordinates for every centre atoms.
    Give the [.., N, 3] Tensor, to be [.., N, N, 3] Tensor.
    """
    def __init__(self):
        super(RelativeCoord, self).__init__()

    def forward(self, c: Tensor):
        """
        Args:
            c(Tensor): the absolute coordinates

        Returns:

        """
        rc = c.unsequeeze(-2) - c.unsequeeze(-1)  # relative coordinates


class Residue(torch.nn.Module):
    """ The residue neural network """
    def __init__(self, wrapped_module: torch.nn.Module):
        super(Residue, self).__init__()
        self.wrapped_module = wrapped_module

    def forward(self, x: Tensor):
        return x + self.wrapped_module(x)

