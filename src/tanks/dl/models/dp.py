"""
python v3.7.9
@Project: hotpot
@File   : dp.py
@Author : Zhiyuan Zhang
@Date   : 2023/3/29
@Time   : 5:27
"""
import torch
from torch import Tensor


class AttQeqPot(torch.nn.Module):
    """ Attention Charge equilibration Potential """

    _atom_elec_struct: Tensor = torch.tensor()
    _atom_cov_radii = None

    def __init__(
            self, atom_emb_size: int, att_heads: int,
            att_layers: int, desc_dims: int
    ):
        """"""

    def forward(self):
        """"""
