"""
python v3.9.0
@Project: graph.py
@File   : gen_decoder
@Auther : Zhiyuan Zhang
@Data   : 2024/9/10
@Time   : 14:51
"""
import torch
import torch.nn as nn


class NodeEdgeDecoder(nn.Module):
    """ Decoder a random vector with Gaussian distribution to a graph with a node-edge structure """
    def __init__(self):
        super(NodeEdgeDecoder, self).__init__()

    def forward(self, z: torch.Tensor):
        """"""
