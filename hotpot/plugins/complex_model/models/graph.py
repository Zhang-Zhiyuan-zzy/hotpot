import torch
from torch import nn
from torch.nn import functional as F


class CompleteGraph(nn.Module):
    def __init__(
            self,
            vic_dim: int,
            nheads: int
    ):
        super(CompleteGraph, self).__init__()
        
