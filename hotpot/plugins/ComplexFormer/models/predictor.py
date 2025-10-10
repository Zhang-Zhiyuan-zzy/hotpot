from typing import Literal, Type

import torch.nn as nn
import torch_geometric.nn as pygnn

from .core import CoreBase

TargetTypeName = Literal['num', 'xyz', 'onehot', 'binary']
class Predictor(nn.Module):
    define_types = ('num', 'xyz', 'onehot', 'binary')
    def __init__(
            self,
            in_size: int,
            target_type: TargetTypeName,
            num_layers: int = 2,
            dropout: float = 0.1,
            out_act: Type[nn.Module] = nn.ReLU,
            **kwargs
    ):
        super(Predictor, self).__init__()
        self.hidden_layers = pygnn.MLP(num_layers * [in_size], dropout=dropout)

        self.target_type = target_type
        if target_type == 'num':
            self.out_layer = nn.Linear(in_size, 1)
            self.out_act = nn.LeakyReLU()
        elif target_type == 'xyz':
            self.out_layer = nn.Linear(in_size, 3)
            self.out_act = out_act()
        elif target_type == 'onehot':
            try:
                self.onehot_type = kwargs['onehot_type']
            except KeyError:
                raise KeyError('For onehot predictor, `onehot_type` arg must be specified`')

            self.out_layer = nn.Linear(in_size, self.onehot_type)
            self.out_act = nn.Softmax(dim=-1)
        elif target_type == 'binary':
            self.out_layer = nn.Linear(in_size, 1)
            self.out_act = lambda out: out
        else:
            raise NotImplementedError(f"{target_type} is not implemented")

    def forward(self, z):
        z = self.hidden_layers(z) + z
        z = self.out_layer(z)
        if self.target_type in ['num', 'xyz']:
            return z
        else:
            return self.out_act(z)

    @classmethod
    def link_with_core_module(cls, core: CoreBase, target_type: TargetTypeName, **kwargs):
        return cls(core.vec_size, target_type, **kwargs)