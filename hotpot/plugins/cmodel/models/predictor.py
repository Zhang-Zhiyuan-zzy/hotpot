from typing import Literal, Type

import torch.nn as nn
import torch_geometric.nn as pygnn

TargetTypeName = Literal['num', 'xyz', 'onehot', 'binary']
class Predictor(nn.Module):
    def __init__(
            self,
            in_size: int,
            target_pattern: TargetTypeName,
            num_layers: int = 2,
            dropout: float = 0.1,
            act: Type[nn.Module] = nn.ReLU,
            out_act: Type[nn.Module] = nn.ReLU,
            **kwargs
    ):
        super(Predictor, self).__init__()
        self.hidden_layers = pygnn.MLP(num_layers * [in_size], dropout=dropout)

        self.target_pattern = target_pattern
        if target_pattern == 'num':
            self.out_layer = nn.Linear(in_size, 1)
            self.out_act = nn.LeakyReLU()
        elif target_pattern == 'xyz':
            self.out_layer = nn.Linear(in_size, 3)
            self.out_act = out_act()
        elif target_pattern == 'onehot':
            self.out_layer = nn.Linear(in_size, kwargs.get("onehot_type", 119))
            self.out_act = nn.Softmax(dim=-1)
        elif target_pattern == 'binary':
            self.out_layer = nn.Linear(in_size, 1)
            self.out_act = nn.Sigmoid()
        else:
            raise NotImplementedError(f"{target_pattern} is not implemented")

    def forward(self, z):
        z = self.hidden_layers(z) + z
        z = self.out_layer(z)
        if self.target_pattern in ['num', 'xyz']:
            return z
        else:
            return self.out_act(z)