from typing import Literal, Type

import torch
import torch.nn as nn
import torch_geometric.nn as pygnn

from peft import get_peft_model, LoraConfig, PeftModel, PeftMixedModel

from .core import CoreBase

__all__ = [
    'TargetTypeName',
    'Predictor',
    'apply_lora_to_predictor'
]


# Helper function to verify and replace layers
def replace_pyg_linear_in_mlp(mlp_module):
    """
    In-place replaces PyG Linear layers within a PyG MLP with torch.nn.Linear,
    preserving weights and biases.
    """
    # PyG MLP stores linear layers in a ModuleList called 'lins'
    if not hasattr(mlp_module, 'lins'):
        return

    for i, layer in enumerate(mlp_module.lins):
        # Check if it's a PyG Linear (or behaves like one)
        # We verify by checking for 'in_channels' which PyG uses, vs 'in_features' for Torch
        if hasattr(layer, 'in_channels') and not isinstance(layer, nn.Linear):

            # Check for Lazy Initialization (uninitialized weights)
            if layer.in_channels == -1 or layer.in_channels is None:
                raise ValueError(f"Layer {i} is lazily initialized. Run a forward pass with data before applying LoRA.")

            # 1. Create standard torch.nn.Linear
            new_layer = nn.Linear(
                in_features=layer.in_channels,
                out_features=layer.out_channels,
                bias=layer.bias is not None,
            )

            # 2. Copy weights and bias
            # PyG Linear weight shape is (out, in), same as nn.Linear
            with torch.no_grad():
                new_layer.weight.copy_(layer.weight)
                new_layer.weight.requires_grad = layer.weight.requires_grad
                if layer.bias is not None:
                    new_layer.bias.copy_(layer.bias)
                    new_layer.bias.requires_grad = layer.bias.requires_grad

            # 3. Replace the layer in the ModuleList
            mlp_module.lins[i] = new_layer
            print(f"Converted layer {i}: PyG Linear -> torch.nn.Linear")


TargetTypeName = Literal['num', 'xyz', 'onehot', 'binary']
class Predictor(nn.Module):
    define_types = ('num', 'xyz', 'onehot', 'binary')
    def __init__(
            self,
            in_size: int,
            target_type: TargetTypeName,
            hidden_dim: int = 1024,
            num_layers: int = 4,
            dropout: float = 0.1,
            out_act: Type[nn.Module] = nn.ReLU,
            **kwargs
    ):
        super(Predictor, self).__init__()
        self.in_layers = pygnn.MLP([in_size, hidden_dim])
        self.hidden_layers = pygnn.MLP(num_layers * [hidden_dim], dropout=dropout, norm=None)

        self.target_type = target_type
        if target_type == 'num':
            # self.out_layer = nn.Linear(in_size, 1)
            self.out_layer = nn.Linear(hidden_dim, 1)
            self.out_act = nn.LeakyReLU()
            # self.out_act = lambda out: torch.exp(out) - 20.
        elif target_type == 'xyz':
            self.out_layer = nn.Linear(hidden_dim, 3)
            self.out_act = out_act()
        elif target_type == 'onehot':
            try:
                self.onehot_type = kwargs['onehot_type']
            except KeyError:
                raise KeyError('For onehot predictor, `onehot_type` arg must be specified`')

            self.out_layer = nn.Linear(hidden_dim, self.onehot_type)
            self.out_act = nn.Softmax(dim=-1)
        elif target_type == 'binary':
            self.out_layer = nn.Linear(hidden_dim, 1)
            self.out_act = lambda out: out
        else:
            raise NotImplementedError(f"{target_type} is not implemented")

    def forward(self, z):
        z = self.in_layers(z)
        z = self.hidden_layers(z) + z
        # z = self.hidden_layers(z)
        z = self.out_layer(z)
        if self.target_type in ['num', 'xyz']:
            return z
            # return self.out_act(z)
        else:
            return self.out_act(z)

    def convert_to_standard_linear(self):
        """Explicitly convert internal PyG layers to Torch layers."""
        replace_pyg_linear_in_mlp(self.in_layers)
        replace_pyg_linear_in_mlp(self.hidden_layers)

    def apply_lora(self, rank=4, alpha=4, dropout=0.3, force: bool = False, target_modules=r"lins\.\d+", **kwargs):
        self.convert_to_standard_linear()
        if force or not isinstance(self.hidden_layers, (PeftModel, PeftMixedModel)):
            self.hidden_layers = apply_lora_to_predictor(
                self.hidden_layers, target_modules=target_modules,
                rank=rank, alpha=alpha,
                dropout=dropout, **kwargs
            )


def apply_lora_to_predictor(predictor_model, target_modules=None, rank=4, alpha=16, dropout=0.3, **kwargs):
    """Applies LoRA to the predictor model."""
    peft_config = LoraConfig(
        r=rank,
        target_modules=target_modules,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=None,
        **kwargs
    )

    return get_peft_model(predictor_model, peft_config)
