"""
python v3.9.0
@Project: hotpot
@File   : pytorch_func
@Auther : Zhiyuan Zhang
@Data   : 2024/8/16
@Time   : 8:53

Notes:
    This module define some convenient functions for running Pytorch
"""
import logging
from typing import Callable, Iterable

import torch
from torch.optim import Adam
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

from hotpot.dataset import load_dataset
from hotpot.plugins.dl.generate.vae import VAE, GraphEncoder, GraphDecoder, graph_vae_loss
from hotpot.dataset.graphs.spectrum import SpectrumDataset
# from hotpot.plugins.dl.function.graph import graph_spectrum_loss


def _graph_data_organizer(batch: Batch):
    inputs = TensorContainer(
        x=batch.x.float(),
        edge_index=batch.edge_index,
        batch=batch.batch,
        ptr=batch.ptr
    )

    target = TensorContainer(
        atom_num=batch.ptr[1:] - batch.ptr[:-1],
        spectrum=batch.spectrum,
        batch=batch.batch,
        ptr=batch.ptr
    )

    return inputs, target
    # return batch, batch.spectrum


class TensorContainer:
    """ A container stored tensors """
    def __init__(self, *args, **kwargs: torch.Tensor):
        """"""
        # Check whether all inputs are Tensor
        for tensor in args:
            assert isinstance(tensor, torch.Tensor)
        for name, tensor in kwargs.items():
            assert isinstance(tensor, torch.Tensor)

        self._args = args
        self._tensors = kwargs

    def __dir__(self) -> Iterable[str]:
        return ('to', 'get') + tuple(self._tensors.keys())

    def __getattr__(self, item):
        if (values := self._tensors.get(item)) is not None:
            return values
        elif (values := getattr(self, item)) is not None:
            return values
        else:
            raise AttributeError(f'the {self.__class__.__name__} object have not attribute {item}')

    def __setattr__(self, key, value):
        if key in ('_args', '_tensors'):
            super().__setattr__(key, value)
        elif isinstance(value, torch.Tensor):
            self._tensors[key] = value
        else:
            raise TypeError(f'the assigned object must be a Tensor, instead of {type(value)}')

    def to(self, where):
        """ Operate all tensors contained in self """
        for name, tensor in self._tensors.items():
            self._tensors[name] = tensor.to(where)

        return self

    @property
    def args(self):
        return self._args

    @property
    def kwargs(self):
        return self._tensors


def train(
        model,
        dataloader,
        loss_fn,
        opti_cls,
        learning_rate: float,
        epoch_num: int,
        device=torch.device("cpu"),
        data_organizer: Callable = None,
        loss_organizer: Callable = None,
        valid_func: Callable = None,
        valid_per_batch: int = 100
):
    model.to(device)
    model.train()
    get_model_size(model)
    optimizer = opti_cls(model.parameters(), lr=learning_rate, weight_decay=0.1 * learning_rate)

    for epoch in range(epoch_num):
        print(f"Epoch -- {epoch}")
        for i, batch in enumerate(iter(dataloader)):

            if i != 0 and i % valid_per_batch == 0 and valid_func:
                validate(i, model, batch, valid_func, device, data_organizer)

            else:
                if isinstance(data_organizer, Callable):
                    inputs, target = data_organizer(batch)
                else:
                    inputs, target = _graph_data_organizer(batch)

                inputs = inputs.to(device)
                target = target.to(device)

                # Forward propagation
                out = model(*inputs.args, **inputs.kwargs)

                if isinstance(loss_organizer, Callable):
                    loss = loss_fn(*loss_organizer(target, out))
                else:
                    loss = loss_fn(target, out)

                # Back propagation
                loss.backward()
                logging.debug("\n".join(f'{n}: {p}' for n, p in model.named_parameters()))
                show_gpu_memory_utils(device)

                has_grad_nan = any(torch.any(torch.isnan(p.grad)) for p in model.parameters() if isinstance(p.grad, torch.Tensor))

                print(loss, max([torch.max(p.grad).item() for p in model.parameters() if isinstance(p.grad, torch.Tensor)]))
                optimizer.step()
                optimizer.zero_grad()


def validate(
        item_num: int,
        model,
        batch,
        valid_func: Callable,
        device: torch,
        data_organizer: Callable = None
):
    if isinstance(data_organizer, Callable):
        inputs, target = data_organizer(batch)
    else:
        inputs, target = _graph_data_organizer(batch)

    inputs = inputs.to(device)
    target = target.to(device)

    with torch.no_grad():
        out = model(*inputs.args, **inputs.kwargs)

        results = valid_func(target, out)

        print(f'---------- valid in {item_num} iteration -----------------')
        for name, value in results.items():
            print(f'\t{name}: \t{value}')


def show_gpu_memory_utils(device=torch.device('cuda')):
    total_memory = torch.cuda.get_device_properties(device).total_memory
    reserved_memory = torch.cuda.memory_reserved(device)
    allocated_memory = torch.cuda.memory_allocated(device)
    free_memory = reserved_memory - allocated_memory

    print(f"Total GPU memory: {total_memory / (1024 ** 3):.2f} GB")
    print(f"Free GPU memory: {free_memory / (1024 ** 3):.2f} GB")


def get_model_size(module):
    total_params = sum(p.numel() for p in module.parameters())
    param_size = 4  # Size of float32 in bytes
    total_size = total_params * param_size / (1024 ** 3)  # Convert to MB

    print(f'Total model size: {total_size:.2f}')
    return total_size


def loss_organizer(target, out):
    true_atom_num, true_spectrum, ptr = target.atom_num, target.spectrum, target.ptr
    (pred_atom_num, pred_spectrum), pred_latent, mu, logvar = out

    bincount = ptr[1:] - ptr[:-1]
    max_nodes = max(bincount)
    sliced_spectrum = torch.split(true_spectrum, bincount.tolist(), dim=1)

    padding_spectrum = [
        F.pad(t, (0, max_nodes - t.shape[1]))
        for t in sliced_spectrum
    ]

    true_spectrum = torch.stack(padding_spectrum)

    return true_atom_num, pred_atom_num, true_spectrum, pred_spectrum, pred_latent, mu, logvar


def to_valid(target, out):
    """"""
    true_atom_num, true_spectrum, ptr = target.atom_num, target.spectrum, target.ptr
    (pred_atom_num, pred_spectrum), pred_latent, mu, logvar = out

    return {
        'Atom number error': F.mse_loss(true_atom_num, pred_atom_num).item(),
        'Atom Error median': torch.median(torch.abs(pred_atom_num - true_atom_num)).item(),
        "KL divergence": torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
    }


if __name__ == "__main__":
    latent_size = 256

    loader = DataLoader(SpectrumDataset(), batch_size=256)
    model = VAE(
        encoder=GraphEncoder(
            in_channels=9,
            gnn_pool_ratio=0.8,
            latent_size=latent_size
        ),
        decoder=GraphDecoder(
            latent_size=latent_size
        )
    )
    model.decoder.pred_node_num.bias = torch.nn.Parameter(torch.tensor([43.]))

    train(
        model=model,
        dataloader=loader,
        loss_fn=graph_vae_loss,
        opti_cls=Adam,
        learning_rate=1e-3,
        epoch_num=10,
        device=torch.device('cuda'),
        loss_organizer=loss_organizer,
        valid_func=to_valid,
        valid_per_batch=10
    )
