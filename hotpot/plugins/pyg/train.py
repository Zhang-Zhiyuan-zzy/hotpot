"""
python v3.9.0
@Project: hotpot
@File   : train
@Auther : Zhiyuan Zhang
@Data   : 2025/1/13
@Time   : 16:40
"""
from typing import Iterable, Type, Callable, Any

import torch
import torch.nn as nn
from click.core import batch
from torch.utils.data import DataLoader

# import optuna


def model_forward(model, batch_, *batch_input_attr):
    """"""
    if batch_input_attr:
        inp = {n: getattr(batch_, n) for n in batch_input_attr}
        return model(**inp)

    else:
        return model(batch_)


def model_backward(res, loss):
    ...


class Evaluator:
    """"""
    def __init__(
            self,
            eval_data,
            eval_step: int = 200,
            **test_func: Callable[[nn.Module, Any], float]
    ):
        self.eval_step = eval_step
        self.eval_data = eval_data
        self.test_func = test_func

        self.step = 0

    def __call__(self, model):
        self.step += 1

        if self.step % self.eval_step == 0:
            print(f"-----------Evaluate in step {self.step}:-------------")
            for name, func in self.test_func.items():
                metric = func(model, self.eval_data)
                print(f"{name}: {metric}")
            print(f"-----------Evaluate in step {self.step}:-------------")


class Printer(object):
    def __init__(
            self,
            print_step: int = 100,
            **metrics: Callable,
    ):
        self.print_step = print_step
        self.metrics = metrics

        self.step = 0
        self.contents = {n: 0 for n in metrics}

    def __call__(self, loss, res, batch_):
        for metric_name, func in self.metrics.items():
            self.contents[metric_name] += func(loss, res, batch_)

        self.step += 1
        if self.step % self.print_step == 0:
            for name, metric in self.contents.items():
                print(f"{self.step}, {name}: {metric}")


def train_nn(
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        loss_func: Callable,
        input_attr_names: tuple = (),
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: dict = None,
        batch_size: int = 128,
        epochs: int = 100,
        printer: Printer = Printer(loss=lambda l, r, b: l.item()),
        evaluator: Callable[[nn.Module], None] = None,
        **kwargs,
):
    """"""
    loader = loader_class(dataset, batch_size=batch_size)

    model = model.to(device)
    optimizer = optimizer(model.parameters(), **optimizer_kwargs)

    for i in range(epochs):
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            res = model(batch, *input_attr_names)
            loss = loss_func(res, batch)
            loss.backward()
            optimizer.step()

            if printer:
                printer(loss, res, batch)

            if evaluator:
                evaluator(model)
