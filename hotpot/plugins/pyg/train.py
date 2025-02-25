"""
python v3.9.0
@Project: hotpot
@File   : train
@Auther : Zhiyuan Zhang
@Data   : 2025/1/13
@Time   : 16:40
"""
import os
from glob import glob
import os.path as osp
from os import PathLike
from typing import Iterable, Type, Callable, Any, Union, Literal
import datetime

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


class Trainer(object):
    def __init__(
            self,
            work_dir: Union[PathLike, str],
            model: nn.Module,
            train_func: Callable[[nn.Module], None],
            not_save: bool = False,
    ):
        self.model = model
        self.train_func = train_func

        if osp.exists(work_dir):
            assert osp.isdir(work_dir)
        else:
            if not osp.exists(osp.dirname(work_dir)):
                raise NotADirectoryError(f'The parent directory of {work_dir} does not exist')
            os.mkdir(work_dir)
        self.work_dir = osp.abspath(work_dir)
        self.not_save = not_save

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.not_save:
            self.save_model()

    @staticmethod
    def load_last_model(work_dir: Union[PathLike, str], which: Literal['model', 'state_dict']='model'):
        last_datetime = max(int(osp.basename(p).split('_')[-1]) for p in glob(osp.join(work_dir, 'cp_*')))
        model_dir = osp.join(work_dir, f'cp_{last_datetime}')

        if which == 'model':
            return torch.load(osp.join(model_dir, 'model.pt'))
        elif which == 'state_dict':
            return torch.load(osp.join(model_dir, 'state_dict.pt'))

    def train(self):
        self.train_func(self.model)

    def save_model(self):
        now = datetime.datetime.now()
        formatted_datetime = now.strftime("%y%m%d%H%M%S")

        model_dir = osp.join(self.work_dir, f"cp_{formatted_datetime}")
        os.mkdir(model_dir)

        torch.save(self.model.state_dict(), osp.join(model_dir, f'state_dict.pt'))
        torch.save(self.model, osp.join(self.work_dir, f'model.pt'))



