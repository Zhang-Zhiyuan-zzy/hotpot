import copy
import os
import re
import sys
import random
import shutil
import os.path as osp
import datetime
from glob import glob
import socket
import typing
from typing import Callable, Union, Sequence, Optional, Protocol, Type

from torch_geometric.datasets.qm9 import atomrefs
from tqdm import tqdm
from operator import attrgetter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn import metrics

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Optimizer, Adam

from torch_geometric.loader import DataLoader

from torch_geometric.data import Batch

from hotpot.plugins.complex_model import models as M


# ###########################################################################
def get_x_input_attrs(*inputs, input_x_index: Union[list, torch.Tensor]):
    x = inputs[0][:, input_x_index]
    return (x,) + inputs[1:]

def x_masker_func(inputs: tuple, masked_vec: torch.Tensor):
    masked_x, atom_label, masked_node_idx = M.get_masked_input_and_labels(inputs[0], masked_vec, inputs[0][:, 0].long())
    return (masked_x,) + inputs[1:], masked_node_idx

# ###########################################################################

class FeatureExtractorTemplate(typing.Protocol):
    @staticmethod
    def __call__(
            seq: torch.Tensor,
            X_mask: torch.Tensor,
            R_mask: torch.Tensor,
            batch: Batch,
            batch_getter: Callable[[Batch], Union[tuple, torch.Tensor]]=None
    ):
        ...

class ModelProtocol(Protocol):
    def __call__(self, x, edge_index, edge_attr, rings_node_index, rings_node_nums, mol_rings_nums, batch, ptr):
        ...


class Hypers:
    """ A handle of hyperparameters. """
    def __init__(self):
        self.lr = 1e-3
        self.weight_decay = 4e-5
        self.batch_size = 256


class PretrainComplex:
    # Matchers
    dataset_matcher = re.compile(r'get_.+_dataset')
    work_matcher = re.compile(r'run_.+')

    def __init__(
            self,
            work_dir: str,
            dataset_,
            model: ModelProtocol,
            hypers: Union[Hypers, dict],
            optimizer: Optional[Type[Optimizer]] = None,
            not_save: bool = False,
            dataset_test_ = None,
            eval_first: bool = False,
            eval_steps: Optional[int] = 10,
            debug: bool = False,
            device: Union[str, torch.device] = None,
            epochs: int = 100,
            work_name: Optional[str] = None,
            **kwargs
    ):
        """

        Args:
            work_dir:
            dataset_:
            model:
            not_save:
            dataset_test_:
            eval_first:
            eval_steps:
            debug:
        Keyword Args:
            trainset_shuffle: bool
            evalset_shuffle: bool
        """
        self.work_name = work_name

        self.work_dir = work_dir
        self.dataset = dataset_
        self.dataset_test = dataset_test_
        self.model = model
        if isinstance(hypers, dict):
            self.hypers = Hypers
            for k, v in hypers.items():
                setattr(self.hypers, k, v)
        elif isinstance(hypers, Hypers):
            self.hypers = hypers
        else:
            raise TypeError("The argument hypers must be a dict or a Hypers object.")

        self.not_save = not_save
        self.eval_first = eval_first
        self.eval_steps = eval_steps
        self.metrics = {}
        self.debug = debug
        self.epochs = epochs

        if not device:
            self.device = torch.device('cuda') if torch.cuda.is_available() else None
        else:
            self.device = device

        self.OPTIMIZER = optimizer if isinstance(optimizer, Optimizer) else Adam
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.not_save:
            model_dir = self.save_model()

            # Recording the train curve
            df = pd.DataFrame(self.metrics)
            df.set_index('epoch', inplace=True)
            df.to_csv(osp.join(model_dir, 'metrics.csv'))

    def save_model(self):
        now = datetime.datetime.now()
        formatted_datetime = now.strftime("%y%m%d%H%M%S")

        model_dir = osp.join(self.work_dir, f"cp_{formatted_datetime}")
        os.mkdir(model_dir)

        self.model.save_model(model_dir)

        return model_dir

    def train_func(self, which) -> Callable:
        return getattr(self, f"run_{which}")

    @property
    def datasets(self) -> list[str]:
        dataset_names = []
        for name, attr in self.__dict__.items():
            if self.dataset_matcher.match(name) and isinstance(attr, Callable):
                dataset_names.append(name.split('_')[1])

        return dataset_names

    def get_dataset(self, which):
        return getattr(self, f"get_{which}_dataset")()

    def prepare(self):
        loader = DataLoader(self.dataset, batch_size=self.hypers.batch_size, shuffle=self.kwargs.get('trainset_shuffle', True))
        eval_loader = DataLoader(self.dataset_test, batch_size=self.hypers.batch_size, shuffle=self.kwargs.get('evalset_shuffle', False))
        # Clear cache
        torch.cuda.empty_cache()

        model = self.model.to(self.device)
        model.train()
        optimizer = self.OPTIMIZER(model.parameters(), lr=self.hypers.lr, weight_decay=self.hypers.weight_decay)

        return loader, eval_loader, model, optimizer

    @staticmethod
    def get_target(
            batch: Batch,
            target_getter: Callable[[Batch], torch.Tensor],
            masked_idx: Optional[torch.Tensor] = None,
            to_onehot: bool = False,
            onehot_types: int = None,
            loss_weight_calculator: Callable[[torch.Tensor, int], torch.Tensor] = None,
            **kwargs
    ):
        target = target_getter(batch)
        if isinstance(masked_idx, torch.Tensor):
            target = target[masked_idx]

        if to_onehot:
            target = F.one_hot(target.long(), num_classes=onehot_types)  # Convert to OneHot label
        else:
            target = target.view((-1, 1))

        loss_weight = None
        if loss_weight_calculator:
            loss_weight = loss_weight_calculator(target, onehot_types)

        return target, loss_weight

    def forward(
            self,
            model, batch,
            inputs_getter: Callable[[Batch], tuple[Union[torch.Tensor, Sequence], ...]],
            feature_extractor: FeatureExtractorTemplate,
            predictor: Callable[[torch.Tensor], torch.Tensor],
            x_masker: Callable[[tuple[torch.Tensor, ...], torch.Tensor], tuple[torch.Tensor, torch.Tensor]] = None,
            inputs_preprocessor: Callable[[tuple[torch.Tensor, ...], Union[list, torch.Tensor]], tuple[torch.Tensor, ...]] = None,
            input_x_index: Union[list, torch.Tensor] = None,
            extractor_attr_getter: Callable[[Batch], Union[tuple, torch.Tensor]] = None,
            **kwargs
    ):
        batch = batch.to(self.device)
        inputs = inputs_getter(batch)

        if inputs_preprocessor:
            assert isinstance(input_x_index, (list, torch.Tensor))
            inputs = inputs_preprocessor(*inputs, input_x_index=input_x_index)
        if x_masker:
            inputs, masked_idx = x_masker(inputs, model.core.x_mask_vec)
        else:
            masked_idx = None

        # Core model
        seq, X_not_pad, R_not_pad = model(*inputs)

        # Extract features
        feature = feature_extractor(seq, X_not_pad, R_not_pad, batch, extractor_attr_getter)  # Node level feature
        if isinstance(masked_idx, torch.Tensor):
            feature = feature[masked_idx]

        # Prediction
        # predict atom type
        node_pred = predictor(feature)

        return model, node_pred, masked_idx

    @staticmethod
    def batch_dtype_preprocessor(batch):
        for name in batch.keys():
            if "index" in name:
                batch[name] = batch[name].long()
            elif name in ['batch', 'ptr']:
                batch[name] = batch[name].int()
            elif 'nums' in name:
                batch[name] = batch[name].int()
            elif isinstance(batch[name], torch.Tensor) and torch.is_floating_point(batch[name]):
                batch[name] = batch[name].bfloat16()

    def to_train(
            self,
            loader, model, optimizer,
            inputs_getter: Callable[[Batch], tuple[Union[torch.Tensor, Sequence], ...]],
            feature_extractor: FeatureExtractorTemplate,
            predictor: Callable[[torch.Tensor], torch.Tensor],
            target_getter: Callable[[Batch], torch.Tensor],
            loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            inputs_preprocessor: Callable[[tuple[torch.Tensor, ...], Union[list, torch.Tensor]], tuple[torch.Tensor, ...]] = None,
            input_x_index: Union[list, torch.Tensor] = None,
            x_masker: Callable[[tuple[torch.Tensor, ...], torch.Tensor], tuple[torch.Tensor, torch.Tensor]] = None,
            extractor_attr_getter: Callable[[Batch], Union[tuple, torch.Tensor]] = None,
            to_onehot: bool = False,
            onehot_types: int = None,
            loss_weight_calculator: Callable[[torch.Tensor, int], torch.Tensor] = None,
            **kwargs
    ):
        model.train()
        for batch in loader:
            self.batch_dtype_preprocessor(batch)
            model, pred, masked_index = self.forward(
                model, batch,
                inputs_getter=inputs_getter,
                feature_extractor=feature_extractor,
                predictor=predictor,
                inputs_preprocessor=inputs_preprocessor,
                input_x_index=input_x_index,
                x_masker=x_masker,
                extractor_attr_getter=extractor_attr_getter,
                **kwargs
            )

            target, loss_weight = self.get_target(
                batch,
                target_getter=target_getter,
                masked_idx=masked_index,
                to_onehot=to_onehot,
                onehot_types=onehot_types,
                loss_weight_calculator=loss_weight_calculator,
                **kwargs
            )

            # Back propagation
            if isinstance(loss_weight, torch.Tensor):
                loss = loss_fn(pred, target, loss_weight)
            else:
                loss = loss_fn(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if self.debug:
                break

    def to_eval(
            self,
            model, loader,
            metrics: dict[str, Callable[[np.ndarray, np.ndarray], Union[float, np.ndarray]]],
            inputs_getter: Callable[[Batch], tuple[Union[torch.Tensor, Sequence], ...]],
            feature_extractor: FeatureExtractorTemplate,
            node_attr_predictor: Callable[[torch.Tensor], torch.Tensor],
            target_getter: Callable[[Batch], torch.Tensor],
            inputs_preprocessor: Callable[[tuple[torch.Tensor, ...], Union[list, torch.Tensor]], tuple[torch.Tensor, Optional]] = None,
            input_x_index: Union[list, torch.Tensor] = None,
            x_masker: Callable[[tuple[torch.Tensor, ...], torch.Tensor], tuple[torch.Tensor, torch.Tensor]] = None,
            extractor_attr_getter: Callable[[Batch], Union[tuple, torch.Tensor]] = None,
            to_onehot: bool = False,
            onehot_types: int = None,
            print_pred_target_labels: bool = True,
            eval_one_debug: bool = False,
            **kwargs
    ):
        model.eval()

        pred = []
        target = []
        with torch.no_grad():
            for batch in loader:
                self.batch_dtype_preprocessor(batch)
                model, node_pred, masked_index = self.forward(
                    model, batch,
                    inputs_getter=inputs_getter,
                    feature_extractor=feature_extractor,
                    predictor=node_attr_predictor,
                    inputs_preprocessor=inputs_preprocessor,
                    input_x_index=input_x_index,
                    x_masker=x_masker,
                    extractor_attr_getter=extractor_attr_getter,
                )

                node_target, _ = self.get_target(
                    batch,
                    target_getter=target_getter,
                    masked_idx=masked_index,
                    to_onehot=to_onehot,
                    onehot_types=onehot_types,
                )

                pred.append(node_pred.cpu().detach().float().numpy())
                target.append(node_target.cpu().detach().float().numpy())

                if self.debug:
                    break

            pred = np.concatenate(pred)
            target = np.concatenate(target)

            if print_pred_target_labels:
                pred_label, target_label = M.inverse_onehot(to_onehot, pred, target)
                pred_target_label = np.concatenate([pred_label, target_label], axis=1)
                assert pred_target_label.shape == (target.shape[0], 2)
                # print("Sample pred and target:")
                # print(np.random.choice(pred_target_label, print_sample_number, replace=False))

            return {
                metric_name: metric_func(pred, target)
                for metric_name, metric_func in metrics.items()
            }

    def print_eval_metric(self, epoch: int, metric_results):
        list_epoch = self.metrics.setdefault('epoch', [])
        list_epoch.append(epoch)
        for metric_name, metric_value in metric_results.items():
            print(f'Eval {metric_name} in eval set {self.work_name}: {metric_value}')
            list_metric = self.metrics.setdefault(metric_name, [])
            list_metric.append(metric_value)


    def train_eval(
            self,
            feature_extractor: FeatureExtractorTemplate,
            predictor: Callable,
            target_getter: Callable[[Batch], torch.Tensor],
            loss_fn: Callable[[tuple[torch.Tensor, torch.Tensor], torch.Tensor], torch.Tensor],
            input_x_index: Union[list, torch.Tensor] = None,
            x_masker: Callable[[tuple[torch.Tensor, ...], torch.Tensor], tuple[torch.Tensor, torch.Tensor]] = None,
            extractor_attr_getter: Callable[[Batch], Union[tuple, torch.Tensor]] = None,
            to_onehot: bool = True,
            onehot_types: int = None,
            loss_weight_calculator: Callable[[torch.Tensor, int], torch.Tensor] = None,
            metrics: dict[str, Callable[[np.ndarray, np.ndarray], Union[float, np.ndarray]]] = None,
            **kwargs
    ):
        if to_onehot and not isinstance(onehot_types, int):
            raise ValueError("The type of onehot should be explicitly specified.")

        # Initializing Dataloader, model and optimizer
        inputs_getter = attrgetter(
            'x', 'edge_index', 'edge_attr', 'rings_node_index',
            'rings_node_nums', 'mol_rings_nums', 'batch', 'ptr')
        loader, eval_loader, model, optimizer = self.prepare()

        # Preparing arguments
        train_kw = dict(
            loader=loader,
            model=model,
            optimizer=optimizer,
            inputs_getter=inputs_getter,
            feature_extractor=feature_extractor,
            predictor=predictor,
            target_getter=target_getter,
            loss_fn=loss_fn,
            inputs_preprocessor=get_x_input_attrs,
            input_x_index=input_x_index,
            x_masker=x_masker,
            extractor_attr_getter=extractor_attr_getter,
            to_onehot=to_onehot,
            onehot_types=onehot_types,
            loss_weight_calculator=loss_weight_calculator,
            **kwargs
        )

        eval_kw = dict(
            loader=eval_loader,
            model=model,
            metrics=metrics,
            inputs_getter=inputs_getter,
            feature_extractor=feature_extractor,
            node_attr_predictor=predictor,
            target_getter=target_getter,
            inputs_preprocessor=get_x_input_attrs,
            input_x_index=input_x_index,
            extractor_attr_getter=extractor_attr_getter,
            to_onehot=to_onehot,
            onehot_types=onehot_types,
            **kwargs
        )

        def _print_eval_result():
            ev_kw = copy.copy(eval_kw)
            ev_kw["loader"] = DataLoader(self.dataset_test, batch_size=self.hypers.batch_size, shuffle=True)
            metric_results = self.to_eval(**_ev_kw)
            self.print_eval_metric(-1, metric_results)

        # Training and evaluation
        if self.eval_first:
            _ev_kw = copy.copy(eval_kw)
            _ev_kw["loader"] = DataLoader(self.dataset_test, batch_size=self.hypers.batch_size, shuffle=True)
            metric_results = self.to_eval(**_ev_kw)
            self.print_eval_metric(-1, metric_results)
            del _ev_kw

        for epoch in range(self.epochs):
            self.to_train(**train_kw)
            if isinstance(self.eval_steps, int) and epoch % self.eval_steps == 0:
                metric_results = self.to_eval(**eval_kw)
                self.print_eval_metric(epoch, metric_results)

    def run(self, *args, **kwargs):
        self.train_eval(*args, **kwargs)
