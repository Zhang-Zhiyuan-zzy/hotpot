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
from typing import Callable, Union, Sequence, Optional

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
from torch_geometric.loader import DataLoader

from torch_geometric.data import Batch

from hotpot.plugins.complex_model.data import tmQmDataset

add_dir = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
sys.path.append(add_dir)


import hotpot as hp
from hotpot.cheminfo.elements import elements
from hotpot.plugins.dl.pytorch_func import loss_organizer
from hotpot.plugins.complex_model import data as pyg_data, plots
from hotpot.plugins.complex_model import train, models as M
from datasets import DatasetGetter


def sk_r2(pred, target):
    return metrics.r2_score(target, pred)


# Types definition
class TensorFunc(typing.Protocol):
    def __call__(self, tensor: torch.Tensor, *args: typing.Any) -> torch.Tensor:
        ...

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


# Initialize paths.
machine_name = socket.gethostname()
if machine_name == '4090':
    project_root = '/home/zzy/docker_envs/pretrain/proj'
elif machine_name == 'DESKTOP-G9D9UUB':
    project_root = '/mnt/d/zhang/OneDrive/Papers/BayesDesign/results'
elif machine_name == 'docker':
    project_root = '/app/proj'
else:
    raise ValueError

models_dir = osp.join(project_root, 'models')

# dataset save paths
_tmqm_data_dir = osp.join(project_root, 'datasets', 'tmqm_data0207')


torch.set_default_dtype(torch.bfloat16)
def to_bfloat16(tensor: torch.Tensor) -> torch.Tensor:
    if torch.is_floating_point(tensor):
        return tensor.bfloat16()
    else:
        return tensor

if torch.cuda.is_available():
    device = torch.device("cuda:1")
else:
    device = torch.device("cpu")


def count_labels(labels, is_onehot:bool=False, print_result: bool = True):
    if is_onehot:
        labels = torch.argmax(labels, dim=1)

    uni, count = torch.unique(labels, return_counts=True)
    if print_result:
        for u, c in zip(uni, count):
            print(f"{u}: {c}")

    return uni, count


def move_random_files(source_dir, target_dir, num_files=10000):
    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Get all .pt files in the source directory
    pt_files = glob(os.path.join(source_dir, "*.pt"))

    # Check if there are enough files to move
    if len(pt_files) < num_files:
        raise ValueError(f"Not enough .pt files in the source directory. Found {len(pt_files)}, but need {num_files}.")

    # Randomly select N files
    selected_files = random.sample(pt_files, num_files)

    # Move the selected files to the target directory
    for file in selected_files:
        shutil.move(file, target_dir)
        print(f"Moved: {file} -> {target_dir}")


def restore_tmqm_data():
    source_dir = osp.join(project_root, 'datasets', 'tmqm_data0207')
    target_dir = osp.join(project_root, 'datasets', 'tmqm_data0207', 'test_data')

    move_random_files(source_dir, target_dir)

def binary_accuracy(pred: torch.Tensor, target: torch.Tensor):
    pred_results = (pred >= 0.5).int()
    return (pred_results == target).float().mean()


def get_x_input_attrs(*inputs):
    x = inputs[0][:, INPUT_X_INDEX]
    return (x,) + inputs[1:]

def x_masker_func(inputs: tuple, masked_vec: torch.Tensor):
    masked_x, atom_label, masked_node_idx = M.get_masked_input_and_labels(inputs[0], masked_vec, inputs[0][:, 0].long())
    return (masked_x,) + inputs[1:], masked_node_idx

def r2_score(y_pred, y_true):
    # Compute residual sum of squares
    ss_res = torch.sum((y_true - y_pred) ** 2)

    # Compute total sum of squares
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)

    # Return the R² score
    return 1 - ss_res / ss_tot


def get_atom_type_weight(scale=10., atom_types: int = 119):
    path_tmqm_type_weight = osp.join(project_root, 'datasets', 'tmqm_node_type_weight.csv')
    df_weight = pd.read_csv(path_tmqm_type_weight, index_col=0)
    weight_values = df_weight.values.flatten()
    atom_type = df_weight.index

    weight = torch.zeros(atom_types)
    for t, w in zip(atom_type, weight_values):
        weight[t] = w

    return scale * weight

def loss_func(res, batch):
    """"""
    print(res)


class PairIndexIter:
    def __init__(
            self,
            pair_index,
            batch_size=32*1024
    ):
        self.pair_index = pair_index
        self.batch_size = batch_size
        self.size = pair_index.shape[-1]

        self.start = 0

    def __iter__(self):
        while self.start < self.size-1:
            yield (self.start, self.start + self.batch_size), self.pair_index[:, self.start:self.start + self.batch_size]
            self.start += self.batch_size


class PretrainData:
    _tmqm_data_dir = osp.join(project_root, 'datasets', 'tmqm_data0207')
    _tmqm_test_dir = osp.join(project_root, 'datasets', 'tmqm_data0207', 'test_data')

    def __init__(self, name):
        self.name = name

    @classmethod
    def get_tmqm_data(cls):
        ds = pyg_data.tmQmDataset(cls._tmqm_data_dir)
        ds_test = pyg_data.tmQmDataset(cls._tmqm_test_dir)

        input_x_name = ('atomic_number', 'n', 's', 'p', 'd', 'f', 'g', 'x', 'y', 'z')
        input_x_index = [ds[0].x_names.index(n) for n in input_x_name]

        atom_type_index = 0

        atom_charge_name = 'partial_charge'
        atom_charge_index = ds[0].x_names.index('partial_charge')

        atom_aromatic_name = 'is_aromatic'
        atom_aromatic_index = ds[0].x_names.index('is_aromatic')

        rings_attr_names = ('is_aromatic',)
        rings_attr_index = 0

        pair_attr_names = ds[0].pair_attr_names
        pair_step_index = pair_attr_names.index('length_shortest_path')
        pair_wbo_index = pair_attr_names.index('wiberg_bond_order')

        y_attr_names = ds[0].y_names
        y_attr_index = True

        return (
            ds, ds_test,
            input_x_name, input_x_index,
            atom_type_index,
            atom_charge_name, atom_charge_index,
            atom_aromatic_name, atom_aromatic_index,
            rings_attr_names, rings_attr_index,
            pair_attr_names, pair_step_index, pair_wbo_index,
            y_attr_names, y_attr_index,
        )

# (  # Specify which data apply to train
#     dataset, dataset_test,
#     INPUT_X_NAME, INPUT_X_INDEX,
#     TYPE_INDEX,
#     ATOM_CHRG_NAME, ATOM_CHRG_INDEX,
#     ATOM_AROMATIC_NAME, ATOM_AROMATIC_INDEX,
#     RINGS_ATTR_NAMES, RING_AROMATIC_INDEX,
#     PAIR_ATTR_NAMES, PAIR_STEP_INDEX, PAIR_WBO_INDEX,
#     Y_ATTR_NAMES, Y_ATTR_INDEX,
# ) = PretrainData.get_tmqm_data()


tmqm_getter = DatasetGetter(project_root, "tmqm")

dataset, dataset_test = tmqm_getter.get_datasets()
INPUT_X_INDEX = tmqm_getter.get_index('x', ('atomic_number', 'n', 's', 'p', 'd', 'f', 'g', 'x', 'y', 'z'))
TYPE_INDEX = tmqm_getter.get_index('x', 'atomic_number')
ATOM_CHRG_INDEX = tmqm_getter.get_index('x', 'partial_charge')
ATOM_AROMATIC_INDEX = tmqm_getter.get_index('x', 'is_aromatic')
RING_AROMATIC_INDEX = tmqm_getter.get_index('ring_attr', 'is_aromatic')
PAIR_STEP_INDEX = tmqm_getter.get_index('pair_attr', 'length_shortest_path')
PAIR_WBO_INDEX = tmqm_getter.get_index('pair_attr', 'wiberg_bond_order')
Y_ATTR_NAMES = tmqm_getter.get_y_attrs()



# Hyperparameters
BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 5e-3
WEIGHT_DECAY = 4e-5
OPTIMIZER = torch.optim.Adam

X_DIM = len(INPUT_X_INDEX)
EDGE_DIM = dataset[0].edge_attr.shape[-1]
VEC_DIM = 64
MASK_VEC = (-1 * torch.ones(X_DIM)).to(device)

ATOM_TYPES = 119  # Arguments for atom type loss

_batch_getter = attrgetter(
    'x', 'x_names',
    'edge_index', 'edge_attr', 'edge_attr_names',
    'pair_index', 'pair_attr', 'pair_attr_name',
    'rings_node_index', 'rings_node_nums', 'mol_rings_nums', 'mol_rings_node_nums', 'rings_attr',  # rings_attr_name
    'y', 'y_names',
    'batch', 'ptr',
    'identifier'
)

def load_trained_model(model_name, shuffle_data: bool = False):
    model_path = osp.join(models_dir, model_name)
    model = torch.load(model_path)

    loader = iter(DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle_data))

    return model, next(loader)

def calc_oh_accuracy(pred_oh, target_label):
    pred_label = torch.argmax(pred_oh, dim=1)
    return (pred_label == target_label).float().mean()

def get_pred_target_node_type(model_name, shuffle_data: bool = False):
    model, batch = load_trained_model(model_name, shuffle_data)
    model = model.to(device)
    batch = batch.to(device)

    (x, x_names,
     edge_index, edge_attr, edge_attr_names,
     pair_index, pair_attr, pair_attr_name,
     rings_node_index, rings_node_nums, mol_rings_nums, mol_rings_node_nums, rings_attr,  # rings_attr_name
     y, y_names,
     batch, ptr,
     identifier) = _batch_getter(batch)

    masked_X, atom_label, masked_node_idx = (
        M.get_masked_input_and_labels(x[:, INPUT_X_INDEX], MASK_VEC, x[:, 0].long()))
    atom_types_oh = F.one_hot(atom_label, num_classes=ATOM_TYPES)  # Convert to OneHot label
    type_weight_oh = M.atom_label_weight(atom_label, num_types=ATOM_TYPES)

    # Core model
    seq, X_not_pad, R_not_pad = model(masked_X, edge_index, edge_attr, rings_node_index, rings_node_nums,
                                      mol_rings_nums, batch, ptr)

    # Extract features
    Znode = model.extract_atom_vec(seq, X_not_pad)  # Node level feature

    # Prediction
    # predict atom type
    pred_atom_type = model.predict_atom_type(Znode)


    return model, batch, pred_atom_type[masked_node_idx], atom_types_oh


class PretrainComplex:
    # Matchers
    dataset_matcher = re.compile(r'get_.+_dataset')
    work_matcher = re.compile(r'run_.+')

    def __init__(
            self,
            work_dir: str,
            dataset_,
            model: nn.Module = None,
            work_name: str = 'atom_type',
            not_save: bool = False,
            dataset_test_ = None,
            eval_first: bool = False,
            eval_steps: Optional[int] = 10,
            debug: bool = False,
            **kwargs
    ):
        """

        Args:
            work_dir:
            dataset_:
            model:
            work_name:
            not_save:
            dataset_test_:
            eval_first:
            eval_steps:
            debug:
        Keyword Args:
            trainset_shuffle: bool
            evalset_shuffle: bool
        """
        self.work_dir = work_dir
        self.dataset = dataset_
        self.dataset_test = dataset_test_

        self.work_name = work_name
        self.not_save = not_save
        self.eval_first = eval_first
        self.eval_steps = eval_steps
        self.metrics = {}
        self.debug = debug

        self.kwargs = kwargs

        if model:
            self.model = model
        else:
            self.model = M.ComplexFormer(
                x_dim=X_DIM,
                edge_dim=EDGE_DIM,
                vec_dim=VEC_DIM,
            )

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
    def work_names(self) -> list[str]:
        work_names = []
        for name, attr in self.__dict__.items():
            if self.work_matcher.match(name) and isinstance(attr, Callable):
                work_names.append(name)

        return work_names

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
        loader = DataLoader(self.dataset, batch_size=BATCH_SIZE, shuffle=self.kwargs.get('trainset_shuffle', True))
        eval_loader = DataLoader(self.dataset_test, batch_size=BATCH_SIZE, shuffle=self.kwargs.get('evalset_shuffle', False))
        # Clear cache
        torch.cuda.empty_cache()

        model = self.model.to(device)
        model.train()
        optimizer = OPTIMIZER(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

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

    @staticmethod
    def forward(
            model, batch,
            inputs_getter: Callable[[Batch], tuple[Union[torch.Tensor, Sequence], ...]],
            feature_extractor: FeatureExtractorTemplate,
            predictor: Callable[[torch.Tensor], torch.Tensor],
            x_masker: Callable[[tuple[torch.Tensor, ...], torch.Tensor], tuple[torch.Tensor, torch.Tensor]] = None,
            inputs_preprocessor: Callable[[tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]] = None,
            extractor_attr_getter: Callable[[Batch], Union[tuple, torch.Tensor]] = None,
            **kwargs
    ):
        batch = batch.to(device)
        inputs = inputs_getter(batch)

        if inputs_preprocessor:
            inputs = inputs_preprocessor(*inputs)
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
            inputs_preprocessor: Callable[[tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]] = None,
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
            inputs_preprocessor: Callable[[tuple[torch.Tensor, ...]], tuple[torch.Tensor, Optional]] = None,
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
            predictor_name: str,
            target_getter: Callable[[Batch], torch.Tensor],
            loss_fn: Callable[[tuple[torch.Tensor, torch.Tensor], torch.Tensor], torch.Tensor],
            x_masker: Callable[[tuple[torch.Tensor, ...], torch.Tensor], tuple[torch.Tensor, torch.Tensor]] = None,
            extractor_attr_getter: Callable[[Batch], Union[tuple, torch.Tensor]] = None,
            to_onehot: bool = True,
            onehot_types: int = ATOM_TYPES,
            loss_weight_calculator: Callable[[torch.Tensor, int], torch.Tensor] = None,
            metrics: dict[str, Callable[[np.ndarray, np.ndarray], Union[float, np.ndarray]]] = None,
            **kwargs
    ):
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
            predictor=getattr(model, predictor_name),
            target_getter=target_getter,
            loss_fn=loss_fn,
            inputs_preprocessor=get_x_input_attrs,
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
            node_attr_predictor=getattr(model, predictor_name),
            target_getter=target_getter,
            inputs_preprocessor=get_x_input_attrs,
            extractor_attr_getter=extractor_attr_getter,
            to_onehot=to_onehot,
            onehot_types=onehot_types,
            **kwargs
        )

        def _print_eval_result():
            ev_kw = copy.copy(eval_kw)
            ev_kw["loader"] = DataLoader(self.dataset_test, batch_size=BATCH_SIZE, shuffle=True)
            metric_results = self.to_eval(**_ev_kw)
            self.print_eval_metric(-1, metric_results)

        # Training and evaluation
        if self.eval_first:
            _ev_kw = copy.copy(eval_kw)
            _ev_kw["loader"] = DataLoader(self.dataset_test, batch_size=BATCH_SIZE, shuffle=True)
            metric_results = self.to_eval(**_ev_kw)
            self.print_eval_metric(-1, metric_results)
            del _ev_kw

        for epoch in range(EPOCHS):
            self.to_train(**train_kw)
            if isinstance(self.eval_steps, int) and epoch % self.eval_steps == 0:
                metric_results = self.to_eval(**eval_kw)
                self.print_eval_metric(epoch, metric_results)

    def run_atom_types(self):
        self.train_eval(
            feature_extractor=M.FeatureExtractors.extract_atom_vec,
            predictor_name='predict_atom_type',
            target_getter=lambda batch: batch.x[:, TYPE_INDEX],
            x_masker=x_masker_func,
            loss_fn=M.LossMethods.calc_atom_type_loss,
            to_onehot=True,
            onehot_types=ATOM_TYPES,
            loss_weight_calculator=lambda t, n: M.atom_label_weight_(t, n, 'inverse-count'),
            metrics={'Accuracy': lambda p, t: M.Metrics.calc_oh_accuracy(p, t, is_onehot=True)}
        )

    def run_atom_charge(self):
        self.train_eval(
            feature_extractor=M.FeatureExtractors.extract_atom_vec,
            predictor_name='predict_atom_charge',
            target_getter=lambda batch: batch.x[:, ATOM_CHRG_INDEX],
            loss_fn=F.mse_loss,
            to_onehot=False,
            metrics={'R^2': sk_r2}
        )

    def run_atom_aromatic(self):
        self.train_eval(
            feature_extractor=M.FeatureExtractors.extract_atom_vec,
            predictor_name='predict_atom_aromatic',
            target_getter=lambda batch: batch.x[:, ATOM_AROMATIC_INDEX],
            loss_fn=F.binary_cross_entropy,
            to_onehot=False,
            metrics={'Binary Accuracy': M.Metrics.binary_accuracy}
        )

    def run_pair_step(self):
        self.train_eval(
            feature_extractor=M.FeatureExtractors.extract_pair_vec,
            extractor_attr_getter=attrgetter('pair_index'),
            predictor_name='predict_pair_step',
            target_getter=lambda batch: batch.pair_attr[:, PAIR_STEP_INDEX],
            loss_fn=F.mse_loss,
            to_onehot=False,
            metrics={'R^2': sk_r2}
        )

    def run_cbond(self):
        self.train_eval(
            feature_extractor=M.FeatureExtractors.extract_cbond_pair,
            predictor_name='predict_atom_aromatic',
            target_getter=lambda batch: batch.is_cbond,
            loss_fn=F.binary_cross_entropy,
            to_onehot=False,
            metrics={'Binary Accuracy': M.Metrics.binary_accuracy}
        )

    def run_rings_aromatic(self):
        self.train_eval(
            feature_extractor=M.FeatureExtractors.extract_ring_vec,
            predictor_name='predict_rings_aromatic',
            target_getter=lambda batch: batch.ring_attr[:, RING_AROMATIC_INDEX],
            loss_fn=F.mse_loss,
            to_onehot=False,
            metrics={'R^2': sk_r2}
        )

    def run_pair_step_(self):
        batch_getter = attrgetter(
            'x', 'edge_index', 'edge_attr', 'rings_node_index',
            'pair_index', 'pair_attr',
            'rings_node_nums', 'mol_rings_nums', 'batch', 'ptr')

        loader, model, optimizer = self.prepare()
        model.pair_step_predictor = model.pair_step_predictor.to('cuda:1')
        model.pair_step_linear = model.pair_step_linear.to('cuda:1')

        for i in range(EPOCHS):
            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad()

                (
                    x, edge_index, edge_attr, rings_node_index,
                    pair_index, pair_attr,
                    rings_node_nums, mol_rings_nums, batch, ptr
                ) = batch_getter(batch)
                x, edge_attr, pair_attr, = map(to_bfloat16, [x, edge_attr, pair_attr])

                # Core model
                seq, X_not_pad, R_not_pad = model(x[:, INPUT_X_INDEX], edge_index, edge_attr, rings_node_index,
                                                  rings_node_nums, mol_rings_nums, batch, ptr)

                # Extract features
                Znode = model.extract_atom_vec(seq, X_not_pad)  # Node level feature

                pred_steps = []
                losses = []
                for i, ((start, end), batch_pair_index) in enumerate(PairIndexIter(pair_index)):
                    Zpair = model.extract_pair_vec(Znode, batch_pair_index)

                    pred_pair_step = model.predict_pair_step(Zpair).to('cuda:1')

                    # Extract pair step
                    target_pair_step = pair_attr[start:end, PAIR_STEP_INDEX].to('cuda:1')

                    mse_loss = F.mse_loss(pred_pair_step.flatten(), target_pair_step)

                    # Back propagation
                    mse_loss.backward(retain_graph=True)

                    # Save pred_result and clone to cpu
                    losses.append(mse_loss.item())
                    pred_steps.append(pred_pair_step.cpu())

                optimizer.step()

                pred_steps = torch.cat(pred_steps, dim=0)
                print(f"MSE loss: {np.mean(losses)}, R^2 loss: {r2_score(pred_steps.flatten(),pair_attr[:, PAIR_STEP_INDEX].cpu()).item()}")
                pred_steps = []

    def run_ring_aromatic(self):
        # TODO: aromatic might be all zero!
        batch_getter = attrgetter(
            'x', 'edge_index', 'edge_attr', 'rings_node_index', 'rings_attr',
            'rings_node_nums', 'mol_rings_nums', 'batch', 'ptr')

        loader, model, optimizer = self.prepare()

        for i in range(EPOCHS):
            for batch in loader:
                batch = batch.to(device)

                optimizer.zero_grad()

                (
                    x, edge_index, edge_attr, rings_node_index, rings_attr,
                    rings_node_nums, mol_rings_nums, batch, ptr
                ) = batch_getter(batch)
                x, edge_attr, rings_attr = map(to_bfloat16, [x, edge_attr, rings_attr])

                # Extractor target charge
                target_aromatic = rings_attr[:, RING_AROMATIC_INDEX]

                # Core model
                seq, X_not_pad, R_not_pad = model(x[:, INPUT_X_INDEX], edge_index, edge_attr, rings_node_index,
                                                  rings_node_nums, mol_rings_nums, batch, ptr)

                # Extract features
                Zring = model.extract_ring_vec(seq, R_not_pad)  # Node level feature

                pred_aromatic = model.predict_rings_aromatic(Zring).flatten()

                bce_loss = F.binary_cross_entropy(pred_aromatic, target_aromatic)

                # Back propagation
                bce_loss.backward()
                optimizer.step()

                print(f"BCE loss: {bce_loss.item()}, Binary Accuracy: {binary_accuracy(pred_aromatic, target_aromatic).item()}")


    def run(self):
        train_func = getattr(self, f'run_{self.work_name}')
        train_func()

    def run_mol_attrs(self):
        batch_getter = attrgetter(
            'x', 'edge_index', 'edge_attr', 'rings_node_index', 'y',
            'rings_node_nums', 'mol_rings_nums', 'batch', 'ptr')

        loader, model, optimizer = self.prepare()
        model.mol_attr_predictors = {
            n: M.MolAttrPredictor(VEC_DIM, 3).to(device) for n in Y_ATTR_NAMES
        }

        for i in range(EPOCHS):
            for batch in loader:
                batch = batch.to(device)

                optimizer.zero_grad()

                (
                    x, edge_index, edge_attr, rings_node_index, y,
                    rings_node_nums, mol_rings_nums, batch, ptr
                ) = batch_getter(batch)
                x, edge_attr, y = map(to_bfloat16, [x, edge_attr, y])

                # Core model
                seq, X_not_pad, R_not_pad = model(x[:, INPUT_X_INDEX], edge_index, edge_attr, rings_node_index,
                                                  rings_node_nums, mol_rings_nums, batch, ptr)

                # Extract features
                mol_vec = model.extract_mol_vec(seq)  # Node level feature

                # Forward for each mol attrs:
                for i, name in enumerate(Y_ATTR_NAMES):
                    target_y = y[i]

                    pred_y = model.mol_attr_predictors[name](mol_vec)

                    mse_loss = F.mse_loss(pred_y, target_y)

                    # Back propagation
                    mse_loss.backward()
                    optimizer.step()

                    print(f"MSE loss: {mse_loss.item()}, R^2 loss: {r2_score(pred_y, target_y).item()}")


def coord_disturb_recover(model):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


if __name__ == '__main__':

    # mod = M.ComplexFormer(
    #     x_dim=X_DIM,
    #     edge_dim=EDGE_DIM,
    #     vec_dim=VEC_DIM,
    # )

    # # training for atom type
    # with train.Trainer('/home/zzy/proj/bayes/models', mod, pretrain_atom_type) as trainer:
    #     trainer.train()

    # training for atom charge
    # with train.Trainer('/home/zzy/proj/bayes/models', mod, pretrain_atom_charge) as trainer:
    #     trainer.train()

    # m, b, p, t = get_pred_target_node_type('model_250209072654.pt')


    with PretrainComplex(
        not_save=True,
        work_dir=models_dir,
        dataset_=dataset,
        dataset_test_=dataset_test,
        eval_steps=1,
        # work_name='atom_charge',
        work_name='atom_types',
        # work_name='pair_step',
        # work_name='atom_aromatic'
        # work_name='ring_aromatic',
        # work_name='mol_attrs'
        eval_first=True,
        # debug=True,
    ) as pretrain:
        pretrain.run()
