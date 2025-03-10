import os
import re
import os.path as osp
import datetime
import typing
from typing import Callable, Union, Sequence, Optional, Any, Type, Literal, Iterable

from operator import attrgetter

from tqdm import tqdm
import pandas as pd
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Optimizer, Adam
import torch.optim.lr_scheduler as lrs
from torch_geometric.graphgym.register import train_dict

from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

from hotpot.plugins.complex_model import models as M


######################## Utils ######################################
def torch_numpy_exchanger(nf: Callable, **kw):
    # torch-numpy exchanger
    def wrapper(*inputs: Union[torch.Tensor, np.ndarray]):
        return nf(*inputs, **kw)

    return wrapper


# ###########################################################################
def get_xyz(*inputs, xyz_index: Union[int, torch.Tensor]) -> torch.Tensor:
    return inputs[0][:, xyz_index]

def get_x_input_attrs(*inputs, input_x_index: Union[list, torch.Tensor]):
    x = inputs[0][:, input_x_index]
    return (x,) + inputs[1:]

def get_labeled_x_input_attrs(*inputs, input_x_index: Union[list, torch.Tensor]):
    return (inputs[0][:, 0],) + inputs[1:]

def x_masker_func(inputs: tuple, masked_vec: torch.Tensor):
    x = inputs[0]
    if x.dim == 2:
        masked_x, atom_label, masked_node_idx = M.get_masked_input_and_labels(inputs[0], masked_vec, x[:, 0].long())
    else:
        masked_x, atom_label, masked_node_idx = M.get_masked_input_and_labels(inputs[0], masked_vec, x.long(), label_mask=True)

    return (masked_x,) + inputs[1:], masked_node_idx

def remove_cbond_edges(batch: Batch):
    """ Remove the cbond edges for predict """
    edge_index = batch.edge_index
    edge_attr = batch.edge_attr if hasattr(batch, 'edge_attr') else None
    is_cbond: torch.Tensor = getattr(batch, 'is_cbond')

    cbond_indices = torch.nonzero(is_cbond == 1).squeeze()

    # if cbond_indices is not empty，remove the edge in the edge_index
    if len(cbond_indices) > 0:
        # 创建一个mask来标记不需要删除的边
        mask = torch.ones(edge_index.size(1), dtype=torch.bool)
        mask[cbond_indices] = False  # 将要删除的边标记为False

        edge_index = edge_index[:, mask]

        if edge_attr is not None:
            edge_attr = edge_attr[mask]

        batch.edge_index = edge_index
        batch.edge_attr = edge_attr

    return batch

# ###########################################################################
############################## Loss Func ####################################
def mean_maximum_displacement(
        pred: Union[torch.Tensor, np.ndarray],
        target: Union[torch.Tensor, np.ndarray],
        *args, **kwargs
) -> Union[torch.Tensor, np.ndarray, float]:
    if isinstance(target, torch.Tensor):
        norm = torch_numpy_exchanger(torch.norm, dim=-1)
    elif isinstance(target, np.ndarray):
        norm = torch_numpy_exchanger(np.linalg.norm, axis=-1)
    else:
        raise TypeError("The target and pred data should be torch.Tensor or np.ndarray")

    return norm(pred - target).mean()

class FeatureExtractorTemplate(typing.Protocol):
    @staticmethod
    def __call__(
            seq: torch.Tensor,
            X_mask: torch.Tensor,
            R_mask: torch.Tensor,
            batch: Batch,
            batch_getter: Callable[[Batch], Union[tuple, torch.Tensor]]=None
    ) -> torch.Tensor:
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

    def  __init__(
            self,
            work_dir: str,
            train_dataset,
            core_model: nn.Module,
            predictor: nn.Module,
            hypers: Union[Hypers, dict],
            feature_extractor: FeatureExtractorTemplate,
            target_getter: Callable[[Batch], torch.Tensor],
            loss_fn: Callable[[torch.Tensor, torch.Tensor, Optional[Any]], torch.Tensor],
            metrics: dict[str, Callable[[np.ndarray, np.ndarray], Union[float, np.ndarray]]],
            optimizer: Optional[Type[Optimizer]] = None,
            constant_lr: bool = False,
            lr_scheduler: Optional[Type[torch.optim.lr_scheduler.LRScheduler]] = None,
            lr_scheduler_kwargs: Optional[dict] = None,
            has_xyz: bool = False,
            not_save: bool = False,
            save_max_acc_state: bool = True,
            test_dataset = None,
            eval_first: bool = False,
            eval_steps: Optional[int] = 1,
            debug: bool = False,
            device: Union[str, torch.device] = None,
            epochs: int = 100,
            work_name: Optional[str] = None,
            primary_metric: str = "accuracy",
            minimize_metric: bool = False,
            early_stopping: bool = False,
            early_stop_step: int = 5,
            load_all_data: bool = False,
            show_batch_pbar: bool = False,
            **kwargs
    ):
        """

        Args:
            work_dir:
            train_dataset:
            core_model:
            not_save:
            test_dataset:
            eval_first:
            eval_steps:
            debug:
        Keyword Args:
            trainset_shuffle: bool
            evalset_shuffle: bool
        """
        self.work_name = work_name
        self.has_xyz = has_xyz

        self.work_dir = work_dir
        self.train_dataset = train_dataset
        self.dataset_test = test_dataset
        self.load_all_data = load_all_data
        self.core_model = core_model
        self.predictor = predictor
        self.feature_extractor = feature_extractor
        self.target_getter = target_getter
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.metrics_results = {}
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
        self.debug = debug
        self.epochs = epochs
        self.lazy_eval = None

        if not device:
            self.device = torch.device('cuda') if torch.cuda.is_available() else None
        else:
            self.device = device

        # Optimizer control arguments
        self.OPTIMIZER = optimizer if isinstance(optimizer, Optimizer) else Adam
        if not constant_lr:
            if lr_scheduler:
                self.lr_scheduler = lr_scheduler
                default_lrs_kwargs = {}
            else:
                self.lr_scheduler = lrs.ExponentialLR
                default_lrs_kwargs = {'gamma': 0.95}

            default_lrs_kwargs.update(lr_scheduler_kwargs if lr_scheduler_kwargs else {})
            self.lrs_kwargs = default_lrs_kwargs
        else:
            self.lr_scheduler = None
            self.lrs_kwargs = None

        self.kwargs = kwargs

        self.model_dir = self._init_model_dir()
        self.model_name = osp.basename(self.model_dir)
        self.primary_metric = primary_metric
        self.best_primary_metric = None
        self.save_max_acc_state = save_max_acc_state
        self.minimize_metric = minimize_metric

        # Early stop control
        self.early_stopping = early_stopping
        self.early_stop_step = early_stop_step
        self.early_stop_clock = 0

        # Visualize
        self.show_batch_pbar = show_batch_pbar
        self.sample_num = len(train_dataset)
        self.epoch_batch_counts = self.sample_num // self.hypers.batch_size + 1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.not_save:

            if not osp.exists(self.model_dir):
                os.mkdir(self.model_dir)

            self.save_model()

            # Recording the train curve
            df = pd.DataFrame(self.metrics_results)
            df.set_index('epoch', inplace=True)
            df.to_csv(osp.join(self.model_dir, 'metrics.csv'))

    def _init_model_dir(self):
        now = datetime.datetime.now()
        formatted_datetime = now.strftime("%y%m%d%H%M%S")
        model_dir = osp.join(self.work_dir, f"cp_{formatted_datetime}")

        return model_dir

    def load_model_params(
            self,
            which: Union[int, str] = -1,
            prefix: Optional[str] = "best",
            *,
            core_only: bool = False,
            path: Optional[str] = None,
    ):
        # If the state dict is directly given.
        if isinstance(path, str):
            if not osp.exists(path):
                raise FileNotFoundError(path)
            self.core_model.load_state_dict(torch.load(path, map_location=self.device))
            return

        list_models = sorted(filter(lambda f: f != self.model_name, os.listdir(self.work_dir)))
        if isinstance(which, int):
            model_dir = osp.join(self.work_dir, list_models[which])
        elif isinstance(which, str):
            if which not in list_models:
                raise ValueError("The model you are trying to load does not exist.")
            model_dir = osp.join(self.work_dir, which)
        else:
            raise TypeError("The argument which is not a int or str.")

        # Loader core
        if isinstance(prefix, str):
            state_dict_name = f"{prefix}state_dict.pt"
        else:
            state_dict_name = f"state_dict.pt"
        state_dict = torch.load(osp.join(model_dir, state_dict_name))
        self.core_model.load_state_dict(state_dict)

        if not core_only:
            if isinstance(prefix, str):
                state_dict_name = f"{prefix}predictor_dict.pt"
            else:
                state_dict_name = f"predictor_dict.pt"
            state_dict = torch.load(osp.join(model_dir, state_dict_name))
            self.predictor.load_state_dict(state_dict)

    def save_model(self, prefix: Optional[str] = None):
        if not osp.exists(self.model_dir):
            os.mkdir(self.model_dir)

        # Save core
        torch.save(self.core_model, osp.join(self.model_dir, f'{prefix}model.pt'))
        torch.save(self.core_model.state_dict(), osp.join(self.model_dir, f'{prefix}state_dict.pt'))
        # Save Predictor
        torch.save(self.predictor, osp.join(self.model_dir, f'{prefix}predictor.pt'))
        torch.save(self.predictor.state_dict(), osp.join(self.model_dir, f'{prefix}predictor_dict.pt'))

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
        loader = DataLoader(
            self.train_dataset.load_all() if self.load_all_data else self.train_dataset,
            batch_size=self.hypers.batch_size,
            shuffle=self.kwargs.get('trainset_shuffle', True)
        )
        eval_loader = DataLoader(
            self.dataset_test.load_all() if self.load_all_data else self.dataset_test,
            batch_size=self.hypers.batch_size,
            shuffle=self.kwargs.get('evalset_shuffle', False)
        )
        # Clear cache
        torch.cuda.empty_cache()

        self.core_model = self.core_model.to(self.device)
        self.predictor = self.predictor.to(self.device)

        optimizer = self.OPTIMIZER(
            list(self.core_model.parameters()) + list(self.predictor.parameters()),
            lr=self.hypers.lr,
            weight_decay=self.hypers.weight_decay
        )

        if self.lr_scheduler:
            lr_scheduler = self.lr_scheduler(optimizer, **self.lrs_kwargs)
        else:
            lr_scheduler = None

        return loader, eval_loader, optimizer, lr_scheduler

    def get_target(
            self,
            batch: Batch,
            masked_idx: Optional[torch.Tensor] = None,
            to_onehot: bool = False,
            onehot_labels: int = None,
            loss_weight_calculator: Callable[[torch.Tensor, int], torch.Tensor] = None,
            **kwargs
    ):
        target = self.target_getter(batch)
        if isinstance(masked_idx, torch.Tensor):
            target = target[masked_idx]

        if to_onehot:
            target = F.one_hot(target.long(), num_classes=onehot_labels).to(target.dtype)  # Convert to OneHot label
        else:
            target = target.view((-1, 1))

        loss_weight = None
        if loss_weight_calculator:
            loss_weight = loss_weight_calculator(target, onehot_labels)

        return target, loss_weight

    def forward(
            self,
            batch,
            inputs_getter: Callable[[Batch], tuple[Union[torch.Tensor, Sequence], ...]],
            x_masker: Callable[[tuple[torch.Tensor, ...], torch.Tensor], tuple[torch.Tensor, torch.Tensor]] = None,
            batch_preprocessor: Callable[[Batch], Batch] = None,
            inputs_preprocessor: Callable[[tuple[torch.Tensor, ...], Union[list, torch.Tensor]], tuple[torch.Tensor, ...]] = None,
            input_x_index: Union[list, torch.Tensor] = None,
            xyz_index: Union[list, torch.Tensor] = None,
            extractor_attr_getter: Callable[[Batch], Union[tuple, torch.Tensor]] = None,
            **kwargs
    ):
        batch = batch.to(self.device)
        if batch_preprocessor:
            batch = batch_preprocessor(batch)
        inputs = inputs_getter(batch)

        if xyz_index is not None:
            xyz = get_xyz(*inputs, xyz_index=xyz_index)
        else:
            xyz = None

        if inputs_preprocessor:
            assert isinstance(input_x_index, (list, torch.Tensor))
            inputs = inputs_preprocessor(*inputs, input_x_index=input_x_index)
        if x_masker:
            inputs, masked_idx = x_masker(inputs, self.core_model.x_mask_vec)
        else:
            masked_idx = None

        # Core model
        seq, X_not_pad, R_not_pad = self.core_model(*inputs, xyz=xyz)

        # Extract features
        feature = self.feature_extractor(seq, X_not_pad, R_not_pad, batch, extractor_attr_getter)  # Node level feature
        if isinstance(masked_idx, torch.Tensor):
            feature = feature[masked_idx]

        # Prediction
        # predict atom type
        node_pred = self.predictor(feature)

        return node_pred, masked_idx

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
            epoch, loader, optimizer,
            inputs_getter: Callable[[Batch], tuple[Union[torch.Tensor, Sequence], ...]],
            inputs_preprocessor: Callable[[tuple[torch.Tensor, ...], Union[list, torch.Tensor]], tuple[torch.Tensor, ...]] = None,
            batch_preprocessor: Callable[[Batch], Batch] = None,
            input_x_index: Union[list, torch.Tensor] = None,
            xyz_index: Union[list, torch.Tensor] = None,
            x_masker: Callable[[tuple[torch.Tensor, ...], torch.Tensor], tuple[torch.Tensor, torch.Tensor]] = None,
            extractor_attr_getter: Callable[[Batch], Union[tuple, torch.Tensor]] = None,
            to_onehot: bool = False,
            onehot_labels: int = None,
            loss_weight_calculator: Callable[[torch.Tensor, int], torch.Tensor] = None,
            eval_batch_step: Optional[int] = None,
            **kwargs
    ):
        p_bar = tqdm(desc=f"Epoch: {epoch}:", total=len(loader)) if self.show_batch_pbar else None
        self.core_model.train(), self.predictor.train()
        for i, batch in enumerate(loader, 1):
            self.batch_dtype_preprocessor(batch)
            pred, masked_index = self.forward(
                batch,
                inputs_getter=inputs_getter,
                batch_preprocessor=batch_preprocessor,
                inputs_preprocessor=inputs_preprocessor,
                input_x_index=input_x_index,
                xyz_index=xyz_index,
                x_masker=x_masker,
                extractor_attr_getter=extractor_attr_getter,
                **kwargs
            )

            target, loss_weight = self.get_target(
                batch,
                masked_idx=masked_index,
                to_onehot=to_onehot,
                onehot_labels=onehot_labels,
                loss_weight_calculator=loss_weight_calculator,
                **kwargs
            )

            # Back propagation
            if isinstance(loss_weight, torch.Tensor):
                loss = self.loss_fn(pred, target, loss_weight)
            else:
                loss = self.loss_fn(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if eval_batch_step and i % eval_batch_step and isinstance(self.lazy_eval, Callable):
                self.lazy_eval(eval_max_batch=3)

            if self.debug:
                break

            if p_bar:
                p_bar.update(1)

    def _lazy_eval(self, *args, **kwargs):
        def lazy_wrapper(epoch=None, **kw):
            kwargs.update(kw)
            metric_results = self.to_eval(*args, **kwargs)
            self.print_eval_metric(metric_results, epoch)
            return metric_results
        return lazy_wrapper

    def to_eval(
            self,
            loader,
            inputs_getter: Callable[[Batch], tuple[Union[torch.Tensor, Sequence], ...]],
            inputs_preprocessor: Callable[[tuple[torch.Tensor, ...], Union[list, torch.Tensor]], tuple[torch.Tensor, Optional]] = None,
            batch_preprocessor: Callable[[Batch], Batch] = None,
            input_x_index: Union[list, torch.Tensor] = None,
            xyz_index: Union[list, torch.Tensor] = None,
            x_masker: Callable[[tuple[torch.Tensor, ...], torch.Tensor], tuple[torch.Tensor, torch.Tensor]] = None,
            extractor_attr_getter: Callable[[Batch], Union[tuple, torch.Tensor]] = None,
            to_onehot: bool = False,
            onehot_labels: int = None,
            print_pred_target_labels: bool = False,
            eval_max_batch: Optional[int] = None,
            **kwargs
    ):
        self.core_model.eval(), self.predictor.eval()

        pred = []
        target = []
        with torch.no_grad():
            for i, batch in enumerate(loader):
                self.batch_dtype_preprocessor(batch)
                node_pred, masked_index = self.forward(
                    batch,
                    inputs_getter=inputs_getter,
                    inputs_preprocessor=inputs_preprocessor,
                    batch_preprocessor=batch_preprocessor,
                    input_x_index=input_x_index,
                    xyz_index=xyz_index,
                    x_masker=x_masker,
                    extractor_attr_getter=extractor_attr_getter,
                )

                node_target, _ = self.get_target(
                    batch,
                    masked_idx=masked_index,
                    to_onehot=to_onehot,
                    onehot_labels=onehot_labels,
                )

                pred.append(node_pred.cpu().detach().float().numpy())
                target.append(node_target.cpu().detach().float().numpy())

                if self.debug:
                    break

                if eval_max_batch and i >= eval_max_batch:
                    break

            pred = np.concatenate(pred)
            target = np.concatenate(target)

            if print_pred_target_labels:
                pred_label, target_label = M.inverse_onehot(to_onehot, pred, target)
                pred_target_label = np.concatenate([pred_label, target_label], axis=1)
                assert pred_target_label.shape == (target.shape[0], 2)

            return {
                metric_name: metric_func(pred, target)
                for metric_name, metric_func in self.metrics.items()
            }

    def inspect_model(self, eval_results: dict):
        def update_best_model(pm):
            if self.not_save:
                return

            nonlocal is_update
            self.best_primary_metric = pm

            if not osp.exists(self.model_dir):
                os.mkdir(self.model_dir)
            torch.save(self.core_model.state_dict(), osp.join(self.model_dir, 'beststate_dict.pt'))
            torch.save(self.predictor.state_dict(), osp.join(self.model_dir, 'bestpredict_dict.pt'))
            is_update = True

        is_update = False
        primary_metric = eval_results[self.primary_metric]
        if self.best_primary_metric is None:
            update_best_model(primary_metric)
        else:
            if self.minimize_metric and primary_metric < self.best_primary_metric:
                update_best_model(primary_metric)
            elif not self.minimize_metric and primary_metric > self.best_primary_metric:
                update_best_model(primary_metric)

        return is_update

    def print_eval_metric(self, metric_results, epoch: Optional[int] = None):
        for metric_name, metric_value in metric_results.items():
            print(f'Eval {metric_name} in eval set {self.work_name}, epoch: {epoch}/{self.epochs}: {metric_value}')

            if isinstance(epoch, int):
                list_metric = self.metrics_results.setdefault(metric_name, [])
                list_metric.append(metric_value)

        list_epoch = self.metrics_results.setdefault('epoch', [])
        if isinstance(epoch, int):
            list_epoch.append(epoch)

    def train_eval(
            self,
            input_x_index: Union[list, torch.Tensor] = None,
            xyz_index: Union[list, torch.Tensor] = None,
            x_masker: Callable[[tuple[torch.Tensor, ...], torch.Tensor], tuple[torch.Tensor, torch.Tensor]] = None,
            extractor_attr_getter: Callable[[Batch], Union[tuple, torch.Tensor]] = None,
            to_onehot: bool = False,
            onehot_labels: int = None,
            loss_weight_calculator: Callable[[torch.Tensor, int], torch.Tensor] = None,
            **kwargs
    ):
        if to_onehot and not isinstance(onehot_labels, int):
            raise ValueError("The type of onehot should be explicitly specified.")

        # Initializing Dataloader, model and optimizer
        inputs_getter = attrgetter(
            'x', 'edge_index', 'edge_attr', 'rings_node_index',
            'rings_node_nums', 'mol_rings_nums', 'batch', 'ptr')
        loader, eval_loader, optimizer, lr_sche = self.prepare()
        if isinstance(getattr(self.core_model, 'x_label_nums', None), int):
            inputs_preprocessor = get_labeled_x_input_attrs
        else:
            inputs_preprocessor = get_x_input_attrs

        # Preparing arguments
        train_kw = dict(
            loader=loader,
            optimizer=optimizer,
            inputs_getter=inputs_getter,
            inputs_preprocessor=inputs_preprocessor,
            input_x_index=input_x_index,
            xyz_index=xyz_index,
            x_masker=x_masker,
            extractor_attr_getter=extractor_attr_getter,
            to_onehot=to_onehot,
            onehot_labels=onehot_labels,
            loss_weight_calculator=loss_weight_calculator,
            **kwargs
        )

        eval_kw = dict(
            loader=eval_loader,
            inputs_getter=inputs_getter,
            inputs_preprocessor=inputs_preprocessor,
            input_x_index=input_x_index,
            xyz_index=xyz_index,
            extractor_attr_getter=extractor_attr_getter,
            to_onehot=to_onehot,
            onehot_labels=onehot_labels,
            **kwargs
        )

        self.lazy_eval = self._lazy_eval(**eval_kw)

        # Training and evaluation
        if self.eval_first:
            self.lazy_eval(eval_max_batch=3)

        for epoch in range(self.epochs):

            # Training block
            self.to_train(epoch, **train_kw)
            if lr_sche:
                lr_sche.step()

            # Eval and early step
            if isinstance(self.eval_steps, int) and epoch % self.eval_steps == 0:
                metric_results = self.lazy_eval(epoch)
                is_update = self.inspect_model(metric_results)

                # Control early stop
                if self.early_stopping and not is_update:
                    self.early_stop_clock += 1
                else:
                    self.early_stop_clock = 0

                if self.early_stop_clock > self.early_stop_step:
                    print(RuntimeWarning(f"Early stopping in {epoch} epochs"))
                    break

    def run(self, *args, **kwargs):
        self.train_eval(*args, **kwargs)

############################## Pretrain Run ###################################
MetricType = Literal['r2score', 'rmse', 'mse', 'mae', 'accuracy', 'binary_accuracy']
metrics_options = {
    'r2score': M.Metrics.r2_score,
    'rmse': M.Metrics.rmse,
    'mae': M.Metrics.mae,
    'mse': M.Metrics.mse,
    'accuracy': M.Metrics.calc_oh_accuracy,
    'binary_accuracy': M.Metrics.binary_accuracy,
}
extractor_options = {
    "atom": M.FeatureExtractors.extract_atom_vec,
    "pair": M.FeatureExtractors.extract_pair_vec,
    "ring": M.FeatureExtractors.extract_ring_vec,
    "mol": M.FeatureExtractors.extract_mol_vec,
    "cbond": M.FeatureExtractors.extract_cbond_pair
},
loss_options = {
    'mse': F.mse_loss,
    'cross_entropy': F.cross_entropy,
    'binary_cross_entropy': F.binary_cross_entropy,
    'mean_maximum_displace': mean_maximum_displacement
}


def _get_index(first_data, data_item: str, attrs: Union[str, Iterable[str]] = None) -> Union[int, list[int]]:
    item_names = first_data[f"{data_item}_names"]
    if attrs is None:
        return list(range(len(item_names)))
    elif isinstance(attrs, str):
        return item_names.index(attrs)
    elif isinstance(attrs, Iterable):
        return [item_names.index(a) for a in attrs]

def run(
        work_name: str,
        work_dir: str,
        core_model: M.Core,
        train_dataset,
        test_dataset,
        hypers: Union[dict, Hypers],
        checkpoint_path: Union[str, int] = None,
        load_core_only: bool = False,
        epochs: int = 100,
        with_xyz: bool = True,
        save_model: bool = True,
        optimizer: Optional[Type[Optimizer]] = None,
        constant_lr: bool = False,
        lr_schedular: Optional[Callable] = None,
        lr_schedular_kwargs: Optional[dict] = None,
        target_type: Optional[M.TargetTypeName] = None,
        feature_extractor: Optional[Union[Callable, str]] = None,
        predictor: Optional[Union[nn.Module, str]] = None,
        target_getter: Optional[Union[str, Callable]] = None,
        loss_fn: Optional[Union[Callable, str]] = None,
        primary_metric: Optional[MetricType] = None,
        other_metric: Optional[Union[MetricType, Iterable[MetricType], dict[str, Callable]]] = None,
        device: Optional[Union[torch.device, str]] = None,
        eval_steps: int = 1,
        minimize_metric: bool = False,
        early_stopping: bool = True,
        early_stop_step: int = 5,
        loss_weight_calculator: Optional[Union[Callable, bool]] = None,
        loss_weight_method: Literal['inverse-count', 'cross-entropy'] = 'inverse-count',
        onehot_labels: Optional[int] = None,
        eval_each_step: Optional[int] = 1,
        **kwargs,
):
    first_data = train_dataset[0]

    if target_type is None:
        if work_name == "AtomType":
            target_type = 'onehot'
        elif work_name.startswith("xyz"):
            target_type = 'xyz'
        elif work_name in ['Cbond', 'RingAromatic']:
            target_type = 'binary'
        else:
            target_type = 'num'

    # FeatureExtractor, Predictor, LossFunc, Metrics, and TargetGetter
    fplmt = {}

    # Specify default feature extractor
    if isinstance(feature_extractor, Callable):
        fplmt['feature_extractor'] = feature_extractor
    elif isinstance(feature_extractor, str):
        if feature_extractor.lower() == 'atom':
            fplmt['feature_extractor'] = M.FeatureExtractors.extract_atom_vec
        elif feature_extractor.lower() == 'pair':
            fplmt['feature_extractor'] = M.FeatureExtractors.extract_pair_vec
        elif feature_extractor.lower() == 'ring':
            fplmt['feature_extractor'] = M.FeatureExtractors.extract_ring_vec
        elif feature_extractor.lower() == 'cbond':
            fplmt['feature_extractor'] = M.FeatureExtractors.extract_cbond_pair
        elif feature_extractor.lower() == 'mol':
            fplmt['feature_extractor'] = M.FeatureExtractors.extract_mol_vec
        else:
            raise ValueError(f"Unknown feature extractor: Named {feature_extractor}")
    else:
        if "Atom" in work_name or "xyz" in work_name:
            fplmt['feature_extractor'] = M.FeatureExtractors.extract_atom_vec
        elif "Ring" in work_name:
            fplmt['feature_extractor'] = M.FeatureExtractors.extract_ring_vec
        elif "Cbond" in work_name:
            fplmt['feature_extractor'] = M.FeatureExtractors.extract_cbond_pair
        elif "Pair" in work_name:
            fplmt['feature_extractor'] = M.FeatureExtractors.extract_pair_vec
        elif "Mol" in work_name:
            fplmt['feature_extractor'] = M.FeatureExtractors.extract_mol_vec
        else:
            raise ValueError("Unknown feature extractor type")

    # Specify default predictor
    if isinstance(predictor, (Callable, nn.Module)):
        fplmt['predictor'] = predictor
    elif isinstance(predictor, str):
        fplmt['predictor'] = M.Predictor(core_model.vec_size, predictor)
    elif target_type in ['onehot', 'xyz', 'binary', 'num']:
        fplmt['predictor'] = M.Predictor(core_model.vec_size, target_type)
    else:
        raise ValueError(f"Unknown predictor type: {target_type}")

    # Specify loss func
    if isinstance(loss_fn, Callable):
        fplmt['loss_fn'] = loss_fn
    elif isinstance(loss_fn, str):
        try:
            fplmt['loss_fn'] = loss_options[loss_fn]
        except KeyError:
            raise ValueError(f"Unknown loss function: {loss_fn}")
    else:
        if target_type == 'onehot':
            fplmt['loss_fn'] = F.cross_entropy
        elif target_type == 'xyz':
            fplmt['loss_fn'] = mean_maximum_displacement
        elif target_type == 'binary':
            fplmt['loss_fn'] = F.binary_cross_entropy
        elif target_type == 'num':
            fplmt['loss_fn'] = F.mse_loss
        else:
            raise ValueError(f"Loss function has not been specified, pass by argument `loss_fn`")

    # Specify primary metric
    if isinstance(primary_metric, str):
        try:
            fplmt['metrics'] = {primary_metric: metrics_options[primary_metric]}
        except KeyError:
            raise ValueError(f"Unknown primary metric: {primary_metric}\n, choose from: {list(metrics_options.keys())}")
    else:
        if target_type == 'onehot':
            primary_metric = 'accuracy'
            fplmt['metrics'] = {primary_metric: lambda p, t: M.Metrics.calc_oh_accuracy(p, t, is_onehot=True)}
        elif target_type == 'xyz':
            primary_metric = 'AMD'  # Average maximum displacement
            fplmt['metrics'] = {primary_metric: mean_maximum_displacement}
        elif target_type == 'binary':
            primary_metric = 'binary_accuracy'
            fplmt['metrics'] = {primary_metric: M.Metrics.binary_accuracy}
        elif target_type == 'num':
            primary_metric = 'r2score'
            fplmt['metrics'] = {primary_metric: M.Metrics.r2_score}
        else:
            raise ValueError(f"The primary metric has not been specified, pass by argument `primary_metric`")

    # Specify other target getter
    if isinstance(other_metric, str):
        if other_metric in metrics_options:
            fplmt['metrics'].update({other_metric: metrics_options[other_metric]})
        else:
            raise ValueError(f"Unknown other metric: {other_metric}\n, choose from: {list(metrics_options.keys())}")
    elif isinstance(other_metric, Iterable) and not isinstance(other_metric, dict):
        try:
            fplmt['metrics'].update({n: metrics_options[n] for n in other_metric})
        except KeyError as e:
            print(e)
            raise ValueError(f"Unknown other metric, choose from: {list(metrics_options.keys())}")
    elif isinstance(other_metric, dict):
        for n, c in other_metric.items():
            if not isinstance(n, str):
                raise TypeError(f"The metric name should be a string, instead got {type(n)}")
            elif not isinstance(c, Callable):
                raise TypeError(f"The metric value should be a callable, instead got {type(c)}")
            fplmt['metrics'].update({n: c})

    # Specify target_getter
    if isinstance(target_getter, Callable):
        fplmt['target_getter'] = target_getter
    elif isinstance(target_getter, str):
        if target_type == 'xyz':
            XYZ_INDEX = _get_index(first_data, 'x', ('x', 'y', 'z'))
            fplmt['target_getter'] = lambda batch: batch.x[:, XYZ_INDEX]
        else:
            attr_type, attr_name = target_getter.rsplit('.')
            TARGETINDEX = _get_index(first_data, attr_type, attr_name)
            fplmt['target_getter'] = lambda batch: _get_index(batch, attr_type)[:, TARGETINDEX]
    else:
        if target_type == 'xyz':
            XYZ_INDEX = _get_index(first_data, 'x', ('x', 'y', 'z'))
            fplmt['target_getter'] = lambda batch: batch.x[:, XYZ_INDEX]
        elif work_name == 'AtomType':
            fplmt['target_getter'] = lambda batch: batch.x[:, 0]

    if loss_weight_calculator is None and target_type == 'onehot':
        loss_weight_calculator = lambda t, n: M.atom_label_weight_(t, n, loss_weight_method)

    with PretrainComplex(
        work_name=work_name,
        work_dir=work_dir,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        hypers=hypers,
        core_model=core_model,
        optimizer=optimizer,
        constant_lr=constant_lr,
        lr_schedular=lr_schedular,
        lr_schedular_kwargs=lr_schedular_kwargs,
        not_save=not save_model,
        eval_first=True,
        device=device,
        epochs=epochs,
        eval_steps=eval_steps,
        early_stopping=early_stopping,
        early_stop_steps=early_stop_step,
        minimize_metric=minimize_metric,
        **fplmt,
        **kwargs,
    ) as pt:
        if checkpoint_path is not None:
            pt.load_model_params(checkpoint_path, core_only=load_core_only)

        pt.run(
            xyz_index=_get_index(first_data, 'x', ('x', 'y', 'z')) if with_xyz else None,
            loss_weight_calculator=loss_weight_calculator,
            input_x_index=_get_index(
                first_data, 'x', ('atomic_number', 'n', 's', 'p', 'd', 'f', 'g', 'x', 'y', 'z')),
            to_onehot=True if target_type == 'onehot' else False,
            onehot_labels=119 if work_name == 'AtomType' else onehot_labels,
            eval_each_step=eval_each_step,
            x_masker=x_masker_func if work_name == 'AtomType' else None,
        )


