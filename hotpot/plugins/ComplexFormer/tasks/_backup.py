from typing import Optional, Union

import torch
from torch_geometric.data import Batch
import lightning as L

from .tasks import BaseTask, MultiTask, _perform_current_task


class MultiDataTask(BaseTask):
    def __init__(self, *tasks: MultiTask):
        self._tasks = tasks
        self.current_task = None

    def choose_task(self, batch):
        assert hasattr(batch, 'dataset_idx'), "The task choice depends on batch attr `dataset_idx`, but it's not found"
        dataset_idx = batch.dataset_idx
        assert len(torch.unique(dataset_idx)), (f'The implementation of MultiDataTask requires all Data in the Batch '
            f'from a same dataset\n, but they are from various Dataset: Index{torch.unique(dataset_idx)}')
        self.current_task = self._tasks[dataset_idx[0]]

    def batch_preprocessor(self, batch: Batch) -> Batch:
        self.choose_task(batch)
        self.current_task.batch_preprocessor(batch)

    @_perform_current_task
    def inputs_getter(self, batch: Batch) -> torch.Tensor:
        """"""

    @_perform_current_task
    def get_xyz(self, inputs: tuple[torch.Tensor, ...]) -> Optional[torch.Tensor]:
        """"""

    @_perform_current_task
    def perturb_xyz(self, xyz):
        """"""

    @_perform_current_task
    def inputs_preprocessor(self, inputs: tuple[torch.Tensor, ...], **kwargs) -> tuple[torch.Tensor, ...]:
        pass

    @_perform_current_task
    def x_masker(self, inputs: tuple[torch.Tensor, ...]) -> (tuple[torch.Tensor, ...], torch.Tensor):
        pass

    @_perform_current_task
    def feature_extractor(self, *args, **kwargs) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
        pass

    @_perform_current_task
    def target_getter(self, batch: Batch) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
        pass

    @_perform_current_task
    def loss_weight_calculator(self, target):
        pass

    @_perform_current_task
    def loss_fn(self, pred, target, loss_weight):
        pass

    @_perform_current_task
    def label2oh_conversion(self, target: Union[torch.Tensor, dict[str, torch.Tensor]]):
        pass

    @staticmethod
    @_perform_current_task
    def peel_unmaksed_obj(feature_target: Union[torch.Tensor, dict[str, torch.Tensor]],
                          mask_idx: Optional[Union[torch.Tensor, dict[str, torch.Tensor]]] = None):
        pass

    @_perform_current_task
    def log_on_train_batch_end(self, pl_module: L.LightningModule, loss: Union[torch.Tensor, dict[str, torch.Tensor]],
                               pred: Union[torch.Tensor, dict[str, torch.Tensor]],
                               target: Union[torch.Tensor, dict[str, torch.Tensor]]) -> None:
        pass

    @_perform_current_task
    def add_val_pred_target(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]],
                            target: Union[torch.Tensor, dict[str, torch.Tensor]]):
        pass

    @_perform_current_task
    def eval_on_val_end(self, pl_module: L.LightningModule):
        pass
