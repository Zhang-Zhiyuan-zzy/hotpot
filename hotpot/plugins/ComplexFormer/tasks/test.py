from typing import Optional, Union

import torch
from torch_geometric.data import Batch
import lightning as L

from .tasks import BaseTask

class TestTask(BaseTask):
    """"""
    def batch_preprocessor(self, batch: Batch) -> Batch:
        print(f'Test batch_preprocessor successful')

    def inputs_getter(self, batch: Batch) -> torch.Tensor:
        print(f'Test input getter successful')

    def get_xyz(self, inputs: tuple[torch.Tensor, ...]) -> Optional[torch.Tensor]:
        print(f'Test get_xyz successful')

    def perturb_xyz(self, xyz):
        print(f'Test perturb_xyz successful')

    def inputs_preprocessor(self, inputs: tuple[torch.Tensor, ...], **kwargs) -> tuple[torch.Tensor, ...]:
        print(f'Test inputs_preprocessor successful')

    def x_masker(self, inputs: tuple[torch.Tensor, ...]) -> (tuple[torch.Tensor, ...], torch.Tensor):
        print(f'Test x_masker successful')

    def feature_extractor(self, *args, **kwargs) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
        print(f'Test feature_extractor successful')

    def target_getter(self, batch: Batch) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
        print(f'Test target_getter successful')

    def loss_weight_calculator(self, target):
        print(f'Test loss_weight_calculator successful')

    def loss_fn(self, pred, target, loss_weight):
        print(f'Test loss_fn successful')

    def label2oh_conversion(self, target: Union[torch.Tensor, dict[str, torch.Tensor]]):
        print(f'Test label2oh_conversion successful')

    @staticmethod
    def peel_unmaksed_obj(feature_target: Union[torch.Tensor, dict[str, torch.Tensor]],
                          mask_idx: Optional[Union[torch.Tensor, dict[str, torch.Tensor]]] = None):
        print(f'Test peel_unmaksed_obj successful')

    def log_on_train_batch_end(self, pl_module: L.LightningModule, loss: Union[torch.Tensor, dict[str, torch.Tensor]],
                               pred: Union[torch.Tensor, dict[str, torch.Tensor]],
                               target: Union[torch.Tensor, dict[str, torch.Tensor]]) -> None:
        print(f'Test log_on_train_batch_end successful')

    def add_val_pred_target(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]],
                            target: Union[torch.Tensor, dict[str, torch.Tensor]]):
        print(f'Test add_val_pred_target successful')

    def eval_on_val_end(self, pl_module: L.LightningModule):
        print(f'Test eval_on_val_end successful')


class TestBatch:
    dataset_idx = torch.zeros(256)

test_batch = TestBatch()