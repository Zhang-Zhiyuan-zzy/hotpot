from typing import Union, Optional, Any
import torch
import torch.nn as nn

import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT


class Model(L.LightningModule):
    def __init__(
            self,
            core: Union[nn.Module, str],
            predictor: nn.Module
    ):
        super().__init__()
        self.core = core
        self.predictor = predictor

    def forward(self, *args, **kw):
        return self.core(*args, **kw)

    @property
    def x_label_nums(self) -> Optional[int]:
        return getattr(self.core, 'x_label_nums', None)

    @property
    def x_mask_vec(self) -> Optional[torch.Tensor]:
        return getattr(self.core, 'x_mask_vec', None)