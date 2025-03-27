import lightning as L
from torch import nn


class ComplexFormer(L.LightningModule):
    def __init__(self, core: nn.Module, predictor: nn.Module):
        super().__init__()
        self.core = core
        self.predictor = predictor
