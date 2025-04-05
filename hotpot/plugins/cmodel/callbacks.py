import io
import math
from typing import Any, Union, Optional

import lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from lightning.pytorch.callbacks import Callback, ProgressBar



def get_metric_table(
        metrics_dict,
        table_kw: dict = None,
        title: str = None,
):
    if table_kw is None:
        table_kw = {}

    assert len(metrics_dict) > 0
    t_cols = min(4, len(metrics_dict))
    # t_rows = math.ceil(len(metrics_dict) / t_cols)
    t_rest = len(metrics_dict) % t_cols

    table = Table(title=title, **table_kw)
    for _ in range(t_cols):
        table.add_column('ID', no_wrap=True)
        table.add_column('Metric', no_wrap=True)
        table.add_column('Value', no_wrap=True)

    rows = []
    row = []
    for i, (name, value) in enumerate(metrics_dict.items(), 1):
        if len(row) == 3 * t_cols:
            rows.append(row)
            row = []

        row.extend(map(str, (i, name, f'{value:.3g}')))

    for _ in range(t_rest):
        row.extend([''] * 3)
    rows.append(row)

    for row in rows:
        table.add_row(*row)

    return table


def _update_n(bar: tqdm, value: int) -> None:
    if not bar.disable:
        bar.n = value
        bar.refresh()


class Pbar(ProgressBar):
    def __init__(self):
        super().__init__()
        self.buf = None
        self.train_pbar = None
        self.layout = Layout()
        self.layout.split(Layout(name='val'), Layout(name='train'))
        self.liver = None

    def init_train_tqdm(self):
        """Override this to customize the tqdm bar for training."""
        return tqdm(
            desc=self.train_description,
            total=self.total_train_batches,
            leave=True,
            dynamic_ncols=True,
            file=self.buf,
            smoothing=0,
        )

    def end_liver(self):
        self.liver.stop()

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.buf = io.StringIO()
        self.train_pbar = self.init_train_tqdm()
        self.liver = Live(self.layout, auto_refresh=False)
        self.liver.start()

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        metrics = self.get_metrics(trainer, pl_module)
        table = get_metric_table(metrics)
        _update_n(self.train_pbar, batch_idx+1)

        table.title = f"Eval in Training Step  (Epoch {pl_module.current_epoch})"
        caption = self.buf.getvalue().split('\r')[-1]
        table.caption = caption
        # self.liver.update(table)

        self.layout['train'].update(table)
        self.liver.refresh()

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.buf.close()
        self.end_liver()
        self.train_pbar = None

    def on_sanity_check_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.liver = Live(self.layout, auto_refresh=False)
        self.liver.start()

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        epoch = pl_module.current_epoch
        metrics = pl_module.val_metrics

        if len(metrics) > 0:
            table = get_metric_table(metrics, {'style': 'magenta'}, title=f'Eval in Validation Step (Epoch {epoch})')

            self.layout['val'].update(table)
            self.liver.refresh()

    def on_sanity_check_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.liver.stop()
