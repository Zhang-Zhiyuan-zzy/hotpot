import io
from typing import Any

import torch
import torch.distributed as dist
import lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from tqdm import tqdm
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from lightning.pytorch.callbacks import Callback, ProgressBar

from hotpot.utils import fmt_print
from .data.loader import DistConcatLoader

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

def get_inspect_table(
        inspect_dict,
        table_kw: dict = None,
        title: str = None,
):
    if table_kw is None:
        table_kw = {}

    table = Table(title=title, **table_kw)
    table.add_column('Task', no_wrap=True)
    table.add_column('Mean', no_wrap=True)
    table.add_column('Std', no_wrap=True)
    table.add_column('Median', no_wrap=True)
    table.add_column('Max', no_wrap=True)
    table.add_column('Min', no_wrap=True)

    for tsk, list_inspect in inspect_dict.items():
        row = [tsk] + [f'{v:.3g}' for v in list_inspect]
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

    def disable(self):
        pass

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

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.layout = Layout()
        if getattr(pl_module, 'pred_inspect', None):
            self.layout.split(Layout(name='val'), Layout(name='train'), Layout(name='inspect'))
        else:
            self.layout.split(Layout(name='val'), Layout(name='train'))

        self.liver = Live(self.layout, auto_refresh=False)
        self.liver.start()

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.buf = io.StringIO()
        self.train_pbar = self.init_train_tqdm()

    # def on_train_batch_start(
    #     self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int
    # ) -> None:
    #     self.layout = Layout()
    #     if getattr(pl_module, 'pred_inspect', None):
    #         self.layout.split(Layout(name='val'), Layout(name='train'), Layout(name='inspect'))
    #     else:
    #         self.layout.split(Layout(name='val'), Layout(name='train'))
    #
    #     self.liver = Live(self.layout, auto_refresh=False)
    #     self.liver.start()

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        # metrics = self.get_metrics(trainer, pl_module)
        metrics = pl_module.train_metrics
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
        self.train_pbar = None

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.end_liver()

    def on_sanity_check_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.liver = Live(self.layout, auto_refresh=False)
        self.liver.start()

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        epoch = pl_module.current_epoch
        metrics = pl_module.val_metrics
        inspect = pl_module.pred_inspect

        if len(metrics) > 0:
            table = get_metric_table(metrics, {'style': 'magenta'}, title=f'Eval in Validation Step (Epoch {epoch})')
            self.layout['val'].update(table)

        if inspect:
            inspect_table = get_inspect_table(inspect, {'style': 'green'}, title=f'Pred Inspect in Validation Step (Epoch {epoch})')
            self.layout['inspect'].update(inspect_table)
            pl_module.pred_inspect = None

        self.liver.refresh()


    def on_sanity_check_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.liver.stop()


class Debugger(Callback):
    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        loader = trainer.train_dataloader
        fmt_print.bold_magenta(str(len(loader)))

        # Check whether all data in a batch from same dataset
        batches = {}
        for batch in iter(loader):
            if isinstance(loader, DistConcatLoader):
                assert len(torch.unique(batch.dataset_idx)) == 1
                dataset_idx = batch.dataset_idx[0]
            else:
                dataset_idx = None

            list_batch = batches.setdefault(dataset_idx, [])
            list_batch.append(batch)

        # batch_dataset_idx = torch.cat([batch[0].dataset_idx for batch in iter(loader)])
        rank = f' {dist.get_rank()} ' if isinstance(dist.get_rank(), int) else ''
        # fmt_print.dark_green(f"Batch dataset_idx in{rank}{batch_dataset_idx.__repr__()}")

        for dataset_idx, list_batch in batches.items():
            batch = list_batch[0]
            if isinstance(dataset_idx, int):
                pl_module.tasks.choose_task(batch)

            target = pl_module.tasks.target_getter(batch)
            for tsk_name, tgt in target.items():
                print(f'{rank}in dataset {dataset_idx} {tsk_name}: {tgt.shape}')
