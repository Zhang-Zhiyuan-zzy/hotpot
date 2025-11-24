from typing import Union, Iterable, Callable

import torch
from torch_geometric.data import Batch, Data

from .models import where_metal


target_getter_condition_fns = {
    'where_metal': where_metal
}


def specify_target_getter(
        first_data: Data,
        data_item: str,
        attrs: str = None,
        condition_fn: str = None
):
    return TargetGetter(first_data, data_item, attrs, target_getter_condition_fns.get(condition_fn, None))


class TargetGetter:
    def __init__(
            self,
            first_data,
            data_item: str,
            attrs: Union[str, Iterable[str]] = None,
            condition_fn: Callable = None,
    ):
        self.data_item = data_item
        self.attrs = attrs
        self.data_idx =  get_index(first_data, self.data_item, attrs)
        self.condition_fn = condition_fn

    def __call__(self, batch: Batch) -> torch.Tensor:
        target = self.get_target(batch)
        if isinstance(self.condition_fn, Callable):
            judge = self.condition_fn(target)
            target = target[judge]

        return target

    def get_target(self, batch: Batch) -> torch.Tensor:
        try:
            if self.data_idx is None:
                return getattr(batch, self.data_item)
            else:
                return getattr(batch, self.data_item)[:, self.data_idx]

        except Exception as e:
            msg = e.args[0] + f'\tdata.item={self.data_item} attr={self.attrs}'
            raise type(e)(msg)


def get_index(first_data, data_item: str, attrs: Union[str, Iterable[str]] = None) -> Union[int, list[int]]:
    try:
        item_names = first_data[f"{data_item}_names"]
    except KeyError:
        return None

    try:
        if attrs is None:
            return list(range(len(item_names)))
        elif isinstance(attrs, str):
            return item_names.index(attrs)
        elif isinstance(attrs, Iterable):
            return [item_names.index(a) for a in attrs]
    except Exception as e:
        msg = e.args[0] + f'\nItem={data_item} attrs={attrs}'
