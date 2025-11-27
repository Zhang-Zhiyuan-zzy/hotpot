import json
import logging
from pathlib import Path
from typing import Union, Iterable, Callable, Optional

import torch
from torch_geometric.data import Batch, Data

from .models import where_metal
from .data_process.normalize_data import TensorNormalizer, DataProfile


target_getter_condition_fns = {
    'where_metal': where_metal
}


def specify_target_getter(
        first_data: Data,
        data_item: str,
        attrs: str = None,
        condition_fn: str = None
):
    return TargetGetter(
        first_data,
        data_item, attrs,
        target_getter_condition_fns.get(condition_fn, None)
    )


class TargetGetter:
    def __init__(
            self,
            first_data,
            data_item: str,
            attrs: Union[str, Iterable[str]] = None,
            condition_fn: Callable = None,
            normalizer: Optional[TensorNormalizer] = None
    ):
        self.data_item = data_item
        self.attrs = attrs
        self.data_idx =  get_index(first_data, self.data_item, attrs)
        self.condition_fn = condition_fn
        self.normalizer = self._specify_normalizer(first_data, normalizer)

    def _specify_normalizer(self, first_data: Data, normalizer: TensorNormalizer) -> Optional[TensorNormalizer]:
        if normalizer is not None:
            return normalizer

        if not hasattr(first_data, 'dir_datasets') or not hasattr(first_data, 'dataset_name'):
            return None

        if isinstance(self.attrs, str):
            attr = self.attrs
        elif isinstance(self.attrs, Iterable):
            attr = list(self.attrs)
            if len(attr) != 1:
                return None
        else:
            return None

        profile_path = Path(first_data.dir_datasets) / '.profile' / 'stats.json'
        if not profile_path.is_file():
            return None

        try:
            with open(profile_path, 'r') as f:
                full_stats = json.load(f)

            path = f'{first_data.dataset_name}/{self.data_item}/{attr}'
            profile_dict = full_stats.get(path, None)
            if profile_dict is None:
                return None

            logging.debug(f"Loading TensorNormalizer for dataset path: [#3f51b5]`{path}`[/]")
            return TensorNormalizer(path, DataProfile(path, **profile_dict))

        except (json.JSONDecodeError, TypeError, ValueError) as e:
            raise e


    def __call__(self, batch: Batch, norm=False) -> torch.Tensor:
        target = self.get_target(batch)
        if isinstance(self.condition_fn, Callable):
            judge = self.condition_fn(target)
            target = target[judge]

        if norm:
            target = self.normalize(target)

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

    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.normalizer is not None:
            tensor = self.normalizer.transform(tensor)
        return tensor

    def inverse(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.normalizer is not None:
            tensor = self.normalizer.inverse(tensor)
        return tensor



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
