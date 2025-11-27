"""
@File Name:        num_normalize
@Project:          
@Author:           Zhiyuan Zhang
@Created On:       2025/11/24 19:27
@Project:          Hotpot
"""
import logging
import os
import json
import math
from enum import Enum
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Union, Iterable

import numpy as np
from scipy import stats
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch_geometric.data import Batch

from hotpot.utils.configs.logging_config import RateLimitLogger
from hotpot.utils import fmt_print




# ==============================================================================
# Constants & Enums
# ==============================================================================
SKEWNESS_THRESHOLD = 1.0
VARIANCE_THRESHOLD = 1e-6
EPSILON = 1e-8


class Method(Enum):
    IDENTITY = "identity"
    MIN_MAX = "min_max"
    Z_SCORE = "z_score"
    LOG = "log"

WorkItemDict = dict[str, Union[None, list[str]]]

# ==============================================================================
# Data Structures
# ==============================================================================

@dataclass
class DataProfile:
    """Holds statistical descriptors for a specific target variable."""
    path: str
    method: Method
    data: Optional[np.ndarray] = None
    mean: float = 0.0
    std: float = 0.0
    min_val: float = 0.0
    max_val: float = 1.0
    shift: float = 0.0

    def __post_init__(self):
        if isinstance(self.method, str):
            try:
                self.method = Method(self.method)
            except ValueError:
                raise ValueError(f"Unknown method: {self.method}")

    def to_dict(self):
        d = self.__dict__.copy()
        d['method'] = self.method.value
        return d


class DatasetAnalyzer:
    def __init__(
            self,
            datasets_root: Union[str, os.PathLike],
            profile_dir: Optional[os.PathLike] = None,
            strict_mode: bool = False,
    ):
        self.datasets_root = Path(datasets_root)

        self.profile_dir = Path(profile_dir) if profile_dir else self.datasets_root / '.profile'
        if not self.profile_dir.exists():
            self.profile_dir.mkdir(parents=True, exist_ok=True)

        self.strict_mode = strict_mode
        self.profiles = {}

    ####################### Helpers ############################
    @staticmethod
    def _parse_path(path):
        item_idx = None
        if len(path) == 1:
            dataset_name = path[0]
            data_item = 'y'
        elif len(path) == 2:
            dataset_name = path[0]
            data_item = path[1]
        elif len(path) == 3:
            dataset_name = path[0]
            data_item = path[1]
            item_idx = path[2]
        else:
            raise ValueError(f'Unregular path: {path}')
        return dataset_name, data_item, item_idx

    def _resolve_dataset_names(self, paths: Optional[Union[str, Iterable[str]]]) -> list[str]:
        """
        Standardizes input paths into a list of dataset names.
        Scans the datasets_root for directories if paths is None.
        """
        if paths is None:
            return [
                p for p in os.listdir(self.datasets_root)
                if self.datasets_root.joinpath(p).is_dir() and not p.startswith('.')
            ]

        if isinstance(paths, str):
            return [paths]

        if isinstance(paths, Iterable):
            return list(paths)

        raise TypeError('paths must be a string or an iterable of strings')

    def _define_io_works_from_paths(self, paths) -> dict[str, WorkItemDict]:
        works = defaultdict(dict)
        for path in paths:
            path = path.split('/')
            dataset_name, data_item, item_idx = self._parse_path(path)
            works[dataset_name].setdefault(data_item, []).append(item_idx)

        for dataset_name, items_dict in works.items():
            for item_name, item_idx in items_dict.items():
                if any(iti is None for iti in item_idx):
                    items_dict[item_name] = None

        return dict(works)

    def _recommend_method(self, data: np.ndarray, verbose: bool) -> Method:
        if len(data) < 2 or np.var(data) < VARIANCE_THRESHOLD:
            if verbose: print("Variance too low. Using Identity.")
            return Method.IDENTITY

        skewness = float(stats.skew(data))
        if verbose: print(f"Skewness: {skewness:.4f}")

        if abs(skewness) > SKEWNESS_THRESHOLD:
            if verbose: print("High skew detected. Using Log.")
            return Method.LOG

        if self._is_uniform_better_fit(data, verbose):
            if verbose: print("Uniform distribution detected. Using MinMax.")
            return Method.MIN_MAX

        if verbose: print("Normal distribution detected. Using Z-Score.")
        return Method.Z_SCORE

    @staticmethod
    def _determine_params_minmax(path: str, data: np.ndarray) -> DataProfile:
        return DataProfile(
            path=path,
            method=Method.MIN_MAX,
            data=data,
            min_val=float(np.min(data)),
            max_val=float(np.max(data)),
            mean=float(np.mean(data)),
            std=float(np.std(data)),
        )

    @staticmethod
    def _determine_params_zscore(path: str, data: np.ndarray) -> DataProfile:
        mean = float(np.mean(data))
        std = float(np.std(data))

        if std < EPSILON:
            std = 1.0

        return DataProfile(
            path=path,
            method=Method.Z_SCORE,
            data=data,
            mean=mean,
            std=std,
            min_val=float(np.min(data)),
            max_val=float(np.max(data))
        )

    @staticmethod
    def _determine_params_log(path: str, data: np.ndarray) -> DataProfile:
        # Shift data to be positive before log: log(x - min + 1)
        min_val = np.min(data)
        shift = 0.0
        if min_val <= 0:
            shift = abs(min_val) + 1.0

        log_data = np.log(data + shift)

        # Normalize the log-transformed data to N(0,1)
        mean = float(np.mean(log_data))
        std = float(np.std(log_data))

        if std < EPSILON:
            std = 1.0

        return DataProfile(
            path=path,
            method=Method.LOG,
            data=data,
            shift=shift,
            mean=mean,
            std=std,
            min_val=min_val,
            max_val=float(data.max()),
        )

    def _get_normal_params(self, path: str, data: np.ndarray, verbose: bool = False) -> DataProfile:
        """
        Analyzes the data distribution, fits the parameters.

        Strategies:
        - Constant Data -> Identity
        - Highly Skewed (> 1.0) -> Log Transform + Z-Score
        - Uniform-like -> MinMax Scaling (0, 1)
        - Normal-like -> Z-Score Standardization (0, 1)

        Args:
            path: the data path name, `dataset_name/item_name/item_idx_name`
            data: 1D numpy array of numerical values.
            verbose: If True, prints the selected method and metrics.

        Returns:
            Normalized 1D numpy array.
        """
        cleaned_data = data[np.isfinite(data)]
        method = self._recommend_method(cleaned_data, verbose)

        if method == Method.IDENTITY:
            return DataProfile(path, method, data)

        if method == Method.MIN_MAX:
            return self._determine_params_minmax(path, data)

        if method == Method.LOG:
            return self._determine_params_log(path, data)

        return self._determine_params_zscore(path, data)

    @staticmethod
    def _is_uniform_better_fit(data: np.ndarray, verbose: bool) -> bool:
        mu, std = stats.norm.fit(data)
        ks_norm, _ = stats.kstest(data, 'norm', args=(mu, std))

        min_val, max_val = np.min(data), np.max(data)
        ks_uniform, _ = stats.kstest(data, 'uniform', args=(min_val, max_val - min_val))

        if verbose: print(f"KS-Norm: {ks_norm:.4f}, KS-Uniform: {ks_uniform:.4f}")
        return ks_uniform < ks_norm

    def _visualize_data_distribution(self, cols: int = 3) -> None:
        num_plots = len(self.profiles)
        rows = math.ceil(num_plots / cols)

        fig, axs = plt.subplots(rows, cols, figsize=(15, rows * 5))
        axs = axs.flatten()
        for i, (path, data_stat) in enumerate(self.profiles.items()):
            fmt_print.bold_magenta(f'Path{path}, mix={data_stat.data.mean():.4f}, max={data_stat.data.max():.4f}, mean={data_stat.data.mean():.4f}')
            axs[i].hist(data_stat.data, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
            axs[i].set_title(f'{data_stat.path}')
            axs[i].set_xlabel('Values')
            axs[i].set_ylabel('Density')

            info_box = (
                f"Method: {data_stat.method}\n"
                f"Mean: {data_stat.mean:.4f}\n"
                f"Std:  {data_stat.std:.4f}\n"
                f"Min:  {data_stat.min_val:.4f}\n"
                f"Max:  {data_stat.max_val:.4f}\n"
                f"Shift:  {data_stat.shift:.4f}"
            )
            axs[i].text(
                0.95, 0.95, info_box, transform=axs[i].transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.75)
            )

        fig.tight_layout()
        fig.savefig(self.profile_dir / f'dist_vis.png')

    def _save_profile(self) -> None:
        save_dict = {
            data_stat.path: {
                'method': data_stat.method.value,
                'min_val': float(data_stat.min_val),
                'max_val': float(data_stat.max_val),
                'mean': float(data_stat.mean),
                'std': float(data_stat.std),
                'shift': float(data_stat.shift),
            } for path, data_stat in self.profiles.items()
        }
        json.dump(save_dict, open(self.profile_dir / 'stats.json', 'w'), indent=4)

    def _process_single_dataset(
            self,
            dataset_name: str,
            item_dict: WorkItemDict,
    ) -> None:
        dataset_path = self.datasets_root.joinpath(dataset_name)

        data = None
        item_values = defaultdict(list)
        for data_path in tqdm(dataset_path.glob('*.pt'), f'Reading {dataset_name} ...'):
            data = torch.load(data_path, weights_only=False, map_location='cpu')

            for item_name in item_dict.keys():
                item = getattr(data, item_name, None)
                if item is None:
                    if self.strict_mode:
                        raise AttributeError(f'Dataset {dataset_name} has no attribute {item_name}')
                    print(RuntimeWarning(f'Dataset {dataset_name} has no attribute {item_name}'))
                    return

                item_values[item_name].append(
                    item.view(1, -1) if item.ndim == 1 else item
                )

        if data is None:
            raise IOError(f'No data found for {dataset_name}')

        for item_name, values in item_values.items():
            values = torch.cat(values, dim=0)

            chosen_idx_names = item_dict[item_name]
            if chosen_idx_names is not None:
                assert isinstance(chosen_idx_names, list)
                existed_idx_names: list = getattr(data, f'{item_name}_names')
                chosen_idx = torch.tensor([existed_idx_names.index(cin) for cin in chosen_idx_names])
                values = values[:, chosen_idx]
            else:
                chosen_idx_names = getattr(data, f'{item_name}_names')

            for i, idx_name in enumerate(chosen_idx_names):
                # Summary and statistic
                path = f'{dataset_name}/{item_name}/{idx_name}'
                self.profiles[path] = self.compute_stats(
                    path=f'{dataset_name}/{item_name}/{idx_name}',
                    data=values[:, i].detach().cpu().numpy()
                )

    def compute_stats(self, path: str, data: np.ndarray) -> DataProfile:
        return self._get_normal_params(path, data)

    ################# End of Helpers ###########################

    ################ Public Interface ##########################
    def analyze_datasets(self, paths: Optional[Union[str, Iterable[str]]] = None):
        paths = self._resolve_dataset_names(paths)
        works = self._define_io_works_from_paths(paths)

        # Read each data.pt just once
        self.profiles = {}  # Clear the profile
        for dataset_name, item_dict in works.items():
            self._process_single_dataset(dataset_name, item_dict)

        self._visualize_data_distribution()
        self._save_profile()

    def batch_transform(self, batch: Batch) -> Batch:
        ...

class TensorNormalizer:
    def __init__(self, path: str, profile: DataProfile):
        self.path = path
        self.profile = profile
        self.trans_logger = RateLimitLogger(interval_count=100)
        self.inv_logger = RateLimitLogger(interval_count=100)

    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        self.trans_logger.debug(f"Transform tensor from path: [#3f51b5]`{self.path}`[/] with method [#3f51b5]{self.profile.method.value}[/]")
        profile = self.profile
        if profile.method == Method.IDENTITY:
            return tensor
        elif profile.method == Method.MIN_MAX:
            return (tensor - profile.min_val) / (profile.max_val - profile.min_val)
        elif profile.method == Method.Z_SCORE:
            return (tensor - profile.mean) / profile.std
        elif profile.method == Method.LOG:
            return (torch.log(tensor + profile.shift) - profile.mean) / profile.std
        else:
            raise NotImplementedError(f'Unknown transformation method {profile.method}')

    def inverse(self, normalized_tensor: torch.Tensor) -> torch.Tensor:
        self.inv_logger.debug(f"Inverse tensor from path: [#3f51b5]`{self.path}`[\] with method [#3f51b5]{self.profile.method.value}[\]")
        profile = self.profile
        if profile.method == Method.IDENTITY:
            return normalized_tensor
        elif profile.method == Method.MIN_MAX:
            return normalized_tensor * (profile.max_val - profile.min_val) + profile.min_val
        elif profile.method == Method.Z_SCORE:
            return normalized_tensor * profile.std + profile.mean
        elif profile.method == Method.LOG:
            return torch.exp(normalized_tensor * profile.std + profile.mean) - profile.shift
        else:
            raise NotImplementedError(f'Unknown transformation method {profile.method}')
