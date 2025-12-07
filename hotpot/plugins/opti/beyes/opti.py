"""
@File Name:        opti
@Project:          
@Author:           Zhiyuan Zhang
@Created On:       2025/12/7 19:43
@Project:          Hotpot
"""
import copy
import logging
import os
from pathlib import Path
from typing import Callable, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import gpytorch
import numpy as np
import pandas as pd
import torch
from gpytorch.kernels import RBFKernel, ScaleKernel
from sklearn.manifold import MDS, TSNE
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from hotpot.plugins.plots import BayesDesignSpaceMap

__all__ = [
    "next_params",
    "draw_comics_map",
]

ArrayLike = Union[np.ndarray, torch.Tensor]

DEFAULT_BATCH_SIZE = 5
DEFAULT_ACQ_EPS_MAX = 3.0
DEFAULT_ACQ_POWER = 2.5
DEFAULT_EI_EPS = 0.01
MIN_SIGMA = 1e-9
DEFAULT_MESH_COUNTS = 20
DEFAULT_BO_ITER = 150
DEFAULT_GP_LR = 0.1


def generate_power_ladder(
    index: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_epsilon: float = DEFAULT_ACQ_EPS_MAX,
    power: float = DEFAULT_ACQ_POWER,
) -> float:
    if batch_size <= 1:
        return DEFAULT_EI_EPS
    ratio = index / (batch_size - 1)
    return max_epsilon * ratio**power


class AcquisitionFunction:
    @staticmethod
    def expected_improvement(
        mean: torch.Tensor,
        sigma: torch.Tensor,
        best_observed: torch.Tensor,
        epsilon: float = 0.02,
    ) -> torch.Tensor:
        clamped_sigma = sigma.clamp(min=MIN_SIGMA)
        diff = mean - best_observed - epsilon
        standardized = diff / clamped_sigma

        normal_dist = torch.distributions.Normal(0, 1)
        cdf_values = normal_dist.cdf(standardized)
        pdf_values = normal_dist.log_prob(standardized).exp()

        improvement = diff * cdf_values + clamped_sigma * pdf_values
        improvement[sigma <= 0.0] = 0.0
        return improvement


class GaussianProcess(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        mean_module: Optional[gpytorch.means.Mean] = None,
        covar_module: Optional[gpytorch.kernels.Kernel] = None,
        likelihood: Optional[gpytorch.likelihoods.GaussianLikelihood] = None,
    ) -> None:
        if mean_module is None:
            mean_module = gpytorch.means.ConstantMean()
        if covar_module is None:
            covar_module = ScaleKernel(RBFKernel())
        if likelihood is None:
            likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=gpytorch.constraints.GreaterThan(0.0)
            )

        super().__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module
        self.covar_module = covar_module

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class BayesianOptimizer:
    """Bayesian optimizer based on an ExactGP surrogate model."""

    def __init__(
        self,
        surrogate: gpytorch.models.ExactGP,
        acquisition_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, float], torch.Tensor] = AcquisitionFunction.expected_improvement,
        batch_size: int = 1,
    ) -> None:
        self.surrogate = surrogate
        self.acquisition_fn = acquisition_fn
        self.batch_size = batch_size
        self.is_trained = False
        self.surrogate_snapshots: List[gpytorch.models.ExactGP] = []

    def __call__(
        self,
        design_points: ArrayLike,
        n_iter: int = DEFAULT_BO_ITER,
        lr: float = DEFAULT_GP_LR,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(design_points, np.ndarray):
            design_points = torch.as_tensor(design_points, dtype=torch.float32)

        train_x, train_y = self.surrogate.train_inputs[0], self.surrogate.train_targets

        logging.info(
            "\n".join(
                f"{name}, {param}"
                for name, param in self.surrogate.named_parameters()
            )
        )

        selected_points: List[torch.Tensor] = []
        selected_means: List[torch.Tensor] = []
        selected_stds: List[torch.Tensor] = []
        selected_indices: List[torch.Tensor] = []

        for batch_index in range(self.batch_size):
            self.train_gp(n_iter=n_iter, lr=lr)
            mean, sigma = self.predict(design_points)

            epsilon = generate_power_ladder(
                index=batch_index,
                batch_size=self.batch_size,
            )
            acquisition_values = self.acquisition_fn(
                mean,
                sigma,
                train_y.max(),
                epsilon=epsilon,
            )

            best_index = torch.argmax(acquisition_values)
            selected_indices.append(best_index)
            selected_points.append(design_points[best_index])
            selected_means.append(mean[best_index])
            selected_stds.append(sigma[best_index])

            new_x = selected_points[-1].detach().clone().unsqueeze(0)
            new_y = selected_means[-1].detach().clone().unsqueeze(0)

            train_x = train_x.detach()
            train_y = train_y.detach()

            train_x = torch.cat([train_x, new_x], dim=0)
            train_y = torch.cat([train_y, new_y], dim=0)
            self.surrogate.set_train_data(train_x, train_y, strict=False)

        return (
            torch.stack(selected_points),
            torch.stack(selected_means),
            torch.stack(selected_stds),
            torch.stack(selected_indices),
        )

    def train_gp(
        self,
        n_iter: int = DEFAULT_BO_ITER,
        lr: float = DEFAULT_GP_LR,
        report_gap: Optional[int] = None,
    ) -> None:
        if report_gap is None:
            report_gap = n_iter

        self.surrogate.train()
        optimizer = torch.optim.Adam(self.surrogate.parameters(), lr=lr)
        likelihood = self.surrogate.likelihood
        marginal_log_likelihood = gpytorch.mlls.ExactMarginalLogLikelihood(
            likelihood, self.surrogate
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            output = self.surrogate(self.surrogate.train_inputs[0])
            loss = -marginal_log_likelihood(output, self.surrogate.train_targets)
            loss.backward()
            return loss

        for iteration in range(n_iter):
            loss = optimizer.step(closure)
            if (iteration + 1) % report_gap == 0:
                print(f"Iter {iteration + 1:3d}/{n_iter} - Loss: {loss.item():.3f}")

        self.surrogate.eval()
        self.surrogate_snapshots.append(copy.deepcopy(self.surrogate))
        self.is_trained = True

    def predict(
        self,
        inputs: torch.Tensor,
        which: Literal["first", "last"] = "last",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if which == "last":
            prediction = self.surrogate(inputs)
        elif which == "first":
            prediction = self.surrogate_snapshots[-1](inputs)
        else:
            raise ValueError(f"Unknown surrogate selector: {which}")

        mean = prediction.mean
        variance = prediction.variance
        sigma = torch.sqrt(variance)
        return mean, sigma

    def generate_2d_embedding_design_space(
        self,
        design_points: ArrayLike,
        optimal_indices: Optional[ArrayLike] = None,
        n_iter: int = DEFAULT_BO_ITER,
        lr: float = DEFAULT_GP_LR,
        original_x: Optional[np.ndarray] = None,
        original_y: Optional[np.ndarray] = None,
        embedding_method: Union[TSNE, MDS, None] = TSNE(),
        figpath: Optional[Union[str, os.PathLike]] = None,
        embedded_x: Optional[np.ndarray] = None,
        show_fig: bool = False,
        y_scaler: Optional[StandardScaler] = None,
        to_contourf: bool = True,
        cmap: str = "Greys",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if original_x is not None:
            num_original = len(original_x)
            if original_y is not None:
                if len(original_y) != num_original:
                    raise ValueError("Length of original_x and original_y must match.")

            design_points = np.vstack([design_points, original_x])
            original_indices = np.arange(num_original) + len(design_points)
        else:
            original_indices = None
            _ = original_indices  # keep for potential future use

        if not self.is_trained:
            self.train_gp(n_iter=n_iter, lr=lr)

        if isinstance(design_points, np.ndarray):
            design_points = torch.from_numpy(design_points).to(dtype=torch.float32)
        elif isinstance(design_points, torch.Tensor):
            design_points = design_points.to(dtype=torch.float32)
        else:
            raise TypeError("design_points must be a numpy array or torch.Tensor.")

        mean, sigma = self.predict(design_points, which="first")
        mean_np = mean.detach().numpy()
        sigma_np = sigma.detach().numpy()

        if y_scaler is not None:
            mean_np = y_scaler.inverse_transform(mean_np.reshape(-1, 1)).flatten()
            sigma_np = y_scaler.inverse_transform(sigma_np.reshape(-1, 1)).flatten()

        if embedded_x is None:
            if embedding_method is None:
                raise ValueError(
                    "embedding_method is None and no embedded_x provided."
                )
            embedded_x = embedding_method.fit_transform(design_points)

        if show_fig or figpath:
            bayes_map = BayesDesignSpaceMap(
                embedded_x,
                mean_np,
                sigma_np,
                optimal_indices,
                to_coutourf=to_contourf,
                cmap=cmap,
            )
            fig, _ = bayes_map()

            if show_fig:
                fig.show()
            if figpath:
                logging.info(f"Saved parameters space to {figpath}")
                fig.savefig(figpath)

        return embedded_x, mean_np, sigma_np


class ParamPreprocessor:
    """
    Preprocessor for numerical parameters.

    This class provides:
      - optional base-10 log transformation for selected dimensions
      - feature scaling for X and y
      - construction of a meshed design space within given parameter ranges
    """

    def __init__(
        self,
        scaler: Callable[..., MinMaxScaler] = MinMaxScaler(),
        y_scaler: Optional[StandardScaler] = StandardScaler(),
        param_range: Optional[Union[torch.Tensor, np.ndarray]] = None,
        param_names: Optional[List[str]] = None,
        param_mesh_counts: int = DEFAULT_MESH_COUNTS,
        logX_indices: Optional[Union[int, List[int]]] = None,
    ) -> None:
        self.scaler = scaler
        self.y_scaler = y_scaler
        self.param_range = param_range
        self.param_names = param_names
        self.param_mesh_counts = param_mesh_counts

        if self.param_range is not None and self.param_range.shape[1] != 2:
            raise ValueError("param_range must have shape (n_params, 2).")

        if self.param_names is not None and self.param_range is not None:
            if len(self.param_names) != len(self.param_range):
                raise ValueError(
                    "param_names length must match the number of parameters."
                )

        if isinstance(logX_indices, int):
            self.logX_indices: Optional[List[int]] = [logX_indices]
        else:
            self.logX_indices = logX_indices

    def get_design_space(self) -> torch.Tensor:
        if not isinstance(self.param_range, (torch.Tensor, np.ndarray)):
            raise AttributeError("param_range must be set to generate design space.")

        param_range = copy.deepcopy(self.param_range)

        if isinstance(self.logX_indices, list):
            for index in self.logX_indices:
                param_range[index] = np.log10(param_range[index])

        axes: List[torch.Tensor] = []
        for low, high in param_range:
            axes.append(torch.linspace(low, high, self.param_mesh_counts))

        meshgrid = torch.meshgrid(*axes, indexing="ij")
        design_points = torch.vstack([grid.flatten() for grid in meshgrid]).T
        scaled_design = torch.as_tensor(self.scaler.transform(design_points))
        return scaled_design

    def fit(self, X: torch.Tensor, y: Optional[torch.Tensor] = None) -> None:
        self.scaler.fit(X)
        if y is not None and self.y_scaler is not None:
            self.y_scaler.fit(y.reshape(-1, 1))

    def log10_transform(self, X: torch.Tensor) -> torch.Tensor:
        if not isinstance(self.logX_indices, list):
            return copy.deepcopy(X)

        subset = torch.as_tensor(X[:, self.logX_indices])
        if torch.any(subset <= 0):
            raise ValueError("log10_transform cannot be applied to non-positive values.")

        transformed = copy.deepcopy(X)
        transformed[:, self.logX_indices] = torch.log10(subset)
        return transformed

    def scale_features(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray]:
        X_scaled = self.scaler.transform(X)
        if self.y_scaler is None:
            raise ValueError("y_scaler is not set.")
        y_scaled = self.y_scaler.transform(y.reshape(-1, 1)).flatten()
        return X_scaled, y_scaled

    def inverse_log10(self, X_log: ArrayLike) -> ArrayLike:
        if not isinstance(self.logX_indices, list):
            return copy.deepcopy(X_log)

        output = copy.deepcopy(X_log)
        if isinstance(output, torch.Tensor):
            output[:, self.logX_indices] = torch.pow(10.0, output[:, self.logX_indices])
        else:
            output[:, self.logX_indices] = np.power(10.0, output[:, self.logX_indices])
        return output

    def inverse_scale(
        self,
        values: ArrayLike,
        mean: ArrayLike,
        sigma: ArrayLike,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        values_np = (
            values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else values
        )
        mean_np = (
            mean.detach().cpu().numpy() if isinstance(mean, torch.Tensor) else mean
        )
        sigma_np = (
            sigma.detach().cpu().numpy() if isinstance(sigma, torch.Tensor) else sigma
        )

        values_inv = self.scaler.inverse_transform(values_np)

        if self.y_scaler is None:
            raise ValueError("y_scaler is not set.")

        mean_inv = self.y_scaler.inverse_transform(mean_np.reshape(-1, 1))
        sigma_inv = self.y_scaler.inverse_transform(sigma_np.reshape(-1, 1))
        return values_inv, mean_inv, sigma_inv


def preprocess_inputs(
    X: ArrayLike,
    y: ArrayLike,
    param_range: np.ndarray,
    param_names: List[str],
    log_indices: Optional[Union[int, List[int]]],
    mesh_counts: int,
) -> Tuple[ParamPreprocessor, np.ndarray, np.ndarray, torch.Tensor]:
    preprocessor = ParamPreprocessor(
        param_range=param_range,
        param_names=param_names,
        logX_indices=log_indices,
        param_mesh_counts=mesh_counts,
    )

    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32)

    X_log = preprocessor.log10_transform(X)
    preprocessor.fit(X_log, y)
    X_scaled, y_scaled = preprocessor.scale_features(X_log, y)
    X_design_scaled = preprocessor.get_design_space()
    return preprocessor, X_scaled, y_scaled, X_design_scaled


def run_bayesian_optimization(
    X: ArrayLike,
    y: ArrayLike,
    design_points: ArrayLike,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Tuple[BayesianOptimizer, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    X_tensor, y_tensor, design_tensor = (
        torch.as_tensor(arr, dtype=torch.float32) for arr in (X, y, design_points)
    )

    input_dim = X_tensor.shape[1]
    covar_module = ScaleKernel(RBFKernel(ard_num_dims=input_dim))
    gp = GaussianProcess(X_tensor, y_tensor, covar_module=covar_module)

    optimizer = BayesianOptimizer(gp, batch_size=batch_size)
    X_optimal, mu_optimal, sigma_optimal, indices = optimizer(
        design_tensor, n_iter=300
    )

    for name, param in optimizer.surrogate.named_parameters():
        print(
            f"Parameter name: {name:42} value = {param.detach().cpu().tolist()}"
        )

    return optimizer, X_optimal, mu_optimal, sigma_optimal, indices


def next_params(
    X: ArrayLike,
    y: ArrayLike,
    param_range: np.ndarray,
    param_names: List[str],
    next_param_path: Union[str, os.PathLike],
    mesh_counts: int = DEFAULT_MESH_COUNTS,
    figpath: Optional[Union[str, os.PathLike]] = None,
    log_indices: Optional[Union[int, List[int]]] = None,
    to_coutourf: bool = True,
    cmap: str = "Greys",
) -> None:
    preprocessor, X_scaled, y_scaled, design_scaled = preprocess_inputs(
        X,
        y,
        param_range,
        param_names,
        log_indices,
        mesh_counts,
    )

    optimizer, X_optimal_scaled, mu_scaled, sigma_scaled, indices = run_bayesian_optimization(
        X_scaled,
        y_scaled,
        design_scaled,
        batch_size=DEFAULT_BATCH_SIZE,
    )

    optimizer.generate_2d_embedding_design_space(
        design_scaled,
        indices,
        figpath=figpath,
        to_contourf=to_coutourf,
        cmap=cmap,
        y_scaler=preprocessor.y_scaler,
    )

    X_optimal, mu_original, sigma_original = preprocessor.inverse_scale(
        X_optimal_scaled,
        mu_scaled,
        sigma_scaled,
    )
    X_optimal = preprocessor.inverse_log10(X_optimal)

    data = np.concatenate([X_optimal, mu_original, sigma_original], axis=1)
    df = pd.DataFrame(data, columns=param_names + ["mu", "sigma"])
    df.to_csv(next_param_path, index=False)


def _plot_one_by_one(
        embedded_snapshots,
        mu_snapshots,
        sigma_snapshots,
        optimal_index_snapshots,
        to_coutourf,
        cmap,
        figpath_dir_path
):
    mu_min, mu_max = BayesDesignSpaceMap.list_array_min_max(mu_snapshots)
    sigma_colors, _, sigma_label = BayesDesignSpaceMap.normalize_list_sigma(sigma_snapshots, mu_min, mu_max)
    sigma_min, sigma_max = BayesDesignSpaceMap.list_array_min_max(sigma_colors)

    for i, (
        embedded_x,
        mu,
        sigma,
        optimal_indices,
    ) in enumerate(
        zip(
            embedded_snapshots,
            mu_snapshots,
            sigma_snapshots,
            optimal_index_snapshots,
        )
    ):
        bayes_map = BayesDesignSpaceMap(
            embedded_x,
            mu,
            sigma_colors[i],
            optimal_indices,
            mu_norm=(mu_min, mu_max),
            sigma_norm=(sigma_min, sigma_max),
            cmap_mu="Grays",
            cmap_sigma="Grays",
            superscript=False,
            to_coutourf=to_coutourf,
            sigma_is_color=True,
            sigma_label=sigma_label,
            cmap=cmap,
        )
        fig, _ = bayes_map()
        fig.savefig(figpath_dir_path.joinpath(f"comics_{i}.png"))


def draw_comics_map(
    X: ArrayLike,
    y: ArrayLike,
    init_index: int,
    batch_size: int,
    param_range: Optional[np.ndarray] = None,
    param_names: Optional[Sequence[str]] = None,
    mesh_counts: int = DEFAULT_MESH_COUNTS,
    log_indices: Optional[Union[int, List[int]]] = None,
    figpath_dir: Optional[Union[str, os.PathLike]] = None,
    emb_method: Union[TSNE, MDS] = TSNE(),
    to_coutourf: bool = True,
    cmap: str = "Greys",
) -> None:
    if not (X.shape[0] == y.shape[0] > init_index):
        raise ValueError("X and y must have same length and be longer than init_index.")

    (
        preprocessor,
        X_scaled,
        y_scaled,
        design_scaled,
    ) = preprocess_inputs(
        X,
        y,
        param_range,
        list(param_names) if param_names is not None else [],
        log_indices,
        mesh_counts,
    )

    embedded_design = emb_method.fit_transform(design_scaled)

    embedded_snapshots: List[np.ndarray] = []
    mu_snapshots: List[np.ndarray] = []
    sigma_snapshots: List[np.ndarray] = []
    optimal_index_snapshots: List[np.ndarray] = []

    last_index = X.shape[0] + batch_size
    for iteration, end in enumerate(
        range(init_index, last_index, batch_size),
        start=1,
    ):
        X_batch = X_scaled[:end]
        y_batch = y_scaled[:end]
        optimizer, _, _, _, optimal_indices = run_bayesian_optimization(
            X_batch,
            y_batch,
            design_scaled,
            batch_size=batch_size,
        )

        embedded_x, mu, sigma = optimizer.generate_2d_embedding_design_space(
            design_scaled,
            optimal_indices.detach().numpy(),
            embedding_method=None,
            embedded_x=embedded_design,
            y_scaler=preprocessor.y_scaler,
            to_contourf=to_coutourf,
            cmap=cmap,
        )

        embedded_snapshots.append(embedded_x)
        mu_snapshots.append(mu)
        sigma_snapshots.append(sigma)
        optimal_index_snapshots.append(optimal_indices.detach().numpy())

    figpath_dir_path = Path(figpath_dir) if figpath_dir is not None else Path(".")

    _plot_one_by_one(
        embedded_snapshots,
        mu_snapshots,
        sigma_snapshots,
        optimal_index_snapshots,
        to_coutourf,
        cmap,
        figpath_dir_path=figpath_dir_path,
    )

    aggregate_map = BayesDesignSpaceMap(
        embedded_snapshots,
        mu_snapshots,
        sigma_snapshots,
        optimal_index_snapshots,
        cmap="viridis",
    )
    fig, _ = aggregate_map()
    fig.savefig(figpath_dir_path.joinpath("comics.png"))
