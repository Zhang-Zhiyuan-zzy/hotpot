"""
python v3.9.0
@Project: hotpot
@File   : beyes
@Auther : Zhiyuan Zhang
@Data   : 2024/10/19
@Time   : 9:17
"""
import logging
from typing import Union

import torch
import numpy as np
from matplotlib.colors import Normalize

from ..plotter import SciPlotter
from .. import utils, colors


__all__ = ['BayesDesignSpaceMap']

EPSILON = 1e-8


class BayesDesignSpaceMap(SciPlotter):
    """  Visualize the BayesDesignSpace """
    def _draw_mu_map(self, ax, sciplot, *args, **kwargs):
        emb_x = self.emb_x[self.plot_index]
        mu = self.mus[self.plot_index]

        self._draw_map(emb_x[:, 0], emb_x[:, 1], mu, ax)
        sciplot.add_colorbar(
            ax,
            colorbar_label=r"$\mathrm{Mean\ values}\ (\mu)$",
            norm=self.mu_norm,
            cmap=self.cmap
        )

    def _draw_sigma_map(self, ax, sciplot, *args, **kwargs):
        emb_x = self.emb_x[self.plot_index]
        sigma = self.sigma_colors[self.plot_index]

        self._draw_map(emb_x[:, 0], emb_x[:, 1], sigma, ax)
        sciplot.add_colorbar(
            ax,
            colorbar_label=self.sigma_colorbar_label,
            norm=self.sigma_norm,
            cmap=self.cmap
        )

        self.plot_index += 1

    def _draw_map(self, x, y, c, ax, mesh_num=50):
        if self.to_coutourf:
            ax.contourf(
                *utils.scatter_to_coutourf(x, y, c, mesh_num),
                cmap=self.cmap
            )

        else:
            ax.scatter(x, y, c=c, alpha=0.3, cmap=self.cmap)

        if self.X_opti_idx is not None:
            ax.scatter(
                x[self.X_opti_idx[self.plot_index]],
                y[self.X_opti_idx[self.plot_index]],
                c='r', marker='*', s=150
            )

        if self.X_orig_idx is not None:
            ax.scatter(
                x[self.X_orig_idx[self.plot_index]],
                y[self.X_orig_idx[self.plot_index]],
                c='orange', marker='*', s=150
            )

    @staticmethod
    def list_array_min_max(list_arr: list[np.ndarray]) -> tuple[float, float]:
        mu_min = min(mu.min() for mu in list_arr)
        mu_max = max(mu.max() for mu in list_arr)
        return mu_min, mu_max

    @staticmethod
    def normalizelist_sigma(sigmas, mu_min, mu_max):
        delta_mu = mu_max - mu_min
        max_abs_mu = max(abs(mu_max), abs(mu_min))

        # === Calculate the color for sigma ==========================
        logging.info(f"delta_mu = {delta_mu}; max_abs_mu = {max_abs_mu}")
        if delta_mu < 0.01 * max_abs_mu:
            if max_abs_mu < EPSILON:
                sigma_ref_value = 1.
                label = r"$\mathrm{Absolute std. dev. (\sigma)}$"
            else:
                sigma_ref_value = max_abs_mu
                label = r"$\mathrm{Relative std. dev. (\sigma / \max(\mu))}$"
        else:
            sigma_ref_value = delta_mu
            label = r"$\mathrm{Relative std. dev. (\sigma / \Delta\mu)}$"

        sigma_colors: list[np.ndarray] = []
        for sigma in sigmas:
            if np.any(sigma < 0.):
                raise ValueError("sigma must be > 0.")

            sigma_colors.append(sigma / sigma_ref_value)

        return sigma_colors, sigma_ref_value, label

    def __init__(
            self,
            emb_x: Union[list[np.ndarray], np.ndarray],
            mus: Union[list[np.ndarray], np.ndarray],
            sigmas: Union[list[np.ndarray], np.ndarray],
            X_opti_idx: Union[list[np.ndarray], np.ndarray] = None,
            X_orig_idx: Union[list[np.ndarray], np.ndarray] = None,
            to_coutourf=True,
            mesh_num=100,
            mu_norm: tuple[float, float] = None,
            sigma_norm: tuple[float, float] = None,
            sigma_is_color: bool = False,
            sigma_label: str = None,
            cmap='Greys',
            **kwargs
    ):
        """

        Args:
            emb_x:
            mus:
            sigmas:
            X_opti_idx:
            to_coutourf:
            mesh_num:
            mu_norm(tuple(float, float)): the min and max limit of mu in colorbar
            sigma_norm(tuple(float, float)): the min and max limit of sigma in colorbar
            **kwargs:
        """
        # Convert the Numpy Array to list of Array.
        if isinstance(emb_x, np.ndarray):
            emb_x = [emb_x]
        if isinstance(mus, np.ndarray):
            mus = [mus]
        if isinstance(sigmas, np.ndarray):
            sigmas = [sigmas]
        if X_opti_idx is not None and isinstance(X_opti_idx, (np.ndarray, torch.Tensor)):
            X_opti_idx = [X_opti_idx]
        if X_orig_idx is not None and isinstance(X_orig_idx, (np.ndarray, torch.Tensor)):
            X_orig_idx = [X_orig_idx]

        # Check whether the number of emb_x, mus, and sigmas are equal.
        if not (len(emb_x) == len(mus) == len(sigmas)):
            raise ValueError("the length of emb_x and mus and sigmas must match, "
                             f"got emb_x: {len(emb_x)} and mus: {len(mus)} and sigmas: {len(sigmas)}")
        if X_opti_idx is not None and len(X_opti_idx) != len(emb_x):
            raise ValueError("the given X_opti_idx and emb_x must have the same length, "
                             f"got X_opti_idx: {len(X_opti_idx)} and emb_x: {len(emb_x)}")
        if X_orig_idx is not None and len(X_orig_idx) != len(emb_x):
            raise ValueError("the given X_orig_idx and emb_x must have the same length, "
                             f"got X_opti_idx: {len(X_orig_idx)} and emb_x: {len(emb_x)}")

        self.plot_index = 0
        self.plots_num = len(emb_x)

        self.emb_x = emb_x
        self.mus = mus
        self.sigmas = sigmas
        self.X_opti_idx = X_opti_idx
        self.X_orig_idx = X_orig_idx
        self.to_coutourf = to_coutourf
        self.mesh_num = mesh_num
        self.cmap = colors.load_cmap(cmap)

        if mu_norm is None:
            mu_min, mu_max = self.list_array_min_max(self.mus)
        else:
            mu_min, mu_max = mu_norm

        sigma_colors, sigma_ref_value, sigma_colorbar_label = self.normalize_list_sigma(self.sigmas, mu_min, mu_max)

        self.sigma_ref_value = sigma_ref_value
        self.sigma_colorbar_label = sigma_label if isinstance(sigma_label, str) else sigma_colorbar_label
        self.sigma_colors = self.sigmas if sigma_is_color else sigma_colors
        # ===================================================================

        if sigma_norm is None:
            # 使用 sigma / Δmu 的真实范围，并让 colorbar 从 0 开始
            max_val = max(col.max() for col in self.sigma_colors)
            sigma_norm = (0.0, max_val)
        else:
            sigma_norm = (0.0, sigma_norm[1])

        self.mu_norm = Normalize(mu_min, mu_max)
        self.sigma_norm = Normalize(*sigma_norm)

        plotters = np.array([self._draw_mu_map, self._draw_sigma_map] * self.plots_num).reshape((self.plots_num, 2))
        super().__init__(plotters, **kwargs)

