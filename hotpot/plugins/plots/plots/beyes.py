"""
python v3.9.0
@Project: hotpot
@File   : beyes
@Auther : Zhiyuan Zhang
@Data   : 2024/10/19
@Time   : 9:17
"""
from typing import Union

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from .base import Plot
from ..plotter import SciPlotter
from .. import utils


__all__ = ['BayesDesignSpaceMap']


class BayesDesignSpaceMap(SciPlotter):
    """  Visualize the BayesDesignSpace """
    def _draw_mu_map(self, ax, sciplot, *args, **kwargs):
        emb_x = self.emb_x[self.plot_index]
        mu = self.mus[self.plot_index]

        self._draw_map(emb_x[:, 0], emb_x[:, 1], mu, ax)
        sciplot.add_colorbar(ax, colorbar_label='mu', norm=self.mu_norm, cmap=self.cmap)

    def _draw_sigma_map(self, ax, sciplot, *args, **kwargs):
        emb_x = self.emb_x[self.plot_index]
        sigma = self.sigmas[self.plot_index]

        self._draw_map(emb_x[:, 0], emb_x[:, 1], sigma, ax)
        sciplot.add_colorbar(ax, colorbar_label='sigma', norm=self.sigma_norm, cmap=self.cmap)

        self.plot_index += 1

    def _draw_map(self, x, y, c, ax, mesh_num=50):
        if self.to_coutourf:
            # Create a regular grid
            # xi, yi = np.meshgrid(np.linspace(x.min(), x.max(), mesh_num), np.linspace(y.min(), y.max(), mesh_num))
            #
            # # Interpolate the scattered data onto the regular grid
            # zi = griddata((x, y), c, (xi, yi), method='linear')
            # zi = np.nan_to_num(zi, nan=0.0)
            ax.contourf(
                *utils.scatter_to_coutourf(x, y, c, mesh_num),
                # cmap='viridis'
                cmap=self.cmap
            )

        else:
            ax.scatter(x, y, c=c, alpha=0.3)

        if self.X_opti_idx is not None:
            ax.scatter(
                x[self.X_opti_idx[self.plot_index]],
                y[self.X_opti_idx[self.plot_index]],
                c='r', marker='*', s=150
            )

    def __init__(
            self,
            emb_x: Union[list[np.ndarray], np.ndarray],
            mus: Union[list[np.ndarray], np.ndarray],
            sigmas: Union[list[np.ndarray], np.ndarray],
            X_opti_idx: Union[list[np.ndarray], np.ndarray] = None,
            to_coutourf=True,
            mesh_num=100,
            mu_norm: tuple[int, int] = None,
            sigma_norm: tuple[int, int] = None,
            cmap='Grays',
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
            mu_norm(tuple(int, int)): the min and max limit of mu in colorbar
            sigma_norm(tuple(int, int)): the min and max limit of sigma in colorbar
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

        # Check whether the number of emb_x, mus, and sigmas are equal.
        if not (len(emb_x) == len(mus) == len(sigmas)):
            raise ValueError("the length of emb_x and mus and sigmas must match, "
                             f"got emb_x: {len(emb_x)} and mus: {len(mus)} and sigmas: {len(sigmas)}")
        if X_opti_idx is not None and len(X_opti_idx) != len(emb_x):
            raise ValueError("the given X_opti_idx and emb_x must have the same length, "
                             f"got X_opti_idx: {len(X_opti_idx)} and emb_x: {len(emb_x)}")

        self.plot_index = 0
        self.plots_num = len(emb_x)

        self.emb_x = emb_x
        self.mus = mus
        self.sigmas = sigmas
        self.X_opti_idx = X_opti_idx
        self.to_coutourf = to_coutourf
        self.mesh_num = mesh_num
        self.cmap = cmap

        if mu_norm is None:
            mu_norm = min(mu.min() for mu in mus), max(mu.max() for mu in mus)
        if sigma_norm is None:
            sigma_norm = min(sig.min() for sig in sigmas), max(sig.max() for sig in sigmas)

        self.mu_norm = Normalize(*mu_norm)
        self.sigma_norm = Normalize(*sigma_norm)

        plotters = np.array([self._draw_mu_map, self._draw_sigma_map] * self.plots_num).reshape((self.plots_num, 2))
        super().__init__(plotters, **kwargs)

