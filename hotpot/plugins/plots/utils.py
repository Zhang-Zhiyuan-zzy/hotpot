"""
python v3.9.0
@Project: hotpot
@File   : utils
@Auther : Zhiyuan Zhang
@Data   : 2024/10/14
@Time   : 15:54
"""
import functools
from typing import *

import numpy as np
from scipy.stats import gaussian_kde
from scipy.interpolate import griddata

import matplotlib as mpl
from matplotlib import pyplot as plt

from .defaults import Settings


def calc_active_span(nrows, ncols, i, j):
    """ calculate the active span of Axes, according to the arranging of all Axes
    and the position of calculated Axes """
    x_dividers = [i * 1 / ncols for i in range(ncols)]
    y_dividers = list(reversed([i * 1 / nrows for i in range(nrows)]))

    return x_dividers[j], y_dividers[i]


def abs2rela_pos(
        ax: plt.Axes, left: float, bottom: float, width: float, height: float,
        rows_interval: float = None, cols_interval: float = None
):
    """
    Calculate the position of the image on the entire canvas based on its span ratio relative to its active area
    Args:
        ax: the Axes whose position is calculated
        left: the fraction span from the left active boundary to the left spline
        bottom: the fraction span from the bottom active boundary to the bottom spline
        width: the fraction span from the left spline to the right spline
        height: the fraction span from the bottom spline to the top spline
        rows_interval: the interval between rows of the plots
        cols_interval: the interval between columns of the plots

    Returns:
        the (left, bottom, width, height) of calculated Axes relate to the entire canvas.
    """
    # right = 1. - left - width
    # top = 1. - bottom - height
    # if not rows_interval:
    #     rows_interval = 0.5 * (right + left)
    # if not cols_interval:
    #     cols_interval = 0.5 * (top + bottom)

    # Retrieve the axes grid position in the subplot
    subplotspec = ax.get_subplotspec().get_topmost_subplotspec()
    i, j = subplotspec.rowspan[0], subplotspec.colspan[0]
    nrows, ncols = subplotspec.get_gridspec().nrows, subplotspec.get_gridspec().ncols
    # i, j = map(int, np.where(self.axs == ax))
    left_boundary, bottom_boundary = calc_active_span(nrows, ncols, i, j)

    # left = left_boundary + 1 / ncols * (left - 0. if not i else rows_interval)
    # bottom = bottom_boundary + 1 / nrows * (bottom - 0. if not (nrows - j - 1) else cols_interval)

    left = left_boundary + 1 / ncols * left
    bottom = bottom_boundary + 1 / nrows * bottom

    width = 1 / ncols * width
    height = 1 / nrows * height

    return left, bottom, width, height


def insert_axes(main_ax, ins_ax, relative_pos: Sequence[float] = None):
    if relative_pos is None:
        relative_pos = Settings.axes_pos

    pos = abs2rela_pos(main_ax, *relative_pos)
    ins_ax.set_position(pos)


def specify_colorbar_pos(main_ax: plt.Axes, orientation: Literal["horizontal", "vertical"] = 'vertical'):
    # colorbar_pos = list(main_ax.get_position().bounds)
    colorbar_pos = list(Settings.axes_pos)
    if orientation == 'vertical':
        colorbar_pos[0], colorbar_pos[2] = 0.85, 0.05
    elif orientation == 'horizontal':
        colorbar_pos[1], colorbar_pos[3] = 0.05, 0.05

    return colorbar_pos


def add_colorbar(
        ax: plt.Axes, mappable=None,
        value_lim: Sequence = None,
        colorbar_label: str = None,
        colorbar_pos: Sequence[float] = None,
        orientation: Literal['horizontal', 'vertical'] = 'vertical',
        norm: mpl.colors.Normalize = None,
        cmap: Union[str, mpl.colors.Colormap] = None,
        **kwargs
):
    if not mappable:
        mappable = plt.cm.ScalarMappable(norm, cmap)

    ax_pos = list(ax.get_position().bounds)
    if orientation == 'vertical':
        ax_pos[2] -= 0.025

    colorbar = ax.figure.colorbar(mappable, ax=ax, orientation=orientation, **kwargs)
    ax.set_position(ax_pos)

    # specify colorbar position
    if not colorbar_pos:
        colorbar_pos = specify_colorbar_pos(ax, orientation)

    insert_axes(ax, colorbar.ax, colorbar_pos)

    if value_lim:
        colorbar.ax.set_ylim(*value_lim)

    if colorbar_label:
        colorbar.ax.set_ylabel(
            colorbar_label, fontdict={'font': Settings.font, 'fontsize': 28, 'fontweight': 'bold'})

    # adjust the ticklabel of colorbar
    if orientation == 'vertical':
        for ticklabel in colorbar.ax.get_yticklabels():
            ticklabel.set(font=Settings.font, fontsize=16)
    elif orientation == 'horizontal':
        for ticklabel in colorbar.ax.get_xticklabels():
            ticklabel.set(font=Settings.font, fontsize=16)


def add_text(ax, xfrac: float, yfrac: float, s: str, fontdict: dict = None):
    (xb, xu), (yb, yu) = ax.get_xlim(), ax.get_ylim()
    x, y = (1 - xfrac) * xb + xfrac * xu, (1 - yfrac) * yb + yfrac * yu

    ax.text(x, y, s, fontdict=fontdict)

def _densi(x: np.ndarray, y: np.ndarray) -> np.array:
    return np.argsort(np.abs(x * y))[::-1]

def sample_density(a, b, ratio=0.4):
    """ sample according to density distribution, useful when the number of samples is too large """
    return a[(den := _densi(a.flatten(), b.flatten()))], a[den] + (b[den] - a[den]) * (1 - ratio)


def later_call(func: Callable):

    @functools.wraps(func)
    def decorator(*args, **kwargs):
        def wrapper():
            return func(*args, **kwargs)
        return wrapper

    return decorator


def calculate_scatter_density(xy: np.ndarray):
    """ Calculate the point density """
    xy = np.float64(xy)
    d = gaussian_kde(xy)(xy)
    return np.log2(d)


def scatter_to_coutourf(x, y, c, mesh_num=50):
    """"""
    xi, yi = np.meshgrid(np.linspace(x.min(), x.max(), mesh_num), np.linspace(y.min(), y.max(), mesh_num))

    zi = griddata((x, y), c, (xi, yi), method='linear')
    zi = np.nan_to_num(zi, nan=0.0)

    return xi, yi, zi
