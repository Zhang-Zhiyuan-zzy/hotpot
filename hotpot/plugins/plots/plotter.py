"""
python v3.9.0
@Project: hotpot
@File   : core
@Auther : Zhiyuan Zhang
@Data   : 2024/10/14
@Time   : 13:50
"""
import string
import functools
from copy import copy
from typing import *

import numpy as np
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt

from .base import Plotter, Plot
from . import utils
from .defaults import Settings


_axes_settings_methods = []
def axes_set_method(method: Callable):
    """
    A decorator for SciPlotter Axes setting methods,
    these methods could be skipped after passing keyword args named not_{method_name} with a True value
    into the SciPlotter instance.
    """
    @functools.wraps(method)
    def setting_wrapper(self: "SciPlotter", *args, **kwargs):
        if not getattr(self.settings, f"not_{method.__name__}", False):
            return method(self, *args, **kwargs)
        else:
            print(f"skip {method.__name__}")

    return setting_wrapper


class SciPlotter(Plotter):
    def __init__(
            self,
            plotters: Union["Plot", Callable, np.ndarray[Union["Plot", Callable]], Sequence[Callable]],
            **kwargs
    ):
        """"""
        self.settings = Settings()
        self.set_arguments(**kwargs)

        # Adjust the shape of given plotters
        if not isinstance(plotters, np.ndarray):
            if isinstance(plotters, Callable):
                plotters = np.array([[plotters]])
            elif isinstance(plotters, Sequence):
                plotters = np.array(plotters)
            else:
                raise TypeError('the given plotters must be Callable, Sequence or np.ndarray objects')

        if len(plotters.shape) > 2:
            raise AttributeError('the dimension of given plotters should less than or equal to 2')
        elif len(plotters.shape) == 1:
            plotters = plotters.reshape(-1, plotters.size)

        self.plotters = plotters
        self.nrows, self.ncols = self.plotters.shape

        fig, axs = plt.subplots(self.nrows, self.ncols)

        self.fig = fig
        if self.nrows == self.ncols == 1:
            self.axs = np.array([[axs]])
        elif self.nrows == 1 or self.ncols == 1:
            self.axs = np.array([axs])
        else:
            self.axs = axs

        # self.figwidth = kwargs.get('figwidth', self.settings.figwidth * self.ncols)
        # self.figheight = kwargs.get('figheight', self.settings.figheight * self.nrows)
        self.figwidth, self.figheight = self.init_figure_size(self.nrows, self.ncols)
        self.fig.set(figwidth=self.figwidth, figheight=self.figheight, dpi=self.settings.dpi)

        super().__init__()

    @property
    def rows_interval(self):
        left, top, width, height = self.settings.axes_pos
        right = 1. - left - width
        return 0.5 * (right + left)

    @property
    def cols_interval(self):
        left, top, width, height = self.settings.axes_pos
        bottom = 1. - top - height
        return 0.5 * (bottom + top)

    def init_figure_size(self, nrows, ncols):
        left, top, width, height = self.settings.axes_pos
        right = 1. - left - width
        bottom = 1. - top - height

        rows_interval = 0.5 * (right + left)
        cols_interval = 0.5 * (bottom + top)

        # figwidth = self.settings.figwidth * (ncols - (ncols-1) * 0.5 * (right + left))
        # figheight = self.settings.figheight * (nrows - (nrows-1) * 0.5 * (bottom + top))

        figwidth = self.settings.figwidth * ncols
        figheight = self.settings.figheight * nrows

        return figwidth, figheight

    def set_arguments(self, **kwargs):
        for name, value in kwargs.items():
            setattr(self.settings, name, value)

    def print_settings(self):
        for name, value in self.settings.__dict__.items():
            print(f"{name}: ", value)

    def __call__(self, *args, **kwargs):
        """"""
        # Settings in Axes level
        for i, (ax, plot) in enumerate(zip(self.axs.flatten(), self.plotters.flatten())):
            twin_axs = plot(ax, self)  # Making plot

            # General settings for Axes
            self.set_spline(ax, twin_axs)
            self.set_xylabel_font(ax, twin_axs)
            self.set_ticks(ax, twin_axs)
            self.set_axes_position(ax, twin_axs)
            # self.rotate_tick_labels(ax)

            if self.plotters.size > 1:
                self.add_superscript(ax, i)

        # post process by modifier
        self.post_process_axes()

        return self.fig, self.axs

    @axes_set_method
    def set_spline(self, main_ax: plt.Axes, *twin_axs: plt.Axes):
        for _, spline in main_ax.spines.items():
            spline.set_linewidth(self.settings.splinewidth)

        for tax in  twin_axs:
            for _, spline in tax.spines.items():
                if isinstance(tax, plt.Axes):
                    spline.set_linewidth(self.settings.splinewidth)

    @axes_set_method
    def set_xylabel_font(self, main_ax: plt.Axes, *twin_axs: plt.Axes):
        """"""
        def to_set(ax: plt.Axes):
            ax.xaxis.label.set_font(self.settings.font)
            ax.xaxis.label.set_fontsize(self.settings.xy_label_fontsize)
            ax.xaxis.label.set_fontweight('bold')

            ax.yaxis.label.set_font(self.settings.font)
            ax.yaxis.label.set_fontsize(self.settings.xy_label_fontsize)
            ax.yaxis.label.set_fontweight('bold')

        to_set(main_ax)
        for tax in twin_axs:
            if isinstance(tax, plt.Axes):
                to_set(tax)

    # General settings
    @axes_set_method
    def set_spline(self, main_ax: plt.Axes, *twin_axs: plt.Axes):
        for _, spline in main_ax.spines.items():
            spline.set_linewidth(self.settings.splinewidth)

        for tax in twin_axs:
            if isinstance(tax, plt.Axes):
                for _, spline in tax.spines.items():
                    spline.set_linewidth(self.settings.splinewidth)

    @axes_set_method
    def set_ticks(self, main_ax: plt.Axes, *twin_axs: plt.Axes):
        """
        Set the tick properties
        Args:
            main_ax: the main Axes object
            twin_axs: the twin Axes object
        """
        def to_set(ax: plt.Axes):
            ax.tick_params(width=self.settings.splinewidth, length=self.settings.splinewidth * 2)

            for xtick_labels in ax.xaxis.get_ticklabels():
                xtick_labels.set(
                    font=self.settings.font,
                    fontsize=self.settings.xticklabels_fontsize or self.settings.ticklabels_fontsize,
                    rotation=self.settings.xticklabels_rotation or self.settings.ticklabels_rotation
                )
            for ytick_labels in ax.yaxis.get_ticklabels():
                ytick_labels.set(
                    font=self.settings.font,
                    fontsize=self.settings.yticklabels_fontsize or self.settings.ticklabels_fontsize,
                    rotation=self.settings.yticklabels_rotation or self.settings.ticklabels_rotation
                )

        to_set(main_ax)

        for tax in twin_axs:
            if isinstance(tax, plt.Axes):
                to_set(tax)

    @axes_set_method
    def set_axes_position(self, main_ax, *twin_axs):
        pos = utils.abs2rela_pos(main_ax, *self.settings.axes_pos)

        main_ax.set_position(pos)
        for tax in twin_axs:
            if isinstance(tax, plt.Axes):
                tax.set_position(pos)

    @axes_set_method
    def add_superscript(self, ax, i, font_dict: dict = None):
        xfrac, yfrac = self.settings.superscript_position
        utils.add_text(ax, xfrac, yfrac, string.ascii_lowercase[i], self.settings.superscript_font_dict)

    def add_colorbar(
            self, ax: plt.Axes, mappable=None,
            value_lim: Sequence = None,
            colorbar_label: str = None,
            colorbar_pos: Sequence[float] = None,
            orientation: Literal['horizontal', 'vertical'] = 'vertical',
            **kwargs
    ):
        arguments = locals()
        arguments.pop('self')
        arguments.pop('kwargs')
        arguments.update(kwargs)

        self.add_axes_modifier('ax', utils.later_call(utils.add_colorbar)(**arguments))


class LegendOrganizer:
    def __init__(self):
        """"""
        self.legend_names = []

    # Methods applied in external codes
    def add_plot(self, obj_name, plot_func, *args, **kwargs):
        """
        Make the plots and recording the name of the plot.
        Calling this method instead of directly calling `plot_func` will be convenient when add
        the legend to the plots, by calling the `add_legend` method in the end.
        Args:
            obj_name: the name of the adding object to plot.
            plot_func: the function to make the object on the plot
            *args: arguments to pass to the `plot_func`
            **kwargs: keyword arguments to pass to the `plot_func`

        Returns:
            None
        """
        # Make plot
        plot_func(*args, **kwargs)
        self.legend_names.append(obj_name)

    def add_legend(self, ax: plt.Axes, *args, **kwargs):
        """ Add the legend to the plot, after record the objects by make_plots method """
        ax.legend(self.legend_names, *args, **kwargs)

