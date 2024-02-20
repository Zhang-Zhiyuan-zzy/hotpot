"""
python v3.9.0
@Project: hotpot
@File   : plots
@Auther : Zhiyuan Zhang
@Data   : 2023/11/9
@Time   : 20:43

Notes:
    this module define some easy methods for drawing of common scientific plots.

"""
import itertools
from typing import *
import string
import re
from copy import copy

import numpy as np
import shap
from scipy.stats import gaussian_kde
from sklearn import linear_model
from sklearn.feature_selection import r_regression
from sklearn.metrics import r2_score
from scipy.cluster.hierarchy import dendrogram, linkage

import matplotlib.pyplot as plt


def axes_setting(setting: Callable):
    def not_setting(self, *args, **kwargs):
        self.kwargs.pop(f"not_{setting.__name__}")

    def setting_wrapper(self, *args, **kwargs):
        if not self.kwargs.get(f"not_{setting.__name__}", False):
            return setting(self, *args, **kwargs)
        else:
            return not_setting(self, *args, **kwargs)

    return setting_wrapper


class SciPlotter:
    """
    An automatic plot maker to draw scientific plots, with the style like the Origin software:
        1) set the size of figure and the position of Axes
        2) set the font Properties for xy labels, ticks and insert text
        3) set the spline line widtho
    """
    _figwidth = 10.72  # inch
    _figheight = 8.205  # inch
    _dpi = 300

    default_axes_pos = [0.1483, 0.1657, 0.7163, 0.7185]
    _font = 'Arial'
    _splinewidth = 3
    _ticklabels_fontsize = 18
    _xy_label_fontsize = 28

    _superscript_xy_frac = (0.025, 0.925)
    superscript_dict = {'font': _font, 'fontsize': 32, "fontweight": 'bold'}

    def __init__(
            self,
            plotters: Union["Plot", Callable, np.ndarray[Union["Plot", Callable]]],
            superscript: bool = True,
            ax_adjust: dict[plt.Axes, Callable] = None,
            ax_adjust_kwargs: dict[plt.Axes, dict] = None,
            fig_adjust: Callable = None,
            fig_adjust_kwargs: dict = None,
            **kwargs  # Custom format keywords arguments
    ):
        """

        Args:
            plotters:
            superscript: whether to add superscript into axes
            fig_adjust: post process in figure level
            fig_adjust_kwargs: kwargs for post_process function

        Keyword Args:
            figwidth(float|int): figure width
            figheight(float|int): figure height
            axes_position(tuple): axes position, (left, bottom, width, height)
            font(str): the fonts for plotting
            splinewidth(float|int): the width of splines
            ticklabels_fontsize(float|int): the font size for tick labels
            xticklabels_fontsize(float|int): the font size for xtick labels
            yticklabels_fontsize(float|int): the font size for ytick labels
            xy_label_fontsize(float|int): the font size for xy-axis labels
            ticklabels_rotation(float|int): rotation for tick labels
            xticklabels_rotation(float|int): rotation for xtick labels
            yticklabels_rotation(float|int): rotation for ytick labels
        """
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

        self.figwidth = kwargs.get('figwidth', self._figwidth * self.ncols)
        self.figheight = kwargs.get('figheight', self._figheight * self.nrows)
        self.fig.set(figwidth=self.figwidth, figheight=self.figheight, dpi=self._dpi)

        self.axes_position = kwargs.get('axes_position', self.default_axes_pos)

        self.ax_modifier = ax_adjust if isinstance(ax_adjust, dict) else {}
        self.ax_modifier_kwargs = ax_adjust_kwargs if isinstance(ax_adjust_kwargs, dict) else {}

        self.fig_modifier = fig_adjust
        self.fig_modifier_kwargs = fig_adjust_kwargs if fig_adjust_kwargs else {}

        # Whether add the Axes superscript
        self.superscript = superscript
        # Custom format arguments
        self.font = kwargs.get('font', self._font)
        self.splinewidth = kwargs.get('splinewidth', self._splinewidth)
        self.tick_fontsize = kwargs.get('tick_fontsize', self._ticklabels_fontsize)
        self.xy_label_fontsize = kwargs.get('xy_label_fontsize', self._xy_label_fontsize)

        # Other keyword arguments
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        """"""
        # Settings in Axes level
        for i, (ax, plotter) in enumerate(zip(self.axs.flatten(), self.plotters.flatten())):
            ins_ax = plotter(ax, self)  # Making plot

            # General settings for Axes
            self.set_spline(ax, ins_ax)
            self.set_xylabel_font(ax, ins_ax)
            self.set_ticks(ax, ins_ax)
            self.set_axes_position(ax, ins_ax)
            # self.rotate_tick_labels(ax)

            if self.plotters.size > 1 and self.superscript:
                self.add_superscript(ax, i)

            # Custom settings
            if ax_modifier := self.ax_modifier.get(ax):
                ax_modifier(self.fig, ax, **self.ax_modifier_kwargs.get(ax, {}))

        # Setting in Figure level
        if self.fig_modifier:
            self.fig_modifier(self.fig, self.axs, **self.fig_modifier_kwargs)

        return self.fig, self.axs

    def add_axes_colorbar(self, fig: plt.Figure, ax: plt.Axes, mappable):
        colorbar = ax.figure.colorbar(mappable)
        pos = copy(self.axes_position)
        pos[0], pos[2] = 0.90, 0.05

        self.insert_other_axes_into(ax, colorbar, pos)

        colorbar.ax.set_ylim(-1., 1.)
        # adjust the colorbar labels
        colorbar.ax.set_ylabel('Pearson Correlation',
                            fontdict={'font': self.font, 'fontsize': 28, 'fontweight': 'bold'})

        # adjust the ticklabel of colorbar
        for ticklabel in colorbar.ax.get_yticklabels():
            ticklabel.set(font=self.font, fontsize=16)

    def add_superscript(self, ax, i):
        xfrac, yfrac = self._superscript_xy_frac

        self.add_text(ax, xfrac, yfrac, string.ascii_lowercase[i], self.superscript_dict)

    @staticmethod
    def add_text(ax, xfrac: float, yfrac: float, s: str, fontdict=None):
        (xb, xu), (yb, yu) = ax.get_xlim(), ax.get_ylim()
        x, y = (1 - xfrac) * xb + xfrac * xu, (1 - yfrac) * yb + yfrac * yu

        ax.text(x, y, s, fontdict=fontdict)

    @staticmethod
    def axes_modifier_container(*modifiers_kwargs: tuple[Callable, dict]):
        def wrapper(fig, ax, **kwargs):
            for modifier, kws in modifiers_kwargs:
                modifier(fig, ax, **kws)

        return wrapper

    @staticmethod
    def calculate_scatter_density(xy: np.ndarray):
        """ Calculate the point density """
        xy = np.float_(xy)
        d = gaussian_kde(xy)(xy)
        return np.log2(d)

    @staticmethod
    def calc_active_span(nrows, ncols, i, j):
        """ calculate the active span of Axes, according to the arranging of all Axes
        and the position of calculated Axes """
        x_dividers = [i * 1 / ncols for i in range(ncols)]
        y_dividers = list(reversed([i * 1 / nrows for i in range(nrows)]))

        return x_dividers[j], y_dividers[i]

    @staticmethod
    def calc_subaxes_pos(ax: plt.Axes, left: float, bottom: float, width: float, height: float):
        """
        Calculate the position of the image on the entire canvas based on its span ratio relative to its active area
        Args:
            ax: the Axes whose position is calculated
            left: the fraction span from the left active boundary to the left spline
            bottom: the fraction span from the bottom active boundary to the bottom spline
            width: the fraction span from the left spline to the right spline
            height: the fraction span from the bottom spline to the top spline

        Returns:
            the (left, bottom, width, height) of calculated Axes relate to the entire canvas.
        """
        # Retrieve the axes grid position in the subplot
        subplotspec = ax.get_subplotspec().get_topmost_subplotspec()
        i, j = subplotspec.rowspan[0], subplotspec.colspan[0]
        nrows, ncols = subplotspec.get_gridspec().nrows, subplotspec.get_gridspec().ncols
        # i, j = map(int, np.where(self.axs == ax))
        left_boundary, bottom_boundary = SciPlotter.calc_active_span(nrows, ncols, i, j)

        left = left_boundary + 1 / ncols * left
        bottom = bottom_boundary + 1 / nrows * bottom
        width = 1 / ncols * width
        height = 1 / nrows * height

        return left, bottom, width, height

    def insert_other_axes_into(self, main_ax: plt.Axes, insert_ax: plt.Axes, relative_pos: Union[list, np.ndarray]):
        """
        insert other Axes into the main Axes, given the given position relating to span of main Axes as the base.
        Args:
            main_ax: where the other Axes is inserted into
            insert_ax: the inserted Axes
            relative_pos: the relative position for the main Axes that the inserted Axes places. all values of
             the relative_pos is from 0 to 1.
        """
        self.set_axes_position(insert_ax, pos=self.calc_subaxes_pos(main_ax, *relative_pos))

    def rotate_tick_labels(self, ax: plt.Axes):
        if degree := (self.kwargs.get('ticklables_rotation') or self.kwargs.get('xticklabels_rotation')):
            for xticklabels in ax.get_xticklabels():
                xticklabels.set_rotation(degree)

        if degree := (self.kwargs.get('ticklables_rotation') or self.kwargs.get('yticklabels_rotation')):
            for yticklabels in ax.get_yticklabels():
                yticklabels.set_rotation(degree)

    @axes_setting
    def set_axes_position(self, *axs, pos: Optional[Union[np.ndarray, Sequence]] = None):
        if not pos:
            pos = self.calc_subaxes_pos(axs[0], *self.axes_position)

        for ax in axs:
            if isinstance(ax, plt.Axes):
                ax.set_position(pos)

    @axes_setting
    def set_xylabel_font(self, main_ax: plt.Axes, ins_ax: plt.Axes = None):
        """"""

        def to_set(ax):
            ax.xaxis.label.set_font(self.font)
            ax.xaxis.label.set_fontsize(self.xy_label_fontsize)
            ax.xaxis.label.set_fontweight('bold')

            ax.yaxis.label.set_font(self.font)
            ax.yaxis.label.set_fontsize(self.xy_label_fontsize)
            ax.yaxis.label.set_fontweight('bold')

        to_set(main_ax)
        if ins_ax:
            to_set(ins_ax)

    @axes_setting
    def set_spline(self, main_ax: plt.Axes, ins_ax: plt.Axes = None):
        for _, spline in main_ax.spines.items():
            spline.set_linewidth(self.splinewidth)

        if ins_ax:
            for _, spline in ins_ax.spines.items():
                spline.set_linewidth(self.splinewidth)

    @axes_setting
    def set_ticks(self, main_ax: plt.Axes, twin_ax: plt.Axes = None):
        """
        Set the tick properties
        Args:
            main_ax: the main Axes object
            twin_ax: the twin Axes object
        """
        def to_set(ax):
            ax.tick_params(width=self.splinewidth, length=self.splinewidth * 2)

            for xtick_labels in ax.xaxis.get_ticklabels():
                xtick_labels.set(
                    font=self.font,
                    fontsize=self.kwargs.get(
                        'xticklables_fontsize',
                        self.kwargs.get('ticklables_fontsize', self._ticklabels_fontsize)
                    ),
                    rotation=self.kwargs.get('ticklables_rotation') or self.kwargs.get('xticklabels_rotation')
                )
            for ytick_labels in ax.yaxis.get_ticklabels():
                ytick_labels.set(
                    font=self.font,
                    fontsize=self.kwargs.get(
                        'yticklabels_fontsize',
                        self.kwargs.get('ticklabels_fontsize', self._ticklabels_fontsize)
                    ),
                    rotation=self.kwargs.get('ticklables_rotation') or self.kwargs.get('yticklabels_rotation')
                )

        to_set(main_ax)
        if twin_ax:
            to_set(twin_ax)


class Plot:
    """"""
    def __call__(self, ax: plt.Axes, sciplot: SciPlotter = None):
        """
        Args:
            ax: the Axes object to be made Plots
            sciplot: the SciPlot object to make Plots
        """
        raise NotImplemented


class R2Regression(Plot):
    def __init__(
            self, xy1, xy2=None, unit: str = None, xy_lim: Union[Sequence, np.ndarray] = None,
            s1=None, s2=None, c1=None, c2='green', marker1=None, marker2=None, err1=None, err2=None,
            *args, **kwargs
    ):
        """
        The templet for R-squared regression plot, which is usually used to show the performance of trained ML model.
        The plot allow plot the R^2 in bath train set and test set together, by giving the args both 'xy1' and 'xy2'.

        When just 'xy1' is given, the points density will be calculated and showed in plot.
        When both 'xy1' and 'xy2' are given, xy1 is seen as train set and xy2 the test_set.

        This plot will calculate the R2 score and show it on plot automatically.
        Args:
            xy1: numpy array with shape [2, n-sample], where the x (i.e., the first row) is target value,
                y is the predicted value.
            xy2: numpy array with shape [2, n-sample], where the x (i.e., the first row) is target value,
                y is the predicted values
            unit(str): the unit of target and prediction, if given will be showed.
            xy_lim(Sequence|np.ndarray): set the minimum and maximum for both xy-axis, the default value
                is the minimum and maximum of xy1 and xy2
            s1(float): sample size for xy1
            s2(float): sample size for xy2
            c1: color for xy1
            c2: color for xy2
            marker1: marker for xy1
            marker2: marker for xy2
            *args: other args for Matplotlib.Axes.Scatter
            **kwargs: other kwargs for Matplotlib.Axes.Scatter
        """
        self.xy1 = xy1 if isinstance(xy1, np.ndarray) else np.array(xy1)
        self.xy2 = xy2 if (xy2 is None) or isinstance(xy2, np.ndarray) else np.array(xy2)
        self.unit = f' ({unit})' if unit else ''

        if xy_lim:
            self.xy_lim = xy_lim
        elif xy2 is None:
            xy_diff = self.xy1.max() - self.xy1.min()
            self.xy_lim = (self.xy1.min() - 0.05*xy_diff, self.xy1.max() + 0.05*xy_diff)
        else:
            xy_diff = max([self.xy1.max(), self.xy2.max()]) - min([self.xy1.min(), self.xy2.min()])
            self.xy_lim = (
                min([self.xy1.min(), self.xy2.min()]) - 0.05*xy_diff,
                max([self.xy1.max(), self.xy2.max()]) + 0.05*xy_diff
            )

        self.s1 = s1
        self.s2 = s1 if not s2 else s1
        self.c1 = c1
        self.c2 = c2
        self.marker1 = marker1
        self.marker2 = marker2 if not marker2 else marker1
        self.err1 = err1
        self.err2 = err2

        self.args = args
        self.kwargs = {}

        self.r2_1 = r2_score(xy1[0], xy1[1])
        if self.xy2 is not None:
            self.r2_2 = r2_score(xy2[0], xy2[1])
            self.kwargs.update({'alpha': 0.3})
        else:
            self.c1 = SciPlotter.calculate_scatter_density(xy1)

        self.kwargs.update(kwargs)

    @staticmethod
    def add_diagonal(ax: plt.Axes):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_autoscale_on(False)
        ax.plot(xlim, ylim, linewidth=3, c='black')

    def __call__(self, ax: plt.Axes, sciplot: SciPlotter = None):
        """
        Args:
            sciplot:
        """
        if isinstance(self.err1, np.ndarray):
            ax.errorbar(self.xy1[0], self.xy1[1], fmt='o', yerr=self.err1)
        else:
            ax.scatter(self.xy1[0], self.xy1[1], self.s1, self.c1, self.marker1, *self.args, **self.kwargs)

        if self.xy2 is not None:
            if isinstance(self.err2, np.ndarray):
                ax.errorbar(self.xy2[0], self.xy2[1], yerr=self.err2, fmt='o')
            else:
                ax.scatter(self.xy2[0], self.xy2[1], self.s2, self.c2, self.marker2, *self.args, **self.kwargs)

        ax.set_xlabel(f'Target{self.unit}')
        ax.set_ylabel(f'Predicted{self.unit}')

        ax.set_xlim(self.xy_lim[0], self.xy_lim[1])
        ax.set_ylim(self.xy_lim[0], self.xy_lim[1])

        self.add_diagonal(ax)

        if self.xy2 is None:
            SciPlotter.add_text(ax, 0.025, 0.875, r"$\mathdefault{R^2}$=" + f"{round(self.r2_1, 3)}", {'fontsize': 20})
        else:
            SciPlotter.add_text(ax, 0.025, 0.850, r"train $\mathdefault{R^2}$=" + f"{round(self.r2_1, 3)}", {'fontsize': 20})
            SciPlotter.add_text(ax, 0.025, 0.775, r"test $\mathdefault{R^2}$=" + f"{round(self.r2_2, 3)}", {'fontsize': 20})

        ax.legend(['train set', 'test set'], loc='lower right', fontsize='xx-large')


class FeatureImportance:
    def __init__(self, feature_name, imp):
        self.feature_name = feature_name
        self.imp = imp

    def __call__(self, ax: plt.Axes, sciplot: SciPlotter = None):
        ax.bar(self.feature_name, self.imp)
        sciplot.kwargs['xticklabels_rotation'] = sciplot.kwargs.get('xticklabels_rotation', 60)

        ax.set_xlabel('Features No.')
        ax.set_ylabel(r'Importance')


class HierarchicalTree(Plot):
    def __init__(self, x: np.ndarray, xlabels: list[str], color_threshold=0.6):
        if x.shape[1] != len(xlabels):
            raise AssertionError('the row counts of "x" should be equal to length of "xlabels"')

        self.x = x
        self.xlabels = np.array(xlabels)
        self.threshold = color_threshold

    def __call__(self, ax: plt.Axes, sciplot: SciPlotter = None):
        abs_correlation_mat = np.abs(np.corrcoef(self.x.T))

        # Perform hierarchical clustering
        Z = linkage(abs_correlation_mat, 'average')

        dendrogram(Z, ax=ax, orientation='top', labels=self.xlabels, color_threshold=self.threshold)

        # adjust the line width of the tree
        for collection in ax.collections:
            collection.set_linewidth(1.0)

        for label in ax.get_xticklabels():
            label.set_rotation(sciplot.kwargs.get('xticklabels_rotation', 60))


class Hist(Plot):
    def __init__(self, x, bins=None, range=None, density=False, weights=None, cumulative=False, bottom=None,
                 histtype='bar', align='mid', orientation='vertical', rwidth=None, log=False, color='sandybrown',
                 label=None, stacked=False, *, data=None, **kwargs):
        """"""
        kwargs.update(locals())
        kwargs.pop('self')
        kwargs.pop('kwargs')
        self.kwargs = kwargs

    def __call__(self, ax: plt.axes, sciplot: SciPlotter = None):
        n, bins, patches = ax.hist(**self.kwargs)

        ax.set_xlabel(r'Capacity $\mathdefault{(mmol CO_2/g)}$')
        ax.set_ylabel(r'Counts')

        cum_n = np.concatenate([[0.], np.cumsum(n) / n.sum() * 100])
        ax_twin = ax.twinx()
        ax_twin.plot(bins, cum_n, 'b', linewidth=2.0)
        ax_twin.set_ylabel('Cumulative(%)')

        return ax_twin


class Pearson(Plot):
    """ Performing univariate analysis and calculate the Pearson coeffiecient """
    def __init__(self, xy, xlabel, ylabel):
        self.xy = xy
        self.c = SciPlotter.calculate_scatter_density(xy)

        self.xlabel = xlabel
        self.ylabel = ylabel

    @staticmethod
    def fit_line(x: np.ndarray, y: np.ndarray, ax: plt.Axes):
        reg = linear_model.LinearRegression()
        reg.fit(x.reshape(-1, 1), y)
        low, high = ax.get_xlim()
        lin_space = np.linspace(low, high)
        return lin_space, reg.predict(lin_space.reshape(-1, 1))

    def add_line(self, ax: plt.Axes):
        line_x, line_y = self.fit_line(self.xy[0], self.xy[1], ax)
        ax.plot(line_x, line_y, linewidth=2.0, color="black")

    def __call__(self, ax: plt.Axes, sciplot: SciPlotter = None):
        ax.scatter(self.xy[0], self.xy[1], c=self.c)
        coeff = r_regression(self.xy[0].reshape(-1, 1), self.xy[1])[0]

        self.add_line(ax)

        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)

        SciPlotter.add_text(ax, 0.125, 0.900, f"Pearson R={round(coeff, 3)}", {'fontsize': 48})


class PearsonMatrix(Plot):
    def __init__(self, x: np.ndarray, xlabels: list[str], show_values=False, num_round=3, colorbar=True):
        if x.shape[1] != len(xlabels):
            raise AssertionError('the row counts of "x" should be equal to length of "xlabels"')

        self.x = x
        self.xlabels = xlabels
        self.round = num_round
        self.show_values = show_values
        self.colorbar = colorbar

    def __call__(self, ax: plt.Axes, sciplot: SciPlotter = None):
        p_mat = np.corrcoef(self.x.T)
        mappable = ax.imshow(p_mat, cmap='coolwarm')

        ax.xaxis.set_major_locator(plt.MultipleLocator())
        ax.yaxis.set_major_locator(plt.MultipleLocator())

        # Design the xy ticks
        if self.x.shape[1] == len(self.xlabels) <= 15:
            ax.xaxis.set_ticks(range(len(self.xlabels)), self.xlabels)
            ax.yaxis.set_ticks(range(len(self.xlabels)), self.xlabels)
        elif self.x.shape[1] <= 15:
            ax.xaxis.set_ticks(range(len(self.xlabels)), len(self.xlabels))
            ax.yaxis.set_ticks(range(len(self.xlabels)), len(self.xlabels))
        else:
            ax.xaxis.set_ticks(range(0, len(self.xlabels), 5), range(0, len(self.xlabels), 5))
            ax.yaxis.set_ticks(range(0, len(self.xlabels), 5), range(0, len(self.xlabels), 5))

        # Add the pearson coefficient for each image block
        if self.show_values == True:
            for i, j in itertools.product(range(self.x.shape[1]), range(self.x.shape[1])):
                ax.text(i, j, round(p_mat[i, j], self.round), ha='center', va='center',
                        color='w' if p_mat[i, j] < 0.5 else 'b')

        if self.colorbar:
            sciplot.ax_modifier[ax] = SciPlotter.axes_modifier_container(
                (sciplot.add_axes_colorbar, {'mappable': mappable}))


class SHAPlot(Plot):
    """ this class is used to make plots for analyzing the SHAP results """
    _plot_type = Literal['bar', 'beeswarm', 'scatter', 'waterfull']

    def __init__(self, exp: shap.Explanation, plot_type: _plot_type = 'bar', *args, **kwargs):
        """"""
        self.exp = exp
        self.plot_type = plot_type
        self.args = args
        self.kwargs = kwargs

    def __call__(self, ax: plt.Axes, sciplot: SciPlotter = None):
        """
        Args:
            sciplot:
        """
        plt.sca(ax)  # Set current Axes to the given

        fig = plt.gcf()

        fig_params = {'figwidth': fig.get_figwidth(), 'figheight': fig.get_figheight(), 'dpi':  fig.get_dpi()}

        if self.plot_type == 'bar':
            self._bar(ax, sciplot)

        elif self.plot_type == 'beeswarm':
            self._beeswarm(ax, sciplot)

        fig.set(**fig_params)  # recover the attributes of whole figure.

    def _bar(self, ax: plt.Axes, sciplot: SciPlotter):
        """ make the SHAP bar plot """
        shap.plots.bar(self.exp, *self.args, **self.kwargs, show=False)

        for bar_value_text in ax.texts:
            bar_value_text.set(font='Arial', fontsize=20, fontweight='bold')

        for yticklabels in ax.get_yticklabels():
            yticklabels.set(color='black', alpha=1.0)

        pos = copy(sciplot.axes_position)
        pos[0] = 0.18
        sciplot.set_axes_position(ax, pos=sciplot.calc_subaxes_pos(ax, *pos))
        sciplot.kwargs['not_set_axes_position'] = True

    def _beeswarm(self, ax: plt.Axes, sciplot: SciPlotter):
        """ make the SHAP beeswarm plot """
        shap.plots.beeswarm(self.exp, *self.args, **self.kwargs, show=False)

        # Setting the colorbar
        fig = ax.figure
        colorbar = [a for a in fig.get_axes() if a.get_label() == '<colorbar>'][0]

        # adjust the position for colorbar
        pos = copy(sciplot.axes_position)
        pos[0], pos[2] = 0.90, 0.05
        sciplot.insert_other_axes_into(ax, colorbar, pos)

        # adjust the colorbar labels
        colorbar.set_ylabel('Feature Value (Normalized)',
                            fontdict={'font': "Arial", 'fontsize': 20, 'fontweight': 'bold'})

        # adjust the ticklabel of colorbar
        for ticklabel in colorbar.get_yticklabels():
            ticklabel.set(font='Arial', fontsize=16)

    @staticmethod
    def add_superscript(fig: plt.Figure, axs: np.ndarray):
        def autowrap(text: str, max_len=12):
            """ autowrap when the label is too long """
            new_label = ''
            line_len = 0
            for s in text.split():
                if len(s) > max_len and line_len:
                    new_label = f'\n{s}\n'
                    line_len = 0

                elif line_len + len(s) > max_len:
                    new_label += f'\n{s}'
                    line_len = len(s)

                elif line_len:
                    new_label += f' {s}'
                    line_len += len(s) + 1
                else:
                    new_label += s
                    line_len = len(s)

            return new_label

        marcher = re.compile(r'Sum of \d+ other features')

        x_frac, y_frac = -0.05, 1.05  # the superscript of Axes
        for i, ax in enumerate(axs.flatten()):
            SciPlotter.add_text(ax, x_frac, y_frac, string.ascii_lowercase[i], SciPlotter.superscript_dict)

            if ax.get_xlabel() == 'SHAP value (impact on model output)':
                ax.set_xlabel('SHAP value', fontdict={'fontsize': 28})
            else:
                ax.set_xlabel(ax.get_xlabel(), fontdict={'fontsize': 28})

            ticks = []
            for tick_label in ax.xaxis.get_ticklabels():
                if marcher.fullmatch(tick_label.get_text()):
                    tick_label.set(text=f'other {tick_label.get_text().split()[2]}', fontsize=22)
                else:
                    tick_label.set(text=autowrap(tick_label.get_text()), fontsize=22)
                ticks.append(tick_label)
            ax.set_xticklabels(ticks)

            ticks = []
            for tick_label in ax.yaxis.get_ticklabels():
                if marcher.fullmatch(tick_label.get_text()):
                    tick_label.set(text=f'other {tick_label.get_text().split()[2]}', fontsize=22, rotation=45)
                else:
                    tick_label.set(text=autowrap(tick_label.get_text()), fontsize=22, rotation=45)

                ticks.append(tick_label)
            ax.set_yticklabels(ticks)
