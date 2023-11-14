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
import matplotlib.pyplot as plt


class SciPlot:
    """
    The base class to draw scientific plots, this class given method to:
        1) set the size of figure and the position of Axes
        2) set the font Properties for xy labels, ticks and insert text
        3) set the spline line width
    """
    _figwidth = 10.72  # inch
    _figheight = 8.205  # inch
    _dpi = 300

    default_axes_pos = [0.1483, 0.1657, 0.7163, 0.7185]
    _font = 'Arial'
    _splinewidth = 3
    _tick_fontsize = 28
    _xy_label_fontsize = 32

    _superscript_xy_frac = (0.025, 0.925)
    superscript_dict = {'font': _font, 'fontsize': 32, "fontweight": 'bold'}

    def __init__(self, plotters: Union[Callable, np.ndarray[Callable]], superscript: bool = True,
                 post_process: Callable = None, post_process_kwargs: dict = None):
        assert isinstance(plotters, np.ndarray) or isinstance(plotters, Callable)
        self.plotters = plotters if isinstance(plotters, np.ndarray) else np.array([[plotters]])
        self.nrows, self.ncols = self.plotters.shape

        fig, axs = plt.subplots(self.nrows, self.ncols)

        self.fig = fig
        if self.nrows == self.ncols == 1:
            self.axs = np.array([[axs]])
        elif self.nrows == 1 or self.ncols == 1:
            self.axs = np.array([axs])
        else:
            self.axs = axs

        self.figwidth = self._figwidth * self.ncols
        self.figheight = self._figheight * self.nrows

        self.fig.set(figwidth=self.figwidth, figheight=self.figheight, dpi=self._dpi)

        self.superscript = superscript

        self.post_process = post_process
        self.post_process_kwargs = post_process_kwargs if post_process_kwargs else {}

    def __call__(self, *args, **kwargs):
        """"""
        for i, (ax, plotter) in enumerate(zip(self.axs.flatten(), self.plotters.flatten())):
            ins_ax = plotter(ax)

            self.set_spline(ax, ins_ax)
            self.set_xylabel_font(ax, ins_ax)
            self.set_ticks(ax, ins_ax)
            self.set_axes_position(ax, ins_ax)

            if self.plotters.size > 1 and self.superscript:
                self.add_superscript(ax, i)

        if self.post_process:
            self.post_process(self.fig, self.axs, **self.post_process_kwargs)

        return self.fig, self.axs

    def add_superscript(self, ax, i):
        xfrac, yfrac = self._superscript_xy_frac

        self.add_text(ax, xfrac, yfrac, string.ascii_lowercase[i], self.superscript_dict)

    @staticmethod
    def add_text(ax, xfrac: float, yfrac: float, s: str, fontdict=None):
        (xb, xu), (yb, yu) = ax.get_xlim(), ax.get_ylim()
        x, y = (1 - xfrac) * xb + xfrac * xu, (1 - yfrac) * yb + yfrac * yu

        ax.text(x, y, s, fontdict=fontdict)

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
        left_boundary, bottom_boundary = SciPlot.calc_active_span(nrows, ncols, i, j)

        left = left_boundary + 1 / ncols * left
        bottom = bottom_boundary + 1 / nrows * bottom
        width = 1 / ncols * width
        height = 1 / nrows * height

        return left, bottom, width, height

    @classmethod
    def insert_other_axes_into(cls, main_ax: plt.Axes, insert_ax: plt.Axes, relative_pos: Union[tuple, np.ndarray]):
        """
        insert other Axes into the main Axes, given the given position relating to span of main Axes as the base.
        Args:
            main_ax: where the other Axes is inserted into
            insert_ax: the inserted Axes
            relative_pos: the relative position for the main Axes that the inserted Axes places. all values of
             the relative_pos is from 0 to 1.
        """
        cls.set_axes_position(insert_ax, pos=cls.calc_subaxes_pos(main_ax, *relative_pos))

    @classmethod
    def set_axes_position(cls, *axs, pos: Optional[Union[np.ndarray, Sequence]] = None):
        if not pos:
            pos = cls.calc_subaxes_pos(axs[0], *cls.default_axes_pos)

        for ax in axs:
            if isinstance(ax, plt.Axes):
                ax.set_position(pos)

    def set_xylabel_font(self, main_ax: plt.Axes, ins_ax: plt.Axes = None):
        """"""

        def to_set(ax):
            ax.xaxis.label.set_font(self._font)
            ax.xaxis.label.set_fontsize(self._xy_label_fontsize)
            ax.xaxis.label.set_fontweight('bold')

            ax.yaxis.label.set_font(self._font)
            ax.yaxis.label.set_fontsize(self._xy_label_fontsize)
            ax.yaxis.label.set_fontweight('bold')

        to_set(main_ax)
        if ins_ax:
            to_set(ins_ax)

    def set_spline(self, main_ax: plt.Axes, ins_ax: plt.Axes = None):
        for _, spline in main_ax.spines.items():
            spline.set_linewidth(self._splinewidth)

        if ins_ax:
            for _, spline in ins_ax.spines.items():
                spline.set_linewidth(self._splinewidth)

    def set_ticks(self, main_ax: plt.Axes, ins_ax: plt.Axes = None):
        def to_set(ax):
            ax.tick_params(width=self._splinewidth, length=self._splinewidth * 2)

            for tick in ax.xaxis.get_ticklabels():
                tick.set(font=self._font, fontsize=self._tick_fontsize)
            for tick in ax.yaxis.get_ticklabels():
                tick.set(font=self._font, fontsize=self._tick_fontsize)

        to_set(main_ax)
        if ins_ax:
            to_set(ins_ax)


class PlotTemplet:
    """"""
    def __call__(self, ax: plt.Axes):
        raise NotImplemented


class R2Regression(PlotTemplet):
    def __init__(self, xy1, xy2=None, unit: str = None, xy_lim: Union[Sequence, np.ndarray] = None,
                 s1=None, s2=None, c1=None, c2='green', marker1=None, marker2=None, *args, **kwargs):
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
        self.xy1 = xy1
        self.xy2 = xy2
        self.unit = f' ({unit})' if unit else ''
        if xy_lim:
            self.xy_lim = xy_lim
        elif xy2 is None:
            self.xy_lim = (self.xy1.min(), self.xy1.max())
        else:
            self.xy_lim = (min([self.xy1.min(), self.xy2.min()]), max([self.xy1.max(), self.xy2.max()]))

        self.s1 = s1
        self.s2 = s1 if not s2 else s1
        self.c1 = c1
        self.c2 = c2
        self.marker1 = marker1
        self.marker2 = marker2 if not marker2 else marker1

        self.args = args
        self.kwargs = {}

        self.r2_1 = r2_score(xy1[0], xy1[1])
        if self.xy2 is not None:
            self.r2_2 = r2_score(xy2[0], xy2[1])
            self.kwargs.update({'alpha': 0.3})
        else:
            self.c1 = SciPlot.calculate_scatter_density(xy1)

        self.kwargs.update(kwargs)

    @staticmethod
    def add_diagonal(ax: plt.Axes):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_autoscale_on(False)
        ax.plot(xlim, ylim, linewidth=3, c='black')

    def __call__(self, ax: plt.Axes):
        """"""
        ax.scatter(self.xy1[0], self.xy1[1], self.s1, self.c1, self.marker1, *self.args, **self.kwargs)
        if self.xy2 is not None:
            ax.scatter(self.xy2[0], self.xy2[1], self.s2, self.c2, self.marker2, *self.args, **self.kwargs)

        ax.set_xlabel(f'Target{self.unit}')
        ax.set_ylabel(f'Predicted{self.unit}')

        ax.set_xlim(self.xy_lim[0], self.xy_lim[1])
        ax.set_ylim(self.xy_lim[0], self.xy_lim[1])

        self.add_diagonal(ax)

        if self.xy2 is None:
            SciPlot.add_text(ax, 0.025, 0.875, r"$\mathdefault{R^2}$=" + f"{round(self.r2_1), 3}", {'fontsize': 20})
        else:
            SciPlot.add_text(ax, 0.025, 0.850, r"train $\mathdefault{R^2}$=" + f"{round(self.r2_1, 3)}", {'fontsize': 20})
            SciPlot.add_text(ax, 0.025, 0.775, r"test $\mathdefault{R^2}$=" + f"{round(self.r2_2, 3)}", {'fontsize': 20})

        ax.legend(['train set', 'test set'], loc='lower right', fontsize='xx-large')


class Hist(PlotTemplet):
    def __init__(self, x, bins=None, range=None, density=False, weights=None, cumulative=False, bottom=None,
                 histtype='bar', align='mid', orientation='vertical', rwidth=None, log=False, color='sandybrown',
                 label=None, stacked=False, *, data=None, **kwargs):
        """"""
        kwargs.update(locals())
        kwargs.pop('self')
        kwargs.pop('kwargs')
        self.kwargs = kwargs

    def __call__(self, ax: plt.axes):
        n, bins, patches = ax.hist(**self.kwargs)

        ax.set_xlabel(r'Capacity $\mathdefault{(mmol CO_2/g)}$')
        ax.set_ylabel(r'Counts')

        cum_n = np.concatenate([[0.], np.cumsum(n) / n.sum() * 100])
        ax_twin = ax.twinx()
        ax_twin.plot(bins, cum_n, 'b', linewidth=2.0)
        ax_twin.set_ylabel('Cumulative(%)')

        return ax_twin


class Pearson(PlotTemplet):
    """ Performing univariate analysis and calculate the Pearson coeffiecient """
    def __init__(self, xy, xlabel, ylabel):
        self.xy = xy
        self.c = SciPlot.calculate_scatter_density(xy)

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

    def __call__(self, ax: plt.Axes):
        ax.scatter(self.xy[0], self.xy[1], c=self.c)
        coeff = r_regression(self.xy[0].reshape(-1, 1), self.xy[1])[0]

        self.add_line(ax)

        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)

        SciPlot.add_text(ax, 0.125, 0.900, f"Pearson R={round(coeff, 3)}", {'fontsize': 48})


class SHAPlot(PlotTemplet):
    """ this class is used to make plots for analyzing the SHAP results """
    _plot_type = Literal['bar', 'beeswarm', 'scatter', 'waterfull']

    def __init__(self, exp: shap.Explanation, plot_type: _plot_type = 'bar', *args, **kwargs):
        """"""
        self.exp = exp
        self.plot_type = plot_type
        self.args = args
        self.kwargs = kwargs

    def __call__(self, ax: plt.Axes):
        """"""
        plt.sca(ax)  # Set current Axes to the given

        fig = plt.gcf()

        fig_params = {'figwidth': fig.get_figwidth(), 'figheight': fig.get_figheight(), 'dpi':  fig.get_dpi()}

        if self.plot_type == 'bar':
            self._bar(ax)

        elif self.plot_type == 'beeswarm':
            self._beeswarm(ax)

        fig.set(**fig_params)  # recover the attributes of whole figure.

    def _bar(self, ax: plt.Axes):
        """ make the SHAP bar plot """
        shap.plots.bar(self.exp, *self.args, **self.kwargs, show=False)

        for bar_value_text in ax.texts:
            bar_value_text.set(font='Arial', fontsize=20, fontweight='bold')

    def _beeswarm(self, ax: plt.Axes):
        """ make the SHAP beeswarm plot """
        shap.plots.beeswarm(self.exp, *self.args, **self.kwargs, show=False)

        # Setting the colorbar
        fig = ax.figure
        colorbar = [a for a in fig.get_axes() if a.get_label() == '<colorbar>'][0]

        # adjust the position for colorbar
        pos = copy(SciPlot.default_axes_pos)
        pos[0], pos[2] = 0.90, 0.05
        SciPlot.insert_other_axes_into(ax, colorbar, pos)

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
            SciPlot.add_text(ax, x_frac, y_frac, string.ascii_lowercase[i], SciPlot.superscript_dict)

            if ax.get_xlabel() == 'SHAP value (impact on model output)':
                ax.set_xlabel('SHAP value', fontdict={'fontsize': 28})
            else:
                ax.set_xlabel(ax.get_xlabel(), fontdict={'fontsize': 28})

            ticks = []
            for tick in ax.xaxis.get_ticklabels():
                if marcher.fullmatch(tick.get_text()):
                    tick.set(text=f'other {tick.get_text().split()[2]}', fontsize=22)
                else:
                    tick.set(text=autowrap(tick.get_text()), fontsize=22)
                ticks.append(tick)
            ax.set_xticklabels(ticks)

            ticks = []
            for tick in ax.yaxis.get_ticklabels():
                if marcher.fullmatch(tick.get_text()):
                    tick.set(text=f'other {tick.get_text().split()[2]}', fontsize=22)
                else:
                    tick.set(text=autowrap(tick.get_text()), fontsize=22)

                ticks.append(tick)
            ax.set_yticklabels(ticks)




