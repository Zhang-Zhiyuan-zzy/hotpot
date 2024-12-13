"""
python v3.9.0
@Project: hotpot
@File   : ml
@Auther : Zhiyuan Zhang
@Data   : 2024/10/19
@Time   : 9:14
"""
import re
import string
import itertools
from copy import copy
from typing import *

import torch
import numpy as np
import shap
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.feature_selection import r_regression
from sklearn import linear_model
from scipy.stats import norm
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.interpolate import griddata
from sklearn.manifold import TSNE

from matplotlib import pyplot as plt

from .base import Plot
from ..defaults import Settings
from ..plotter import SciPlotter, LegendOrganizer
from .. import utils


__all__ = [
    'R2Regression',
    'FeatureImportance',
    'HierarchicalTree',
    'PearsonMatrix',
    'Pearson',
    'SciPlotter',
    'EmbeddingDataTo2dMap'
]


class R2Regression(Plot):
    def __init__(
            self, xy1, xy2=None, unit: str = None, xy_lim: Union[Sequence, np.ndarray] = None,
            s1=None, s2=None, c1=None, c2=None, ch=None,
            marker1=None, marker2=None, err1=None, err2=None,
            show_mae=False, show_rmse=False,
            *args,
            sample1_name: str = None, sample2_name: str = None, sampleh_name: str = None,
            xy1_highlight_indices=None, xy2_highlight_indices=None,
            **kwargs
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
            show_mae: whether to show the MAE value in plot
            show_rmse: whether to show the RMSE value in plot
            *args: other args for Matplotlib.Axes.Scatter
            **kwargs: other kwargs for Matplotlib.Axes.Scatter
        """

        self.xy1 = xy1 if isinstance(xy1, np.ndarray) else np.array(xy1)
        self.xy2 = xy2 if (xy2 is None) or isinstance(xy2, np.ndarray) else np.array(xy2)
        self.sample1_name = ''
        self.sample2_name = ''
        self.sampleh_name = ''

        self.sample_num = len(self.xy1) + (len(self.xy2) if isinstance(self.xy2, np.ndarray) else 0)
        self.is_small_sample = self.sample_num < 1000

        self.unit = f' ({unit})' if unit else ''

        # Setting the Xy limiting
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
        self.s2 = s2 or s1
        self.c1 = None
        self.c2 = None
        self.ch = None  # color for highlight samples
        self.marker1 = marker1 or 'o'
        self.marker2 = marker2 or 'x'
        self.err1 = err1
        self.err2 = err2

        self.args = args

        self.kwargs = {}
        if self.is_small_sample:
            self.kwargs.update({'alpha': 1.0})
        else:
            self.kwargs.update({'alpha': 0.3})

        self.show_mae = show_mae
        self.show_rmse = show_rmse
        self.metrics = {}  # dict to store metrics

        self.xy_highlight = []
        self.xy1_highlight_indices = np.array(xy1_highlight_indices).tolist()
        self.xy2_highlight_indices = np.array(xy2_highlight_indices).tolist()
        self._article_names = []

        self.kwargs.update(kwargs)

        self._specify_sample_names(sample1_name, sample2_name, sampleh_name)
        self._organize_highlight_samples()
        self._calc_metrics()
        self._configure_colors(c1, c2, ch)

    def _specify_sample_names(self, sample1_name, sample2_name, sampleh_name):
        if isinstance(self.xy2, np.ndarray):
            self.sample1_name = 'train'
            self.sample2_name = 'test'

        if isinstance(self.xy_highlight, np.ndarray):
            self.sampleh_name = 'highlight'

        self.sample1_name = sample1_name or self.sample1_name
        self.sample2_name = sample2_name or self.sample2_name
        self.sampleh_name = sampleh_name or self.sampleh_name

    @staticmethod
    def add_diagonal(ax: plt.Axes):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_autoscale_on(False)
        ax.plot(xlim, ylim, linewidth=3, c='black')

    def _calc_metrics(self):
        if self.kwargs.pop('to_sparse', False):
            self.xy1[0], self.xy1[1] = utils.sample_density(self.xy1[0], self.xy1[1])

        self.metrics['r2_1'] = r2_score(self.xy1[0], self.xy1[1])
        self.metrics['mae_1'] = mean_absolute_error(self.xy1[0], self.xy1[1])
        self.metrics['rmse_1'] = np.sqrt(mean_squared_error(self.xy1[0], self.xy1[1]))

        if isinstance(self.xy2, np.ndarray):
            self.metrics['r2_2'] = r2_score(self.xy2[0], self.xy2[1])
            self.metrics['mae_2'] = mean_absolute_error(self.xy2[0], self.xy2[1])
            self.metrics['rmse_2'] = np.sqrt(mean_squared_error(self.xy2[0], self.xy2[1]))

        if isinstance(self.xy_highlight, np.ndarray):
            self.metrics['r2_h'] = r2_score(self.xy_highlight[0], self.xy_highlight[1])
            self.metrics['mae_h'] = mean_absolute_error(self.xy_highlight[0], self.xy_highlight[1])
            self.metrics['rmse_h'] = np.sqrt(mean_squared_error(self.xy_highlight[0], self.xy_highlight[1]))

    def add_metric_info(self, ax: plt.Axes):
        """ Add metric information """

        def _add(_text: str):
            nonlocal count
            utils.add_text(ax, 0.025, 0.925 - count*0.075, _text, Settings.text_fontdict)
            count += 1

        count = 0

        if self.xy2 is None:
            _add(r"$\mathdefault{R^2}$=" + f"{round(self.metrics['r2_1'], 3)}")
            if self.show_mae:
                _add(r"MAE=" + f"{round(self.metrics['mae_1'], 3)}")
            if self.show_rmse:
                _add(r"RMSE =" + f"{round(self.metrics['rmse_1'], 3)}")

        else:
            _add(r"train $\mathdefault{R^2}$=" + f"{round(self.metrics['r2_1'], 3)}")
            _add(r"test $\mathdefault{R^2}$=" + f"{round(self.metrics['r2_2'], 3)}")
            if self.show_mae:
                _add(r"test MAE=" + f"{round(self.metrics['mae_2'], 3)}")
            if self.show_rmse:
                _add(r"test RMSE=" + f"{round(self.metrics['rmse_2'], 3)}")

        if isinstance(self.xy_highlight, np.ndarray):
            _add(r"highlight MAE=" + f"{round(self.metrics['mae_h'], 3)}")
            _add(r"highlight RMSE=" + f"{round(self.metrics['rmse_h'], 3)}")

    def _organize_highlight_samples(self):
        if self.xy1_highlight_indices:
            self.xy_highlight.append(self.xy1[:, self.xy1_highlight_indices])
        if self.xy2_highlight_indices:
            self.xy_highlight.append(self.xy2[:, self.xy2_highlight_indices])

        try:
            self.xy_highlight = np.vstack(self.xy_highlight)
        except ValueError:
            self.xy_highlight = []

        if self.xy1_highlight_indices:
            self.xy1 = np.delete(self.xy1, self.xy1_highlight_indices, axis=1)
        if self.xy2_highlight_indices:
            self.xy2 = np.delete(self.xy2, self.xy2_highlight_indices, axis=1)

    def _configure_colors(self, c1=None, c2=None, ch=None):
        """"""
        if isinstance(self.xy_highlight, np.ndarray):
            self.ch = '#F4A666'

        if isinstance(self.xy2, np.ndarray):
            if isinstance(self.xy_highlight, np.ndarray):
                self.c1 = '#888888'
            else:
                self.c1 = '#3F54C7'

            self.c2 = '#6B0086'
        else:
            self.c1 = utils.calculate_scatter_density(self.xy1)

        self.c1 = c1 or self.c1
        self.c2 = c2 or self.c2
        self.ch = ch or self.ch

    def __call__(self, ax: plt.Axes, sciplot: SciPlotter = None):
        """
        Args:
            sciplot:
        """
        # if self.xy1_highlight_indices is not None and len(self.xy1_highlight_indices) > 0:
        #     xy1_highlight = self.xy1[:, self.xy1_highlight_indices]
        #     # ax.scatter(xy1_highlight[0], xy1_highlight[1], 120, c='#F4A666', marker='*')
        # if self.xy2_highlight_indices is not None and len(self.xy2_highlight_indices) > 0:
        #     xy2_highlight = self.xy2[:, self.xy2_highlight_indices]
        #     # ax.scatter(xy2_highlight[0], xy2_highlight[1], 120, c='#F4A666', marker='*')

        legend_organizer = LegendOrganizer()
        if isinstance(self.err1, np.ndarray):
            legend_organizer.add_plot('Train error bar', ax.errorbar, self.xy1[0], self.xy1[1], fmt='o', yerr=self.err1)
            # ax.errorbar(self.xy1[0], self.xy1[1], fmt='o', yerr=self.err1)
        else:
            legend_organizer.add_plot(
                'Train samples', ax.scatter,
                self.xy1[0], self.xy1[1], self.s1, self.c1, self.marker1, cmap='plasma', *self.args, **self.kwargs
            )
            # ax.scatter(
            #     self.xy1[0], self.xy1[1], self.s1, self.c1, self.marker1, cmap='plasma', *self.args, **self.kwargs)

        if self.xy2 is not None:
            if isinstance(self.err2, np.ndarray):
                legend_organizer.add_plot('Test error bar', ax.errorbar, self.xy2[0], self.xy2[1], fmt='o', yerr=self.err2)
                # ax.errorbar(self.xy2[0], self.xy2[1], yerr=self.err2, fmt='o')
            else:
                legend_organizer.add_plot(
                    'Test samples', ax.scatter,
                    self.xy2[0], self.xy2[1], self.s2, self.c2, self.marker2, *self.args, **self.kwargs
                )
                # ax.scatter(self.xy2[0], self.xy2[1], self.s2, self.c2, self.marker2, *self.args, **self.kwargs)

        if isinstance(self.xy_highlight, np.ndarray):
            legend_organizer.add_plot(
                'highlight', ax.scatter, self.xy_highlight[0], self.xy_highlight[1],
                200, c=self.ch, marker='*'
            )
            # ax.scatter(self.xy_highlight[0], self.xy_highlight[1], 200, c=self.ch, marker='*')

        ax.set_xlabel(f'Target{self.unit}')
        ax.set_ylabel(f'Predicted{self.unit}')

        ax.set_xlim(self.xy_lim[0], self.xy_lim[1])
        ax.set_ylim(self.xy_lim[0], self.xy_lim[1])

        self.add_diagonal(ax)
        self.add_metric_info(ax)

        legend_organizer.add_legend(ax, loc='lower right', fontsize='xx-large')
        # ax.legend(['train set', 'test set', 'highlight'], loc='lower right', fontsize='xx-large')


class FeatureImportance:
    def __init__(self, feature_name, imp, force_show_label=False):
        self.feature_name = feature_name
        self.imp = imp
        self.force_show_label = force_show_label

    def __call__(self, ax: plt.Axes, sciplot: SciPlotter = None):
        ax.bar(self.feature_name, self.imp)

        if not self.force_show_label and len(self.feature_name) > 15:
            xtick_locs = range(0, len(self.feature_name), 5)
            ax.set_xticks(xtick_locs)
            ax.set_xlabel('Features No.')
        else:
            # ax.set_xticklabels(self.feature_name)
            ax.set_xticks(self.feature_name)
            ax.set_xlabel('Feature name')

        ax.set_ylabel(r'Importance')

        sciplot.settings.xticklabels_rotation = sciplot.settings.xticklabels_rotation or 60


class HierarchicalTree(Plot):
    def __init__(self, x: np.ndarray, xlabels: Sequence[Union[str, str]], color_threshold=0.6):
        if x.shape[1] != len(xlabels):
            raise AssertionError('the row counts of "x" should be equal to length of "xlabels"')

        self.x = x
        self.xlabels = np.array(xlabels)
        self.threshold = color_threshold

    def __call__(self, ax: plt.Axes, sciplot: SciPlotter = None):
        abs_correlation_mat = np.abs(np.nan_to_num(np.corrcoef(self.x.T)))

        # Perform hierarchical clustering
        Z = linkage(abs_correlation_mat, 'average')

        dendrogram(Z, ax=ax, orientation='top', labels=self.xlabels, color_threshold=self.threshold)

        # adjust the line width of the tree
        for collection in ax.collections:
            collection.set_linewidth(2.0)

        for label in ax.get_xticklabels():
            degree = sciplot.settings.xticklabels_rotation or 60
            label.set_rotation(degree)


class Pearson(Plot):
    """ Performing univariate analysis and calculate the Pearson coeffiecient """
    def __init__(self, xy, xlabel, ylabel):
        self.xy = xy
        self.c = utils.calculate_scatter_density(xy)

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

        utils.add_text(ax, 0.125, 0.900, f"Pearson R={round(coeff, 3)}", {'fontsize': 48})


class PearsonMatrix(Plot):
    def __init__(
            self,
            x: np.ndarray,
            xlabels: list[str],
            is_matrix: bool = False,
            show_values=False,
            num_round=3,
            colorbar=True
    ):
        if x.shape[1] != len(xlabels):
            raise AssertionError('the row counts of "x" should be equal to length of "xlabels"')

        self.x = x
        self.xlabels = xlabels
        self.is_matrix = is_matrix
        self.round = num_round
        self.show_values = show_values
        self.colorbar = colorbar

    def __call__(self, ax: plt.Axes, sciplot: SciPlotter = None):
        if self.is_matrix:
            p_mat = self.x
        else:
            p_mat = np.nan_to_num(np.corrcoef(self.x.T))

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
                ax.text(i, j, str(round(float(p_mat[i, j]), self.round)), ha='center', va='center',
                        color='w' if p_mat[i, j] < 0.5 else 'b')

        if self.colorbar:
            sciplot.add_colorbar(ax, mappable, value_lim=(-1, 1), colorbar_label='Pearson Correlation')


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

        pos = copy(Settings.axes_pos)
        pos[0] = 0.18
        ax.set_position(pos=utils.abs2rela_pos(ax, *pos))
        sciplot.settings.not_set_axes_position = True

    def _beeswarm(self, ax: plt.Axes, sciplot: SciPlotter):
        """ make the SHAP beeswarm plot """
        shap.plots.beeswarm(self.exp, *self.args, **self.kwargs, show=False)

        for bee_value_text in ax.texts:
            bee_value_text.set(font='Arial', fontsize=20, fontweight='bold')

        # Setting the colorbar
        fig = ax.figure
        colorbar = [a for a in fig.get_axes() if a.get_label() == '<colorbar>'][0]

        # adjust the position for colorbar
        utils.insert_axes(ax, colorbar, utils.specify_colorbar_pos(ax))

        # adjust the colorbar labels
        colorbar.set_ylabel('Feature Value (Normalized)',
                            fontdict={'font': "Arial", 'fontsize': 20, 'fontweight': 'bold'})

        # adjust the ticklabel of colorbar
        for ticklabel in colorbar.get_yticklabels():
            ticklabel.set(font='Arial', fontsize=16)

    @staticmethod
    def add_superscript(fig: plt.Figure, axs: np.ndarray[plt.Axes]):

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
            utils.add_text(ax, x_frac, y_frac, string.ascii_lowercase[i], Settings.superscript_font_dict)

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


class EmbeddingDataTo2dMap(Plot):
    def __init__(
            self,
            X_train, X_test,
            y_train, y_test,
            p_train, p_test,
            embedding=TSNE(),
            feature_weight=None,
            color_data: Union[str, np.ndarray] = 'error'
    ):
        """

        Args:
            X_train:
            X_test:
            y_train:
            y_test:
            p_train:
            p_test:
            embedding:
            feature_weight:
            color_data:
                'error': the difference between true value and predicted value
                'true': the true value of simples
        """
        self.color_data = color_data

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.p_train = p_train
        self.p_test = p_test
        self.embedding = embedding
        self.feature_weight = feature_weight if feature_weight else np.ones(X_train.shape[1])

        self.total_X = np.vstack((X_train, X_test))
        self.total_y = np.hstack((y_train, y_test))

        self.weighted_X = self.total_X * self.feature_weight

        self.emb_X = embedding.fit_transform(self.weighted_X).T
        self.emb_Xtrain, self.emb_Xtest = self.emb_X[:, :len(X_train)], self.emb_X[:, len(X_train):]

    def __call__(self, ax: plt.Axes, *args, **kwargs):
        if self.color_data == 'error':
            error = self.p_test - self.y_test
            ax.scatter(self.emb_Xtrain[0], self.emb_Xtrain[1], c='#eeeeee', s=100)
            ax.scatter(self.emb_Xtest[0], self.emb_Xtest[1], c=error)

        if self.color_data == 'true':
            ax.scatter(self.emb_X[0], self.emb_X[1], c=self.total_y, s=100)
