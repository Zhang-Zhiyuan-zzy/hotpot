"""
python v3.9.0
@Project: hotpot
@File   : general
@Auther : Zhiyuan Zhang
@Data   : 2024/10/19
@Time   : 9:23
"""

from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

from .. import utils
from ..plotter import SciPlotter
from ..defaults import Settings
from .base import Plot


__all__ = ['Hist']


class Hist(Plot):
    def __init__(self, x, x_name, bins=None, range=None, density=False, weights=None, cumulative=False, bottom=None,
                 histtype='bar', align='mid', orientation='vertical', rwidth=None, log=False, color='#F5CD5E',
                 label=None, stacked=False, edgecolor='black', linewidth=1., *, data=None, **kwargs):
        """"""
        kwargs.update(locals())
        kwargs.pop('self')
        kwargs.pop('kwargs')
        self.x_name = kwargs.pop('x_name')
        self.kwargs = kwargs

    def __call__(self, ax: plt.axes, sciplot: SciPlotter = None):
        n, bins, patches = ax.hist(**self.kwargs)

        # Initialize distribution fitting line
        mu, std = norm.fit(self.kwargs['x'])
        x_min, x_max = ax.get_xlim()
        x_line = np.linspace(x_min, x_max, 100)
        y_line = norm.pdf(x_line, mu, std)

        ax.set_xlabel(f'{self.x_name}')

        if not self.kwargs.get('density', False):
            ax.set_ylabel(r'Counts')
            ax_densi = ax.twinx()
            ax_densi.set_ylabel('Distribution')
        else:
            ax.set_ylabel(r'Distribution')
            ax_densi = ax


        ax_densi.plot(x_line, y_line, 'k', linewidth=2)

        utils.add_text(ax, 0.01, 0.95, r'$\sigma=%.3f$' % std, fontdict=Settings.text_fontdict)
        utils.add_text(ax, 0.01, 0.90, r'$\mu=%.3f$' % mu, fontdict=Settings.text_fontdict)

        return ax_densi
