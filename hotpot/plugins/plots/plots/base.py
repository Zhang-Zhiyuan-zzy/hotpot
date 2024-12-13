"""
python v3.9.0
@Project: hotpot
@File   : base
@Auther : Zhiyuan Zhang
@Data   : 2024/10/19
@Time   : 9:17
"""
import matplotlib.pyplot as plt
from ..plotter import SciPlotter


__all__ = ['Plot']

class Plot:
    """"""
    def __call__(self, ax: plt.Axes, sciplot: SciPlotter = None):
        """
        Args:
            ax: the Axes object to be made Plots
            sciplot: the SciPlot object to make Plots
        """
        raise NotImplemented
