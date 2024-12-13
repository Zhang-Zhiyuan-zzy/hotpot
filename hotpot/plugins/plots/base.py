"""
python v3.9.0
@Project: hotpot
@File   : base
@Auther : Zhiyuan Zhang
@Data   : 2024/10/14
@Time   : 14:06
"""
from abc import abstractmethod, ABC
from typing import Callable, Literal
from matplotlib import pyplot as plt


class ImmutableDict(dict):
    def __setitem__(self, key, value):
        raise TypeError("This dictionary is immutable and cannot be modified.")

    def __delitem__(self, key):
        raise TypeError("This dictionary is immutable and cannot be modified.")

    def update(self, __m, **kwargs):
        raise TypeError("This dictionary is immutable and cannot be modified.")

    def popitem(self):
        raise TypeError("This dictionary is immutable and cannot be modified.")

    def clear(self):
        raise TypeError("This dictionary is immutable and cannot be modified.")

    def pop(self, __key):
        raise TypeError("This dictionary is immutable and cannot be modified.")


class Plotter:
    """ The base class for all plotters. """
    def __init__(self):
        """"""
        self._modifiers: ImmutableDict[str, list['FigAxesModifier']] = ImmutableDict({'fig': [], 'ax': []})

    def add_axes_modifier(self, which: Literal['fig', 'ax'], modifier: "FigAxesModifier"):
        """ Add the axes post-processing modifier. """
        self._modifiers[which].append(modifier)

    def post_process_axes(self) -> None:
        """ Post-processing of Axes by modifiers """
        for modifier in self._modifiers['fig']:
            modifier()
        for modifier in self._modifiers['ax']:
            modifier()


class Plot(ABC):
    """"""
    def __call__(self, ax: plt.Axes, sciplot: Plotter = None):
        """
        Args:
            ax: the Axes object to be made Plots
            sciplot: the SciPlot object to make Plots
        """
        raise NotImplemented


class FigAxesModifier(ABC):
    def __init__(self, plotter: Plotter, fig: plt.Figure, ax: plt.Axes, **kwargs):
        self.plotter = plotter
        self.fig = fig
        self.ax = ax
        self.kwargs = kwargs

    @abstractmethod
    def modify(self):
        raise NotImplemented

