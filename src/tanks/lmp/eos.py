"""
python v3.7.9
@Project: hotpot
@File   : eos.py
@Author : Zhiyuan Zhang
@Date   : 2023/4/27
@Time   : 8:08
Notes:
    PV = nRT
    Pv=RT
"""
from typing import *

import numpy as np
from scipy.optimize import fsolve

# Ideal gas constant
R = 8.3144626181532  # J/(mol.k)


def solve_ideal_gas_equation(
        p: Union[np.ndarray, float] = None, v: Union[np.ndarray, float] = None, t: Union[np.ndarray, float] = None
):
    if p is not None and t is not None and v is None:
        which = 'v'
    elif p is not None and v is not None and t is None:
        which = 't'
    elif t is not None and v is not None and p is None:
        which = 'p'
    else:
        raise ValueError('the two of p, t, v should be given')

    if which == 'v':
        if isinstance(p, float) and isinstance(t, float):
            return 'v', (R * t) / p
        elif isinstance(p, np.ndarray) and isinstance(t, np.ndarray):
            if len(p) == len(t):
                return 'v', (R * t) / p
            else:
                raise AttributeError('the given values of temperature and pressure should have same length')
        else:
            raise TypeError('the temperature(t) and pressure(p) should either be both float or both np.ndarray')

    if which == 'p':
        if isinstance(t, float) and isinstance(v, float):
            return 'p', (R * t) / v
        elif isinstance(t, np.ndarray) and isinstance(v, np.ndarray):
            if len(t) == len(v):
                return 'p', (R * t) / v
            else:
                raise AttributeError('the given values of temperature(t) and volume(v) should have same length')
        else:
            raise TypeError('the temperature(t) and volume(v) should either be both float or both np.ndarray')

    if which == 't':
        if isinstance(p, float) and isinstance(v, float):
            return 't', (p * v) / R
        elif isinstance(t, np.ndarray) and isinstance(v, np.ndarray):
            if len(t) == len(v):
                return 't', (p * v) / R
            else:
                raise AttributeError('the given values of volume(v) and pressure(p) should have same length')
        else:
            raise TypeError('the pressure(p) and volume(v) should either be both float or both np.ndarray')


class PengRobinson:
    """ Peng-Robinson equation """
    def __init__(self, tc, pc, w):
        """"""
        self.tc = tc
        self.pc = pc
        self.w = w
        self.r = 8.3144626181532

    def __call__(self, **kwargs):
        f: Callable = self._equations(**kwargs)
        x_name, x = solve_ideal_gas_equation(**kwargs)

        return x_name, fsolve(f, x0=x, args=(x_name,))

    def _equations(self, **kwargs):
        def PR_eqa(x, which):
            nonlocal p, t, v

            if which == 'v':
                v = x
            elif which == 't':
                t = x
            elif which == 'p':
                p = x
            else:
                raise ValueError('the two of p, t, v should be given')

            tc = self.tc
            pc = self.pc
            w = self.w
            tr = t / tc
            k = 0.37464 + (1.54226 * w) - 0.26992 * (w ** 2)

            a = 0.457235 * (((R ** 2) * (tc ** 2)) / pc)
            b = 0.077796 * ((R * tc) / pc)
            alpha = (1 + k * (1 - (tr ** 0.5))) ** 2

            return p - ((R * t) / (v - b)) + ((a * alpha) / ((v ** 2) + (2 * b * v) - (b ** 2)))

        p = kwargs.get('p')  # real pressure
        t = kwargs.get('t')  # temperature
        v = kwargs.get('v')  # molar volume

        return PR_eqa
