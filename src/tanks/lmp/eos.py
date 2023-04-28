"""
python v3.7.9
@Project: hotpot
@File   : mc.py
@Author : Zhiyuan Zhang
@Date   : 2023/4/27
@Time   : 8:08
Notes:
    PV = nRT
    Pv=RT
"""
from typing import *
from scipy.optimize import fsolve


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
        x_name, x = self._solve_ideal_gas_equation(**kwargs)

        res = fsolve(f, x0=x, args=(x_name,))

        return res

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
            r = 8.3144626181532  # J/(mol.k)
            tr = t / tc
            k = 0.37464 + (1.54226 * w) - 0.26992 * (w ** 2)

            a = 0.457235 * (((r ** 2) * (tc ** 2)) / pc)
            b = 0.077796 * ((r * tc) / pc)
            alpha = (1 + k * (1 - (tr ** 0.5))) ** 2

            return p - ((r * t) / (v - b)) + ((a * alpha) / ((v ** 2) + (2 * b * v) - (b ** 2)))

        p = kwargs.get('p')  # real pressure
        t = kwargs.get('t')  # temperature
        v = kwargs.get('v')  # molar volume

        return PR_eqa

    def _solve_ideal_gas_equation(self, **kwargs):
        """ Given two of p v t, solve the rest one by ideal gas state equation: P(V/n)=RT """

        p = kwargs.get('p')  # real pressure
        t = kwargs.get('t')  # temperature
        v = kwargs.get('v')  # real molar volume

        if p and t and not v:
            which = 'v'
        elif p and v and not t:
            which = 't'
        elif t and v and not p:
            which = 'p'
        else:
            raise ValueError('the two of p, t, v should be given')

        v = 0.0224  # m^3/mol, ideal molar volume
        if which == 'v':
            return 'v', v
        if which == 'p':
            if not t:
                raise ValueError('the temperature(t) should give for the calculation of pressure')
            return 'p', (self.r*t)/v
        if which == 't':
            if not p:
                raise ValueError('the pressure(p) should give for the calculation of temperature')
            return 't', p*v/self.r


def peng_robinson(tc, pc, w, t=None, p=None, v=None):
    """"""
    r = 8.3144626181532  # J/(mol.k)



    def t_equation(x):
        """"""



    vm_iso = 0.0224  # m^3/mol, the molar volume of ideal gas in iso state

    # vm = (p/101325)*(298.15/t)*vm_iso
    vm = vm_iso
    tr = t/tc
    k = 0.37464 + (1.54226*w) - 0.26992*(w ** 2)

    a = 0.457235*(((r ** 2) * (tc ** 2)) / pc)
    b = 0.077796 * ((r * tc) / pc)
    alpha = (1 + k*(1-(tr ** 0.5))) ** 2

    fugacity = (r * t)/(vm - b) - ((a * alpha) / ((vm ** 2) + (2*b*vm) - (b ** 2)))

    return fugacity



