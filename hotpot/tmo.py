"""
python v3.7.9
@Project: hotpot
@File   : tmo.py
@Author : Zhiyuan Zhang
@Date   : 2023/5/9
@Time   : 5:26
"""
from typing import Sequence
import thermo as tmo
import thermo.chemical as cmc


class Thermo:
    """ To determine thermodynamical properties of Csubstance """
    def __init__(self, mol, **kwargs):
        """"""
        self._data = {
            'mol': mol
        }

        self._data.update(kwargs)
        self._init_property()  # initialization

    def __dir__(self) -> Sequence[str]:
        d = set(self.__dict__.keys())
        chem = self.chem
        eos = self.eos

        if chem:
            d.update(dir(chem))
        if eos:
            d.update(dir(eos))

        return list(d)

    def __getattr__(self, item):
        chem = self._data.get('chem')  # Get chemical substance
        eos = self._data.get('eos_')  # Get equation of state

        if chem:
            attr = getattr(chem, item, None)
            if attr:
                return attr

        if eos:
            attr = getattr(eos, item, None)
            if attr:
                return attr

        return None

    def _init_property(self):
        # Retrieve the substance data by molecule smiles
        T = self._data.get('T')
        P = self._data.get('P')
        V = self._data.get('V')
        chem_kwargs = {}
        if T:
            chem_kwargs['T'] = T
        if P:
            chem_kwargs['P'] = P

        try:
            chem = cmc.Chemical(self.mol.inchi, **chem_kwargs)
            Tc = chem.Tc
            Pc = chem.Pc
            omega = chem.omega

            self._data['chem'] = chem

        except ValueError:
            Tc = self._data.get('Tc')
            Pc = self._data.get('Pc')
            omega = self._data.get('omega')

        # Retrieve the state of equation for thermo.eos module
        eos_name = self._data.get('eos', 'PR')
        eos = getattr(tmo.eos, 'PR')

        if eos and Tc and Pc and omega:
            # if the eos and the critical params are all be retrieved, build eos
            if T and P:
                self._data['eos_'] = eos(T=T, P=P, Tc=Tc, Pc=Pc, omega=omega)
            elif T and V:
                self._data['eos_'] = eos(T=T, V=V, Tc=Tc, Pc=Pc, omega=omega)
            elif P and T:
                self._data['eos_'] = eos(P=P, V=V, Tc=Tc, Pc=Pc, omega=omega)
            else:
                self._data['eos_'] = eos(T=298.15, P=101325, Tc=Tc, Pc=Pc, omega=omega)

    @property
    def chem(self):
        return self._data.get('chem')

    @property
    def eos(self):
        return self._data.get('eos_')

    @property
    def mol(self):
        return self._data.get('mol')

    @property
    def Tc(self):
        return self.chem.Tc

    @property
    def mu_l(self):
        """ TODO: chemical potential in certain state """
        return
