from typing import Union

import numpy as np
import pandas as pd

from .reference import elements as _reference
from .reference.properties import (
    _atomic_number_for_symbol, _element_properties_snapshot, _mutable_copy,
)


__all__ = [
    'elements',
    "element_properties",
    "PropertiesSheet"
]

# TODO: Lacking ionic coordination radii, refer to https://www.matbd.cn/sjjs/userInfo/checkDataSet?dataset_id=5b60fc21-db0b-48d2-8c3a-8d993910d535&template_id=57&type=home
element_properties = _element_properties_snapshot()
_shared_prop = {
    'cid_number', 'volume_magnetic_susceptibility', 'lattice_constant_c', 'electrical_type', 'critical_temperature',
    'speed_of_sound', 'electronegativity', 'electron_affinity', 'valence', 'space_group_number', 'alternate_names',
    'superconducting_point', 'electron_configuration', 'covalent_radius', 'critical_pressure', 'period', 'series',
    'neutron_cross_section', 'resistivity', 'percent_in_humans', 'symbol', 'atomic_weight', 'heat_of_fusion',
    'curie_point', 'ionization_energies', 'rtecs_number', 'lifetime', 'absolute_boiling_point', 'known_isotopes',
    'name', 'lattice_constant_a', 'atomic_radius', 'half_life', 'group', 'brinell_hardness', 'quantum_numbers',
    'melting_point', 'poisson_ratio', 'density_liquid', 'percent_in_universe', 'heat_of_vaporization',
    'dot_hazard_class', 'decay_mode', 'color', 'shear_modulus', 'dot_numbers', 'neutron_mass_absorption',
    'electrical_conductivity', 'mass_magnetic_susceptibility', 'percent_in_earth_crust', 'density', 'gas_phase',
    'refractive_index', 'bulk_modulus', 'percent_in_meteorites', 'nfpa_label', 'space_group_name',
    'molar_magnetic_susceptibility', 'adiabatic_index', 'thermal_expansion', 'percent_in_oceans', 'percent_in_sun',
    'cas_number', 'magnetic_type', 'lattice_constant_b', 'block', 'molar_volume', 'van_der_waals_radius',
    'phase', 'neel_point', 'names_of_allotropes', 'lattice_angles', 'absolute_melting_point', 'vickers_hardness',
    'mohs_hardness', 'specific_heat', 'crystal_structure', 'stable_isotopes', 'discovery', 'young_modulus',
    'boiling_point', 'thermal_conductivity', 'atomic_number'
}


class PropertiesSheet:
    def __init__(self, sheet: pd.DataFrame = None):
        if sheet is None:
            sheet = self._get_source_sheet()
        self._sheet = sheet

    def __getitem__(self, item):
        return self._sheet.loc[item]

    @staticmethod
    def _get_source_sheet():
        return pd.DataFrame(element_properties).T

    def get_by_properties(self, properties: Union[str, list]):
        return self._sheet.loc[properties, :]

    def get_by_elements(self, ele: Union[str, list]):
        return self._sheet.loc[:, ele]

    def _separate_value_units(self, index: int = 0):
        if index not in [0, 1]:
            raise ValueError('index must be 0 (value) or 1 (units)')
        prop_names = self._sheet.columns
        elem_names = self._sheet.index
        data = []
        for p in range(len(prop_names)):
            row_data = []
            for e in range(len(elem_names)):
                couple = self._sheet.iloc[e, p]
                if isinstance(couple, list):
                    assert len(couple) == 2, f'Error list of value {couple} for {elem_names[e]} and {prop_names[p]}'
                    row_data.append(self._sheet.iloc[e, p][index])
                else:
                    assert isinstance(couple, float)
                    assert np.isnan(couple), f'Error value {couple}, {type(couple)}'
                    row_data.append('NaN')
            data.append(row_data)
        return pd.DataFrame(data, columns=prop_names, index=elem_names)

    def drop_units(self):
        return self._separate_value_units(0)

    def get_units(self):
        return self._separate_value_units(1)

    def complete_values(self):
        complete_properties = []
        for p in self._sheet:
            if all(isinstance(v, list) and len(v) == 2 and v[0] is not None for v in self._sheet[p].tolist()):
                complete_properties.append(p)
        return PropertiesSheet(self._sheet[complete_properties])

    def number_values(self):
        number_properties = []
        for p in self._sheet:
            if all(
                isinstance(v, list) and len(v) == 2 and isinstance(v[0], (int, float))
                for v in self._sheet[p].tolist()
            ):
                number_properties.append(p)
        return PropertiesSheet(self._sheet[number_properties])


    def shared_sheet(self):
        return PropertiesSheet(self._sheet[list(_shared_prop)])

    def normalize_sheet(self):
        normalized_properties = []
        for prop_name in self._sheet:
            if all(
                (isinstance(v, list) and len(v) == 2) or (isinstance(v, float) and np.isnan(v))
                for v in self._sheet[prop_name].tolist()
            ):
                normalized_properties.append(prop_name)
        return PropertiesSheet(self._sheet[normalized_properties])


class Element:
    """
    Represents chemical elements, their symbols, atomic orbitals, and default valences.

    This class provides static data structures containing essential information about
    chemical elements such as symbols, electron configurations, and typical valences.
    The data supports scientific computation, referencing chemical properties, and periodic
    classifications.

    Attributes
    ----------
    symbols : tuple[str]
        A tuple where each index corresponds to the chemical element's atomic number. The
        tuple contains the symbol for every element in the periodic table. Empty strings are
        used in unfilled atomic number slots.
    atomic_orbital : list[list[int]]
        Nested list where each sublist represents electron configurations in orbitals for
        periodic rows. Each integer corresponds to the maximum number of electrons that fit
        in each orbital within a given energy level.
    default_valence : dict[int, int]
        A dictionary mapping atomic numbers (keys) of elements to their most commonly observed
        valence states (values). Inert elements are denoted by a valence of 0.
    """

    symbols = _reference._ELEMENT_SYMBOLS
    atomic_orbital = _mutable_copy(_reference._ATOMIC_ORBITAL)
    default_valence = _mutable_copy(_reference._DEFAULT_VALENCE)
    valence_dict = _mutable_copy(_reference._VALENCE_DICT)
    atomic_radii = _mutable_copy(_reference._ATOMIC_RADII)
    ionic_radii = _mutable_copy(_reference._IONIC_RADII)
    ionization_energies = _mutable_copy(_reference._IONIZATION_ENERGIES)
    element_densities = _mutable_copy(_reference._ELEMENT_DENSITIES)
    electronegativity = _mutable_copy(_reference._ELECTRONEGATIVITY)
    electron_configs = _reference._ELECTRON_CONFIGS
    Z_TO_GROUP = _mutable_copy(_reference._Z_TO_GROUP)
    alkali_metals = set(_reference._ALKALI_METALS)
    alkaline_earth_metals = set(_reference._ALKALINE_EARTH_METALS)
    transition_metals = set(_reference._TRANSITION_METALS)
    post_transition_metals = set(_reference._POST_TRANSITION_METALS)
    lanthanides = set(_reference._LANTHANIDES)
    actinides = set(_reference._ACTINIDES)
    metal = set(_reference._METAL)
    metalloid_1st = set(_reference._METALLOID_1ST)
    metalloid_2nd = set(_reference._METALLOID_2ND)
    metalloid = set(_reference._METALLOID)
    nonmetals = _mutable_copy(_reference._NONMETALS)
    metalloids = _mutable_copy(_reference._METALLOIDS)
    noble_gases = _mutable_copy(_reference._NOBLE_GASES)
    halogens = _mutable_copy(_reference._HALOGENS)
    covalent_radii = _mutable_copy(_reference._COVALENT_RADII)
    density = _mutable_copy(_reference._DENSITY)


    def __dir__(self):
        return list(super().__dir__()) + list(self._shared_prop)

    def __getattr__(self, item):
        try:
            super().__getattribute__(item)
        except AttributeError as e:
            if item in _shared_prop:
                return {sym: element_properties[sym][item] for sym in Element.symbols[1:]}
            raise e

    def __getitem__(self, item: Union[int, str]):
        if isinstance(item, str):
            item = _atomic_number_for_symbol(item)

        return {
            "atomic_number": item,
            "symbol": self.symbols[item],
            "default_valence": self.default_valence[item],
            "valence": self.valence_dict[item],
            'electronegativity': self.electronegativity[item],
            'covalent_radii': self.covalent_radii[item],
            'density': self.density[item],
        }

elements = Element()
