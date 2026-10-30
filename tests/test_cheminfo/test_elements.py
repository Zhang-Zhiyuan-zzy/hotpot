"""Scientific-data invariants for the periodic-element table."""

from math import isfinite

import pytest

from hotpot.cheminfo import elements as element_module
from hotpot.cheminfo.reference import elements as element_reference
from hotpot.cheminfo.rules.chemistry import elements as element_rules
from hotpot.cheminfo.rules.chemistry.descriptors.graph_spectrum import (
    calc_electron_config as graph_electron_config,
)
from hotpot.utils.chem.atom import calc_electron_config


def test_element_property_table_covers_the_periodic_table_uniformly() -> None:
    properties = element_module.element_properties
    symbols = element_module.elements.symbols[1:]
    shared_fields = set.intersection(*(set(record) for record in properties.values()))

    assert len(properties) == 118
    assert tuple(properties) == symbols
    assert len(shared_fields) == 83
    assert {
        "atomic_number",
        "symbol",
        "atomic_weight",
        "electron_configuration",
        "ionization_energies",
    } <= shared_fields


def test_complete_properties_are_non_null_for_every_element() -> None:
    complete = element_module.PropertiesSheet().complete_values()._sheet

    assert set(complete.columns) == {
        "name",
        "symbol",
        "atomic_number",
        "atomic_weight",
        "alternate_names",
        "names_of_allotropes",
        "block",
        "period",
        "series",
        "electron_configuration",
        "discovery",
        "cas_number",
        "half_life",
        "lifetime",
        "quantum_numbers",
        "known_isotopes",
    }
    assert not complete.isna().any(axis=None)


def test_first_ionization_energy_missingness_is_explicit_and_bounded() -> None:
    symbols = element_module.elements.symbols[1:]
    energies = element_module.elements.ionization_energies
    missing = {
        symbol
        for symbol, values in zip(symbols, energies)
        if not values or values[0] is None
    }

    assert missing == {
        "Db",
        "Sg",
        "Bh",
        "Hs",
        "Mt",
        "Ds",
        "Rg",
        "Cn",
        "Nh",
        "Fl",
        "Mc",
        "Lv",
        "Ts",
        "Og",
    }
    assert all(
        isfinite(values[0]) and values[0] > 0.0
        for symbol, values in zip(symbols, energies)
        if symbol not in missing
    )


def test_element_reference_is_immutable_and_facade_snapshots_are_isolated() -> None:
    with pytest.raises(TypeError):
        element_reference._DEFAULT_VALENCE[6] = 9
    with pytest.raises(TypeError):
        element_reference._VALENCE_DICT[6]["stable"][0] = 9
    with pytest.raises(AttributeError):
        element_reference._METAL.add(6)

    facade = element_module.Element
    old_valence = facade.default_valence[6]
    old_stable = facade.valence_dict[6]["stable"][:]
    try:
        facade.default_valence[6] = 9
        facade.valence_dict[6]["stable"][0] = 9
        facade.metal.add(6)
        assert element_rules._default_valence(6) == 4
        assert element_reference._VALENCE_DICT[6]["stable"] == (4,)
        assert not element_rules._is_metal(6)
    finally:
        facade.default_valence[6] = old_valence
        facade.valence_dict[6]["stable"][:] = old_stable
        facade.metal.remove(6)


@pytest.mark.parametrize(
    "atomic_number,n,l,conf",
    [(0, 0, 0, []), (1, 0, 1, [1]), (2, 0, 1, [2]),
     (3, 1, 1, [1]), (29, 3, 2, [2, 9]),
     (118, 6, 4, [2, 14, 10, 6]), (168, 7, 5, [2, 18, 14, 10, 6])],
)
def test_existing_electron_projections_keep_shell_and_padding_conventions(
    atomic_number, n, l, conf,
) -> None:
    assert calc_electron_config(atomic_number) == (n, l, conf + [0] * (5 - len(conf)))
    assert graph_electron_config(atomic_number) == (n, conf + [0] * (4 - len(conf)))
    assert graph_electron_config(atomic_number, length=1) == (n, conf + [0] * (1 - len(conf)))


def test_elemental_electron_table_keeps_exceptional_configurations() -> None:
    assert element_rules._electron_configuration(24) == (4, 1, 0, 5, 0, 0)
    assert element_rules._electron_configuration(29) == (4, 1, 0, 10, 0, 0)
    assert element_rules._electron_configuration(46) == (5, 0, 0, 10, 0, 0)
    assert element_rules._electron_configuration(103) == (7, 2, 1, 0, 14, 0)
