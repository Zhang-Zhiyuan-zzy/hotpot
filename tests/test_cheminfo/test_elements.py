"""Scientific-data invariants for the periodic-element table."""

from math import isfinite

from hotpot.cheminfo import elements as element_module


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
