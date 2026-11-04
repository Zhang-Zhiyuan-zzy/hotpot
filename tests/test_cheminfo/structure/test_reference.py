"""Reference ownership, legacy data-shape, and lazy leaf-module contracts."""

import importlib.util
from importlib import resources
import json

import pytest

from hotpot.cheminfo import elements as element_module
from hotpot.cheminfo.reference import properties as reference


def _source_json(name):
    resource = resources.files("hotpot.cheminfo.reference").joinpath("data", name)
    return json.loads(resource.read_text(encoding="utf-8"))


def test_reference_tables_preserve_source_values_and_bond_key_mapping():
    assert reference._element_properties_snapshot() == _source_json(
        "element_properties.json"
    )["data"]

    expected = {}
    for key, value in _source_json("ideal_bond_lengths.json").items():
        first, second, order = json.loads(key)
        expected.setdefault(frozenset((first, second)), {})[order] = value
    assert reference._ideal_bond_lengths() == expected


@pytest.mark.parametrize(
    "symbol,atomic_number,atomic_weight",
    (("H", 1, 1.008), ("C", 6, 12.011), ("O", 8, 15.999), ("Fe", 26, 55.845)),
)
def test_reference_elements_match_known_element_identities(
    symbol, atomic_number, atomic_weight
):
    """Fixed periodic-table observations are independent of JSON decoding."""
    properties = reference._element_properties_snapshot()[symbol]

    assert properties["symbol"] == [symbol, None]
    assert properties["atomic_number"] == [atomic_number, None]
    assert properties["atomic_weight"][0] == pytest.approx(atomic_weight, abs=0.001)


@pytest.mark.parametrize(
    "elements,order,expected_angstrom,tolerance_angstrom",
    (
        ((1, 1), 1, 0.74, 0.01),
        ((1, 6), 1, 1.09, 0.02),
        ((6, 6), 1, 1.54, 0.04),
        ((6, 6), 2, 1.34, 0.04),
        ((6, 6), 3, 1.20, 0.04),
        ((6, 8), 2, 1.21, 0.03),
    ),
)
def test_reference_lengths_match_characteristic_covalent_scales(
    elements, order, expected_angstrom, tolerance_angstrom
):
    """Approximate chemical scales check units and indexing, not universal lengths."""
    length = reference._ideal_bond_lengths()[frozenset(elements)][order]

    assert length == pytest.approx(expected_angstrom, abs=tolerance_angstrom)


def test_carbon_bond_lengths_decrease_with_bond_order():
    lengths = reference._ideal_bond_lengths()[frozenset((6, 6))]

    assert lengths[3] < lengths[2] < lengths[1]


@pytest.mark.parametrize(
    "current_name,retired_name",
    (
        ("element_properties.json", "ElementProperties.json"),
        ("ideal_bond_lengths.json", "ideal_bond_length.json"),
    ),
)
def test_reference_assets_have_one_packaged_owner(current_name, retired_name):
    assert resources.files("hotpot.cheminfo.reference").joinpath(
        "data", current_name
    ).is_file()
    assert not resources.files("hotpot.cheminfo").joinpath(
        "ChemData", retired_name
    ).is_file()


def test_public_element_table_keeps_dict_list_semantics_and_isolates_source():
    public = element_module.element_properties
    source = reference._element_properties()
    assert isinstance(public, dict)
    assert isinstance(public["H"], dict)
    assert isinstance(public["H"]["atomic_number"], list)

    original = public["H"]["atomic_number"][0]
    try:
        public["H"]["atomic_number"][0] = -1
        assert source["H"]["atomic_number"][0] == original
        assert reference._element_properties_snapshot()["H"]["atomic_number"][0] == original
    finally:
        public["H"]["atomic_number"][0] = original


def test_separate_public_snapshots_own_their_deeply_nested_lists():
    first = reference._element_properties_snapshot()
    second = reference._element_properties_snapshot()

    first["H"]["atomic_number"][0] = -1
    first["H"]["ionization_energies"][0].append(-1)

    assert second["H"]["atomic_number"] == [1, None]
    assert second["H"]["ionization_energies"] == [[1312], "kj/mol"]
    assert reference._element_properties()["H"]["ionization_energies"] == (
        (1312,), "kj/mol"
    )


def test_reference_sources_are_nested_read_only():
    elements = reference._element_properties()
    with pytest.raises(TypeError):
        elements["H"] = {}
    with pytest.raises(TypeError):
        elements["H"]["atomic_number"] = (0, "")
    with pytest.raises(TypeError):
        elements["H"]["atomic_number"][0] = 0

    lengths = reference._ideal_bond_lengths()
    with pytest.raises(TypeError):
        lengths[frozenset((1, 1))] = {}
    with pytest.raises(TypeError):
        lengths[frozenset((1, 1))][1] = 0


def test_fresh_reference_leaf_defers_and_reuses_json_reads(monkeypatch):
    reads = []
    owners = []
    original_files = resources.files

    class RecordedResource:
        def __init__(self, resource):
            self.resource = resource

        def joinpath(self, *parts):
            return RecordedResource(self.resource.joinpath(*parts))

        def open(self, *args, **kwargs):
            reads.append(self.resource.name)
            return self.resource.open(*args, **kwargs)

    def record_files(owner):
        owners.append(owner)
        return RecordedResource(original_files(owner))

    monkeypatch.setattr(resources, "files", record_files)
    spec = importlib.util.spec_from_file_location(
        "_hotpot_reference_leaf_probe", reference.__file__
    )
    leaf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(leaf)
    assert reads == []
    assert owners == []

    first = leaf._element_properties()
    assert leaf._element_properties() is first
    lengths = leaf._ideal_bond_lengths()
    assert leaf._ideal_bond_lengths() is lengths
    assert reads == ["element_properties.json", "ideal_bond_lengths.json"]
    assert owners == ["hotpot.cheminfo.reference", "hotpot.cheminfo.reference"]
