"""Reference ownership, legacy data-shape, and lazy leaf-module contracts."""

import importlib.util
import json
from pathlib import Path

import pytest

from hotpot.cheminfo import elements as element_module
from hotpot.cheminfo.reference import properties as reference


def _source_json(name):
    path = Path(reference.__file__).resolve().parents[1] / "ChemData" / name
    return json.loads(path.read_text(encoding="utf-8"))


def test_reference_tables_preserve_source_values_and_bond_key_mapping():
    assert reference._element_properties_snapshot() == _source_json(
        "ElementProperties.json"
    )["data"]

    expected = {}
    for key, value in _source_json("ideal_bond_length.json").items():
        first, second, order = json.loads(key)
        expected.setdefault(frozenset((first, second)), {})[order] = value
    assert reference._ideal_bond_lengths() == expected


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
    original_open = Path.open

    def record_open(path, *args, **kwargs):
        reads.append(path.name)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", record_open)
    spec = importlib.util.spec_from_file_location(
        "_hotpot_reference_leaf_probe", reference.__file__
    )
    leaf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(leaf)
    assert reads == []

    first = leaf._element_properties()
    assert leaf._element_properties() is first
    lengths = leaf._ideal_bond_lengths()
    assert leaf._ideal_bond_lengths() is lengths
    assert reads == ["ElementProperties.json", "ideal_bond_length.json"]
