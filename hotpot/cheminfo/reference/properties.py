"""Load the existing reference assets on demand, without carrier ownership."""

from functools import cache
import json
from pathlib import Path
from types import MappingProxyType


__all__ = ()


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _mutable_copy(value):
    if isinstance(value, MappingProxyType):
        return {key: _mutable_copy(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_mutable_copy(item) for item in value]
    return value


@cache
def _element_properties():
    """Return the one immutable parsed element-property source."""
    path = Path(__file__).resolve().parents[1] / "ChemData" / "ElementProperties.json"
    with path.open(encoding="utf-8") as source:
        return _freeze(json.load(source)["data"])


def _element_properties_snapshot():
    """Preserve the existing public dict/list table without exposing the source."""
    return _mutable_copy(_element_properties())


@cache
def _ideal_bond_lengths():
    """Return the immutable CSD reference indexed by element pair and order."""
    path = Path(__file__).resolve().parents[1] / "ChemData" / "ideal_bond_length.json"
    with path.open(encoding="utf-8") as source:
        raw_data = json.load(source)
    lengths = {}
    for key, value in raw_data.items():
        first, second, order = json.loads(key)
        lengths.setdefault(frozenset((first, second)), {})[order] = value
    return _freeze(lengths)
