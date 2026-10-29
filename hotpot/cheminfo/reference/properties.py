"""Load the existing reference assets on demand, without carrier ownership."""

from functools import cache
import json
from pathlib import Path
from types import MappingProxyType


__all__ = ()


_ELEMENT_SYMBOLS = (
    "0",
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
)


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


def _atomic_number_for_symbol(symbol):
    """Translate the existing symbol spelling without loading a toolkit."""
    if not symbol.isascii():
        return 0
    if symbol in ("D", "T"):
        return 1
    try:
        return _ELEMENT_SYMBOLS.index(symbol[:1].upper() + symbol[1:])
    except ValueError:
        return 0


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
