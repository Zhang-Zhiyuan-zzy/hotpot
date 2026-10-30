"""The retained ordered formula, count and mass projections."""

from collections import Counter

from ....reference.properties import _ELEMENT_SYMBOLS
from ..elements import _mass

__all__ = ()


def _element_counts(atomic_numbers):
    return Counter(_ELEMENT_SYMBOLS[number] for number in atomic_numbers)


def _formula(atomic_numbers):
    return ''.join(f'{element}{count}' for element, count in _element_counts(atomic_numbers).items())


def _weight(atomic_numbers):
    return sum(_mass(number) for number in atomic_numbers)
