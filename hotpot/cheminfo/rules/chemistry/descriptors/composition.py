"""Composition predicates over explicit element and metal facts."""


__all__ = ()


def _organic_element_allowed(is_metal):
    return not is_metal


def _contains_carbon(atomic_numbers):
    return any(number == 6 for number in atomic_numbers)
