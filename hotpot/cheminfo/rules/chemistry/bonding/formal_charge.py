"""The retained default formal-charge heuristic, separate from declarations."""

from ..elements import _default_valence, _is_halogen, _is_metal
from ....reference.elements import _METALLOID_2ND

__all__ = ()


def _calculate_formal_charge(atomic_number, neighbour_numbers, covalent_order, symbol):
    if _is_metal(atomic_number) or atomic_number in _METALLOID_2ND:
        return _default_valence(atomic_number)
    if atomic_number in (6, 14):
        return 4
    if atomic_number == 8:
        return -2
    if atomic_number in (7, 15):
        oxygens = neighbour_numbers.count(8)
        return max(5, 2 * oxygens + 1) if oxygens else (-3 if atomic_number == 7 else 3)
    if atomic_number == 16:
        if 8 not in neighbour_numbers:
            return -2
        return 4 if covalent_order <= 4 else 6
    if atomic_number in (5, 33):
        return 3
    if atomic_number == 1:
        return 1
    if _is_halogen(atomic_number):
        return -1
    raise NotImplementedError(f"The formal charge of {symbol} is not implemented.")


def _sum_formal_charges(charges):
    return sum(int(charge) for charge in charges)


def _default_molecular_charge(components):
    return sum(explicit - implicit if organic else isolated
               for organic, explicit, implicit, isolated in components)
