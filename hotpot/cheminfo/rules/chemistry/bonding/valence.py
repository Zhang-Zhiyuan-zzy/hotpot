"""The existing Hotpot valence heuristic over explicit local facts."""

from ....reference.elements import _ATOMIC_ORBITAL
from ..elements import (
    _default_valence, _electron_configuration, _is_halogen, _is_metal, _is_noble_gas,
)

__all__ = ()


def _missing_electrons(atomic_number):
    configuration = _electron_configuration(atomic_number)
    return sum(_ATOMIC_ORBITAL[configuration[0]]) - sum(configuration[1:])


def _calculate_valence(atomic_number, neighbour_numbers, covalent_order):
    if atomic_number == 0:
        return 0
    if atomic_number in (6, 14):
        return 4
    if atomic_number == 8:
        return 2
    if atomic_number in (7, 15):
        oxygens = neighbour_numbers.count(8)
        return max(5, 2 * oxygens + 1) if oxygens else 3
    if atomic_number == 16:
        if 8 not in neighbour_numbers:
            return 2
        return 4 if covalent_order <= 4 else 6
    if atomic_number == 5:
        return 3
    if atomic_number == 1 or _is_halogen(atomic_number):
        return 1
    if _is_metal(atomic_number):
        return _default_valence(atomic_number)
    if _is_noble_gas(atomic_number):
        return 0
    return _default_valence(atomic_number)


def _hybridization(atomic_number, orders, aromatic, heavy_order):
    if atomic_number == 1:
        return 0
    if atomic_number in (6, 14):
        if aromatic or 2 in orders:
            return 2
        return 1 if 3 in orders else 3
    if atomic_number in (7, 15):
        return 2 if heavy_order == 3 else 3
    return 0


def _default_unpaired_electrons(atomic_numbers, charge):
    return (sum(atomic_numbers) - charge) % 2


def _default_spin_mult(atomic_numbers, charge):
    return _default_unpaired_electrons(atomic_numbers, charge) + 1


def _bonded_missing_electrons(element_missing, bond_order_sum):
    return element_missing - bond_order_sum


def _open_shell_electrons(configuration):
    return sum(configuration)


def _calculate_electron_config(atomic_number):
    from ..elements import _fill_electron_shell
    n, l, configuration = _fill_electron_shell(atomic_number)
    return n, l, tuple(configuration) + (0,) * (5 - len(configuration))


def _legacy_oxidation_order_sign(electronegativity, neighbour_electronegativity):
    return -1 if electronegativity > neighbour_electronegativity else 1


def _legacy_oxidation_contribution(sign, order):
    return sign * int(order)


def _legacy_oxidation_total(contributions, hydrogen_count, carbon_count):
    """Complete the existing local heuristic; this is not the state solver."""
    state = sum(contributions) + hydrogen_count
    if state > 0:
        return state + carbon_count
    if state < 0:
        return state - carbon_count
    return state
