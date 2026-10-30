"""Finite, charge-aware hydrogen completion of represented local facts.

The result is the historical site-total-H target. None denotes missing required
input; a known unsupported valence convention adds no unspecified hydrogen.
"""

from ..elements import _is_metal

__all__ = ()


_DEFAULT_VALENCES = {
    5: {-1: (4,), 0: (3,), 1: (2,)},
    6: {-1: (3,), 0: (4,), 1: (3,)},
    7: {-1: (2,), 0: (3,), 1: (4,)},
    8: {-2: (0,), -1: (1,), 0: (2,), 1: (3,)},
    14: {-1: (3, 5), 0: (4,), 1: (3,)},
    15: {-1: (2, 4, 6), 0: (3, 5), 1: (4,)},
    33: {-1: (2, 4, 6), 0: (3, 5), 1: (4,)},
    **{number: {-2: (0,), -1: (1, 3, 5), 0: (2, 4, 6),
                1: (3, 5), 2: (4,)} for number in (16, 34, 52)},
    **{number: {-1: (0,), 0: (1,), 1: (2,)}
       for number in (9, 17, 35, 53, 85)},
}
_RADICAL_COMPLETION = frozenset((5, 6, 7, 8, 14, 9, 17, 35, 53, 85))
_COVALENT_VALENCE = {'single': 1, 'double': 2, 'triple': 3}


def _calculate_implicit_hydrogens(atomic_number, neighbour_numbers, orders, aromatic,
                                  formal_charge, bond_kinds, bond_aromatic, dative_roles,
                                  radical_electrons, bracket_hydrogens, no_implicit):
    graph_hydrogens = neighbour_numbers.count(1)
    declared = graph_hydrogens + bracket_hydrogens
    if atomic_number == 0:
        return None
    if _is_metal(atomic_number):
        return 0
    if atomic_number == 1:
        return graph_hydrogens
    if 0 in neighbour_numbers:
        return None
    if no_implicit:
        return declared
    if 'unknown' in bond_kinds:
        return None
    if any(kind == 'dative' and role is None
           for kind, role in zip(bond_kinds, dative_roles)):
        return None
    if radical_electrons < 0 or (radical_electrons and atomic_number not in _RADICAL_COMPLETION):
        return declared
    if aromatic:
        target = 3 if ((atomic_number, formal_charge) in ((6, 0), (14, 0), (5, -1))
                       or (atomic_number == 7 and formal_charge == 1 and not radical_electrons)) else None
        valences = () if target is None else (target,)
    else:
        valences = _DEFAULT_VALENCES.get(atomic_number, {}).get(formal_charge, ())
    occupied = bracket_hydrogens + radical_electrons
    for order, kind, edge_aromatic, role in zip(orders, bond_kinds, bond_aromatic, dative_roles):
        if kind == 'zero':
            continue
        if kind == 'dative':
            occupied += role == 'acceptor'
        elif aromatic and (edge_aromatic or kind == 'aromatic'):
            occupied += 1
        elif kind == 'aromatic':
            occupied += order
        else:
            occupied += _COVALENT_VALENCE[kind]
    if occupied != int(occupied):
        return declared
    target = next((valence for valence in valences if valence >= occupied), occupied)
    return declared + int(target - occupied)
