"""Read-only elemental classifications and existing electron-shell arithmetic."""

from hotpot.cheminfo.reference import elements as _reference


__all__ = ()


def _electron_configuration(atomic_number):
    return _reference._ELECTRON_CONFIGS[atomic_number]


def _fill_electron_shell(atomic_number):
    """Return the existing zero-based shell, occupied count and unpadded values."""
    shells = _reference._ATOMIC_ORBITAL
    n = 0
    l = 0
    conf = []
    while atomic_number > 0:
        if l >= len(shells[n]):
            n += 1
            l = 0
            conf = []
        conf.append(min(atomic_number, shells[n][l]))
        atomic_number -= shells[n][l]
        l += 1
    return n, l, conf


def _group(atomic_number):
    return _reference._Z_TO_GROUP[atomic_number]


def _is_metal(atomic_number):
    return atomic_number in _reference._METAL


def _is_lanthanide(atomic_number):
    return atomic_number in _reference._LANTHANIDES


def _is_actinide(atomic_number):
    return atomic_number in _reference._ACTINIDES


def _is_noble_gas(atomic_number):
    return atomic_number in _reference._NOBLE_GASES


def _is_halogen(atomic_number):
    return atomic_number in _reference._HALOGENS


def _mass(atomic_number):
    return _reference._MASS[atomic_number]


def _exact_mass(atomic_number):
    return _reference._EXACT_MASS[atomic_number]


def _electronegativity(atomic_number):
    return _reference._ELECTRONEGATIVITY[atomic_number]


def _covalent_radius(atomic_number):
    return _reference._COVALENT_RADII[atomic_number]


def _density(atomic_number):
    return _reference._DENSITY[atomic_number]


def _default_valence(atomic_number):
    return _reference._DEFAULT_VALENCE[atomic_number]
