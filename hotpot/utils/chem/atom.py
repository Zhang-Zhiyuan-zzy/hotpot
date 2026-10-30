from hotpot.cheminfo.rules.chemistry.elements import _fill_electron_shell


def calc_electron_config(_atomic_number):
    n, l, conf = _fill_electron_shell(_atomic_number)
    return n, l, conf + [0] * (5 - len(conf))
