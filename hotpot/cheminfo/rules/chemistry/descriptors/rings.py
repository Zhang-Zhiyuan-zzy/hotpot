"""Ring classifications over explicit size and bond-overlap counts."""


__all__ = ()


def _is_small_ring(size):
    return size <= 8


def _rings_are_joint(shared_bond_count):
    if shared_bond_count == 0:
        return False
    if shared_bond_count == 1:
        return True
    raise ArithmeticError("incorrect Ring")
