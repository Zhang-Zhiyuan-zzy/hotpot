"""Value-only arithmetic for the named SMARTS semantic views."""

from ....structure.substructure import SmartsSemantics

def resolve_smarts_semantics(value) -> SmartsSemantics:
    """Return a semantics enum from its public enum or string representation."""

    if isinstance(value, SmartsSemantics):
        return value
    return SmartsSemantics(value)


def degree(bond_count):
    return bond_count


def connectivity(bond_count, implicit_hydrogens):
    return bond_count + implicit_hydrogens


def valence(bond_order_sum, implicit_hydrogens):
    return bond_order_sum + implicit_hydrogens


def ligand_valence(bond_orders, implicit_hydrogens):
    return sum(bond_orders) + implicit_hydrogens
