"""Read-only typed SMARTS comparisons over lazily supplied values."""

from ....structure.substructure import _AtomOp, _BondOp


def match_atom_expression(expression, values):
    op = expression.op
    if op is _AtomOp.TRUE:
        return True
    if op is _AtomOp.FIELD_EQ:
        return bool(values(expression) == expression.value)
    if op is _AtomOp.FIELD_IN:
        return bool(values(expression) in expression.value)
    if op is _AtomOp.PERIOD_IN:
        return _period_number(values(expression)) in expression.value
    if op in {_AtomOp.DEGREE_EQ, _AtomOp.CONNECTIVITY_EQ, _AtomOp.VALENCE_EQ}:
        return values(expression) == expression.value
    if op is _AtomOp.RING_ANY:
        return bool(values(expression))
    if op is _AtomOp.RING_COUNT_EQ:
        return values(expression) == expression.value
    if op is _AtomOp.RING_NONE:
        return not values(expression)
    if op is _AtomOp.RING_SIZE_EQ:
        return any(
            values(expression, index) == expression.value
            for index in range(values(expression))
        )
    if op is _AtomOp.TOTAL_H_EQ:
        explicit, implicit = values(expression)
        return explicit + implicit == expression.value
    if op is _AtomOp.RECURSIVE:
        return bool(values(expression))
    if op is _AtomOp.AND:
        return all(match_atom_expression(child, values) for child in expression.children)
    if op is _AtomOp.OR:
        return any(match_atom_expression(child, values) for child in expression.children)
    if op is _AtomOp.NOT:
        return not match_atom_expression(expression.children[0], values)
    raise ValueError(f"Unknown atom predicate operation: {op!r}")


def match_bond_expression(expression, values):
    op = expression.op
    if op is _BondOp.TRUE:
        return True
    if op is _BondOp.KIND_IS:
        return values(expression) is expression.kind
    if op is _BondOp.AROMATIC:
        return bool(values(expression))
    if op is _BondOp.OR:
        return any(match_bond_expression(child, values) for child in expression.children)
    raise ValueError(f"Unknown bond predicate operation: {op!r}")


def _period_number(atomic_number: int) -> int:
    for period, upper_bound in enumerate((2, 10, 18, 36, 54, 86, 118), start=1):
        if atomic_number <= upper_bound:
            return period
    return 0
