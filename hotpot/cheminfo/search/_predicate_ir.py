"""Immutable typed predicates emitted by Hotpot's SMARTS parser."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import FrozenSet, Optional, Tuple

from ..core import BondKind
from .semantics import SmartsSemantics


class _AtomField(str, Enum):
    ATOMIC_NUMBER = "atomic_number"
    FORMAL_CHARGE = "formal_charge"
    IS_AROMATIC = "is_aromatic"
    GROUP = "group"
    IS_LANTHANIDE = "is_lanthanide"
    IS_ACTINIDE = "is_actinide"
    IS_METAL = "is_metal"


class _AtomOp(str, Enum):
    TRUE = "true"
    FIELD_EQ = "field_eq"
    FIELD_IN = "field_in"
    PERIOD_IN = "period_in"
    DEGREE_EQ = "degree_eq"
    CONNECTIVITY_EQ = "connectivity_eq"
    VALENCE_EQ = "valence_eq"
    RING_ANY = "ring_any"
    RING_COUNT_EQ = "ring_count_eq"
    RING_NONE = "ring_none"
    RING_SIZE_EQ = "ring_size_eq"
    TOTAL_H_EQ = "total_h_eq"
    RECURSIVE = "recursive"
    AND = "and"
    OR = "or"
    NOT = "not"


@dataclass(frozen=True, slots=True, eq=False)
class _AtomExpression:
    op: _AtomOp
    description: str
    children: Tuple["_AtomExpression", ...] = ()
    field: Optional[_AtomField] = None
    value: object = None
    semantics: Optional[SmartsSemantics] = None
    aromatic_states: FrozenSet[bool] = frozenset((False, True))

    def __call__(self, atom: object) -> bool:
        if self.op is _AtomOp.TRUE:
            return True
        if self.op is _AtomOp.FIELD_EQ:
            return bool(getattr(atom, self.field.value) == self.value)
        if self.op is _AtomOp.FIELD_IN:
            return bool(getattr(atom, self.field.value) in self.value)
        if self.op is _AtomOp.PERIOD_IN:
            return _period_number(atom.atomic_number) in self.value
        if self.op is _AtomOp.DEGREE_EQ:
            return self.semantics.degree(atom) == self.value
        if self.op is _AtomOp.CONNECTIVITY_EQ:
            return self.semantics.connectivity(atom) == self.value
        if self.op is _AtomOp.VALENCE_EQ:
            return self.semantics.valence(atom) == self.value
        if self.op is _AtomOp.RING_ANY:
            return bool(self.semantics.rings_for(atom))
        if self.op is _AtomOp.RING_COUNT_EQ:
            return len(self.semantics.rings_for(atom)) == self.value
        if self.op is _AtomOp.RING_NONE:
            return not self.semantics.rings_for(atom)
        if self.op is _AtomOp.RING_SIZE_EQ:
            return any(
                len(ring) == self.value for ring in self.semantics.rings_for(atom)
            )
        if self.op is _AtomOp.TOTAL_H_EQ:
            return atom.explicit_hydrogens + atom.implicit_hydrogens == self.value
        if self.op is _AtomOp.RECURSIVE:
            return bool(self.value(atom))
        if self.op is _AtomOp.AND:
            return all(child(atom) for child in self.children)
        if self.op is _AtomOp.OR:
            return any(child(atom) for child in self.children)
        if self.op is _AtomOp.NOT:
            return not self.children[0](atom)
        raise ValueError(f"Unknown atom predicate operation: {self.op!r}")

    def __repr__(self) -> str:
        return self.description


class _BondOp(str, Enum):
    TRUE = "true"
    KIND_IS = "kind_is"
    AROMATIC = "aromatic"
    OR = "or"


@dataclass(frozen=True, slots=True, eq=False)
class _BondExpression:
    op: _BondOp
    description: str
    children: Tuple["_BondExpression", ...] = ()
    kind: Optional[BondKind] = None

    def __call__(self, bond: object) -> bool:
        if self.op is _BondOp.TRUE:
            return True
        if self.op is _BondOp.KIND_IS:
            return bond.bond_kind is self.kind
        if self.op is _BondOp.AROMATIC:
            return bool(bond.is_aromatic)
        if self.op is _BondOp.OR:
            return any(child(bond) for child in self.children)
        raise ValueError(f"Unknown bond predicate operation: {self.op!r}")

    def __repr__(self) -> str:
        return self.description


def _period_number(atomic_number: int) -> int:
    for period, upper_bound in enumerate((2, 10, 18, 36, 54, 86, 118), start=1):
        if atomic_number <= upper_bound:
            return period
    return 0
