"""Named chemical semantics for Hotpot SMARTS compilation."""

from __future__ import annotations

from enum import Enum

from .bond_kind import BondKind
from dataclasses import dataclass
from typing import FrozenSet, Optional, Tuple


class SmartsSemantics(str, Enum):
    """Select the molecular view used by topology-sensitive SMARTS primitives."""

    FULL_GRAPH = "full_graph"
    LIGAND_SKELETON = "ligand_skeleton"

    def rings_for(self, atom):
        from ..operations.search.semantics import rings_for
        return rings_for(self, atom)

    def bonds_for(self, atom):
        from ..operations.search.semantics import bonds_for
        return bonds_for(self, atom)

    def degree(self, atom) -> int:
        from ..operations.search.semantics import degree
        return degree(self, atom)

    def connectivity(self, atom) -> int:
        from ..operations.search.semantics import connectivity
        return connectivity(self, atom)

    def valence(self, atom) -> float:
        from ..operations.search.semantics import valence
        return valence(self, atom)



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


@dataclass(frozen=True, eq=False)
class _AtomExpression:
    op: _AtomOp
    description: str
    children: Tuple["_AtomExpression", ...] = ()
    field: Optional[_AtomField] = None
    value: object = None
    semantics: Optional[SmartsSemantics] = None
    aromatic_states: FrozenSet[bool] = frozenset((False, True))

    def __call__(self, atom: object) -> bool:
        from ..operations.search.predicates import match_atom_expression
        return match_atom_expression(self, atom)

    def __repr__(self) -> str:
        return self.description


class _BondOp(str, Enum):
    TRUE = "true"
    KIND_IS = "kind_is"
    AROMATIC = "aromatic"
    OR = "or"


@dataclass(frozen=True, eq=False)
class _BondExpression:
    op: _BondOp
    description: str
    children: Tuple["_BondExpression", ...] = ()
    kind: Optional[BondKind] = None

    def __call__(self, bond: object) -> bool:
        from ..operations.search.predicates import match_bond_expression
        return match_bond_expression(self, bond)

    def __repr__(self) -> str:
        return self.description
