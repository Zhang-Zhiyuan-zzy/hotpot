"""Immutable, field-selective bonding inputs and complete aromatic proposals."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Tuple, Union

from ._source_identity import _SourceIdentity
from .stereo import StereoPayload


__all__ = ()


@dataclass(frozen=True)
class _BondingSnapshot:
    source: _SourceIdentity
    atom_uids: tuple[int, ...]
    atomic_numbers: tuple[int, ...]
    implicit_hydrogens: tuple[int | None, ...]
    aromatic: tuple[bool, ...]
    bonds: tuple[tuple[int, int, int, float, str, bool], ...]
    coordinates: tuple[tuple[float, float, float], ...]
    rings: tuple[tuple[int, ...], ...]
    ring_has_3d: tuple[bool, ...]

    def __post_init__(self):
        for field in ("atom_uids", "atomic_numbers", "implicit_hydrogens", "aromatic", "ring_has_3d"):
            object.__setattr__(self, field, tuple(getattr(self, field)))
        for field in ("bonds", "coordinates", "rings"):
            object.__setattr__(self, field, tuple(tuple(row) for row in getattr(self, field)))


@dataclass(frozen=True)
class _AromaticAssignment:
    source: _SourceIdentity
    atom_uids: tuple[int, ...]
    bond_uids: tuple[int, ...]
    atom_aromatic: tuple[bool, ...]
    bond_aromatic: tuple[bool, ...]
    bond_orders: tuple[float, ...]
    bond_kinds: tuple[str, ...]
    covered_rows: tuple[int, ...]
    method: str

    def __post_init__(self):
        for field in ("atom_uids", "bond_uids", "atom_aromatic", "bond_aromatic", "bond_orders", "bond_kinds", "covered_rows"):
            object.__setattr__(self, field, tuple(getattr(self, field)))


@dataclass(frozen=True)
class _StereoSnapshot:
    """One immutable capture of the raw facts consumed by stereo rules."""

    payload: StereoPayload
    atom_keys: tuple[int, ...]
    bonds: tuple[tuple[int, int, float, str, str | None], ...]


@dataclass(frozen=True)
class _Gate6Graph:
    """One immutable augmented represented-graph snapshot for Gate 6."""

    target_spec: Mapping
    directed_dative_edges: frozenset[tuple[int, int]]
    real_atom_count: int
    virtual_hydrogens: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class MolecularFactsIdentity:
    """Coordinate-independent identity of one chemical boundary."""

    chemical_identity_digest: 'str'


@dataclass(frozen=True)
class ComponentIdentity:
    """Stable component identity with indices in source atom order."""

    identity: 'str'
    atom_indices: 'Tuple[int, ...]'


@dataclass(frozen=True)
class HydrogenInventory:
    """Graph-H counts and stored site targets in current atom order.

    Ordinary nonmetal targets include graph H; consumers derive the non-graph
    residual. Metal and real-H raw-count exceptions follow the H convention.
    """

    explicit_counts: 'Tuple[Union[int, None], ...]'
    implicit_counts: 'Tuple[Union[int, None], ...]'


SemanticBondKind = Literal["covalent", "coordination", "unknown"]


@dataclass(frozen=True)
class SemanticBond:
    """One immutable bond fact without backend-owned objects."""

    atom_indices: 'Tuple[int, int]'
    order: 'Union[float, None]'
    aromatic: 'bool'
    kind: 'SemanticBondKind'


@dataclass(frozen=True)
class MolecularFacts:
    """Immutable chemical and optional geometric facts extracted from a molecule."""

    identity: 'MolecularFactsIdentity'
    geometry_digest: 'Union[str, None]'
    atomic_numbers: 'Tuple[int, ...]'
    mass_numbers: 'Tuple[Union[int, None], ...]'
    atom_map: 'Tuple[int, ...]'
    source_components: 'Tuple[ComponentIdentity, ...]'
    covalent_components: 'Tuple[ComponentIdentity, ...]'
    hydrogen_inventory: 'HydrogenInventory'
    semantic_bonds: 'Tuple[SemanticBond, ...]'
    coordinates: 'Union[Tuple[Tuple[float, float, float], ...], None]'
    nuclear_charge_sum: 'int'
