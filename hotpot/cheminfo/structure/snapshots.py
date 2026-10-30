"""Immutable, field-selective bonding inputs and complete aromatic proposals."""

from collections.abc import Mapping
from dataclasses import dataclass

from ._identity import _SourceIdentity
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
