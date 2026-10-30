"""Immutable, field-selective bonding inputs and complete aromatic proposals."""

from dataclasses import dataclass

from ._identity import _SourceIdentity


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
