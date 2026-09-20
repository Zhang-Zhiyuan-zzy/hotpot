"""Structural contracts used by the isolated aromaticity implementation."""

from __future__ import annotations

from typing import Protocol, Sequence

from ..geometry import Cycle


class AtomLike(Protocol):
    atomic_number: int
    implicit_hydrogens: int
    is_aromatic: bool

    @property
    def heavy_neighbours(self) -> Sequence["AtomLike"]: ...

    @property
    def neigh_idx(self) -> Sequence[int]: ...

    @property
    def bonds(self) -> Sequence["BondLike"]: ...

    @property
    def sum_heavy_cov_orders(self) -> float: ...


class BondLike(Protocol):
    bond_order: float

    @property
    def atoms(self) -> Sequence[AtomLike]: ...


class RingLike(Protocol):
    _atoms: Sequence[AtomLike]
    _bonds: Sequence[BondLike]

    @property
    def is_aromatic(self) -> bool: ...

    @is_aromatic.setter
    def is_aromatic(self, value: bool) -> None: ...

    @property
    def has_3d(self) -> bool: ...

    @property
    def has_metal(self) -> bool: ...

    @property
    def geometry_cycle(self) -> Cycle: ...


class JointRingLike(Protocol):
    @property
    def atoms(self) -> Sequence[AtomLike]: ...


class MoleculeLike(Protocol):
    @property
    def rings(self) -> Sequence[RingLike]: ...

    @property
    def ligand_rings(self) -> Sequence[RingLike]: ...

    def hide_metal_ligand_bonds(self, clear_conformers: bool = False) -> None: ...

    def recover_hided_metal_ligand_bonds(
            self,
            clear_conformers: bool = False,
    ) -> None: ...
