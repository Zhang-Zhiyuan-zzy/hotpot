"""Named chemical semantics for Hotpot SMARTS compilation."""

from __future__ import annotations

from enum import Enum

from ..core import BondKind


class SmartsSemantics(str, Enum):
    """Select the molecular view used by topology-sensitive SMARTS primitives."""

    FULL_GRAPH = "full_graph"
    LIGAND_SKELETON = "ligand_skeleton"

    def rings_for(self, atom):
        rings = (
            atom.mol.rings
            if self is SmartsSemantics.FULL_GRAPH
            else atom.mol.ligand_rings
        )
        return tuple(ring for ring in rings if atom in ring)

    def bonds_for(self, atom):
        if self is SmartsSemantics.FULL_GRAPH or atom.is_metal:
            return tuple(atom.bonds)
        return tuple(
            bond for bond in atom.bonds if not bond.is_metal_ligand_bond
        )

    def degree(self, atom) -> int:
        return len(self.bonds_for(atom))

    def connectivity(self, atom) -> int:
        return self.degree(atom) + atom.implicit_hydrogens

    def valence(self, atom) -> float:
        if self is SmartsSemantics.FULL_GRAPH:
            return atom.sum_bond_orders + atom.implicit_hydrogens
        valence_kinds = {
            BondKind.SINGLE,
            BondKind.DOUBLE,
            BondKind.TRIPLE,
            BondKind.AROMATIC,
        }
        return (
            sum(
                bond.bond_order
                for bond in self.bonds_for(atom)
                if bond.bond_kind in valence_kinds
            )
            + atom.implicit_hydrogens
        )


def resolve_smarts_semantics(value) -> SmartsSemantics:
    """Return a semantics enum from its public enum or string representation."""

    if isinstance(value, SmartsSemantics):
        return value
    return SmartsSemantics(value)

