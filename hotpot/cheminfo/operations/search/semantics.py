"""Carrier reads for the public SMARTS semantic convenience methods."""

from ...structure.bond_kind import BondKind
from ...structure.substructure import SmartsSemantics
from ...rules.chemistry.substructure import semantics as rules
from ..chemistry import _atom_non_graph_hydrogens


def rings_for(semantics, atom):
    rings = atom.mol.rings if semantics is SmartsSemantics.FULL_GRAPH else atom.mol.ligand_rings
    return tuple(ring for ring in rings if atom in ring)


def bonds_for(semantics, atom):
    if semantics is SmartsSemantics.FULL_GRAPH or atom.is_metal:
        return tuple(atom.bonds)
    return tuple(bond for bond in atom.bonds if not bond.is_metal_ligand_bond)


def degree(semantics, atom):
    return rules.degree(len(bonds_for(semantics, atom)))


def connectivity(semantics, atom):
    return rules.connectivity(semantics.degree(atom), _atom_non_graph_hydrogens(atom))


def valence(semantics, atom):
    if semantics is SmartsSemantics.FULL_GRAPH:
        return rules.valence(atom.sum_bond_orders, _atom_non_graph_hydrogens(atom))
    valence_kinds = {BondKind.SINGLE, BondKind.DOUBLE, BondKind.TRIPLE, BondKind.AROMATIC}
    return rules.ligand_valence(
        tuple(bond.bond_order for bond in semantics.bonds_for(atom) if bond.bond_kind in valence_kinds),
        _atom_non_graph_hydrogens(atom),
    )
