"""Shared, perception-free coordination graphs for SMARTS conformance tests."""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import hotpot as hp
from hotpot.cheminfo.core import BondKind


COORDINATION_FIXTURES = Path(__file__).parent / "fixtures" / "coordination"
REPOSITORY_INPUTS = Path(__file__).parents[1] / "input"


@dataclass(frozen=True)
class PerceptionCase:
    name: str
    filename: str
    bond_order: float
    bond_kind: BondKind
    donor_degree: int
    donor_implicit_hydrogens: int
    donor_connectivity: int
    donor_valence: int


PERCEPTION_CASES: Tuple[PerceptionCase, ...] = (
    PerceptionCase(
        name="mol2_single",
        filename="cu_trimethylamine_single.mol2",
        bond_order=1.0,
        bond_kind=BondKind.SINGLE,
        donor_degree=4,
        donor_implicit_hydrogens=0,
        donor_connectivity=4,
        donor_valence=4,
    ),
    PerceptionCase(
        name="mol2_zero",
        filename="cu_trimethylamine_zero.mol2",
        bond_order=0.0,
        bond_kind=BondKind.UNKNOWN,
        donor_degree=4,
        donor_implicit_hydrogens=0,
        donor_connectivity=4,
        donor_valence=3,
    ),
    PerceptionCase(
        name="mol2_unknown",
        filename="cu_trimethylamine_un.mol2",
        bond_order=0.0,
        bond_kind=BondKind.UNKNOWN,
        donor_degree=4,
        donor_implicit_hydrogens=0,
        donor_connectivity=4,
        donor_valence=3,
    ),
    PerceptionCase(
        name="mol2_not_connected",
        filename="cu_trimethylamine_nc.mol2",
        bond_order=0.0,
        bond_kind=BondKind.UNKNOWN,
        donor_degree=4,
        donor_implicit_hydrogens=0,
        donor_connectivity=4,
        donor_valence=3,
    ),
    PerceptionCase(
        name="sdf_single",
        filename="cu_trimethylamine_single.sdf",
        bond_order=1.0,
        bond_kind=BondKind.SINGLE,
        donor_degree=4,
        donor_implicit_hydrogens=1,
        donor_connectivity=5,
        donor_valence=5,
    ),
)


def build_graph(atom_specs, bonds):
    """Build a Hotpot graph directly, without invoking Open Babel perception."""

    molecule = hp.Molecule()
    for atomic_number, implicit_hydrogens in atom_specs:
        molecule.create_atom(
            atomic_number=atomic_number,
            implicit_hydrogens=implicit_hydrogens,
        )
    for begin, end, order in bonds:
        molecule.add_bond(begin, end, order)
    return molecule


def tertiary_amine(coordinate_order=None):
    atom_specs = [(7, 0), (6, 3), (6, 3), (6, 3)]
    bonds = [(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0)]
    if coordinate_order is not None:
        atom_specs.append((29, 0))
        bonds.append((0, 4, coordinate_order))
    return build_graph(atom_specs, bonds)


def metal_star(connectivity, bond_order=1.0):
    atom_specs = [(26, 0)] + [(7, 0)] * connectivity
    bonds = [(0, ligand, bond_order) for ligand in range(1, connectivity + 1)]
    return build_graph(atom_specs, bonds)


def ethylenediamine_chelate():
    # Cu-N-C-C-N-Cu is a five-membered full-graph cycle.  The ligand skeleton
    # is acyclic after the existing metal-ligand bonds are hidden.
    return build_graph(
        [(29, 0), (7, 1), (6, 2), (6, 2), (7, 1)],
        [
            (0, 1, 1.0),
            (1, 2, 1.0),
            (2, 3, 1.0),
            (3, 4, 1.0),
            (4, 0, 1.0),
        ],
    )


def metal_bound_ligand_ring():
    # A six-membered covalent ligand ring with the metal as a leaf node.
    return build_graph(
        [(29, 0), (7, 0)] + [(6, 1)] * 5,
        [
            (0, 1, 1.0),
            (1, 2, 1.0),
            (2, 3, 1.0),
            (3, 4, 1.0),
            (4, 5, 1.0),
            (5, 6, 1.0),
            (6, 1, 1.0),
        ],
    )


def matching_indices(molecule, smarts):
    return {
        atom_index
        for hit in molecule.search_substructure(smarts)
        for atom_index in hit.atom_indices
    }


def matching_atom_sets(molecule, smarts):
    return {frozenset(hit.atom_indices) for hit in molecule.search_substructure(smarts)}
