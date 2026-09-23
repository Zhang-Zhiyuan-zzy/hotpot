"""Metal--ligand coordination topology and placement interfaces."""

from __future__ import annotations

from typing import Iterator, Optional, Tuple, TYPE_CHECKING

import networkx as nx

from .contracts import (
    CoordinationEnvironment,
    CoordinationGeometryResult,
)


if TYPE_CHECKING:
    from ..core import Atom, Molecule


__all__ = (
    "collect_coordination_environments",
    "prepare_coordination_geometry",
)


def _iter_metal_donor_pairs(
    mol: "Molecule",
) -> Iterator[Tuple["Atom", "Atom"]]:
    """Yield the metal and donor atoms of each explicit coordination bond."""
    for bond in mol.bonds:
        if not bond.is_metal_ligand_bond:
            continue
        if bond.atom1.is_metal:
            yield bond.atom1, bond.atom2
        else:
            yield bond.atom2, bond.atom1


def _require_explicit_complex(mol: "Molecule") -> None:
    """Require a metal center with at least one explicit metal--ligand bond."""
    if not mol.has_metal or not any(
        bond.is_metal_ligand_bond for bond in mol.bonds
    ):
        raise ValueError(
            "The complex workflow requires a molecule with at least one "
            "explicit metal-ligand bond"
        )


def collect_coordination_environments(
    mol: "Molecule",
) -> Tuple[CoordinationEnvironment, ...]:
    """Describe explicit metal--donor connectivity without assigning geometry."""
    metal_donor_pairs = tuple(_iter_metal_donor_pairs(mol))
    ligand_graph = mol.graph.copy()
    ligand_graph.remove_edges_from(
        (metal.idx, donor.idx) for metal, donor in metal_donor_pairs
    )
    component_by_atom = {}
    for component_index, nodes in enumerate(nx.connected_components(ligand_graph)):
        for atom_idx in nodes:
            component_by_atom[atom_idx] = component_index

    donors_by_metal = {metal.idx: [] for metal in mol.metals}
    for metal, donor in metal_donor_pairs:
        donors_by_metal[metal.idx].append(donor.idx)

    environments = []
    for metal in mol.metals:
        donors = sorted(donors_by_metal[metal.idx])
        grouped = {}
        for donor_idx in donors:
            grouped.setdefault(component_by_atom[donor_idx], []).append(donor_idx)
        environments.append(
            CoordinationEnvironment(
                metal_idx=metal.idx,
                donor_indices=tuple(donors),
                coordination_number=len(donors),
                metal_atomic_number=metal.atomic_number,
                metal_formal_charge=metal.formal_charge,
                donor_atomic_numbers=tuple(
                    mol.atoms[index].atomic_number for index in donors
                ),
                chelate_groups=tuple(
                    tuple(indices) for _, indices in sorted(grouped.items())
                ),
            )
        )
    return tuple(environments)


def prepare_coordination_geometry(
    mol: "Molecule",
    *,
    environments: Optional[Tuple[CoordinationEnvironment, ...]] = None,
    strategy: Optional[str] = None,
    seed: Optional[int] = None,
) -> CoordinationGeometryResult:
    """Reserved hook for coordination-number-aware initial placement."""
    _require_explicit_complex(mol)
    raise NotImplementedError(
        "Coordination-number-aware placement is reserved but not implemented"
    )
