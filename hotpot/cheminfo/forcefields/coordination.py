"""Metal--ligand coordination topology and placement interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin, sqrt
from typing import Iterator, Literal, Optional, Sequence, Tuple, TYPE_CHECKING

import networkx as nx
import numpy as np

from .. import geometry as geo
from .contracts import (
    CoordinationEnvironment,
    CoordinationGeometryResult,
)
from .settings import (
    _BOND_RING_MAX_SIZE,
    _METAL_RELOCATION_COORDINATION_DISTANCE_RATIO_MAX,
    _METAL_RELOCATION_COORDINATION_DISTANCE_RATIO_MIN,
    _METAL_RELOCATION_COORDINATION_DISTANCE_SCALE,
    _METAL_RELOCATION_DIRECTION_EPSILON,
    _METAL_RELOCATION_DUPLICATE_TOLERANCE,
    _METAL_RELOCATION_MAX_CANDIDATES,
    _METAL_RELOCATION_MIN_ATOM_CLEARANCE,
    _METAL_RELOCATION_MIN_BOND_CLEARANCE,
    _METAL_RELOCATION_SPHERE_DIRECTION_COUNT,
)


if TYPE_CHECKING:
    from ..core import Atom, Bond, Molecule


__all__ = (
    "collect_coordination_environments",
    "prepare_coordination_geometry",
)


@dataclass(frozen=True)
class _MetalPlacementCandidate:
    coordinates: Tuple[float, float, float]
    safe_donor_indices: Tuple[int, ...]
    minimum_normalized_clearance: float
    coordination_distance_deviation: float
    sequence_index: int


@dataclass(frozen=True)
class _MetalPlacementContext:
    metal: "Atom"
    donors: Tuple["Atom", ...]
    nonmetal_atoms: Tuple["Atom", ...]
    covalent_bonds: Tuple["Bond", ...]
    cycles: Tuple[geo.Cycle, ...]


@dataclass(frozen=True)
class _MetalRelocationResult:
    metal_idx: int
    status: Literal["relocated", "infeasible"]
    candidates_evaluated: int
    original_coordinates: Tuple[float, float, float]
    selected_coordinates: Optional[Tuple[float, float, float]]
    safe_donor_indices: Tuple[int, ...] = ()
    minimum_normalized_clearance: Optional[float] = None
    coordination_distance_deviation: Optional[float] = None

    @property
    def moved(self) -> bool:
        return self.status == "relocated"


def _unit_vector(vector: np.ndarray) -> Optional[np.ndarray]:
    length = float(np.linalg.norm(vector))
    if length <= _METAL_RELOCATION_DIRECTION_EPSILON:
        return None
    return vector / length


def _fibonacci_directions(count: int) -> Iterator[np.ndarray]:
    """Yield a deterministic, approximately uniform sequence on the unit sphere."""
    golden_angle = pi * (3.0 - sqrt(5.0))
    for index in range(count):
        z = 1.0 - 2.0 * ((index + 0.5) / count)
        radius = sqrt(max(0.0, 1.0 - z * z))
        angle = index * golden_angle
        yield np.array((radius * cos(angle), radius * sin(angle), z))


def _candidate_directions(
    donor_coordinates: np.ndarray,
    ligand_centroid: np.ndarray,
    metal_coordinates: np.ndarray,
) -> Tuple[np.ndarray, ...]:
    preferred = (
        donor_coordinates - ligand_centroid,
        metal_coordinates - donor_coordinates,
        np.array((1.0, 0.0, 0.0)),
        np.array((-1.0, 0.0, 0.0)),
        np.array((0.0, 1.0, 0.0)),
        np.array((0.0, -1.0, 0.0)),
        np.array((0.0, 0.0, 1.0)),
        np.array((0.0, 0.0, -1.0)),
    )
    normalized = tuple(
        direction
        for vector in preferred
        if (direction := _unit_vector(vector)) is not None
    )
    return normalized + tuple(_fibonacci_directions(
        _METAL_RELOCATION_SPHERE_DIRECTION_COUNT
    ))


def _pending_donors(
    metal: "Atom",
    pending_bonds: Sequence["Bond"],
) -> Tuple["Atom", ...]:
    return tuple(sorted(
        (
            bond.another_end(metal)
            for bond in pending_bonds
            if bond.atom1 is metal or bond.atom2 is metal
        ),
        key=lambda atom: atom.idx,
    ))


def _coordination_distance(metal: "Atom", donor: "Atom") -> float:
    return _METAL_RELOCATION_COORDINATION_DISTANCE_SCALE * (
        float(metal.covalent_radius) + float(donor.covalent_radius)
    )


def _normalized_atom_clearance(
    segment: geo.Segment,
    metal: "Atom",
    obstacle: "Atom",
) -> float:
    radius_sum = max(
        float(metal.covalent_radius) + float(obstacle.covalent_radius),
        _METAL_RELOCATION_DIRECTION_EPSILON,
    )
    return geo.point_segment_distance(
        geo.Point(obstacle.coordinates),
        segment,
    ) / radius_sum


def _normalized_bond_clearance(
    segment: geo.Segment,
    obstacle_bond: "Bond",
) -> float:
    capsule_radius = max(
        0.5 * (
            float(obstacle_bond.atom1.covalent_radius)
            + float(obstacle_bond.atom2.covalent_radius)
        ),
        _METAL_RELOCATION_DIRECTION_EPSILON,
    )
    return geo.segment_segment_distance(
        segment,
        geo.segment_from_bond(obstacle_bond),
    ) / capsule_radius


def _prepare_metal_placement_context(
    mol: "Molecule",
    metal: "Atom",
    donors: Tuple["Atom", ...],
) -> _MetalPlacementContext:
    return _MetalPlacementContext(
        metal=metal,
        donors=donors,
        nonmetal_atoms=tuple(
            atom
            for atom in mol.atoms
            if atom.idx != metal.idx and not atom.is_metal
        ),
        covalent_bonds=tuple(
            bond
            for bond in mol.bonds
            if bond.is_covalent
            and bond.atom1 is not metal
            and bond.atom2 is not metal
        ),
        cycles=tuple(
            ring.cycle
            for ring in geo.iter_ring_geometries(
                mol,
                ring_scope="full_graph",
                max_ring_size=_BOND_RING_MAX_SIZE,
            )
        ),
    )


def _segment_pierces_any_cycle(
    segment: geo.Segment,
    cycles: Tuple[geo.Cycle, ...],
) -> bool:
    return any(
        next(geo.iter_segment_cycle_screenings((segment,), cycle)).state
        is geo.PiercingState.PIERCES
        for cycle in cycles
    )


def _evaluate_metal_candidate(
    context: _MetalPlacementContext,
    coordinates: np.ndarray,
    sequence_index: int,
) -> Optional[_MetalPlacementCandidate]:
    safe_donor_indices = []
    clearances = []
    distance_deviations = []

    for donor in context.donors:
        segment = geo.Segment(coordinates, donor.coordinates)
        target_distance = _coordination_distance(context.metal, donor)
        distance_ratio = segment.length / max(
            target_distance,
            _METAL_RELOCATION_DIRECTION_EPSILON,
        )
        if not (
            _METAL_RELOCATION_COORDINATION_DISTANCE_RATIO_MIN
            <= distance_ratio
            <= _METAL_RELOCATION_COORDINATION_DISTANCE_RATIO_MAX
        ):
            continue

        atom_clearances = tuple(
            _normalized_atom_clearance(segment, context.metal, atom)
            for atom in context.nonmetal_atoms
            if atom is not donor
        )
        if (
            atom_clearances
            and min(atom_clearances) < _METAL_RELOCATION_MIN_ATOM_CLEARANCE
        ):
            continue

        unrelated_bonds = tuple(
            bond
            for bond in context.covalent_bonds
            if bond.atom1 is not donor and bond.atom2 is not donor
        )
        bond_clearances = tuple(
            _normalized_bond_clearance(segment, bond)
            for bond in unrelated_bonds
        )
        if (
            bond_clearances
            and min(bond_clearances) < _METAL_RELOCATION_MIN_BOND_CLEARANCE
        ):
            continue

        if _segment_pierces_any_cycle(segment, context.cycles):
            continue

        safe_donor_indices.append(donor.idx)
        clearances.extend(atom_clearances)
        clearances.extend(bond_clearances)
        distance_deviations.append(abs(distance_ratio - 1.0))

    if not safe_donor_indices:
        return None
    return _MetalPlacementCandidate(
        coordinates=tuple(float(value) for value in coordinates),
        safe_donor_indices=tuple(safe_donor_indices),
        minimum_normalized_clearance=min(clearances, default=float("inf")),
        coordination_distance_deviation=(
            sum(distance_deviations) / len(distance_deviations)
        ),
        sequence_index=sequence_index,
    )


def _iter_metal_candidate_coordinates(
    mol: "Molecule",
    metal: "Atom",
    donors: Tuple["Atom", ...],
) -> Iterator[np.ndarray]:
    ligand_coordinates = np.asarray(
        [atom.coordinates for atom in mol.atoms if not atom.is_metal],
        dtype=float,
    )
    ligand_centroid = np.mean(ligand_coordinates, axis=0)
    metal_coordinates = np.asarray(metal.coordinates, dtype=float)
    candidate_sets = tuple(
        (
            donor,
            _candidate_directions(
                np.asarray(donor.coordinates, dtype=float),
                ligand_centroid,
                metal_coordinates,
            ),
        )
        for donor in donors
    )
    seen = set()
    emitted = 0
    direction_count = max(
        (len(directions) for _, directions in candidate_sets),
        default=0,
    )
    for direction_index in range(direction_count):
        for donor, directions in candidate_sets:
            if direction_index >= len(directions):
                continue
            donor_coordinates = np.asarray(donor.coordinates, dtype=float)
            target_distance = _coordination_distance(metal, donor)
            direction = directions[direction_index]
            coordinates = donor_coordinates + target_distance * direction
            key = tuple(np.round(
                coordinates / _METAL_RELOCATION_DUPLICATE_TOLERANCE,
            ).astype(np.int64))
            if key in seen:
                continue
            seen.add(key)
            yield coordinates
            emitted += 1
            if emitted >= _METAL_RELOCATION_MAX_CANDIDATES:
                return


def _relocate_unbound_metal(
    mol: "Molecule",
    metal: "Atom",
    pending_bonds: Sequence["Bond"],
) -> _MetalRelocationResult:
    """Move one unbound metal to the best bounded, geometrically visible site."""
    original_coordinates = tuple(float(value) for value in metal.coordinates)
    donors = _pending_donors(metal, pending_bonds)
    context = _prepare_metal_placement_context(mol, metal, donors)
    candidates = []
    evaluated_count = 0
    for sequence_index, coordinates in enumerate(
        _iter_metal_candidate_coordinates(mol, metal, donors)
    ):
        evaluated_count += 1
        candidate = _evaluate_metal_candidate(
            context,
            coordinates,
            sequence_index,
        )
        if candidate is not None:
            candidates.append(candidate)

    if not candidates:
        return _MetalRelocationResult(
            metal_idx=metal.idx,
            status="infeasible",
            candidates_evaluated=evaluated_count,
            original_coordinates=original_coordinates,
            selected_coordinates=None,
        )

    selected = min(
        candidates,
        key=lambda candidate: (
            -len(candidate.safe_donor_indices),
            -candidate.minimum_normalized_clearance,
            candidate.coordination_distance_deviation,
            candidate.sequence_index,
        ),
    )
    metal.coordinates = selected.coordinates
    return _MetalRelocationResult(
        metal_idx=metal.idx,
        status="relocated",
        candidates_evaluated=evaluated_count,
        original_coordinates=original_coordinates,
        selected_coordinates=selected.coordinates,
        safe_donor_indices=selected.safe_donor_indices,
        minimum_normalized_clearance=selected.minimum_normalized_clearance,
        coordination_distance_deviation=selected.coordination_distance_deviation,
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
