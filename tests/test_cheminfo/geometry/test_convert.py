"""Tests for typed chemical-object to geometry conversion."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Dict, Optional, Sequence, Tuple

import pytest

from hotpot.cheminfo.core import Molecule
from hotpot.cheminfo.geometry import convert, native
from hotpot.cheminfo.geometry.relation import (
    PiercingState,
)
from hotpot.cheminfo.geometry.object import Segment
from hotpot.cheminfo.geometry.settings import DEFAULT_GEOMETRY_SETTINGS


@dataclass(frozen=True)
class FakeAtom:
    idx: int
    coordinates: Tuple[float, float, float]


@dataclass(frozen=True)
class FakeBond:
    atom1: FakeAtom
    atom2: FakeAtom


@dataclass(frozen=True)
class FakeRing:
    atoms: Tuple[FakeAtom, ...]


@dataclass
class FakeMolecule:
    atoms: Sequence[FakeAtom]
    bonds: Sequence[FakeBond]
    rings_by_scope: Dict[str, Sequence[FakeRing]]
    ring_queries: list = field(default_factory=list)

    def rings_for_scope(
            self,
            ring_scope: str,
            *,
            max_size: Optional[int] = None,
            max_cycles: Optional[int] = None,
    ) -> Sequence[FakeRing]:
        self.ring_queries.append((ring_scope, max_size, max_cycles))
        rings = tuple(self.rings_by_scope[ring_scope])
        if max_size is None:
            return rings
        return tuple(ring for ring in rings if len(ring.atoms) <= max_size)


if TYPE_CHECKING:
    from hotpot.cheminfo.core import Atom, Bond, Ring

    def _check_fake_source_types(
            mol: FakeMolecule,
            ring: FakeRing,
            bond: FakeBond,
    ) -> None:
        atom_geometry: convert.AtomGeometry[FakeAtom] = next(
            convert.iter_atom_geometries(mol)
        )
        ring_geometry: convert.RingGeometry[FakeRing] = next(
            convert.iter_ring_geometries(
                mol,
                ring_scope="full_graph",
                max_ring_size=8,
            )
        )
        finding: convert.BondRingFinding[FakeRing, FakeBond] = (
            convert.determine_bond_ring_relation(ring, bond)
        )
        del atom_geometry, ring_geometry, finding

    def _check_hotpot_source_types(mol: Molecule) -> None:
        atom_geometry: convert.AtomGeometry[Atom] = next(
            convert.iter_atom_geometries(mol)
        )
        report: convert.BondRingScanReport[Ring, Bond] = (
            convert.scan_bond_ring_relations(
                mol,
                ring_scope="full_graph",
                max_ring_size=8,
            )
        )
        del atom_geometry, report


@pytest.fixture
def square_molecule() -> FakeMolecule:
    ring_atoms = (
        FakeAtom(0, (-1.0, -1.0, 0.0)),
        FakeAtom(1, (1.0, -1.0, 0.0)),
        FakeAtom(2, (1.0, 1.0, 0.0)),
        FakeAtom(3, (-1.0, 1.0, 0.0)),
    )
    outside = FakeAtom(4, (-2.0, -2.0, 1.0))
    crossing_start = FakeAtom(5, (0.0, 0.0, -1.0))
    crossing_end = FakeAtom(6, (0.0, 0.0, 1.0))
    boundary_bonds = tuple(
        FakeBond(ring_atoms[index], ring_atoms[(index + 1) % 4])
        for index in range(4)
    )
    shared_endpoint_bond = FakeBond(ring_atoms[0], outside)
    crossing_bond = FakeBond(crossing_start, crossing_end)
    return FakeMolecule(
        atoms=ring_atoms + (outside, crossing_start, crossing_end),
        bonds=boundary_bonds + (shared_endpoint_bond, crossing_bond),
        rings_by_scope={
            "full_graph": (FakeRing(ring_atoms),),
            "ligand_skeleton": (FakeRing(tuple(reversed(ring_atoms))),),
        },
    )


def test_point_segment_and_cycle_conversion_preserve_sources() -> None:
    first = FakeAtom(8, (1.0, 2.0, 3.0))
    second = FakeAtom(2, (4.0, 5.0, 6.0))
    bond = FakeBond(first, second)
    ring = FakeRing((first, second, FakeAtom(5, (0.0, 0.0, 0.0))))

    assert convert.point_from_atom(first).coordinates == (1.0, 2.0, 3.0)
    assert convert.segment_from_bond(bond).start.coordinates == (1.0, 2.0, 3.0)
    assert convert.segment_from_bond(bond).end.coordinates == (4.0, 5.0, 6.0)
    assert tuple(point.coordinates for point in convert.cycle_from_ring(ring).vertices) == (
        (1.0, 2.0, 3.0),
        (4.0, 5.0, 6.0),
        (0.0, 0.0, 0.0),
    )


def test_atom_pair_scopes_and_measurements_use_the_chemical_graph() -> None:
    atoms = (
        FakeAtom(10, (0.0, 0.0, 0.0)),
        FakeAtom(20, (3.0, 4.0, 0.0)),
        FakeAtom(30, (0.0, 0.0, 12.0)),
    )
    structure = FakeMolecule(
        atoms=atoms,
        bonds=(FakeBond(atoms[0], atoms[1]),),
        rings_by_scope={"full_graph": (), "ligand_skeleton": ()},
    )

    bonded = convert.measure_atom_pair_distances(structure, "bonded")
    nonbonded = convert.measure_atom_pair_distances(structure, "nonbonded")
    all_pairs = convert.measure_atom_pair_distances(structure, "all")

    assert [(item.target.first.key, item.target.second.key) for item in bonded] == [(10, 20)]
    assert bonded[0].measurement.distance == pytest.approx(5.0)
    assert len(nonbonded) == 2
    assert len(all_pairs) == 3
    assert all(not item.target.bonded for item in nonbonded)


def test_atom_pair_measurement_requests_only_the_selected_native_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    atoms = (
        FakeAtom(10, (0.0, 0.0, 0.0)),
        FakeAtom(20, (3.0, 4.0, 0.0)),
        FakeAtom(30, (0.0, 0.0, 12.0)),
    )
    structure = FakeMolecule(
        atoms=atoms,
        bonds=(FakeBond(atoms[0], atoms[2]),),
        rings_by_scope={"full_graph": (), "ligand_skeleton": ()},
    )
    calls = []
    measure = native._selected_point_pair_distances

    def counted_measure(points, pair_indices):
        pairs = tuple(pair_indices)
        calls.append(pairs)
        return measure(points, pairs)

    monkeypatch.setattr(native, "_selected_point_pair_distances", counted_measure)

    result = convert.measure_atom_pair_distances(structure, "bonded")

    assert calls == [((0, 2),)]
    assert result[0].measurement.distance == pytest.approx(12.0)


def test_invalid_pair_scope_is_rejected() -> None:
    atoms = (
        FakeAtom(0, (0.0, 0.0, 0.0)),
        FakeAtom(1, (1.0, 0.0, 0.0)),
    )
    structure = FakeMolecule(
        atoms=atoms,
        bonds=(),
        rings_by_scope={"full_graph": (), "ligand_skeleton": ()},
    )

    with pytest.raises(ValueError, match="Unsupported atom-pair scope"):
        tuple(convert.iter_atom_pair_targets(structure, "invalid"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unsupported atom-pair scope"):
        convert.measure_atom_pair_distances(structure, "invalid")  # type: ignore[arg-type]


def test_ring_key_is_invariant_to_rotation_and_reversal() -> None:
    atoms = tuple(FakeAtom(index, (float(index), 0.0, 0.0)) for index in (7, 2, 9, 4))
    rotated = atoms[2:] + atoms[:2]
    reversed_atoms = tuple(reversed(atoms))
    mol = FakeMolecule(
        atoms=atoms,
        bonds=(),
        rings_by_scope={
            "full_graph": (
                FakeRing(atoms),
                FakeRing(rotated),
                FakeRing(reversed_atoms),
            ),
            "ligand_skeleton": (),
        },
    )

    keys = tuple(
        item.key
        for item in convert.iter_ring_geometries(
            mol,
            ring_scope="full_graph",
            max_ring_size=8,
        )
    )

    assert keys == ((2, 7, 4, 9),) * 3


def test_candidate_pairs_exclude_only_ring_edges(square_molecule: FakeMolecule) -> None:
    targets = tuple(convert.iter_bond_ring_targets(
        square_molecule,
        ring_scope="full_graph",
        max_ring_size=8,
    ))

    assert tuple(target.bond.key for target in targets) == ((0, 4), (5, 6))
    assert 0 in targets[0].bond.key


def test_single_relation_delegates_to_the_geometric_kernel(
        square_molecule: FakeMolecule,
) -> None:
    ring = square_molecule.rings_by_scope["full_graph"][0]
    crossing_bond = square_molecule.bonds[-1]

    finding = convert.determine_bond_ring_relation(ring, crossing_bond)

    assert finding.relation.state is PiercingState.PIERCES
    assert finding.target.ring.ring is ring
    assert finding.target.bond.bond is crossing_bond


def test_core_ring_scope_query_does_not_populate_ring_caches() -> None:
    mol = Molecule()
    for index, coordinates in enumerate((
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    )):
        mol.create_atom(atomic_number=6, coordinates=coordinates, id=index)
    mol.add_bond(0, 1, bond_order=1.0)
    mol.add_bond(1, 2, bond_order=1.0)
    mol.add_bond(2, 0, bond_order=1.0)

    assert mol._rings == []
    assert mol._ligand_rings is None

    full_graph = mol.rings_for_scope("full_graph")
    ligand_skeleton = mol.rings_for_scope("ligand_skeleton")

    assert len(full_graph) == len(ligand_skeleton) == 1
    assert mol._rings == []
    assert mol._ligand_rings is None


def test_dense_scan_records_every_pair_and_coverage(
        square_molecule: FakeMolecule,
) -> None:
    report = convert.scan_bond_ring_relations(
        square_molecule,
        ring_scope="full_graph",
        max_ring_size=8,
    )

    assert square_molecule.ring_queries == [("full_graph", None, None)]
    assert report.selected_ring_count == 1
    assert report.excluded_ring_count == 0
    assert report.candidate_pair_count == len(report.findings) == 2
    assert report.piercing_pair_count == 1
    assert report.does_not_pierce_pair_count == 1
    assert report.undetermined_pair_count == 0
    assert report.piercings == (report.findings[1],)
    assert report.scan_complete


def test_dense_scan_batches_all_bonds_for_each_ring(
        square_molecule: FakeMolecule,
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    determine = native._determine_segment_cycle_relations

    def counted_determine(cycles, segments, candidate_pairs):
        segment_batch = tuple(segments)
        pair_batch = tuple(candidate_pairs)
        calls.append((cycles.cycle_count, len(segment_batch), pair_batch))
        return determine(cycles, segment_batch, pair_batch)

    monkeypatch.setattr(
        native,
        "_determine_segment_cycle_relations",
        counted_determine,
    )

    report = convert.scan_bond_ring_relations(
        square_molecule,
        ring_scope="full_graph",
        max_ring_size=8,
    )

    assert report.candidate_pair_count == 2
    assert calls == [(1, 6, ((2, 0), (5, 0)))]


def test_screening_report_covers_all_pairs_and_retains_only_actionable_findings(
        square_molecule: FakeMolecule,
) -> None:
    far_atoms = (
        FakeAtom(7, (10.0, 10.0, 3.0)),
        FakeAtom(8, (11.0, 10.0, 3.0)),
    )
    screened_molecule = FakeMolecule(
        atoms=tuple(square_molecule.atoms) + far_atoms,
        bonds=tuple(square_molecule.bonds[:4])
        + (square_molecule.bonds[-1], FakeBond(*far_atoms)),
        rings_by_scope=square_molecule.rings_by_scope,
    )
    report = convert.screen_bond_ring_relations(
        screened_molecule,
        ring_scope="full_graph",
        max_ring_size=8,
    )

    assert report.candidate_pair_count == 2
    assert report.piercing_pair_count == 1
    assert report.does_not_pierce_pair_count == 1
    assert report.undetermined_pair_count == 0
    assert report.aabb_separated_pair_count == 1
    assert report.exact_pair_count == 1
    assert report.actionable_findings == report.piercings
    assert len(report.piercings) == 1
    assert report.scan_complete


def test_explicit_bond_screen_includes_bond_absent_from_molecular_bond_table(
        square_molecule: FakeMolecule,
) -> None:
    crossing_bond = square_molecule.bonds[-1]
    molecule_without_crossing_bond = FakeMolecule(
        atoms=square_molecule.atoms,
        bonds=square_molecule.bonds[:-1],
        rings_by_scope=square_molecule.rings_by_scope,
    )

    report = convert.screen_bonds_against_rings(
        molecule_without_crossing_bond,
        (crossing_bond,),
        ring_scope="full_graph",
        max_ring_size=8,
    )

    assert report.candidate_pair_count == 1
    assert report.piercing_pair_count == 1
    assert report.piercings[0].target.bond.bond is crossing_bond


def test_explicit_bond_screen_excludes_ring_edges_and_counts_aabb_paths(
        square_molecule: FakeMolecule,
) -> None:
    far_atoms = (
        FakeAtom(7, (10.0, 10.0, 3.0)),
        FakeAtom(8, (11.0, 10.0, 3.0)),
    )
    far_bond = FakeBond(*far_atoms)

    report = convert.screen_bonds_against_rings(
        square_molecule,
        (square_molecule.bonds[0], square_molecule.bonds[-1], far_bond),
        ring_scope="full_graph",
        max_ring_size=8,
    )

    assert report.candidate_pair_count == 2
    assert report.piercing_pair_count == 1
    assert report.does_not_pierce_pair_count == 1
    assert report.aabb_separated_pair_count == 1
    assert report.exact_pair_count == 1


def test_full_molecule_screen_wraps_explicit_bond_screen(
        square_molecule: FakeMolecule,
) -> None:
    explicit_report = convert.screen_bonds_against_rings(
        square_molecule,
        square_molecule.bonds,
        ring_scope="full_graph",
        max_ring_size=8,
    )
    full_report = convert.screen_bond_ring_relations(
        square_molecule,
        ring_scope="full_graph",
        max_ring_size=8,
    )

    assert full_report == explicit_report


def test_screening_plan_and_workspace_preserve_one_shot_behavior(
        square_molecule: FakeMolecule,
) -> None:
    plan = convert.prepare_bond_ring_screening_plan(
        square_molecule,
        ring_scope="full_graph",
        max_ring_size=8,
    )
    workspace = convert.prepare_bond_ring_frame(plan)

    workspace_report = convert.screen_bond_ring_workspace(workspace)
    one_shot_report = convert.screen_bond_ring_relations(
        square_molecule,
        ring_scope="full_graph",
        max_ring_size=8,
    )

    assert plan.ring_atom_keys == ((0, 1, 2, 3),)
    assert plan.ring_edge_keys == (frozenset({
        (0, 1),
        (1, 2),
        (2, 3),
        (0, 3),
    }),)
    assert plan.candidate_bond_keys_by_ring == (((0, 4), (5, 6)),)
    assert workspace_report == one_shot_report


def test_dense_batch_preserves_ring_major_order_and_source_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_ring_atoms = tuple(
        FakeAtom(index, coordinates)
        for index, coordinates in enumerate((
            (10.0, -1.0, 0.0),
            (12.0, -1.0, 0.0),
            (12.0, 1.0, 0.0),
            (10.0, 1.0, 0.0),
        ))
    )
    second_ring_atoms = tuple(
        FakeAtom(index + 10, coordinates)
        for index, coordinates in enumerate((
            (-1.0, -1.0, 0.0),
            (1.0, -1.0, 0.0),
            (1.0, 1.0, 0.0),
            (-1.0, 1.0, 0.0),
        ))
    )
    first_crossing_atoms = (
        FakeAtom(20, (11.0, 0.0, -1.0)),
        FakeAtom(21, (11.0, 0.0, 1.0)),
    )
    second_crossing_atoms = (
        FakeAtom(30, (0.0, 0.0, -1.0)),
        FakeAtom(31, (0.0, 0.0, 1.0)),
    )
    first_ring = FakeRing(first_ring_atoms)
    second_ring = FakeRing(second_ring_atoms)
    first_bond = FakeBond(*first_crossing_atoms)
    second_bond = FakeBond(*second_crossing_atoms)
    molecule = FakeMolecule(
        atoms=(
            first_ring_atoms
            + second_ring_atoms
            + first_crossing_atoms
            + second_crossing_atoms
        ),
        bonds=(first_bond, second_bond),
        rings_by_scope={
            "full_graph": (second_ring, first_ring),
            "ligand_skeleton": (),
        },
    )
    calls = []
    determine = native._determine_segment_cycle_relations

    def counted_determine(cycles, segments, candidate_pairs):
        segment_batch = tuple(segments)
        pair_batch = tuple(candidate_pairs)
        calls.append((cycles.cycle_count, len(segment_batch), pair_batch))
        return determine(cycles, segment_batch, pair_batch)

    monkeypatch.setattr(
        native,
        "_determine_segment_cycle_relations",
        counted_determine,
    )

    report = convert.scan_bond_ring_relations(
        molecule,
        ring_scope="full_graph",
        max_ring_size=8,
    )

    assert len(calls) == 1
    assert calls[0] == (2, 2, ((0, 0), (1, 0), (0, 1), (1, 1)))
    assert all(
        finding.target.ring.ring is expected
        for finding, expected in zip(
            report.findings,
            (first_ring, first_ring, second_ring, second_ring),
        )
    )
    assert all(
        finding.target.bond.bond is expected
        for finding, expected in zip(
            report.findings,
            (first_bond, second_bond, first_bond, second_bond),
        )
    )
    assert tuple(finding.relation.state for finding in report.findings) == (
        PiercingState.PIERCES,
        PiercingState.DOES_NOT_PIERCE,
        PiercingState.DOES_NOT_PIERCE,
        PiercingState.PIERCES,
    )


def test_dense_batch_maps_equal_key_bonds_by_position_not_dictionary_key(
    square_molecule: FakeMolecule,
) -> None:
    first_bond = square_molecule.bonds[-1]
    second_bond = FakeBond(first_bond.atom1, first_bond.atom2)
    molecule = FakeMolecule(
        atoms=square_molecule.atoms,
        bonds=tuple(square_molecule.bonds[:4]) + (first_bond, second_bond),
        rings_by_scope=square_molecule.rings_by_scope,
    )

    report = convert.scan_bond_ring_relations(
        molecule,
        ring_scope="full_graph",
        max_ring_size=8,
    )

    assert len(report.findings) == 2
    assert report.findings[0].target.bond.bond is first_bond
    assert report.findings[1].target.bond.bond is second_bond


def test_workspace_screen_matches_public_scalar_pair_iteration(
        square_molecule: FakeMolecule,
) -> None:
    far_atoms = (
        FakeAtom(7, (10.0, 10.0, 3.0)),
        FakeAtom(8, (11.0, 10.0, 3.0)),
    )
    bonds = (square_molecule.bonds[-1], FakeBond(*far_atoms))
    workspace = convert.prepare_bond_ring_frame(
        convert.prepare_bond_ring_screening_plan(
            square_molecule,
            bonds=bonds,
            ring_scope="full_graph",
            max_ring_size=8,
        )
    )
    report = convert.screen_bond_ring_workspace(workspace)
    ring = square_molecule.rings_by_scope["full_graph"][0]
    scalar = tuple(
        convert.determine_bond_ring_relation(ring, bond)
        for bond in bonds
    )

    assert report.selected_ring_count == 1
    assert report.excluded_ring_count == 0
    assert report.candidate_pair_count == len(scalar)
    assert report.piercing_pair_count == sum(
        finding.relation.state is PiercingState.PIERCES for finding in scalar
    )
    assert report.does_not_pierce_pair_count == sum(
        finding.relation.state is PiercingState.DOES_NOT_PIERCE
        for finding in scalar
    )
    assert report.undetermined_pair_count == sum(
        finding.relation.state is PiercingState.UNDETERMINED
        for finding in scalar
    )
    assert tuple(
        (
            finding.target.ring.key,
            finding.target.bond.key,
            finding.relation.state,
        )
        for finding in report.actionable_findings
    ) == tuple(
        (
            finding.target.ring.key,
            finding.target.bond.key,
            finding.relation.state,
        )
        for finding in scalar
        if finding.relation.state is not PiercingState.DOES_NOT_PIERCE
    )


def test_frame_workspace_is_a_read_only_coordinate_snapshot(
        square_molecule: FakeMolecule,
) -> None:
    plan = convert.prepare_bond_ring_screening_plan(
        square_molecule,
        ring_scope="full_graph",
        max_ring_size=8,
    )
    original_workspace = convert.prepare_bond_ring_frame(plan)
    crossing_start = square_molecule.atoms[5]
    coordinate_row = original_workspace.coordinate_keys.index(crossing_start.idx)

    with pytest.raises(ValueError, match="read-only"):
        original_workspace.coordinates[coordinate_row, 0] = 20.0

    object.__setattr__(crossing_start, "coordinates", (10.0, 10.0, -1.0))
    refreshed_workspace = convert.prepare_bond_ring_frame(plan)

    assert tuple(original_workspace.coordinates[coordinate_row]) == (0.0, 0.0, -1.0)
    assert tuple(refreshed_workspace.coordinates[coordinate_row]) == (
        10.0,
        10.0,
        -1.0,
    )
    assert (
        convert.screen_bond_ring_workspace(original_workspace).state
        is PiercingState.PIERCES
    )
    assert (
        convert.screen_bond_ring_workspace(refreshed_workspace).state
        is not PiercingState.PIERCES
    )


def test_topology_plan_remains_a_snapshot_until_rebuilt(
        square_molecule: FakeMolecule,
) -> None:
    plan = convert.prepare_bond_ring_screening_plan(
        square_molecule,
        ring_scope="full_graph",
        max_ring_size=8,
    )
    square_molecule.bonds = square_molecule.bonds[:-1]

    stale_plan_report = convert.screen_bond_ring_workspace(
        convert.prepare_bond_ring_frame(plan)
    )
    rebuilt_plan_report = convert.screen_bond_ring_relations(
        square_molecule,
        ring_scope="full_graph",
        max_ring_size=8,
    )

    assert stale_plan_report.state is PiercingState.PIERCES
    assert rebuilt_plan_report.state is not PiercingState.PIERCES


def test_workspace_key_selection_preserves_canonical_pair_order(
        square_molecule: FakeMolecule,
) -> None:
    workspace = convert.prepare_bond_ring_frame(
        convert.prepare_bond_ring_screening_plan(
            square_molecule,
            ring_scope="full_graph",
            max_ring_size=8,
        )
    )

    crossing_only = convert.screen_bond_ring_workspace(
        workspace,
        bond_keys=((5, 6),),
        ring_keys=((0, 1, 2, 3),),
    )
    no_rings = convert.screen_bond_ring_workspace(
        workspace,
        ring_keys=((20, 21, 22),),
    )

    assert crossing_only.candidate_pair_count == 1
    assert crossing_only.piercing_pair_count == 1
    assert crossing_only.piercings[0].target.bond.key == (5, 6)
    assert no_rings.selected_ring_count == 0
    assert no_rings.candidate_pair_count == 0


def test_workspace_can_stop_after_first_confirmed_piercing(
        square_molecule: FakeMolecule,
) -> None:
    far_atoms = (
        FakeAtom(7, (10.0, 10.0, 3.0)),
        FakeAtom(8, (11.0, 10.0, 3.0)),
    )
    plan = convert.prepare_bond_ring_screening_plan(
        square_molecule,
        bonds=(square_molecule.bonds[-1], FakeBond(*far_atoms)),
        ring_scope="full_graph",
        max_ring_size=8,
    )

    report = convert.screen_bond_ring_workspace(
        convert.prepare_bond_ring_frame(plan),
        stop_after_confirmed=True,
    )

    assert report.state is PiercingState.PIERCES
    assert report.candidate_pair_count == 1
    assert not report.scan_complete


def test_explicit_segment_snapshot_uses_workspace_rings(
        square_molecule: FakeMolecule,
) -> None:
    crossing_bond = square_molecule.bonds[-1]
    workspace = convert.prepare_bond_ring_frame(
        convert.prepare_bond_ring_screening_plan(
            square_molecule,
            bonds=(crossing_bond,),
            ring_scope="full_graph",
            max_ring_size=8,
        )
    )
    displaced_segment = convert.BondGeometry(
        bond=crossing_bond,
        segment=Segment((10.0, 10.0, -1.0), (10.0, 10.0, 1.0)),
        key=(5, 6),
    )

    report = convert.screen_segments_against_ring_workspace(
        (displaced_segment,),
        workspace,
    )

    assert report.state is PiercingState.DOES_NOT_PIERCE
    assert report.aabb_separated_pair_count == 1


def test_dense_scan_is_incomplete_when_surface_enumeration_is_incomplete(
) -> None:
    ring_atoms = (
        FakeAtom(0, (0.0, 0.0, 0.0)),
        FakeAtom(1, (2.0, 0.0, 0.0)),
        FakeAtom(2, (2.0, 2.0, 0.4)),
        FakeAtom(3, (0.0, 2.0, 0.0)),
    )
    crossing_atoms = (
        FakeAtom(4, (0.6, 0.8, -1.0)),
        FakeAtom(5, (0.6, 0.8, 1.0)),
    )
    molecule = FakeMolecule(
        atoms=ring_atoms + crossing_atoms,
        bonds=tuple(
            FakeBond(ring_atoms[index], ring_atoms[(index + 1) % 4])
            for index in range(4)
        ) + (FakeBond(*crossing_atoms),),
        rings_by_scope={
            "full_graph": (FakeRing(ring_atoms),),
            "ligand_skeleton": (),
        },
    )
    settings = replace(
        DEFAULT_GEOMETRY_SETTINGS,
        surface=replace(
            DEFAULT_GEOMETRY_SETTINGS.surface,
            maximum_surface_count=1,
        ),
    )

    report = convert.scan_bond_ring_relations(
        molecule,
        ring_scope="full_graph",
        max_ring_size=8,
        settings=settings,
    )

    assert report.candidate_pair_count == 1
    assert report.undetermined_pair_count == 1
    assert not report.scan_complete


def test_ring_size_coverage_distinguishes_excluded_and_empty_scan(
        square_molecule: FakeMolecule,
) -> None:
    report = convert.scan_bond_ring_relations(
        square_molecule,
        ring_scope="full_graph",
        max_ring_size=3,
    )

    assert report.selected_ring_count == 0
    assert report.excluded_ring_count == 1
    assert report.candidate_pair_count == 0
    assert report.scan_complete


def test_state_query_uses_state_only_native_early_stop(
        square_molecule: FakeMolecule,
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    screen = native._screen_segments

    def counted_screen(
        cycles,
        segments,
        candidate_pairs,
        detail,
        *,
        stop_after_confirmed=False,
    ):
        result = screen(
            cycles,
            tuple(segments),
            tuple(candidate_pairs),
            detail,
            stop_after_confirmed=stop_after_confirmed,
        )
        calls.append((detail, stop_after_confirmed, result.evaluated_pair_count))
        return result

    monkeypatch.setattr(native, "_screen_segments", counted_screen)

    state = convert.determine_bond_ring_piercing_state(
        square_molecule,
        ring_scope="full_graph",
        max_ring_size=8,
    )

    assert state is PiercingState.PIERCES
    assert calls == [(native.DetailLevel.STATE_ONLY, True, 2)]


def test_state_query_prepares_one_packed_frame_and_one_native_batch(
        square_molecule: FakeMolecule,
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepare_calls = []
    screen_calls = []
    prepare = native._prepare_cycles
    screen = native._screen_segments

    def counted_prepare(coordinates, indices, offsets, settings):
        prepare_calls.append((tuple(indices), tuple(offsets)))
        return prepare(coordinates, indices, offsets, settings)

    def counted_screen(*args, **kwargs):
        screen_calls.append((args, kwargs))
        return screen(*args, **kwargs)

    monkeypatch.setattr(native, "_prepare_cycles", counted_prepare)
    monkeypatch.setattr(native, "_screen_segments", counted_screen)

    state = convert.determine_bond_ring_piercing_state(
        square_molecule,
        ring_scope="full_graph",
        max_ring_size=8,
    )

    assert state is PiercingState.PIERCES
    assert prepare_calls == [((0, 1, 2, 3), (0, 4))]
    assert len(screen_calls) == 1


def test_state_query_preserves_undetermined_without_a_piercing(
) -> None:
    ring_atoms = (
        FakeAtom(0, (0.0, 0.0, 0.0)),
        FakeAtom(1, (2.0, 2.0, 0.0)),
        FakeAtom(2, (0.0, 2.0, 0.0)),
        FakeAtom(3, (2.0, 0.0, 0.0)),
    )
    far_atoms = (
        FakeAtom(4, (10.0, 10.0, 4.0)),
        FakeAtom(5, (11.0, 10.0, 4.0)),
    )
    molecule = FakeMolecule(
        atoms=ring_atoms + far_atoms,
        bonds=tuple(
            FakeBond(ring_atoms[index], ring_atoms[(index + 1) % 4])
            for index in range(4)
        ) + (FakeBond(*far_atoms),),
        rings_by_scope={
            "full_graph": (FakeRing(ring_atoms),),
            "ligand_skeleton": (),
        },
    )

    state = convert.determine_bond_ring_piercing_state(
        molecule,
        ring_scope="full_graph",
        max_ring_size=8,
    )

    assert state is PiercingState.UNDETERMINED
