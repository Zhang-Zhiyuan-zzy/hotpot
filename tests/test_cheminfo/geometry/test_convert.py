"""Tests for typed chemical-object to geometry conversion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional, Sequence, Tuple

import pytest

from hotpot.cheminfo.core import Molecule
from hotpot.cheminfo.geometry import convert
from hotpot.cheminfo.geometry.relation import (
    PiercingState,
    SegmentCycleScreening,
    SurfaceFamilyEvidence,
)


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


@dataclass(frozen=True)
class FakeRelation:
    state: PiercingState
    surface_evidence: SurfaceFamilyEvidence


def surface_evidence(*, enumeration_complete: bool = True) -> SurfaceFamilyEvidence:
    return SurfaceFamilyEvidence(
        enumeration_complete=enumeration_complete,
        enumerated_surface_count=0,
        embedded_surface_count=0,
        proven_non_embedded_surface_count=0,
        construction_undetermined_count=0,
        intersecting_surface_count=0,
        non_piercing_surface_count=0,
        evaluation_undetermined_count=0,
        segment_triangle_tests_used=0,
        triangle_pair_tests_used=0,
    )


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
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    states = iter((PiercingState.DOES_NOT_PIERCE, PiercingState.UNDETERMINED))
    monkeypatch.setattr(
        convert,
        "iter_segment_cycle_relations",
        lambda segments, cycle, settings: (
            FakeRelation(next(states), surface_evidence()) for _ in segments
        ),
    )

    report = convert.scan_bond_ring_relations(
        square_molecule,
        ring_scope="full_graph",
        max_ring_size=8,
    )

    assert square_molecule.ring_queries == [("full_graph", None, None)]
    assert report.selected_ring_count == 1
    assert report.excluded_ring_count == 0
    assert report.candidate_pair_count == len(report.findings) == 2
    assert report.piercing_pair_count == 0
    assert report.does_not_pierce_pair_count == 1
    assert report.undetermined_pair_count == 1
    assert report.undetermined == (report.findings[1],)
    assert report.scan_complete


def test_dense_scan_batches_all_bonds_for_each_ring(
        square_molecule: FakeMolecule,
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch_sizes = []

    def fake_relations(segments, cycle, settings):
        batch = tuple(segments)
        batch_sizes.append(len(batch))
        return iter(
            FakeRelation(PiercingState.DOES_NOT_PIERCE, surface_evidence())
            for _ in batch
        )

    monkeypatch.setattr(convert, "iter_segment_cycle_relations", fake_relations)

    report = convert.scan_bond_ring_relations(
        square_molecule,
        ring_scope="full_graph",
        max_ring_size=8,
    )

    assert report.candidate_pair_count == 2
    assert batch_sizes == [2]


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


def test_dense_scan_is_incomplete_when_surface_enumeration_is_incomplete(
        square_molecule: FakeMolecule,
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        convert,
        "iter_segment_cycle_relations",
        lambda segments, cycle, settings: (
            FakeRelation(
                PiercingState.UNDETERMINED,
                surface_evidence(enumeration_complete=False),
            )
            for _ in segments
        ),
    )

    report = convert.scan_bond_ring_relations(
        square_molecule,
        ring_scope="full_graph",
        max_ring_size=8,
    )

    assert report.candidate_pair_count == 2
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


def test_lazy_state_stops_on_first_confirmed_piercing(
        square_molecule: FakeMolecule,
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumed = []

    def fake_screenings(segments, cycle, settings):
        for segment in segments:
            consumed.append(segment)
            state = (
                PiercingState.PIERCES
                if len(consumed) == 1
                else PiercingState.UNDETERMINED
            )
            relation = FakeRelation(state, surface_evidence())
            yield SegmentCycleScreening(state, relation, False, True)

    monkeypatch.setattr(
        convert,
        "iter_segment_cycle_screenings",
        fake_screenings,
    )

    state = convert.determine_bond_ring_piercing_state(
        square_molecule,
        ring_scope="full_graph",
        max_ring_size=8,
    )

    assert state is PiercingState.PIERCES
    assert len(consumed) == 1


def test_lazy_state_does_not_construct_targets_after_first_piercing(
        square_molecule: FakeMolecule,
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    converted_bonds = []
    original_segment_from_bond = convert.segment_from_bond

    def counted_segment_from_bond(bond):
        converted_bonds.append(bond)
        return original_segment_from_bond(bond)

    def first_relation_pierces(segments, cycle, settings):
        for _ in segments:
            relation = FakeRelation(PiercingState.PIERCES, surface_evidence())
            yield SegmentCycleScreening(
                PiercingState.PIERCES,
                relation,
                False,
                True,
            )

    monkeypatch.setattr(convert, "segment_from_bond", counted_segment_from_bond)
    monkeypatch.setattr(
        convert, "iter_segment_cycle_screenings", first_relation_pierces
    )

    state = convert.determine_bond_ring_piercing_state(
        square_molecule,
        ring_scope="full_graph",
        max_ring_size=8,
    )

    assert state is PiercingState.PIERCES
    assert len(converted_bonds) == 1


def test_lazy_state_consumes_all_pairs_to_preserve_undetermined(
        square_molecule: FakeMolecule,
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    states = iter((PiercingState.UNDETERMINED, PiercingState.DOES_NOT_PIERCE))
    consumed = []

    def fake_screenings(segments, cycle, settings):
        for segment in segments:
            consumed.append(segment)
            state = next(states)
            relation = FakeRelation(state, surface_evidence())
            yield SegmentCycleScreening(state, relation, False, True)

    monkeypatch.setattr(
        convert,
        "iter_segment_cycle_screenings",
        fake_screenings,
    )

    state = convert.determine_bond_ring_piercing_state(
        square_molecule,
        ring_scope="full_graph",
        max_ring_size=8,
    )

    assert state is PiercingState.UNDETERMINED
    assert len(consumed) == 2
