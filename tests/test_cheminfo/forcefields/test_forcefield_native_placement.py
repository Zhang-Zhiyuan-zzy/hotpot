"""Behavior fence for the native metal-placement engine."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from hotpot.cheminfo.forcefields.native import (
    MetalPlacementOptions,
    assess_metal_position,
    create_coordination_session,
    place_metal,
    place_metals,
    set_coordination_active_mask,
    snapshot_structure,
)
from hotpot.cheminfo.forcefields.native_packing import ComplexSessionInput
from hotpot.cheminfo.obWrappers import _ob_native


def _session_input(
    atomic_numbers: tuple[int, ...],
    coordinates: tuple[tuple[float, float, float], ...],
    *,
    ligand_bonds: tuple[tuple[int, int], ...] = (),
    coordination_bonds: tuple[tuple[int, int], ...],
    metal_indices: tuple[int, ...] = (0,),
) -> ComplexSessionInput:
    atom_count = len(atomic_numbers)
    ligand_bond_count = len(ligand_bonds)
    coordination_bond_count = len(coordination_bonds)
    return ComplexSessionInput(
        schema_version=1,
        atomic_numbers=np.asarray(atomic_numbers, dtype=np.int32),
        formal_charges=np.zeros(atom_count, dtype=np.int32),
        partial_charges=np.zeros(atom_count, dtype=np.float64),
        coordinates=np.asarray(coordinates, dtype=np.float64),
        atom_aromatic=np.zeros(atom_count, dtype=np.uint8),
        ligand_bond_indices=np.asarray(
            ligand_bonds,
            dtype=np.int32,
        ).reshape((-1, 2)),
        ligand_bond_orders=np.ones(ligand_bond_count, dtype=np.float64),
        ligand_bond_kinds=np.full(
            ligand_bond_count,
            1,
            dtype=np.uint8,
        ),
        ligand_bond_aromatic=np.zeros(
            ligand_bond_count,
            dtype=np.uint8,
        ),
        metal_indices=np.asarray(metal_indices, dtype=np.int32),
        intended_coordination_bonds=np.asarray(
            coordination_bonds,
            dtype=np.int32,
        ).reshape((-1, 2)),
        intended_coordination_orders=np.ones(
            coordination_bond_count,
            dtype=np.float64,
        ),
        intended_coordination_kinds=np.full(
            coordination_bond_count,
            6,
            dtype=np.uint8,
        ),
        unit_cell=None,
    )


def _target_distance(atomic_number: int) -> float:
    return (
        _ob_native.covalent_radius(63).angstrom
        + _ob_native.covalent_radius(atomic_number).angstrom
    )


def _single_donor_input(
    metal_coordinates: tuple[float, float, float],
    donor_coordinates: tuple[float, float, float] = (0.0, 0.0, 0.0),
    *,
    extra_coordinates: tuple[tuple[float, float, float], ...] = (),
    extra_atomic_numbers: tuple[int, ...] = (),
) -> ComplexSessionInput:
    return _session_input(
        (63, 7, *extra_atomic_numbers),
        (metal_coordinates, donor_coordinates, *extra_coordinates),
        coordination_bonds=((0, 1),),
    )


def test_current_fully_feasible_position_is_bitwise_unchanged() -> None:
    target = _target_distance(7)
    source = _single_donor_input(
        (-target, 0.0, 0.0),
        extra_coordinates=((20.0, 20.0, 20.0),),
        extra_atomic_numbers=(1,),
    )
    session = create_coordination_session(source)
    before = snapshot_structure(session)

    result = place_metal(session, 0)
    after = snapshot_structure(session)

    assert result.status == _ob_native.PlacementStatus.FULLY_FEASIBLE
    assert result.selected_evidence.proposal_kind == (
        _ob_native.PlacementProposalKind.CURRENT
    )
    assert not result.moved
    assert result.candidates_evaluated == 1
    np.testing.assert_array_equal(after.coordinates, before.coordinates)
    np.testing.assert_array_equal(
        after.active_coordination_mask,
        before.active_coordination_mask,
    )


@pytest.mark.parametrize(
    ("ratio", "expected"),
    (
        (0.70, "FULLY_FEASIBLE"),
        (0.70 - 1.0e-6, "INFEASIBLE"),
        (1.50, "FULLY_FEASIBLE"),
        (1.50 + 1.0e-6, "INFEASIBLE"),
    ),
)
def test_reachability_threshold_is_the_existing_070_to_150_contract(
    ratio: float,
    expected: str,
) -> None:
    target = _target_distance(7)
    session = create_coordination_session(
        _single_donor_input((ratio * target, 0.0, 0.0))
    )

    evidence = assess_metal_position(
        session,
        0,
        (ratio * target, 0.0, 0.0),
    )

    assert evidence.status == getattr(_ob_native.PlacementStatus, expected)


def test_center_clash_threshold_and_aabb_broad_phase_are_explicit() -> None:
    target = _target_distance(7)
    carbon_cutoff = 0.55 * _target_distance(6)
    source = _single_donor_input(
        (0.0, 0.0, 0.0),
        (target, 0.0, 0.0),
        extra_coordinates=(
            (0.0, carbon_cutoff + 1.0e-5, 0.0),
            (40.0, 40.0, 40.0),
        ),
        extra_atomic_numbers=(6, 1),
    )
    safe_session = create_coordination_session(source)
    safe = assess_metal_position(safe_session, 0, (0.0, 0.0, 0.0))
    assert safe.status == _ob_native.PlacementStatus.FULLY_FEASIBLE
    assert safe.atom_aabb_rejected_pair_count > 0
    assert safe.atom_pair_count > safe.atom_aabb_rejected_pair_count

    colliding_coordinates = np.array(source.coordinates, copy=True)
    colliding_coordinates[2, 1] = carbon_cutoff - 1.0e-5
    colliding = create_coordination_session(replace(
        source,
        coordinates=colliding_coordinates,
    ))
    blocked = assess_metal_position(colliding, 0, (0.0, 0.0, 0.0))
    assert blocked.status == _ob_native.PlacementStatus.INFEASIBLE
    assert blocked.hard_obstruction_count > 0


def test_shared_donor_endpoint_is_allowed_but_interior_contact_is_not() -> None:
    target = _target_distance(7)
    shared = create_coordination_session(_session_input(
        (63, 7, 6),
        ((-target, 0.0, 0.0), (0.0, 0.0, 0.0), (1.4, 0.0, 0.0)),
        ligand_bonds=((1, 2),),
        coordination_bonds=((0, 1),),
    ))
    shared_evidence = assess_metal_position(
        shared,
        0,
        (-target, 0.0, 0.0),
    )
    assert shared_evidence.donor_paths[0].status == (
        _ob_native.DonorPathStatus.SAFE
    )
    assert not shared_evidence.donor_paths[0].bond_obstructed
    assert np.isinf(
        shared_evidence.donor_paths[0].normalized_bond_clearance
    )
    assert len(shared_evidence.donor_paths[0].approach_angles) == 1
    assert shared_evidence.donor_paths[0].approach_angles[
        0
    ].metal_donor_neighbour_angle_degrees == pytest.approx(180.0)

    overlap = create_coordination_session(_session_input(
        (63, 7, 6),
        ((target, 0.0, 0.0), (0.0, 0.0, 0.0), (4.0, 0.0, 0.0)),
        ligand_bonds=((1, 2),),
        coordination_bonds=((0, 1),),
    ))
    overlap_evidence = assess_metal_position(
        overlap,
        0,
        (target, 0.0, 0.0),
    )
    assert overlap_evidence.donor_paths[0].status != (
        _ob_native.DonorPathStatus.SAFE
    )
    assert overlap_evidence.hard_obstruction_count > 0

    crossing = create_coordination_session(_session_input(
        (63, 7, 6, 6),
        (
            (target, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (1.3, -2.0, 0.0),
            (1.3, 2.0, 0.0),
        ),
        ligand_bonds=((2, 3),),
        coordination_bonds=((0, 1),),
    ))
    crossing_evidence = assess_metal_position(
        crossing,
        0,
        (target, 0.0, 0.0),
    )
    assert crossing_evidence.donor_paths[0].status == (
        _ob_native.DonorPathStatus.BOND_OBSTRUCTION
    )
    assert crossing_evidence.donor_paths[0].bond_obstructed
    assert crossing_evidence.bond_obstruction_count == 1


def test_aabb_rejection_preserves_exact_global_clearance_for_ranking() -> None:
    target = _target_distance(7)
    source = _single_donor_input(
        (-target, 0.0, 0.0),
        extra_coordinates=((40.0, 0.0, 0.0),),
        extra_atomic_numbers=(1,),
    )
    evidence = assess_metal_position(
        create_coordination_session(source),
        0,
        (-target, 0.0, 0.0),
    )

    assert evidence.atom_aabb_rejected_pair_count > 0
    assert np.isfinite(evidence.minimum_normalized_clearance)
    expected_center_clearance = (40.0 + target) / _target_distance(1)
    expected_path_clearance = 40.0 / _target_distance(1)
    assert evidence.minimum_normalized_clearance == pytest.approx(
        min(expected_center_clearance, expected_path_clearance)
    )


def test_primary_status_does_not_hide_other_donor_path_failures() -> None:
    target = _target_distance(7)
    candidate = (2.0 * target, 0.0, 0.0)
    source = _single_donor_input(
        candidate,
        extra_coordinates=((target, 0.0, 0.0),),
        extra_atomic_numbers=(6,),
    )
    evidence = assess_metal_position(
        create_coordination_session(source),
        0,
        candidate,
    )

    donor = evidence.donor_paths[0]
    assert donor.status == _ob_native.DonorPathStatus.OUT_OF_RANGE
    assert not donor.distance_reachable
    assert donor.atom_obstructed
    assert evidence.out_of_range_donor_count == 1
    assert evidence.atom_obstruction_count == 1


def test_ring_relations_preserve_definite_and_undetermined_states() -> None:
    target = _target_distance(7)
    piercing = create_coordination_session(_session_input(
        (63, 7, 6, 6, 6, 6),
        (
            (0.0, 0.0, 0.5 * target),
            (0.0, 0.0, -0.5 * target),
            (-3.0, -3.0, 0.0),
            (3.0, -3.0, 0.0),
            (3.0, 3.0, 0.0),
            (-3.0, 3.0, 0.0),
        ),
        ligand_bonds=((2, 3), (3, 4), (4, 5), (5, 2)),
        coordination_bonds=((0, 1),),
    ))
    pierced = assess_metal_position(
        piercing,
        0,
        (0.0, 0.0, 0.5 * target),
    )
    assert pierced.donor_paths[0].status == (
        _ob_native.DonorPathStatus.RING_PIERCING
    )
    assert pierced.definite_piercing_count == 1

    undetermined = create_coordination_session(_session_input(
        (63, 7, 7, 6, 6, 6, 6),
        (
            (2.0, 3.0, -1.0),
            (2.0, 3.0, 1.0),
            (2.0 - target, 3.0, -1.0),
            (0.0, 0.0, 0.0),
            (10.0, 10.0, 0.0),
            (0.0, 10.0, 0.0),
            (10.0, 0.0, 0.0),
        ),
        ligand_bonds=((3, 4), (4, 5), (5, 6), (6, 3)),
        coordination_bonds=((0, 1), (0, 2)),
    ))
    uncertain = assess_metal_position(
        undetermined,
        0,
        (2.0, 3.0, -1.0),
    )
    assert uncertain.donor_paths[0].status == (
        _ob_native.DonorPathStatus.UNDETERMINED
    )
    assert uncertain.donor_paths[1].status == _ob_native.DonorPathStatus.SAFE
    assert uncertain.status == _ob_native.PlacementStatus.PARTIAL


def test_donor_endpoint_on_ligand_ring_is_not_a_piercing() -> None:
    target = _target_distance(7)
    session = create_coordination_session(_session_input(
        (63, 7, 6, 6, 6),
        (
            (-target, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 2.0, 0.0),
            (0.0, 2.0, 0.0),
        ),
        ligand_bonds=((1, 2), (2, 3), (3, 4), (4, 1)),
        coordination_bonds=((0, 1),),
    ))

    evidence = assess_metal_position(session, 0, (-target, 0.0, 0.0))

    assert evidence.donor_paths[0].status == _ob_native.DonorPathStatus.SAFE
    assert evidence.donor_paths[0].definite_piercing_count == 0
    assert evidence.donor_paths[0].undetermined_relation_count == 0


def test_single_undetermined_donor_is_partial_not_infeasible() -> None:
    target = _target_distance(7)
    source = _session_input(
        (63, 7, 1, 6, 6, 6, 6),
        (
            (2.0, 3.0, -1.0),
            (2.0, 3.0, 1.0),
            (30.0, 30.0, 30.0),
            (0.0, 0.0, 0.0),
            (10.0, 10.0, 0.0),
            (0.0, 10.0, 0.0),
            (10.0, 0.0, 0.0),
        ),
        ligand_bonds=((3, 4), (4, 5), (5, 6), (6, 3)),
        coordination_bonds=((0, 1),),
    )
    evidence = assess_metal_position(
        create_coordination_session(source),
        0,
        (2.0, 3.0, 1.0 - target),
    )

    assert evidence.donor_paths[0].status == (
        _ob_native.DonorPathStatus.UNDETERMINED
    )
    assert evidence.status == _ob_native.PlacementStatus.PARTIAL


def test_large_ligand_ring_is_reported_and_excluded_from_placement_policy() -> None:
    target = _target_distance(7)
    ring_coordinates = tuple(
        (
            5.0 * np.cos(2.0 * np.pi * index / 17),
            5.0 * np.sin(2.0 * np.pi * index / 17),
            0.0,
        )
        for index in range(17)
    )
    source = _session_input(
        (63, 7, *(6 for _ in range(17))),
        (
            (0.0, 0.0, 0.5 * target),
            (0.0, 0.0, -0.5 * target),
            *ring_coordinates,
        ),
        ligand_bonds=tuple(
            (index + 2, ((index + 1) % 17) + 2) for index in range(17)
        ),
        coordination_bonds=((0, 1),),
    )

    session = create_coordination_session(source)
    evidence = assess_metal_position(
        session,
        0,
        tuple(source.coordinates[0]),
    )
    result = place_metal(session, 0)

    assert evidence.excluded_large_cycle_count == 1
    assert result.status == _ob_native.PlacementStatus.FULLY_FEASIBLE
    assert result.excluded_large_cycle_count == 1
    assert "metal_placement_large_cycles_excluded" in result.warning_codes


def test_metal_incident_edges_do_not_change_ligand_skeleton_cycles() -> None:
    target = _target_distance(7)
    metal_cycle_coordinates = tuple(
        (
            30.0 + 5.0 * np.cos(2.0 * np.pi * index / 17),
            5.0 * np.sin(2.0 * np.pi * index / 17),
            0.0,
        )
        for index in range(17)
    )
    organic_triangle = (
        (30.0, 30.0, 0.0),
        (31.5, 30.0, 0.0),
        (30.75, 31.3, 0.0),
    )
    metal_cycle_bonds = (
        (0, 2),
        *((index, index + 1) for index in range(2, 18)),
        (18, 0),
    )
    source = _session_input(
        (63, 7, *(6 for _ in range(20))),
        (
            (0.0, 0.0, 0.0),
            (-target, 0.0, 0.0),
            *metal_cycle_coordinates,
            *organic_triangle,
        ),
        ligand_bonds=(
            *metal_cycle_bonds,
            (19, 20),
            (20, 21),
            (21, 19),
        ),
        coordination_bonds=((0, 1),),
    )

    evidence = assess_metal_position(
        create_coordination_session(source),
        0,
        (0.0, 0.0, 0.0),
    )

    assert evidence.excluded_large_cycle_count == 0
    assert evidence.donor_paths[0].cycle_pair_count == 1


@pytest.mark.parametrize(
    ("obstacle_atomic_number", "metal_indices"),
    ((1, (0,)), (63, (0, 2))),
)
def test_hydrogen_and_other_declared_metals_participate_in_center_clash_checks(
    obstacle_atomic_number: int,
    metal_indices: tuple[int, ...],
) -> None:
    target = _target_distance(7)
    session = create_coordination_session(_session_input(
        (63, 7, obstacle_atomic_number),
        ((0.0, 0.0, 0.0), (target, 0.0, 0.0), (0.4, 0.0, 0.0)),
        coordination_bonds=((0, 1),),
        metal_indices=metal_indices,
    ))

    evidence = assess_metal_position(session, 0, (0.0, 0.0, 0.0))

    assert evidence.status == _ob_native.PlacementStatus.INFEASIBLE
    assert evidence.hard_obstruction_count > 0


@pytest.mark.parametrize(
    ("donor_atomic_numbers", "expected_kind"),
    (
        ((7,), "TARGET_SPHERE"),
        ((7, 8), "SPHERE_INTERSECTION"),
        ((7, 8, 16), "LEAST_SQUARES"),
    ),
)
def test_candidate_generators_are_deterministic(
    donor_atomic_numbers: tuple[int, ...],
    expected_kind: str,
) -> None:
    donor_coordinates = tuple(
        (1.5 * np.cos(2.0 * np.pi * index / len(donor_atomic_numbers)),
         1.5 * np.sin(2.0 * np.pi * index / len(donor_atomic_numbers)),
         0.0)
        for index in range(len(donor_atomic_numbers))
    )
    source = _session_input(
        (63, *donor_atomic_numbers),
        ((0.0, 0.0, 0.0), *donor_coordinates),
        coordination_bonds=tuple(
            (0, index + 1) for index in range(len(donor_atomic_numbers))
        ),
    )
    options = MetalPlacementOptions(retain_candidate_evidence=True)
    first = place_metal(create_coordination_session(source), 0, options=options)
    second = place_metal(create_coordination_session(source), 0, options=options)

    np.testing.assert_array_equal(
        first.selected_coordinates,
        second.selected_coordinates,
    )
    assert first.selected_evidence.proposal_ordinal == (
        second.selected_evidence.proposal_ordinal
    )
    assert any(
        evidence.proposal_kind
        == getattr(_ob_native.PlacementProposalKind, expected_kind)
        for evidence in first.retained_candidates
    )
    if len(donor_atomic_numbers) >= 2:
        assert first.selected_evidence.donor_pairs
        pair = first.selected_evidence.donor_pairs[0]
        assert pair.first_donor_index != pair.second_donor_index
        assert pair.donor_separation_angstrom > 0.0
        assert isinstance(pair.target_shells_intersect, bool)
        assert pair.donor_metal_donor_angle_degrees is not None


def test_placement_requires_coordination_bonds_to_be_inactive() -> None:
    target = _target_distance(7)
    session = create_coordination_session(
        _single_donor_input((-target, 0.0, 0.0))
    )
    set_coordination_active_mask(session, np.asarray([1], dtype=np.uint8))

    evidence = assess_metal_position(session, 0, (-target, 0.0, 0.0))
    assert evidence.status == _ob_native.PlacementStatus.FULLY_FEASIBLE
    with pytest.raises(RuntimeError, match="requires all intended"):
        place_metal(session, 0)


def test_multi_metal_final_recheck_preserves_selected_displacement() -> None:
    target = _target_distance(7)
    source = _session_input(
        (63, 63, 7, 7, 6),
        (
            (0.6607, 0.0, 0.0),
            (50.0, 0.0, 0.0),
            (3.35, 0.0, 0.0),
            (50.0 - target, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        ),
        coordination_bonds=((0, 2), (1, 3)),
        metal_indices=(0, 1),
    )

    report = place_metals(create_coordination_session(source))

    assert len(report.metals) == 2
    assert report.metals[0].moved
    assert report.metals[0].selected_evidence.displacement_angstrom > 0.0


def test_multi_metal_final_recheck_synchronizes_downgrade_warnings() -> None:
    first_target = 1.49 * _target_distance(7)
    second_center = np.array((2.35, 0.0, 0.0))
    donor_radius = _target_distance(7) / np.sqrt(3.0)
    donor_offsets = (
        (1.0, 1.0, 1.0),
        (1.0, -1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
    )
    second_donors = tuple(
        tuple(second_center + donor_radius * np.asarray(offset))
        for offset in donor_offsets
    )
    session = create_coordination_session(_session_input(
        (63, 7, 63, 7, 7, 7, 7),
        (
            (0.0, 0.0, 0.0),
            (first_target, 0.0, 0.0),
            (2.35, 10.0, 0.0),
            *second_donors,
        ),
        coordination_bonds=(
            (0, 1),
            (2, 3),
            (2, 4),
            (2, 5),
            (2, 6),
        ),
        metal_indices=(0, 2),
    ))

    report = place_metals(session)

    first = report.metals[0]
    assert first.status == _ob_native.PlacementStatus.INFEASIBLE
    assert "metal_placement_post_batch_recheck_changed" in first.warning_codes
    assert "metal_placement_infeasible" in first.warning_codes
    assert "metal_placement_partial" not in first.warning_codes
    assert "metal_placement_infeasible" in report.warning_codes


def test_retain_evidence_does_not_change_selection() -> None:
    source = _single_donor_input(
        (0.6607, 0.0, 0.0),
        (3.35, 0.0, 0.0),
        extra_coordinates=((0.0, 0.0, 0.0), (30.0, 0.0, 0.0)),
        extra_atomic_numbers=(6, 1),
    )
    discarded = place_metal(
        create_coordination_session(source),
        0,
        options=MetalPlacementOptions(retain_candidate_evidence=False),
    )
    retained = place_metal(
        create_coordination_session(source),
        0,
        options=MetalPlacementOptions(retain_candidate_evidence=True),
    )

    np.testing.assert_array_equal(
        discarded.selected_coordinates,
        retained.selected_coordinates,
    )
    assert discarded.retained_candidates == []
    assert len(retained.retained_candidates) == retained.candidates_evaluated


def test_no_usable_candidate_retains_original_coordinates_and_warns() -> None:
    source = _single_donor_input(
        (0.4, 0.0, 0.0),
        (3.0, 0.0, 0.0),
        extra_coordinates=((0.0, 0.0, 0.0),),
        extra_atomic_numbers=(6,),
    )
    session = create_coordination_session(source)
    before = snapshot_structure(session)
    result = place_metal(
        session,
        0,
        options=MetalPlacementOptions(maximum_candidate_count=1),
    )

    assert result.status == _ob_native.PlacementStatus.INFEASIBLE
    assert not result.moved
    assert "metal_placement_infeasible" in result.warning_codes
    np.testing.assert_array_equal(
        snapshot_structure(session).coordinates,
        before.coordinates,
    )


def test_python_facade_and_bound_cpp_api_share_one_backend() -> None:
    target = _target_distance(7)
    session = create_coordination_session(
        _single_donor_input((-target, 0.0, 0.0))
    )
    facade = assess_metal_position(session, 0, (-target, 0.0, 0.0))
    direct = _ob_native.assess_metal_position(
        session,
        0,
        (-target, 0.0, 0.0),
        _ob_native.MetalPlacementOptions(),
    )

    assert facade.status == direct.status
    assert facade.proposal_kind == direct.proposal_kind
    assert facade.donor_paths[0].status == direct.donor_paths[0].status
    assert facade.donor_paths[0].distance_ratio == pytest.approx(
        direct.donor_paths[0].distance_ratio,
        abs=0.0,
    )


def test_case109_initial_collision_is_removed_before_any_bond_is_active() -> None:
    """Regression proxy using the diagnosed 0.6607 Angstrom Eu--C clash."""
    source = _session_input(
        (63, 6, 7, 7),
        (
            (0.0, 0.0, 0.0),
            (0.0, 0.6607, 0.0),
            (-1.5, 0.0, 0.0),
            (1.5, 0.0, 0.0),
        ),
        coordination_bonds=((0, 2), (0, 3)),
    )
    session = create_coordination_session(source)
    before = snapshot_structure(session)
    initial = assess_metal_position(session, 0, (0.0, 0.0, 0.0))
    assert initial.status == _ob_native.PlacementStatus.INFEASIBLE
    assert initial.hard_obstruction_count > 0
    assert not np.any(before.active_coordination_mask)

    result = place_metal(session, 0)
    after = snapshot_structure(session)
    assert result.status == _ob_native.PlacementStatus.FULLY_FEASIBLE
    assert result.moved
    assert not np.any(after.active_coordination_mask)
    assert np.linalg.norm(after.coordinates[0] - after.coordinates[1]) >= (
        0.55 * _target_distance(6)
    )
    np.testing.assert_array_equal(
        after.coordinates[1:],
        before.coordinates[1:],
    )


def test_case54_initial_phosphorus_distance_is_not_misreported_as_a_clash() -> None:
    """Phase 6 must not claim the post-relaxation case 54 collapse is fixed."""
    target = _target_distance(16)
    source = _session_input(
        (63, 16, 15),
        ((0.0, 0.0, 0.0), (-target, 0.0, 0.0), (0.0, 1.9756, 0.0)),
        coordination_bonds=((0, 1),),
    )
    session = create_coordination_session(source)
    evidence = assess_metal_position(session, 0, (0.0, 0.0, 0.0))

    assert evidence.status == _ob_native.PlacementStatus.FULLY_FEASIBLE
    assert evidence.hard_obstruction_count == 0


def test_native_geometry_and_forcefield_responsibilities_are_separated() -> None:
    repository = Path(__file__).resolve().parents[3]
    construction = (
        repository / "hotpot/cheminfo/geometry/_native/construction.cpp"
    ).read_text(encoding="utf-8")
    candidates = (
        repository / "hotpot/cheminfo/forcefields/_native/placement_candidates.cpp"
    ).read_text(encoding="utf-8")
    policy = (
        repository / "hotpot/cheminfo/forcefields/_native/placement_policy.cpp"
    ).read_text(encoding="utf-8")

    assert not {"Metal", "Donor", "CovalentRadius", "Reasonable", "Accepted"} & {
        token for token in construction.replace("::", " ").split()
    }
    assert "solve_three_by_three" not in candidates
    assert "fit_point_to_spheres" in candidates
    assert "std::acos" not in policy
    assert "measure_sphere_pair" in policy
