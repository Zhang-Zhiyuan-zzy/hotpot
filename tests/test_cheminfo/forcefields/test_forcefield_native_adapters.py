"""Behavior fence for Python adapters around native complex stages."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pytest
from numpy.typing import NDArray

from hotpot import read_mol
from hotpot.cheminfo.forcefields.coordinates import _perturbed_coordinates
from hotpot.cheminfo.forcefields.native import (
    ComplexOptimizationOptions,
    CoordinationStageOptions,
    create_coordination_session,
    run_complex_workflow,
)
from hotpot.cheminfo.forcefields.native_adapters import (
    NATIVE_WARNING_MESSAGES,
    apply_native_selected_structure,
    coordination_restoration_report,
    forcefield_run_report,
    ingest_native_trajectory,
    native_coordination_offsets,
    native_optimization_offsets,
    native_perturbation_streams,
    native_warning_messages,
)
from hotpot.cheminfo.forcefields.native_packing import (
    ComplexSessionInput,
    pack_complex_session_input,
)
from hotpot.cheminfo.forcefields.trajectory import (
    ForceFieldTrajectory,
    TrajectoryStart,
)


@dataclass(frozen=True)
class _SelectedStructure:
    selected_coordinates: NDArray[np.float64]
    final_active_coordination_mask: NDArray[np.uint8]


def _complex():
    mol = read_mol("[Eu]N", fmt="smi")
    mol.coordinates = np.asarray(
        ((0.0, 0.0, 0.0), (2.4, 0.0, 0.0)),
        dtype=np.float64,
    )
    return mol


def _multi_coordination_complex():
    mol = read_mol("[Eu](N)(O)F", fmt="smi")
    mol.coordinates = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (2.4, 0.0, 0.0),
            (0.0, 2.4, 0.0),
            (0.0, 0.0, 2.4),
        ),
        dtype=np.float64,
    )
    return mol


def _native_workflow_result(
    session_input: ComplexSessionInput,
):
    coordination_options = CoordinationStageOptions(
        attempt_limit=1,
        relaxation_steps=1,
    )
    optimization_options = ComplexOptimizationOptions(
        epochs=1,
        steps_per_epoch=1,
        untangling_attempt_limit=1,
    )
    streams = native_perturbation_streams(
        len(session_input.atomic_numbers),
        2026,
        coordination_options=coordination_options,
        optimization_options=optimization_options,
    )
    return run_complex_workflow(
        create_coordination_session(session_input),
        streams.coordination,
        streams.untangling,
        streams.optimization,
        coordination_options=coordination_options,
        optimization_options=optimization_options,
    )


def test_coordination_offsets_exactly_reproduce_stage_2_rng() -> None:
    coordination_options = CoordinationStageOptions(
        attempt_limit=4,
        perturb_sigma=0.5,
    )

    offsets = native_coordination_offsets(
        4,
        19,
        options=coordination_options,
    )
    rng = np.random.default_rng(19)
    origin = np.zeros((4, 3), dtype=np.float64)
    expected = np.stack(tuple(
        _perturbed_coordinates(origin, sigma=0.5, rng=rng)
        for _ in range(3)
    ))

    np.testing.assert_array_equal(offsets, expected)
    assert not offsets.flags.writeable
    assert np.max(np.abs(offsets)) <= 1.0


def test_optimization_offsets_exactly_reproduce_independent_stage_3_rngs(
) -> None:
    options = ComplexOptimizationOptions(
        epochs=7,
        untangling_attempt_limit=2,
        perturb_interval=2,
        perturb_sigma=0.5,
    )

    streams = native_optimization_offsets(4, 19, options=options)
    origin = np.zeros((4, 3), dtype=np.float64)
    untangling_rng = np.random.default_rng(19)
    expected_untangling = np.stack(tuple(
        _perturbed_coordinates(origin, sigma=0.5, rng=untangling_rng)
        for _ in range(2)
    ))
    optimization_rng = np.random.default_rng(19)
    expected_optimization = np.stack(tuple(
        _perturbed_coordinates(origin, sigma=0.5, rng=optimization_rng)
        for _ in range(3)
    ))

    np.testing.assert_array_equal(streams.untangling, expected_untangling)
    np.testing.assert_array_equal(streams.optimization, expected_optimization)
    assert not streams.untangling.flags.writeable
    assert not streams.optimization.flags.writeable


def test_combined_offset_streams_compose_stage_specific_outputs() -> None:
    coordination_options = CoordinationStageOptions(
        attempt_limit=4,
        perturb_sigma=0.25,
    )
    optimization_options = ComplexOptimizationOptions(
        epochs=7,
        untangling_attempt_limit=2,
        perturb_interval=2,
        perturb_sigma=0.5,
    )

    combined = native_perturbation_streams(
        4,
        19,
        coordination_options=coordination_options,
        optimization_options=optimization_options,
    )
    coordination = native_coordination_offsets(
        4,
        19,
        options=coordination_options,
    )
    optimization = native_optimization_offsets(
        4,
        19,
        options=optimization_options,
    )

    np.testing.assert_array_equal(combined.coordination, coordination)
    np.testing.assert_array_equal(combined.untangling, optimization.untangling)
    np.testing.assert_array_equal(
        combined.optimization,
        optimization.optimization,
    )


def test_warning_mapping_is_complete_and_rejects_unknown_codes() -> None:
    codes = tuple(NATIVE_WARNING_MESSAGES)

    assert native_warning_messages(codes) == tuple(
        NATIVE_WARNING_MESSAGES[code] for code in codes
    )
    assert native_warning_messages((codes[0], codes[0])) == (
        NATIVE_WARNING_MESSAGES[codes[0]],
    )
    with pytest.raises(ValueError, match="unmapped_native_warning"):
        native_warning_messages(("unmapped_native_warning",))


def test_native_stage_results_map_to_existing_python_reports() -> None:
    mol = _complex()
    result = _native_workflow_result(pack_complex_session_input(mol))
    coordination_result = replace(
        result.coordination,
        warning_codes=("coordination_bonds_forced",),
    )
    optimization_result = replace(
        result.optimization,
        warning_codes=("ring_relation_undetermined",),
    )

    coordination = coordination_restoration_report(coordination_result)
    optimization = forcefield_run_report(
        optimization_result,
        requested_forcefield=None,
        effective_forcefield="UFF",
    )

    assert coordination.attempt_limit == coordination_result.attempt_limit
    assert coordination.bond_count == 1
    assert coordination.warning_messages == (
        NATIVE_WARNING_MESSAGES["coordination_bonds_forced"],
    )
    assert coordination.elapsed_seconds == coordination_result.elapsed_seconds
    assert optimization.requested_forcefield is None
    assert optimization.effective_forcefield == "UFF"
    assert optimization.setup_succeeded
    assert optimization.steps_completed is None
    assert optimization.energy_unit == "kJ/mol"
    assert optimization.final_energy == optimization_result.final_energy_kj_mol
    assert optimization.elapsed_seconds == optimization_result.elapsed_seconds
    assert optimization.untangling is not None
    assert optimization.untangling.final_piercing_count == (
        optimization_result.final_piercing_count
    )
    assert optimization.untangling.warning_messages == (
        NATIVE_WARNING_MESSAGES["ring_relation_undetermined"],
    )


def test_topology_blocked_result_does_not_report_setup_success() -> None:
    mol = _complex()
    result = _native_workflow_result(pack_complex_session_input(mol))
    blocked_result = replace(
        result.optimization,
        termination_reason="topology_blocked",
        epochs_completed=0,
        steps_submitted=0,
        initialization_steps=0,
    )

    report = forcefield_run_report(
        blocked_result,
        requested_forcefield="MMFF94",
        effective_forcefield="UFF",
    )

    assert not report.setup_succeeded
    assert report.termination_reason == "topology_blocked"


def test_topology_blocked_tail_overrides_earlier_optimizer_work() -> None:
    mol = _complex()
    result = _native_workflow_result(pack_complex_session_input(mol))
    blocked_result = replace(
        result.optimization,
        termination_reason="topology_blocked",
        epochs_completed=3,
        steps_submitted=300,
        initialization_steps=1,
    )

    report = forcefield_run_report(
        blocked_result,
        requested_forcefield=None,
        effective_forcefield="UFF",
    )

    assert not report.setup_succeeded


@pytest.mark.parametrize('mask', [(0,), (1,)])
def test_selected_structure_updates_coordinates_and_reuses_complete_bond_objects(mask):
    mol = _complex()
    session_input = pack_complex_session_input(mol)
    original_bonds = tuple(mol.bonds)
    original_uids = tuple(bond._uid for bond in original_bonds)
    original_attrs = tuple(bond.attr_dict for bond in original_bonds)
    mol.conformer_add(mol.coordinates)
    frames = mol.conformers._coordinates
    shifted = mol.coordinates + np.asarray((.1, .2, .3))
    apply_native_selected_structure(mol, session_input, _SelectedStructure(
        selected_coordinates=shifted, final_active_coordination_mask=np.asarray(mask, dtype=np.uint8)))
    assert tuple(mol.bonds) == original_bonds
    assert tuple(bond._uid for bond in mol.bonds) == original_uids
    assert tuple(bond.attr_dict for bond in mol.bonds) == original_attrs
    assert mol.conformers._coordinates is frames
    np.testing.assert_array_equal(mol.coordinates, shifted)


@pytest.mark.parametrize('mask', [(0, 0, 0), (0, 1, 1), (1, 1, 1)])
def test_selected_structure_masks_leave_complete_coordination_facts_in_place(mask):
    mol = _multi_coordination_complex()
    original_bonds = tuple(mol.bonds)
    original_uids = tuple(bond._uid for bond in original_bonds)
    original_attrs = tuple(bond.attr_dict for bond in original_bonds)
    session_input = pack_complex_session_input(mol)
    shifted = mol.coordinates + np.asarray((.2, .3, .4))
    apply_native_selected_structure(mol, session_input, _SelectedStructure(
        selected_coordinates=shifted, final_active_coordination_mask=np.asarray(mask, dtype=np.uint8)))
    assert tuple(mol.bonds) == original_bonds
    assert tuple(bond._uid for bond in mol.bonds) == original_uids
    assert tuple(bond.attr_dict for bond in mol.bonds) == original_attrs
    assert all(mol.bond(bond.atom1.idx, bond.atom2.idx) is bond for bond in original_bonds)
    np.testing.assert_array_equal(mol.coordinates, shifted)


def test_native_batch_ingests_into_an_existing_python_trajectory() -> None:
    mol = _complex()
    session_input = pack_complex_session_input(mol)
    result = _native_workflow_result(session_input)
    trajectory = ForceFieldTrajectory.from_molecule(
        mol,
        start=TrajectoryStart.COORDINATION_RESTORATION,
    )

    frame_indices = ingest_native_trajectory(
        trajectory,
        result.trajectory,
        session_input,
    )

    assert len(frame_indices) == result.trajectory.frame_count
    assert frame_indices == tuple(range(result.trajectory.frame_count))
    assert trajectory.selected_index == result.trajectory.selected_frame_index
    assert trajectory.terminal_index == result.trajectory.terminal_frame_index
    assert trajectory.selected_frame is not None
    np.testing.assert_array_equal(
        trajectory.coordinates(trajectory.selected_index),
        result.selected_coordinates,
    )


@pytest.mark.parametrize('invalid', [
    'short_mask', 'matrix_mask', 'nonbinary_mask', 'short_coordinates', 'nonfinite_coordinates',
    'atom_mapping', 'coordination_mapping', 'duplicate_mapping', 'reversed_mapping', 'order_mapping',
])
def test_invalid_selected_publication_rejects_before_any_factual_change(invalid):
    from tests.test_cheminfo._stage5_fixtures import factual_state
    mol = _multi_coordination_complex()
    session_input = pack_complex_session_input(mol)
    coordinates = mol.coordinates + 1.
    mask = np.ones(3, dtype=np.uint8)
    if invalid == 'short_mask':
        mask = mask[:2]
    elif invalid == 'matrix_mask':
        mask = mask.reshape(3, 1)
    elif invalid == 'nonbinary_mask':
        mask[1] = 2
    elif invalid == 'short_coordinates':
        coordinates = coordinates[:-1]
    elif invalid == 'nonfinite_coordinates':
        coordinates[1, 0] = np.nan
    elif invalid == 'atom_mapping':
        session_input = replace(session_input, atomic_numbers=np.zeros(4, dtype=np.int32))
    elif invalid == 'coordination_mapping':
        session_input = replace(session_input, intended_coordination_bonds=np.zeros((2, 2), dtype=np.int32))
    elif invalid == 'duplicate_mapping':
        rows = session_input.intended_coordination_bonds.copy()
        rows[1] = rows[0]
        session_input = replace(session_input, intended_coordination_bonds=rows)
    elif invalid == 'reversed_mapping':
        session_input = replace(session_input, intended_coordination_bonds=session_input.intended_coordination_bonds[:, ::-1])
    else:
        session_input = replace(session_input, intended_coordination_orders=session_input.intended_coordination_orders + 1.)
    before = factual_state(mol)
    atoms, bonds, index = tuple(mol.atoms), tuple(mol.bonds), mol._topology_index
    with pytest.raises((ValueError, RuntimeError)):
        apply_native_selected_structure(mol, session_input, _SelectedStructure(coordinates, mask))
    assert factual_state(mol) == before
    assert tuple(mol.atoms) == atoms and tuple(mol.bonds) == bonds
    assert mol._topology_index is index


def test_session_packing_sorts_operation_rows_without_reordering_complete_facts():
    from hotpot import BondKind, Molecule
    mol = Molecule()
    for number in (30, 7, 8, 29):
        mol.create_atom(atomic_number=number)
    first = mol.add_bond(2, 0, bond_order=1.25, bond_kind=BondKind.DATIVE,
                         bond_direction='atom1_to_atom2')
    ligand = mol.add_bond(1, 2, bond_order=2.)
    second = mol.add_bond(0, 1, bond_order=1.)
    metal_metal = mol.add_bond(0, 3, bond_order=1.)
    bonds = tuple(mol.bonds)
    session_input = pack_complex_session_input(mol)
    np.testing.assert_array_equal(session_input.intended_coordination_bonds, ((0, 1), (0, 2)))
    np.testing.assert_array_equal(session_input.intended_coordination_orders, (1., 1.25))
    np.testing.assert_array_equal(session_input.ligand_bond_indices, ((1, 2), (0, 3)))
    np.testing.assert_array_equal(session_input.ligand_bond_orders, (2., 1.))
    assert tuple(mol.bonds) == bonds == (first, ligand, second, metal_metal)
    assert first.bond_kind is BondKind.DATIVE and first.bond_direction == 'atom1_to_atom2'
