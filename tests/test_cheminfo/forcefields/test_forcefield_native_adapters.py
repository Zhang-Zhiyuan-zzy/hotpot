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
    assert optimization.requested_forcefield is None
    assert optimization.effective_forcefield == "UFF"
    assert optimization.setup_succeeded
    assert optimization.steps_completed is None
    assert optimization.energy_unit == "kJ/mol"
    assert optimization.final_energy == optimization_result.final_energy_kj_mol
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


def test_selected_structure_updates_coordinates_and_reuses_bond_objects() -> None:
    mol = _complex()
    session_input = pack_complex_session_input(mol)
    coordination_bond = next(
        bond for bond in mol.bonds if bond.is_metal_ligand_bond
    )
    mol.conformer_add(mol.coordinates)
    conformer_count = mol.conformers_number
    shifted = mol.coordinates + np.asarray((0.1, 0.2, 0.3))

    apply_native_selected_structure(
        mol,
        session_input,
        _SelectedStructure(
            selected_coordinates=shifted,
            final_active_coordination_mask=np.zeros(1, dtype=np.uint8),
        ),
    )

    assert all(bond is not coordination_bond for bond in mol.bonds)
    assert any(bond is coordination_bond for bond in mol._hided_metal_bonds)
    assert mol.conformers_number == conformer_count
    np.testing.assert_array_equal(mol.coordinates, shifted)

    apply_native_selected_structure(
        mol,
        session_input,
        _SelectedStructure(
            selected_coordinates=shifted + 1.0,
            final_active_coordination_mask=np.ones(1, dtype=np.uint8),
        ),
    )

    assert any(bond is coordination_bond for bond in mol.bonds)
    assert all(
        bond is not coordination_bond for bond in mol._hided_metal_bonds
    )
    assert mol.conformers_number == conformer_count
    np.testing.assert_array_equal(mol.coordinates, shifted + 1.0)


def test_selected_structure_applies_mixed_coordination_mask_in_place() -> None:
    mol = _multi_coordination_complex()
    original_bonds = tuple(mol.bonds)
    coordination_bonds = tuple(
        bond for bond in original_bonds if bond.is_metal_ligand_bond
    )
    mol.conformer_add(mol.coordinates)
    mol.hide_bonds(coordination_bonds[1], clear_conformers=False)
    conformer_count = mol.conformers_number
    session_input = pack_complex_session_input(mol)
    shifted = mol.coordinates + np.asarray((0.2, 0.3, 0.4))

    apply_native_selected_structure(
        mol,
        session_input,
        _SelectedStructure(
            selected_coordinates=shifted,
            final_active_coordination_mask=np.asarray(
                (0, 1, 1),
                dtype=np.uint8,
            ),
        ),
    )

    active_bond_ids = {id(bond) for bond in mol.bonds}
    hidden_bond_ids = {id(bond) for bond in mol._hided_metal_bonds}
    assert id(coordination_bonds[0]) in hidden_bond_ids
    assert id(coordination_bonds[1]) in active_bond_ids
    assert id(coordination_bonds[2]) in active_bond_ids
    assert {id(bond) for bond in coordination_bonds} == (
        active_bond_ids | hidden_bond_ids
    )
    assert all(bond in original_bonds for bond in mol.bonds)
    assert all(bond in original_bonds for bond in mol._hided_metal_bonds)
    assert mol.conformers_number == conformer_count
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
