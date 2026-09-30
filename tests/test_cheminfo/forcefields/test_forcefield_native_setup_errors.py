"""Typed setup-failure contracts at the native Python boundary."""

from __future__ import annotations

import numpy as np
import pytest

from hotpot.cheminfo.forcefields.native import (
    ComplexOptimizationOptions,
    CoordinationStageOptions,
    create_coordination_session,
    create_optimization_session,
    optimize_complex,
    restore_coordination,
    run_complex_workflow,
)
from hotpot.cheminfo.forcefields.native_packing import ComplexSessionInput
from hotpot.cheminfo.obWrappers import _ob_native


UNKNOWN_FORCEFIELD = "not-a-forcefield"


def _complex() -> ComplexSessionInput:
    return ComplexSessionInput(
        schema_version=1,
        atomic_numbers=np.asarray((63, 7), dtype=np.int32),
        formal_charges=np.zeros(2, dtype=np.int32),
        partial_charges=np.zeros(2, dtype=np.float64),
        coordinates=np.asarray(
            ((0.0, 0.0, 0.0), (2.4, 0.0, 0.0)),
            dtype=np.float64,
        ),
        atom_aromatic=np.zeros(2, dtype=np.uint8),
        ligand_bond_indices=np.empty((0, 2), dtype=np.int32),
        ligand_bond_orders=np.empty(0, dtype=np.float64),
        ligand_bond_kinds=np.empty(0, dtype=np.uint8),
        ligand_bond_aromatic=np.empty(0, dtype=np.uint8),
        metal_indices=np.asarray((0,), dtype=np.int32),
        intended_coordination_bonds=np.asarray(((0, 1),), dtype=np.int32),
        intended_coordination_orders=np.ones(1, dtype=np.float64),
        intended_coordination_kinds=np.full(1, 6, dtype=np.uint8),
        unit_cell=None,
    )


def _offsets(frame_count: int) -> np.ndarray:
    return np.zeros((frame_count, 2, 3), dtype=np.float64)


def _assert_setup_error(
    error: _ob_native.ForceFieldSetupError,
    workflow_stage: str,
) -> None:
    assert error.forcefield == UNKNOWN_FORCEFIELD
    assert error.stage == "lookup"
    assert error.workflow_stage == workflow_stage


def test_stage_two_setup_failure_has_typed_context() -> None:
    with pytest.raises(_ob_native.ForceFieldSetupError) as caught:
        restore_coordination(
            create_coordination_session(_complex()),
            _offsets(0),
            options=CoordinationStageOptions(
                forcefield=UNKNOWN_FORCEFIELD,
                attempt_limit=1,
                relaxation_steps=1,
            ),
        )

    _assert_setup_error(caught.value, "coordination_restoration")
    assert caught.value.completed_coordination is None


def test_stage_three_setup_failure_has_typed_context() -> None:
    with pytest.raises(_ob_native.ForceFieldSetupError) as caught:
        optimize_complex(
            create_optimization_session(_complex()),
            _offsets(1),
            _offsets(0),
            options=ComplexOptimizationOptions(
                forcefield=UNKNOWN_FORCEFIELD,
                epochs=1,
                steps_per_epoch=1,
                untangling_attempt_limit=1,
            ),
        )

    _assert_setup_error(caught.value, "complex_optimization")
    assert caught.value.completed_coordination is None


def test_workflow_stage_three_failure_carries_completed_coordination() -> None:
    with pytest.raises(_ob_native.ForceFieldSetupError) as caught:
        run_complex_workflow(
            create_coordination_session(_complex()),
            _offsets(0),
            _offsets(1),
            _offsets(0),
            coordination_options=CoordinationStageOptions(
                attempt_limit=1,
                relaxation_steps=1,
            ),
            optimization_options=ComplexOptimizationOptions(
                forcefield=UNKNOWN_FORCEFIELD,
                epochs=1,
                steps_per_epoch=1,
                untangling_attempt_limit=1,
            ),
        )

    _assert_setup_error(caught.value, "complex_optimization")
    completed = caught.value.completed_coordination
    assert isinstance(completed, _ob_native.CoordinationStageResult)
    assert completed.bond_count == 1
    assert completed.final_active_coordination_mask.tolist() == [1]


def test_workflow_stage_two_failure_has_no_completed_coordination() -> None:
    with pytest.raises(_ob_native.ForceFieldSetupError) as caught:
        run_complex_workflow(
            create_coordination_session(_complex()),
            _offsets(0),
            _offsets(1),
            _offsets(0),
            coordination_options=CoordinationStageOptions(
                forcefield=UNKNOWN_FORCEFIELD,
                attempt_limit=1,
                relaxation_steps=1,
            ),
            optimization_options=ComplexOptimizationOptions(
                epochs=1,
                steps_per_epoch=1,
                untangling_attempt_limit=1,
            ),
        )

    _assert_setup_error(caught.value, "coordination_restoration")
    assert caught.value.completed_coordination is None


def _molecule_data() -> _ob_native.MoleculeData:
    return _ob_native.MoleculeData(
        1,
        np.asarray((6,), dtype=np.int32),
        np.zeros(1, dtype=np.int32),
        np.zeros(1, dtype=np.float64),
        np.zeros((1, 3), dtype=np.float64),
        np.zeros(1, dtype=np.uint8),
        np.empty((0, 2), dtype=np.int32),
        np.empty(0, dtype=np.float64),
        np.empty(0, dtype=np.uint8),
        np.empty(0, dtype=np.uint8),
        None,
    )


def test_single_optimization_uses_the_same_setup_error_contract() -> None:
    with pytest.raises(_ob_native.ForceFieldSetupError) as caught:
        _ob_native.single_optimize(_molecule_data(), UNKNOWN_FORCEFIELD, 1)

    _assert_setup_error(caught.value, "single_optimization")
    assert caught.value.completed_coordination is None


def test_optimizer_uses_the_same_setup_error_contract() -> None:
    with pytest.raises(_ob_native.ForceFieldSetupError) as caught:
        _ob_native.optimize(
            _molecule_data(),
            UNKNOWN_FORCEFIELD,
            "conjugate",
            1,
            1,
        )

    _assert_setup_error(caught.value, "optimization")
    assert caught.value.completed_coordination is None
