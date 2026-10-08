"""Low-level contracts of the native Open Babel force-field engine."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from importlib import import_module

import numpy as np
import pytest

from hotpot import read_mol
from hotpot.cheminfo.obWrappers.packing import _pack_molecule


native = import_module("hotpot.cheminfo.obWrappers._ob_native")


def _native_molecule(smiles: str = "CC"):
    mol = read_mol(smiles)
    buffers = _pack_molecule(mol)
    return mol, native.MoleculeData(
        schema_version=buffers.schema_version,
        atomic_numbers=buffers.atomic_numbers,
        formal_charges=buffers.formal_charges,
        partial_charges=buffers.partial_charges,
        coordinates=buffers.coordinates,
        atom_aromatic=buffers.atom_aromatic,
        bond_indices=buffers.bond_indices,
        bond_orders=buffers.bond_orders,
        bond_kinds=buffers.bond_kinds,
        bond_aromatic=buffers.bond_aromatic,
        unit_cell=buffers.unit_cell,
    )


def _native_molecule_with_coordinates(coordinates: np.ndarray):
    mol = read_mol("CC")
    mol.coordinates = coordinates
    buffers = _pack_molecule(mol)
    return native.MoleculeData(
        schema_version=buffers.schema_version,
        atomic_numbers=buffers.atomic_numbers,
        formal_charges=buffers.formal_charges,
        partial_charges=buffers.partial_charges,
        coordinates=buffers.coordinates,
        atom_aromatic=buffers.atom_aromatic,
        bond_indices=buffers.bond_indices,
        bond_orders=buffers.bond_orders,
        bond_kinds=buffers.bond_kinds,
        bond_aromatic=buffers.bond_aromatic,
        unit_cell=buffers.unit_cell,
    )


def _optimize(molecule, **overrides):
    options = {
        "forcefield": "UFF",
        "algorithm": "steepest",
        "epochs": 1,
        "steps_per_epoch": 1,
    }
    options.update(overrides)
    return native.optimize(molecule, **options)


@pytest.mark.parametrize(
    ("operation", "message"),
    (
        (
            lambda molecule: native.inspect_rules(
                molecule,
                native.RuleStage.PRE_BUILD,
                singularity_threshold=float("nan"),
            ),
            "singularity_threshold",
        ),
        (
            lambda molecule: native.single_optimize(
                molecule,
                "UFF",
                0,
            ),
            "steps",
        ),
        (
            lambda molecule: native.single_optimize(
                molecule,
                "",
                1,
            ),
            "forcefield",
        ),
        (
            lambda molecule: _optimize(
                molecule,
                forcefield="",
            ),
            "forcefield",
        ),
        (
            lambda molecule: _optimize(
                molecule,
                epochs=2**31,
            ),
            "step limit",
        ),
        (
            lambda molecule: _optimize(
                molecule,
                vdw_cutoff_start=float("nan"),
            ),
            "vdw cutoffs",
        ),
        (
            lambda molecule: _optimize(
                molecule,
                energy_tolerance=float("inf"),
            ),
            "energy_tolerance",
        ),
        (
            lambda molecule: _optimize(
                molecule,
                stopping_window=1,
                maximum_energy_change_kj_mol=-1.0,
            ),
            "stopping thresholds",
        ),
    ),
)
def test_native_entry_points_reject_invalid_shared_parameters(
    operation,
    message,
):
    _, molecule = _native_molecule()

    with pytest.raises(ValueError, match=message):
        operation(molecule)


def test_forcefield_setup_failure_is_structured():
    _, molecule = _native_molecule()

    with pytest.raises(native.ForceFieldSetupError) as caught:
        native.single_optimize(molecule, "not-a-forcefield", 1)

    assert caught.value.forcefield == "not-a-forcefield"
    assert caught.value.stage == "lookup"
    assert issubclass(native.ForceFieldEnergyUnitError, RuntimeError)
    assert issubclass(native.OptimizationFrameError, RuntimeError)


@pytest.mark.parametrize(
    "level_name",
    ("OPENBABEL", "FAST", "BALANCED", "STRICT"),
)
def test_native_result_records_requested_convergence_level(level_name):
    _, molecule = _native_molecule()
    level = getattr(native.ConvergenceLevel, level_name)

    result = _optimize(molecule, convergence_level=level)

    assert result.convergence_level == level


def test_native_optimizer_defaults_to_fast_convergence() -> None:
    _, molecule = _native_molecule()

    result = _optimize(molecule)

    assert result.convergence_level == native.ConvergenceLevel.FAST


@pytest.mark.parametrize("retain_frames", (False, True))
def test_nonfinite_backend_state_returns_terminal_evidence(retain_frames):
    molecule = _native_molecule_with_coordinates(
        np.asarray(
            ((1.0e200, 0.0, 0.0), (-1.0e200, 0.0, 0.0)),
            dtype=np.float64,
        )
    )

    result = _optimize(
        molecule,
        epochs=5,
        steps_per_epoch=2,
        retain_frames=retain_frames,
        retain_epoch_history=True,
    )

    assert result.epochs_completed == 1
    assert not result.terminal_converged
    assert result.termination_reason in {
        "nonfinite_coordinates",
        "nonfinite_energy",
        "nonfinite_gradients",
        "explosion_detected",
    }
    assert result.terminal_coordinates.shape == (2, 3)
    assert len(result.epoch_energies) == 1
    assert not np.isfinite(result.final_energy) or result.exploded
    if retain_frames:
        assert len(result.frames) == 1
        assert np.allclose(
            result.frames[0].coordinates,
            result.terminal_coordinates,
            equal_nan=True,
        )
    else:
        assert result.frames == []


def test_seed_and_rule_inspection_share_a_thread_safe_native_boundary():
    _, molecule = _native_molecule("CCC")

    def seed_and_inspect(seed: int) -> int:
        native.seed_random(seed)
        report = native.inspect_rules(
            molecule,
            native.RuleStage.PRE_FORCEFIELD_SETUP,
        )
        return len(report.applications)

    with ThreadPoolExecutor(max_workers=4) as executor:
        counts = tuple(executor.map(seed_and_inspect, range(32)))

    assert counts == (0,) * 32
