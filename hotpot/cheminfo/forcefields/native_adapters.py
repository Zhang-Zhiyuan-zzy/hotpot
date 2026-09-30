"""Adapters between native complex stages and public force-field contracts.

This module owns Python workflow semantics at the native boundary.  The
low-level :mod:`.native` module remains limited to typed native calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    cast,
)

import numpy as np
from numpy.typing import NDArray

from .contracts import (
    CoordinationBondRestorationReport,
    ForceFieldRunReport,
    RingUntanglingReport,
    TerminationReason,
)
from .coordinates import _perturbed_coordinates
from .native import ComplexOptimizationOptions, CoordinationStageOptions
from .native_packing import ComplexSessionInput, _coordination_bonds
from .native_reports import (
    ComplexOptimizationResult,
    CoordinationStageResult,
    NativeTrajectoryBatch,
)
from .trajectory import ForceFieldTrajectory, ForceFieldTrajectoryArchive

if TYPE_CHECKING:
    from ..core import Bond, Molecule


__all__ = (
    "NATIVE_WARNING_MESSAGES",
    "NativeOptimizationPerturbationStreams",
    "NativePerturbationStreams",
    "apply_native_selected_structure",
    "coordination_restoration_report",
    "forcefield_run_report",
    "ingest_native_trajectory",
    "native_coordination_offsets",
    "native_optimization_offsets",
    "native_perturbation_streams",
    "native_warning_messages",
    "ring_untangling_report",
)


NATIVE_WARNING_MESSAGES: Mapping[str, str] = MappingProxyType({
    "metal_placement_partial": (
        "Metal placement found only a partially acceptable position."
    ),
    "metal_placement_infeasible": (
        "Metal placement found no geometrically feasible position."
    ),
    "metal_placement_large_cycles_excluded": (
        "Metal placement excluded rings larger than the configured actionable "
        "size."
    ),
    "metal_placement_post_batch_recheck_changed": (
        "A metal placement changed status when rechecked against the final "
        "batch geometry."
    ),
    "coordination_large_cycles_excluded": (
        "Coordination-bond screening excluded rings larger than the configured "
        "actionable size."
    ),
    "coordination_relation_undetermined": (
        "A coordination-bond and ring relation could not be determined "
        "mathematically."
    ),
    "coordination_bonds_forced": (
        "One or more coordination bonds were restored after no nonpiercing "
        "restoration path was found."
    ),
    "ring_piercing_has_no_opening_edge": (
        "Confirmed ring piercing remains, but no eligible ring-opening edge "
        "was available."
    ),
    "ring_relation_undetermined": (
        "A final bond-ring relation could not be determined mathematically."
    ),
    "ring_untangling_nonfinite_frame": (
        "Ring untangling produced a non-finite trial frame; the trial was "
        "rejected."
    ),
    "ring_untangling_attempt_limit_reached": (
        "Confirmed ring piercing remains after the untangling attempt limit."
    ),
    "ring_piercing_blocks_complex_optimization": (
        "Confirmed ring piercing prevented numerical complex optimization."
    ),
})


class _SelectedStructureResult(Protocol):
    selected_coordinates: NDArray[np.float64]
    final_active_coordination_mask: NDArray[np.uint8]


@dataclass(frozen=True)
class NativeOptimizationPerturbationStreams:
    """Independent deterministic offset schedules for native Stage 3."""

    untangling: NDArray[np.float64]
    optimization: NDArray[np.float64]


@dataclass(frozen=True)
class NativePerturbationStreams:
    """Independent deterministic offset schedules for native Stages 2 and 3."""

    coordination: NDArray[np.float64]
    untangling: NDArray[np.float64]
    optimization: NDArray[np.float64]


def _offset_schedule(
    atom_count: int,
    frame_count: int,
    sigma: float,
    seed: Optional[int],
) -> NDArray[np.float64]:
    if frame_count == 0:
        offsets = np.empty((0, atom_count, 3), dtype=np.float64)
    else:
        origin = np.zeros((atom_count, 3), dtype=np.float64)
        rng = np.random.default_rng(seed)
        offsets = np.ascontiguousarray(
            tuple(
                _perturbed_coordinates(origin, sigma=sigma, rng=rng)
                for _ in range(frame_count)
            ),
            dtype=np.float64,
        )
    offsets.setflags(write=False)
    return offsets


def native_coordination_offsets(
    atom_count: int,
    seed: Optional[int],
    *,
    options: CoordinationStageOptions = CoordinationStageOptions(),
) -> NDArray[np.float64]:
    """Generate the deterministic perturbation offsets used by Stage 2."""
    return _offset_schedule(
        atom_count,
        options.attempt_limit - 1,
        options.perturb_sigma,
        seed,
    )


def native_optimization_offsets(
    atom_count: int,
    seed: Optional[int],
    *,
    options: ComplexOptimizationOptions = ComplexOptimizationOptions(),
) -> NativeOptimizationPerturbationStreams:
    """Generate the independent untangling and optimization Stage 3 offsets."""
    optimization_count = (
        0
        if options.perturb_interval is None
        else (options.epochs - 1) // options.perturb_interval
    )
    return NativeOptimizationPerturbationStreams(
        untangling=_offset_schedule(
            atom_count,
            options.untangling_attempt_limit,
            options.perturb_sigma,
            seed,
        ),
        optimization=_offset_schedule(
            atom_count,
            optimization_count,
            options.perturb_sigma,
            seed,
        ),
    )


def native_perturbation_streams(
    atom_count: int,
    seed: Optional[int],
    *,
    coordination_options: CoordinationStageOptions = CoordinationStageOptions(),
    optimization_options: ComplexOptimizationOptions = (
        ComplexOptimizationOptions()
    ),
) -> NativePerturbationStreams:
    """Reproduce the three independent random streams used by Python workflows."""
    coordination = native_coordination_offsets(
        atom_count,
        seed,
        options=coordination_options,
    )
    optimization = native_optimization_offsets(
        atom_count,
        seed,
        options=optimization_options,
    )
    return NativePerturbationStreams(
        coordination=coordination,
        untangling=optimization.untangling,
        optimization=optimization.optimization,
    )


def native_warning_messages(codes: Sequence[str]) -> Tuple[str, ...]:
    """Translate native warning codes, rejecting an incomplete mapping."""
    unknown_codes = tuple(
        code for code in codes if code not in NATIVE_WARNING_MESSAGES
    )
    if unknown_codes:
        joined = ", ".join(dict.fromkeys(unknown_codes))
        raise ValueError(f"Unknown native force-field warning code(s): {joined}")
    return tuple(dict.fromkeys(NATIVE_WARNING_MESSAGES[code] for code in codes))


def coordination_restoration_report(
    result: CoordinationStageResult,
) -> CoordinationBondRestorationReport:
    """Map one native Stage 2 result to the established Python report."""
    return CoordinationBondRestorationReport(
        attempt_limit=result.attempt_limit,
        attempts_completed=result.attempts_completed,
        bond_count=result.bond_count,
        metal_relocation_attempt_count=result.metal_relocation_attempt_count,
        relocated_metal_indices=result.relocated_metal_indices,
        infeasible_metal_indices=result.infeasible_metal_indices,
        forced_bond_keys=result.forced_bond_keys,
        rejected_piercing_trial_count=result.rejected_piercing_trial_count,
        undetermined_trial_count=result.undetermined_trial_count,
        excluded_ring_observation_count=(
            result.excluded_ring_observation_count
        ),
        warning_messages=native_warning_messages(result.warning_codes),
        elapsed_seconds=result.elapsed_seconds,
    )


def ring_untangling_report(
    result: ComplexOptimizationResult,
) -> RingUntanglingReport:
    """Map native Stage 3 topology observations to the Python ring report."""
    return RingUntanglingReport(
        attempt_limit=result.untangling_attempt_limit,
        attempts_completed=result.untangling_attempts_completed,
        initial_piercing_count=result.initial_piercing_count,
        final_piercing_count=result.final_piercing_count,
        minimum_piercing_count=result.minimum_piercing_count,
        resolved=result.untangling_resolved,
        warning_messages=native_warning_messages(result.warning_codes),
    )


def forcefield_run_report(
    result: ComplexOptimizationResult,
    *,
    requested_forcefield: Optional[str],
    effective_forcefield: str,
    trajectory: Optional[ForceFieldTrajectoryArchive] = None,
) -> ForceFieldRunReport:
    """Map one native Stage 3 result to the established optimizer report."""
    setup_succeeded = result.termination_reason != "topology_blocked"
    return ForceFieldRunReport(
        requested_forcefield=requested_forcefield,
        effective_forcefield=effective_forcefield,
        setup_succeeded=setup_succeeded,
        converged=result.converged,
        epochs_completed=result.epochs_completed,
        steps_submitted=result.steps_submitted,
        initialization_steps=result.initialization_steps,
        steps_completed=None,
        final_energy=result.final_energy_kj_mol,
        best_energy=result.best_energy_kj_mol,
        energy_unit="kJ/mol",
        rms_gradient=result.rms_gradient_kj_mol_angstrom,
        max_gradient=result.max_gradient_kj_mol_angstrom,
        exploded=result.exploded,
        backend_energy_unit=result.backend_energy_unit,
        gradient_unit="kJ/(mol*angstrom)",
        energy_changes=result.energy_changes,
        max_displacements=result.max_displacements,
        best_epoch=result.best_epoch,
        selected_segment_epochs_completed=(
            result.selected_segment_epochs_completed
        ),
        epoch_energies=result.epoch_energies,
        termination_reason=cast(
            TerminationReason,
            result.termination_reason,
        ),
        terminal_converged=result.terminal_converged,
        untangling=ring_untangling_report(result),
        trajectory=trajectory,
        elapsed_seconds=result.elapsed_seconds,
    )


def _ordered_coordination_bonds(
    mol: "Molecule",
    session_input: ComplexSessionInput,
) -> Tuple["Bond", ...]:
    atom_rows = {id(atom): row for row, atom in enumerate(mol.atoms)}
    by_key = {
        tuple(sorted((atom_rows[id(bond.atom1)], atom_rows[id(bond.atom2)]))): bond
        for bond in _coordination_bonds(mol)
    }
    return tuple(
        by_key[tuple(sorted((int(indices[0]), int(indices[1]))))]
        for indices in session_input.intended_coordination_bonds
    )


def apply_native_selected_structure(
    mol: "Molecule",
    session_input: ComplexSessionInput,
    result: _SelectedStructureResult,
) -> None:
    """Apply native selected coordinates and coordination topology in place."""
    ordered_bonds = _ordered_coordination_bonds(mol, session_input)
    active_bond_ids = {id(bond) for bond in mol.bonds}
    bonds_to_hide = tuple(
        bond
        for bond, active in zip(
            ordered_bonds,
            result.final_active_coordination_mask,
        )
        if not active and id(bond) in active_bond_ids
    )
    hidden_bond_ids = {id(bond) for bond in mol._hided_metal_bonds}
    bonds_to_restore = tuple(
        bond
        for bond, active in zip(
            ordered_bonds,
            result.final_active_coordination_mask,
        )
        if active and id(bond) in hidden_bond_ids
    )
    if bonds_to_hide:
        mol.hide_bonds(*bonds_to_hide, clear_conformers=False)
    if bonds_to_restore:
        mol.restore_bonds(*bonds_to_restore, clear_conformers=False)
    mol.coordinates = result.selected_coordinates


def ingest_native_trajectory(
    trajectory: ForceFieldTrajectory,
    batch: NativeTrajectoryBatch,
    session_input: ComplexSessionInput,
) -> Tuple[int, ...]:
    """Append one native batch to an existing Python trajectory."""
    return trajectory.ingest_native_batch(batch, session_input)
