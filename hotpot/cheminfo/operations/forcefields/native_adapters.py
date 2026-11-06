"""Adapters between native complex stages and public force-field contracts.

This module owns Python workflow semantics at the native boundary.  The
low-level :mod:`hotpot.cheminfo.adapters.native.forcefields.native` module
remains limited to typed native calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import (
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
from hotpot.cheminfo.adapters.native.forcefields.native import (
    ComplexOptimizationOptions,
    CoordinationStageOptions,
)
from hotpot.cheminfo.adapters.native.forcefields.contracts import ComplexSessionInput
from hotpot.cheminfo.obWrappers.packing import _BOND_KIND_CODES
from hotpot.cheminfo.adapters.structure.forcefields import _coordination_bonds
from hotpot.cheminfo.adapters.native.forcefields.reports import (
    ComplexOptimizationResult,
    CoordinationStageResult,
    NativeTrajectoryBatch,
)
from .trajectory import ForceFieldTrajectory, ForceFieldTrajectoryArchive


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
        attempts_completed=result.attempts_completed,
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
        attempts_completed=result.untangling_attempts_completed,
        initial_piercing_count=result.initial_piercing_count,
        final_piercing_count=result.final_piercing_count,
        minimum_piercing_count=result.minimum_piercing_count,
        warning_messages=native_warning_messages(result.warning_codes),
    )


def forcefield_run_report(
    result: ComplexOptimizationResult,
    *,
    effective_forcefield: str,
    trajectory: Optional[ForceFieldTrajectoryArchive] = None,
) -> ForceFieldRunReport:
    """Map one native Stage 3 result to the established optimizer report."""
    setup_succeeded = result.termination_reason != "topology_blocked"
    return ForceFieldRunReport(
        effective_forcefield=effective_forcefield,
        setup_succeeded=setup_succeeded,
        converged=result.converged,
        epochs_completed=result.epochs_completed,
        steps_submitted=result.steps_submitted,
        initialization_steps=result.initialization_steps,
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
    atom_rows = {id(atom): row for row, atom in enumerate(mol._atoms)}
    bonds = _coordination_bonds(mol)
    indices = np.asarray(session_input.intended_coordination_bonds)
    if indices.shape != (len(bonds), 2) or indices.dtype.kind not in "iu":
        raise ValueError("Native coordination mapping must align with complete Bond rows")
    if np.any(indices < 0) or np.any(indices >= len(atom_rows)):
        raise ValueError("Native coordination mapping contains an invalid Atom row")
    by_key = {
        tuple(sorted((atom_rows[id(bond.atom1)], atom_rows[id(bond.atom2)]))): bond
        for bond in bonds
    }
    keys = tuple(tuple(sorted((int(first), int(second)))) for first, second in indices)
    if len(set(keys)) != len(keys) or set(keys) != set(by_key):
        raise RuntimeError("Native coordination mapping differs from complete Bond topology")
    ordered = tuple(by_key[key] for key in keys)
    directed = tuple((atom_rows[id(bond.atom1)], atom_rows[id(bond.atom2)])
                     if bond.atom1.is_metal else
                     (atom_rows[id(bond.atom2)], atom_rows[id(bond.atom1)])
                     for bond in ordered)
    if tuple(map(tuple, indices)) != directed:
        raise ValueError("Native coordination mapping requires directed metal-donor rows")
    orders = np.asarray(session_input.intended_coordination_orders)
    kinds = np.asarray(session_input.intended_coordination_kinds)
    if orders.shape != (len(bonds),) or kinds.shape != (len(bonds),):
        raise ValueError("Native coordination values must align with intended rows")
    if (not np.array_equal(orders, np.asarray([bond.bond_order for bond in ordered]), equal_nan=True)
            or not np.array_equal(kinds, np.asarray([
                _BOND_KIND_CODES[bond.bond_kind.value] for bond in ordered]))):
        raise RuntimeError("Native coordination values differ from complete Bond facts")
    return ordered


def apply_native_selected_structure(
    mol: "Molecule",
    session_input: ComplexSessionInput,
    result: _SelectedStructureResult,
) -> None:
    """Publish selected coordinates while retaining all complete Bond facts."""
    from hotpot.cheminfo.structure._commit import _FactTransaction
    from hotpot.cheminfo.structure._fields import ATOM_SOURCE_CODES

    with mol._mutation_lock:
        atom_count = len(mol._atoms)
        atomic_numbers = np.asarray(session_input.atomic_numbers)
        if (atomic_numbers.shape != (atom_count,)
                or np.asarray(session_input.coordinates).shape != (atom_count, 3)
                or not np.array_equal(atomic_numbers, np.fromiter(
                    (atom.atomic_number for atom in mol._atoms), dtype=np.int32, count=atom_count))):
            raise RuntimeError("Native Atom mapping differs from the current Molecule")
        ordered_bonds = _ordered_coordination_bonds(mol, session_input)
        mask = np.asarray(result.final_active_coordination_mask)
        if mask.shape != (len(ordered_bonds),) or not np.isin(mask, (0, 1)).all():
            raise ValueError("Native coordination mask must contain one binary value per intended row")
        coordinates = np.array(result.selected_coordinates, dtype=np.float64, copy=True)
        if coordinates.shape != (atom_count, 3) or not np.isfinite(coordinates).all():
            raise ValueError("Native selected coordinates must be finite and align with Atom rows")
        with _FactTransaction(mol, producer=ATOM_SOURCE_CODES["forcefield"]):
            mol.coordinates = coordinates


def ingest_native_trajectory(
    trajectory: ForceFieldTrajectory,
    batch: NativeTrajectoryBatch,
    session_input: ComplexSessionInput,
) -> Tuple[int, ...]:
    """Append one native batch to an existing Python trajectory."""
    return trajectory.ingest_native_batch(batch, session_input)
