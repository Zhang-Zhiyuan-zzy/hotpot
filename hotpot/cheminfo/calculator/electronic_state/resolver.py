"""Composition of independent charge and spin evidence."""

from __future__ import annotations

from typing import Optional

from ...core import Molecule
from ..formal_charges import infer_charge
from .contracts import (
    ChargeEstimator,
    ChargeInferenceError,
    ChargeInferenceResult,
    ChargeInferenceSource,
    ElectronicState,
    SpinEstimator,
    SpinInferenceResult,
    SpinInferenceSource,
)
from .spin import infer_lowest_spin

__all__ = ["resolve_electronic_state"]


def _resolve_charge(
    mol: Molecule,
    charge: Optional[int],
    charge_estimator: Optional[ChargeEstimator],
) -> ChargeInferenceResult:
    if charge is None:
        return (
            charge_estimator.infer(mol)
            if charge_estimator is not None
            else infer_charge(mol)
        )

    if charge_estimator is None:
        return infer_charge(
            mol,
            model="valence-constrained",
            target_charge=charge,
        )

    charge_result = charge_estimator.infer(mol)
    if charge_result.total_charge != charge:
        raise ChargeInferenceError(
            f"Custom charge estimator returned {charge_result.total_charge}, "
            f"which does not match explicit total charge {charge}"
        )
    return charge_result


def _explicit_spin_result(
    mol: Molecule,
    charge: int,
    unpaired_electrons: int,
) -> SpinInferenceResult:
    parity_result = infer_lowest_spin(mol, charge)
    if unpaired_electrons < 0:
        raise ValueError("Unpaired-electron count cannot be negative")
    if unpaired_electrons > parity_result.electron_count:
        raise ValueError("Unpaired-electron count exceeds the electron count")
    if unpaired_electrons % 2 != parity_result.electron_count % 2:
        raise ValueError(
            "Explicit unpaired-electron count has inconsistent electron parity"
        )
    return SpinInferenceResult(
        unpaired_electrons=unpaired_electrons,
        multiplicity=unpaired_electrons + 1,
        electron_count=parity_result.electron_count,
        source=SpinInferenceSource.EXPLICIT,
        assumptions=(
            f"Used explicit unpaired-electron count {unpaired_electrons}.",
        ),
    )


def resolve_electronic_state(
    mol: Molecule,
    *,
    charge: Optional[int] = None,
    unpaired_electrons: Optional[int] = None,
    charge_estimator: Optional[ChargeEstimator] = None,
    spin_estimator: Optional[SpinEstimator] = None,
) -> ElectronicState:
    """Resolve backend-ready charge and spin facts with explicit provenance."""
    charge_result = _resolve_charge(mol, charge, charge_estimator)
    resolved_charge = charge_result.total_charge if charge is None else charge
    charge_source = (
        charge_result.source
        if charge is None
        else ChargeInferenceSource.EXPLICIT
    )
    charge_assumptions = charge_result.assumptions
    if charge is not None:
        charge_assumptions += (f"Used explicit total charge {charge}.",)

    if unpaired_electrons is not None:
        spin_result = _explicit_spin_result(
            mol,
            resolved_charge,
            unpaired_electrons,
        )
    elif spin_estimator is not None:
        spin_result = spin_estimator.infer(mol, resolved_charge)
    else:
        spin_result = infer_lowest_spin(mol, resolved_charge)

    return ElectronicState(
        charge=resolved_charge,
        unpaired_electrons=spin_result.unpaired_electrons,
        multiplicity=spin_result.multiplicity,
        fragment_charges=tuple(
            fragment.charge for fragment in charge_result.fragments
        ),
        charge_source=charge_source,
        spin_source=spin_result.source,
        assumptions=charge_assumptions + spin_result.assumptions,
    )
