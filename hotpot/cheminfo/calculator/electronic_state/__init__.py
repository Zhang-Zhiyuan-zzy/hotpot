"""Typed electronic-state facts shared by calculator services."""

from .contracts import (
    AmbiguousHydrogenRepresentationError,
    ChargeEstimator,
    ChargeInferenceError,
    ChargeInferenceResult,
    ChargeInferenceSource,
    ElectronicState,
    ElectronicStateError,
    FragmentCharge,
    IncompleteExplicitAtomError,
    SpinEstimator,
    SpinInferenceResult,
    SpinInferenceSource,
)
from .spin import LowestSpinEstimator, infer_lowest_spin

__all__ = [
    "AmbiguousHydrogenRepresentationError",
    "ChargeEstimator",
    "ChargeInferenceError",
    "ChargeInferenceResult",
    "ChargeInferenceSource",
    "ElectronicState",
    "ElectronicStateError",
    "FragmentCharge",
    "IncompleteExplicitAtomError",
    "LowestSpinEstimator",
    "SpinEstimator",
    "SpinInferenceResult",
    "SpinInferenceSource",
    "infer_lowest_spin",
]

