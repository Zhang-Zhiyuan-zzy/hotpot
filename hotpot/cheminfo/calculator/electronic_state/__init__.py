"""Typed electronic-state facts shared by calculator services."""

from .contracts import (
    AmbiguousHydrogenRepresentationError,
    ChargeEstimator,
    ChargeInferenceError,
    ChargeInferenceResult,
    ChargeInferenceSource,
    ElectronicState,
    ElectronicStateCandidate,
    ElectronicStateCandidateEvidence,
    ElectronicStateError,
    ElectronicStateInferenceResult,
    FragmentCharge,
    IncompleteExplicitAtomError,
    NeutralAtomReference,
    SpinContributionCandidate,
    SpinContributionScope,
    SpinEstimator,
    SpinInferenceResult,
    SpinInferenceSource,
)
from .spin import (
    LowestSpinEstimator,
    infer_lowest_spin,
    require_complete_explicit_atoms,
)
from .neutral_atom import neutral_atom_reference

__all__ = [
    "AmbiguousHydrogenRepresentationError",
    "ChargeEstimator",
    "ChargeInferenceError",
    "ChargeInferenceResult",
    "ChargeInferenceSource",
    "ElectronicState",
    "ElectronicStateCandidateEvidence",
    "SpinContributionScope",
    "SpinContributionCandidate",
    "ElectronicStateCandidate",
    "ElectronicStateInferenceResult",
    "NeutralAtomReference",
    "ElectronicStateError",
    "FragmentCharge",
    "IncompleteExplicitAtomError",
    "LowestSpinEstimator",
    "SpinEstimator",
    "SpinInferenceResult",
    "SpinInferenceSource",
    "infer_lowest_spin",
    "require_complete_explicit_atoms",
    "neutral_atom_reference",
]

