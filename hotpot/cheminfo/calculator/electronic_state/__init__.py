"""Typed electronic-state facts shared by calculator services."""

from .contracts import (
    ElectronicStateCandidate,
    ElectronicStateCandidateEvidence,
    ElectronicStateInferenceResult,
    NeutralAtomReference,
    SpinContributionCandidate,
    SpinContributionScope,
)
from .neutral_atom import neutral_atom_reference
from .inference import infer_electronic_states

__all__ = [
    "infer_electronic_states",
    "neutral_atom_reference",
    "ElectronicStateCandidateEvidence",
    "SpinContributionScope",
    "SpinContributionCandidate",
    "ElectronicStateCandidate",
    "ElectronicStateInferenceResult",
    "NeutralAtomReference",
]

