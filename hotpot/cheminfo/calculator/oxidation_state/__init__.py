"""Coordination-counting and oxidation-state contracts and operations."""

from .contracts import (
    CoordinationCountingCandidate,
    CoordinationElectronCountingResult,
    CoordinationFragmentAssignment,
    CountingConvention,
    ElementOxidationStateDomain,
    OxidationStateCandidateEvidence,
    OxidationStateInferenceResult,
    OxidationStateOverride,
)
from .registry import element_oxidation_state_domain
from .inference import infer_oxidation_states
from .coordination_counting import infer_coordination_electron_counting

__all__ = [
    "infer_coordination_electron_counting",
    "infer_oxidation_states",
    "element_oxidation_state_domain",
    "CountingConvention",
    "OxidationStateOverride",
    "CoordinationFragmentAssignment",
    "CoordinationCountingCandidate",
    "CoordinationElectronCountingResult",
    "OxidationStateCandidateEvidence",
    "OxidationStateInferenceResult",
    "ElementOxidationStateDomain",
]
