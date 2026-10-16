"""Backend-context state projection contracts and operations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

from .contracts import (
    CandidateBudget,
    BackendContext,
    BackendIdentity,
    DEFAULT_CANDIDATE_BUDGET,
    MolecularFacts,
    MolecularFactsIdentity,
    ProviderSet,
    ScientificOperationRequirements,
    StatePolicy,
)
from .electronic_state import (
    ElectronicStateCandidate,
    SpinContributionCandidate,
    SpinContributionScope,
)
from .state_composition import (
    CoupledScientificStateCandidate,
    ScientificStateIdentity,
    StateCandidate,
)
from .state_resolver import (
    ProvidedScientificResults,
    ScientificStateBundle,
    StateRequest,
)
from .state_rules import RuleSet
from .state_selection import DecisionReason

__all__ = [
    "project_state",
    "resolve_and_project_state",
    "BackendStateValues",
    "BackendStateRequest",
    "BackendDecisionSource",
    "BackendExecutability",
    "BackendProjectionFinding",
    "BackendExecutionDecision",
    "BackendExecutionEvidence",
    "ExecutionPolicySet",
    "BackendStateProjection",
    "BackendProjectionStatus",
    "BackendProjectionResult",
    "ScientificStateWorkflowResult",
]


# Backend execution-selection contracts.


@dataclass(frozen=True)
class BackendStateValues:
    """Backend-facing charge, UHF and multiplicity values."""

    charge: int
    uhf: Optional[int]
    multiplicity: Optional[int]


@dataclass(frozen=True)
class BackendStateRequest:
    """Explicit backend-only state settings supplied by a caller."""

    uhf: Optional[int] = None


class BackendDecisionSource(Enum):
    """Authority responsible for one backend execution-state choice."""

    EXPLICIT = "explicit"
    SCIENTIFIC_STATE = "scientific_state"
    EXACT_CONTEXT_HISTORY = "exact_context_history"
    EXECUTION_DEFAULT = "execution_default"


class BackendExecutability(Enum):
    """Static executability of one state in an exact backend context."""

    NOT_EVALUATED = "not_evaluated"
    SUPPORTED = "supported"
    UNSUPPORTED_METHOD = "unsupported_method"
    UNSUPPORTED_ELEMENT = "unsupported_element"
    INVALID_EFFECTIVE_ELECTRON_COUNT = "invalid_effective_electron_count"


@dataclass(frozen=True)
class BackendProjectionFinding:
    """Typed diagnostic emitted while projecting a scientific state."""

    code: str
    message: str
    related_identities: Tuple[str, ...] = ()


@dataclass(frozen=True)
class BackendExecutionDecision:
    """Auditable backend-state decision independent of scientific evidence."""

    backend_identity: BackendIdentity
    requested_state: BackendStateValues
    selected_state: BackendStateValues
    source: BackendDecisionSource
    rationale: Tuple[DecisionReason, ...]


@dataclass(frozen=True)
class BackendExecutionEvidence:
    """Verified execution evidence tied to an exact state and backend."""

    scientific_state_identity: ScientificStateIdentity
    backend_identity: BackendIdentity
    requested_state: BackendStateValues
    executed_state: BackendStateValues
    artifact_digests: Tuple[str, ...]


@dataclass(frozen=True)
class ExecutionPolicySet:
    """Named backend execution-selection policies enabled for projection."""

    policy_ids: Tuple[str, ...] = ("EXECUTION-001", "DEFAULT-004")


@dataclass(frozen=True)
class BackendStateProjection:
    """Validated mapping from one scientific state into a backend context."""

    scientific_state_identity: ScientificStateIdentity
    facts_identity: MolecularFactsIdentity
    backend_identity: BackendIdentity
    effective_electron_count: int
    charge: int
    uhf: Optional[int]
    multiplicity: Optional[int]
    projection_convention: str
    execution_decision: BackendExecutionDecision
    executability: BackendExecutability


class BackendProjectionStatus(Enum):
    """Terminal status of one backend projection attempt."""

    NOT_PROJECTED = "not_projected"
    PROJECTED = "projected"
    PROJECTION_CONFLICT = "projection_conflict"
    BACKEND_UNSUPPORTED = "backend_unsupported"


@dataclass(frozen=True)
class BackendProjectionResult:
    """Tagged backend projection result with no fabricated failure values."""

    status: BackendProjectionStatus
    backend_identity: BackendIdentity
    request: BackendStateRequest
    projection: Optional[BackendStateProjection]
    findings: Tuple[BackendProjectionFinding, ...]

    def __post_init__(self) -> None:
        projected = self.status is BackendProjectionStatus.PROJECTED
        if projected != (self.projection is not None):
            raise ValueError("Only PROJECTED results may carry a projection")
        if (
            self.projection is not None
            and self.projection.backend_identity != self.backend_identity
        ):
            raise ValueError(
                "Projection and result must identify the same backend context"
            )


@dataclass(frozen=True)
class ScientificStateWorkflowResult:
    """Scientific-state resolution paired with its backend projection."""

    scientific_state: ScientificStateBundle
    backend_projection: BackendProjectionResult


# Backend electron-count and selected-state helpers.


_HIGHEST_MS_PROJECTION = "pure_spin_highest_ms"
_CHARGE_ONLY_PROJECTION = "charge_only"
_KNOWN_EXECUTION_POLICY_IDS = frozenset(
    {"EXECUTION-001", "DEFAULT-004"}
)


def _effective_electron_count(
    facts: MolecularFacts,
    backend: BackendContext,
    charge: int,
) -> int:
    explicit_electrons = sum(
        backend.effective_nuclear_charge(atomic_number)
        for atomic_number in facts.atomic_numbers
    )
    implicit_hydrogens = sum(
        implicit_count
        for atomic_number, explicit_count, implicit_count in zip(
            facts.atomic_numbers,
            facts.hydrogen_inventory.explicit_counts,
            facts.hydrogen_inventory.implicit_counts,
        )
        if atomic_number != 1
        and explicit_count == 0
        and implicit_count is not None
    )
    return (
        explicit_electrons
        + implicit_hydrogens * backend.effective_nuclear_charge(1)
        - charge
    )


def _uhf_legality_finding(
    effective_electron_count: int,
    uhf: int,
) -> Optional[BackendProjectionFinding]:
    if effective_electron_count < 0:
        return BackendProjectionFinding(
            "HARD-013.NEGATIVE_EFFECTIVE_ELECTRON_COUNT",
            "The backend effective-electron count is negative.",
        )
    if uhf < 0 or uhf > effective_electron_count:
        return BackendProjectionFinding(
            "HARD-013.UHF_OUT_OF_RANGE",
            "UHF must lie between zero and the effective-electron count.",
        )
    if (effective_electron_count - uhf) % 2:
        return BackendProjectionFinding(
            "HARD-013.UHF_PARITY",
            "UHF and the effective-electron count have incompatible parity.",
        )
    return None


def _lowest_legal_uhf(effective_electron_count: int) -> int:
    return effective_electron_count % 2


def _validated_execution_policy_ids(
    policies: ExecutionPolicySet,
) -> frozenset[str]:
    if len(set(policies.policy_ids)) != len(policies.policy_ids):
        raise ValueError("Execution policy IDs must be unique")
    unknown = set(policies.policy_ids) - _KNOWN_EXECUTION_POLICY_IDS
    if unknown:
        raise ValueError(f"Unknown execution policy IDs: {tuple(sorted(unknown))}")
    return frozenset(policies.policy_ids)


def _selected_path(
    state: ScientificStateBundle,
) -> Optional[CoupledScientificStateCandidate]:
    selected_identity = state.execution_decision.selected_state_identity
    if selected_identity is None:
        return None
    return next(
        (
            path
            for path in state.candidate_space.expanded_paths
            if path.identity == selected_identity
        ),
        None,
    )


def _candidate_index(
    state: ScientificStateBundle,
) -> Dict[str, StateCandidate]:
    return {
        value.candidate.provenance.identity: value.candidate
        for factor in state.candidate_space.factors
        for value in factor.values
    }


def _selected_pure_spin(
    state: ScientificStateBundle,
    path: CoupledScientificStateCandidate,
) -> Optional[Tuple[int, int]]:
    candidates = _candidate_index(state)
    electronic_identity = path.electronic_state_candidate_identity
    electronic = candidates.get(electronic_identity or "")
    if isinstance(electronic, ElectronicStateCandidate):
        if not electronic.pure_spin or electronic.multiplicity is None:
            return None
        return electronic.two_s, electronic.multiplicity

    contributions = tuple(
        candidate
        for evidence_id in path.evidence_ids
        for candidate in (candidates.get(evidence_id),)
        if isinstance(candidate, SpinContributionCandidate)
        and candidate.scope is SpinContributionScope.MOLECULAR
    )
    if len(contributions) != 1 or not contributions[0].pure_spin:
        return None
    return contributions[0].two_s, contributions[0].two_s + 1


def _projection_result(
    status: BackendProjectionStatus,
    backend: BackendContext,
    request: BackendStateRequest,
    code: str,
    message: str,
    related_identities: Tuple[str, ...] = (),
) -> BackendProjectionResult:
    return BackendProjectionResult(
        status=status,
        backend_identity=backend.identity,
        request=request,
        projection=None,
        findings=(
            BackendProjectionFinding(
                code,
                message,
                related_identities,
            ),
        ),
    )


def _matching_history(
    state_identity: ScientificStateIdentity,
    backend_identity: BackendIdentity,
    charge: int,
    electron_count: int,
    history: Tuple[BackendExecutionEvidence, ...],
) -> Tuple[BackendExecutionEvidence, ...]:
    return tuple(
        evidence
        for evidence in history
        if evidence.scientific_state_identity == state_identity
        and evidence.backend_identity == backend_identity
        and evidence.executed_state.charge == charge
        and evidence.executed_state.uhf is not None
        and _uhf_legality_finding(
            electron_count,
            evidence.executed_state.uhf,
        )
        is None
    )


def _selected_backend_state(
    state: ScientificStateBundle,
    path: CoupledScientificStateCandidate,
    backend: BackendContext,
    request: BackendStateRequest,
    policies: frozenset[str],
    electron_count: int,
    execution_history: Tuple[BackendExecutionEvidence, ...],
) -> Tuple[
    Optional[BackendStateValues],
    Optional[BackendDecisionSource],
    Tuple[DecisionReason, ...],
    Tuple[BackendProjectionFinding, ...],
]:
    charge = path.net_charge.value
    scientific_spin = _selected_pure_spin(state, path)
    scientific_uhf = scientific_spin[0] if scientific_spin is not None else None
    scientific_multiplicity = (
        scientific_spin[1] if scientific_spin is not None else None
    )

    if not backend.consumes_uhf:
        if backend.projection_convention != _CHARGE_ONLY_PROJECTION:
            raise ValueError(
                "A charge-only backend must use the charge_only convention"
            )
        if request.uhf is not None:
            return (
                None,
                None,
                (),
                (
                    BackendProjectionFinding(
                        "HARD-008.UHF_NOT_CONSUMED",
                        "The selected backend context does not consume UHF.",
                    ),
                ),
            )
        return (
            BackendStateValues(charge, None, None),
            BackendDecisionSource.SCIENTIFIC_STATE,
            (
                DecisionReason(
                    "HARD-008.CHARGE_ONLY_PROJECTION",
                    "The backend context consumes only the selected charge.",
                ),
            ),
            (),
        )

    if backend.projection_convention != _HIGHEST_MS_PROJECTION:
        raise ValueError(
            "Unknown backend projection convention: "
            f"{backend.projection_convention}"
        )

    if request.uhf is not None:
        legality_finding = _uhf_legality_finding(electron_count, request.uhf)
        if legality_finding is not None:
            return None, None, (), (legality_finding,)
        if scientific_uhf is not None and request.uhf != scientific_uhf:
            return (
                None,
                None,
                (),
                (
                    BackendProjectionFinding(
                        "HARD-013.EXPLICIT_UHF_SCIENTIFIC_SPIN_CONFLICT",
                        "Explicit UHF conflicts with the selected pure-spin "
                        "state under the highest-MS projection convention.",
                        (path.identity.digest,),
                    ),
                ),
            )
        return (
            BackendStateValues(
                charge,
                request.uhf,
                scientific_multiplicity,
            ),
            BackendDecisionSource.EXPLICIT,
            (
                DecisionReason(
                    "HARD-013.EXPLICIT_UHF",
                    "A hard-valid explicit backend UHF was selected.",
                ),
            ),
            (),
        )

    matching_history = (
        _matching_history(
            path.identity,
            backend.identity,
            charge,
            electron_count,
            execution_history,
        )
        if "EXECUTION-001" in policies
        else ()
    )
    compatible_history = tuple(
        evidence
        for evidence in matching_history
        if scientific_uhf is None
        or evidence.executed_state.uhf == scientific_uhf
    )
    historical_uhf = frozenset(
        evidence.executed_state.uhf for evidence in compatible_history
    )
    if len(historical_uhf) > 1:
        return (
            None,
            None,
            (),
            (
                BackendProjectionFinding(
                    "EXECUTION-001.CONFLICTING_HISTORY",
                    "Exact-context execution history contains conflicting "
                    "hard-valid UHF values.",
                    tuple(
                        digest
                        for evidence in compatible_history
                        for digest in evidence.artifact_digests
                    ),
                ),
            ),
        )
    if historical_uhf:
        uhf = next(iter(historical_uhf))
        if uhf is None:
            raise RuntimeError("Matched execution history is missing UHF")
        return (
            BackendStateValues(
                charge,
                uhf,
                scientific_multiplicity,
            ),
            BackendDecisionSource.EXACT_CONTEXT_HISTORY,
            (
                DecisionReason(
                    "EXECUTION-001.EXACT_CONTEXT_HISTORY",
                    "Verified exact-context history selected the backend UHF.",
                ),
            ),
            (),
        )

    if scientific_uhf is not None:
        legality_finding = _uhf_legality_finding(electron_count, scientific_uhf)
        if legality_finding is not None:
            return None, None, (), (legality_finding,)
        return (
            BackendStateValues(
                charge,
                scientific_uhf,
                scientific_multiplicity,
            ),
            BackendDecisionSource.SCIENTIFIC_STATE,
            (
                DecisionReason(
                    "HARD-008.HIGHEST_MS_PROJECTION",
                    "The selected pure-spin state was projected as UHF=2S.",
                ),
            ),
            (),
        )

    if "DEFAULT-004" not in policies:
        return (
            None,
            None,
            (),
            (
                BackendProjectionFinding(
                    "DEFAULT-004.DISABLED",
                    "No backend UHF is selected and DEFAULT-004 is disabled.",
                ),
            ),
        )
    uhf = _lowest_legal_uhf(electron_count)
    return (
        BackendStateValues(charge, uhf, None),
        BackendDecisionSource.EXECUTION_DEFAULT,
        (
            DecisionReason(
                "DEFAULT-004.LOWEST_LEGAL_UHF",
                "The lowest parity-compatible UHF was selected for execution.",
            ),
        ),
        (),
    )


# Public backend projection operation.


def project_state(
    state: ScientificStateBundle,
    backend: BackendContext,
    *,
    request: BackendStateRequest = BackendStateRequest(),
    execution_policies: ExecutionPolicySet = ExecutionPolicySet(),
    execution_history: Tuple[BackendExecutionEvidence, ...] = (),
) -> BackendProjectionResult:
    """Project one selected scientific state into an exact backend context."""
    from .state_selection import StateDecisionAction

    policies = _validated_execution_policy_ids(execution_policies)
    if state.execution_decision.action is not StateDecisionAction.SELECT_STATE:
        return _projection_result(
            BackendProjectionStatus.NOT_PROJECTED,
            backend,
            request,
            "HARD-008.SCIENTIFIC_STATE_NOT_SELECTED",
            "Backend projection requires a SELECT_STATE scientific decision.",
            (state.execution_decision.action.value,),
        )

    path = _selected_path(state)
    if path is None:
        raise ValueError("Selected scientific-state identity is not materialized")
    if not backend.method_supported:
        return _projection_result(
            BackendProjectionStatus.BACKEND_UNSUPPORTED,
            backend,
            request,
            "HARD-008.UNSUPPORTED_METHOD",
            "The exact backend context does not support this method.",
        )
    unsupported_elements = tuple(
        sorted(
            {
                atomic_number
                for atomic_number in state.facts.atomic_numbers
                if not backend.supports_atomic_number(atomic_number)
            }
        )
    )
    if unsupported_elements:
        return _projection_result(
            BackendProjectionStatus.BACKEND_UNSUPPORTED,
            backend,
            request,
            "HARD-008.UNSUPPORTED_ELEMENT",
            "The exact backend context does not support every input element.",
            tuple(str(atomic_number) for atomic_number in unsupported_elements),
        )

    charge = path.net_charge.value
    electron_count = _effective_electron_count(state.facts, backend, charge)
    if electron_count < 0:
        return _projection_result(
            BackendProjectionStatus.PROJECTION_CONFLICT,
            backend,
            request,
            "HARD-013.NEGATIVE_EFFECTIVE_ELECTRON_COUNT",
            "The backend effective-electron count is negative.",
            (path.identity.digest,),
        )

    selected, source, rationale, findings = _selected_backend_state(
        state,
        path,
        backend,
        request,
        policies,
        electron_count,
        execution_history,
    )
    if selected is None or source is None:
        return BackendProjectionResult(
            status=BackendProjectionStatus.PROJECTION_CONFLICT,
            backend_identity=backend.identity,
            request=request,
            projection=None,
            findings=findings,
        )

    requested = BackendStateValues(
        charge=charge,
        uhf=request.uhf,
        multiplicity=None,
    )
    decision = BackendExecutionDecision(
        backend_identity=backend.identity,
        requested_state=requested,
        selected_state=selected,
        source=source,
        rationale=rationale,
    )
    projection = BackendStateProjection(
        scientific_state_identity=path.identity,
        facts_identity=state.facts.identity,
        backend_identity=backend.identity,
        effective_electron_count=electron_count,
        charge=selected.charge,
        uhf=selected.uhf,
        multiplicity=selected.multiplicity,
        projection_convention=backend.projection_convention,
        execution_decision=decision,
        executability=BackendExecutability.SUPPORTED,
    )
    return BackendProjectionResult(
        status=BackendProjectionStatus.PROJECTED,
        backend_identity=backend.identity,
        request=request,
        projection=projection,
        findings=(),
    )


def resolve_and_project_state(
    facts: MolecularFacts,
    backend: BackendContext,
    request: StateRequest = StateRequest(),
    *,
    policy: StatePolicy = StatePolicy.STRICT,
    requirements: Optional[ScientificOperationRequirements] = None,
    provider_set: ProviderSet = ProviderSet(),
    provided_results: ProvidedScientificResults = ProvidedScientificResults(),
    rule_set: RuleSet = RuleSet(),
    candidate_budget: CandidateBudget = DEFAULT_CANDIDATE_BUDGET,
    backend_request: BackendStateRequest = BackendStateRequest(),
    execution_policies: ExecutionPolicySet = ExecutionPolicySet(),
    execution_history: Tuple[BackendExecutionEvidence, ...] = (),
) -> ScientificStateWorkflowResult:
    """Resolve scientific state once, then project it for one backend."""
    from .operation_requirements import _workflow_requirements
    from .state_resolver import resolve_scientific_state

    workflow_requirements = _workflow_requirements(
        requirements,
        backend.scientific_operation_requirements,
    )
    scientific_state = resolve_scientific_state(
        facts,
        request,
        policy=policy,
        requirements=workflow_requirements,
        provider_set=provider_set,
        provided_results=provided_results,
        rule_set=rule_set,
        candidate_budget=candidate_budget,
    )
    projection = project_state(
        scientific_state,
        backend,
        request=backend_request,
        execution_policies=execution_policies,
        execution_history=execution_history,
    )
    return ScientificStateWorkflowResult(scientific_state, projection)
