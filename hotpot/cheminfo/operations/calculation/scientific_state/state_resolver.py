"""Scientific-state orchestration over immutable molecular facts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from hotpot.cheminfo.rules.chemistry.scientific_state.request import StateRequest

from hotpot.cheminfo.rules.chemistry.scientific_state.charge.contracts import (
    LewisFormalChargeResult,
    NetChargeResult,
)
from hotpot.cheminfo.rules.chemistry.scientific_state.contracts import (
    CandidateBudget,
    DEFAULT_CANDIDATE_BUDGET,
    ResolutionStatus,
    ScientificAdmissibility,
    ScientificOperationRequirements,
    StateValueAuthority,
)
from hotpot.cheminfo.structure.snapshots import MolecularFacts
from hotpot.cheminfo.operations.calculation.scientific_state.contracts import (
    ProviderSet,
    StatePolicy,
)
from hotpot.cheminfo.rules.chemistry.scientific_state.electronic_state.contracts import (
    ElectronicStateInferenceResult,
)
from hotpot.cheminfo.rules.chemistry.scientific_state.oxidation_state.contracts import (
    CoordinationElectronCountingResult,
    OxidationStateInferenceResult,
)
from hotpot.cheminfo.rules.chemistry.scientific_state.composition import (
    CandidateProposal,
    ScientificResolutionResult,
    ScientificStateCandidateSpace,
)
from hotpot.cheminfo.rules.chemistry.scientific_state.providers.contracts import (
    RankAdjustment,
    RuleTrace,
)
from hotpot.cheminfo.operations.calculation.scientific_state.contracts import RuleSet
from hotpot.cheminfo.operations.calculation.scientific_state.state_selection import (
    ExecutionStateDecision,
)

__all__ = [
    'resolve_scientific_state',
    'ProvidedScientificResults',
    'ScientificStateBundle',
]


# Orchestrator inputs.


@dataclass(frozen=True)
class ProvidedScientificResults:
    """Precomputed scientific results admitted to one resolution run."""

    net_charge_results: Tuple[NetChargeResult, ...] = ()
    lewis_results: Tuple[LewisFormalChargeResult, ...] = ()
    coordination_counting_results: Tuple[
        CoordinationElectronCountingResult, ...
    ] = ()
    oxidation_state_results: Tuple[OxidationStateInferenceResult, ...] = ()
    electronic_state_results: Tuple[ElectronicStateInferenceResult, ...] = ()


@dataclass(frozen=True)
class ScientificStateBundle:
    """Complete attributable output of one scientific-state resolution run."""

    facts: MolecularFacts
    net_charge_results: Tuple[NetChargeResult, ...]
    lewis_results: Tuple[LewisFormalChargeResult, ...]
    coordination_counting_results: Tuple[
        CoordinationElectronCountingResult,
        ...,
    ]
    oxidation_state_results: Tuple[OxidationStateInferenceResult, ...]
    electronic_state_results: Tuple[ElectronicStateInferenceResult, ...]
    candidate_space: ScientificStateCandidateSpace
    scientific_resolution: ScientificResolutionResult
    execution_decision: ExecutionStateDecision
    rule_trace: Tuple[RuleTrace, ...]


# Public scientific-state orchestration.


_KNOWN_PROVIDER_IDS = frozenset(
    {
        "PROVIDER-001",
        "PROVIDER-002",
        "PROVIDER-003",
    }
)


def _enabled_provider_ids(provider_set: ProviderSet) -> frozenset[str]:
    provider_ids = tuple(toggle.provider_id for toggle in provider_set.toggles)
    if len(set(provider_ids)) != len(provider_ids):
        raise ValueError("Provider toggles contain duplicate provider IDs")
    unknown = set(provider_ids) - _KNOWN_PROVIDER_IDS
    if unknown:
        raise ValueError(f"Unknown provider IDs: {tuple(sorted(unknown))}")
    configured = tuple(
        toggle.provider_id
        for toggle in provider_set.toggles
        if toggle.configuration
    )
    if configured:
        raise ValueError(
            "Provider configuration is not defined for "
            f"{tuple(sorted(configured))}"
        )
    return frozenset(
        toggle.provider_id
        for toggle in provider_set.toggles
        if toggle.enabled
    )


def _validated_lewis_result(
    facts: MolecularFacts,
    result: LewisFormalChargeResult,
    charge_results: Tuple[NetChargeResult, ...],
) -> LewisFormalChargeResult:
    from hotpot.cheminfo.rules.chemistry.scientific_state import legality

    charge_by_identity = {
        candidate.provenance.identity: candidate
        for charge_result in charge_results
        if charge_result.facts_identity == facts.identity
        and charge_result.admissibility is ScientificAdmissibility.ADMISSIBLE
        for candidate in charge_result.candidates
    }
    assessments = [
        legality._validate_facts_identity(
            facts.identity,
            result.facts_identity,
            related_identities=(result.identity,),
        )
    ]
    for candidate in result.candidates:
        charge = charge_by_identity.get(
            candidate.net_charge_candidate_identity
        )
        assessments.append(
            legality._validate_lewis_candidate(facts, charge, candidate)
            if charge is not None
            else legality._hard_failure(
                "HARD-007.LEWIS_CHARGE_REFERENCE_MISSING",
                "Lewis evidence references an unavailable charge candidate.",
                related_identities=(candidate.provenance.identity,),
            )
        )
    assessment = legality._merge_hard_rule_assessments(tuple(assessments))
    if assessment.admissibility is ScientificAdmissibility.ADMISSIBLE:
        return result
    return LewisFormalChargeResult(
        identity=result.identity,
        facts_identity=result.facts_identity,
        candidates=result.candidates,
        admissibility=assessment.admissibility,
        resolution=assessment.terminal_status or ResolutionStatus.UNRESOLVED,
        confidence=result.confidence,
        findings=result.findings + assessment.findings,
    )


def _validated_counting_result(
    facts: MolecularFacts,
    result: CoordinationElectronCountingResult,
) -> CoordinationElectronCountingResult:
    from hotpot.cheminfo.rules.chemistry.scientific_state import legality

    assessments = (
        legality._validate_facts_identity(
            facts.identity,
            result.facts_identity,
            related_identities=(result.identity,),
        ),
        *tuple(
            legality._validate_counting_candidate_for_facts(facts, candidate)
            for candidate in result.candidates
        ),
    )
    assessment = legality._merge_hard_rule_assessments(assessments)
    if assessment.admissibility is ScientificAdmissibility.ADMISSIBLE:
        return result
    return CoordinationElectronCountingResult(
        identity=result.identity,
        facts_identity=result.facts_identity,
        candidates=result.candidates,
        admissibility=assessment.admissibility,
        resolution=assessment.terminal_status or ResolutionStatus.UNRESOLVED,
        confidence=result.confidence,
        findings=result.findings + assessment.findings,
    )


def _explicit_candidate_ids(
    charge_results: Tuple[NetChargeResult, ...],
    oxidation_state_results: Tuple[OxidationStateInferenceResult, ...],
    electronic_state_results: Tuple[ElectronicStateInferenceResult, ...],
) -> frozenset[str]:
    return frozenset(
        candidate.provenance.identity
        for results in (
            charge_results,
            oxidation_state_results,
            electronic_state_results,
        )
        for result in results
        for candidate in result.candidates
        if candidate.provenance.authority
        is StateValueAuthority.VERIFIED_EXPLICIT
    )


def resolve_scientific_state(
    facts: MolecularFacts,
    request: StateRequest = StateRequest(),
    *,
    policy: StatePolicy = StatePolicy.STRICT,
    requirements: Optional[ScientificOperationRequirements] = None,
    provider_set: ProviderSet = ProviderSet(),
    provided_results: ProvidedScientificResults = ProvidedScientificResults(),
    rule_set: RuleSet = RuleSet(),
    candidate_budget: CandidateBudget = DEFAULT_CANDIDATE_BUDGET,
) -> ScientificStateBundle:
    """Resolve, compose and decide one immutable molecular scientific state."""
    from hotpot.cheminfo.rules.chemistry.scientific_state.charge.net_charge import (
        resolve_net_charge,
    )
    from hotpot.cheminfo.rules.chemistry.scientific_state.charge.lewis import (
        solve_main_group_lewis,
    )
    from hotpot.cheminfo.operations.calculation.scientific_state.defaults import (
        _decide_scientific_state,
        _decision_search_compiler,
    )
    from hotpot.cheminfo.rules.chemistry.scientific_state.electronic_state.inference import (
        infer_electronic_states,
    )
    from hotpot.cheminfo.operations.calculation.scientific_state.operation_requirements import (
        _effective_requirements,
    )
    from hotpot.cheminfo.rules.chemistry.scientific_state.oxidation_state.inference import (
        infer_oxidation_states,
    )
    from hotpot.cheminfo.operations.calculation.scientific_state.profile_eligibility import (
        _static_profile_eligibility,
    )
    from hotpot.cheminfo.operations.calculation.scientific_state.composition.compose import (
        _compose_scientific_state_space,
    )
    from hotpot.cheminfo.operations.calculation.scientific_state.composition.constraints import (
        _reconcile_authoritative_results,
    )
    from hotpot.cheminfo.operations.calculation.scientific_state.composition.resolution import (
        _resolve_candidate_space,
    )

    if policy is StatePolicy.STRICT and rule_set.layers:
        raise ValueError("STRICT policy cannot execute optional rules")
    enabled_provider_ids = _enabled_provider_ids(provider_set)
    effective_requirements = _effective_requirements(policy, requirements)

    charge_results = resolve_net_charge(
        facts,
        request,
        provided=provided_results.net_charge_results,
    )
    validated_provided_lewis = tuple(
        _validated_lewis_result(facts, result, charge_results)
        for result in provided_results.lewis_results
    )
    generated_lewis = (
        tuple(
            lewis_result
            for charge_result in charge_results
            for lewis_result in solve_main_group_lewis(
                facts,
                charge_result,
                candidate_budget=candidate_budget,
            )
        )
        if "PROVIDER-001" in enabled_provider_ids
        else ()
    )
    lewis_results = validated_provided_lewis + generated_lewis
    counting_results = tuple(
        _validated_counting_result(facts, result)
        for result in provided_results.coordination_counting_results
    )
    oxidation_state_results = infer_oxidation_states(
        facts,
        request,
        counting_results=counting_results,
        provided=provided_results.oxidation_state_results,
    )
    electronic_state_results = infer_electronic_states(
        facts,
        request,
        net_charge_results=charge_results,
        oxidation_state_results=oxidation_state_results,
        provided=provided_results.electronic_state_results,
    )
    cross_result_constraints, cross_result_conflict = (
        _reconcile_authoritative_results(
            facts,
            charge_results,
            oxidation_state_results,
            electronic_state_results,
            request.oxidation_overrides,
        )
    )
    eligibility = _static_profile_eligibility(
        facts,
        policy,
        effective_requirements,
        provider_set,
        rule_set,
        candidate_budget,
    )

    def compose_with_rule_outputs(
        candidate_proposals: Tuple[CandidateProposal, ...],
        rank_adjustments: Tuple[RankAdjustment, ...],
    ) -> ScientificStateCandidateSpace:
        return _compose_scientific_state_space(
            facts,
            net_charge_results=charge_results,
            lewis_results=lewis_results,
            coordination_counting_results=counting_results,
            oxidation_state_results=oxidation_state_results,
            electronic_state_results=electronic_state_results,
            candidate_proposals=candidate_proposals,
            constraints=cross_result_constraints,
            candidate_budget=candidate_budget,
            decision_search=_decision_search_compiler(
                policy,
                effective_requirements,
                eligibility,
                rank_adjustments,
            ),
        )

    candidate_space = compose_with_rule_outputs((), ())
    rule_trace: Tuple[RuleTrace, ...] = ()
    final_rank_adjustments: Tuple[RankAdjustment, ...] = ()
    if rule_set.layers:
        from hotpot.cheminfo.operations.calculation.scientific_state.rule_controller import (
            _execute_rule_layers,
        )

        rule_execution = _execute_rule_layers(
            rule_set,
            facts,
            effective_requirements,
            candidate_budget,
            candidate_space,
            compose_with_rule_outputs,
        )
        candidate_space = rule_execution.candidate_space
        rule_trace = rule_execution.traces
        final_rank_adjustments = rule_execution.rank_adjustments
    scientific_resolution = _resolve_candidate_space(candidate_space)
    all_results = (
        charge_results
        + lewis_results
        + counting_results
        + oxidation_state_results
        + electronic_state_results
    )
    result_states = tuple(
        (result.admissibility, result.resolution)
        for result in all_results
    )
    if cross_result_conflict:
        result_states += (
            (
                ScientificAdmissibility.INADMISSIBLE,
                ResolutionStatus.CONFLICT,
            ),
        )
    decision = _decide_scientific_state(
        policy,
        effective_requirements,
        candidate_space,
        eligibility,
        result_states,
        _explicit_candidate_ids(
            charge_results,
            oxidation_state_results,
            electronic_state_results,
        ),
        _decision_search_compiler(
            policy,
            effective_requirements,
            eligibility,
            final_rank_adjustments,
        ),
        facts=facts,
    )
    return ScientificStateBundle(
        facts=facts,
        net_charge_results=charge_results,
        lewis_results=lewis_results,
        coordination_counting_results=counting_results,
        oxidation_state_results=oxidation_state_results,
        electronic_state_results=electronic_state_results,
        candidate_space=candidate_space,
        scientific_resolution=scientific_resolution,
        execution_decision=decision,
        rule_trace=rule_trace,
    )
