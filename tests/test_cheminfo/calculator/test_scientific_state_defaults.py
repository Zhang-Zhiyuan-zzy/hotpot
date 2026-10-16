"""Named scientific-state decision policy tests."""

from dataclasses import replace

import hotpot
import pytest

from hotpot.cheminfo.calculator.charge import resolve_net_charge
from hotpot.cheminfo.calculator.contracts import (
    DEFAULT_CANDIDATE_BUDGET,
    ProviderSet,
    ResolutionStatus,
    ScientificAdmissibility,
    ScientificOperationRequirements,
    ScientificQuantity,
    StatePolicy,
)
from hotpot.cheminfo.calculator.defaults import (
    _certified_candidate_path,
    _decide_scientific_state,
    _decision_search_compiler,
)
from hotpot.cheminfo.calculator.electronic_state import (
    infer_electronic_states,
)
from hotpot.cheminfo.calculator.profile_eligibility import (
    _static_profile_eligibility,
)
from hotpot.cheminfo.calculator.state_composition import (
    CandidateSearchStatus,
    ScientificStateIdentity,
    compose_scientific_state_space,
)
from hotpot.cheminfo.calculator.state_composition.decision_search import (
    _AuthorityVector,
    _DecisionEquivalenceKey,
    _DecisionRecord,
    _DecisionRelation,
    _PreferenceKey,
)
from hotpot.cheminfo.calculator.state_resolver import (
    StateRequest,
    extract_molecular_facts,
)
from hotpot.cheminfo.calculator.state_rules import RankAdjustment, RuleSet
from hotpot.cheminfo.calculator.state_selection import (
    DecisionCertificationStatus,
    StateDecisionAction,
    StateDecisionSource,
    StaticProfileEligibility,
)


class _ChargePreferenceCompiler:
    identity = "test-charge-preference-v1"
    frontier_schema_version = "scientific-state-frontier-v1"

    def record_for_path(self, path, factors):
        del factors
        rank = float(-abs(path.net_charge.value))
        return _DecisionRecord(
            path.identity,
            _AuthorityVector((("charge", 1),)),
            _PreferenceKey((rank,), ()),
            _DecisionEquivalenceKey(
                path.net_charge.value,
                "none",
                None,
            ),
        )

    def compare_records(self, left, right):
        if left.preference.rank_values > right.preference.rank_values:
            return _DecisionRelation.DOMINATES
        if left.preference.rank_values < right.preference.rank_values:
            return _DecisionRelation.DOMINATED
        return _DecisionRelation.EQUIVALENT


def _resolved_space():
    facts = extract_molecular_facts(hotpot.read_mol("CCO", "smi"))
    request = StateRequest(net_charge=0, two_s=0)
    charge_results = resolve_net_charge(facts, request)
    electronic_results = infer_electronic_states(
        facts,
        request,
        net_charge_results=charge_results,
    )
    space = compose_scientific_state_space(
        facts,
        net_charge_results=charge_results,
        electronic_state_results=electronic_results,
    )
    result_states = tuple(
        (result.admissibility, result.resolution)
        for result in charge_results + electronic_results
    )
    explicit_ids = frozenset(
        candidate.provenance.identity
        for result in charge_results + electronic_results
        for candidate in result.candidates
    )
    return facts, space, result_states, explicit_ids


def _requirements(*quantities):
    return ScientificOperationRequirements(
        "test-operation",
        frozenset(quantities),
        frozenset(),
    )


def _eligibility(requirements):
    return _static_profile_eligibility(
        StatePolicy.STRICT,
        requirements,
        ProviderSet(),
        RuleSet(),
        DEFAULT_CANDIDATE_BUDGET,
    )


def test_strict_analysis_returns_not_requested_after_hard_validation() -> None:
    _, space, result_states, explicit_ids = _resolved_space()
    requirements = _requirements()

    decision = _decide_scientific_state(
        StatePolicy.STRICT,
        requirements,
        space,
        _eligibility(requirements),
        result_states,
        explicit_ids,
    )

    assert decision.action is StateDecisionAction.NOT_REQUESTED
    assert decision.selected_state_identity is None


def test_inactive_profiles_do_not_invent_a_decision_search_compiler() -> None:
    requirements = _requirements(ScientificQuantity.NET_CHARGE)
    screening_eligibility = _static_profile_eligibility(
        StatePolicy.COORDINATION_SCREENING_V1,
        requirements,
        ProviderSet(),
        RuleSet(),
        DEFAULT_CANDIDATE_BUDGET,
    )

    assert _decision_search_compiler(
        StatePolicy.STRICT,
        requirements,
        _eligibility(requirements),
        (),
    ) is None
    assert _decision_search_compiler(
        StatePolicy.COORDINATION_SCREENING_V1,
        requirements,
        screening_eligibility,
        (RankAdjustment("state", "test", 1.0, "test"),),
    ) is None


def test_strict_policy_never_consumes_rule_rank_adjustments() -> None:
    requirements = _requirements(ScientificQuantity.NET_CHARGE)

    with pytest.raises(ValueError, match="STRICT"):
        _decision_search_compiler(
            StatePolicy.STRICT,
            requirements,
            _eligibility(requirements),
            (RankAdjustment("state", "test", 1.0, "test"),),
        )


def test_strict_execution_selects_one_attributable_explicit_path() -> None:
    _, space, result_states, explicit_ids = _resolved_space()
    requirements = _requirements(
        ScientificQuantity.NET_CHARGE,
        ScientificQuantity.ELECTRONIC_STATE,
    )

    decision = _decide_scientific_state(
        StatePolicy.STRICT,
        requirements,
        space,
        _eligibility(requirements),
        result_states,
        explicit_ids,
    )

    assert decision.action is StateDecisionAction.SELECT_STATE
    assert decision.source is StateDecisionSource.EXPLICIT
    assert (
        decision.decision_certifiability.status
        is DecisionCertificationStatus.CERTIFIED
    )


def test_authoritative_conflict_precedes_not_requested_decision() -> None:
    _, space, _, explicit_ids = _resolved_space()
    requirements = _requirements()

    decision = _decide_scientific_state(
        StatePolicy.STRICT,
        requirements,
        space,
        _eligibility(requirements),
        ((ScientificAdmissibility.INADMISSIBLE, ResolutionStatus.CONFLICT),),
        explicit_ids,
    )

    assert decision.action is StateDecisionAction.CONFLICT


def test_missing_required_oxidation_state_declines_automatic_selection() -> None:
    _, space, result_states, explicit_ids = _resolved_space()
    requirements = _requirements(ScientificQuantity.OXIDATION_STATE)

    decision = _decide_scientific_state(
        StatePolicy.STRICT,
        requirements,
        space,
        _eligibility(requirements),
        result_states,
        explicit_ids,
    )

    assert decision.action is StateDecisionAction.NO_AUTOMATIC_STATE
    assert decision.selected_state_identity is None


def test_default_selects_the_compiler_winner_not_path_order() -> None:
    _, space, result_states, explicit_ids = _resolved_space()
    preferred = replace(
        space.expanded_paths[0],
        identity=ScientificStateIdentity("preferred-neutral"),
    )
    other = replace(
        preferred,
        identity=ScientificStateIdentity("first-but-charged"),
        net_charge=replace(preferred.net_charge, value=1),
    )
    candidate_space = replace(
        space,
        expanded_paths=(other, preferred),
        search_status=CandidateSearchStatus.DECISION_CERTIFIED,
    )
    compiler = _ChargePreferenceCompiler()
    requirements = _requirements(ScientificQuantity.NET_CHARGE)
    eligibility = StaticProfileEligibility(
        "coordination-screening-v1",
        "test-profile",
        True,
        (),
    )

    selected = _certified_candidate_path(
        candidate_space,
        candidate_space.expanded_paths,
        compiler,
    )
    decision = _decide_scientific_state(
        StatePolicy.COORDINATION_SCREENING_V1,
        requirements,
        candidate_space,
        eligibility,
        result_states,
        explicit_ids,
        compiler,
    )

    assert selected is preferred
    assert decision.selected_state_identity == preferred.identity
    assert decision.source is StateDecisionSource.EXECUTION_DEFAULT


def test_truncated_certified_status_remains_selectable() -> None:
    _, space, _, _ = _resolved_space()
    candidate_space = replace(
        space,
        search_status=(
            CandidateSearchStatus.TRUNCATED_BUT_DECISION_CERTIFIED
        ),
    )

    selected = _certified_candidate_path(
        candidate_space,
        candidate_space.expanded_paths,
        _ChargePreferenceCompiler(),
    )

    assert selected == candidate_space.expanded_paths[0]
