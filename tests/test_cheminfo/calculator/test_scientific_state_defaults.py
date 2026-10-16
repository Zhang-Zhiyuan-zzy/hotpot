"""Named scientific-state decision policy tests."""

import hotpot

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
from hotpot.cheminfo.calculator.defaults import _decide_scientific_state
from hotpot.cheminfo.calculator.electronic_state import (
    infer_electronic_states,
)
from hotpot.cheminfo.calculator.profile_eligibility import (
    _static_profile_eligibility,
)
from hotpot.cheminfo.calculator.state_composition import (
    compose_scientific_state_space,
)
from hotpot.cheminfo.calculator.state_resolver import (
    StateRequest,
    extract_molecular_facts,
)
from hotpot.cheminfo.calculator.state_rules import RuleSet
from hotpot.cheminfo.calculator.state_selection import (
    DecisionCertificationStatus,
    StateDecisionAction,
    StateDecisionSource,
)


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
