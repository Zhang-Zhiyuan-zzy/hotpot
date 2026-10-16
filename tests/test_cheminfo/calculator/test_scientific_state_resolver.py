"""End-to-end strict scientific-state resolver tests."""

from dataclasses import asdict, replace

import hotpot
import pytest

from hotpot.cheminfo.calculator import resolve_scientific_state
from hotpot.cheminfo.calculator.charge import (
    NetChargeCandidateEvidence,
    resolve_net_charge,
    solve_main_group_lewis,
)
from hotpot.cheminfo.calculator.contracts import (
    CandidateProvenance,
    CandidateBudget,
    CoverageStatus,
    ProviderSet,
    ProviderToggle,
    ResolutionStatus,
    ScientificOperationRequirements,
    ScientificQuantity,
    StatePolicy,
    StateValueAuthority,
)
from hotpot.cheminfo.calculator.electronic_state import (
    infer_electronic_states,
)
from hotpot.cheminfo.calculator.oxidation_state import (
    CountingConvention,
    OxidationStateCandidateEvidence,
    OxidationStateOverride,
    infer_oxidation_states,
)
from hotpot.cheminfo.calculator.state_resolver import (
    ProvidedScientificResults,
    StateRequest,
    extract_molecular_facts,
)
from hotpot.cheminfo.calculator.state_selection import StateDecisionAction


def _facts(smiles: str = "CCO"):
    return extract_molecular_facts(hotpot.read_mol(smiles, "smi"))


def _requirements(*quantities):
    return ScientificOperationRequirements(
        "test-operation",
        frozenset(quantities),
        frozenset(),
    )


def test_strict_default_returns_a_complete_analysis_bundle() -> None:
    bundle = resolve_scientific_state(
        _facts(),
        StateRequest(net_charge=0, two_s=0),
    )

    assert bundle.execution_decision.action is StateDecisionAction.NOT_REQUESTED
    assert bundle.net_charge_results
    assert bundle.electronic_state_results
    assert len(bundle.candidate_space.expanded_paths) == 1
    assert bundle.rule_trace == ()
    assert asdict(bundle)["facts"]["nuclear_charge_sum"] == 26


def test_strict_execution_selects_explicit_charge_and_spin() -> None:
    bundle = resolve_scientific_state(
        _facts(),
        StateRequest(net_charge=0, two_s=0),
        requirements=_requirements(
            ScientificQuantity.NET_CHARGE,
            ScientificQuantity.ELECTRONIC_STATE,
        ),
    )

    assert bundle.execution_decision.action is StateDecisionAction.SELECT_STATE
    assert bundle.execution_decision.selected_state_identity is not None


def test_charge_only_operation_selects_path_with_unresolved_spin() -> None:
    bundle = resolve_scientific_state(
        _facts("[N]"),
        StateRequest(net_charge=0),
        requirements=_requirements(ScientificQuantity.NET_CHARGE),
    )

    assert bundle.scientific_resolution.status is ResolutionStatus.UNRESOLVED
    assert bundle.execution_decision.action is StateDecisionAction.SELECT_STATE
    assert bundle.execution_decision.selected_state_identity is not None


def test_missing_required_charge_returns_no_automatic_state() -> None:
    bundle = resolve_scientific_state(
        _facts(),
        requirements=_requirements(ScientificQuantity.NET_CHARGE),
    )

    assert (
        bundle.execution_decision.action
        is StateDecisionAction.NO_AUTOMATIC_STATE
    )


def test_provider_001_runs_only_when_explicitly_enabled_under_strict() -> None:
    facts = _facts()
    request = StateRequest(net_charge=0, two_s=0)

    disabled = resolve_scientific_state(facts, request)
    enabled = resolve_scientific_state(
        facts,
        request,
        provider_set=ProviderSet((ProviderToggle("PROVIDER-001", True),)),
    )

    assert disabled.lewis_results == ()
    assert enabled.lewis_results


def test_foreign_provided_lewis_result_is_hard_inadmissible() -> None:
    source_facts = _facts("CO")
    source_charge = resolve_net_charge(
        source_facts,
        StateRequest(net_charge=0),
    )[0]
    foreign_lewis = solve_main_group_lewis(
        source_facts,
        source_charge,
    )

    bundle = resolve_scientific_state(
        _facts(),
        provided_results=ProvidedScientificResults(
            lewis_results=foreign_lewis,
        ),
    )

    assert bundle.execution_decision.action is StateDecisionAction.INADMISSIBLE


def test_screening_policy_never_falls_back_to_strict_while_inactive() -> None:
    bundle = resolve_scientific_state(
        _facts(),
        StateRequest(net_charge=0, two_s=0),
        policy=StatePolicy.COORDINATION_SCREENING_V1,
    )

    assert (
        bundle.execution_decision.action
        is StateDecisionAction.NO_AUTOMATIC_STATE
    )
    assert (
        bundle.execution_decision.static_profile_eligibility.in_domain is False
    )


def test_unknown_provider_is_a_malformed_api_request() -> None:
    with pytest.raises(ValueError, match="Unknown provider"):
        resolve_scientific_state(
            _facts(),
            provider_set=ProviderSet((ProviderToggle("UNKNOWN", True),)),
        )


def test_conflicting_authoritative_charge_envelopes_do_not_form_a_union() -> None:
    facts = _facts()
    provided_charge = resolve_net_charge(
        facts,
        StateRequest(net_charge=1),
    )[0]

    bundle = resolve_scientific_state(
        facts,
        StateRequest(net_charge=0, two_s=0),
        provided_results=ProvidedScientificResults(
            net_charge_results=(provided_charge,),
        ),
    )

    assert bundle.execution_decision.action is StateDecisionAction.CONFLICT
    assert bundle.candidate_space.expanded_paths == ()
    assert bundle.candidate_space.constraints


def test_conflicting_authoritative_spin_envelopes_do_not_form_a_union() -> None:
    facts = _facts()
    charge_results = resolve_net_charge(
        facts,
        StateRequest(net_charge=0),
    )
    provided_spin = infer_electronic_states(
        facts,
        StateRequest(two_s=2),
        net_charge_results=charge_results,
    )[0]

    bundle = resolve_scientific_state(
        facts,
        StateRequest(net_charge=0, two_s=0),
        provided_results=ProvidedScientificResults(
            electronic_state_results=(provided_spin,),
        ),
    )

    assert bundle.execution_decision.action is StateDecisionAction.CONFLICT
    assert bundle.candidate_space.expanded_paths == ()


def test_authoritative_charge_constraint_precedes_factor_budget() -> None:
    facts = _facts()
    explicit_result = resolve_net_charge(
        facts,
        StateRequest(net_charge=0),
    )[0]
    q0 = explicit_result.candidates[0]
    q1 = NetChargeCandidateEvidence(
        CandidateProvenance(
            "0-excluded-q1",
            StateValueAuthority.VERIFIED_SCIENTIFIC_SOURCE,
            "test",
        ),
        1,
    )
    provided_result = replace(
        explicit_result,
        identity="provided:q0-or-q1",
        candidates=(q1, q0),
        resolution=ResolutionStatus.AMBIGUOUS,
    )

    bundle = resolve_scientific_state(
        facts,
        StateRequest(net_charge=0),
        provided_results=ProvidedScientificResults(
            net_charge_results=(provided_result,),
        ),
        candidate_budget=CandidateBudget(1, 8, 8),
    )

    assert tuple(
        path.net_charge.value
        for path in bundle.candidate_space.expanded_paths
    ) == (0,)


def test_explicit_oxidation_override_constrains_provided_envelope() -> None:
    facts = _facts("[Fe].[Fe]")
    oxidation_candidate = OxidationStateCandidateEvidence(
        CandidateProvenance(
            "os:all-zero",
            StateValueAuthority.VERIFIED_SCIENTIFIC_SOURCE,
            "test",
        ),
        CountingConvention.IONIC,
            (2, 3),
        CoverageStatus.COMPLETE,
    )
    provided_oxidation = infer_oxidation_states(
        facts,
        StateRequest(oxidation_candidates=(oxidation_candidate,)),
    )[0]

    bundle = resolve_scientific_state(
        facts,
        StateRequest(
            net_charge=5,
            oxidation_overrides=(OxidationStateOverride(0, 3),),
        ),
        provided_results=ProvidedScientificResults(
            oxidation_state_results=(provided_oxidation,),
        ),
    )

    assert bundle.execution_decision.action is StateDecisionAction.CONFLICT
    assert bundle.candidate_space.constraints
