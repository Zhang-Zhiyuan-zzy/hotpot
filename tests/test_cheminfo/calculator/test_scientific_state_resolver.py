"""End-to-end strict scientific-state resolver tests."""

from dataclasses import asdict

import hotpot
import pytest

from hotpot.cheminfo.calculator import resolve_scientific_state
from hotpot.cheminfo.calculator.charge import (
    resolve_net_charge,
    solve_main_group_lewis,
)
from hotpot.cheminfo.calculator.contracts import (
    ProviderSet,
    ProviderToggle,
    ScientificOperationRequirements,
    ScientificQuantity,
    StatePolicy,
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
