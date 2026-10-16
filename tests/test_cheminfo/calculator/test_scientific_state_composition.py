"""Public scientific-state composition and resolution tests."""

from dataclasses import replace

import hotpot
import pytest

from hotpot.cheminfo.calculator.charge import (
    resolve_net_charge,
    solve_main_group_lewis,
)
from hotpot.cheminfo.calculator.contracts import (
    CandidateBudget,
    ResolutionStatus,
    ScientificQuantity,
)
from hotpot.cheminfo.calculator.electronic_state import (
    infer_electronic_states,
)
from hotpot.cheminfo.calculator.state_composition import (
    CandidateProposal,
    CandidateDomainScope,
    CandidateSearchStatus,
    FactorKey,
    MaterializationStatus,
    ProposedCandidateFactor,
    ProposedCandidateValue,
    StateValueOrigin,
    compose_scientific_state_space,
)
from hotpot.cheminfo.calculator.state_composition.resolution import (
    _resolve_candidate_space,
)
from hotpot.cheminfo.calculator.state_resolver import (
    StateRequest,
    extract_molecular_facts,
)


def _explicit_results():
    facts = extract_molecular_facts(hotpot.read_mol("CCO", "smi"))
    request = StateRequest(net_charge=0, two_s=0)
    net_charge_results = resolve_net_charge(facts, request)
    electronic_results = infer_electronic_states(
        facts,
        request,
        net_charge_results=net_charge_results,
    )
    return facts, net_charge_results, electronic_results


def test_public_composition_retains_result_envelope_lineage() -> None:
    facts, net_charge_results, electronic_results = _explicit_results()

    space = compose_scientific_state_space(
        facts,
        net_charge_results=net_charge_results,
        electronic_state_results=electronic_results,
    )

    assert len(space.factors) == 2
    assert len(space.expanded_paths) == 1
    assert space.search_status is CandidateSearchStatus.DECISION_CERTIFIED
    path = space.expanded_paths[0]
    assert path.net_charge.origin is StateValueOrigin.RESULT
    assert path.net_charge.origin_identity == net_charge_results[0].identity
    assert path.electronic_state_result_id == electronic_results[0].identity
    resolution = _resolve_candidate_space(space)
    assert resolution.status is ResolutionStatus.UNIQUE
    assert resolution.exhaustive is True


def test_bounded_lewis_factor_prevents_false_scientific_uniqueness() -> None:
    facts, net_charge_results, electronic_results = _explicit_results()
    lewis_results = solve_main_group_lewis(facts, net_charge_results[0])

    space = compose_scientific_state_space(
        facts,
        net_charge_results=net_charge_results,
        lewis_results=lewis_results,
        electronic_state_results=electronic_results,
    )

    lewis_factor = next(
        factor
        for factor in space.factors
        if factor.factor_key.quantity.value == "lewis_formal_charge"
    )
    assert lewis_factor.domain_scope is CandidateDomainScope.DOMAIN_BOUNDED
    assert space.search_status is CandidateSearchStatus.DECISION_CERTIFIED
    resolution = _resolve_candidate_space(space)
    assert resolution.status is ResolutionStatus.AMBIGUOUS
    assert resolution.exhaustive is False


def test_unresolved_electronic_envelope_remains_explicitly_unresolved() -> None:
    facts = extract_molecular_facts(hotpot.read_mol("CCO", "smi"))
    net_charge_results = resolve_net_charge(
        facts,
        StateRequest(net_charge=0),
    )
    electronic_results = infer_electronic_states(
        facts,
        net_charge_results=net_charge_results,
    )

    space = compose_scientific_state_space(
        facts,
        net_charge_results=net_charge_results,
        electronic_state_results=electronic_results,
    )

    assert len(space.expanded_paths) == 1
    assert (
        space.expanded_paths[0].electronic_state_candidate_identity is None
    )
    assert space.search_status is CandidateSearchStatus.NO_CERTIFIED_DECISION
    assert (
        _resolve_candidate_space(space).status
        is ResolutionStatus.UNRESOLVED
    )


def test_foreign_result_envelopes_do_not_enter_candidate_factors() -> None:
    facts, net_charge_results, electronic_results = _explicit_results()
    other_facts = extract_molecular_facts(hotpot.read_mol("CO", "smi"))

    space = compose_scientific_state_space(
        other_facts,
        net_charge_results=net_charge_results,
        electronic_state_results=electronic_results,
    )

    assert space.factors == ()
    assert space.expanded_paths == ()
    assert space.search_status is CandidateSearchStatus.NOT_EVALUATED


def test_direct_candidate_proposal_cannot_bypass_producer_budget() -> None:
    facts, net_charge_results, _ = _explicit_results()
    charge = net_charge_results[0].candidates[0]
    other_charge = replace(
        charge,
        provenance=replace(charge.provenance, identity="charge:other"),
        charge=1,
    )
    proposal = CandidateProposal(
        "direct:over-budget",
        (
            ProposedCandidateFactor(
                FactorKey(ScientificQuantity.NET_CHARGE, "molecule"),
                "direct-charge-domain-v1",
                tuple(
                    ProposedCandidateValue(candidate)
                    for candidate in (charge, other_charge)
                ),
                CandidateDomainScope.DOMAIN_BOUNDED,
                MaterializationStatus.COMPLETE,
                None,
            ),
        ),
        (),
    )

    with pytest.raises(ValueError, match="materialization budget"):
        compose_scientific_state_space(
            facts,
            net_charge_results=net_charge_results,
            candidate_proposals=(proposal,),
            candidate_budget=CandidateBudget(1, 8, 8),
        )
