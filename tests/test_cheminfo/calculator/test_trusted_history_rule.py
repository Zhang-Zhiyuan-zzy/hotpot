"""OPTIONAL-014 exact-identity trusted-history tests."""

from dataclasses import replace

import hotpot

from hotpot.cheminfo.rules.chemistry.scientific_state.contracts import StateValueAuthority
from hotpot.cheminfo.operations.calculation.scientific_state.facts import extract_molecular_facts
from hotpot.cheminfo.rules.chemistry.scientific_state.providers import (
    RuleContext,
    RulePhase,
    RuleToggle,
)
from hotpot.cheminfo.rules.chemistry.scientific_state.providers.trusted_history import (
    TrustedHistoryRule,
    TrustedScientificStateRecord,
)
from hotpot.cheminfo.operations.calculation.scientific_state.operation_requirements import (
    _effective_requirements,
)
from hotpot.cheminfo.rules.chemistry.scientific_state.contracts import DEFAULT_CANDIDATE_BUDGET
from hotpot.cheminfo.operations.calculation.scientific_state.contracts import StatePolicy


def _context(facts):
    return RuleContext(
        facts=facts,
        phase=RulePhase.SOURCE_EVIDENCE,
        requirements=_effective_requirements(
            StatePolicy.COORDINATION_SCREENING_V1,
            None,
        ),
        candidate_space=None,
        prior_outcomes=(),
        candidate_budget=DEFAULT_CANDIDATE_BUDGET,
    )


def _record(facts, identity="history:fe-ii"):
    return TrustedScientificStateRecord(
        identity=identity,
        facts_identity_digest=facts.identity.chemical_identity_digest,
        atom_map=facts.atom_map,
        source_identity="curated-scientific-history-v1",
        source_digest="sha256:trusted-source",
        authority=StateValueAuthority.EXACT_IDENTITY_TRUSTED_SOURCE,
        net_charge=2,
        oxidation_states=(2,),
        two_s=4,
        multiplicity=5,
        assumption_ids=("domain_template:trusted-history",),
        locator="record://fe-ii",
    )


def test_exact_identity_history_produces_correlated_state_candidates() -> None:
    facts = extract_molecular_facts(hotpot.read_mol("[Fe]", "smi"))
    outcome = TrustedHistoryRule((_record(facts),)).evaluate(
        RuleToggle("OPTIONAL-014", True),
        _context(facts),
    )

    assert outcome.applicability.matches
    assert outcome.candidate_proposal is not None
    assert len(outcome.candidate_proposal.factors) == 3
    candidates = tuple(
        value.candidate
        for factor in outcome.candidate_proposal.factors
        for value in factor.materialized_candidates
    )
    assert all(
        candidate.provenance.authority
        is StateValueAuthority.EXACT_IDENTITY_TRUSTED_SOURCE
        for candidate in candidates
    )
    assert {
        assignment.option_id
        for factor in outcome.candidate_proposal.factors
        for value in factor.materialized_candidates
        for assignment in value.branch_assignments
    } == {"history:fe-ii"}
    assert len(outcome.rank_adjustments) == 3


def test_history_with_different_atom_map_abstains() -> None:
    facts = extract_molecular_facts(hotpot.read_mol("[Fe]", "smi"))
    record = _record(facts)
    mismatched = replace(record, atom_map=(999,))

    outcome = TrustedHistoryRule((mismatched,)).evaluate(
        RuleToggle("OPTIONAL-014", True),
        _context(facts),
    )

    assert not outcome.applicability.matches
    assert outcome.applicability.reason_codes == (
        "EXACT_IDENTITY_HISTORY_UNAVAILABLE",
    )
    assert outcome.candidate_proposal is None


def test_history_alternatives_remain_distinct_correlated_branches() -> None:
    facts = extract_molecular_facts(hotpot.read_mol("[Fe]", "smi"))
    first = _record(facts, "history:fe-ii")
    second = replace(
        first,
        identity="history:fe-iii",
        net_charge=3,
        oxidation_states=(3,),
        two_s=5,
        multiplicity=6,
    )

    outcome = TrustedHistoryRule((first, second)).evaluate(
        RuleToggle("OPTIONAL-014", True),
        _context(facts),
    )

    option_ids = {
        assignment.option_id
        for factor in outcome.candidate_proposal.factors
        for value in factor.materialized_candidates
        for assignment in value.branch_assignments
    }
    assert option_ids == {"history:fe-ii", "history:fe-iii"}
