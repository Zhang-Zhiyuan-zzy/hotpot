"""Budgeted expansion tests for coupled scientific-state paths."""

import hotpot

from hotpot.cheminfo.calculator.charge import resolve_net_charge
from hotpot.cheminfo.calculator.contracts import (
    CandidateBudget,
    CandidateProvenance,
    ScientificQuantity,
    StateValueAuthority,
)
from hotpot.cheminfo.calculator.electronic_state import (
    ElectronicStateCandidate,
)
from hotpot.cheminfo.calculator.state_composition.candidate_space import (
    _merge_candidate_proposals,
)
from hotpot.cheminfo.calculator.state_composition.contracts import (
    BranchAssignment,
    CandidateConstraint,
    CandidateDomainScope,
    CandidateProposal,
    CandidateSearchStatus,
    ConstraintGuard,
    FactorKey,
    GuardedCandidateConstraint,
    MaterializationStatus,
    ProposedCandidateFactor,
    ProposedCandidateValue,
)
from hotpot.cheminfo.calculator.state_composition.expand import (
    _expand_scientific_paths,
)
from hotpot.cheminfo.calculator.state_resolver import (
    StateRequest,
    extract_molecular_facts,
)


def _scientific_candidates():
    facts = extract_molecular_facts(hotpot.read_mol("CCO", "smi"))
    charge_result = resolve_net_charge(facts, StateRequest(net_charge=0))[0]
    charge = charge_result.candidates[0]
    electronic = ElectronicStateCandidate(
        provenance=CandidateProvenance(
            "spin:singlet",
            StateValueAuthority.VERIFIED_SCIENTIFIC_SOURCE,
            "test",
        ),
        net_charge_candidate_identity=charge.provenance.identity,
        oxidation_state_candidate_identity=None,
        electron_count=facts.nuclear_charge_sum,
        two_s=0,
        multiplicity=1,
        pure_spin=True,
    )
    return facts, charge, electronic


def _factor(
    quantity,
    candidate,
    *,
    branch_assignments=(),
):
    return ProposedCandidateFactor(
        FactorKey(quantity, "molecule"),
        "test-domain-v1",
        (ProposedCandidateValue(candidate, branch_assignments),),
        CandidateDomainScope.EXHAUSTIVE,
        MaterializationStatus.COMPLETE,
        None,
    )


def _factors(charge, electronic, *, charge_branches=(), spin_branches=()):
    proposal = CandidateProposal(
        "test:proposal",
        (
            _factor(
                ScientificQuantity.NET_CHARGE,
                charge,
                branch_assignments=charge_branches,
            ),
            _factor(
                ScientificQuantity.ELECTRONIC_STATE,
                electronic,
                branch_assignments=spin_branches,
            ),
        ),
        (),
    )
    return _merge_candidate_proposals(
        (proposal,),
        CandidateBudget(8, 8, 8),
    )[0]


def test_one_exhaustive_hard_valid_path_is_certified() -> None:
    facts, charge, electronic = _scientific_candidates()
    factors = _factors(charge, electronic)

    paths, rejections, frontier, status = _expand_scientific_paths(
        facts,
        factors,
        (),
        {electronic.provenance.identity: "electronic-result"},
        (),
        max_expanded_paths=8,
        max_frontier_paths=8,
    )

    assert len(paths) == 1
    assert paths[0].net_charge.value == 0
    assert rejections == frontier == ()
    assert status is CandidateSearchStatus.DECISION_CERTIFIED


def test_hard_incompatible_path_is_retained_as_rejection() -> None:
    facts, charge, electronic = _scientific_candidates()
    incompatible = ElectronicStateCandidate(
        provenance=CandidateProvenance(
            "spin:wrong-charge",
            StateValueAuthority.VERIFIED_SCIENTIFIC_SOURCE,
            "test",
        ),
        net_charge_candidate_identity="another-charge",
        oxidation_state_candidate_identity=None,
        electron_count=facts.nuclear_charge_sum,
        two_s=0,
        multiplicity=1,
        pure_spin=True,
    )
    factors = _factors(charge, incompatible)

    paths, rejections, _, _ = _expand_scientific_paths(
        facts,
        factors,
        (),
        {incompatible.provenance.identity: "electronic-result"},
        (),
        max_expanded_paths=8,
        max_frontier_paths=8,
    )

    assert paths == ()
    assert rejections[0].constraint_identity == (
        "hard-scientific-consistency-v1"
    )
    assert "HARD-007.NET_CHARGE_REFERENCE_MISMATCH" in (
        rejections[0].reason_codes
    )


def test_branch_conflict_is_rejected_before_hard_validation() -> None:
    facts, charge, electronic = _scientific_candidates()
    factors = _factors(
        charge,
        electronic,
        charge_branches=(BranchAssignment("state", "a"),),
        spin_branches=(BranchAssignment("state", "b"),),
    )

    paths, rejections, _, _ = _expand_scientific_paths(
        facts,
        factors,
        (),
        {electronic.provenance.identity: "electronic-result"},
        (),
        max_expanded_paths=8,
        max_frontier_paths=8,
    )

    assert paths == ()
    assert rejections[0].reason_codes == ("COMPOSITION.BRANCH_CONFLICT",)


def test_resource_limit_retains_a_frontier_and_refuses_certification() -> None:
    facts, charge, electronic = _scientific_candidates()
    factors = _factors(charge, electronic)

    paths, _, frontier, status = _expand_scientific_paths(
        facts,
        factors,
        (),
        {electronic.provenance.identity: "electronic-result"},
        (),
        max_expanded_paths=0,
        max_frontier_paths=8,
    )

    assert paths == ()
    assert len(frontier) == 1
    assert status is CandidateSearchStatus.RESOURCE_LIMIT_NO_CERTIFIED_DECISION


def test_unknown_constraint_type_never_silently_passes() -> None:
    facts, charge, electronic = _scientific_candidates()
    factors = _factors(charge, electronic)
    constraint = CandidateConstraint(
        "unknown-constraint",
        "unimplemented_relation",
        "v1",
        (),
        (),
    )

    paths, rejections, _, _ = _expand_scientific_paths(
        facts,
        factors,
        (GuardedCandidateConstraint(ConstraintGuard(), constraint),),
        {electronic.provenance.identity: "electronic-result"},
        (),
        max_expanded_paths=8,
        max_frontier_paths=8,
    )

    assert paths == ()
    assert rejections[0].constraint_identity == "unknown-constraint"
    assert rejections[0].reason_codes == ("CONSTRAINT.UNSUPPORTED_TYPE",)
