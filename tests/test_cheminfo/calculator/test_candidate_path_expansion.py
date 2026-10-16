"""Budgeted expansion tests for coupled scientific-state paths."""

import hotpot

from hotpot.cheminfo.calculator.charge import (
    NetChargeCandidateEvidence,
    resolve_net_charge,
)
from hotpot.cheminfo.calculator.contracts import (
    CandidateBudget,
    CandidateProvenance,
    CoverageStatus,
    ScientificQuantity,
    StateValueAuthority,
)
from hotpot.cheminfo.calculator.electronic_state import (
    ElectronicStateCandidate,
    SpinContributionCandidate,
    SpinContributionScope,
)
from hotpot.cheminfo.calculator.oxidation_state import (
    CountingConvention,
    OxidationStateCandidateEvidence,
)
from hotpot.cheminfo.calculator.state_composition.candidate_space import (
    _merge_candidate_proposals,
)
from hotpot.cheminfo.calculator.state_composition.contracts import (
    BranchAssignment,
    CandidateConstraint,
    CandidateDomainScope,
    CandidateFactor,
    CandidateDomainSummary,
    CandidateProposal,
    CandidateSearchStatus,
    ConstraintGuard,
    FactorKey,
    FrontierBound,
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


def _merged_factors(*factors):
    return _merge_candidate_proposals(
        (CandidateProposal("test:proposal", factors, ()),),
        CandidateBudget(8, 8, 8),
        (),
    )[0]


def _factors(charge, electronic, *, charge_branches=(), spin_branches=()):
    return _merged_factors(
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
    )


def test_one_exhaustive_hard_valid_path_is_certified() -> None:
    facts, charge, electronic = _scientific_candidates()
    factors = _factors(charge, electronic)

    paths, rejections, frontier, status = _expand_scientific_paths(
        facts,
        factors,
        (),
        {charge.provenance.identity: ("charge-result",)},
        {electronic.provenance.identity: ("electronic-result",)},
        (),
        max_expanded_paths=8,
        max_frontier_paths=8,
    )

    assert len(paths) == 1
    assert paths[0].net_charge.value == 0
    assert paths[0].net_charge.origin_identity == "charge-result"
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
        {charge.provenance.identity: ("charge-result",)},
        {incompatible.provenance.identity: ("electronic-result",)},
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
        {charge.provenance.identity: ("charge-result",)},
        {electronic.provenance.identity: ("electronic-result",)},
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
        {charge.provenance.identity: ("charge-result",)},
        {electronic.provenance.identity: ("electronic-result",)},
        (),
        max_expanded_paths=0,
        max_frontier_paths=8,
    )

    assert paths == ()
    assert len(frontier) == 1
    assert status is CandidateSearchStatus.RESOURCE_LIMIT_NO_CERTIFIED_DECISION


def test_prefix_frontier_compacts_an_unexpanded_cartesian_remainder() -> None:
    facts, _, _ = _scientific_candidates()
    charges = tuple(
        NetChargeCandidateEvidence(
            CandidateProvenance(
                f"charge:{index:02d}",
                StateValueAuthority.VERIFIED_SCIENTIFIC_SOURCE,
                "test",
            ),
            0,
        )
        for index in range(10)
    )
    electronic_states = tuple(
        ElectronicStateCandidate(
            CandidateProvenance(
                f"spin:{index:02d}",
                StateValueAuthority.VERIFIED_SCIENTIFIC_SOURCE,
                "test",
            ),
            charge.provenance.identity,
            None,
            facts.nuclear_charge_sum,
            0,
            1,
            True,
        )
        for index, charge in enumerate(charges)
    )
    factors = _merge_candidate_proposals(
        (
            CandidateProposal(
                "test:large-domain",
                (
                    ProposedCandidateFactor(
                        FactorKey(ScientificQuantity.NET_CHARGE, "molecule"),
                        "charge-domain-v1",
                        tuple(
                            ProposedCandidateValue(charge)
                            for charge in charges
                        ),
                        CandidateDomainScope.EXHAUSTIVE,
                        MaterializationStatus.COMPLETE,
                        None,
                    ),
                    ProposedCandidateFactor(
                        FactorKey(
                            ScientificQuantity.ELECTRONIC_STATE,
                            "molecule",
                        ),
                        "spin-domain-v1",
                        tuple(
                            ProposedCandidateValue(electronic)
                            for electronic in electronic_states
                        ),
                        CandidateDomainScope.EXHAUSTIVE,
                        MaterializationStatus.COMPLETE,
                        None,
                    ),
                ),
                (),
            ),
        ),
        CandidateBudget(16, 128, 128),
        (),
    )[0]
    charge_mapping = {
        charge.provenance.identity: (f"charge-result:{index}",)
        for index, charge in enumerate(charges)
    }
    electronic_mapping = {
        electronic.provenance.identity: (f"spin-result:{index}",)
        for index, electronic in enumerate(electronic_states)
    }

    paths, rejections, frontier, status = _expand_scientific_paths(
        facts,
        factors,
        (),
        charge_mapping,
        electronic_mapping,
        (),
        max_expanded_paths=1,
        max_frontier_paths=128,
    )

    assert len(paths) + len(rejections) == 1
    assert 0 < len(frontier) < 99
    assert len({bound.branch_identity for bound in frontier}) == len(frontier)
    assert status is CandidateSearchStatus.RESOURCE_LIMIT_NO_CERTIFIED_DECISION

    all_paths, all_rejections, all_frontier, _ = _expand_scientific_paths(
        facts,
        factors,
        (),
        charge_mapping,
        electronic_mapping,
        (),
        max_expanded_paths=128,
        max_frontier_paths=128,
    )

    assert len(all_paths) + len(all_rejections) == 100
    assert all_frontier == ()


def test_provider_frontier_bound_reaches_the_expanded_space() -> None:
    facts, charge, electronic = _scientific_candidates()
    bound = FrontierBound(
        "provider:next",
        "scientific-state-frontier-v1",
        ("best",),
    )
    charge_factor = ProposedCandidateFactor(
        FactorKey(ScientificQuantity.NET_CHARGE, "molecule"),
        "charge-domain-v1",
        (ProposedCandidateValue(charge),),
        CandidateDomainScope.DOMAIN_BOUNDED,
        MaterializationStatus.RESOURCE_TRUNCATED,
        CandidateDomainSummary(1, "provider:continuation", bound),
    )
    factors = _merged_factors(
        charge_factor,
        _factor(ScientificQuantity.ELECTRONIC_STATE, electronic),
    )

    _, _, frontier, status = _expand_scientific_paths(
        facts,
        factors,
        (),
        {charge.provenance.identity: ("charge-result",)},
        {electronic.provenance.identity: ("electronic-result",)},
        (),
        max_expanded_paths=8,
        max_frontier_paths=8,
    )

    assert bound in frontier
    assert status is CandidateSearchStatus.RESOURCE_LIMIT_NO_CERTIFIED_DECISION


def test_unknown_constraint_type_never_silently_passes() -> None:
    facts, charge, electronic = _scientific_candidates()
    factors = _factors(charge, electronic)
    constraint = CandidateConstraint(
        "unknown-constraint",
        "unimplemented_relation",
        "candidate-constraint-v1",
        (),
        (("candidate_ids", (charge.provenance.identity,)),),
    )

    paths, rejections, _, _ = _expand_scientific_paths(
        facts,
        factors,
        (GuardedCandidateConstraint(ConstraintGuard(), constraint),),
        {charge.provenance.identity: ("charge-result",)},
        {electronic.provenance.identity: ("electronic-result",)},
        (),
        max_expanded_paths=8,
        max_frontier_paths=8,
    )

    assert paths == ()
    assert rejections[0].constraint_identity == "unknown-constraint"
    assert rejections[0].reason_codes == ("CONSTRAINT.UNSUPPORTED_TYPE",)


def test_partial_oxidation_state_factor_is_retained_in_the_path() -> None:
    facts, charge, electronic = _scientific_candidates()
    oxidation_state = OxidationStateCandidateEvidence(
        CandidateProvenance(
            "os:partial",
            StateValueAuthority.VERIFIED_SCIENTIFIC_SOURCE,
            "test",
        ),
        CountingConvention.IONIC,
        (0, None, None),
        CoverageStatus.PARTIAL,
    )
    factors = _merged_factors(
        _factor(ScientificQuantity.NET_CHARGE, charge),
        _factor(ScientificQuantity.OXIDATION_STATE, oxidation_state),
        _factor(ScientificQuantity.ELECTRONIC_STATE, electronic),
    )

    paths, _, _, _ = _expand_scientific_paths(
        facts,
        factors,
        (),
        {charge.provenance.identity: ("charge-result",)},
        {electronic.provenance.identity: ("electronic-result",)},
        (),
        max_expanded_paths=8,
        max_frontier_paths=8,
    )

    assert paths[0].oxidation_state_assignment_id == "os:partial"


def test_spin_contribution_factor_is_not_silently_discarded() -> None:
    facts, charge, electronic = _scientific_candidates()
    contribution = SpinContributionCandidate(
        CandidateProvenance(
            "spin:component",
            StateValueAuthority.DOMAIN_RULE,
            "test",
        ),
        SpinContributionScope.COMPONENT,
        facts.covalent_components[0].identity,
        (),
        0,
        True,
    )
    factors = _merged_factors(
        _factor(ScientificQuantity.NET_CHARGE, charge),
        ProposedCandidateFactor(
            FactorKey(
                ScientificQuantity.ELECTRONIC_STATE,
                facts.covalent_components[0].identity,
            ),
            "spin-contribution-domain-v1",
            (ProposedCandidateValue(contribution),),
            CandidateDomainScope.DOMAIN_BOUNDED,
            MaterializationStatus.COMPLETE,
            None,
        ),
        _factor(ScientificQuantity.ELECTRONIC_STATE, electronic),
    )

    paths, _, _, status = _expand_scientific_paths(
        facts,
        factors,
        (),
        {charge.provenance.identity: ("charge-result",)},
        {electronic.provenance.identity: ("electronic-result",)},
        (),
        max_expanded_paths=8,
        max_frontier_paths=8,
    )

    assert contribution.provenance.identity in paths[0].evidence_ids
    assert status is CandidateSearchStatus.DECISION_CERTIFIED


def test_present_but_empty_factor_cannot_produce_a_certified_path() -> None:
    facts, charge, electronic = _scientific_candidates()
    populated = _factors(charge, electronic)
    empty = CandidateFactor(
        "empty-factor",
        FactorKey(ScientificQuantity.OXIDATION_STATE, "molecule"),
        "empty-domain-v1",
        (),
        (),
        None,
        CandidateDomainScope.EXHAUSTIVE,
        MaterializationStatus.COMPLETE,
    )

    paths, _, _, status = _expand_scientific_paths(
        facts,
        populated + (empty,),
        (),
        {charge.provenance.identity: ("charge-result",)},
        {electronic.provenance.identity: ("electronic-result",)},
        (),
        max_expanded_paths=8,
        max_frontier_paths=8,
    )

    assert paths == ()
    assert status is CandidateSearchStatus.NOT_EVALUATED


def test_unconditional_required_candidate_is_pruned_before_path_budget() -> None:
    facts, excluded_charge, _ = _scientific_candidates()
    required_charge = NetChargeCandidateEvidence(
        CandidateProvenance(
            "charge:+1",
            StateValueAuthority.VERIFIED_SCIENTIFIC_SOURCE,
            "test",
        ),
        1,
    )
    electronic = ElectronicStateCandidate(
        CandidateProvenance(
            "spin:doublet",
            StateValueAuthority.VERIFIED_SCIENTIFIC_SOURCE,
            "test",
        ),
        required_charge.provenance.identity,
        None,
        facts.nuclear_charge_sum - 1,
        1,
        2,
        True,
    )
    charge_key = FactorKey(ScientificQuantity.NET_CHARGE, "molecule")
    factors = _merged_factors(
        ProposedCandidateFactor(
            charge_key,
            "charge-domain-v1",
            (
                ProposedCandidateValue(excluded_charge),
                ProposedCandidateValue(required_charge),
            ),
            CandidateDomainScope.EXHAUSTIVE,
            MaterializationStatus.COMPLETE,
            None,
        ),
        _factor(ScientificQuantity.ELECTRONIC_STATE, electronic),
    )
    constraint = CandidateConstraint(
        "require-charge:+1",
        "requires_all",
        "candidate-constraint-v1",
        (charge_key,),
        (("candidate_ids", (required_charge.provenance.identity,)),),
    )

    paths, _, frontier, status = _expand_scientific_paths(
        facts,
        factors,
        (GuardedCandidateConstraint(ConstraintGuard(), constraint),),
        {required_charge.provenance.identity: ("charge-result:+1",)},
        {electronic.provenance.identity: ("electronic-result",)},
        (),
        max_expanded_paths=1,
        max_frontier_paths=8,
    )

    assert len(paths) == 1
    assert paths[0].net_charge.value == 1
    assert frontier == ()
    assert status is CandidateSearchStatus.DECISION_CERTIFIED
