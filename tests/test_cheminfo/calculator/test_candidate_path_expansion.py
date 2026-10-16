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
from hotpot.cheminfo.calculator.state_composition.decision_search import (
    _AuthorityVector,
    _DecisionBound,
    _DecisionEquivalenceKey,
    _DecisionRecord,
    _DecisionRelation,
    _PreferenceKey,
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


class _RankedSearchCompiler:
    def __init__(self, lower_rank: float = 1.0) -> None:
        self._lower_rank = lower_rank

    @property
    def identity(self):
        return "test-ranked-search-v1"

    @property
    def frontier_schema_version(self):
        return "scientific-state-frontier-v1"

    def _rank_for_candidate_ids(self, candidate_ids):
        zero = {"charge:zero", "spin:zero"}
        one = {"charge:one", "spin:one"}
        selected = set(candidate_ids)
        if selected.intersection(zero) and selected.intersection(one):
            return 0.0
        if selected.intersection(zero):
            return 2.0
        if selected.intersection(one):
            return self._lower_rank
        return max(2.0, self._lower_rank)

    def record_for_path(self, path, factors):
        del factors
        rank = 2.0 if path.net_charge.value == 0 else self._lower_rank
        return _DecisionRecord(
            path.identity,
            _AuthorityVector((("state", 1),)),
            _PreferenceKey((rank,), ()),
            _DecisionEquivalenceKey(
                path.net_charge.value,
                "none",
                path.net_charge.value,
            ),
        )

    def optimistic_bound_for_prefix(self, selected_values, remaining_factors):
        del remaining_factors
        rank = self._rank_for_candidate_ids(
            tuple(
                value.candidate.provenance.identity
                for value in selected_values
            )
        )
        return _DecisionBound(
            _AuthorityVector((("state", 1),)),
            (rank,),
            (),
            None,
            None,
        )

    def compare_records(self, left, right):
        left_rank = left.preference.rank_values
        right_rank = right.preference.rank_values
        if left_rank > right_rank:
            return _DecisionRelation.DOMINATES
        if left_rank < right_rank:
            return _DecisionRelation.DOMINATED
        return _DecisionRelation.EQUIVALENT

    def record_strictly_beats_bound(self, record, bound):
        return record.preference.rank_values > bound.rank_upper

    def aggregate_bounds(self, bounds):
        return max(bounds, key=lambda bound: bound.rank_upper)

    def bound_order_key(self, bound):
        return (f"{1000.0 - bound.rank_upper[0]:012.6f}",)

    def encode_bound(self, bound):
        return (f"rank:{bound.rank_upper[0]:.6f}",)

    def decode_bound(self, bound):
        if bound.schema_version != self.frontier_schema_version:
            return None
        prefix = "rank:"
        if len(bound.optimistic_key) != 1 or not bound.optimistic_key[0].startswith(
            prefix
        ):
            return None
        return _DecisionBound(
            _AuthorityVector((("state", 1),)),
            (float(bound.optimistic_key[0][len(prefix) :]),),
            (),
            None,
            None,
        )


def _ranked_search_fixture():
    facts, _, _ = _scientific_candidates()
    zero_charge = NetChargeCandidateEvidence(
        CandidateProvenance(
            "charge:zero",
            StateValueAuthority.VERIFIED_SCIENTIFIC_SOURCE,
            "test",
        ),
        0,
    )
    one_charge = NetChargeCandidateEvidence(
        CandidateProvenance(
            "charge:one",
            StateValueAuthority.VERIFIED_SCIENTIFIC_SOURCE,
            "test",
        ),
        1,
    )
    zero_spin = ElectronicStateCandidate(
        CandidateProvenance(
            "spin:zero",
            StateValueAuthority.VERIFIED_SCIENTIFIC_SOURCE,
            "test",
        ),
        zero_charge.provenance.identity,
        None,
        facts.nuclear_charge_sum,
        0,
        1,
        True,
    )
    one_spin = ElectronicStateCandidate(
        CandidateProvenance(
            "spin:one",
            StateValueAuthority.VERIFIED_SCIENTIFIC_SOURCE,
            "test",
        ),
        one_charge.provenance.identity,
        None,
        facts.nuclear_charge_sum - 1,
        1,
        2,
        True,
    )
    factors = _merged_factors(
        ProposedCandidateFactor(
            FactorKey(ScientificQuantity.NET_CHARGE, "molecule"),
            "charge-domain-v1",
            tuple(
                ProposedCandidateValue(candidate)
                for candidate in (zero_charge, one_charge)
            ),
            CandidateDomainScope.EXHAUSTIVE,
            MaterializationStatus.COMPLETE,
            None,
        ),
        ProposedCandidateFactor(
            FactorKey(ScientificQuantity.ELECTRONIC_STATE, "molecule"),
            "spin-domain-v1",
            tuple(
                ProposedCandidateValue(candidate)
                for candidate in (zero_spin, one_spin)
            ),
            CandidateDomainScope.EXHAUSTIVE,
            MaterializationStatus.COMPLETE,
            None,
        ),
    )
    return (
        facts,
        factors,
        {
            zero_charge.provenance.identity: ("charge-result:zero",),
            one_charge.provenance.identity: ("charge-result:one",),
        },
        {
            zero_spin.provenance.identity: ("spin-result:zero",),
            one_spin.provenance.identity: ("spin-result:one",),
        },
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


def test_policy_bound_certifies_a_truncated_best_first_decision() -> None:
    facts, factors, charge_mapping, electronic_mapping = (
        _ranked_search_fixture()
    )

    paths, _, frontier, status = _expand_scientific_paths(
        facts,
        factors,
        (),
        charge_mapping,
        electronic_mapping,
        (),
        max_expanded_paths=1,
        max_frontier_paths=8,
        decision_search=_RankedSearchCompiler(),
    )

    assert len(paths) == 1
    assert paths[0].net_charge.value == 0
    assert frontier
    assert status is (
        CandidateSearchStatus.TRUNCATED_BUT_DECISION_CERTIFIED
    )


def test_equal_non_equivalent_frontier_prevents_truncated_certification() -> None:
    facts, factors, charge_mapping, electronic_mapping = (
        _ranked_search_fixture()
    )

    paths, _, frontier, status = _expand_scientific_paths(
        facts,
        factors,
        (),
        charge_mapping,
        electronic_mapping,
        (),
        max_expanded_paths=1,
        max_frontier_paths=8,
        decision_search=_RankedSearchCompiler(lower_rank=2.0),
    )

    assert len(paths) == 1
    assert frontier
    assert status is CandidateSearchStatus.RESOURCE_LIMIT_NO_CERTIFIED_DECISION


def test_complete_non_equivalent_tie_is_not_decision_certified() -> None:
    facts, factors, charge_mapping, electronic_mapping = (
        _ranked_search_fixture()
    )

    paths, _, frontier, status = _expand_scientific_paths(
        facts,
        factors,
        (),
        charge_mapping,
        electronic_mapping,
        (),
        max_expanded_paths=8,
        max_frontier_paths=8,
        decision_search=_RankedSearchCompiler(lower_rank=2.0),
    )

    assert len(paths) == 2
    assert frontier == ()
    assert status is CandidateSearchStatus.NO_CERTIFIED_DECISION


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

    _, _, compact_frontier, _ = _expand_scientific_paths(
        facts,
        factors,
        (),
        charge_mapping,
        electronic_mapping,
        (),
        max_expanded_paths=1,
        max_frontier_paths=2,
    )
    _, _, repeated_frontier, _ = _expand_scientific_paths(
        facts,
        factors,
        (),
        charge_mapping,
        electronic_mapping,
        (),
        max_expanded_paths=1,
        max_frontier_paths=2,
    )

    assert len(compact_frontier) == 2
    assert compact_frontier == repeated_frontier
    assert compact_frontier[-1].optimistic_key == ("unbounded",)


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
