"""Safe coupled-spin preference bounds through the actual candidate search."""

from dataclasses import replace

import hotpot
import pytest

from hotpot.cheminfo.operations.calculation.scientific_state.composition.decision_search import (
    _record_beats_frontier,
    _select_decision_record,
)
from hotpot.cheminfo.operations.calculation.scientific_state.composition.expand import (
    _expand_scientific_paths,
)
from hotpot.cheminfo.operations.calculation.scientific_state.defaults import (
    _AutomaticCoordinationDecisionCompiler,
)
from hotpot.cheminfo.operations.calculation.scientific_state.facts import (
    extract_molecular_facts,
)
from hotpot.cheminfo.operations.calculation.scientific_state.profiles import (
    _coordination_screening_profile,
)
from hotpot.cheminfo.rules.chemistry.scientific_state.charge import (
    NetChargeCandidateEvidence,
)
from hotpot.cheminfo.rules.chemistry.scientific_state.composition import (
    CandidateDomainScope,
    CandidateFactor,
    CandidateSearchStatus,
    FactorCandidateValue,
    FactorKey,
    MaterializationStatus,
)
from hotpot.cheminfo.rules.chemistry.scientific_state.contracts import (
    CandidateProvenance,
    ScientificOperationRequirements,
    ScientificQuantity,
    StateValueAuthority,
)
from hotpot.cheminfo.rules.chemistry.scientific_state.electronic_state import (
    ElectronicStateCandidate,
)
from hotpot.cheminfo.rules.chemistry.scientific_state.providers import RankAdjustment


def _spin_search(last_rank_adjustment):
    facts = extract_molecular_facts(hotpot.read_mol("O", "smi"))
    authority = StateValueAuthority.DOMAIN_RULE
    charge = NetChargeCandidateEvidence(
        CandidateProvenance("charge:zero", authority, "bound-test"), 0,
    )
    spins = tuple(
        ElectronicStateCandidate(
            CandidateProvenance(identity, authority, "bound-test"),
            charge.provenance.identity, None, facts.nuclear_charge_sum,
            two_s, two_s + 1, True,
        )
        for identity, two_s in (("spin:a-triplet", 2), ("spin:z-singlet", 0))
    )
    factors = tuple(
        CandidateFactor(
            f"factor:{quantity.value}", FactorKey(quantity, "molecule"),
            "bound-test-domain", tuple(
                FactorCandidateValue(candidate, ("bound-test",), ())
                for candidate in candidates
            ), (), None, CandidateDomainScope.EXHAUSTIVE,
            MaterializationStatus.COMPLETE,
        )
        for quantity, candidates in (
            (ScientificQuantity.NET_CHARGE, (charge,)),
            (ScientificQuantity.ELECTRONIC_STATE, spins),
        )
    )
    profile = _coordination_screening_profile()
    compiler = _AutomaticCoordinationDecisionCompiler(
        profile,
        ScientificOperationRequirements(
            "spin-bound-test",
            frozenset((ScientificQuantity.NET_CHARGE, ScientificQuantity.ELECTRONIC_STATE)),
            frozenset(),
        ),
        (RankAdjustment(
            spins[1].provenance.identity, profile.decision_rank_dimensions[-1],
            last_rank_adjustment, "discarded rank replaced by coupled-spin preference",
        ),),
    )
    return facts, factors, compiler, charge, spins


@pytest.mark.parametrize("last_rank_adjustment", (-10.0, 0.0, 10.0))
def test_truncated_spin_search_cannot_certify_over_a_better_unseen_spin(
    last_rank_adjustment,
):
    facts, factors, compiler, charge, spins = _spin_search(last_rank_adjustment)

    def expand(limit):
        return _expand_scientific_paths(
            facts, factors, (),
            {charge.provenance.identity: ("charge-result",)},
            {candidate.provenance.identity: ("spin-result",) for candidate in spins},
            (), max_expanded_paths=limit, max_frontier_paths=8,
            decision_search=compiler,
        )

    all_paths, all_rejections, all_frontier, full_status = expand(8)
    selected = _select_decision_record(all_paths, factors, compiler)
    assert len(all_paths) == 2
    assert all_rejections == all_frontier == ()
    assert full_status is CandidateSearchStatus.DECISION_CERTIFIED
    assert selected.equivalence.molecular_two_s == 0

    paths, rejections, frontier, status = expand(1)
    assert len(paths) == len(frontier) == 1
    assert rejections == ()
    assert status is CandidateSearchStatus.RESOURCE_LIMIT_NO_CERTIFIED_DECISION
    partial = compiler.record_for_path(paths[0], factors)
    bounds = tuple(compiler.decode_bound(entry) for entry in frontier)
    assert not _record_beats_frontier(partial, bounds, compiler)


def test_spin_bound_retains_an_unboundable_assumption():
    _, factors, compiler, _, _ = _spin_search(-10.0)
    value = factors[1].values[1]
    candidate = replace(
        value.candidate,
        provenance=replace(
            value.candidate.provenance, assumption_ids=("unregistered:test",),
        ),
    )

    bound = compiler.optimistic_bound_for_prefix(
        (replace(value, candidate=candidate),), (factors[0],),
    )

    assert bound is None
