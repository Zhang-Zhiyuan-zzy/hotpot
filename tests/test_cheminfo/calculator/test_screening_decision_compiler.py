"""Production coordination-screening decision compiler tests."""

from dataclasses import replace

import pytest

from hotpot.cheminfo.calculator.charge import NetChargeCandidateEvidence
from hotpot.cheminfo.calculator.contracts import (
    CandidateProvenance,
    ScientificOperationRequirements,
    ScientificQuantity,
    StateValueAuthority,
)
from hotpot.cheminfo.calculator.electronic_state import (
    ElectronicStateCandidate,
    SpinContributionCandidate,
    SpinContributionScope,
)
from hotpot.cheminfo.calculator.state_composition import (
    CandidateDomainScope,
    CandidateFactor,
    CoupledScientificStateCandidate,
    FactorCandidateValue,
    FactorKey,
    FrontierBound,
    MaterializationStatus,
    ScientificStateIdentity,
    StateValueOrigin,
    StateValueRef,
)
from hotpot.cheminfo.calculator.state_composition.decision_search import (
    _DecisionRelation,
)
from hotpot.cheminfo.calculator.state_rules import RankAdjustment
from hotpot.cheminfo.calculator.state_rules.profile_decision import (
    _CoordinationScreeningDecisionCompiler,
)
from hotpot.cheminfo.calculator.state_rules.profiles import (
    _coordination_screening_profile,
)


def _provenance(identity, authority, assumptions=()):
    return CandidateProvenance(
        identity,
        authority,
        "test",
        assumption_ids=assumptions,
    )


def _factor(quantity, *candidates):
    return CandidateFactor(
        f"factor:{quantity.value}",
        FactorKey(quantity, "molecule"),
        f"domain:{quantity.value}",
        tuple(
            FactorCandidateValue(candidate, ("test",), ())
            for candidate in candidates
        ),
        (),
        None,
        CandidateDomainScope.DOMAIN_BOUNDED,
        MaterializationStatus.COMPLETE,
    )


def _fixture():
    q0 = NetChargeCandidateEvidence(
        _provenance(
            "charge:zero",
            StateValueAuthority.VERIFIED_EXPLICIT,
        ),
        0,
    )
    q1 = NetChargeCandidateEvidence(
        _provenance("charge:one", StateValueAuthority.DOMAIN_RULE),
        1,
    )
    spin0 = ElectronicStateCandidate(
        _provenance("spin:zero", StateValueAuthority.DOMAIN_RULE),
        q0.provenance.identity,
        None,
        10,
        0,
        1,
        True,
    )
    spin1 = ElectronicStateCandidate(
        _provenance(
            "spin:one",
            StateValueAuthority.VERIFIED_EXPLICIT,
        ),
        q1.provenance.identity,
        None,
        9,
        1,
        2,
        True,
    )
    factors = (
        _factor(ScientificQuantity.NET_CHARGE, q0, q1),
        _factor(ScientificQuantity.ELECTRONIC_STATE, spin0, spin1),
    )
    zero_path = CoupledScientificStateCandidate(
        ScientificStateIdentity("state:zero"),
        StateValueRef(
            ScientificQuantity.NET_CHARGE,
            0,
            StateValueOrigin.RESULT,
            "charge-result:zero",
        ),
        (),
        (),
        None,
        "spin-result:zero",
        spin0.provenance.identity,
        (),
        (q0.provenance.identity, spin0.provenance.identity),
        (),
    )
    one_path = CoupledScientificStateCandidate(
        ScientificStateIdentity("state:one"),
        StateValueRef(
            ScientificQuantity.NET_CHARGE,
            1,
            StateValueOrigin.RESULT,
            "charge-result:one",
        ),
        (),
        (),
        None,
        "spin-result:one",
        spin1.provenance.identity,
        (),
        (q1.provenance.identity, spin1.provenance.identity),
        (),
    )
    return factors, zero_path, one_path


def _compiler(*adjustments):
    requirements = ScientificOperationRequirements(
        "test-operation",
        frozenset(
            {
                ScientificQuantity.NET_CHARGE,
                ScientificQuantity.ELECTRONIC_STATE,
            }
        ),
        frozenset(),
    )
    return _CoordinationScreeningDecisionCompiler(
        _coordination_screening_profile(),
        requirements,
        tuple(adjustments),
    )


def test_profile_rank_resolves_a_cross_quantity_authority_tradeoff() -> None:
    factors, zero_path, one_path = _fixture()
    compiler = _compiler(
        RankAdjustment(
            "charge:zero",
            "oxidation_state_commonality",
            1.0,
            "test preference",
        )
    )
    zero = compiler.record_for_path(zero_path, factors)
    one = compiler.record_for_path(one_path, factors)

    assert compiler.compare_records(zero, one) is _DecisionRelation.DOMINATES
    assert zero.equivalence.net_charge == 0
    assert zero.equivalence.molecular_two_s == 0


def test_profile_reads_two_s_from_a_molecular_spin_contribution() -> None:
    charge = NetChargeCandidateEvidence(
        _provenance("charge:derived-spin", StateValueAuthority.DOMAIN_RULE),
        0,
    )
    contribution = SpinContributionCandidate(
        _provenance(
            "spin:derived-contribution",
            StateValueAuthority.DOMAIN_RULE,
        ),
        SpinContributionScope.MOLECULAR,
        "molecule:test",
        (charge.provenance.identity,),
        2,
        True,
    )
    factors = (
        _factor(ScientificQuantity.NET_CHARGE, charge),
        _factor(ScientificQuantity.ELECTRONIC_STATE, contribution),
    )
    path = CoupledScientificStateCandidate(
        ScientificStateIdentity("state:derived-spin"),
        StateValueRef(
            ScientificQuantity.NET_CHARGE,
            0,
            StateValueOrigin.RESULT,
            "charge-result:derived-spin",
        ),
        (),
        (),
        None,
        "rule:derived-spin",
        "electronic:derived-spin",
        (),
        (charge.provenance.identity, contribution.provenance.identity),
        (),
    )

    record = _compiler().record_for_path(path, factors)

    assert record.equivalence.molecular_two_s == 2


def test_unknown_assumption_marks_a_record_uncertifiable() -> None:
    factors, zero_path, _ = _fixture()
    compiler = _compiler()
    unknown = replace(
        zero_path,
        assumption_ids=("unreviewed:value",),
    )

    record = compiler.record_for_path(unknown, factors)

    assert record.certifiable is False


def test_rank_assumptions_participate_in_severity_and_prefix_bounds() -> None:
    factors, zero_path, _ = _fixture()
    compiler = _compiler(
        RankAdjustment(
            "charge:zero",
            "oxidation_state_commonality",
            1.0,
            "execution prior",
            ("execution_prior:test",),
        )
    )

    record = compiler.record_for_path(zero_path, factors)
    bound = compiler.optimistic_bound_for_prefix(
        (factors[0].values[0],),
        (factors[1],),
    )

    assert record.preference.assumption_severity[0] == 1
    assert bound.assumption_severity_lower[0] == 1


def test_prefix_bound_round_trip_and_aggregation_are_stable() -> None:
    factors, _, _ = _fixture()
    compiler = _compiler(
        RankAdjustment(
            "charge:zero",
            "oxidation_state_commonality",
            1.0,
            "test preference",
        )
    )
    prefix = (factors[0].values[0],)
    bound = compiler.optimistic_bound_for_prefix(prefix, (factors[1],))

    assert bound is not None
    encoded = compiler.encode_bound(bound)
    decoded = compiler.decode_bound(
        FrontierBound(
            "branch",
            compiler.frontier_schema_version,
            encoded,
        )
    )
    assert decoded == bound
    assert compiler.aggregate_bounds((bound, decoded)) == bound
    assert compiler.bound_order_key(bound) == compiler.bound_order_key(decoded)


def test_prefix_bound_excludes_ranked_sibling_candidates() -> None:
    factors, _, _ = _fixture()
    compiler = _compiler(
        RankAdjustment(
            "charge:zero",
            "oxidation_state_commonality",
            1.0,
            "zero preference",
        ),
        RankAdjustment(
            "charge:one",
            "oxidation_state_commonality",
            2.0,
            "one preference",
        ),
    )

    zero_bound = compiler.optimistic_bound_for_prefix(
        (factors[0].values[0],),
        (factors[1],),
    )
    one_bound = compiler.optimistic_bound_for_prefix(
        (factors[0].values[1],),
        (factors[1],),
    )

    assert zero_bound is not None
    assert one_bound is not None
    assert zero_bound.rank_upper < one_bound.rank_upper


def test_unknown_rank_dimension_is_rejected_by_profile_schema() -> None:
    with pytest.raises(ValueError, match="profile schema"):
        _compiler(
            RankAdjustment(
                "charge:zero",
                "unregistered-rank",
                1.0,
                "test preference",
            )
        )
