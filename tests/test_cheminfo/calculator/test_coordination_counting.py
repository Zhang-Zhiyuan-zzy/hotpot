"""Convention-labelled coordination electron-counting tests."""

from dataclasses import replace

import hotpot

from hotpot.cheminfo.calculator.charge import (
    resolve_net_charge,
    solve_main_group_lewis,
)
from hotpot.cheminfo.calculator.contracts import (
    CandidateProvenance,
    CoverageStatus,
    ScientificAdmissibility,
    StateValueAuthority,
)
from hotpot.cheminfo.calculator.oxidation_state import (
    CountingConvention,
    OxidationStateCandidateEvidence,
    infer_oxidation_states,
    infer_coordination_electron_counting,
)
from hotpot.cheminfo.calculator.state_resolver import (
    StateRequest,
    extract_molecular_facts,
)


def _lewis(smiles: str, charge: int = 0):
    facts = extract_molecular_facts(hotpot.read_mol(smiles, "smi"))
    net_charge = resolve_net_charge(facts, StateRequest(net_charge=charge))[0]
    return facts, solve_main_group_lewis(facts, net_charge)[0]


def test_no_convention_does_not_run_hidden_counting_policy() -> None:
    facts, lewis = _lewis("CCO")

    assert infer_coordination_electron_counting(
        facts,
        lewis_results=(lewis,),
    ) == ()


def test_ionic_convention_preserves_component_bookkeeping_charges() -> None:
    facts, lewis = _lewis("[NH4+].[Cl-]")

    result = infer_coordination_electron_counting(
        facts,
        lewis_results=(lewis,),
        conventions=(CountingConvention.IONIC,),
    )[0]

    assert result.admissibility is ScientificAdmissibility.ADMISSIBLE
    assert tuple(
        fragment.ionic_bookkeeping_charge
        for fragment in result.candidates[0].fragments
    ) == tuple(charge for _, charge in lewis.candidates[0].component_charges)
    assert all(
        fragment.lxz_class is None and fragment.donated_electrons is None
        for fragment in result.candidates[0].fragments
    )


def test_nonionic_conventions_require_independent_rule_evidence() -> None:
    facts, lewis = _lewis("N")

    results = infer_coordination_electron_counting(
        facts,
        lewis_results=(lewis,),
        conventions=(CountingConvention.DONOR_RETURN, CountingConvention.CBC),
    )

    assert all(result.candidates == () for result in results)
    assert all(
        result.findings[0].code
        == "COUNTING.CONVENTION_REQUIRES_RULE_EVIDENCE"
        for result in results
    )


def test_inadmissible_lewis_envelope_cannot_feed_counting() -> None:
    facts, lewis = _lewis("CCO")

    result = infer_coordination_electron_counting(
        facts,
        lewis_results=(
            replace(
                lewis,
                admissibility=ScientificAdmissibility.INADMISSIBLE,
            ),
        ),
        conventions=(CountingConvention.IONIC,),
    )[0]

    assert result.admissibility is ScientificAdmissibility.NOT_EVALUATED
    assert result.findings[0].code == "HARD-005.LEWIS_RESULT_UNAVAILABLE"


def test_inadmissible_counting_envelope_cannot_feed_oxidation_state() -> None:
    facts, lewis = _lewis("CCO")
    counting = infer_coordination_electron_counting(
        facts,
        lewis_results=(lewis,),
        conventions=(CountingConvention.IONIC,),
    )[0]
    oxidation_candidate = OxidationStateCandidateEvidence(
        provenance=CandidateProvenance(
            "os",
            StateValueAuthority.VERIFIED_SCIENTIFIC_SOURCE,
            "test",
        ),
        convention=CountingConvention.IONIC,
        atom_states=(0, 0, 0),
        coverage=CoverageStatus.COMPLETE,
        coordination_counting_candidate_identity=(
            counting.candidates[0].provenance.identity
        ),
    )

    result = infer_oxidation_states(
        facts,
        StateRequest(oxidation_candidates=(oxidation_candidate,)),
        counting_results=(
            replace(
                counting,
                admissibility=ScientificAdmissibility.INADMISSIBLE,
            ),
        ),
    )[0]

    assert result.candidates == ()
    assert result.findings[0].code == "HARD-005.COUNTING_REFERENCE_MISSING"
