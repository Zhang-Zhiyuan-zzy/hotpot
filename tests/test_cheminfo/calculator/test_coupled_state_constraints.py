"""Coupled scientific candidate hard-consistency tests."""

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
from hotpot.cheminfo.calculator.electronic_state import (
    ElectronicStateCandidate,
)
from hotpot.cheminfo.calculator.oxidation_state import (
    CountingConvention,
    OxidationStateCandidateEvidence,
    infer_coordination_electron_counting,
)
from hotpot.cheminfo.calculator.state_composition.constraints import (
    _validate_coupled_candidates,
)
from hotpot.cheminfo.calculator.state_resolver import (
    StateRequest,
    extract_molecular_facts,
)


def _path_inputs():
    facts = extract_molecular_facts(
        hotpot.read_mol(
            "[H]C([H])([H])C([H])([H])O[H]",
            "smi",
        )
    )
    net_result = resolve_net_charge(facts, StateRequest(net_charge=0))[0]
    charge = net_result.candidates[0]
    lewis = solve_main_group_lewis(facts, net_result)[0].candidates[0]
    counting = infer_coordination_electron_counting(
        facts,
        lewis_results=(solve_main_group_lewis(facts, net_result)[0],),
        conventions=(CountingConvention.IONIC,),
    )[0].candidates[0]
    oxidation_state = OxidationStateCandidateEvidence(
        provenance=CandidateProvenance(
            "os:zero",
            StateValueAuthority.VERIFIED_SCIENTIFIC_SOURCE,
            "test",
        ),
        convention=CountingConvention.IONIC,
        atom_states=tuple(0 for _ in facts.atomic_numbers),
        coverage=CoverageStatus.COMPLETE,
        coordination_counting_candidate_identity=counting.provenance.identity,
    )
    electronic_state = ElectronicStateCandidate(
        provenance=CandidateProvenance(
            "spin:singlet",
            StateValueAuthority.VERIFIED_SCIENTIFIC_SOURCE,
            "test",
        ),
        net_charge_candidate_identity=charge.provenance.identity,
        oxidation_state_candidate_identity=oxidation_state.provenance.identity,
        electron_count=facts.nuclear_charge_sum,
        two_s=0,
        multiplicity=1,
        pure_spin=True,
    )
    return facts, charge, lewis, counting, oxidation_state, electronic_state


def test_complete_compatible_path_passes_all_hard_checks() -> None:
    facts, charge, lewis, counting, oxidation_state, electronic_state = (
        _path_inputs()
    )

    assessment = _validate_coupled_candidates(
        facts,
        charge,
        lewis=lewis,
        counting=counting,
        oxidation_state=oxidation_state,
        electronic_state=electronic_state,
    )

    assert assessment.admissibility is ScientificAdmissibility.ADMISSIBLE


def test_individually_structured_but_charge_incompatible_os_is_rejected() -> None:
    facts, charge, lewis, counting, oxidation_state, electronic_state = (
        _path_inputs()
    )
    incompatible = replace(
        oxidation_state,
        atom_states=(1,) + oxidation_state.atom_states[1:],
    )

    assessment = _validate_coupled_candidates(
        facts,
        charge,
        lewis=lewis,
        counting=counting,
        oxidation_state=incompatible,
    )

    assert assessment.admissibility is ScientificAdmissibility.INADMISSIBLE
    assert assessment.findings[0].code == "HARD-004.OS_SUM_MISMATCH"


def test_dangling_electronic_os_reference_is_rejected() -> None:
    facts, charge, lewis, counting, oxidation_state, electronic_state = (
        _path_inputs()
    )

    assessment = _validate_coupled_candidates(
        facts,
        charge,
        electronic_state=electronic_state,
    )

    assert assessment.admissibility is ScientificAdmissibility.INADMISSIBLE
    assert assessment.findings[-1].code == (
        "HARD-007.OXIDATION_STATE_REFERENCE_MISSING"
    )


def test_path_cannot_mix_counting_and_oxidation_conventions() -> None:
    facts, charge, lewis, counting, oxidation_state, electronic_state = (
        _path_inputs()
    )
    mismatched = replace(
        oxidation_state,
        convention=CountingConvention.CBC,
        coordination_counting_candidate_identity=None,
    )

    assessment = _validate_coupled_candidates(
        facts,
        charge,
        counting=counting,
        oxidation_state=mismatched,
    )

    assert assessment.admissibility is ScientificAdmissibility.INADMISSIBLE
    assert assessment.findings[0].code == (
        "HARD-005.COUNTING_CONVENTION_MISMATCH"
    )
