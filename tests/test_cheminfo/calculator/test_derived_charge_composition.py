"""Hard-equation derivation tests for coupled scientific states."""

import hotpot

from hotpot.cheminfo.calculator.charge import resolve_net_charge
from hotpot.cheminfo.calculator.contracts import (
    CandidateProvenance,
    ChemicalConfidence,
    CoverageStatus,
    ResolutionStatus,
    ScientificAdmissibility,
    ScientificQuantity,
    StateValueAuthority,
)
from hotpot.cheminfo.calculator.electronic_state import (
    ElectronicStateInferenceResult,
)
from hotpot.cheminfo.calculator.oxidation_state import (
    CoordinationCountingCandidate,
    CoordinationElectronCountingResult,
    CoordinationFragmentAssignment,
    CountingConvention,
    OxidationStateCandidateEvidence,
    OxidationStateInferenceResult,
)
from hotpot.cheminfo.calculator.state_composition import (
    DerivedStateValue,
    StateValueOrigin,
    compose_scientific_state_space,
)
from hotpot.cheminfo.calculator.state_resolver import (
    StateRequest,
    extract_molecular_facts,
)
from hotpot.cheminfo.core import BondKind


def _provenance(identity: str) -> CandidateProvenance:
    return CandidateProvenance(
        identity,
        StateValueAuthority.VERIFIED_SCIENTIFIC_SOURCE,
        "derived-charge-test",
    )


def _oxidation_result(facts, candidate):
    return OxidationStateInferenceResult(
        identity=f"result:{candidate.provenance.identity}",
        facts_identity=facts.identity,
        candidates=(candidate,),
        admissibility=ScientificAdmissibility.ADMISSIBLE,
        resolution=ResolutionStatus.UNIQUE,
        confidence=ChemicalConfidence.HIGH,
        findings=(),
    )


def _counting_result(facts, candidate):
    return CoordinationElectronCountingResult(
        identity=f"result:{candidate.provenance.identity}",
        facts_identity=facts.identity,
        candidates=(candidate,),
        admissibility=ScientificAdmissibility.ADMISSIBLE,
        resolution=ResolutionStatus.UNIQUE,
        confidence=ChemicalConfidence.HIGH,
        findings=(),
    )


def _unresolved_electronic_result(facts):
    return ElectronicStateInferenceResult(
        identity="electronic:unresolved",
        facts_identity=facts.identity,
        candidates=(),
        admissibility=ScientificAdmissibility.NOT_EVALUATED,
        resolution=ResolutionStatus.UNRESOLVED,
        confidence=ChemicalConfidence.UNRESOLVED,
        findings=(),
    )


def _eu_ionic_inputs():
    mol = hotpot.read_mol("[Eu].O", "smi")
    mol.add_bond(0, 1, bond_kind=BondKind.DATIVE)
    facts = extract_molecular_facts(mol)
    fragments = tuple(
        CoordinationFragmentAssignment(
            component.identity,
            -1 if facts.atomic_numbers[component.atom_indices[0]] == 8 else None,
            None,
            None,
        )
        for component in facts.covalent_components
    )
    counting = CoordinationCountingCandidate(
        _provenance("counting:ionic"),
        CountingConvention.IONIC,
        fragments,
    )
    oxidation_state = OxidationStateCandidateEvidence(
        provenance=_provenance("os:eu-iii"),
        convention=CountingConvention.IONIC,
        atom_states=(3, None),
        coverage=CoverageStatus.PARTIAL,
        coordination_counting_candidate_identity=(
            counting.provenance.identity
        ),
    )
    return facts, counting, oxidation_state


def test_complete_oxidation_assignment_derives_molecular_charge() -> None:
    facts = extract_molecular_facts(hotpot.read_mol("[Na+].[Cl-]", "smi"))
    oxidation_state = OxidationStateCandidateEvidence(
        provenance=_provenance("os:complete-zero"),
        convention=CountingConvention.IONIC,
        atom_states=(1, -1),
        coverage=CoverageStatus.COMPLETE,
    )

    space = compose_scientific_state_space(
        facts,
        net_charge_results=(),
        oxidation_state_results=(
            _oxidation_result(facts, oxidation_state),
        ),
        electronic_state_results=(
            _unresolved_electronic_result(facts),
        ),
    )

    derived = space.expanded_paths[0].derived_values[0]
    assert isinstance(derived, DerivedStateValue)
    assert derived.value == 0
    assert derived.parent_candidate_ids == ("os:complete-zero",)
    assert space.expanded_paths[0].net_charge.origin is (
        StateValueOrigin.DERIVED_CONSTRAINT
    )
    assert space.expanded_paths[0].derived_values == (derived,)


def test_partial_ionic_accounting_derives_intact_system_charge() -> None:
    facts, counting, oxidation_state = _eu_ionic_inputs()

    space = compose_scientific_state_space(
        facts,
        net_charge_results=(),
        coordination_counting_results=(
            _counting_result(facts, counting),
        ),
        oxidation_state_results=(
            _oxidation_result(facts, oxidation_state),
        ),
        electronic_state_results=(
            _unresolved_electronic_result(facts),
        ),
    )

    derived = space.expanded_paths[0].derived_values[0]
    assert isinstance(derived, DerivedStateValue)
    assert derived.value == 2
    assert set(derived.parent_candidate_ids) == {
        "counting:ionic",
        "os:eu-iii",
    }
    assert space.expanded_paths[0].net_charge.value == 2


def test_explicit_charge_retains_compatible_derived_lineage() -> None:
    facts, counting, oxidation_state = _eu_ionic_inputs()

    space = compose_scientific_state_space(
        facts,
        net_charge_results=resolve_net_charge(
            facts,
            StateRequest(net_charge=2),
        ),
        coordination_counting_results=(
            _counting_result(facts, counting),
        ),
        oxidation_state_results=(
            _oxidation_result(facts, oxidation_state),
        ),
        electronic_state_results=(
            _unresolved_electronic_result(facts),
        ),
    )

    path = space.expanded_paths[0]
    assert path.net_charge.origin is StateValueOrigin.RESULT
    assert path.net_charge.value == 2
    assert path.derived_values[0].value == 2


def test_explicit_charge_conflicting_with_derived_charge_is_rejected() -> None:
    facts, counting, oxidation_state = _eu_ionic_inputs()

    space = compose_scientific_state_space(
        facts,
        net_charge_results=resolve_net_charge(
            facts,
            StateRequest(net_charge=0),
        ),
        coordination_counting_results=(
            _counting_result(facts, counting),
        ),
        oxidation_state_results=(
            _oxidation_result(facts, oxidation_state),
        ),
        electronic_state_results=(
            _unresolved_electronic_result(facts),
        ),
    )

    assert space.expanded_paths == ()
    assert space.rejected_path_summaries[0].reason_codes == (
        "HARD-002.DERIVED_CHARGE_CONFLICT",
    )


def test_complete_oxidation_state_does_not_ignore_implicit_hydrogen() -> None:
    facts = extract_molecular_facts(hotpot.read_mol("O", "smi"))
    oxidation_state = OxidationStateCandidateEvidence(
        provenance=_provenance("os:water-oxygen-only"),
        convention=CountingConvention.IONIC,
        atom_states=(-2,),
        coverage=CoverageStatus.COMPLETE,
    )

    space = compose_scientific_state_space(
        facts,
        net_charge_results=(),
        oxidation_state_results=(
            _oxidation_result(facts, oxidation_state),
        ),
        electronic_state_results=(
            _unresolved_electronic_result(facts),
        ),
    )

    assert space.expanded_paths == ()
    assert all(
        factor.factor_key.quantity is not ScientificQuantity.NET_CHARGE
        for factor in space.factors
    )


def test_incomplete_ionic_accounting_does_not_invent_charge() -> None:
    mol = hotpot.read_mol("[Eu].O", "smi")
    mol.add_bond(0, 1, bond_kind=BondKind.DATIVE)
    facts = extract_molecular_facts(mol)
    counting = CoordinationCountingCandidate(
        _provenance("counting:incomplete"),
        CountingConvention.IONIC,
        (),
    )
    oxidation_state = OxidationStateCandidateEvidence(
        provenance=_provenance("os:eu-iii-incomplete"),
        convention=CountingConvention.IONIC,
        atom_states=(3, None),
        coverage=CoverageStatus.PARTIAL,
        coordination_counting_candidate_identity=(
            counting.provenance.identity
        ),
    )

    space = compose_scientific_state_space(
        facts,
        net_charge_results=(),
        coordination_counting_results=(
            _counting_result(facts, counting),
        ),
        oxidation_state_results=(
            _oxidation_result(facts, oxidation_state),
        ),
        electronic_state_results=(
            _unresolved_electronic_result(facts),
        ),
    )

    assert all(
        factor.factor_key.quantity is not ScientificQuantity.NET_CHARGE
        for factor in space.factors
    )
    assert space.expanded_paths == ()
