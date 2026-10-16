"""Internal hard scientific-state validators.

This module reports mathematical and conservation facts. It does not infer,
rank, repair, or select scientific states.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from .charge.contracts import (
    AtomFormalChargeEvidence,
    ComponentChargeEvidence,
    NetChargeCandidateEvidence,
)
from .contracts import (
    CoverageStatus,
    MolecularFacts,
    MolecularFactsIdentity,
    ResolutionStatus,
    ScientificAdmissibility,
    ScientificFinding,
)
from .electronic_state.contracts import ElectronicStateCandidate
from .oxidation_state.contracts import (
    CoordinationCountingCandidate,
    CountingConvention,
    OxidationStateCandidateEvidence,
    OxidationStateOverride,
)

__all__ = []


# Shared internal assessment structure.


@dataclass(frozen=True)
class _HardRuleAssessment:
    admissibility: ScientificAdmissibility
    terminal_status: Optional[ResolutionStatus]
    findings: Tuple[ScientificFinding, ...]


_HARD_PASS = _HardRuleAssessment(
    admissibility=ScientificAdmissibility.ADMISSIBLE,
    terminal_status=None,
    findings=(),
)


def _hard_failure(
    code: str,
    message: str,
    *,
    conflict: bool = False,
    related_identities: Tuple[str, ...] = (),
) -> _HardRuleAssessment:
    return _HardRuleAssessment(
        admissibility=ScientificAdmissibility.INADMISSIBLE,
        terminal_status=ResolutionStatus.CONFLICT if conflict else None,
        findings=(
            ScientificFinding(
                code=code,
                message=message,
                related_identities=related_identities,
            ),
        ),
    )


def _hard_not_evaluated(
    code: str,
    message: str,
    *,
    related_identities: Tuple[str, ...] = (),
) -> _HardRuleAssessment:
    return _HardRuleAssessment(
        admissibility=ScientificAdmissibility.NOT_EVALUATED,
        terminal_status=None,
        findings=(
            ScientificFinding(
                code=code,
                message=message,
                related_identities=related_identities,
            ),
        ),
    )


def _merge_hard_rule_assessments(
    assessments: Tuple[_HardRuleAssessment, ...],
) -> _HardRuleAssessment:
    findings = tuple(
        finding
        for assessment in assessments
        for finding in assessment.findings
    )
    if any(
        assessment.terminal_status is ResolutionStatus.CONFLICT
        for assessment in assessments
    ):
        return _HardRuleAssessment(
            ScientificAdmissibility.INADMISSIBLE,
            ResolutionStatus.CONFLICT,
            findings,
        )
    if any(
        assessment.admissibility is ScientificAdmissibility.INADMISSIBLE
        for assessment in assessments
    ):
        return _HardRuleAssessment(
            ScientificAdmissibility.INADMISSIBLE,
            None,
            findings,
        )
    if any(
        assessment.admissibility is ScientificAdmissibility.NOT_EVALUATED
        for assessment in assessments
    ):
        return _HardRuleAssessment(
            ScientificAdmissibility.NOT_EVALUATED,
            None,
            findings,
        )
    return _HardRuleAssessment(
        ScientificAdmissibility.ADMISSIBLE,
        None,
        findings,
    )


# HARD-001: identity binding.


def _validate_facts_identity(
    expected: MolecularFactsIdentity,
    observed: MolecularFactsIdentity,
    *,
    related_identities: Tuple[str, ...] = (),
) -> _HardRuleAssessment:
    if expected == observed:
        return _HARD_PASS
    return _hard_failure(
        "HARD-001.FACTS_IDENTITY_MISMATCH",
        "Scientific evidence is bound to a different molecular-facts identity.",
        related_identities=related_identities,
    )


# HARD-002: net, atom and component charge conservation.


def _validate_charge_conservation(
    facts: MolecularFacts,
    net_charge: NetChargeCandidateEvidence,
    *,
    atom_evidence: Optional[AtomFormalChargeEvidence] = None,
    component_evidence: Optional[ComponentChargeEvidence] = None,
) -> _HardRuleAssessment:
    assessments = []
    if type(net_charge.charge) is not int:
        assessments.append(
            _hard_failure(
                "HARD-002.NON_INTEGER_NET_CHARGE",
                "Molecular net charge must be an integer.",
                related_identities=(net_charge.provenance.identity,),
            )
        )

    if atom_evidence is not None:
        atom_charges = atom_evidence.charges
        known_atom_charges_are_integers = all(
            charge is None or type(charge) is int for charge in atom_charges
        )
        complete_atom_coverage = (
            len(atom_charges) == len(facts.atomic_numbers)
            and all(charge is not None for charge in atom_charges)
        )
        if not known_atom_charges_are_integers or (
            atom_evidence.coverage is CoverageStatus.COMPLETE
            and not complete_atom_coverage
        ):
            assessments.append(
                _hard_failure(
                    "HARD-002.ATOM_COVERAGE_INVALID",
                    "Atom formal-charge evidence has invalid values or coverage.",
                    related_identities=(atom_evidence.provenance.identity,),
                )
            )
        elif (
            type(net_charge.charge) is int
            and atom_evidence.coverage is CoverageStatus.COMPLETE
            and sum(charge for charge in atom_charges if charge is not None)
            != net_charge.charge
        ):
            assessments.append(
                _hard_failure(
                    "HARD-002.ATOM_CHARGE_SUM_MISMATCH",
                    "Complete atom formal charges do not sum to molecular net charge.",
                    conflict=True,
                    related_identities=(
                        net_charge.provenance.identity,
                        atom_evidence.provenance.identity,
                    ),
                )
            )

    if component_evidence is not None:
        component_charges = component_evidence.component_charges
        supplied_ids = tuple(identity for identity, _ in component_charges)
        expected_ids = tuple(
            component.identity for component in facts.source_components
        )
        values_are_integers = all(
            type(charge) is int for _, charge in component_charges
        )
        complete_component_coverage = (
            len(supplied_ids) == len(set(supplied_ids))
            and set(supplied_ids) == set(expected_ids)
        )
        if not values_are_integers or (
            component_evidence.coverage is CoverageStatus.COMPLETE
            and not complete_component_coverage
        ):
            assessments.append(
                _hard_failure(
                    "HARD-002.COMPONENT_COVERAGE_INVALID",
                    "Component charge evidence has invalid values or coverage.",
                    related_identities=(component_evidence.provenance.identity,),
                )
            )
        elif (
            type(net_charge.charge) is int
            and component_evidence.coverage is CoverageStatus.COMPLETE
            and sum(charge for _, charge in component_charges) != net_charge.charge
        ):
            assessments.append(
                _hard_failure(
                    "HARD-002.COMPONENT_CHARGE_SUM_MISMATCH",
                    "Complete component charges do not sum to molecular net charge.",
                    conflict=True,
                    related_identities=(
                        net_charge.provenance.identity,
                        component_evidence.provenance.identity,
                    ),
                )
            )

    if not assessments:
        return _HARD_PASS
    return _merge_hard_rule_assessments(tuple(assessments))


# HARD-005: coordination-counting convention exclusivity.


def _validate_counting_candidate(
    candidate: CoordinationCountingCandidate,
) -> _HardRuleAssessment:
    mixed_components = []
    for fragment in candidate.fragments:
        mixed = (
            candidate.convention is CountingConvention.IONIC
            and fragment.lxz_class is not None
        ) or (
            candidate.convention is CountingConvention.DONOR_RETURN
            and (
                fragment.ionic_bookkeeping_charge is not None
                or fragment.lxz_class is not None
            )
        ) or (
            candidate.convention is CountingConvention.CBC
            and fragment.ionic_bookkeeping_charge is not None
        )
        if mixed:
            mixed_components.append(fragment.component_identity)

    if not mixed_components:
        return _HARD_PASS
    return _hard_failure(
        "HARD-005.MIXED_COUNTING_FIELDS",
        "One coordination-counting candidate mixes incompatible conventions.",
        related_identities=(
            candidate.provenance.identity,
            *tuple(mixed_components),
        ),
    )


def _validate_counting_reference(
    oxidation_candidate: OxidationStateCandidateEvidence,
    counting_candidates: Tuple[CoordinationCountingCandidate, ...],
) -> _HardRuleAssessment:
    reference = oxidation_candidate.coordination_counting_candidate_identity
    if reference is None:
        return _HARD_PASS

    matches = tuple(
        candidate
        for candidate in counting_candidates
        if candidate.provenance.identity == reference
    )
    if len(matches) != 1:
        return _hard_failure(
            "HARD-005.COUNTING_REFERENCE_MISSING",
            "Oxidation-state evidence does not reference exactly one counting candidate.",
            related_identities=(
                oxidation_candidate.provenance.identity,
                reference,
            ),
        )
    if matches[0].convention is not oxidation_candidate.convention:
        return _hard_failure(
            "HARD-005.COUNTING_CONVENTION_MISMATCH",
            "Oxidation-state and coordination-counting conventions differ.",
            related_identities=(
                oxidation_candidate.provenance.identity,
                reference,
            ),
        )
    return _HARD_PASS


# HARD-004: oxidation-state conservation and explicit overrides.


def _validate_oxidation_state_evidence(
    facts: MolecularFacts,
    candidate: OxidationStateCandidateEvidence,
) -> _HardRuleAssessment:
    atom_states = candidate.atom_states
    valid_values = all(
        oxidation_state is None or type(oxidation_state) is int
        for oxidation_state in atom_states
    )
    complete_coverage = (
        len(atom_states) == len(facts.atomic_numbers)
        and all(oxidation_state is not None for oxidation_state in atom_states)
    )
    if len(atom_states) != len(facts.atomic_numbers) or not valid_values or (
        candidate.coverage is CoverageStatus.COMPLETE and not complete_coverage
    ):
        return _hard_failure(
            "HARD-004.OS_COVERAGE_INVALID",
            "Oxidation-state evidence has invalid values or atom coverage.",
            related_identities=(candidate.provenance.identity,),
        )
    return _HARD_PASS


def _validate_oxidation_state_conservation(
    facts: MolecularFacts,
    net_charge: NetChargeCandidateEvidence,
    candidate: OxidationStateCandidateEvidence,
) -> _HardRuleAssessment:
    structure_assessment = _validate_oxidation_state_evidence(facts, candidate)
    if structure_assessment.admissibility is not ScientificAdmissibility.ADMISSIBLE:
        return structure_assessment
    atom_states = candidate.atom_states
    if (
        type(net_charge.charge) is int
        and candidate.coverage is CoverageStatus.COMPLETE
        and sum(
            oxidation_state
            for oxidation_state in atom_states
            if oxidation_state is not None
        )
        != net_charge.charge
    ):
        return _hard_failure(
            "HARD-004.OS_SUM_MISMATCH",
            "Complete atom oxidation states do not sum to molecular net charge.",
            conflict=True,
            related_identities=(
                net_charge.provenance.identity,
                candidate.provenance.identity,
            ),
        )
    return _HARD_PASS


def _validate_oxidation_overrides(
    facts: MolecularFacts,
    overrides: Tuple[OxidationStateOverride, ...],
    complete_candidates: Tuple[OxidationStateCandidateEvidence, ...],
) -> _HardRuleAssessment:
    validity = _validate_oxidation_override_values(facts, overrides)
    if validity.admissibility is not ScientificAdmissibility.ADMISSIBLE:
        return validity

    override_values = {
        override.atom_index: override.oxidation_state for override in overrides
    }
    if not overrides:
        return _HARD_PASS

    eligible_candidates = tuple(
        candidate
        for candidate in complete_candidates
        if candidate.coverage is CoverageStatus.COMPLETE
        and len(candidate.atom_states) == len(facts.atomic_numbers)
        and all(state is not None for state in candidate.atom_states)
    )
    if not eligible_candidates:
        return _hard_not_evaluated(
            "HARD-004.OS_OVERRIDE_EXTENSION_NOT_EVALUATED",
            "No complete oxidation-state candidate is available to test overrides.",
        )

    if any(
        all(
            candidate.atom_states[atom_index] == oxidation_state
            for atom_index, oxidation_state in override_values.items()
        )
        for candidate in eligible_candidates
    ):
        return _HARD_PASS
    return _hard_failure(
        "HARD-007.OS_OVERRIDE_EMPTY_INTERSECTION",
        "No complete oxidation-state candidate extends all explicit overrides.",
        conflict=True,
        related_identities=tuple(
            candidate.provenance.identity for candidate in eligible_candidates
        ),
    )


def _validate_oxidation_override_values(
    facts: MolecularFacts,
    overrides: Tuple[OxidationStateOverride, ...],
) -> _HardRuleAssessment:
    override_values = {}
    for override in overrides:
        if (
            type(override.atom_index) is not int
            or not 0 <= override.atom_index < len(facts.atomic_numbers)
            or type(override.oxidation_state) is not int
        ):
            return _hard_failure(
                "HARD-004.OS_OVERRIDE_INVALID",
                "Oxidation-state override has an invalid atom index or value.",
            )
        incumbent = override_values.get(override.atom_index)
        if incumbent is not None and incumbent != override.oxidation_state:
            return _hard_failure(
                "HARD-007.OS_OVERRIDE_CONFLICT",
                "Explicit oxidation-state overrides conflict on one atom.",
                conflict=True,
            )
        override_values[override.atom_index] = override.oxidation_state
    return _HARD_PASS


# HARD-006: backend-independent molecular electron and spin legality.


def _validate_molecular_spin(
    facts: MolecularFacts,
    net_charge: NetChargeCandidateEvidence,
    *,
    electron_count: int,
    two_s: int,
    multiplicity: Optional[int],
    pure_spin: bool,
    related_identities: Tuple[str, ...] = (),
) -> _HardRuleAssessment:
    if type(net_charge.charge) is not int or type(electron_count) is not int:
        return _hard_failure(
            "HARD-006.NON_INTEGER_ELECTRON_COUNT",
            "Charge and molecular electron count must be integers.",
            related_identities=related_identities,
        )

    expected_electron_count = facts.nuclear_charge_sum - net_charge.charge
    if expected_electron_count < 0:
        return _hard_failure(
            "HARD-006.NEGATIVE_ELECTRON_COUNT",
            "Molecular charge leaves a negative all-electron count.",
            related_identities=related_identities,
        )
    if electron_count != expected_electron_count:
        return _hard_failure(
            "HARD-006.ELECTRON_COUNT_MISMATCH",
            "Reported electron count does not equal nuclear charge minus net charge.",
            related_identities=related_identities,
        )
    if type(two_s) is not int or not 0 <= two_s <= electron_count:
        return _hard_failure(
            "HARD-006.TWO_S_OUT_OF_RANGE",
            "Molecular two_s must be an integer between zero and electron count.",
            related_identities=related_identities,
        )
    if (electron_count - two_s) % 2:
        return _hard_failure(
            "HARD-006.PARITY_MISMATCH",
            "Molecular two_s has parity inconsistent with electron count.",
            related_identities=related_identities,
        )
    if multiplicity is not None and type(multiplicity) is not int:
        return _hard_failure(
            "HARD-006.MULTIPLICITY_INVALID",
            "Molecular multiplicity must be an integer when supplied.",
            related_identities=related_identities,
        )
    if pure_spin and multiplicity is not None and multiplicity != two_s + 1:
        return _hard_failure(
            "HARD-006.MULTIPLICITY_MISMATCH",
            "Pure-spin multiplicity must equal two_s plus one.",
            related_identities=related_identities,
        )
    return _HARD_PASS


def _validate_electronic_state_candidate(
    facts: MolecularFacts,
    net_charge: NetChargeCandidateEvidence,
    candidate: ElectronicStateCandidate,
) -> _HardRuleAssessment:
    related_identities = (
        net_charge.provenance.identity,
        candidate.provenance.identity,
    )
    if candidate.net_charge_candidate_identity != net_charge.provenance.identity:
        return _hard_failure(
            "HARD-007.NET_CHARGE_REFERENCE_MISMATCH",
            "Electronic-state candidate references a different net-charge candidate.",
            related_identities=related_identities,
        )
    return _validate_molecular_spin(
        facts,
        net_charge,
        electron_count=candidate.electron_count,
        two_s=candidate.two_s,
        multiplicity=candidate.multiplicity,
        pure_spin=candidate.pure_spin,
        related_identities=related_identities,
    )


def _validate_electronic_oxidation_reference(
    candidate: ElectronicStateCandidate,
    oxidation_candidates: Tuple[OxidationStateCandidateEvidence, ...],
) -> _HardRuleAssessment:
    reference = candidate.oxidation_state_candidate_identity
    if reference is None:
        return _HARD_PASS
    matches = tuple(
        oxidation_candidate
        for oxidation_candidate in oxidation_candidates
        if oxidation_candidate.provenance.identity == reference
    )
    if len(matches) == 1:
        return _HARD_PASS
    return _hard_failure(
        "HARD-007.OXIDATION_STATE_REFERENCE_MISSING",
        "Electronic-state evidence references an unavailable oxidation-state candidate.",
        related_identities=(candidate.provenance.identity, reference),
    )
