"""Internal hard scientific-state validators.

This module reports mathematical and conservation facts. It does not infer,
rank, repair, or select scientific states.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from .charge import (
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
