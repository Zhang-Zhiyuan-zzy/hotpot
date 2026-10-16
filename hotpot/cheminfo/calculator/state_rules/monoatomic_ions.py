"""OPTIONAL-002 common disconnected monoatomic-ion rule."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Dict, Tuple

from ..contracts import CandidateProvenance, ScientificQuantity, StateValueAuthority
from ..oxidation_state.contracts import (
    CoordinationCountingCandidate,
    CoordinationFragmentAssignment,
    CountingConvention,
)
from ..state_composition.contracts import (
    BranchAssignment,
    CandidateDomainScope,
    CandidateProposal,
    FactorKey,
    MaterializationStatus,
    ProposedCandidateFactor,
    ProposedCandidateValue,
)
from .base import InferenceRule
from .contracts import (
    EvidenceItem,
    ResultType,
    RuleApplicability,
    RuleContext,
    RuleKind,
    RuleOutcome,
    RulePhase,
    RuleSpecification,
    RuleValue,
)

__all__ = ["MonoatomicIonRule"]


_RULE_ID = "OPTIONAL-002"
_RULE_VERSION = "1.0.0"
_REGISTRY_SCHEMA = "hotpot-monoatomic-ion-registry-v1"
_REGISTRY_RESOURCE = "data/rules/monoatomic_ions_v1.json"
_DOMAIN_DEFINITION = "common-disconnected-monoatomic-ion-v1"


def _digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class _IonEntry:
    atomic_number: int
    charges: Tuple[int, ...]
    references: Tuple[str, ...]


@dataclass(frozen=True)
class _IonRegistry:
    version: str
    source_identity: str
    content_digest: str
    entries: Tuple[_IonEntry, ...]


@lru_cache(maxsize=1)
def _ion_registry() -> _IonRegistry:
    resource = resources.files(__package__).joinpath(_REGISTRY_RESOURCE)
    encoded = resource.read_bytes()
    payload = json.loads(encoded.decode("utf-8"))
    expected = {
        "schema_version",
        "registry_version",
        "source_identity",
        "entries",
    }
    if set(payload) != expected or payload["schema_version"] != _REGISTRY_SCHEMA:
        raise ValueError("Monoatomic-ion registry does not match its schema")
    entries = tuple(
        _IonEntry(
            atomic_number=item["atomic_number"],
            charges=tuple(item["charges"]),
            references=tuple(item["references"]),
        )
        for item in payload["entries"]
    )
    if (
        any(
            type(entry.atomic_number) is not int
            or entry.atomic_number <= 0
            or not entry.charges
            or any(type(charge) is not int for charge in entry.charges)
            or not entry.references
            for entry in entries
        )
        or len({entry.atomic_number for entry in entries}) != len(entries)
    ):
        raise ValueError("Monoatomic-ion registry contains invalid entries")
    return _IonRegistry(
        version=payload["registry_version"],
        source_identity=payload["source_identity"],
        content_digest=hashlib.sha256(encoded).hexdigest(),
        entries=entries,
    )


def _matching_components(
    context: RuleContext,
) -> Tuple[Tuple[str, int, _IonEntry], ...]:
    entries = {entry.atomic_number: entry for entry in _ion_registry().entries}
    covalent_by_atoms = {
        component.atom_indices: component.identity
        for component in context.facts.covalent_components
    }
    return tuple(
        (
            covalent_by_atoms[component.atom_indices],
            component.atom_indices[0],
            entries[context.facts.atomic_numbers[component.atom_indices[0]]],
        )
        for component in context.facts.source_components
        if len(component.atom_indices) == 1
        and component.atom_indices in covalent_by_atoms
        and context.facts.atomic_numbers[component.atom_indices[0]] in entries
    )


class MonoatomicIonRule(InferenceRule):
    """Propose bounded ionic charges for explicit one-atom source components."""

    specification = RuleSpecification(
        rule_id=_RULE_ID,
        rule_version=_RULE_VERSION,
        kind=RuleKind.EMPIRICAL,
        phase=RulePhase.CHARGE_EVIDENCE,
        requires=(),
        consumes=(),
        produces=(ResultType.CANDIDATE_PROPOSAL, ResultType.EVIDENCE),
        applicability_schema=_DOMAIN_DEFINITION,
    )

    def applicable(self, context: RuleContext) -> RuleApplicability:
        return RuleApplicability(
            bool(_matching_components(context)),
            (
                ()
                if _matching_components(context)
                else ("NO_SUPPORTED_MONOATOMIC_SOURCE_COMPONENT",)
            ),
        )

    def action(
        self,
        context: RuleContext,
        configuration: Tuple[Tuple[str, RuleValue], ...],
    ) -> RuleOutcome:
        del configuration
        registry = _ion_registry()
        components = _matching_components(context)
        convention_branch = BranchAssignment(
            f"counting-convention:{context.facts.identity.chemical_identity_digest}",
            CountingConvention.IONIC.value,
        )
        factors = []
        evidence_by_atomic_number: Dict[int, EvidenceItem] = {}
        for component_identity, atom_index, entry in components:
            values = []
            for charge in entry.charges:
                identity = _digest(
                    {
                        "rule": _RULE_ID,
                        "version": _RULE_VERSION,
                        "registry": registry.content_digest,
                        "facts": context.facts.identity.chemical_identity_digest,
                        "component": component_identity,
                        "charge": charge,
                    }
                )
                candidate = CoordinationCountingCandidate(
                    provenance=CandidateProvenance(
                        identity=identity,
                        authority=StateValueAuthority.DOMAIN_RULE,
                        source_identity=registry.source_identity,
                        evidence_ids=(registry.content_digest,),
                        assumption_ids=("domain_template:optional-002",),
                    ),
                    convention=CountingConvention.IONIC,
                    fragments=(
                        CoordinationFragmentAssignment(
                            component_identity=component_identity,
                            ionic_bookkeeping_charge=charge,
                            lxz_class=None,
                            donated_electrons=None,
                        ),
                    ),
                )
                values.append(
                    ProposedCandidateValue(candidate, (convention_branch,))
                )
            factors.append(
                ProposedCandidateFactor(
                    factor_key=FactorKey(
                        ScientificQuantity.COORDINATION_COUNTING,
                        component_identity,
                    ),
                    domain_definition_id=(
                        f"{_DOMAIN_DEFINITION}:{registry.content_digest}"
                    ),
                    materialized_candidates=tuple(values),
                    domain_scope=CandidateDomainScope.DOMAIN_BOUNDED,
                    materialization_status=MaterializationStatus.COMPLETE,
                    unexpanded_domain_summary=None,
                )
            )
            evidence_by_atomic_number.setdefault(
                entry.atomic_number,
                EvidenceItem(
                    identity=_digest(
                        {
                            "registry": registry.content_digest,
                            "atomic_number": entry.atomic_number,
                        }
                    ),
                    authority=StateValueAuthority.VERIFIED_SCIENTIFIC_SOURCE,
                    source_identity=registry.source_identity,
                    claim=(
                        f"Common monoatomic ion domain Z={entry.atomic_number}: "
                        f"{entry.charges}"
                    ),
                    locator="; ".join(entry.references),
                ),
            )
        proposal = CandidateProposal(
            proposal_id=_digest(
                {
                    "rule": _RULE_ID,
                    "version": _RULE_VERSION,
                    "facts": context.facts.identity.chemical_identity_digest,
                    "components": tuple(
                        component_identity
                        for component_identity, _, _ in components
                    ),
                }
            ),
            factors=tuple(factors),
            constraints=(),
        )
        return RuleOutcome(
            rule_id=_RULE_ID,
            rule_version=_RULE_VERSION,
            applicability=RuleApplicability(True),
            candidate_proposal=proposal,
            evidence=tuple(
                evidence_by_atomic_number[atomic_number]
                for atomic_number in sorted(evidence_by_atomic_number)
            ),
            rank_adjustments=(),
            assumptions=("domain_template:optional-002",),
            warnings=(),
        )
