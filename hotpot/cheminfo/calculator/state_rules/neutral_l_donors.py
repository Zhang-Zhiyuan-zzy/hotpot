"""OPTIONAL-003 neutral L-type donor rule."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Dict, Optional, Tuple

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

__all__ = ["NeutralLDonorRule"]


_RULE_ID = "OPTIONAL-003"
_RULE_VERSION = "1.0.0"
_REGISTRY_SCHEMA = "hotpot-neutral-l-donor-registry-v1"
_REGISTRY_RESOURCE = "data/rules/neutral_l_donors_v1.json"
_DOMAIN_DEFINITION = "fixed-protonation-neutral-l-donor-v1"
_DEFAULT_CONVENTIONS = (CountingConvention.IONIC,)


def _digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class _AtomSignature:
    atomic_number: int
    covalent_valence: float
    label: str


@dataclass(frozen=True)
class _ComponentSignature:
    atomic_numbers: Tuple[int, ...]
    bond_orders: Tuple[float, ...]
    donor_atomic_number: int
    label: str


@dataclass(frozen=True)
class _DonorRegistry:
    version: str
    source_identity: str
    reference: str
    content_digest: str
    atom_signatures: Tuple[_AtomSignature, ...]
    component_signatures: Tuple[_ComponentSignature, ...]


@lru_cache(maxsize=1)
def _donor_registry() -> _DonorRegistry:
    resource = resources.files(__package__).joinpath(_REGISTRY_RESOURCE)
    encoded = resource.read_bytes()
    payload = json.loads(encoded.decode("utf-8"))
    expected = {
        "schema_version",
        "registry_version",
        "source_identity",
        "reference",
        "atom_signatures",
        "component_signatures",
    }
    if set(payload) != expected or payload["schema_version"] != _REGISTRY_SCHEMA:
        raise ValueError("Neutral L-donor registry does not match its schema")
    atom_signatures = tuple(
        _AtomSignature(
            item["atomic_number"],
            float(item["covalent_valence"]),
            item["label"],
        )
        for item in payload["atom_signatures"]
    )
    component_signatures = tuple(
        _ComponentSignature(
            tuple(item["atomic_numbers"]),
            tuple(float(value) for value in item["bond_orders"]),
            item["donor_atomic_number"],
            item["label"],
        )
        for item in payload["component_signatures"]
    )
    return _DonorRegistry(
        version=payload["registry_version"],
        source_identity=payload["source_identity"],
        reference=payload["reference"],
        content_digest=hashlib.sha256(encoded).hexdigest(),
        atom_signatures=atom_signatures,
        component_signatures=component_signatures,
    )


def _configured_conventions(
    configuration: Tuple[Tuple[str, RuleValue], ...],
) -> Tuple[CountingConvention, ...]:
    values = dict(configuration)
    unknown = set(values) - {"conventions"}
    if unknown:
        raise ValueError(f"Unsupported OPTIONAL-003 configuration: {unknown}")
    configured = values.get("conventions")
    if configured is None:
        return _DEFAULT_CONVENTIONS
    if not isinstance(configured, tuple):
        raise ValueError("OPTIONAL-003 conventions must be a tuple")
    conventions = tuple(CountingConvention(value) for value in configured)
    if not conventions or len(set(conventions)) != len(conventions):
        raise ValueError("OPTIONAL-003 conventions must be unique and non-empty")
    return conventions


def _covalent_valences(context: RuleContext) -> Tuple[Optional[float], ...]:
    values = [0.0] * len(context.facts.atomic_numbers)
    valid = [True] * len(context.facts.atomic_numbers)
    for bond in context.facts.semantic_bonds:
        if bond.kind == "coordination":
            continue
        begin, end = bond.atom_indices
        if bond.kind != "covalent" or bond.order is None:
            valid[begin] = False
            valid[end] = False
            continue
        values[begin] += bond.order
        values[end] += bond.order
    for index, implicit_count in enumerate(
        context.facts.hydrogen_inventory.implicit_counts
    ):
        if implicit_count is not None:
            values[index] += implicit_count
    return tuple(
        value if is_valid else None
        for value, is_valid in zip(values, valid)
    )


def _coordinated_atoms(context: RuleContext) -> frozenset[int]:
    supported = {
        signature.atomic_number
        for signature in _donor_registry().atom_signatures
    } | {
        signature.donor_atomic_number
        for signature in _donor_registry().component_signatures
    }
    donors = set()
    for bond in context.facts.semantic_bonds:
        if bond.kind != "coordination":
            continue
        begin, end = bond.atom_indices
        begin_supported = context.facts.atomic_numbers[begin] in supported
        end_supported = context.facts.atomic_numbers[end] in supported
        if begin_supported != end_supported:
            donors.add(begin if begin_supported else end)
    return frozenset(donors)


def _component_signature_matches(
    context: RuleContext,
    component_atoms: Tuple[int, ...],
    donor_index: int,
) -> bool:
    registry = _donor_registry()
    atomic_numbers = tuple(
        sorted(context.facts.atomic_numbers[index] for index in component_atoms)
    )
    bond_orders = tuple(
        sorted(
            bond.order
            for bond in context.facts.semantic_bonds
            if bond.kind == "covalent"
            and bond.order is not None
            and set(bond.atom_indices).issubset(component_atoms)
        )
    )
    return any(
        tuple(sorted(signature.atomic_numbers)) == atomic_numbers
        and tuple(sorted(signature.bond_orders)) == bond_orders
        and context.facts.atomic_numbers[donor_index]
        == signature.donor_atomic_number
        for signature in registry.component_signatures
    )


def _atom_signature_matches(
    context: RuleContext,
    donor_index: int,
    valences: Tuple[Optional[float], ...],
) -> bool:
    valence = valences[donor_index]
    return valence is not None and any(
        context.facts.atomic_numbers[donor_index] == signature.atomic_number
        and math.isclose(valence, signature.covalent_valence, abs_tol=1.0e-8)
        for signature in _donor_registry().atom_signatures
    )


def _matching_components(
    context: RuleContext,
) -> Tuple[Tuple[str, Tuple[int, ...]], ...]:
    donor_atoms = _coordinated_atoms(context)
    valences = _covalent_valences(context)
    matches = []
    for component in context.facts.covalent_components:
        component_donors = tuple(
            index for index in component.atom_indices if index in donor_atoms
        )
        if not component_donors:
            continue
        if all(
            _atom_signature_matches(context, index, valences)
            or _component_signature_matches(
                context,
                component.atom_indices,
                index,
            )
            for index in component_donors
        ):
            matches.append((component.identity, component_donors))
    return tuple(matches)


def _fragment_assignment(
    component_identity: str,
    donor_count: int,
    convention: CountingConvention,
) -> CoordinationFragmentAssignment:
    if convention is CountingConvention.IONIC:
        return CoordinationFragmentAssignment(component_identity, 0, None, None)
    if convention is CountingConvention.DONOR_RETURN:
        return CoordinationFragmentAssignment(
            component_identity,
            None,
            None,
            2 * donor_count,
        )
    lxz_class = "L" if donor_count == 1 else f"L{donor_count}"
    return CoordinationFragmentAssignment(
        component_identity,
        None,
        lxz_class,
        2 * donor_count,
    )


class NeutralLDonorRule(InferenceRule):
    """Propose convention-specific accounting for supported neutral L donors."""

    specification = RuleSpecification(
        rule_id=_RULE_ID,
        rule_version=_RULE_VERSION,
        kind=RuleKind.EMPIRICAL,
        phase=RulePhase.COORDINATION_COUNTING,
        requires=(),
        consumes=(),
        produces=(ResultType.CANDIDATE_PROPOSAL, ResultType.EVIDENCE),
        applicability_schema=_DOMAIN_DEFINITION,
    )

    def applicable(self, context: RuleContext) -> RuleApplicability:
        matches = _matching_components(context)
        return RuleApplicability(
            bool(matches),
            () if matches else ("NO_SUPPORTED_NEUTRAL_L_DONOR",),
        )

    def action(
        self,
        context: RuleContext,
        configuration: Tuple[Tuple[str, RuleValue], ...],
    ) -> RuleOutcome:
        registry = _donor_registry()
        conventions = _configured_conventions(configuration)
        matches = _matching_components(context)
        branch_group = (
            f"counting-convention:{context.facts.identity.chemical_identity_digest}"
        )
        factors = []
        for component_identity, donor_atoms in matches:
            values = []
            for convention in conventions:
                identity = _digest(
                    {
                        "rule": _RULE_ID,
                        "version": _RULE_VERSION,
                        "registry": registry.content_digest,
                        "facts": context.facts.identity.chemical_identity_digest,
                        "component": component_identity,
                        "donors": donor_atoms,
                        "convention": convention.value,
                    }
                )
                candidate = CoordinationCountingCandidate(
                    provenance=CandidateProvenance(
                        identity=identity,
                        authority=StateValueAuthority.DOMAIN_RULE,
                        source_identity=registry.source_identity,
                        evidence_ids=(registry.content_digest,),
                        assumption_ids=("domain_template:optional-003",),
                    ),
                    convention=convention,
                    fragments=(
                        _fragment_assignment(
                            component_identity,
                            len(donor_atoms),
                            convention,
                        ),
                    ),
                )
                values.append(
                    ProposedCandidateValue(
                        candidate,
                        (BranchAssignment(branch_group, convention.value),),
                    )
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
        proposal = CandidateProposal(
            proposal_id=_digest(
                {
                    "rule": _RULE_ID,
                    "version": _RULE_VERSION,
                    "facts": context.facts.identity.chemical_identity_digest,
                    "components": tuple(identity for identity, _ in matches),
                    "conventions": tuple(
                        convention.value for convention in conventions
                    ),
                }
            ),
            factors=tuple(factors),
            constraints=(),
        )
        evidence = EvidenceItem(
            identity=registry.content_digest,
            authority=StateValueAuthority.VERIFIED_SCIENTIFIC_SOURCE,
            source_identity=registry.source_identity,
            claim="Curated neutral L-type donor applicability domain",
            locator=registry.reference,
        )
        return RuleOutcome(
            rule_id=_RULE_ID,
            rule_version=_RULE_VERSION,
            applicability=RuleApplicability(True),
            candidate_proposal=proposal,
            evidence=(evidence,),
            rank_adjustments=(),
            assumptions=("domain_template:optional-003",),
            warnings=(),
        )
