"""Source component bookkeeping and attributable screening completions.

These rules produce existing factors. Charge summation, hard validation, local
spin coupling and execution selection remain with the common state resolver.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Optional, Tuple

from hotpot.cheminfo.rules.chemistry.bonding.hydrogen import _non_graph_hydrogens
from hotpot.cheminfo.rules.chemistry.elements import _is_metal
from hotpot.cheminfo.rules.chemistry.scientific_state.charge.contracts import (
    AtomFormalChargeEvidence, ComponentChargeEvidence,
)
from hotpot.cheminfo.rules.chemistry.scientific_state.contracts import (
    CandidateProvenance, CoverageStatus, ScientificQuantity, StateValueAuthority,
)
from hotpot.cheminfo.rules.chemistry.scientific_state.composition import (
    BranchAssignment, CandidateConstraint, CandidateDomainScope, CandidateDomainSummary, CandidateProposal,
    ConstraintGuard, FactorKey, GuardedCandidateConstraint, MaterializationStatus,
    ProposedCandidateFactor, ProposedCandidateValue,
)
from hotpot.cheminfo.rules.chemistry.scientific_state.electronic_state.contracts import (
    SpinContributionCandidate, SpinContributionScope,
)
from hotpot.cheminfo.rules.chemistry.scientific_state.oxidation_state.contracts import (
    CoordinationCountingCandidate, CoordinationFragmentAssignment,
    CountingConvention, OxidationStateCandidateEvidence,
)
from hotpot.cheminfo.rules.chemistry.scientific_state.oxidation_state.registry import (
    element_oxidation_state_domain,
)
from hotpot.cheminfo.rules.chemistry.scientific_state.providers.base import InferenceRule
from hotpot.cheminfo.rules.chemistry.scientific_state.providers.contracts import (
    EvidenceItem, RankAdjustment, ResultType, RuleApplicability, RuleContext,
    RuleKind, RuleOutcome, RulePhase, RuleSpecification, RuleValue,
)
from hotpot.cheminfo.rules.chemistry.scientific_state.providers.ionic_configuration_spin import (
    _configuration_registry,
)
from hotpot.cheminfo.rules.chemistry.scientific_state.providers.lanthanide_iii import (
    _artifact as _lanthanide_artifact,
)
from hotpot.cheminfo.rules.chemistry.scientific_state.providers.americium_curium_iii import (
    _artifact as _americium_curium_artifact,
)
from hotpot.cheminfo.rules.chemistry.scientific_state.providers.thorium_iv import (
    _artifact as _thorium_artifact,
)

__all__ = []

_VERSION = "1.0.0"
_SOURCE_RULE = "OPTIONAL-015"
_DEFAULT_RULE = "OPTIONAL-016"
_SOURCE_NAME = "molecule:supplied-component-charge-v1"
_DEFAULT_NAME = "coordination-screening:component-periodic-defaults-v1"
_CONSTRAINT_SCHEMA = "candidate-constraint-v1"


def _digest(payload: object) -> str:
    return hashlib.sha256(json.dumps(
        payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True,
        default=lambda value: value.value,
    ).encode("utf-8")).hexdigest()


def _automatic_source_digest(
    atom_charges: Optional[AtomFormalChargeEvidence],
    component_charges: Optional[ComponentChargeEvidence],
    radicals: Tuple[Optional[int], ...],
    source_identity: str = "",
) -> str:
    return _digest({
        "atom_charges": None if atom_charges is None else asdict(atom_charges),
        "component_charges": None if component_charges is None else asdict(component_charges),
        "radicals": radicals, "source_identity": source_identity,
    })


def _ligand_components(context: RuleContext):
    coordinated_centers = {
        index for bond in context.facts.semantic_bonds if bond.kind == "coordination"
        for index in bond.atom_indices if _is_metal(context.facts.atomic_numbers[index])
    }
    return tuple(component for component in context.facts.covalent_components
                 if not coordinated_centers.intersection(component.atom_indices))


def _source_charge_for_component(context, component, atom_charges, component_charges):
    supplied = {} if component_charges is None else dict(component_charges.component_charges)
    if component.identity in supplied:
        return supplied[component.identity], component_charges.provenance
    for source_component in context.facts.source_components:
        if (source_component.identity in supplied
                and source_component.atom_indices == component.atom_indices):
            return supplied[source_component.identity], component_charges.provenance
    if atom_charges is not None and all(
            atom_charges.charges[index] is not None for index in component.atom_indices):
        return sum(atom_charges.charges[index] for index in component.atom_indices), atom_charges.provenance
    return None


def _source_counted_component_ids(context: RuleContext) -> frozenset[str]:
    if context.candidate_space is None:
        return frozenset()
    return frozenset(
        fragment.component_identity
        for factor in context.candidate_space.factors
        for value in factor.values
        for candidate in (value.candidate,)
        if isinstance(candidate, CoordinationCountingCandidate)
        and candidate.provenance.authority in {
            StateValueAuthority.VERIFIED_EXPLICIT,
            StateValueAuthority.VERIFIED_SCIENTIFIC_SOURCE,
            StateValueAuthority.EXACT_IDENTITY_TRUSTED_SOURCE,
        }
        for fragment in candidate.fragments
    )


def _factor(quantity, subject, candidates, definition, limit):
    retained = tuple(candidates)[:limit]
    omitted = len(candidates) - len(retained)
    return ProposedCandidateFactor(
        FactorKey(quantity, subject), definition,
        tuple(ProposedCandidateValue(candidate) for candidate in retained),
        CandidateDomainScope.DOMAIN_BOUNDED,
        MaterializationStatus.RESOURCE_TRUNCATED if omitted else MaterializationStatus.COMPLETE,
        CandidateDomainSummary(omitted, _digest((definition, subject, limit)), None)
        if omitted else None,
    )


def _counting_factor(context, component, charge, provenance, definition):
    candidate = CoordinationCountingCandidate(
        provenance, CountingConvention.IONIC,
        (CoordinationFragmentAssignment(component.identity, charge, None, None),),
    )
    branch = BranchAssignment(
        f"counting-convention:{context.facts.identity.chemical_identity_digest}",
        CountingConvention.IONIC.value,
    )
    return ProposedCandidateFactor(
        FactorKey(ScientificQuantity.COORDINATION_COUNTING, component.identity),
        definition, (ProposedCandidateValue(candidate, (branch,)),),
        CandidateDomainScope.DOMAIN_BOUNDED, MaterializationStatus.COMPLETE, None,
    )


def _parent_constraint(child, child_factor, parent, parent_factor):
    return GuardedCandidateConstraint(
        ConstraintGuard(all_candidate_ids=(child,)),
        CandidateConstraint(
            _digest((child, parent)), "requires_all", _CONSTRAINT_SCHEMA,
            (child_factor, parent_factor), (("candidate_ids", (child, parent)),),
        ),
    )


class SourceComponentChargeRule(InferenceRule):
    """Publish complete supplied ligand charges before optional donor templates."""

    specification = RuleSpecification(
        _SOURCE_RULE, _VERSION, RuleKind.EMPIRICAL, RulePhase.SOURCE_EVIDENCE,
        (), (), (ResultType.CANDIDATE_PROPOSAL, ResultType.EVIDENCE),
        "supplied-complete-ligand-component-charge-v1",
    )

    def __init__(self, atom_charges=None, component_charges=None, source_digest=""):
        self._atom_charges = atom_charges
        self._component_charges = component_charges
        self._source_digest = source_digest

    def _matches(self, context):
        return tuple(
            (component, source)
            for component in _ligand_components(context)
            for source in (_source_charge_for_component(
                context, component, self._atom_charges, self._component_charges,
            ),)
            if source is not None
        )

    def applicable(self, context):
        return RuleApplicability(bool(self._matches(context)), ("SOURCE_COMPONENT_CHARGE_UNAVAILABLE",)
                                 if not self._matches(context) else ())

    def action(self, context, configuration):
        if configuration != (("source_evidence_digest", self._source_digest),):
            raise ValueError("Source component rule configuration does not bind its supplied evidence")
        factors, evidence = [], []
        for component, (charge, source) in self._matches(context):
            identity = _digest((_SOURCE_RULE, _VERSION, self._source_digest,
                                context.facts.identity.chemical_identity_digest,
                                component.identity, charge, source.identity))
            provenance = CandidateProvenance(
                identity, source.authority, _SOURCE_NAME,
                (source.identity, self._source_digest),
            )
            factors.append(_counting_factor(
                context, component, charge, provenance, f"{_SOURCE_NAME}:{self._source_digest}",
            ))
            evidence.append(EvidenceItem(source.identity, source.authority,
                                         source.source_identity,
                                         f"Complete supplied charge {charge} for component {component.identity}"))
        return RuleOutcome(
            _SOURCE_RULE, _VERSION, RuleApplicability(True),
            CandidateProposal(_digest((_SOURCE_RULE, self._source_digest,
                                       context.facts.identity.chemical_identity_digest)),
                              tuple(factors), ()),
            tuple(evidence), (), (), (),
        )


class AutomaticComponentDefaultsRule(InferenceRule):
    """Complete uncounted components and generic center domains with named defaults."""

    specification = RuleSpecification(
        _DEFAULT_RULE, _VERSION, RuleKind.EMPIRICAL, RulePhase.REDOX_CANDIDATE,
        (), (ResultType.COORDINATION_COUNTING_RESULT,),
        (ResultType.CANDIDATE_PROPOSAL, ResultType.EVIDENCE, ResultType.RANK_ADJUSTMENT),
        "source-aware-component-periodic-screening-v1",
    )

    def __init__(self, atom_charges=None, component_charges=None, radicals=(), source_digest=""):
        self._atom_charges = atom_charges
        self._component_charges = component_charges
        self._radicals = radicals
        self._source_digest = source_digest

    def applicable(self, context):
        matches = any(
            bond.kind == "coordination"
            and any(_is_metal(context.facts.atomic_numbers[index]) for index in bond.atom_indices)
            for bond in context.facts.semantic_bonds
        )
        return RuleApplicability(matches, () if matches else ("COORDINATION_BOUNDARY_REQUIRED",))

    def action(self, context, configuration):
        if configuration != (("source_evidence_digest", self._source_digest),):
            raise ValueError("Automatic component rule configuration does not bind its supplied evidence")
        facts = context.facts
        factors, constraints, ranks, evidence = [], [], [], []
        limit = context.candidate_budget.max_materialized_values_per_factor
        counting = {}
        if context.candidate_space is not None:
            for factor in context.candidate_space.factors:
                for value in factor.values:
                    candidate = value.candidate
                    if (isinstance(candidate, CoordinationCountingCandidate)
                            and candidate.convention is CountingConvention.IONIC):
                        for fragment in candidate.fragments:
                            if fragment.ionic_bookkeeping_charge is not None:
                                counting.setdefault(fragment.component_identity, []).append((
                                    factor.factor_key, candidate, fragment.ionic_bookkeeping_charge,
                                ))
        for component in _ligand_components(context):
            hydrogen_counts = tuple(
                _non_graph_hydrogens(facts.atomic_numbers[index],
                                     facts.hydrogen_inventory.explicit_counts[index],
                                     facts.hydrogen_inventory.implicit_counts[index])
                for index in component.atom_indices
            )
            if any(count is None for count in hydrogen_counts) or any(
                    facts.atomic_numbers[index] == 0 for index in component.atom_indices):
                continue
            if component.identity not in counting:
                charge = sum(
                    self._atom_charges.charges[index] or 0
                    for index in component.atom_indices
                ) if self._atom_charges is not None else 0
                assumptions = ("representation_default:unspecified-site-formal-charge-zero-v1",)
                identity = _digest((_DEFAULT_RULE, self._source_digest,
                                    facts.identity.chemical_identity_digest, component.identity, charge))
                provenance = CandidateProvenance(
                    identity, StateValueAuthority.DOMAIN_RULE, _DEFAULT_NAME,
                    (self._source_digest,), assumptions,
                )
                factor = _counting_factor(context, component, charge, provenance, _DEFAULT_NAME)
                factors.append(factor)
                counting[component.identity] = [(factor.factor_key, factor.materialized_candidates[0].candidate, charge)]
            spin_candidates = []
            spin_factor = FactorKey(ScientificQuantity.ELECTRONIC_STATE, component.identity)
            radical_sum = sum(self._radicals[index] or 0 for index in component.atom_indices) if self._radicals else 0
            for parent_factor, parent, charge in counting[component.identity]:
                electron_count = (sum(facts.atomic_numbers[index] for index in component.atom_indices)
                                  + sum(hydrogen_counts) - charge)
                two_s = radical_sum if radical_sum else electron_count % 2
                assumptions = (("execution_prior:source-radical-local-highest-spin-v1",)
                               if radical_sum else ("execution_prior:lowest-component-parity-spin-v1",))
                assumptions += ("execution_prior:lowest-coupled-molecular-spin-v1",)
                identity = _digest((_DEFAULT_RULE, self._source_digest, component.identity,
                                    parent.provenance.identity, two_s, "component-spin"))
                spin_candidates.append(SpinContributionCandidate(
                    CandidateProvenance(identity, StateValueAuthority.DOMAIN_RULE,
                                        _DEFAULT_NAME, (parent.provenance.identity, self._source_digest),
                                        assumptions),
                    SpinContributionScope.COMPONENT, component.identity,
                    (parent.provenance.identity,), two_s, True,
                ))
                constraints.append(_parent_constraint(identity, spin_factor,
                                                       parent.provenance.identity, parent_factor))
            factors.append(_factor(ScientificQuantity.ELECTRONIC_STATE, component.identity,
                                   spin_candidates, _DEFAULT_NAME, limit))
        scoped_numbers = {
            entry.atomic_number
            for artifact in (_lanthanide_artifact(), _americium_curium_artifact(), _thorium_artifact())
            for entry in artifact.elements
        }
        registered_ions = {(entry.atomic_number, entry.oxidation_state)
                           for entry in _configuration_registry().records}
        coordinated = {index for bond in facts.semantic_bonds if bond.kind == "coordination"
                       for index in bond.atom_indices}
        for index in sorted(coordinated):
            number = facts.atomic_numbers[index]
            if not _is_metal(number) or number in scoped_numbers:
                continue
            # Only a one-atom metal component admits a local ion bookkeeping default.
            if not any(component.atom_indices == (index,) for component in facts.covalent_components):
                continue
            domain = element_oxidation_state_domain(number)
            if domain.default_state is None:
                continue
            subject = f"atom:{facts.atom_map[index]}"
            oxidation_candidates, spin_candidates = [], []
            oxidation_factor = FactorKey(ScientificQuantity.OXIDATION_STATE, subject)
            spin_factor = FactorKey(ScientificQuantity.ELECTRONIC_STATE, subject)
            assumption = "execution_prior:periodic-oxidation-default-v1"
            # Possible states stay alternatives; only the declared table default is ranked.
            states_to_offer = tuple(dict.fromkeys((domain.default_state,) + domain.states))
            for oxidation_state in states_to_offer:
                states = [None] * len(facts.atomic_numbers)
                states[index] = oxidation_state
                identity = _digest((_DEFAULT_RULE, self._source_digest, domain.content_digest,
                                    facts.identity.chemical_identity_digest, index, oxidation_state))
                candidate = OxidationStateCandidateEvidence(
                    CandidateProvenance(identity, StateValueAuthority.DOMAIN_RULE,
                                        domain.source_identity, (domain.content_digest, self._source_digest),
                                        (assumption,)),
                    CountingConvention.IONIC, tuple(states), CoverageStatus.PARTIAL,
                )
                oxidation_candidates.append(candidate)
                if oxidation_state == domain.default_state:
                    ranks.append(RankAdjustment(identity, "oxidation_state_commonality", 1.0,
                                                f"Periodic screening default: {domain.default_basis}", (assumption,)))
                if (number, oxidation_state) not in registered_ions:
                    two_s = (number - oxidation_state) % 2
                    spin_id = _digest((identity, "lowest-ion-parity-spin", two_s))
                    spin_candidates.append(SpinContributionCandidate(
                        CandidateProvenance(spin_id, StateValueAuthority.DOMAIN_RULE,
                                            _DEFAULT_NAME, (identity,),
                                            ("execution_prior:lowest-ion-parity-spin-v1",)),
                        SpinContributionScope.ATOM, subject, (identity,), two_s, True,
                    ))
                    constraints.append(_parent_constraint(spin_id, spin_factor, identity, oxidation_factor))
            factors.append(_factor(ScientificQuantity.OXIDATION_STATE, subject,
                                   oxidation_candidates, f"{_DEFAULT_NAME}:{domain.content_digest}", limit))
            if spin_candidates:
                factors.append(_factor(ScientificQuantity.ELECTRONIC_STATE, subject,
                                       spin_candidates, _DEFAULT_NAME, limit))
            evidence.append(EvidenceItem(domain.content_digest, StateValueAuthority.VERIFIED_SCIENTIFIC_SOURCE,
                                         domain.source_identity,
                                         f"Z={number}: possible={domain.states}; common={domain.common_states}; "
                                         f"important={domain.important_states}; default={domain.default_state} "
                                         f"({domain.default_basis})"))
        retained_ids = {
            value.candidate.provenance.identity
            for factor in factors for value in factor.materialized_candidates
        }
        constraints = [constraint for constraint in constraints
                       if constraint.guard.all_candidate_ids[0] in retained_ids]
        assumptions = tuple(sorted({
            assumption for factor in factors for value in factor.materialized_candidates
            for assumption in value.candidate.provenance.assumption_ids
        }))
        return RuleOutcome(
            _DEFAULT_RULE, _VERSION, RuleApplicability(True),
            CandidateProposal(_digest((_DEFAULT_RULE, self._source_digest,
                                       facts.identity.chemical_identity_digest)), tuple(factors), tuple(constraints)),
            tuple(evidence), tuple(ranks), assumptions, (),
        )
