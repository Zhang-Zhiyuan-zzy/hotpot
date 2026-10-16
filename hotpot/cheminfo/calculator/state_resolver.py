"""Scientific-state orchestration and immutable molecular-fact extraction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

from ..core import Bond, BondKind, Molecule
from .charge.contracts import (
    AtomFormalChargeEvidence,
    ComponentChargeEvidence,
    LewisFormalChargeResult,
    NetChargeCandidateEvidence,
    NetChargeResult,
)
from .contracts import (
    CandidateBudget,
    ComponentIdentity,
    DEFAULT_CANDIDATE_BUDGET,
    HydrogenInventory,
    MolecularFacts,
    MolecularFactsIdentity,
    ProviderSet,
    ResolutionStatus,
    SemanticBond,
    SemanticBondKind,
    ScientificAdmissibility,
    ScientificOperationRequirements,
    StatePolicy,
    StateValueAuthority,
)
from .electronic_state.contracts import (
    ElectronicStateCandidateEvidence,
    ElectronicStateInferenceResult,
)
from .oxidation_state.contracts import (
    CoordinationElectronCountingResult,
    OxidationStateCandidateEvidence,
    OxidationStateInferenceResult,
    OxidationStateOverride,
)
from .state_composition.contracts import (
    CandidateProposal,
    ScientificResolutionResult,
    ScientificStateCandidateSpace,
)
from .state_rules.contracts import RankAdjustment, RuleSet, RuleTrace
from .state_selection import ExecutionStateDecision

__all__ = [
    "extract_molecular_facts",
    "resolve_scientific_state",
    "StateRequest",
    "ProvidedScientificResults",
    "ScientificStateBundle",
]


# Orchestrator inputs.


@dataclass(frozen=True)
class StateRequest:
    """Explicit constraints and attributable allowed candidate domains."""

    net_charge: Optional[int] = None
    atom_formal_charges: Optional[AtomFormalChargeEvidence] = None
    component_charges: Optional[ComponentChargeEvidence] = None
    oxidation_overrides: Tuple[OxidationStateOverride, ...] = ()
    two_s: Optional[int] = None
    multiplicity: Optional[int] = None
    charge_candidates: Tuple[NetChargeCandidateEvidence, ...] = ()
    oxidation_candidates: Tuple[OxidationStateCandidateEvidence, ...] = ()
    electronic_candidates: Tuple[ElectronicStateCandidateEvidence, ...] = ()


@dataclass(frozen=True)
class ProvidedScientificResults:
    """Precomputed scientific results admitted to one resolution run."""

    net_charge_results: Tuple[NetChargeResult, ...] = ()
    lewis_results: Tuple[LewisFormalChargeResult, ...] = ()
    coordination_counting_results: Tuple[
        CoordinationElectronCountingResult, ...
    ] = ()
    oxidation_state_results: Tuple[OxidationStateInferenceResult, ...] = ()
    electronic_state_results: Tuple[ElectronicStateInferenceResult, ...] = ()


@dataclass(frozen=True)
class ScientificStateBundle:
    """Complete attributable output of one scientific-state resolution run."""

    facts: MolecularFacts
    net_charge_results: Tuple[NetChargeResult, ...]
    lewis_results: Tuple[LewisFormalChargeResult, ...]
    coordination_counting_results: Tuple[
        CoordinationElectronCountingResult,
        ...,
    ]
    oxidation_state_results: Tuple[OxidationStateInferenceResult, ...]
    electronic_state_results: Tuple[ElectronicStateInferenceResult, ...]
    candidate_space: ScientificStateCandidateSpace
    scientific_resolution: ScientificResolutionResult
    execution_decision: ExecutionStateDecision
    rule_trace: Tuple[RuleTrace, ...]


# Canonical chemical-boundary serialization.


def _sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atom_map(mol: Molecule) -> Tuple[int, ...]:
    atom_ids = tuple(int(atom.id) for atom in mol.atoms)
    if len(set(atom_ids)) == len(atom_ids) and all(atom_id >= 0 for atom_id in atom_ids):
        return atom_ids
    return tuple(range(len(mol.atoms)))


def _semantic_bond_kind(bond: Bond) -> SemanticBondKind:
    if bond.bond_kind is BondKind.DATIVE:
        return "coordination"
    if bond.bond_kind is BondKind.UNKNOWN or bond.is_metal_ligand_bond:
        return "unknown"
    return "covalent"


def _semantic_bonds(mol: Molecule) -> Tuple[SemanticBond, ...]:
    bonds = []
    for bond in mol.bonds:
        atom_indices = tuple(sorted((int(bond.a1idx), int(bond.a2idx))))
        order = None if bond.bond_kind is BondKind.UNKNOWN else float(bond.bond_order)
        bonds.append(
            SemanticBond(
                atom_indices=atom_indices,
                order=order,
                aromatic=bond.bond_kind is BondKind.AROMATIC,
                kind=_semantic_bond_kind(bond),
            )
        )
    return tuple(sorted(bonds, key=lambda item: item.atom_indices))


def _component_indices(
    atom_count: int,
    bonds: Iterable[SemanticBond],
) -> Tuple[Tuple[int, ...], ...]:
    adjacency = [set() for _ in range(atom_count)]
    for bond in bonds:
        begin, end = bond.atom_indices
        adjacency[begin].add(end)
        adjacency[end].add(begin)

    components = []
    unseen = set(range(atom_count))
    while unseen:
        pending = [min(unseen)]
        component = set()
        while pending:
            atom_index = pending.pop()
            if atom_index in component:
                continue
            component.add(atom_index)
            unseen.discard(atom_index)
            pending.extend(adjacency[atom_index] - component)
        components.append(tuple(sorted(component)))
    return tuple(components)


def _component_identities(
    component_kind: str,
    components: Tuple[Tuple[int, ...], ...],
    atom_map: Tuple[int, ...],
) -> Tuple[ComponentIdentity, ...]:
    return tuple(
        ComponentIdentity(
            identity=_sha256(
                {
                    "component_kind": component_kind,
                    "mapped_atom_ids": sorted(atom_map[index] for index in component),
                }
            ),
            atom_indices=component,
        )
        for component in components
    )


def _coordinates(
    mol: Molecule,
) -> Optional[Tuple[Tuple[float, float, float], ...]]:
    if not mol.has_3d:
        return None
    return tuple(
        tuple(float(value) for value in atom.coordinates)
        for atom in mol.atoms
    )


def _chemical_identity_payload(facts: MolecularFacts) -> object:
    atom_records = sorted(
        (
            facts.atom_map[index],
            atomic_number,
            facts.mass_numbers[index],
            facts.hydrogen_inventory.explicit_counts[index],
            facts.hydrogen_inventory.implicit_counts[index],
        )
        for index, atomic_number in enumerate(facts.atomic_numbers)
    )
    bond_records = sorted(
        (
            min(facts.atom_map[begin], facts.atom_map[end]),
            max(facts.atom_map[begin], facts.atom_map[end]),
            bond.order,
            bond.aromatic,
            bond.kind,
        )
        for bond in facts.semantic_bonds
        for begin, end in (bond.atom_indices,)
    )
    return {
        "atoms": atom_records,
        "bonds": bond_records,
        "source_components": sorted(
            sorted(facts.atom_map[index] for index in component.atom_indices)
            for component in facts.source_components
        ),
        "covalent_components": sorted(
            sorted(facts.atom_map[index] for index in component.atom_indices)
            for component in facts.covalent_components
        ),
    }


def _nuclear_charge_sum(
    atomic_numbers: Tuple[int, ...],
    hydrogen_inventory: HydrogenInventory,
) -> int:
    explicit_nuclear_charge = sum(atomic_numbers)
    implicit_hydrogen_count = sum(
        implicit_count
        for atomic_number, explicit_count, implicit_count in zip(
            atomic_numbers,
            hydrogen_inventory.explicit_counts,
            hydrogen_inventory.implicit_counts,
        )
        if atomic_number != 1
        and explicit_count == 0
        and implicit_count is not None
    )
    return explicit_nuclear_charge + implicit_hydrogen_count


# Public fact extraction.


def extract_molecular_facts(mol: Molecule) -> MolecularFacts:
    """Extract immutable chemical facts without mutating ``mol``."""
    atoms = tuple(mol.atoms)
    atomic_numbers = tuple(int(atom.atomic_number) for atom in atoms)
    mass_numbers = tuple(
        int(atom.isotope) if getattr(atom, "isotope", None) else None
        for atom in atoms
    )
    atom_map = _atom_map(mol)
    semantic_bonds = _semantic_bonds(mol)
    source_components = _component_identities(
        "source",
        _component_indices(len(atoms), semantic_bonds),
        atom_map,
    )
    covalent_bonds = tuple(
        bond for bond in semantic_bonds if bond.kind != "coordination"
    )
    covalent_components = _component_identities(
        "covalent",
        _component_indices(len(atoms), covalent_bonds),
        atom_map,
    )
    hydrogen_inventory = HydrogenInventory(
        explicit_counts=tuple(
            sum(neighbour.atomic_number == 1 for neighbour in atom.neighbours)
            for atom in atoms
        ),
        implicit_counts=tuple(
            int(atom.implicit_hydrogens) if atom.implicit_hydrogens else None
            for atom in atoms
        ),
    )
    coordinates = _coordinates(mol)
    provisional = MolecularFacts(
        identity=MolecularFactsIdentity(chemical_identity_digest=""),
        geometry_digest=_sha256(coordinates) if coordinates is not None else None,
        atomic_numbers=atomic_numbers,
        mass_numbers=mass_numbers,
        atom_map=atom_map,
        source_components=source_components,
        covalent_components=covalent_components,
        hydrogen_inventory=hydrogen_inventory,
        semantic_bonds=semantic_bonds,
        coordinates=coordinates,
        nuclear_charge_sum=_nuclear_charge_sum(
            atomic_numbers,
            hydrogen_inventory,
        ),
    )
    return MolecularFacts(
        identity=MolecularFactsIdentity(
            chemical_identity_digest=_sha256(_chemical_identity_payload(provisional))
        ),
        geometry_digest=provisional.geometry_digest,
        atomic_numbers=provisional.atomic_numbers,
        mass_numbers=provisional.mass_numbers,
        atom_map=provisional.atom_map,
        source_components=provisional.source_components,
        covalent_components=provisional.covalent_components,
        hydrogen_inventory=provisional.hydrogen_inventory,
        semantic_bonds=provisional.semantic_bonds,
        coordinates=provisional.coordinates,
        nuclear_charge_sum=provisional.nuclear_charge_sum,
    )


# Public scientific-state orchestration.


_KNOWN_PROVIDER_IDS = frozenset(
    {
        "PROVIDER-001",
        "PROVIDER-002",
        "PROVIDER-003",
    }
)


def _enabled_provider_ids(provider_set: ProviderSet) -> frozenset[str]:
    provider_ids = tuple(toggle.provider_id for toggle in provider_set.toggles)
    if len(set(provider_ids)) != len(provider_ids):
        raise ValueError("Provider toggles contain duplicate provider IDs")
    unknown = set(provider_ids) - _KNOWN_PROVIDER_IDS
    if unknown:
        raise ValueError(f"Unknown provider IDs: {tuple(sorted(unknown))}")
    configured = tuple(
        toggle.provider_id
        for toggle in provider_set.toggles
        if toggle.configuration
    )
    if configured:
        raise ValueError(
            "Provider configuration is not defined for "
            f"{tuple(sorted(configured))}"
        )
    return frozenset(
        toggle.provider_id
        for toggle in provider_set.toggles
        if toggle.enabled
    )


def _validated_lewis_result(
    facts: MolecularFacts,
    result: LewisFormalChargeResult,
    charge_results: Tuple[NetChargeResult, ...],
) -> LewisFormalChargeResult:
    from . import legality

    charge_by_identity = {
        candidate.provenance.identity: candidate
        for charge_result in charge_results
        if charge_result.facts_identity == facts.identity
        and charge_result.admissibility is ScientificAdmissibility.ADMISSIBLE
        for candidate in charge_result.candidates
    }
    assessments = [
        legality._validate_facts_identity(
            facts.identity,
            result.facts_identity,
            related_identities=(result.identity,),
        )
    ]
    for candidate in result.candidates:
        charge = charge_by_identity.get(
            candidate.net_charge_candidate_identity
        )
        assessments.append(
            legality._validate_lewis_candidate(facts, charge, candidate)
            if charge is not None
            else legality._hard_failure(
                "HARD-007.LEWIS_CHARGE_REFERENCE_MISSING",
                "Lewis evidence references an unavailable charge candidate.",
                related_identities=(candidate.provenance.identity,),
            )
        )
    assessment = legality._merge_hard_rule_assessments(tuple(assessments))
    if assessment.admissibility is ScientificAdmissibility.ADMISSIBLE:
        return result
    return LewisFormalChargeResult(
        identity=result.identity,
        facts_identity=result.facts_identity,
        candidates=result.candidates,
        admissibility=assessment.admissibility,
        resolution=assessment.terminal_status or ResolutionStatus.UNRESOLVED,
        confidence=result.confidence,
        findings=result.findings + assessment.findings,
    )


def _validated_counting_result(
    facts: MolecularFacts,
    result: CoordinationElectronCountingResult,
) -> CoordinationElectronCountingResult:
    from . import legality

    assessments = (
        legality._validate_facts_identity(
            facts.identity,
            result.facts_identity,
            related_identities=(result.identity,),
        ),
        *tuple(
            legality._validate_counting_candidate_for_facts(facts, candidate)
            for candidate in result.candidates
        ),
    )
    assessment = legality._merge_hard_rule_assessments(assessments)
    if assessment.admissibility is ScientificAdmissibility.ADMISSIBLE:
        return result
    return CoordinationElectronCountingResult(
        identity=result.identity,
        facts_identity=result.facts_identity,
        candidates=result.candidates,
        admissibility=assessment.admissibility,
        resolution=assessment.terminal_status or ResolutionStatus.UNRESOLVED,
        confidence=result.confidence,
        findings=result.findings + assessment.findings,
    )


def _explicit_candidate_ids(
    charge_results: Tuple[NetChargeResult, ...],
    oxidation_state_results: Tuple[OxidationStateInferenceResult, ...],
    electronic_state_results: Tuple[ElectronicStateInferenceResult, ...],
) -> frozenset[str]:
    return frozenset(
        candidate.provenance.identity
        for results in (
            charge_results,
            oxidation_state_results,
            electronic_state_results,
        )
        for result in results
        for candidate in result.candidates
        if candidate.provenance.authority
        is StateValueAuthority.VERIFIED_EXPLICIT
    )


def resolve_scientific_state(
    facts: MolecularFacts,
    request: StateRequest = StateRequest(),
    *,
    policy: StatePolicy = StatePolicy.STRICT,
    requirements: Optional[ScientificOperationRequirements] = None,
    provider_set: ProviderSet = ProviderSet(),
    provided_results: ProvidedScientificResults = ProvidedScientificResults(),
    rule_set: RuleSet = RuleSet(),
    candidate_budget: CandidateBudget = DEFAULT_CANDIDATE_BUDGET,
) -> ScientificStateBundle:
    """Resolve, compose and decide one immutable molecular scientific state."""
    from .charge import resolve_net_charge, solve_main_group_lewis
    from .defaults import (
        _decide_scientific_state,
        _decision_search_compiler,
    )
    from .electronic_state import infer_electronic_states
    from .operation_requirements import _effective_requirements
    from .oxidation_state import infer_oxidation_states
    from .profile_eligibility import _static_profile_eligibility
    from .state_composition.compose import _compose_scientific_state_space
    from .state_composition.constraints import (
        _reconcile_authoritative_results,
    )
    from .state_composition.resolution import _resolve_candidate_space

    if policy is StatePolicy.STRICT and rule_set.layers:
        raise ValueError("STRICT policy cannot execute optional rules")
    enabled_provider_ids = _enabled_provider_ids(provider_set)
    effective_requirements = _effective_requirements(policy, requirements)

    charge_results = resolve_net_charge(
        facts,
        request,
        provided=provided_results.net_charge_results,
    )
    validated_provided_lewis = tuple(
        _validated_lewis_result(facts, result, charge_results)
        for result in provided_results.lewis_results
    )
    generated_lewis = (
        tuple(
            lewis_result
            for charge_result in charge_results
            for lewis_result in solve_main_group_lewis(
                facts,
                charge_result,
                candidate_budget=candidate_budget,
            )
        )
        if "PROVIDER-001" in enabled_provider_ids
        else ()
    )
    lewis_results = validated_provided_lewis + generated_lewis
    counting_results = tuple(
        _validated_counting_result(facts, result)
        for result in provided_results.coordination_counting_results
    )
    oxidation_state_results = infer_oxidation_states(
        facts,
        request,
        counting_results=counting_results,
        provided=provided_results.oxidation_state_results,
    )
    electronic_state_results = infer_electronic_states(
        facts,
        request,
        net_charge_results=charge_results,
        oxidation_state_results=oxidation_state_results,
        provided=provided_results.electronic_state_results,
    )
    cross_result_constraints, cross_result_conflict = (
        _reconcile_authoritative_results(
            facts,
            charge_results,
            oxidation_state_results,
            electronic_state_results,
            request.oxidation_overrides,
        )
    )
    eligibility = _static_profile_eligibility(
        facts,
        policy,
        effective_requirements,
        provider_set,
        rule_set,
        candidate_budget,
    )

    def compose_with_rule_outputs(
        candidate_proposals: Tuple[CandidateProposal, ...],
        rank_adjustments: Tuple[RankAdjustment, ...],
    ) -> ScientificStateCandidateSpace:
        return _compose_scientific_state_space(
            facts,
            net_charge_results=charge_results,
            lewis_results=lewis_results,
            coordination_counting_results=counting_results,
            oxidation_state_results=oxidation_state_results,
            electronic_state_results=electronic_state_results,
            candidate_proposals=candidate_proposals,
            constraints=cross_result_constraints,
            candidate_budget=candidate_budget,
            decision_search=_decision_search_compiler(
                policy,
                effective_requirements,
                eligibility,
                rank_adjustments,
            ),
        )

    candidate_space = compose_with_rule_outputs((), ())
    rule_trace: Tuple[RuleTrace, ...] = ()
    final_rank_adjustments: Tuple[RankAdjustment, ...] = ()
    if rule_set.layers:
        from .state_rules.controller import _execute_rule_layers

        rule_execution = _execute_rule_layers(
            rule_set,
            facts,
            effective_requirements,
            candidate_budget,
            candidate_space,
            compose_with_rule_outputs,
        )
        candidate_space = rule_execution.candidate_space
        rule_trace = rule_execution.traces
        final_rank_adjustments = rule_execution.rank_adjustments
    scientific_resolution = _resolve_candidate_space(candidate_space)
    all_results = (
        charge_results
        + lewis_results
        + counting_results
        + oxidation_state_results
        + electronic_state_results
    )
    result_states = tuple(
        (result.admissibility, result.resolution)
        for result in all_results
    )
    if cross_result_conflict:
        result_states += (
            (
                ScientificAdmissibility.INADMISSIBLE,
                ResolutionStatus.CONFLICT,
            ),
        )
    decision = _decide_scientific_state(
        policy,
        effective_requirements,
        candidate_space,
        eligibility,
        result_states,
        _explicit_candidate_ids(
            charge_results,
            oxidation_state_results,
            electronic_state_results,
        ),
        _decision_search_compiler(
            policy,
            effective_requirements,
            eligibility,
            final_rank_adjustments,
        ),
    )
    return ScientificStateBundle(
        facts=facts,
        net_charge_results=charge_results,
        lewis_results=lewis_results,
        coordination_counting_results=counting_results,
        oxidation_state_results=oxidation_state_results,
        electronic_state_results=electronic_state_results,
        candidate_space=candidate_space,
        scientific_resolution=scientific_resolution,
        execution_decision=decision,
        rule_trace=rule_trace,
    )
