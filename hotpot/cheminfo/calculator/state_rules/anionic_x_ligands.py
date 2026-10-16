"""OPTIONAL-004 ordinary anionic X-type ligand rule."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Optional, Tuple

from ..contracts import (
    CandidateProvenance,
    ScientificQuantity,
    SemanticBond,
    StateValueAuthority,
)
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

__all__ = ["AnionicXLigandRule"]


_RULE_ID = "OPTIONAL-004"
_RULE_VERSION = "1.0.0"
_REGISTRY_SCHEMA = "hotpot-anionic-x-ligand-registry-v1"
_REGISTRY_RESOURCE = "data/rules/anionic_x_ligands_v1.json"
_DOMAIN_DEFINITION = "ordinary-anionic-x-ligand-v1"
_HALOGENS = frozenset({9, 17, 35, 53, 85})


def _digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class _XLigandRegistry:
    version: str
    source_identity: str
    reference: str
    motifs: Tuple[str, ...]
    content_digest: str


@lru_cache(maxsize=1)
def _x_ligand_registry() -> _XLigandRegistry:
    resource = resources.files(__package__).joinpath(_REGISTRY_RESOURCE)
    encoded = resource.read_bytes()
    payload = json.loads(encoded.decode("utf-8"))
    expected = {
        "schema_version",
        "registry_version",
        "source_identity",
        "reference",
        "motifs",
    }
    if set(payload) != expected or payload["schema_version"] != _REGISTRY_SCHEMA:
        raise ValueError("Anionic X-ligand registry does not match its schema")
    motifs = tuple(payload["motifs"])
    if not motifs or len(set(motifs)) != len(motifs):
        raise ValueError("Anionic X-ligand motifs must be unique and non-empty")
    return _XLigandRegistry(
        version=payload["registry_version"],
        source_identity=payload["source_identity"],
        reference=payload["reference"],
        motifs=motifs,
        content_digest=hashlib.sha256(encoded).hexdigest(),
    )


def _component_bonds(
    context: RuleContext,
    atom_indices: Tuple[int, ...],
) -> Tuple[SemanticBond, ...]:
    atoms = frozenset(atom_indices)
    return tuple(
        bond
        for bond in context.facts.semantic_bonds
        if set(bond.atom_indices).issubset(atoms)
    )


def _coordination_donor(
    context: RuleContext,
    atom_indices: Tuple[int, ...],
) -> Optional[int]:
    atoms = frozenset(atom_indices)
    donors = tuple(
        begin if begin in atoms else end
        for bond in context.facts.semantic_bonds
        if bond.kind == "coordination"
        for begin, end in (bond.atom_indices,)
        if (begin in atoms) != (end in atoms)
    )
    return donors[0] if len(donors) == 1 else None


def _implicit_hydrogens(context: RuleContext, atom_index: int) -> int:
    count = context.facts.hydrogen_inventory.implicit_counts[atom_index]
    return 0 if count is None else count


def _covalent_neighbours(
    context: RuleContext,
    atom_index: int,
) -> Tuple[Tuple[int, float], ...]:
    neighbours = []
    for bond in context.facts.semantic_bonds:
        if (
            bond.kind != "covalent"
            or bond.order is None
            or atom_index not in bond.atom_indices
        ):
            continue
        begin, end = bond.atom_indices
        neighbours.append((end if begin == atom_index else begin, bond.order))
    return tuple(neighbours)


def _is_halide_or_hydride(
    context: RuleContext,
    atom_indices: Tuple[int, ...],
    donor_index: int,
) -> Optional[str]:
    if len(atom_indices) != 1:
        return None
    atomic_number = context.facts.atomic_numbers[donor_index]
    if atomic_number in _HALOGENS:
        return "halide"
    return "hydride" if atomic_number == 1 else None


def _is_hydroxide_or_alkoxide(
    context: RuleContext,
    donor_index: int,
) -> Optional[str]:
    if context.facts.atomic_numbers[donor_index] != 8:
        return None
    neighbours = _covalent_neighbours(context, donor_index)
    valence = sum(order for _, order in neighbours) + _implicit_hydrogens(
        context,
        donor_index,
    )
    if not math.isclose(valence, 1.0, abs_tol=1.0e-8):
        return None
    return "hydroxide" if not neighbours else "alkoxide"


def _is_carboxylate(
    context: RuleContext,
    donor_index: int,
) -> bool:
    if context.facts.atomic_numbers[donor_index] != 8:
        return False
    donor_neighbours = _covalent_neighbours(context, donor_index)
    if len(donor_neighbours) != 1:
        return False
    carbon_index, _ = donor_neighbours[0]
    if context.facts.atomic_numbers[carbon_index] != 6:
        return False
    oxygen_bonds = tuple(
        (index, order)
        for index, order in _covalent_neighbours(context, carbon_index)
        if context.facts.atomic_numbers[index] == 8
    )
    if len(oxygen_bonds) != 2 or not math.isclose(
        sum(order for _, order in oxygen_bonds),
        3.0,
        abs_tol=1.0e-8,
    ):
        return False
    terminal_oxygens = tuple(
        index
        for index, _ in oxygen_bonds
        if len(_covalent_neighbours(context, index)) == 1
        and _implicit_hydrogens(context, index) == 0
    )
    return donor_index in terminal_oxygens


def _is_cyanide(
    context: RuleContext,
    atom_indices: Tuple[int, ...],
    donor_index: int,
) -> bool:
    if tuple(sorted(context.facts.atomic_numbers[index] for index in atom_indices)) != (
        6,
        7,
    ):
        return False
    bonds = tuple(
        bond
        for bond in _component_bonds(context, atom_indices)
        if bond.kind == "covalent" and bond.order is not None
    )
    return (
        len(bonds) == 1
        and math.isclose(bonds[0].order, 3.0, abs_tol=1.0e-8)
        and context.facts.atomic_numbers[donor_index] in {6, 7}
    )


def _motif(
    context: RuleContext,
    atom_indices: Tuple[int, ...],
    donor_index: int,
) -> Optional[str]:
    return (
        _is_halide_or_hydride(context, atom_indices, donor_index)
        or _is_hydroxide_or_alkoxide(context, donor_index)
        or ("carboxylate" if _is_carboxylate(context, donor_index) else None)
        or ("cyanide" if _is_cyanide(context, atom_indices, donor_index) else None)
    )


def _matching_components(
    context: RuleContext,
) -> Tuple[Tuple[str, int, str], ...]:
    matches = []
    for component in context.facts.covalent_components:
        donor_index = _coordination_donor(context, component.atom_indices)
        if donor_index is None:
            continue
        motif = _motif(context, component.atom_indices, donor_index)
        if motif is not None:
            matches.append((component.identity, donor_index, motif))
    return tuple(matches)


class AnionicXLigandRule(InferenceRule):
    """Propose ionic -1 bookkeeping for supported whole-ligand motifs."""

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
            () if matches else ("NO_SUPPORTED_ANIONIC_X_LIGAND",),
        )

    def action(
        self,
        context: RuleContext,
        configuration: Tuple[Tuple[str, RuleValue], ...],
    ) -> RuleOutcome:
        if configuration:
            raise ValueError("OPTIONAL-004 does not accept configuration")
        registry = _x_ligand_registry()
        matches = _matching_components(context)
        branch = BranchAssignment(
            f"counting-convention:{context.facts.identity.chemical_identity_digest}",
            CountingConvention.IONIC.value,
        )
        factors = []
        for component_identity, donor_index, motif in matches:
            identity = _digest(
                {
                    "rule": _RULE_ID,
                    "version": _RULE_VERSION,
                    "registry": registry.content_digest,
                    "facts": context.facts.identity.chemical_identity_digest,
                    "component": component_identity,
                    "donor": donor_index,
                    "motif": motif,
                }
            )
            candidate = CoordinationCountingCandidate(
                provenance=CandidateProvenance(
                    identity=identity,
                    authority=StateValueAuthority.DOMAIN_RULE,
                    source_identity=registry.source_identity,
                    evidence_ids=(registry.content_digest,),
                    assumption_ids=("ionic_approximation:optional-004",),
                ),
                convention=CountingConvention.IONIC,
                fragments=(
                    CoordinationFragmentAssignment(
                        component_identity,
                        -1,
                        None,
                        None,
                    ),
                ),
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
                    materialized_candidates=(
                        ProposedCandidateValue(candidate, (branch,)),
                    ),
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
                    "matches": matches,
                }
            ),
            factors=tuple(factors),
            constraints=(),
        )
        evidence = EvidenceItem(
            identity=registry.content_digest,
            authority=StateValueAuthority.VERIFIED_SCIENTIFIC_SOURCE,
            source_identity=registry.source_identity,
            claim="Curated ordinary anionic X-ligand applicability domain",
            locator=registry.reference,
        )
        return RuleOutcome(
            rule_id=_RULE_ID,
            rule_version=_RULE_VERSION,
            applicability=RuleApplicability(True),
            candidate_proposal=proposal,
            evidence=(evidence,),
            rank_adjustments=(),
            assumptions=("ionic_approximation:optional-004",),
            warnings=(),
        )
