"""Pre-activation scientific-state resolver benchmark."""

from __future__ import annotations

import hashlib
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import hotpot

from hotpot.cheminfo.calculator import (
    ScientificStateBundle,
    extract_molecular_facts,
    resolve_scientific_state,
)
from hotpot.cheminfo.calculator.state_composition import MaterializationStatus
from hotpot.cheminfo.calculator.state_composition._decision_snapshot import (
    _decision_snapshot_to_json,
)
from hotpot.cheminfo.calculator.state_rules.manifest import (
    _coordination_screening_manifest,
)
from hotpot.cheminfo.core import Molecule

from .corpus import DryStateCase, DryStateSourceKind

__all__ = ["DryResolutionRecord", "resolve_case"]


@dataclass(frozen=True)
class DryResolutionRecord:
    """Measured state-resolution facts for one immutable source record."""

    index: int
    case_id: str
    source_kind: str
    format: str
    source_sha256: str
    facts_identity: str
    atom_count: int
    coordination_bond_count: int
    profile_id: str
    profile_version: str
    profile_activation_status: str
    profile_content_digest: str
    parse_seconds: float
    facts_seconds: float
    resolve_seconds: float
    python_peak_bytes: int
    expanded_path_count: int
    rejected_path_count: int
    frontier_count: int
    truncated: bool
    search_status: str
    scientific_resolution_status: str
    scientific_resolution_exhaustive: bool
    decision_action: str
    decision_source: str
    certification_status: str
    profile_in_domain: bool
    profile_finding_codes: tuple[str, ...]
    static_profile_ready_except_activation: bool
    snapshot_sha256: str

    def to_payload(self) -> dict[str, object]:
        """Return the stable JSON-ready record payload."""

        return asdict(self)


def _source_digest(case: DryStateCase) -> str:
    if case.source_kind is DryStateSourceKind.SMILES:
        encoded = case.source.encode("utf-8")
    else:
        encoded = Path(case.source).read_bytes()
    return hashlib.sha256(encoded).hexdigest()


def _load_molecule(case: DryStateCase) -> Molecule:
    if _source_digest(case) != case.source_sha256:
        raise ValueError(f"Source digest changed for case {case.case_id!r}")
    source = (
        Path(case.source)
        if case.source_kind is DryStateSourceKind.FILE
        else case.source
    )
    return hotpot.read_mol(source, case.format)


def _snapshot_digest(bundle: ScientificStateBundle) -> str:
    snapshot = _decision_snapshot_to_json(
        bundle.candidate_space,
        bundle.scientific_resolution,
        bundle.execution_decision,
        bundle.rule_trace,
    )
    return hashlib.sha256(snapshot.encode("utf-8")).hexdigest()


def resolve_case(case: DryStateCase) -> DryResolutionRecord:
    """Resolve one case with the exact inactive screening manifest."""

    parse_started = perf_counter()
    mol = _load_molecule(case)
    parse_seconds = perf_counter() - parse_started

    facts_started = perf_counter()
    facts = extract_molecular_facts(mol)
    facts_seconds = perf_counter() - facts_started

    manifest = _coordination_screening_manifest()
    if manifest.activation_status != "planned":
        raise RuntimeError(
            "The pre-activation benchmark requires a planned profile artifact"
        )

    tracemalloc.start()
    try:
        resolve_started = perf_counter()
        bundle = resolve_scientific_state(
            facts,
            policy=manifest.policy,
            provider_set=manifest.provider_set,
            rule_set=manifest.rule_set,
            candidate_budget=manifest.candidate_budget,
        )
        resolve_seconds = perf_counter() - resolve_started
        _, python_peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    candidate_space = bundle.candidate_space
    decision = bundle.execution_decision
    finding_codes = tuple(
        finding.code for finding in decision.static_profile_eligibility.findings
    )
    activation_code = "PROFILE.SCREENING_ARTIFACT_NOT_ACTIVE"
    truncated = bool(candidate_space.frontier) or any(
        factor.materialization_status is MaterializationStatus.RESOURCE_TRUNCATED
        for factor in candidate_space.factors
    )
    return DryResolutionRecord(
        index=case.index,
        case_id=case.case_id,
        source_kind=case.source_kind.value,
        format=case.format,
        source_sha256=case.source_sha256,
        facts_identity=facts.identity.chemical_identity_digest,
        atom_count=len(facts.atomic_numbers),
        coordination_bond_count=sum(
            bond.kind == "coordination" for bond in facts.semantic_bonds
        ),
        profile_id=manifest.profile_id,
        profile_version=manifest.profile_version,
        profile_activation_status=manifest.activation_status,
        profile_content_digest=manifest.profile_content_digest,
        parse_seconds=parse_seconds,
        facts_seconds=facts_seconds,
        resolve_seconds=resolve_seconds,
        python_peak_bytes=python_peak_bytes,
        expanded_path_count=len(candidate_space.expanded_paths),
        rejected_path_count=len(candidate_space.rejected_path_summaries),
        frontier_count=len(candidate_space.frontier),
        truncated=truncated,
        search_status=candidate_space.search_status.value,
        scientific_resolution_status=bundle.scientific_resolution.status.value,
        scientific_resolution_exhaustive=bundle.scientific_resolution.exhaustive,
        decision_action=decision.action.value,
        decision_source=decision.source.value,
        certification_status=decision.decision_certifiability.status.value,
        profile_in_domain=decision.static_profile_eligibility.in_domain,
        profile_finding_codes=finding_codes,
        static_profile_ready_except_activation=(
            finding_codes == (activation_code,)
        ),
        snapshot_sha256=_snapshot_digest(bundle),
    )
