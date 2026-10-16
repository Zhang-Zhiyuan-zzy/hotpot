"""Pre-activation scientific-state resolver benchmark."""

from __future__ import annotations

import hashlib
import json
import tracemalloc
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import TextIO

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

from .corpus import DryStateCase, DryStateCorpus, DryStateSourceKind

__all__ = [
    "DryCaseError",
    "DryResolutionRecord",
    "resolve_case",
    "run_benchmark",
]


_SCHEMA_VERSION = 1


class DryCaseError(RuntimeError):
    """Expected per-record failure at one named dry-resolution stage."""

    def __init__(self, stage: str, error: Exception) -> None:
        self.stage = stage
        self.error_type = type(error).__name__
        self.error_message = str(error)
        super().__init__(f"{stage}: {self.error_type}: {self.error_message}")


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
    unknown_bond_count: int
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
    try:
        mol = _load_molecule(case)
    except (OSError, ValueError, StopIteration) as error:
        raise DryCaseError("source", error) from error
    parse_seconds = perf_counter() - parse_started

    facts_started = perf_counter()
    try:
        facts = extract_molecular_facts(mol)
    except ValueError as error:
        raise DryCaseError("facts", error) from error
    facts_seconds = perf_counter() - facts_started

    manifest = _coordination_screening_manifest()
    if manifest.activation_status != "planned":
        raise RuntimeError(
            "The pre-activation benchmark requires a planned profile artifact"
        )

    tracemalloc.start()
    try:
        resolve_started = perf_counter()
        try:
            bundle = resolve_scientific_state(
                facts,
                policy=manifest.policy,
                provider_set=manifest.provider_set,
                rule_set=manifest.rule_set,
                candidate_budget=manifest.candidate_budget,
            )
        except ValueError as error:
            raise DryCaseError("resolution", error) from error
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
        unknown_bond_count=sum(
            bond.kind == "unknown" for bond in facts.semantic_bonds
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


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_record(stream: TextIO, payload: dict[str, object]) -> None:
    stream.write(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )


def _increment(counter: Counter[str], value: str) -> None:
    counter[value] += 1


def _case_failure_payload(
    case: DryStateCase,
    error: DryCaseError,
) -> dict[str, object]:
    return {
        "status": "source_or_resolution_error",
        "index": case.index,
        "case_id": case.case_id,
        "source_kind": case.source_kind.value,
        "format": case.format,
        "source_sha256": case.source_sha256,
        "failure_stage": error.stage,
        "error_type": error.error_type,
        "error_message": error.error_message,
    }


def _manifest_payload(corpus: DryStateCorpus, replay: bool) -> dict[str, object]:
    manifest = _coordination_screening_manifest()
    return {
        "schema_version": _SCHEMA_VERSION,
        "benchmark": "scientific_state_pre_activation_baseline",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "corpus": {
            "source": str(corpus.source),
            "source_sha256": corpus.source_sha256,
            "case_count": len(corpus.cases),
        },
        "profile": {
            "profile_id": manifest.profile_id,
            "profile_version": manifest.profile_version,
            "activation_status": manifest.activation_status,
            "content_digest": manifest.profile_content_digest,
        },
        "replay_enabled": replay,
        "memory_metric": "python_tracemalloc_peak_bytes",
        "memory_scope": "Python allocations only; native allocations are excluded",
    }


def _fraction(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def run_benchmark(
    corpus: DryStateCorpus,
    output_directory: Path,
    *,
    replay: bool = True,
) -> dict[str, object]:
    """Stream one pre-activation baseline and optional deterministic replay."""

    output_directory.mkdir(parents=True, exist_ok=True)
    manifest = _manifest_payload(corpus, replay)
    _write_json(output_directory / "manifest.json", manifest)

    action_counts: Counter[str] = Counter()
    decision_source_counts: Counter[str] = Counter()
    certification_counts: Counter[str] = Counter()
    search_status_counts: Counter[str] = Counter()
    format_counts: Counter[str] = Counter()
    coordination_record_counts: Counter[str] = Counter()
    unknown_bond_record_counts: Counter[str] = Counter()
    profile_finding_counts: Counter[str] = Counter()
    resolved_count = 0
    failed_count = 0
    truncated_count = 0
    selected_count = 0
    no_automatic_state_count = 0
    static_ready_count = 0
    replay_checked_count = 0
    replay_mismatch_count = 0
    expanded_path_total = 0
    expanded_path_maximum = 0
    resolve_seconds_total = 0.0
    python_peak_bytes = 0
    started = perf_counter()
    records_path = output_directory / "records.jsonl"
    temporary_records_path = records_path.with_suffix(".jsonl.tmp")
    with temporary_records_path.open("w", encoding="utf-8") as stream:
        for case in corpus.cases:
            try:
                record = resolve_case(case)
            except DryCaseError as error:
                failed_count += 1
                _write_record(stream, _case_failure_payload(case, error))
                continue

            payload = record.to_payload()
            payload["status"] = "resolved"
            resolved_count += 1
            _increment(action_counts, record.decision_action)
            _increment(decision_source_counts, record.decision_source)
            _increment(certification_counts, record.certification_status)
            _increment(search_status_counts, record.search_status)
            _increment(format_counts, record.format)
            if record.coordination_bond_count:
                _increment(coordination_record_counts, record.format)
            if record.unknown_bond_count:
                _increment(unknown_bond_record_counts, record.format)
            for finding_code in record.profile_finding_codes:
                _increment(profile_finding_counts, finding_code)
            truncated_count += int(record.truncated)
            selected_count += int(record.decision_action == "select_state")
            no_automatic_state_count += int(
                record.decision_action == "no_automatic_state"
            )
            static_ready_count += int(
                record.static_profile_ready_except_activation
            )
            expanded_path_total += record.expanded_path_count
            expanded_path_maximum = max(
                expanded_path_maximum,
                record.expanded_path_count,
            )
            resolve_seconds_total += record.resolve_seconds
            python_peak_bytes = max(
                python_peak_bytes,
                record.python_peak_bytes,
            )

            if replay:
                replay_checked_count += 1
                try:
                    replay_record = resolve_case(case)
                except DryCaseError as error:
                    replay_mismatch_count += 1
                    payload["replay"] = {
                        "match": False,
                        "failure_stage": error.stage,
                        "error_type": error.error_type,
                        "error_message": error.error_message,
                    }
                else:
                    replay_matches = (
                        replay_record.facts_identity == record.facts_identity
                        and replay_record.snapshot_sha256
                        == record.snapshot_sha256
                    )
                    replay_mismatch_count += int(not replay_matches)
                    payload["replay"] = {
                        "match": replay_matches,
                        "facts_identity": replay_record.facts_identity,
                        "snapshot_sha256": replay_record.snapshot_sha256,
                    }
            _write_record(stream, payload)
    temporary_records_path.replace(records_path)

    total_seconds = perf_counter() - started
    summary = {
        "schema_version": _SCHEMA_VERSION,
        "benchmark": "scientific_state_pre_activation_baseline",
        "requested_count": len(corpus.cases),
        "resolved_count": resolved_count,
        "failure_count": failed_count,
        "total_seconds": total_seconds,
        "records_per_second": _fraction(resolved_count, total_seconds),
        "resolve_seconds_total": resolve_seconds_total,
        "resolve_only_records_per_second": _fraction(
            resolved_count,
            resolve_seconds_total,
        ),
        "python_peak_bytes": python_peak_bytes,
        "mean_expanded_paths": _fraction(
            expanded_path_total,
            resolved_count,
        ),
        "max_expanded_paths": expanded_path_maximum,
        "truncated_count": truncated_count,
        "truncated_fraction": _fraction(truncated_count, resolved_count),
        "automatic_selection_count": selected_count,
        "automatic_selection_fraction": _fraction(
            selected_count,
            resolved_count,
        ),
        "no_automatic_state_count": no_automatic_state_count,
        "no_automatic_state_fraction": _fraction(
            no_automatic_state_count,
            resolved_count,
        ),
        "static_profile_ready_except_activation_count": static_ready_count,
        "replay_checked_count": replay_checked_count,
        "replay_mismatch_count": replay_mismatch_count,
        "decision_action_counts": dict(sorted(action_counts.items())),
        "decision_source_counts": dict(sorted(decision_source_counts.items())),
        "certification_status_counts": dict(sorted(certification_counts.items())),
        "search_status_counts": dict(sorted(search_status_counts.items())),
        "format_counts": dict(sorted(format_counts.items())),
        "topology_by_format": {
            format_name: {
                "record_count": count,
                "with_coordination_bonds": coordination_record_counts[
                    format_name
                ],
                "with_unknown_bonds": unknown_bond_record_counts[format_name],
            }
            for format_name, count in sorted(format_counts.items())
        },
        "profile_finding_counts": dict(sorted(profile_finding_counts.items())),
        "activation_coverage_is_final": False,
    }
    _write_json(output_directory / "summary.json", summary)
    return summary
