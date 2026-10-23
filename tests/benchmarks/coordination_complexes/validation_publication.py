"""Bridge a completed Plan-17 run into governed dataset publication records.

The bridge deliberately sits after gallery review and before
``validation_dataset.prepare_bundle``.  It selects only the five primary runs
declared by the run manifest, validates the independently completed gallery
review, and emits the exact 182 x 5 record matrix consumed by the dataset
publisher.  The full F1 repeat remains control evidence and can never become a
published coordinate source.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

from .validation_acceptance import GATE_IDS
from .validation_cohort import load_validation_cohort
from .validation_dataset import (
    FORMAL_CASE_INDICES,
    ROUTES,
    prepare_bundle,
    validate_route_results,
)
from .validation_gallery import validate_manual_review_artifact
from .validation_providers import (
    load_frozen_proposals,
    load_mapped_coordinate_proposal,
)
from .validation_runner import (
    _proposal_artifact_identity,
    _tree_identity,
)

RUN_MANIFEST_SCHEMA = "hotpot-plan17-validation-run/v1"
RUN_RESULT_SCHEMA = "hotpot-plan17-automated-route-result/v1"
CONVERSION_SCHEMA = "hotpot-plan17-publication-conversion/v1"
RESULTS_FILENAME = "route-results.jsonl"
MANIFEST_FILENAME = "conversion-manifest.json"
MANIFEST_DIGEST_FILENAME = "conversion-manifest.sha256"
EVIDENCE_SCHEMA = "hotpot-plan17-publication-evidence/v1"
EVIDENCE_INDEX = "evidence/publication/evidence-index.json"
_SAFE_SEGMENT = re.compile(r"[A-Za-z0-9._-]+")


@dataclass(frozen=True)
class PublicationConversion:
    """Validated publication records and their conversion manifest."""

    records: tuple[dict[str, object], ...]
    manifest: dict[str, object]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _load_object(path: Path, label: str) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _artifact(path: Path) -> dict[str, object]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size": path.stat().st_size,
    }


def _safe_segment(value: object, label: str) -> str:
    if not isinstance(value, str) or _SAFE_SEGMENT.fullmatch(value) is None:
        raise ValueError(f"{label} is not a safe path segment")
    return value


def _run_contract(
    manifest: Mapping[str, object],
) -> tuple[
    Mapping[str, object],
    dict[str, Mapping[str, object]],
    Mapping[str, object],
]:
    if manifest.get("schema") != RUN_MANIFEST_SCHEMA:
        raise ValueError("validation run manifest schema is unsupported")
    configuration = _require_mapping(
        manifest.get("scientific_configuration"),
        "scientific_configuration",
    )
    if configuration.get("formal") is not True:
        raise ValueError("publication requires a formal validation run")
    if tuple(configuration.get("case_indices", ())) != FORMAL_CASE_INDICES:
        raise ValueError("publication requires the exact 182-case formal cohort")
    identity = _require_mapping(configuration.get("identity"), "run identity")
    raw_specs = configuration.get("run_specs")
    if not isinstance(raw_specs, list):
        raise ValueError("run_specs must be an array")
    if any(not isinstance(spec, Mapping) for spec in raw_specs):
        raise ValueError("every run spec must be an object")
    run_ids = [_safe_segment(spec.get("run_id"), "run_id") for spec in raw_specs]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("primary and repeat run_id values must be globally unique")

    primary_specs = [
        _require_mapping(spec, "primary run spec")
        for spec in raw_specs
        if spec.get("repeat_of") is None
    ]
    repeat_specs = [
        _require_mapping(spec, "repeat run spec")
        for spec in raw_specs
        if spec.get("repeat_of") is not None
    ]
    if len(primary_specs) != len(ROUTES):
        raise ValueError("run manifest must declare five primary runs")
    primary: dict[str, Mapping[str, object]] = {}
    for spec in primary_specs:
        route = spec.get("route")
        if route not in ROUTES or route in primary:
            raise ValueError("primary runs must uniquely cover O0/F1/F2/C1/C2")
        _safe_segment(spec.get("run_id"), f"{route} run_id")
        primary[str(route)] = spec
    if tuple(primary) != ROUTES:
        raise ValueError("primary runs must be ordered O0/F1/F2/C1/C2")

    if len(repeat_specs) != 1:
        raise ValueError("run manifest must contain exactly one F1 repeat control")
    repeat = repeat_specs[0]
    if repeat.get("route") != "F1" or repeat.get("repeat_of") != primary["F1"].get(
        "run_id"
    ):
        raise ValueError("F1 repeat control must point to the primary F1 run")
    _safe_segment(repeat.get("run_id"), "F1 repeat run_id")
    return identity, primary, repeat


def _result_path(
    run_root: Path,
    spec: Mapping[str, object],
    case_index: int,
) -> Path:
    run_id = _safe_segment(spec.get("run_id"), "run_id")
    route = str(spec["route"])
    return (
        run_root
        / "runs"
        / run_id
        / route
        / "cases"
        / f"{case_index:04d}"
        / "result.json"
    )


def _gate_decisions(record: Mapping[str, object]) -> dict[str, bool]:
    acceptance = _require_mapping(record.get("acceptance"), "acceptance")
    gates = acceptance.get("gates")
    if not isinstance(gates, list):
        raise ValueError("acceptance.gates must be an array")
    decisions: dict[str, bool] = {}
    for gate in gates:
        gate = _require_mapping(gate, "acceptance gate")
        gate_id = gate.get("gate_id")
        passed = gate.get("passed")
        if gate_id not in GATE_IDS or gate_id in decisions or type(passed) is not bool:
            raise ValueError("acceptance must contain unique Boolean A01-A10 gates")
        decisions[str(gate_id)] = bool(passed)
    if tuple(decisions) != GATE_IDS:
        raise ValueError("acceptance gates must be ordered A01-A10")
    if type(acceptance.get("passed")) is not bool:
        raise ValueError("acceptance.passed must be Boolean")
    if bool(acceptance["passed"]) != all(decisions.values()):
        raise ValueError("acceptance.passed differs from A01-A10 decisions")
    if bool(record.get("passed")) != bool(acceptance["passed"]):
        raise ValueError("runner passed status differs from acceptance")
    return decisions


def _recovery_actions(evidence: Optional[Mapping[str, object]]) -> list[str]:
    if evidence is None:
        return []
    actions: list[str] = []
    events = evidence.get("unavailable_gradient_events", [])
    if not isinstance(events, list):
        raise ValueError("unavailable_gradient_events must be an array")
    for event in events:
        event = _require_mapping(event, "unavailable-gradient event")
        recovery = event.get("recovery")
        if recovery is None:
            continue
        recovery = _require_mapping(recovery, "gradient recovery")
        action = recovery.get("action_name")
        if isinstance(action, str) and action and action not in actions:
            actions.append(action)
    trace = _require_mapping(evidence.get("stage_trace"), "stage_trace")
    stages = trace.get("records")
    if not isinstance(stages, list):
        raise ValueError("stage_trace.records must be an array")
    for stage in stages:
        stage = _require_mapping(stage, "stage trace record")
        stage_actions = stage.get("recovery_actions")
        if not isinstance(stage_actions, list):
            raise ValueError("stage recovery_actions must be an array")
        for action in stage_actions:
            if not isinstance(action, str) or not action:
                raise ValueError("recovery action names must be non-empty strings")
            if action not in actions:
                actions.append(action)
    return actions


def _selected_structure(
    record: Mapping[str, object],
    *,
    result_path: Path,
    run_root: Path,
    artifact_result_path: Optional[Path] = None,
    artifact_run_root: Optional[Path] = None,
) -> Optional[dict[str, object]]:
    selected = record.get("selected_structure")
    if record.get("passed") is not True:
        if selected is not None:
            raise ValueError("failed runner result must not expose selected_structure")
        return None
    selected = _require_mapping(selected, "selected_structure")
    path = Path(str(selected.get("path", ""))).expanduser().resolve()
    expected = (result_path.parent / "selected.sdf").resolve()
    try:
        expected.relative_to(run_root.resolve())
    except ValueError as error:
        raise ValueError(
            "selected structure escapes the validation run root"
        ) from error
    if path != expected:
        raise ValueError(
            "selected structure must be the selected.sdf beside its result.json"
        )
    artifact_path = (
        path
        if artifact_result_path is None
        else (artifact_result_path.parent / "selected.sdf").resolve()
    )
    artifact_root = (
        run_root.resolve()
        if artifact_run_root is None
        else artifact_run_root.resolve()
    )
    try:
        artifact_path.relative_to(artifact_root)
    except ValueError as error:
        raise ValueError(
            "selected structure artifact escapes the validation run root"
        ) from error
    digest = selected.get("sha256")
    if not artifact_path.is_file():
        raise FileNotFoundError(artifact_path)
    if not isinstance(digest, str) or _sha256(artifact_path) != digest:
        raise ValueError(f"selected structure SHA-256 mismatch: {artifact_path}")
    atom_mapping = selected.get("atom_row_mapping")
    if not isinstance(atom_mapping, list):
        raise ValueError("selected structure atom_row_mapping must be an array")
    return {
        "path": str(artifact_path),
        "sha256": digest,
        "format": selected.get("format"),
        "atom_row_mapping": list(atom_mapping),
    }


def _proposal_artifact(
    record: Mapping[str, object],
    *,
    route: str,
    declared_case_root: Path,
    artifact_case_root: Optional[Path] = None,
) -> Optional[dict[str, object]]:
    if route not in {"F2", "C1", "C2"}:
        return None
    artifact_case_root = (
        declared_case_root
        if artifact_case_root is None
        else artifact_case_root
    )
    filename = (
        "frozen-proposals.json"
        if route == "F2"
        else "whole-complex-proposal.json"
    )
    declared_candidate = declared_case_root / filename
    artifact_candidate = artifact_case_root / filename
    if declared_candidate.is_symlink() or artifact_candidate.is_symlink():
        raise ValueError(f"{route} proposal artifact must not be a symlink")
    declared_path = declared_candidate.resolve()
    artifact_path = artifact_candidate.resolve()
    for path, root in (
        (declared_path, declared_case_root.resolve()),
        (artifact_path, artifact_case_root.resolve()),
    ):
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"{route} proposal artifact escapes its case") from error
    proposal = record.get("proposal")
    archived = (
        None
        if proposal is None
        else _require_mapping(proposal, "proposal").get("archived_artifact")
    )
    if archived is None:
        if artifact_path.exists():
            raise ValueError(f"{route} proposal artifact exists but is undeclared")
        if record.get("status") != "execution_failed":
            raise ValueError(f"{route} completed result omits its proposal artifact")
        return None

    archived = _require_mapping(archived, "archived proposal artifact")
    if Path(str(archived.get("path", ""))).expanduser().resolve() != declared_path:
        raise ValueError(f"{route} proposal artifact is not beside result.json")
    actual = _proposal_artifact_identity(artifact_path)
    if actual is None or actual.get("valid_json") is not True:
        raise ValueError(f"{route} proposal artifact is unavailable or invalid JSON")
    if route == "F2":
        load_frozen_proposals(artifact_path)
        route_fields = ("proposal_count",)
    else:
        loaded = load_mapped_coordinate_proposal(artifact_path)
        if loaded.get("route") != route:
            raise ValueError(f"{route} proposal artifact route differs")
        route_fields = ("atom_count", "coordinate_sha256")
    for field in ("sha256", "canonical_sha256", *route_fields):
        if archived.get(field) != actual.get(field):
            raise ValueError(f"{route} proposal artifact {field} differs")

    declared = _require_mapping(proposal, "proposal").get("declared_artifact")
    if declared is not None:
        declared = _require_mapping(declared, "declared proposal artifact")
        if Path(str(declared.get("path", ""))).expanduser().resolve() != declared_path:
            raise ValueError(f"{route} declared proposal is not beside result.json")
        for field in ("sha256", "canonical_sha256", *route_fields):
            if declared.get(field) != actual.get(field):
                raise ValueError(f"{route} declared proposal {field} differs")
    return actual


def _trajectory_artifact(
    record: Mapping[str, object],
    *,
    declared_case_root: Path,
    artifact_case_root: Optional[Path] = None,
) -> Optional[dict[str, object]]:
    artifact_case_root = (
        declared_case_root
        if artifact_case_root is None
        else artifact_case_root
    )
    declared_candidate = declared_case_root / "trajectory"
    artifact_candidate = artifact_case_root / "trajectory"
    if declared_candidate.is_symlink() or artifact_candidate.is_symlink():
        raise ValueError("trajectory artifact must not be a symlink")
    declared_path = declared_candidate.resolve()
    artifact_path = artifact_candidate.resolve()
    declared = record.get("trajectory_artifact")
    if declared is None:
        if artifact_path.exists():
            raise ValueError("trajectory artifact exists but is undeclared")
        return None
    declared = _require_mapping(declared, "trajectory_artifact")
    if Path(str(declared.get("path", ""))).expanduser().resolve() != declared_path:
        raise ValueError("trajectory artifact is not beside result.json")
    if not artifact_path.is_dir():
        raise FileNotFoundError(artifact_path)
    if any(path.is_symlink() for path in artifact_path.rglob("*")):
        raise ValueError("trajectory artifact must not contain symlinks")
    actual = {
        **_tree_identity(artifact_path),
        "includes_sdf": any(
            path.is_file() and path.suffix.lower() == ".sdf"
            for path in artifact_path.rglob("*")
        ),
    }
    for field in ("sha256", "file_count", "total_bytes", "includes_sdf"):
        if declared.get(field) != actual[field]:
            raise ValueError(f"trajectory artifact {field} differs")
    return actual


def _validate_case_artifacts(
    record: Mapping[str, object],
    *,
    route: str,
    declared_result_path: Path,
    artifact_result_path: Optional[Path] = None,
) -> None:
    artifact_result_path = (
        declared_result_path
        if artifact_result_path is None
        else artifact_result_path
    )
    _proposal_artifact(
        record,
        route=route,
        declared_case_root=declared_result_path.parent,
        artifact_case_root=artifact_result_path.parent,
    )
    _trajectory_artifact(
        record,
        declared_case_root=declared_result_path.parent,
        artifact_case_root=artifact_result_path.parent,
    )


def _validate_runner_record(
    record: Mapping[str, object],
    *,
    expected_identity: Mapping[str, object],
    spec: Mapping[str, object],
    case: Mapping[str, object],
    result_path: Path,
    run_root: Path,
) -> tuple[dict[str, bool], Optional[dict[str, object]]]:
    if record.get("schema") != RUN_RESULT_SCHEMA:
        raise ValueError("runner result schema is unsupported")
    expected = {
        "route": spec["route"],
        "run_id": spec["run_id"],
        "case_index": case["index"],
        "source_smiles": case["smiles"],
        "semantic_graph_sha256": case["semantic_graph_sha256"],
        "seed": case["seed"],
    }
    if any(record.get(field) != value for field, value in expected.items()):
        raise ValueError(
            f"runner result identity differs for {spec['route']} case {case['index']}"
        )
    if record.get("identity") != expected_identity:
        raise ValueError(
            "runner result runtime/git/cohort identity differs from manifest"
        )
    if type(record.get("passed")) is not bool:
        raise ValueError("runner passed must be Boolean")
    if not isinstance(record.get("status"), str) or not record["status"]:
        raise ValueError("runner status must be non-empty")
    decisions = _gate_decisions(record)
    _validate_case_artifacts(
        record,
        route=str(spec["route"]),
        declared_result_path=result_path,
    )
    selected = _selected_structure(
        record,
        result_path=result_path,
        run_root=run_root,
    )
    return decisions, selected


def _dependency_versions(runtime: Mapping[str, object]) -> dict[str, str]:
    fields = {
        "hotpot": "imported_hotpot_version",
        "rdkit-native": "rdkit_version",
        "rdkit-python": "python_rdkit_version",
        "rdkit-python-distribution": "python_rdkit_distribution_version",
        "openbabel": "openbabel_version",
        "python": "python_version",
    }
    result = {name: runtime.get(field) for name, field in fields.items()}
    if any(not isinstance(value, str) or not value for value in result.values()):
        raise ValueError("run identity omits a required dependency version")
    return {name: str(value) for name, value in result.items()}


def _control(name: str, artifact: Mapping[str, object]) -> dict[str, object]:
    return {
        "name": name,
        "sha256": artifact["sha256"],
        "locator": artifact["path"],
    }


def _review_index(
    review: Mapping[str, object],
) -> dict[tuple[str, int], Mapping[str, object]]:
    rows = review.get("reviews")
    if not isinstance(rows, list):
        raise ValueError("manual review rows must be an array")
    return {
        (str(row["route"]), int(row["case_index"])): row
        for row in rows
        if isinstance(row, Mapping)
    }


def _gallery_index(
    gallery: Mapping[str, object],
) -> dict[tuple[str, int], Mapping[str, object]]:
    if tuple(gallery.get("case_indices", ())) != FORMAL_CASE_INDICES:
        raise ValueError("gallery does not cover the exact 182-case formal cohort")
    cases = gallery.get("cases")
    if not isinstance(cases, list):
        raise ValueError("gallery cases must be an array")
    return {
        (str(case["route"]), int(case["case_index"])): case
        for case in cases
        if isinstance(case, Mapping)
    }


def _cohort_cases(cohort: Mapping[str, object]) -> dict[int, Mapping[str, object]]:
    cases = cohort.get("cases")
    if not isinstance(cases, list):
        raise ValueError("cohort cases must be an array")
    selected = {
        int(case["index"]): case
        for case in cases
        if isinstance(case, Mapping) and case.get("status") == "cbond"
    }
    if tuple(selected) != FORMAL_CASE_INDICES:
        raise ValueError("cohort does not contain the exact 182 positive cases")
    return selected


def build_publication_conversion(
    run_root: Path,
    gallery_manifest: Path,
    manual_review_artifact: Path,
    *,
    cohort_path: Optional[Path] = None,
    source_path: Optional[Path] = None,
) -> PublicationConversion:
    """Validate evidence and construct the exact primary publication matrix."""

    run_root = run_root.resolve()
    run_manifest_path = run_root / "manifest.json"
    gallery_manifest = gallery_manifest.resolve()
    manual_review_artifact = manual_review_artifact.resolve()
    run_manifest = _load_object(run_manifest_path, "run manifest")
    identity, primary, repeat = _run_contract(run_manifest)

    gallery = _load_object(gallery_manifest, "gallery manifest")
    review_value = _load_object(manual_review_artifact, "manual review")
    review = validate_manual_review_artifact(review_value, gallery_manifest)
    gallery_by_case = _gallery_index(gallery)
    review_by_case = _review_index(review)

    cohort_identity = _require_mapping(identity.get("cohort"), "cohort identity")
    declared_cohort = Path(str(cohort_identity.get("path", ""))).expanduser()
    selected_cohort_path = (
        declared_cohort if cohort_path is None else cohort_path.expanduser()
    ).resolve()
    cohort = load_validation_cohort(selected_cohort_path)
    if cohort.get("canonical_sha256") != cohort_identity.get("canonical_sha256"):
        raise ValueError("cohort canonical SHA-256 differs from run manifest")
    cohort_by_case = _cohort_cases(cohort)

    source_identity = _require_mapping(identity.get("source"), "source identity")
    declared_source = Path(str(source_identity.get("path", ""))).expanduser()
    selected_source_path = (
        declared_source if source_path is None else source_path.expanduser()
    ).resolve()
    source_artifact = _artifact(selected_source_path)
    if source_artifact["sha256"] != source_identity.get("sha256"):
        raise ValueError("source SHA-256 differs from run manifest")

    manifest_artifact = _artifact(run_manifest_path)
    cohort_artifact = _artifact(selected_cohort_path)
    gallery_artifact = _artifact(gallery_manifest)
    review_artifact = _artifact(manual_review_artifact)
    base_controls = [
        _control("validation-run-manifest", manifest_artifact),
        _control("semantic-cohort", cohort_artifact),
        _control("extractant-source", source_artifact),
        _control("validation-gallery", gallery_artifact),
        _control("manual-review", review_artifact),
    ]

    runtime = _require_mapping(identity.get("runtime"), "runtime identity")
    git = _require_mapping(identity.get("git"), "git identity")
    hotpot_commit = git.get("commit")
    if not isinstance(hotpot_commit, str) or not hotpot_commit:
        raise ValueError("run identity omits the Hotpot Git commit")
    dependencies = _dependency_versions(runtime)
    scientific_settings = _require_mapping(
        identity.get("settings"),
        "scientific settings",
    )

    repeat_rows: dict[int, tuple[Mapping[str, object], Mapping[str, object]]] = {}
    repeat_artifacts: dict[int, dict[str, object]] = {}
    primary_artifacts: dict[tuple[str, int], dict[str, object]] = {}
    primary_records: dict[tuple[str, int], Mapping[str, object]] = {}
    primary_decisions: dict[tuple[str, int], dict[str, bool]] = {}
    primary_selected: dict[tuple[str, int], Optional[dict[str, object]]] = {}

    for route in ROUTES:
        spec = primary[route]
        for case_index in FORMAL_CASE_INDICES:
            case = cohort_by_case[case_index]
            path = _result_path(run_root, spec, case_index)
            record = _load_object(path, "primary runner result")
            decisions, selected = _validate_runner_record(
                record,
                expected_identity=identity,
                spec=spec,
                case=case,
                result_path=path,
                run_root=run_root,
            )
            key = (route, case_index)
            primary_records[key] = record
            primary_decisions[key] = decisions
            primary_selected[key] = selected
            primary_artifacts[key] = _artifact(path)

    for case_index in FORMAL_CASE_INDICES:
        case = cohort_by_case[case_index]
        path = _result_path(run_root, repeat, case_index)
        repeated = _load_object(path, "F1 repeat runner result")
        _validate_runner_record(
            repeated,
            expected_identity=identity,
            spec=repeat,
            case=case,
            result_path=path,
            run_root=run_root,
        )
        repeat_rows[case_index] = (primary_records[("F1", case_index)], repeated)
        repeat_artifacts[case_index] = _artifact(path)

    expected_keys = {
        (route, case_index) for route in ROUTES for case_index in FORMAL_CASE_INDICES
    }
    if set(gallery_by_case) != expected_keys or set(review_by_case) != expected_keys:
        raise ValueError("gallery/manual review matrix differs from primary run matrix")

    records = []
    for case_index in FORMAL_CASE_INDICES:
        case = cohort_by_case[case_index]
        coordination = _require_mapping(case.get("coordination"), "coordination")
        metal = _require_mapping(coordination.get("metal"), "coordination metal")
        donor_rows = coordination.get("donor_rows")
        if not isinstance(donor_rows, list):
            raise ValueError("coordination donor_rows must be an array")
        for route in ROUTES:
            key = (route, case_index)
            runner_record = primary_records[key]
            selected = primary_selected[key]
            gallery_case = gallery_by_case[key]
            review_row = review_by_case[key]
            expected_selected_digest = None if selected is None else selected["sha256"]
            gallery_identity = {
                "route": route,
                "run_id": primary[route]["run_id"],
                "case_index": case_index,
                "status": runner_record["status"],
                "passed": runner_record["passed"],
                "selected_structure_sha256": expected_selected_digest,
            }
            if any(
                gallery_case.get(field) != value
                for field, value in gallery_identity.items()
            ):
                raise ValueError(f"gallery differs from primary result {key}")

            flagged = bool(review_row["flagged"])
            final_passed = bool(runner_record["passed"]) and not flagged
            final_selected = selected if final_passed else None
            flags = (
                []
                if not flagged
                else [
                    {
                        "source": "exhaustive_gallery_review",
                        "reason": review_row["reason"],
                        "gallery_image_sha256": review_row["gallery_image_sha256"],
                    }
                ]
            )
            acceptance = _require_mapping(
                runner_record["acceptance"],
                "acceptance",
            )
            raw_evidence = runner_record.get("evidence")
            evidence = (
                None
                if raw_evidence is None
                else _require_mapping(raw_evidence, "evidence")
            )
            controls = [
                *base_controls,
                _control("primary-runner-result", primary_artifacts[key]),
            ]
            if route == "F1":
                controls.append(
                    _control("f1-repeat-control-result", repeat_artifacts[case_index])
                )
            record: dict[str, object] = {
                "case_index": case_index,
                "source_smiles": case["smiles"],
                "source_sha256": source_identity["sha256"],
                "semantic_graph_sha256": case["semantic_graph_sha256"],
                "route": route,
                "run_id": primary[route]["run_id"],
                "seed": case["seed"],
                "status": (
                    "passed"
                    if final_passed
                    else "manual_review_failed"
                    if flagged
                    else runner_record["status"]
                ),
                "passed": final_passed,
                "acceptance": {
                    **primary_decisions[key],
                    "diagnostics": list(acceptance.get("diagnostics", [])),
                    "recovery_actions": _recovery_actions(evidence),
                },
                "optimization": dict(
                    _require_mapping(
                        runner_record.get("optimization"),
                        "selected optimization",
                    )
                ),
                "provenance": {
                    "validation_round": gallery["validation_revision"],
                    "hotpot_commit": hotpot_commit,
                    "dependency_versions": dependencies,
                    "controls": controls,
                },
                "settings": {
                    **dict(scientific_settings),
                    "cbond": {
                        "metal_index": metal["row"],
                        "donor_indices": list(donor_rows),
                    },
                    "validation": {
                        "source_identity": dict(source_identity),
                        "cohort_identity": dict(cohort_identity),
                        "git_identity": dict(git),
                        "runtime_identity": dict(runtime),
                        "proposal": runner_record.get("proposal"),
                        "evidence": None if evidence is None else dict(evidence),
                        "trajectory_artifact": runner_record.get("trajectory_artifact"),
                        "timing": runner_record.get("timing"),
                        "resources": runner_record.get("resources"),
                        "runner_terminal_status": runner_record["status"],
                    },
                },
                "manual_review": {
                    "status": "failed" if flagged else "passed",
                    "reviewer": review["reviewer"],
                    "completed_at": review["completed_at"],
                    "flags": flags,
                },
            }
            if final_selected is not None:
                record["selected_structure"] = final_selected
            records.append(record)

    ordered = validate_route_results(
        records,
        selected_source_path,
        case_indices=FORMAL_CASE_INDICES,
        expected_source_sha256=str(source_identity["sha256"]),
    )
    records_bytes = b"".join(_canonical_bytes(record) for record in ordered)
    repeat_matrix = [
        {
            "case_index": case_index,
            "primary_passed": bool(primary_record["passed"]),
            "repeat_passed": bool(repeated["passed"]),
            "primary_gates": primary_decisions[("F1", case_index)],
            "repeat_gates": _gate_decisions(repeated),
            "repeat_result_sha256": repeat_artifacts[case_index]["sha256"],
        }
        for case_index, (primary_record, repeated) in repeat_rows.items()
    ]
    pass_mismatches = [
        row["case_index"]
        for row in repeat_matrix
        if row["primary_passed"] != row["repeat_passed"]
    ]
    gate_mismatches = [
        row["case_index"]
        for row in repeat_matrix
        if row["primary_gates"] != row["repeat_gates"]
    ]
    manifest = {
        "schema": CONVERSION_SCHEMA,
        "inputs": {
            "validation_run_manifest": manifest_artifact,
            "semantic_cohort": {
                **cohort_artifact,
                "canonical_sha256": cohort["canonical_sha256"],
            },
            "extractant_source": source_artifact,
            "gallery_manifest": gallery_artifact,
            "manual_review": review_artifact,
        },
        "primary_runs": [dict(primary[route]) for route in ROUTES],
        "f1_repeat_control": {
            "run_id": repeat["run_id"],
            "repeat_of": repeat["repeat_of"],
            "case_count": len(repeat_matrix),
            "matrix_sha256": hashlib.sha256(
                _canonical_bytes(repeat_matrix)
            ).hexdigest(),
            "acceptance_classification_consistent": not pass_mismatches,
            "acceptance_mismatch_case_indices": pass_mismatches,
            "gate_classification_consistent": not gate_mismatches,
            "gate_mismatch_case_indices": gate_mismatches,
        },
        "publication_matrix": {
            "case_count": len(FORMAL_CASE_INDICES),
            "route_count": len(ROUTES),
            "record_count": len(ordered),
            "passed_count": sum(bool(record["passed"]) for record in ordered),
            "failed_count": sum(not bool(record["passed"]) for record in ordered),
            "manual_flagged_count": int(review["summary"]["flagged_case_count"]),
            "selected_structure_count": sum(
                "selected_structure" in record for record in ordered
            ),
            "records_path": RESULTS_FILENAME,
            "records_sha256": hashlib.sha256(records_bytes).hexdigest(),
            "records_size": len(records_bytes),
        },
    }
    return PublicationConversion(records=ordered, manifest=manifest)


def write_publication_conversion(
    run_root: Path,
    gallery_manifest: Path,
    manual_review_artifact: Path,
    output_root: Path,
    *,
    cohort_path: Optional[Path] = None,
    source_path: Optional[Path] = None,
) -> dict[str, object]:
    """Atomically write validated publication records and conversion identity."""

    conversion = build_publication_conversion(
        run_root,
        gallery_manifest,
        manual_review_artifact,
        cohort_path=cohort_path,
        source_path=source_path,
    )
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    try:
        records_path = staging / RESULTS_FILENAME
        records_path.write_bytes(
            b"".join(_canonical_bytes(record) for record in conversion.records)
        )
        manifest_path = staging / MANIFEST_FILENAME
        manifest_path.write_bytes(_canonical_bytes(conversion.manifest))
        manifest_sha256 = _sha256(manifest_path)
        (staging / MANIFEST_DIGEST_FILENAME).write_text(
            f"{manifest_sha256}  {MANIFEST_FILENAME}\n",
            encoding="utf-8",
        )
        os.replace(staging, output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "output_root": str(output_root),
        "records_path": str(output_root / RESULTS_FILENAME),
        "records_sha256": conversion.manifest["publication_matrix"]["records_sha256"],
        "manifest_path": str(output_root / MANIFEST_FILENAME),
        "manifest_sha256": manifest_sha256,
    }


def _copy_evidence_tree(source: Path, destination: Path) -> None:
    for path in source.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"publication evidence must not contain symlinks: {path}")
    shutil.copytree(source, destination)


def _publication_evidence_index(bundle_root: Path) -> dict[str, object]:
    evidence_root = bundle_root / "evidence" / "publication"
    assets = [
        {
            "path": path.relative_to(bundle_root).as_posix(),
            "sha256": _sha256(path),
            "size": path.stat().st_size,
        }
        for path in sorted(evidence_root.rglob("*"))
        if path.is_file()
    ]
    return {
        "schema": EVIDENCE_SCHEMA,
        "assets": assets,
        "asset_count": len(assets),
        "aggregate_sha256": hashlib.sha256(_canonical_bytes(assets)).hexdigest(),
    }


def _bundle_evidence_locator(
    locator: object,
    *,
    run_root: Path,
    gallery_root: Path,
    cohort_path: Path,
    source_path: Path,
    manual_review_path: Path,
) -> str:
    raw_path = Path(str(locator)).expanduser()
    if not raw_path.is_absolute():
        if (
            raw_path.parts[:2] == ("evidence", "publication")
            and ".." not in raw_path.parts
        ):
            return raw_path.as_posix()
        raise ValueError(
            f"publication control has an invalid relative locator: {raw_path}"
        )
    path = raw_path.resolve()
    fixed = {
        cohort_path: Path("evidence/publication/cohort.json"),
        source_path: Path("evidence/publication/extractants.smi"),
        manual_review_path: Path("evidence/publication/manual-review.json"),
    }
    if path in fixed:
        return fixed[path].as_posix()
    for root, destination in (
        (run_root, Path("evidence/publication/validation-run")),
        (gallery_root, Path("evidence/publication/validation-gallery")),
    ):
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        return (destination / relative).as_posix()
    raise ValueError(f"publication control is outside the frozen evidence: {path}")


def _bundle_records(
    records: Sequence[Mapping[str, object]],
    *,
    run_root: Path,
    gallery_root: Path,
    cohort_path: Path,
    source_path: Path,
    manual_review_path: Path,
) -> tuple[dict[str, object], ...]:
    frozen = copy.deepcopy(records)
    for record in frozen:
        provenance = record["provenance"]
        controls = provenance["controls"]
        for control in controls:
            control["locator"] = _bundle_evidence_locator(
                control["locator"],
                run_root=run_root,
                gallery_root=gallery_root,
                cohort_path=cohort_path,
                source_path=source_path,
                manual_review_path=manual_review_path,
            )
    return tuple(frozen)


def _validate_frozen_publication_evidence(
    prepared: Path,
    conversion: PublicationConversion,
    bundled_records: Sequence[Mapping[str, object]],
    *,
    run_root: Path,
    gallery_manifest: Path,
    manual_review_artifact: Path,
) -> None:
    publication_root = prepared / "evidence" / "publication"
    frozen_run_root = publication_root / "validation-run"
    frozen_gallery_root = publication_root / "validation-gallery"
    frozen_gallery = frozen_gallery_root / gallery_manifest.name
    frozen_review = publication_root / "manual-review.json"
    frozen_cohort = publication_root / "cohort.json"
    frozen_source = publication_root / "extractants.smi"

    inputs = conversion.manifest["inputs"]
    fixed_artifacts = (
        (frozen_run_root / "manifest.json", inputs["validation_run_manifest"]),
        (frozen_cohort, inputs["semantic_cohort"]),
        (frozen_source, inputs["extractant_source"]),
        (frozen_gallery, inputs["gallery_manifest"]),
        (frozen_review, inputs["manual_review"]),
    )
    for path, declared in fixed_artifacts:
        declared = _require_mapping(declared, "conversion input artifact")
        if not path.is_file() or _sha256(path) != declared.get("sha256"):
            raise ValueError(f"frozen publication input differs: {path}")

    cohort = load_validation_cohort(frozen_cohort)
    if cohort.get("canonical_sha256") != inputs["semantic_cohort"].get(
        "canonical_sha256"
    ):
        raise ValueError("frozen cohort canonical SHA-256 differs")
    review = _load_object(frozen_review, "frozen manual review")
    validate_manual_review_artifact(review, frozen_gallery)

    frozen_manifest = _load_object(
        publication_root / "conversion-manifest.json",
        "frozen conversion manifest",
    )
    if frozen_manifest != conversion.manifest:
        raise ValueError("frozen conversion manifest differs")
    frozen_records = tuple(
        json.loads(line)
        for line in (
            publication_root / "conversion-route-results.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    )
    if frozen_records != conversion.records:
        raise ValueError("frozen conversion route results differ")

    for record in bundled_records:
        provenance = _require_mapping(record.get("provenance"), "provenance")
        controls = provenance.get("controls")
        if not isinstance(controls, list):
            raise ValueError("provenance.controls must be an array")
        for control in controls:
            control = _require_mapping(control, "provenance control")
            path = (prepared / str(control.get("locator", ""))).resolve()
            try:
                path.relative_to(prepared.resolve())
            except ValueError as error:
                raise ValueError("frozen provenance control escapes Bundle") from error
            if not path.is_file() or _sha256(path) != control.get("sha256"):
                raise ValueError(f"frozen provenance control differs: {path}")

    manifest = _load_object(frozen_run_root / "manifest.json", "frozen run manifest")
    _, primary, repeat = _run_contract(manifest)
    record_by_key = {
        (str(record["route"]), int(record["case_index"])): record
        for record in conversion.records
    }
    for spec in (*primary.values(), repeat):
        route = str(spec["route"])
        for case_index in FORMAL_CASE_INDICES:
            declared_result = _result_path(run_root, spec, case_index)
            frozen_result = _result_path(frozen_run_root, spec, case_index)
            runner_record = _load_object(frozen_result, "frozen runner result")
            _validate_case_artifacts(
                runner_record,
                route=route,
                declared_result_path=declared_result,
                artifact_result_path=frozen_result,
            )
            selected = _selected_structure(
                runner_record,
                result_path=declared_result,
                run_root=run_root,
                artifact_result_path=frozen_result,
                artifact_run_root=frozen_run_root,
            )
            if spec.get("repeat_of") is not None:
                continue
            published = record_by_key[(route, case_index)].get("selected_structure")
            if published is None:
                continue
            if selected is None or selected["sha256"] != published["sha256"]:
                raise ValueError("frozen selected structure differs from publication")
            fmt = str(published["format"]).lower()
            extension = "smi" if fmt == "smiles" else fmt
            structure = (
                prepared
                / "structures"
                / f"cbond-ff-{case_index:03d}-{route.lower()}.{extension}"
            )
            if not structure.is_file() or _sha256(structure) != published["sha256"]:
                raise ValueError("prepared Bundle structure differs from frozen evidence")


def prepare_bundle_from_validation_run(
    run_root: Path,
    gallery_manifest: Path,
    manual_review_artifact: Path,
    bundle_root: Path,
    *,
    authorization: Mapping[str, object],
    cohort_path: Optional[Path] = None,
    source_path: Optional[Path] = None,
    schema_path: Optional[Path] = None,
) -> dict[str, object]:
    """Validate, freeze, and prepare one publication-ready dataset Bundle."""

    run_root = run_root.resolve()
    gallery_manifest = gallery_manifest.resolve()
    manual_review_artifact = manual_review_artifact.resolve()
    bundle_root = bundle_root.resolve()
    conversion = build_publication_conversion(
        run_root,
        gallery_manifest,
        manual_review_artifact,
        cohort_path=cohort_path,
        source_path=source_path,
    )
    cohort_artifact = conversion.manifest["inputs"]["semantic_cohort"]
    source_artifact = conversion.manifest["inputs"]["extractant_source"]
    frozen_cohort = Path(str(cohort_artifact["path"])).resolve()
    frozen_source = Path(str(source_artifact["path"])).resolve()
    bundled_records = _bundle_records(
        conversion.records,
        run_root=run_root,
        gallery_root=gallery_manifest.parent,
        cohort_path=frozen_cohort,
        source_path=frozen_source,
        manual_review_path=manual_review_artifact,
    )

    if bundle_root.exists():
        raise FileExistsError(bundle_root)
    for evidence_root in (run_root, gallery_manifest.parent):
        try:
            bundle_root.relative_to(evidence_root)
        except ValueError:
            continue
        raise ValueError("bundle_root must be outside validation evidence roots")
    bundle_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{bundle_root.name}.publication.", dir=bundle_root.parent
        )
    )
    prepared = staging / "bundle"
    try:
        summary = prepare_bundle(
            bundled_records,
            source_path=frozen_source,
            bundle_root=prepared,
            authorization=authorization,
            schema_path=schema_path,
            result_root=run_root,
            case_indices=FORMAL_CASE_INDICES,
            expected_source_sha256=str(source_artifact["sha256"]),
        )
        publication_root = prepared / "evidence" / "publication"
        publication_root.mkdir(parents=True)
        _copy_evidence_tree(run_root, publication_root / "validation-run")
        _copy_evidence_tree(
            gallery_manifest.parent,
            publication_root / "validation-gallery",
        )
        shutil.copyfile(frozen_cohort, publication_root / "cohort.json")
        shutil.copyfile(frozen_source, publication_root / "extractants.smi")
        shutil.copyfile(
            manual_review_artifact,
            publication_root / "manual-review.json",
        )
        (publication_root / "conversion-manifest.json").write_bytes(
            _canonical_bytes(conversion.manifest)
        )
        (publication_root / "conversion-route-results.jsonl").write_bytes(
            b"".join(_canonical_bytes(record) for record in conversion.records)
        )

        _validate_frozen_publication_evidence(
            prepared,
            conversion,
            bundled_records,
            run_root=run_root,
            gallery_manifest=gallery_manifest,
            manual_review_artifact=manual_review_artifact,
        )

        index_path = prepared / EVIDENCE_INDEX
        index_path.write_bytes(_canonical_bytes(_publication_evidence_index(prepared)))
        bundle_manifest_path = prepared / "bundle.json"
        bundle_manifest = _load_object(bundle_manifest_path, "bundle manifest")
        controls = bundle_manifest.get("control_assets")
        if not isinstance(controls, list):
            raise ValueError("bundle control_assets must be an array")
        controls.append(
            {
                "category": "validation",
                "name": "plan17-publication-evidence-index",
                "path": EVIDENCE_INDEX,
                "sha256": _sha256(index_path),
                "size": index_path.stat().st_size,
            }
        )
        bundle_manifest_path.write_bytes(_canonical_bytes(bundle_manifest))
        os.replace(prepared, bundle_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    shutil.rmtree(staging)
    return {
        **summary,
        "bundle_root": str(bundle_root),
        "publication_evidence_index": str(bundle_root / EVIDENCE_INDEX),
        "publication_evidence_sha256": _sha256(bundle_root / EVIDENCE_INDEX),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--gallery-manifest", type=Path, required=True)
    parser.add_argument("--manual-review", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cohort", type=Path)
    parser.add_argument("--source", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = write_publication_conversion(
        arguments.run_root,
        arguments.gallery_manifest,
        arguments.manual_review,
        arguments.output_root,
        cohort_path=arguments.cohort,
        source_path=arguments.source,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "CONVERSION_SCHEMA",
    "EVIDENCE_SCHEMA",
    "PublicationConversion",
    "build_publication_conversion",
    "prepare_bundle_from_validation_run",
    "write_publication_conversion",
)
