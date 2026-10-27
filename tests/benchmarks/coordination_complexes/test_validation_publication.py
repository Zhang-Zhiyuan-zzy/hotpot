"""Contracts for the Plan-17 validation-to-publication bridge."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from . import validation_publication as publication
from .validation_dataset import publish_bundle
from .validation_gallery import (
    ValidationGalleryParameters,
    create_manual_review_template,
    render_validation_gallery,
)
from .validation_providers import (
    FrozenCoordinateProposal,
    FrozenCoordinateReplayAdapter,
    coordinate_sha256,
    write_frozen_proposals,
    write_mapped_coordinate_proposal,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _complete_review(gallery_manifest: Path) -> dict[str, object]:
    review = create_manual_review_template(gallery_manifest)
    review["status"] = "complete"
    review["reviewer"] = "human-reviewer@example.test"
    review["completed_at"] = "2026-10-23T18:00:00+08:00"
    for row in review["reviews"]:
        row["reviewed"] = True
    review["summary"]["reviewed_case_count"] = len(review["reviews"])
    return review


def _authorization() -> dict[str, object]:
    return {
        "authorization_id": "owner-plan17-v1",
        "authority": "User",
        "actor": "owner@example.test",
        "approved": True,
        "reason": "Owner-authorized Plan-17 benchmark publication",
        "authorized_at": "2026-10-23T00:00:00Z",
        "license": "CC-BY-4.0",
        "access_policy": "open",
    }


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    case_indices = (1, 2)
    monkeypatch.setattr(publication, "FORMAL_CASE_INDICES", case_indices)
    tmp_path.mkdir(parents=True, exist_ok=True)

    source = tmp_path / "extractants.smi"
    source.write_text("C\nN\n", encoding="utf-8")
    cohort_path = tmp_path / "cohort.json"
    cohort_path.write_text('{"fixture": true}\n', encoding="utf-8")
    source_sha256 = _sha256(source)
    cohort = {
        "canonical_sha256": "c" * 64,
        "cases": [
            {
                "index": index,
                "smiles": smiles,
                "seed": 20260921 + index,
                "status": "cbond",
                "semantic_graph_sha256": hashlib.sha256(
                    f"graph-{index}".encode()
                ).hexdigest(),
                "coordination": {
                    "metal": {"row": 2, "symbol": "Am"},
                    "donor_rows": [0, 1],
                },
            }
            for index, smiles in ((1, "C"), (2, "N"))
        ],
    }
    monkeypatch.setattr(
        publication,
        "load_validation_cohort",
        lambda path: cohort,
    )
    identity = {
        "source": {
            "path": str(source),
            "sha256": source_sha256,
            "physical_line_count": 2,
        },
        "cohort": {
            "path": str(cohort_path),
            "canonical_sha256": cohort["canonical_sha256"],
            "schema_version": 1,
            "cbond_count": 2,
            "no_cbond_count": 0,
        },
        "git": {"commit": "a" * 40, "dirty": False},
        "runtime": {
            "imported_hotpot_version": "0.5.4.5",
            "rdkit_version": "2026.09.1",
            "python_rdkit_version": "2026.09.1",
            "python_rdkit_distribution_version": "2026.9.1",
            "openbabel_version": "3.2.1",
            "python_version": "3.13.0",
        },
        "settings": {"seed": 20260921, "epochs": 100},
    }
    run_prefix = "formal-test"
    run_specs = [
        {
            "route": route,
            "run_id": f"{run_prefix}-{route.lower()}",
            "repeat_of": None,
        }
        for route in publication.ROUTES
    ]
    run_specs.append(
        {
            "route": "F1",
            "run_id": f"{run_prefix}-f1-repeat",
            "repeat_of": f"{run_prefix}-f1",
        }
    )
    run_root = tmp_path / "run"
    _write_json(
        run_root / "manifest.json",
        {
            "schema": publication.RUN_MANIFEST_SCHEMA,
            "scientific_configuration": {
                "formal": True,
                "identity": identity,
                "run_specs": run_specs,
                "case_indices": list(case_indices),
            },
            "execution": {"workers": 2},
            "human_review": {"status": "not_performed"},
        },
    )

    records = []
    result_paths = {}
    selected_paths = {}
    for spec in run_specs:
        for case_index in case_indices:
            route = spec["route"]
            is_repeat = spec["repeat_of"] is not None
            passed = is_repeat or route != "C2" or case_index != 2
            case = cohort["cases"][case_index - 1]
            record = {
                "schema": publication.RUN_RESULT_SCHEMA,
                "case_index": case_index,
                "source_smiles": case["smiles"],
                "semantic_graph_sha256": case["semantic_graph_sha256"],
                "route": route,
                "run_id": spec["run_id"],
                "seed": case["seed"],
                "status": "automated_passed" if passed else "execution_failed",
                "passed": passed,
                "identity": identity,
                "proposal": {"record": {"status": "SUCCESS"}},
                "evidence": (
                    {
                        "unavailable_gradient_events": [
                            {
                                "event_id": "gradient-1",
                                "recovery": {"action_name": "PERTURBED"},
                            }
                        ],
                        "stage_trace": {
                            "records": [
                                {
                                    "name": "optimization",
                                    "status": "recovered",
                                    "recovery_actions": ["PERTURBED"],
                                }
                            ]
                        },
                        "quality": {
                            "passed": True,
                            "check_names": ["standard"],
                        },
                    }
                    if passed
                    else None
                ),
                "acceptance": {
                    "case_id": f"extractant-{case_index:03d}",
                    "route": route,
                    "passed": passed,
                    "gates": [
                        {
                            "gate_id": gate,
                            "passed": passed,
                            "diagnostics": [] if passed else ["not_evaluated"],
                        }
                        for gate in publication.GATE_IDS
                    ],
                    "diagnostics": [] if passed else ["not_evaluated:execution"],
                },
                "optimization": {
                    "best_energy": -10.0 if passed else None,
                    "rms_gradient": 0.1 if passed else None,
                    "max_gradient": 0.2 if passed else None,
                },
                "timing": {"runner_elapsed_seconds": 1.0},
                "resources": {"identity": {"hostname": "fixture"}},
                "trajectory_artifact": {"sha256": "d" * 64},
                # Deliberately forged: publication must derive review state
                # from the independently validated review artifact.
                "manual_review": {"status": "passed", "flags": []},
            }
            result_path = publication._result_path(run_root, spec, case_index)
            if route in {"F2", "C1", "C2"}:
                coordinates = np.arange(9, dtype=np.float64).reshape((3, 3))
                if route == "F2":
                    proposal_path = result_path.parent / "frozen-proposals.json"
                    proposal_identity = write_frozen_proposals(
                        proposal_path,
                        FrozenCoordinateReplayAdapter(
                            (
                                FrozenCoordinateProposal(
                                    component_sha256="f" * 64,
                                    coordinate_sha256=coordinate_sha256(coordinates),
                                    coordinates=tuple(
                                        tuple(float(value) for value in row)
                                        for row in coordinates
                                    ),
                                    ordinal=0,
                                ),
                            )
                        ),
                    )
                else:
                    proposal_path = (
                        result_path.parent / "whole-complex-proposal.json"
                    )
                    proposal_identity = write_mapped_coordinate_proposal(
                        proposal_path,
                        route=route,
                        proposal={
                            "coordinate_sha256": coordinate_sha256(coordinates)
                        },
                        coordinates=coordinates,
                    )
                record["proposal"] = {
                    "record": {"status": "SUCCESS"},
                    "frozen_manifest": [],
                    "declared_artifact": proposal_identity,
                    "archived_artifact": publication._proposal_artifact_identity(
                        proposal_path
                    ),
                }
            trajectory = result_path.parent / "trajectory"
            trajectory.mkdir(parents=True, exist_ok=True)
            _write_json(trajectory / "frames.json", {"frames": []})
            record["trajectory_artifact"] = {
                **publication._tree_identity(trajectory),
                "includes_sdf": False,
            }
            if passed:
                selected = result_path.parent / "selected.sdf"
                selected.parent.mkdir(parents=True, exist_ok=True)
                selected.write_text(
                    f"{spec['run_id']} case {case_index}\n",
                    encoding="utf-8",
                )
                record["selected_structure"] = {
                    "path": str(selected),
                    "sha256": _sha256(selected),
                    "format": "sdf",
                    "atom_row_mapping": [0, 1, 2],
                }
                selected_paths[(spec["run_id"], case_index)] = selected
            _write_json(result_path, record)
            result_paths[(spec["run_id"], case_index)] = result_path
            if not is_repeat:
                records.append(record)

    def renderer(source_path, target, parameters):
        del source_path
        Image.new(
            "RGB",
            (parameters.image_width, parameters.image_height),
            "white",
        ).save(target)

    gallery_root = tmp_path / "gallery"
    render_validation_gallery(
        records,
        gallery_root,
        validation_revision="plan17-validation-test",
        case_indices=case_indices,
        renderer=renderer,
        parameters=ValidationGalleryParameters(
            image_width=30,
            image_height=20,
            ray_dpi=72,
            columns=2,
            cell_width=80,
            cell_height=60,
            header_height=20,
        ),
    )
    gallery_manifest = gallery_root / "gallery.json"
    review = _complete_review(gallery_manifest)
    manual_review = gallery_root / "manual_review.json"
    _write_json(manual_review, review)
    return {
        "run_root": run_root,
        "cohort_path": cohort_path,
        "source": source,
        "gallery_root": gallery_root,
        "gallery_manifest": gallery_manifest,
        "manual_review": manual_review,
        "review": review,
        "result_paths": result_paths,
        "selected_paths": selected_paths,
    }


def test_bridge_selects_only_primary_runs_and_applies_validated_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    review = copy.deepcopy(fixture["review"])
    flagged = next(
        row
        for row in review["reviews"]
        if row["route"] == "O0" and row["case_index"] == 1
    )
    flagged["flagged"] = True
    flagged["reason"] = "visible ligand interpenetration"
    review["summary"]["flagged_case_count"] = 1
    _write_json(fixture["manual_review"], review)

    output = tmp_path / "publication"
    written = publication.write_publication_conversion(
        fixture["run_root"],
        fixture["gallery_manifest"],
        fixture["manual_review"],
        output,
        cohort_path=fixture["cohort_path"],
        source_path=fixture["source"],
    )
    records = [
        json.loads(line)
        for line in (output / publication.RESULTS_FILENAME)
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert len(records) == 10
    assert {record["run_id"] for record in records} == {
        f"formal-test-{route.lower()}" for route in publication.ROUTES
    }
    flagged_record = next(
        record
        for record in records
        if record["route"] == "O0" and record["case_index"] == 1
    )
    assert flagged_record["passed"] is False
    assert flagged_record["status"] == "manual_review_failed"
    assert "selected_structure" not in flagged_record
    assert flagged_record["manual_review"]["flags"][0]["reason"] == (
        "visible ligand interpenetration"
    )
    unflagged = next(
        record
        for record in records
        if record["route"] == "F1" and record["case_index"] == 1
    )
    assert unflagged["passed"] is True
    assert unflagged["selected_structure"]["path"] == str(
        fixture["selected_paths"][("formal-test-f1", 1)].resolve()
    )
    assert unflagged["acceptance"] == {
        **{gate: True for gate in publication.GATE_IDS},
        "diagnostics": [],
        "recovery_actions": ["PERTURBED"],
    }
    assert {control["name"] for control in unflagged["provenance"]["controls"]} >= {
        "semantic-cohort",
        "primary-runner-result",
        "f1-repeat-control-result",
    }
    assert (
        unflagged["settings"]["validation"]["runtime_identity"]["rdkit_version"]
        == "2026.09.1"
    )
    automated_failure = next(
        record
        for record in records
        if record["route"] == "C2" and record["case_index"] == 2
    )
    assert automated_failure["status"] == "execution_failed"
    assert automated_failure["passed"] is False
    assert "selected_structure" not in automated_failure
    assert automated_failure["acceptance"]["recovery_actions"] == []

    manifest = json.loads(
        (output / publication.MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert manifest["publication_matrix"] == {
        "case_count": 2,
        "route_count": 5,
        "record_count": 10,
        "passed_count": 8,
        "failed_count": 2,
        "manual_flagged_count": 1,
        "selected_structure_count": 8,
        "records_path": publication.RESULTS_FILENAME,
        "records_sha256": _sha256(output / publication.RESULTS_FILENAME),
        "records_size": (output / publication.RESULTS_FILENAME).stat().st_size,
    }
    assert manifest["f1_repeat_control"]["case_count"] == 2
    assert written["manifest_sha256"] == _sha256(output / publication.MANIFEST_FILENAME)
    assert (
        (output / publication.MANIFEST_DIGEST_FILENAME)
        .read_text(encoding="utf-8")
        .startswith(written["manifest_sha256"])
    )


def test_bridge_rejects_pending_or_tampered_human_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    pending = copy.deepcopy(fixture["review"])
    pending["status"] = "pending_human_review"
    _write_json(fixture["manual_review"], pending)

    with pytest.raises(ValueError, match="status must be complete"):
        publication.build_publication_conversion(
            fixture["run_root"],
            fixture["gallery_manifest"],
            fixture["manual_review"],
            cohort_path=fixture["cohort_path"],
            source_path=fixture["source"],
        )

    _write_json(fixture["manual_review"], fixture["review"])
    gallery = json.loads(fixture["gallery_manifest"].read_text(encoding="utf-8"))
    contact_sheet = fixture["gallery_root"] / gallery["routes"][0]["contact_sheet"]
    contact_sheet.write_bytes(contact_sheet.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="SHA-256 differs"):
        publication.build_publication_conversion(
            fixture["run_root"],
            fixture["gallery_manifest"],
            fixture["manual_review"],
            cohort_path=fixture["cohort_path"],
            source_path=fixture["source"],
        )


def test_bridge_rejects_incomplete_matrix_and_changed_selected_asset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    missing = fixture["result_paths"][("formal-test-c2", 2)]
    missing.unlink()
    with pytest.raises(FileNotFoundError):
        publication.build_publication_conversion(
            fixture["run_root"],
            fixture["gallery_manifest"],
            fixture["manual_review"],
            cohort_path=fixture["cohort_path"],
            source_path=fixture["source"],
        )

    fixture = _fixture(tmp_path / "changed", monkeypatch)
    fixture["selected_paths"][("formal-test-f1", 1)].write_text(
        "changed after gallery review\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="selected structure SHA-256 mismatch"):
        publication.build_publication_conversion(
            fixture["run_root"],
            fixture["gallery_manifest"],
            fixture["manual_review"],
            cohort_path=fixture["cohort_path"],
            source_path=fixture["source"],
        )


def test_bridge_rejects_duplicate_runs_and_foreign_selection_but_records_repeat_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path / "duplicate", monkeypatch)
    manifest_path = fixture["run_root"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["scientific_configuration"]["run_specs"][4]["run_id"] = "formal-test-o0"
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="globally unique"):
        publication.build_publication_conversion(
            fixture["run_root"],
            fixture["gallery_manifest"],
            fixture["manual_review"],
            cohort_path=fixture["cohort_path"],
            source_path=fixture["source"],
        )

    fixture = _fixture(tmp_path / "repeat", monkeypatch)
    repeated_path = fixture["result_paths"][("formal-test-f1-repeat", 1)]
    repeated = json.loads(repeated_path.read_text(encoding="utf-8"))
    repeated["passed"] = False
    repeated["status"] = "execution_failed"
    repeated["evidence"] = None
    repeated.pop("selected_structure")
    repeated["acceptance"]["passed"] = False
    for gate in repeated["acceptance"]["gates"]:
        gate["passed"] = False
    _write_json(repeated_path, repeated)
    conversion = publication.build_publication_conversion(
        fixture["run_root"],
        fixture["gallery_manifest"],
        fixture["manual_review"],
        cohort_path=fixture["cohort_path"],
        source_path=fixture["source"],
    )
    repeat_control = conversion.manifest["f1_repeat_control"]
    assert repeat_control["acceptance_classification_consistent"] is False
    assert repeat_control["acceptance_mismatch_case_indices"] == [1]
    assert repeat_control["gate_classification_consistent"] is False
    assert repeat_control["gate_mismatch_case_indices"] == [1]

    fixture = _fixture(tmp_path / "foreign", monkeypatch)
    result_path = fixture["result_paths"][("formal-test-f1", 1)]
    record = json.loads(result_path.read_text(encoding="utf-8"))
    foreign = tmp_path / "foreign.sdf"
    foreign.write_text("foreign selection\n", encoding="utf-8")
    record["selected_structure"]["path"] = str(foreign)
    record["selected_structure"]["sha256"] = _sha256(foreign)
    _write_json(result_path, record)
    with pytest.raises(ValueError, match="selected.sdf beside"):
        publication.build_publication_conversion(
            fixture["run_root"],
            fixture["gallery_manifest"],
            fixture["manual_review"],
            cohort_path=fixture["cohort_path"],
            source_path=fixture["source"],
        )


def test_bridge_rejects_mutated_proposal_and_trajectory_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path / "proposal", monkeypatch)
    result = json.loads(
        fixture["result_paths"][("formal-test-f2", 1)].read_text(encoding="utf-8")
    )
    proposal = Path(result["proposal"]["archived_artifact"]["path"])
    proposal.write_text(
        proposal.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="proposal artifact sha256 differs"):
        publication.build_publication_conversion(
            fixture["run_root"],
            fixture["gallery_manifest"],
            fixture["manual_review"],
            cohort_path=fixture["cohort_path"],
            source_path=fixture["source"],
        )

    fixture = _fixture(tmp_path / "trajectory", monkeypatch)
    result_path = fixture["result_paths"][("formal-test-o0", 1)]
    (result_path.parent / "trajectory" / "frames.json").write_text(
        '{"frames":["mutated"]}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="trajectory artifact sha256 differs"):
        publication.build_publication_conversion(
            fixture["run_root"],
            fixture["gallery_manifest"],
            fixture["manual_review"],
            cohort_path=fixture["cohort_path"],
            source_path=fixture["source"],
        )


def test_bundle_preparation_rejects_copy_time_evidence_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path / "input", monkeypatch)
    original_copy = publication._copy_evidence_tree
    selected = fixture["selected_paths"][("formal-test-f1", 1)]
    mutated = False

    def mutate_then_copy(source, destination):
        nonlocal mutated
        if source.resolve() == fixture["run_root"].resolve() and not mutated:
            selected.write_text("copy-time mutation\n", encoding="utf-8")
            mutated = True
        original_copy(source, destination)

    monkeypatch.setattr(publication, "_copy_evidence_tree", mutate_then_copy)

    with pytest.raises(ValueError, match="selected structure SHA-256 mismatch"):
        publication.prepare_bundle_from_validation_run(
            fixture["run_root"],
            fixture["gallery_manifest"],
            fixture["manual_review"],
            tmp_path / "bundle",
            authorization=_authorization(),
            cohort_path=fixture["cohort_path"],
            source_path=fixture["source"],
        )
    assert not (tmp_path / "bundle").exists()


def test_prepare_bundle_freezes_and_certifies_complete_validation_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path / "input", monkeypatch)
    bundle_root = tmp_path / "bundle"
    summary = publication.prepare_bundle_from_validation_run(
        fixture["run_root"],
        fixture["gallery_manifest"],
        fixture["manual_review"],
        bundle_root,
        authorization=_authorization(),
        cohort_path=fixture["cohort_path"],
        source_path=fixture["source"],
    )

    index_path = bundle_root / publication.EVIDENCE_INDEX
    evidence = json.loads(index_path.read_text(encoding="utf-8"))
    assert evidence["schema"] == publication.EVIDENCE_SCHEMA
    assert evidence["asset_count"] == len(evidence["assets"])
    assert {
        "evidence/publication/cohort.json",
        "evidence/publication/extractants.smi",
        "evidence/publication/manual-review.json",
        "evidence/publication/conversion-manifest.json",
        "evidence/publication/conversion-route-results.jsonl",
        "evidence/publication/validation-run/manifest.json",
        "evidence/publication/validation-gallery/gallery.json",
    } <= {asset["path"] for asset in evidence["assets"]}
    for asset in evidence["assets"]:
        frozen = bundle_root / asset["path"]
        assert frozen.stat().st_size == asset["size"]
        assert _sha256(frozen) == asset["sha256"]

    bundle = json.loads((bundle_root / "bundle.json").read_text(encoding="utf-8"))
    control = next(
        asset
        for asset in bundle["control_assets"]
        if asset["name"] == "plan17-publication-evidence-index"
    )
    assert control["path"] == publication.EVIDENCE_INDEX
    assert control["sha256"] == _sha256(index_path)
    assert summary["publication_evidence_sha256"] == control["sha256"]
    bundled_records = [
        json.loads(line)
        for line in (bundle_root / "evidence/control/route-results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    for record in bundled_records:
        for provenance_control in record["provenance"]["controls"]:
            locator = provenance_control["locator"]
            assert not Path(locator).is_absolute()
            assert (bundle_root / locator).is_file()
            assert _sha256(bundle_root / locator) == provenance_control["sha256"]

    original = fixture["selected_paths"][("formal-test-f1", 1)]
    frozen = (
        bundle_root
        / "evidence/publication/validation-run/runs/formal-test-f1/F1"
        / "cases/0001/selected.sdf"
    )
    original.write_text("mutated after Bundle preparation\n", encoding="utf-8")
    frozen_asset = next(
        asset
        for asset in evidence["assets"]
        if asset["path"] == frozen.relative_to(bundle_root).as_posix()
    )
    assert _sha256(frozen) == frozen_asset["sha256"]

    published = publish_bundle(
        bundle_root,
        database=tmp_path / "store" / "datasets.sqlite3",
        reason="End-to-end publication evidence test",
        confirm_ingest=True,
    )
    assert published["receipt"]["included_count"] == 10
    assert published["inventory"]["benchmark_matrix"]["route_case_count"] == 10
