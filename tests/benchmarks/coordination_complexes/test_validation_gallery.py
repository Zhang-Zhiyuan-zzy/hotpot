"""Contracts for exhaustive Plan-17 galleries and human review evidence."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from .validation_dataset import ROUTES
from .validation_gallery import (
    ValidationGalleryParameters,
    create_manual_review_template,
    render_validation_gallery,
    validate_manual_review_artifact,
    write_manual_review_artifact,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _records(
    structure: Path,
    *,
    case_indices: tuple[int, ...] = (1, 2),
) -> list[dict[str, object]]:
    records = []
    selected = {
        "path": str(structure),
        "sha256": _sha256(structure),
        "format": "mol2",
        "atom_row_mapping": [0],
    }
    for route in ROUTES:
        for case_index in case_indices:
            passed = case_index == 1
            records.append(
                {
                    "route": route,
                    "run_id": f"run-{route.lower()}-001",
                    "case_index": case_index,
                    "status": "passed" if passed else "optimization_failed",
                    "passed": passed,
                    # Deliberately attach the rejected asset to prove that a
                    # failed row never enters the structure renderer.
                    "selected_structure": dict(selected),
                }
            )
    return records


def _render_parameters() -> ValidationGalleryParameters:
    return ValidationGalleryParameters(
        image_width=120,
        image_height=90,
        ray_dpi=72,
        columns=2,
        cell_width=180,
        cell_height=150,
        header_height=55,
    )


def _complete(template: dict[str, object]) -> dict[str, object]:
    value = copy.deepcopy(template)
    value["status"] = "complete"
    value["reviewer"] = "human-reviewer@example.test"
    value["completed_at"] = "2026-10-23T16:42:00+08:00"
    for review in value["reviews"]:
        review["reviewed"] = True
    value["summary"]["reviewed_case_count"] = len(value["reviews"])
    return value


def test_gallery_renders_only_passed_selected_structures_and_is_exhaustive(
    tmp_path: Path,
) -> None:
    selected = tmp_path / "selected.mol2"
    selected.write_text("selected structure\n", encoding="utf-8")
    calls: list[Path] = []

    def renderer(source, target, parameters):
        calls.append(source)
        Image.new(
            "RGB", (parameters.image_width, parameters.image_height), "white"
        ).save(target)

    output = tmp_path / "gallery"
    payload = render_validation_gallery(
        _records(selected),
        output,
        validation_revision="plan17-validation-test",
        case_indices=(1, 2),
        renderer=renderer,
        parameters=_render_parameters(),
    )

    assert calls == [selected.resolve()] * len(ROUTES)
    assert payload["case_indices"] == [1, 2]
    assert len(payload["cases"]) == 10
    assert [route["route"] for route in payload["routes"]] == list(ROUTES)
    for route in payload["routes"]:
        assert route["case_count"] == 2
        assert route["rendered_selected_count"] == 1
        assert route["scientific_placeholder_count"] == 1
        assert route["render_error_count"] == 0
        assert (output / route["contact_sheet"]).is_file()
    failed = [case for case in payload["cases"] if case["case_index"] == 2]
    assert {case["display_role"] for case in failed} == {"placeholder"}
    assert {case["render_status"] for case in failed} == {"scientific_placeholder"}
    assert all(case["selected_structure_sha256"] is None for case in failed)
    assert all(case["placeholder_reason"] for case in failed)
    assert (output / "gallery.json").is_file()
    assert (output / "manual_review.template.json").is_file()
    assert str(selected) not in json.dumps(failed)


def test_passed_row_without_selected_structure_is_an_explicit_placeholder(
    tmp_path: Path,
) -> None:
    selected = tmp_path / "selected.mol2"
    selected.write_text("selected structure\n", encoding="utf-8")
    records = _records(selected, case_indices=(1,))
    records[0].pop("selected_structure")
    calls = []

    def renderer(source, target, parameters):
        calls.append(source)
        Image.new("RGB", (120, 90), "white").save(target)

    payload = render_validation_gallery(
        records,
        tmp_path / "gallery",
        validation_revision="revision",
        case_indices=(1,),
        renderer=renderer,
        parameters=_render_parameters(),
    )

    missing = payload["cases"][0]
    assert missing["route"] == "O0"
    assert missing["passed"] is True
    assert missing["display_role"] == "placeholder"
    assert missing["placeholder_reason"] == "no_selected_structure"
    assert len(calls) == len(ROUTES) - 1


def test_pending_template_cannot_be_mistaken_for_completed_human_review(
    tmp_path: Path,
) -> None:
    selected = tmp_path / "selected.mol2"
    selected.write_text("selected structure\n", encoding="utf-8")

    def renderer(source, target, parameters):
        Image.new("RGB", (120, 90), "white").save(target)

    output = tmp_path / "gallery"
    render_validation_gallery(
        _records(selected),
        output,
        validation_revision="revision-under-review",
        case_indices=(1, 2),
        renderer=renderer,
        parameters=_render_parameters(),
    )
    manifest = output / "gallery.json"
    template = create_manual_review_template(manifest)

    assert template["status"] == "pending_human_review"
    assert template["reviewer"] is None
    assert template["completed_at"] is None
    assert len(template["reviews"]) == 10
    assert not any(review["reviewed"] for review in template["reviews"])
    with pytest.raises(ValueError, match="status must be complete"):
        validate_manual_review_artifact(template, manifest)


def test_completed_manual_review_binds_every_route_case_and_gallery_hash(
    tmp_path: Path,
) -> None:
    selected = tmp_path / "selected.mol2"
    selected.write_text("selected structure\n", encoding="utf-8")

    def renderer(source, target, parameters):
        Image.new("RGB", (120, 90), "white").save(target)

    output = tmp_path / "gallery"
    render_validation_gallery(
        _records(selected),
        output,
        validation_revision="revision-under-review",
        case_indices=(1, 2),
        renderer=renderer,
        parameters=_render_parameters(),
    )
    manifest = output / "gallery.json"
    artifact = _complete(create_manual_review_template(manifest))
    artifact["reviews"][0]["flagged"] = True
    artifact["reviews"][0]["reason"] = "visible ligand interpenetration"
    artifact["summary"]["flagged_case_count"] = 1

    validated = validate_manual_review_artifact(artifact, manifest)
    target = output / "manual_review.json"
    write_manual_review_artifact(validated, manifest, target)

    assert validated["validation_revision"] == "revision-under-review"
    assert validated["gallery_manifest_sha256"] == _sha256(manifest)
    assert [route["route"] for route in validated["routes"]] == list(ROUTES)
    assert all(len(route["gallery_sha256"]) == 64 for route in validated["routes"])
    assert len({row["gallery_image_sha256"] for row in validated["reviews"]}) > 0
    assert json.loads(target.read_text(encoding="utf-8")) == validated


def test_manual_review_rejects_missing_rows_reasonless_flags_and_naive_date(
    tmp_path: Path,
) -> None:
    selected = tmp_path / "selected.mol2"
    selected.write_text("selected structure\n", encoding="utf-8")

    def renderer(source, target, parameters):
        Image.new("RGB", (120, 90), "white").save(target)

    output = tmp_path / "gallery"
    render_validation_gallery(
        _records(selected),
        output,
        validation_revision="revision",
        case_indices=(1, 2),
        renderer=renderer,
        parameters=_render_parameters(),
    )
    manifest = output / "gallery.json"
    complete = _complete(create_manual_review_template(manifest))

    missing = copy.deepcopy(complete)
    missing["reviews"].pop()
    missing["summary"]["reviewed_case_count"] -= 1
    with pytest.raises(ValueError, match="missing route/case"):
        validate_manual_review_artifact(missing, manifest)

    reasonless = copy.deepcopy(complete)
    reasonless["reviews"][0]["flagged"] = True
    reasonless["summary"]["flagged_case_count"] = 1
    with pytest.raises(ValueError, match="requires a reason"):
        validate_manual_review_artifact(reasonless, manifest)

    naive = copy.deepcopy(complete)
    naive["completed_at"] = "2026-10-23T16:42:00"
    with pytest.raises(ValueError, match="include a timezone"):
        validate_manual_review_artifact(naive, manifest)


def test_render_errors_and_changed_gallery_assets_cannot_be_human_certified(
    tmp_path: Path,
) -> None:
    selected = tmp_path / "selected.mol2"
    selected.write_text("selected structure\n", encoding="utf-8")
    calls = 0

    def renderer(source, target, parameters):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("synthetic renderer failure")
        Image.new("RGB", (120, 90), "white").save(target)

    output = tmp_path / "gallery"
    payload = render_validation_gallery(
        _records(selected, case_indices=(1,)),
        output,
        validation_revision="revision",
        case_indices=(1,),
        renderer=renderer,
        parameters=_render_parameters(),
    )
    manifest = output / "gallery.json"
    complete = _complete(create_manual_review_template(manifest))

    assert sum(route["render_error_count"] for route in payload["routes"]) == 1
    with pytest.raises(ValueError, match="render errors prevent"):
        validate_manual_review_artifact(complete, manifest)

    sheet = output / payload["routes"][0]["contact_sheet"]
    sheet.write_bytes(sheet.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="SHA-256 differs"):
        create_manual_review_template(manifest)
