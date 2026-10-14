from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

import hotpot.ds.cli as ds_cli
from hotpot.ds.cli import build_parser, run
from hotpot.ds.public._bundle_state import BUNDLE_SCHEMA
from hotpot.ds.public.bundle import load_bundle


def _case(case_id: str, index: int, *, approved: bool = False, complete: bool = True) -> dict:
    value = {
        "id": case_id,
        "local_id": f"local-{case_id}",
        "public_index": index,
        "identifiers": {"CSD": case_id.upper()},
        "core_info": {"sample": case_id} if complete else {},
        "observations": (
            [{"observation_id": f"obs-{case_id}", "property": "spin", "value": index + 1}]
            if complete
            else []
        ),
        "structures": (
            [{"structure_id": f"struct-{case_id}", "role": "primary", "format": "smi", "path": f"structures/{case_id}.smi", "primary": True}]
            if complete
            else []
        ),
    }
    if approved:
        value["attestations"] = [
            {"attestation_id": f"att-{case_id}", "authority": "User", "actor": "curator", "approved": True}
        ]
    return value


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "bundle"
    intake = root / "evidence" / "intake" / "cases.jsonl"
    intake.parent.mkdir(parents=True)
    values = [
        _case("ordinary", 2),
        _case("benchmark", 7, approved=True),
        _case("incomplete", 11, complete=False),
    ]
    intake.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")
    structures = root / "structures"
    structures.mkdir()
    (structures / "ordinary.smi").write_text("O\n", encoding="utf-8")
    (structures / "benchmark.smi").write_text("N\n", encoding="utf-8")
    (root / "bundle.json").write_text(
        json.dumps(
            {
                "schema": BUNDLE_SCHEMA,
                "ds_name": "demo",
                "release_id": "demo-r1",
                "release_serial": 1,
                "version": "1.0",
                "title": "Demo",
                "description": "Stage 13 CLI fixture",
                "license": "CC0-1.0",
                "access_policy": "open",
                "provenance": {"created_by": "test"},
                "core_labels": ["spin"],
                "core_info": ["sample"],
                "intake": {"format": "jsonl", "path": "evidence/intake/cases.jsonl"},
                "identity_schema": {
                    "components": [
                        {"key": "sample", "path": "core_info.sample", "required": True, "normalizer": "text"}
                    ]
                },
                "required_structure_roles": ["primary"],
            }
        ),
        encoding="utf-8",
    )
    return root


def _invoke(arguments: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, dict | None, str]:
    code = run(build_parser().parse_args(arguments))
    captured = capsys.readouterr()
    envelope = json.loads(captured.out) if captured.out else None
    return code, envelope, captured.err


def test_bundle_cli_has_explicit_transitions_and_closed_channels(tmp_path: Path, capsys) -> None:
    root = _root(tmp_path)
    database = tmp_path / "datasets.sqlite3"

    code, result, _ = _invoke(
        ["bundle", "build", str(root), "--dataset", "demo", "--db", str(database)],
        capsys,
    )
    assert code == 0
    assert result["key"] == "hotpot.ds.bundle.build"
    assert result["payload"]["is_synchronized"] is False

    code, result, error = _invoke(
        ["bundle", "--root", str(root), "get", "cases"], capsys
    )
    assert code == 2 and result is None
    assert "synchronize" in error

    code, result, _ = _invoke(
        ["bundle", "--root", str(root), "gate", "benchmark", "--show"], capsys
    )
    assert code == 0
    assert result["key"] == "hotpot.ds.bundle.gate"
    assert result["payload"]["action"] == "show"

    code, result, _ = _invoke(
        ["bundle", "--root", str(root), "synchronize"], capsys
    )
    assert code == 0
    assert result["payload"]["archive_counts"] == {
        "benchmark": 1,
        "incomplete": 1,
        "ordinary": 1,
    }

    code, result, _ = _invoke(
        ["bundle", "--root", str(root), "get", "cases", "--archive-class", "benchmark"],
        capsys,
    )
    assert code == 0
    assert result["channel"] == "cases"
    assert result["payload"]["items"] == [{"case_id": "benchmark"}]

    code, result, _ = _invoke(
        ["bundle", "--root", str(root), "get", "cases", "--min-evidence", "E0"],
        capsys,
    )
    assert code == 0
    assert len(result["payload"]["items"]) == 3
    assert "min_evidence" not in result["payload"]["criteria"]

    mutation = tmp_path / "mutation.json"
    mutation.write_text(
        json.dumps({"case_id": "ordinary", "name": "note", "value": "reviewed"}),
        encoding="utf-8",
    )
    code, result, _ = _invoke(
        ["bundle", "--root", str(root), "set", "case.set_core_info", str(mutation)],
        capsys,
    )
    assert code == 0
    assert result["payload"]["is_synchronized"] is False
    assert result["payload"]["event_ids"][0].startswith("event-")


def test_bundle_cli_preserves_evidence_and_verification_details(
    tmp_path: Path, capsys
) -> None:
    root = _root(tmp_path)
    database = tmp_path / "datasets.sqlite3"
    code, _, _ = _invoke(
        ["bundle", "build", str(root), "--dataset", "demo", "--db", str(database)],
        capsys,
    )
    assert code == 0

    evidence_input = tmp_path / "evidence.json"
    evidence_input.write_text(
        json.dumps(
            {
                "case_id": "ordinary",
                "target": "obs-ordinary",
                "source": "doi:10.1000/cli",
                "locator": "Table 3",
                "details": {"source_lineage_id": "experiment:cli"},
            }
        ),
        encoding="utf-8",
    )
    code, _, _ = _invoke(
        [
            "bundle",
            "--root",
            str(root),
            "set",
            "case.add_evidence",
            str(evidence_input),
        ],
        capsys,
    )
    assert code == 0
    bundle = load_bundle(root, "demo", db=database)
    evidence = bundle.cases[0].evidence[0]

    verification_input = tmp_path / "verification.json"
    verification_input.write_text(
        json.dumps(
            {
                "case_id": "ordinary",
                "evidence_id": evidence["evidence_id"],
                "method": "AI",
                "verifier": "review-tool/current",
                "details": {
                    "procedure_version": "cli-review/current",
                    "verified_at": "2026-10-14T01:30:00Z",
                },
            }
        ),
        encoding="utf-8",
    )
    code, _, _ = _invoke(
        [
            "bundle",
            "--root",
            str(root),
            "set",
            "case.verify_evidence",
            str(verification_input),
        ],
        capsys,
    )
    assert code == 0

    facts = bundle._state.get_case("ordinary")
    assert facts["evidence"][0]["scientific_work_id"] == "doi:10.1000/cli"
    assert facts["evidence"][0]["source_lineage_id"] == "experiment:cli"
    assert facts["verifications"][0]["procedure_version"] == "cli-review/current"
    assert facts["verifications"][0]["verified_at"] == "2026-10-14T01:30:00Z"


def test_certify_ingest_query_archive_opt_in_and_protocol_pipeline(tmp_path: Path, capsys, monkeypatch) -> None:
    root = _root(tmp_path)
    database = tmp_path / "datasets.sqlite3"
    for arguments in (
        ["bundle", "build", str(root), "--db", str(database)],
        ["bundle", "--root", str(root), "synchronize"],
        ["certify", str(root)],
    ):
        code, result, _ = _invoke(arguments, capsys)
        assert code == 0 and result is not None

    code, result, _ = _invoke(
        [
            "ingest",
            str(root),
            "--db",
            str(database),
            "--include-incomplete",
            "--reason",
            "Preserve pending repair",
        ],
        capsys,
    )
    assert code == 0
    assert result["payload"]["included"]["count"] == 3
    assert result["payload"]["excluded"]["count"] == 0
    assert result["payload"]["idempotent"] is False

    code, default_subset, _ = _invoke(
        ["subset", "demo", "--db", str(database)], capsys
    )
    assert code == 0
    assert default_subset["payload"]["count"] == 2

    code, incomplete_subset, _ = _invoke(
        ["subset", "demo", "--archive-class", "incomplete", "--db", str(database)],
        capsys,
    )
    assert code == 0
    assert incomplete_subset["payload"]["count"] == 1
    monkeypatch.setattr(ds_cli.sys, "stdin", io.StringIO(json.dumps(incomplete_subset)))
    code, piped, _ = _invoke(
        [
            "get",
            "demo",
            "-",
            "--archive-class",
            "incomplete",
            "--db",
            str(database),
        ],
        capsys,
    )
    assert code == 0
    assert piped["payload"]["items"][0]["archive_class"] == "incomplete"


def test_export_uses_stable_public_indices_and_aligned_names(tmp_path: Path, capsys) -> None:
    root = _root(tmp_path)
    database = tmp_path / "datasets.sqlite3"
    for arguments in (
        ["bundle", "build", str(root), "--db", str(database)],
        ["bundle", "--root", str(root), "synchronize"],
        ["certify", str(root)],
        ["ingest", str(root), "--db", str(database)],
    ):
        code, _, _ = _invoke(arguments, capsys)
        assert code == 0
    output = tmp_path / "export"
    code, result, _ = _invoke(
        ["export", "demo", str(output), "--db", str(database)], capsys
    )
    assert code == 0
    assert result["payload"]["record_count"] == 2
    assert sorted(path.name for path in output.glob("struct-*")) == [
        "struct-000002.smi",
        "struct-000007.smi",
    ]
    assert sorted(path.name for path in output.glob("info-*")) == [
        "info-000002.json",
        "info-000007.json",
    ]
    assert (output / "summary.json").is_file()
