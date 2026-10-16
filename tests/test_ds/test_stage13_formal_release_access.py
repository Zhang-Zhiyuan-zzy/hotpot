from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from hotpot.ds.cli import build_parser, run
from hotpot.ds.public._bundle_state import BUNDLE_SCHEMA
from hotpot.ds.public.bundle import Bundle
from hotpot.ds.public.catalog import load


def _ingested_release(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "bundle"
    intake = root / "evidence" / "intake" / "cases.jsonl"
    intake.parent.mkdir(parents=True)
    original = root / "evidence" / "original" / "chemist.txt"
    original.parent.mkdir()
    original.write_text("chemist assigned spin = 1\n", encoding="utf-8")
    structures = root / "structures"
    structures.mkdir()
    (structures / "sample.smi").write_text("O\n", encoding="utf-8")
    case = {
        "id": "case-1",
        "local_id": "local-1",
        "public_index": 5,
        "identifiers": {"CSD": "ACCESS1"},
        "core_info": {"sample": "sample-1"},
        "observations": [
            {"observation_id": "observation-1", "property": "spin", "value": 1}
        ],
        "claims": [
            {
                "claim_id": "claim-1",
                "property": "electronic_state",
                "value": "singlet",
            }
        ],
        "structures": [
            {
                "structure_id": "structure-1",
                "role": "primary",
                "format": "smi",
                "path": "structures/sample.smi",
                "primary": True,
            }
        ],
        "evidence": [
            {
                "evidence_id": "evidence-1",
                "target": "observation-1",
                "relation": "supports",
                "source_kind": "expert_statement",
                "source": "file:evidence/original/chemist.txt",
                "path": "evidence/original/chemist.txt",
                "quote": "chemist assigned spin = 1",
            }
        ],
    }
    intake.write_text(json.dumps(case) + "\n", encoding="utf-8")
    (root / "bundle.json").write_text(
        json.dumps(
            {
                "schema": BUNDLE_SCHEMA,
                "ds_name": "access",
                "release_id": "access-r1",
                "release_serial": 1,
                "version": "1.0",
                "title": "Formal access fixture",
                "description": "Target-owned evidence and structure access fixture",
                "license": "CC0-1.0",
                "access_policy": "open",
                "provenance": {"created_by": "test"},
                "core_labels": ["spin"],
                "core_info": ["sample"],
                "intake": {
                    "format": "jsonl",
                    "path": "evidence/intake/cases.jsonl",
                },
                "identity_schema": {
                    "components": [
                        {
                            "key": "sample",
                            "path": "core_info.sample",
                            "required": True,
                            "normalizer": "text",
                        }
                    ]
                },
                "required_structure_roles": ["primary"],
            }
        ),
        encoding="utf-8",
    )
    database = tmp_path / "datasets.sqlite3"
    bundle = Bundle(root, "access", db=database)
    bundle.synchronize()
    bundle.write_certificate()
    bundle.ingest()
    return root, database


def _invoke(
    arguments: list[str], capsys: pytest.CaptureFixture[str]
) -> tuple[int, dict | None, str]:
    code = run(build_parser().parse_args(arguments))
    captured = capsys.readouterr()
    envelope = json.loads(captured.out) if captured.out else None
    return code, envelope, captured.err


def test_formal_release_remains_fully_readable_without_bundle_products(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, database = _ingested_release(tmp_path)
    shutil.rmtree(root)

    data = load("access", db=database)[0]
    assert data.structures.smi == "O\n"
    assert data.info["facts"]["evidence"][0]["path"] == (
        "evidence/original/chemist.txt"
    )
    evidence_asset = data.info["evidence_assets"][0]
    evidence_path = Path(evidence_asset["path"])
    assert evidence_path.read_text(encoding="utf-8") == (
        "chemist assigned spin = 1\n"
    )
    assert evidence_path.is_relative_to(
        database.with_name(database.name + ".assets")
    )

    code, result, _ = _invoke(
        ["get", "access", "0", "--smi", "--db", str(database)], capsys
    )
    assert code == 0
    assert result["payload"]["items"][0]["evidence_assets"] == [evidence_asset]
    assert result["payload"]["representations"][0]["content"] == "O\n"

    output = tmp_path / "export"
    code, result, _ = _invoke(
        ["export", "access", str(output), "--db", str(database)], capsys
    )
    assert code == 0
    assert result["payload"]["record_count"] == 1
    exported_info = json.loads((output / "info.json").read_text(encoding="utf-8"))
    assert exported_info["evidence_assets"] == [evidence_asset]
    assert (output / "struct.smi").read_text(encoding="utf-8") == "O\n"

    code, result, _ = _invoke(
        [
            "detail",
            "query",
            "claims",
            "--dataset",
            "access",
            "--db",
            str(database),
        ],
        capsys,
    )
    assert code == 0
    assert result["payload"]["items"][0]["claim"]["claim_id"] == "claim-1"

    code, result, _ = _invoke(
        ["detail", "audit", "--db", str(database)], capsys
    )
    assert code == 0
    assert result["payload"]["passed"] is True
