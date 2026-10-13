from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

import hotpot.ds as ds
from hotpot.ds.cli import build_parser, run


def _write_folder_submission(root: Path, version: str) -> Path:
    root.mkdir()
    structures = root / "structures"
    structures.mkdir()
    (structures / "water.smi").write_text("O\n", encoding="utf-8")
    manifest = {
        "schema_version": "hotpot-ds-submission/current",
        "dataset": {
            "name": "demo",
            "version": version,
            "title": "Demo",
            "description": "Public ingestion workflow",
            "license": "CC0-1.0",
            "access_policy": "open",
            "provenance": {"created_by": "test"},
        },
        "records": [
            {
                "id": "water",
                "identifiers": {"local_identifier": "water"},
                "metadata": {},
                "structures": [
                    {
                        "role": "reported_structure",
                        "primary": True,
                        "representations": {
                            "smi": {"path": "structures/water.smi"}
                        },
                    }
                ],
                "labels": {
                    "oxidation_state": {
                        "value": 2,
                        "evidence_level": "E2_explicit_reported",
                        "evidence": [
                            {
                                "source": "paper",
                                "evidence_type": "reported_value",
                                "locator": "page 1",
                                "quote": "The oxidation state is +2.",
                                "role": "supports",
                            }
                        ],
                    }
                },
            }
        ],
        "sources": {
            "paper": {
                "source_type": "publication",
                "citation": "Test source",
                "doi": None,
                "license": "CC0-1.0",
                "access_policy": "open",
                "metadata": {},
            }
        },
    }
    (root / "dataset.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return root


def test_folder_certification_and_existing_dataset_integration(tmp_path: Path) -> None:
    first = ds.load_bundle(_write_folder_submission(tmp_path / "release-1", "1.0"))
    database = tmp_path / "datasets.sqlite3"
    certificate = ds.write_certificate(
        first,
        tmp_path / "release-1.certificate.json",
        core_labels=["oxidation_state"],
        core_info=[],
    )

    ds.store.ingest(first, certificate=certificate, db=database)

    second = ds.load_bundle(_write_folder_submission(tmp_path / "release-2", "2.0"))
    established = ds.load(ds_name="demo", db=database)
    update_certificate = ds.write_certificate(
        second,
        tmp_path / "release-2.certificate.json",
        dataset=established,
    )

    ds.store.ingest(second, certificate=update_certificate, db=database)
    latest = ds.load(ds_name="demo", db=database)
    assert latest.version == "2.0"
    assert latest[0].id == "hp-demo[002]-0"
    assert latest[0].structures.smi == "O\n"


def test_cli_subset_applies_inclusive_quality_and_evidence_gates(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _write_folder_submission(tmp_path / "submission", "1.0")
    bundle = ds.load_bundle(source)
    certificate = ds.write_certificate(
        bundle,
        tmp_path / "certificate.json",
        core_labels=["oxidation_state"],
    )
    database = tmp_path / "datasets.sqlite3"
    ds.store.ingest(bundle, certificate=certificate, db=database)

    def subset(*options: str):
        arguments = build_parser().parse_args(
            ["subset", "demo", *options, "--db", str(database)]
        )
        assert run(arguments) == 0
        envelope = json.loads(capsys.readouterr().out)
        assert envelope["key"] == "hotpot.ds.subset"
        return envelope["payload"]["items"]

    assert len(subset()) == 1
    assert len(subset("-q", "0", "-e", "2")) == 1
    assert subset("--quality", "1") == []
    assert subset("--evidence", "3") == []


def test_compact_sqlite_submission_uses_the_same_public_gate(tmp_path: Path) -> None:
    smiles = tmp_path / "water.smi"
    smiles.write_text("O\n", encoding="utf-8")
    source = tmp_path / "submission.sqlite3"
    connection = sqlite3.connect(source)
    connection.executescript(
        """
        CREATE TABLE dataset(
          name TEXT,version TEXT,title TEXT,description TEXT,license TEXT,
          access_policy TEXT,provenance_json TEXT);
        CREATE TABLE data(
          id TEXT PRIMARY KEY,metadata_json TEXT,identifiers_json TEXT);
        CREATE TABLE structures(data_id TEXT,format TEXT,path TEXT);
        CREATE TABLE labels(data_id TEXT,name TEXT,value_json TEXT);
        INSERT INTO dataset VALUES(
          'sqlite-demo','1.0','SQLite demo','Compact exchange','CC0-1.0','open',
          '{"created_by":"test"}');
        INSERT INTO data VALUES('water','{}','{"CSD_identifier":"WATER"}');
        INSERT INTO structures VALUES('water','smi','water.smi');
        """
    )
    connection.commit()
    connection.close()

    bundle = ds.load_bundle(source)
    certificate = ds.write_certificate(
        bundle,
        tmp_path / "sqlite.certificate.json",
        core_labels=[],
        core_info=[],
    )
    database = tmp_path / "sqlite-store.sqlite3"

    ds.store.ingest(bundle, certificate=certificate, db=database)
    data = ds.load(ds_name="sqlite-demo", db=database).CSD_identifier["WATER"]
    assert data.structures.smi == "O\n"


def test_public_store_rejects_mismatched_and_unenveloped_certificates(
    tmp_path: Path,
) -> None:
    first = ds.load_bundle(_write_folder_submission(tmp_path / "first", "1.0"))
    certificate = ds.write_certificate(
        first,
        tmp_path / "certificate.json",
        core_labels=["oxidation_state"],
    )
    second = ds.load_bundle(_write_folder_submission(tmp_path / "second", "2.0"))

    with pytest.raises(RuntimeError, match="does not match"):
        ds.store.ingest(
            second,
            certificate=certificate,
            db=tmp_path / "mismatch.sqlite3",
        )

    envelope = json.loads(certificate.read_text(encoding="utf-8"))
    unenveloped = tmp_path / "unenveloped-certificate.json"
    unenveloped.write_text(json.dumps(envelope["certificate"]), encoding="utf-8")
    with pytest.raises(RuntimeError, match="current public envelope"):
        ds.store.ingest(
            first,
            certificate=unenveloped,
            db=tmp_path / "unenveloped.sqlite3",
        )


def test_cli_ingest_envelopes_the_public_result(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _write_folder_submission(tmp_path / "cli", "1.0")
    bundle = ds.load_bundle(source)
    certificate = ds.write_certificate(
        bundle,
        tmp_path / "cli-certificate.json",
        core_labels=["oxidation_state"],
    )
    arguments = build_parser().parse_args(
        [
            "ingest",
            str(source),
            "--certificate",
            str(certificate),
            "--db",
            str(tmp_path / "cli-store.sqlite3"),
        ]
    )

    assert run(arguments) == 0
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["key"] == "hotpot.ds.ingest"
    result = envelope["payload"]["result"]
    assert set(result) == {
        "dataset",
        "versions",
        "releases",
        "records",
        "structures",
        "observations",
        "claims",
        "descriptors",
    }
