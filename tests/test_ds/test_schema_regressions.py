from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hotpot.ds.database import DatasetSchemaError, DatasetStore
from hotpot.ds.public.certificate import _established_spec


def test_noncurrent_store_requires_rebuild(tmp_path: Path) -> None:
    path = tmp_path / "noncurrent.sqlite3"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE schema_metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        INSERT INTO schema_metadata VALUES('schema_version','old');
        INSERT INTO schema_metadata VALUES('record_standard','hotpot-dataset/old');
        """
    )
    connection.commit()
    connection.close()

    with pytest.raises(DatasetSchemaError, match="rebuild"):
        DatasetStore(path).initialize()


def test_established_spec_uses_greatest_release_serial(tmp_path: Path) -> None:
    path = tmp_path / "established.sqlite3"
    store = DatasetStore(path)
    store.initialize()
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute(
        "INSERT INTO datasets VALUES(?,?,?)", ("demo", "Demo", "Description")
    )
    connection.executemany(
        "INSERT INTO quality_certificates VALUES(?,?,?,?,?,?)",
        (
            ("cert-old", "a" * 64, "admission", "b" * 64, "2026-01-02", "{}"),
            ("cert-new", "c" * 64, "admission", "d" * 64, "2026-01-01", "{}"),
        ),
    )
    connection.executemany(
        "INSERT INTO ingestions(ingestion_id,certificate_id,asset_root) VALUES(?,?,?)",
        ((2, "cert-old", "."), (1, "cert-new", ".")),
    )
    connection.executemany(
        """INSERT INTO dataset_releases(
        release_id,dataset_id,version,release_serial,quality_level,
        core_labels_json,license,access_policy,provenance_json,quality_json,ingestion_id)
        VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
        (
            (
                "demo.serial-1",
                "demo",
                "1.0",
                1,
                "B0_collection",
                '["old_label"]',
                "CC0-1.0",
                "open",
                "{}",
                '{"core_info":["old_info"]}',
                2,
            ),
            (
                "demo.serial-2",
                "demo",
                "2.0",
                2,
                "B1_candidate",
                '["new_label"]',
                "CC0-1.0",
                "open",
                "{}",
                '{"core_info":["new_info"]}',
                1,
            ),
        ),
    )
    connection.commit()
    connection.close()

    specification = _established_spec(store, "demo")

    assert specification["release_id"] == "demo.serial-2"
    assert specification["quality_level"] == "B1_candidate"
    assert specification["core_labels"] == ["new_label"]
    assert specification["core_info"] == ["new_info"]
