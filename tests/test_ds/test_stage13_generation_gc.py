from __future__ import annotations

import json
import multiprocessing
import os
import sqlite3
from pathlib import Path

import pytest

from hotpot.ds.public import _generation as generation_module
from hotpot.ds.public._generation import (
    AssessmentGeneration,
    _GenerationLease,
    _collect_obsolete_generations,
)


def _generation(root: Path, name: str, digest: str) -> Path:
    directory = root / "assessments" / "generations" / name
    directory.mkdir(parents=True)
    (directory / "generation.json").write_text(
        json.dumps(
            {
                "generation_id": name,
                "generation_digest": digest,
            }
        ),
        encoding="utf-8",
    )
    return directory


def _crash_with_lease(root: str, generation_id: str, ready) -> None:
    lease = _GenerationLease(Path(root), generation_id)
    lease.__enter__()
    ready.send(True)
    ready.close()
    os._exit(0)


def test_collection_preserves_every_reference_kind_and_active_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "bundle"
    references = {
        "generation-current": "1" * 64,
        "generation-certificate": "2" * 64,
        "generation-receipt": "3" * 64,
        "generation-reader": "4" * 64,
        "generation-obsolete": "5" * 64,
    }
    for generation_id, digest in references.items():
        _generation(root, generation_id, digest)
    monkeypatch.setattr(
        generation_module,
        "_certificate_generation_reference",
        lambda root: (
            (
                "generation-certificate",
                references["generation-certificate"],
            ),
            False,
        ),
    )

    with _GenerationLease(root, "generation-reader"):
        report = _collect_obsolete_generations(
            root,
            current_reference=(
                "generation-current",
                references["generation-current"],
            ),
            receipt_generation_digests=(references["generation-receipt"],),
        )

    assert report["removed"] == ["generation-obsolete"]
    assert report["retained"] == {
        "generation-certificate": "certificate",
        "generation-current": "current",
        "generation-reader": "active_reader",
        "generation-receipt": "receipt",
    }
    for generation_id in references.keys() - {"generation-obsolete"}:
        assert (root / "assessments" / "generations" / generation_id).is_dir()

    second = _collect_obsolete_generations(
        root,
        current_reference=(
            "generation-current",
            references["generation-current"],
        ),
        receipt_generation_digests=(references["generation-receipt"],),
    )
    assert "generation-reader" in second["removed"]


def test_reader_iterator_holds_lease_until_iteration_finishes(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    generation_id = "generation-reader"
    digest = "6" * 64
    directory = _generation(root, generation_id, digest)
    database = sqlite3.connect(directory / "assessment.sqlite3")
    database.execute(
        "CREATE TABLE case_assessments(public_index INTEGER,case_id TEXT,assessment_json TEXT)"
    )
    database.executemany(
        "INSERT INTO case_assessments VALUES (?,?,?)",
        [(0, "case-a", '{"case_id":"case-a"}'), (1, "case-b", '{"case_id":"case-b"}')],
    )
    database.commit()
    database.close()
    generation = AssessmentGeneration(
        root,
        directory,
        {"generation_id": generation_id, "generation_digest": digest},
    )

    cases = generation.iter_cases()
    assert next(cases) == {"case_id": "case-a"}
    retained = _collect_obsolete_generations(root, current_reference=None)
    assert retained["retained"][generation_id] == "active_reader"
    assert list(cases) == [{"case_id": "case-b"}]

    removed = _collect_obsolete_generations(root, current_reference=None)
    assert removed["removed"] == [generation_id]


def test_crashed_reader_releases_os_lease_without_timeout(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    generation_id = "generation-crashed-reader"
    _generation(root, generation_id, "7" * 64)
    receive, send = multiprocessing.Pipe(duplex=False)
    process = multiprocessing.get_context("fork").Process(
        target=_crash_with_lease,
        args=(str(root), generation_id, send),
    )
    process.start()
    assert receive.recv() is True
    process.join(timeout=5)
    assert process.exitcode == 0

    report = _collect_obsolete_generations(root, current_reference=None)

    assert report["removed"] == [generation_id]
    assert (
        root / "assessments" / "leases" / f"{generation_id}.lock"
    ).is_file()


def test_unreadable_certificate_or_receipt_reference_retains_conservatively(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    generation_id = "generation-unknown-reference"
    _generation(root, generation_id, "8" * 64)
    (root / "certificate.json").write_text("not JSON", encoding="utf-8")

    certificate_report = _collect_obsolete_generations(
        root,
        current_reference=None,
    )
    assert certificate_report["conservative"] is True
    assert certificate_report["retained"][generation_id] == (
        "unknown_durable_reference"
    )

    (root / "certificate.json").unlink()
    receipt_report = _collect_obsolete_generations(
        root,
        current_reference=None,
        receipt_reference_unknown=True,
    )
    assert receipt_report["conservative"] is True
    assert receipt_report["retained"][generation_id] == (
        "unknown_durable_reference"
    )
