from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import pytest

import hotpot.ds.public.bundle as bundle_module
import hotpot.ds.public.certificate as certificate_module
from hotpot.ds.public._bundle_state import BUNDLE_SCHEMA, BundleState
from hotpot.ds.public._generation import AssessmentGeneration, _AssessmentReader
from hotpot.ds.public.bundle import Bundle, SealedIntegrityError


def _selection_bundle(tmp_path: Path, size: int) -> Bundle:
    root = tmp_path / f"bundle-{size}"
    intake = root / "evidence" / "intake" / "cases.jsonl"
    intake.parent.mkdir(parents=True)
    structures = root / "structures"
    structures.mkdir()
    (structures / "shared.smi").write_text("O\n", encoding="utf-8")
    cases = []
    for index in range(size):
        case = {
            "id": f"case-{index}",
            "local_id": f"local-{index}",
            "public_index": index,
            "core_info": {"sample": f"sample-{index}"},
            "observations": [
                {
                    "observation_id": f"observation-{index}",
                    "property": "spin",
                    "value": index + 1,
                }
            ],
            "structures": [
                {
                    "structure_id": f"structure-{index}",
                    "role": "primary",
                    "format": "smi",
                    "path": "structures/shared.smi",
                    "primary": True,
                }
            ],
        }
        if index % 2 == 0:
            case["attestations"] = [
                {
                    "attestation_id": f"attestation-{index}",
                    "authority": "User",
                    "actor": "curator@example.test",
                    "approved": True,
                }
            ]
        cases.append(case)
    intake.write_text(
        "".join(json.dumps(case) + "\n" for case in cases),
        encoding="utf-8",
    )
    (root / "bundle.json").write_text(
        json.dumps(
            {
                "schema": BUNDLE_SCHEMA,
                "ds_name": "selection-efficiency",
                "release_id": f"selection-efficiency-r{size}",
                "release_serial": 1,
                "version": "1.0",
                "title": "Selection efficiency",
                "description": "Deterministic selection-work test",
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
    bundle = Bundle(root, "selection-efficiency", db=tmp_path / f"store-{size}.sqlite3")
    bundle.synchronize()
    return bundle


@pytest.mark.parametrize("size", [3, 24])
def test_case_selection_uses_one_generation_reader_independent_of_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    size: int,
) -> None:
    bundle = _selection_bundle(tmp_path, size)
    collection = bundle.cases
    calls = {
        "status": 0,
        "source_digest": 0,
        "generation": 0,
        "reader": 0,
        "lease": 0,
        "indexed_selection": 0,
    }

    original_status = BundleState.status.fget
    original_source_digest = BundleState._compute_source_digest
    original_generation = Bundle._generation_from_status
    original_reader = AssessmentGeneration._reader
    original_lease = AssessmentGeneration.lease
    original_select = _AssessmentReader.select_cases

    def counted_status(self):
        calls["status"] += 1
        return original_status(self)

    def counted_source_digest(self, connection, manifest):
        calls["source_digest"] += 1
        return original_source_digest(self, connection, manifest)

    def counted_generation(self, status):
        calls["generation"] += 1
        return original_generation(self, status)

    @contextmanager
    def counted_reader(self):
        calls["reader"] += 1
        with original_reader(self) as reader:
            yield reader

    @contextmanager
    def counted_lease(self):
        calls["lease"] += 1
        with original_lease(self) as generation:
            yield generation

    def reject_per_case_read(self, case_id):
        raise AssertionError(f"per-Case assessment read used for {case_id}")

    def counted_select(self, criteria):
        calls["indexed_selection"] += 1
        yield from original_select(self, criteria)

    monkeypatch.setattr(BundleState, "status", property(counted_status))
    monkeypatch.setattr(BundleState, "_compute_source_digest", counted_source_digest)
    monkeypatch.setattr(Bundle, "_generation_from_status", counted_generation)
    monkeypatch.setattr(AssessmentGeneration, "_reader", counted_reader)
    monkeypatch.setattr(AssessmentGeneration, "lease", counted_lease)
    monkeypatch.setattr(AssessmentGeneration, "get_case", reject_per_case_read)
    monkeypatch.setattr(_AssessmentReader, "select_cases", counted_select)

    selected = collection.select(attestation="User")

    assert [case.id for case in selected] == [
        f"case-{index}" for index in range(0, size, 2)
    ]
    assert selected[0] is collection[0]
    assert calls == {
        "status": 1,
        "source_digest": 1,
        "generation": 1,
        "reader": 1,
        "lease": 1,
        "indexed_selection": 1,
    }


def test_derived_case_filters_use_indexed_selection_and_stable_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _selection_bundle(tmp_path, 12)
    statements = []
    original_select = _AssessmentReader.select_cases

    def traced_select(self, criteria):
        self._connection.set_trace_callback(statements.append)
        yield from original_select(self, criteria)

    def reject_full_scan(self):
        raise AssertionError("derived selection used the full assessment iterator")

    monkeypatch.setattr(_AssessmentReader, "select_cases", traced_select)
    monkeypatch.setattr(_AssessmentReader, "iter_cases", reject_full_scan)

    selected = bundle.cases.select(
        archive_class="benchmark",
        quality_level="B0",
        min_quality="B0",
        evidence_level="E0",
        min_evidence="E0",
        benchmark_status="admitted",
    )

    assert [case.id for case in selected] == [
        f"case-{index}" for index in range(0, 12, 2)
    ]
    query = next(
        statement
        for statement in statements
        if "FROM case_assessments WHERE" in statement
    )
    assert "archive_class=" in query
    assert "quality_level>=" in query
    assert "evidence_level>=" in query
    assert "benchmark_status=" in query
    assert "ORDER BY public_index,case_id" in query
    generation = bundle._require_generation()
    with generation._reader() as reader:
        plan = reader._connection.execute(
            "EXPLAIN QUERY PLAN "
            "SELECT public_index,case_id,assessment_json "
            "FROM case_assessments WHERE archive_class=? "
            "ORDER BY public_index,case_id",
            ("benchmark",),
        ).fetchall()
    assert any(
        "case_assessments_archive_selection" in str(row[3]) for row in plan
    )


def test_case_collection_construction_and_selection_do_not_load_all_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _selection_bundle(tmp_path, 8)

    def reject_id_materialization(self):
        raise AssertionError("CaseCollection eagerly materialized every Case id")

    monkeypatch.setattr(Bundle, "_case_ids", reject_id_materialization)
    collection = bundle.cases
    selected = collection.select(archive_class="benchmark")

    assert [case.id for case in selected] == [
        "case-0",
        "case-2",
        "case-4",
        "case-6",
    ]


def test_ingest_uses_three_bulk_scans_and_two_certificate_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _selection_bundle(tmp_path, 6)
    bundle.write_certificate()
    scans = 0
    certificate_proofs = 0
    original_scan = _AssessmentReader.iter_cases
    original_verify = certificate_module.verify_bundle_certificate

    def counted_scan(self):
        nonlocal scans
        scans += 1
        yield from original_scan(self)

    def counted_verify(*args, **kwargs):
        nonlocal certificate_proofs
        certificate_proofs += 1
        return original_verify(*args, **kwargs)

    def reject_point_read(self, case_id):
        raise AssertionError(f"per-Case assessment read used for {case_id}")

    monkeypatch.setattr(_AssessmentReader, "iter_cases", counted_scan)
    monkeypatch.setattr(
        certificate_module, "verify_bundle_certificate", counted_verify
    )
    monkeypatch.setattr(AssessmentGeneration, "get_case", reject_point_read)

    result = bundle.ingest()

    assert result["included_count"] == 6
    assert result["excluded_count"] == 0
    assert scans == 3
    assert certificate_proofs == 2

    scans = 0
    assert bundle._authoritative_receipt() is not None
    assert scans == 1


def test_ingestion_snapshot_hashes_one_shared_asset_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _selection_bundle(tmp_path, 8)
    bundle.write_certificate()
    shared = (bundle.root / "structures" / "shared.smi").resolve()
    calls = 0
    original_hash = bundle_module._hash_file

    def counted_hash(path: Path):
        nonlocal calls
        if Path(path).resolve() == shared:
            calls += 1
        return original_hash(path)

    monkeypatch.setattr(bundle_module, "_hash_file", counted_hash)
    snapshot = bundle._ingestion_snapshot("0" * 64, False)
    list(snapshot["cases"])

    assert calls == 1


def test_ingestion_rejects_fact_assessment_count_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _selection_bundle(tmp_path, 3)
    bundle.write_certificate()
    original_scan = _AssessmentReader.iter_cases

    def truncated_scan(self):
        rows = iter(original_scan(self))
        yield next(rows)

    monkeypatch.setattr(_AssessmentReader, "iter_cases", truncated_scan)

    with pytest.raises(SealedIntegrityError, match="counts do not match"):
        bundle._ingestion_snapshot("0" * 64, False)
