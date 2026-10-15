from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from hotpot.ds.public._bundle_state import BUNDLE_SCHEMA, BundleState
from hotpot.ds.public._generation import AssessmentGeneration, _AssessmentReader
from hotpot.ds.public.bundle import Bundle


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
        "assessment_scan": 0,
    }

    original_status = BundleState.status.fget
    original_source_digest = BundleState._compute_source_digest
    original_generation = Bundle._generation_from_status
    original_reader = AssessmentGeneration._reader
    original_lease = AssessmentGeneration.lease
    original_scan = _AssessmentReader.iter_cases

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

    def counted_scan(self):
        calls["assessment_scan"] += 1
        yield from original_scan(self)

    monkeypatch.setattr(BundleState, "status", property(counted_status))
    monkeypatch.setattr(BundleState, "_compute_source_digest", counted_source_digest)
    monkeypatch.setattr(Bundle, "_generation_from_status", counted_generation)
    monkeypatch.setattr(AssessmentGeneration, "_reader", counted_reader)
    monkeypatch.setattr(AssessmentGeneration, "lease", counted_lease)
    monkeypatch.setattr(AssessmentGeneration, "get_case", reject_per_case_read)
    monkeypatch.setattr(_AssessmentReader, "iter_cases", counted_scan)

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
        "assessment_scan": 1,
    }
