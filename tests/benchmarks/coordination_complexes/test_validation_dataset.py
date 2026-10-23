from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from .validation_dataset import (
    FORMAL_CASE_INDICES,
    NONFINITE_KEY,
    ROUTES,
    SOURCE_SHA256,
    normalize_nonfinite_json,
    prepare_bundle,
    publish_bundle,
    route_case_id,
    route_case_public_index,
    validate_route_results,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _record(
    index: int,
    route: str,
    source_lines: list[str],
    source_sha256: str,
    *,
    structure: Path | None = None,
    failed_value: float = math.inf,
) -> dict[str, object]:
    passed = structure is not None
    decisions = {f"A{number:02d}": passed for number in range(1, 11)}
    result: dict[str, object] = {
        "case_index": index,
        "source_smiles": source_lines[index - 1],
        "source_sha256": source_sha256,
        "semantic_graph_sha256": hashlib.sha256(f"graph-{index}".encode()).hexdigest(),
        "route": route,
        "run_id": f"run-{route.lower()}",
        "seed": 20260921 + index,
        "status": "passed" if passed else "optimization_failed",
        "passed": passed,
        "acceptance": {
            **decisions,
            "diagnostics": [] if passed else [{"gradient": failed_value}],
            "recovery_actions": [],
        },
        "optimization": {
            "best_energy": -10.0 if passed else failed_value,
            "rms_gradient": 0.1 if passed else failed_value,
            "max_gradient": 0.2 if passed else failed_value,
        },
        "provenance": {
            "validation_round": "plan17-validation-v1",
            "hotpot_commit": "1" * 40,
            "dependency_versions": {"rdkit": "2026.09.1"},
            "controls": [
                {
                    "name": "cohort",
                    "sha256": "2" * 64,
                    "locator": "cohort.json",
                }
            ],
        },
        "settings": {
            "cbond": {"metal_index": 0, "donor_indices": [1]},
            "epochs": 10,
        },
        "manual_review": {
            "status": "passed" if passed else "not_applicable",
            "flags": [],
        },
    }
    if structure is not None:
        result["selected_structure"] = {
            "path": str(structure),
            "sha256": _digest(structure),
            "format": "smi",
            "atom_row_mapping": [0],
        }
    return result


def _small_matrix(tmp_path: Path) -> tuple[Path, list[dict[str, object]]]:
    source = tmp_path / "extractants.smi"
    source.write_text("C\n", encoding="utf-8")
    selected = tmp_path / "selected.smi"
    selected.write_text("C\n", encoding="utf-8")
    source_lines = ["C"]
    source_sha256 = _digest(source)
    records = [
        _record(
            1,
            route,
            source_lines,
            source_sha256,
            structure=selected if route in {"O0", "F1", "F2", "C1"} else None,
        )
        for route in ROUTES
    ]
    return source, records


def test_formal_matrix_is_910_stably_ordered_route_cases() -> None:
    source = Path(__file__).parents[3] / "molecules" / "extractant" / "extractants.smi"
    source_lines = source.read_text(encoding="utf-8").splitlines()
    records = [
        _record(index, route, source_lines, SOURCE_SHA256)
        for index in reversed(FORMAL_CASE_INDICES)
        for route in reversed(ROUTES)
    ]

    ordered = validate_route_results(records, source)

    assert len(ordered) == 910
    assert (ordered[0]["case_index"], ordered[0]["route"]) == (1, "O0")
    assert (ordered[-1]["case_index"], ordered[-1]["route"]) == (184, "C2")
    assert route_case_id(1, "O0") == "cbond-ff-001-o0"
    assert route_case_public_index(184, "C2") == 909


def test_nonfinite_diagnostics_are_strict_json() -> None:
    normalized = normalize_nonfinite_json(
        {"nan": math.nan, "positive": math.inf, "negative": -math.inf}
    )

    assert normalized == {
        "nan": {NONFINITE_KEY: "nan"},
        "negative": {NONFINITE_KEY: "-inf"},
        "positive": {NONFINITE_KEY: "+inf"},
    }
    json.dumps(normalized, allow_nan=False)


def test_result_contract_forbids_borrowed_or_missing_selected_structure(
    tmp_path: Path,
) -> None:
    source, records = _small_matrix(tmp_path)
    failed = records[-1]
    failed["selected_structure"] = records[0]["selected_structure"]

    with pytest.raises(ValueError, match="failed result must not contain"):
        validate_route_results(
            records,
            source,
            case_indices=(1,),
            expected_source_sha256=None,
        )

    failed.pop("selected_structure")
    records[0].pop("selected_structure")
    with pytest.raises(ValueError, match="passed result requires exactly"):
        validate_route_results(
            records,
            source,
            case_indices=(1,),
            expected_source_sha256=None,
        )


def test_prepare_materializes_only_success_structures_and_user_authority(
    tmp_path: Path,
) -> None:
    source, records = _small_matrix(tmp_path)
    bundle_root = tmp_path / "bundle"

    summary = prepare_bundle(
        records,
        source_path=source,
        bundle_root=bundle_root,
        authorization=_authorization(),
        result_root=tmp_path,
        case_indices=(1,),
        expected_source_sha256=None,
    )

    assert summary["route_case_count"] == 5
    assert summary["coordinate_assets"] == 4
    assert summary["missing_coordinate_assets"] == 1
    cases = [
        json.loads(line)
        for line in (bundle_root / "evidence/intake/cases.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [case["id"] for case in cases] == [
        "cbond-ff-001-o0",
        "cbond-ff-001-f1",
        "cbond-ff-001-f2",
        "cbond-ff-001-c1",
        "cbond-ff-001-c2",
    ]
    assert len(list((bundle_root / "structures").iterdir())) == 4
    assert cases[-1]["structures"] == []
    assert {
        attestation["authority"]
        for case in cases
        for attestation in case["attestations"]
    } == {"User"}
    raw_results = (bundle_root / "evidence/control/route-results.jsonl").read_text(
        encoding="utf-8"
    )
    assert "Infinity" not in raw_results
    assert '"nonfinite_float":"+inf"' in raw_results


def test_prepare_rejects_raw_hotpot_authority(tmp_path: Path) -> None:
    source, records = _small_matrix(tmp_path)
    authorization = _authorization()
    authorization["authority"] = "Hotpot"

    with pytest.raises(ValueError, match="authority must be User"):
        prepare_bundle(
            records,
            source_path=source,
            bundle_root=tmp_path / "bundle",
            authorization=authorization,
            case_indices=(1,),
            expected_source_sha256=None,
        )


def test_explicit_publication_preserves_incomplete_results_and_emits_inventory(
    tmp_path: Path,
) -> None:
    source, records = _small_matrix(tmp_path)
    bundle_root = tmp_path / "bundle"
    database = tmp_path / "store" / "datasets.sqlite3"
    inventory_path = tmp_path / "candidate-inventory.json"
    prepare_bundle(
        records,
        source_path=source,
        bundle_root=bundle_root,
        authorization=_authorization(),
        case_indices=(1,),
        expected_source_sha256=None,
    )

    with pytest.raises(PermissionError, match="confirm_ingest"):
        publish_bundle(
            bundle_root,
            database=database,
            reason="Preserve failed route results",
        )
    assert not database.exists()

    result = publish_bundle(
        bundle_root,
        database=database,
        reason="Preserve failed route results",
        confirm_ingest=True,
        inventory_output=inventory_path,
    )

    receipt = result["receipt"]
    inventory = result["inventory"]
    assert receipt["include_incomplete"] is True
    assert receipt["reason"] == "Preserve failed route results"
    assert receipt["included_count"] == 5
    assert receipt["excluded_count"] == 0
    assert inventory["release"]["cases"] == 5
    assert inventory["release"]["archive_class"] == {
        "benchmark": 4,
        "incomplete": 1,
    }
    assert inventory["benchmark_matrix"]["coordinate_assets"] == 4
    assert json.loads(inventory_path.read_text(encoding="utf-8")) == inventory
