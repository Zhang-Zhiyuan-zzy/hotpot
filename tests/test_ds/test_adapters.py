import csv
import sqlite3
from pathlib import Path

import pytest

from hotpot.ds.detail import DatasetStore
from hotpot.ds.adapters import convert_csd_electronic_state
from hotpot.ds.quality import assess_bundle
from hotpot.ds.cli import build_parser, run
from hotpot.ds._paths import DATA_ROOT, SOURCES_ROOT


def test_csd_spin_text_remains_unresolved_and_subset_queries_do_not_leak(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    source = SOURCES_ROOT / "csd-electronic-state" / "electronic_states.sqlite3"
    if not source.is_file():
        pytest.skip("portable CSD evidence database is not installed")

    bundle = convert_csd_electronic_state(
        source,
        license_name="test-local-CSD-license",
        access_policy="restricted",
        scope="spin",
    )
    certificate = assess_bundle(bundle, DATA_ROOT, profile="benchmark")
    admission_certificate = assess_bundle(bundle, DATA_ROOT, profile="admission")
    export_certificate = assess_bundle(bundle, DATA_ROOT, profile="export")

    assert len(bundle.observations) == 0
    assert len(bundle.claims) == 811
    assert len(bundle.evidence) == 811
    assert len(bundle.structures) == 0
    assert len(bundle.descriptors) == 350
    assert len(bundle.entity_relations) == 28
    assert {row.status for row in bundle.claims} == {"unresolved"}
    assert {row.evidence_level for row in bundle.claims} == {"E1_extracted"}
    assert len(bundle.claim_memberships) == len(bundle.claims)
    assert certificate.passed
    assert not export_certificate.passed
    assert export_certificate.checks[-1].code == "export_policy"

    store = DatasetStore(tmp_path / "spin.sqlite3")
    store.ingest(bundle, admission_certificate, DATA_ROOT)
    expected_by_subset = {
        subset.subset_id: {
            membership.claim_id
            for membership in bundle.claim_memberships
            if membership.subset_id == subset.subset_id
        }
        for subset in bundle.subsets
    }
    returned_by_subset = {
        subset_id: {row["claim_id"] for row in store.iter_claims(subset_id=subset_id)}
        for subset_id in expected_by_subset
    }
    assert returned_by_subset == expected_by_subset
    assert set().union(*returned_by_subset.values()) == {
        claim.claim_id for claim in bundle.claims
    }
    assert sum(map(len, returned_by_subset.values())) == len(bundle.claims)
    first_claim = bundle.claims[0]
    detail = store.get_claim(first_claim.claim_id)
    assert detail is not None
    assert detail["evidence_level"] == "E1_extracted"
    assert len(detail["evidence"]) == 1

    args = build_parser().parse_args(
        [
            "detail", "query", "claims", "--db", str(store.path), "--evidence-level",
            "E1_extracted", "--limit", "1", "--format", "json",
        ]
    )
    assert run(args) == 0
    assert "E1_extracted" in capsys.readouterr().out


def test_csd_name_conflicts_remain_visible_regression_sentinels() -> None:
    source = SOURCES_ROOT / "csd-electronic-state" / "electronic_states.sqlite3"
    labels_path = SOURCES_ROOT / "oxidation" / "gold" / "sco95" / "labels.csv"
    if not source.is_file() or not labels_path.is_file():
        pytest.skip("CSD evidence and curated SCO95 labels are not installed")

    with labels_path.open(newline="", encoding="utf-8") as handle:
        curated = {
            row["structure_refcode"]: int(row["metal_oxidation_state"])
            for row in csv.DictReader(handle)
        }
    connection = sqlite3.connect(source)
    try:
        extracted = {
            refcode: {
                row[0]
                for row in connection.execute(
                    "SELECT oxidation_state FROM entry_oxidation_states WHERE refcode=? AND element='Fe'",
                    (refcode,),
                )
            }
            for refcode in ("ACAHUJ", "ZEKDOK", "ACAHUJ02", "ZEKDOK01")
        }
    finally:
        connection.close()

    assert curated["ACAHUJ"] == curated["ZEKDOK"] == 2
    assert extracted["ACAHUJ"] == extracted["ZEKDOK"] == {3}
    assert extracted["ACAHUJ02"] == extracted["ZEKDOK01"] == {2}


def test_csd_adapter_preserves_known_cross_dataset_disagreement(tmp_path: Path) -> None:
    source = SOURCES_ROOT / "csd-electronic-state" / "electronic_states.sqlite3"
    if not source.is_file():
        pytest.skip("portable CSD evidence database is not installed")

    bundle = convert_csd_electronic_state(
        source,
        license_name="test-local-CSD-license",
        access_policy="restricted",
        version="conflict-test",
        scope="oxidation",
        limit=725,
    )
    record_id = next(row.record_id for row in bundle.records if row.local_id == "ACAHUJ")
    assert not [row for row in bundle.observations if row.record_id == record_id]
    claim = next(row for row in bundle.claims if row.record_id == record_id)

    assert claim.status == "unresolved"
    assert claim.evidence_level == "E2_explicit_reported"
    assert claim.metadata["owner_disposition"] == "quarantined_cross_dataset_disagreement"
    assert claim.metadata["benchmark_target_eligible"] is False
    certificate = assess_bundle(bundle, DATA_ROOT, profile="benchmark")
    assert certificate.passed
    assert certificate.assessed_levels[
        next(row.subset_id for row in bundle.subsets if row.name.startswith("Oxidation T1"))
    ] == "B1_candidate"

    store = DatasetStore(tmp_path / "conflict.sqlite3")
    store.ingest(bundle, assess_bundle(bundle, DATA_ROOT), DATA_ROOT)
    detail = store.get_claim(claim.claim_id)
    assert detail is not None
    assert {row["link_role"] for row in detail["evidence"]} >= {
        "verbatim_source", "owner_adjudication"
    }

