from __future__ import annotations

import inspect
from collections.abc import Mapping

import hotpot.ds as ds
from hotpot.ds.public.case import Case, CaseCollection, CaseView
from hotpot.ds.public.structures import Structures


EXPECTED_PUBLIC_API = [
    "datasets_list",
    "load",
    "Dataset",
    "Data",
    "BaseSearcher",
    "DataSearcher",
    "DatasetSearcher",
    "SmartsSearcher",
    "load_bundle",
    "Bundle",
    "write_certificate",
]


def _parameters(callable_object):
    return tuple(inspect.signature(callable_object).parameters)


def test_first_level_api_is_closed_and_ordered() -> None:
    assert ds.__all__ == EXPECTED_PUBLIC_API
    assert dir(ds) == sorted(EXPECTED_PUBLIC_API)
    assert not hasattr(ds, "store")
    assert not hasattr(ds, "Certificate")
    assert not hasattr(ds, "Configure")


def test_first_level_function_signature_tree() -> None:
    assert _parameters(ds.datasets_list) == ("db",)
    assert _parameters(ds.load) == (
        "ds_name",
        "version",
        "db",
        "archive_class",
    )
    assert _parameters(ds.load_bundle) == ("root", "ds_name", "db")
    assert _parameters(ds.Bundle) == ("root", "ds_name", "db", "gates")
    assert _parameters(ds.Bundle.from_certificate) == ("path", "db")
    assert _parameters(ds.write_certificate) == ("bundle",)
    assert _parameters(ds.Bundle.ingest) == (
        "self",
        "include_incomplete",
        "reason",
    )


def test_returned_object_signature_tree_is_complete() -> None:
    assert _parameters(ds.Dataset.subset) == ("self", "criteria")
    assert _parameters(Structures.__call__) == ("self",)
    assert _parameters(Structures.content) == ("self", "fmt")
    assert _parameters(Structures.path) == ("self", "fmt")
    assert _parameters(Structures.molecule) == ("self",)
    assert _parameters(CaseCollection.select) == ("self", "criteria")
    assert _parameters(Case.set_core_info) == ("self", "name", "value", "reason")
    assert _parameters(Case.add_evidence) == (
        "self",
        "target",
        "source",
        "relation",
        "locator",
        "note",
        "details",
    )
    assert _parameters(Case.verify_evidence) == (
        "self",
        "evidence_id",
        "method",
        "verifier",
        "note",
        "details",
    )
    assert _parameters(Case.attest_benchmark) == (
        "self",
        "approved",
        "authority",
        "actor",
        "reason",
    )

    assert {
        name
        for name, value in Case.__dict__.items()
        if not name.startswith("_") and isinstance(value, property)
    } == {
        "id",
        "info",
        "core_info",
        "structures",
        "labels",
        "observations",
        "claims",
        "evidence",
        "attestations",
        "evidence_levels",
        "evidence_level",
        "quality_level",
        "archive_class",
        "benchmark",
        "benchmark_status",
        "reasons",
        "assessment",
    }
    assert {
        name
        for name, value in Case.__dict__.items()
        if not name.startswith("_") and callable(value)
    } == {
        "set_core_info",
        "remove_core_info",
        "add_structure",
        "replace_structure",
        "withdraw_structure",
        "add_observation",
        "replace_observation",
        "withdraw_observation",
        "add_claim",
        "replace_claim",
        "withdraw_claim",
        "add_evidence",
        "verify_evidence",
        "withdraw_verification",
        "replace_evidence",
        "withdraw_evidence",
        "attest_benchmark",
        "withdraw_attestation",
    }
    assert set(CaseCollection.__dict__) >= {"select"}
    assert set(CaseView.__dict__) >= {"__len__", "__iter__", "__getitem__"}


def test_data_exposes_parent_dataset_and_structures_helpers() -> None:
    dataset = ds.Dataset(
        object(),
        {"dataset": "demo", "title": "Demo", "description": "Demo"},
        {
            "release": "demo.v001",
            "version": "1",
            "release_serial": 1,
            "metadata": {},
        },
        [],
        archive_classes=("ordinary", "benchmark"),
    )
    data = ds.Data(
        dataset,
        {
            "id": "case-1",
            "local_id": "case-1",
            "public_index": 0,
            "facts": {},
            "assessment": {
                "quality_level": "B0",
                "evidence_level": None,
                "archive_class": "ordinary",
            },
        },
    )

    assert data.dataset is dataset
    assert isinstance(data.structures, Structures)
    assert data.structures() == []
    assert data.structures.content("smi") is None
    assert data.structures.path("mol2") is None


def test_search_and_gate_signatures_remain_compact() -> None:
    assert _parameters(ds.DataSearcher) == ("criteria",)
    assert _parameters(ds.DatasetSearcher) == ("criteria",)
    assert _parameters(ds.SmartsSearcher) == ("smarts", "semantics")
    assert isinstance(ds.DataSearcher(example=1).criteria, Mapping)
    assert isinstance(ds.DatasetSearcher(example=1).criteria, Mapping)

    from hotpot.ds.gates import BenchmarkGate, EvidenceGate, QualityGate

    assert _parameters(EvidenceGate) == (
        "checks",
        "level_checks",
        "require_locator",
    )
    assert _parameters(QualityGate) == (
        "checks",
        "level_checks",
        "structure_checks",
    )
    assert _parameters(BenchmarkGate) == (
        "checklist",
        "direct_entry",
        "vetoes",
        "numeric_tolerances",
    )
