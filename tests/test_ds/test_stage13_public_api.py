from __future__ import annotations

import inspect

import hotpot.ds as ds


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


def test_search_and_gate_signatures_remain_compact() -> None:
    assert _parameters(ds.DataSearcher) == ("criteria",)
    assert _parameters(ds.DatasetSearcher) == ("criteria",)
    assert _parameters(ds.SmartsSearcher) == ("smarts", "semantics")

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
