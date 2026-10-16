from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest

from hotpot.ds._communication import (
    CommunicationError,
    emit_envelope,
    extract_path,
    load_protocol,
    make_envelope,
    protocol_sha256,
    read_input,
)


def _payload(ids: list[str]) -> dict:
    return {
        "dataset": {
            "name": "spin",
            "release_id": "spin-r1",
            "version": "1",
            "version_serial": "001",
            "database": "/data/datasets.sqlite3",
        },
        "items": [{"id": value} for value in ids],
        "representations": [],
        "count": len(ids),
    }


def test_protocol_inventory_is_closed_and_every_route_has_a_schema() -> None:
    protocol = load_protocol()
    assert set(protocol["producers"]) == {
        "hotpot.ds.list",
        "hotpot.ds.info",
        "hotpot.ds.labels",
        "hotpot.ds.get",
        "hotpot.ds.subset",
        "hotpot.ds.search",
        "hotpot.ds.export",
        "hotpot.ds.detail",
        "hotpot.ds.detail.audit",
        "hotpot.ds.detail.query.claims",
        "hotpot.ds.detail.status",
        "hotpot.ds.bundle.build",
        "hotpot.ds.bundle.get",
        "hotpot.ds.bundle.set",
        "hotpot.ds.bundle.gate",
        "hotpot.ds.bundle.status",
        "hotpot.ds.bundle.synchronize",
        "hotpot.ds.bundle.rm",
        "hotpot.ds.certify",
        "hotpot.ds.ingest",
    }
    assert all(
        contract in protocol["contracts"]
        for channels in protocol["producers"].values()
        for contract in channels.values()
    )
    assert all(contract is not None for channels in protocol["producers"].values() for contract in channels.values())


def test_protocol_fingerprint_transitively_binds_exact_schema_bytes() -> None:
    protocol = load_protocol()
    canonical = json.dumps(
        protocol,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert protocol_sha256() == hashlib.sha256(canonical).hexdigest()
    package = Path(__import__("hotpot.ds").ds.__file__).resolve().parent
    for contract in protocol["contracts"].values():
        reference = contract["schema"]
        assert hashlib.sha256((package / reference["path"]).read_bytes()).hexdigest() == reference["sha256"]


def test_emit_writes_one_exact_four_field_envelope() -> None:
    stream = io.StringIO()
    emit_envelope("hotpot.ds.subset", _payload(["hp-spin[001]-0"]), stream=stream)
    envelope = json.loads(stream.getvalue())
    assert set(envelope) == {"key", "channel", "protocol_sha256", "payload"}
    assert stream.getvalue().count("\n") == 1


def test_schema_rejects_missing_and_unknown_payload_fields() -> None:
    missing = _payload(["one"])
    missing.pop("representations")
    with pytest.raises(CommunicationError, match="missing fields"):
        make_envelope("hotpot.ds.subset", missing)
    extra = _payload(["one"])
    extra["unexpected"] = True
    with pytest.raises(CommunicationError, match="unknown fields"):
        make_envelope("hotpot.ds.subset", extra)
    wrong = _payload(["one"])
    wrong["items"][0]["id"] = 1
    with pytest.raises(CommunicationError, match="JSON type"):
        make_envelope("hotpot.ds.subset", wrong)


def test_declared_paths_and_input_cardinality_are_enforced() -> None:
    document = {"payload": {"items": [{"id": "a"}, {"id": "b"}]}}
    assert extract_path(document, "payload.items[].id") == ["a", "b"]
    with pytest.raises(CommunicationError, match="invalid communication path"):
        extract_path(document, "payload.items[0].id")

    empty = make_envelope("hotpot.ds.subset", _payload([]))
    assert read_input("hotpot.ds.get", io.StringIO(json.dumps(empty)))["bindings"]["selectors"] == []
    with pytest.raises(CommunicationError, match="cardinality"):
        read_input("hotpot.ds.export", io.StringIO(json.dumps(empty)))

    duplicate = make_envelope("hotpot.ds.subset", _payload(["same", "same"]))
    with pytest.raises(CommunicationError, match="duplicate"):
        read_input("hotpot.ds.export", io.StringIO(json.dumps(duplicate)))


def test_input_fails_closed_for_wrong_digest_route_or_json_document() -> None:
    envelope = make_envelope("hotpot.ds.subset", _payload(["one"]))
    envelope["protocol_sha256"] = "0" * 64
    with pytest.raises(CommunicationError, match="fingerprint"):
        read_input("hotpot.ds.get", io.StringIO(json.dumps(envelope)))
    with pytest.raises(CommunicationError, match="one JSON document"):
        read_input("hotpot.ds.get", io.StringIO("{}\n{}"))
    with pytest.raises(CommunicationError, match="duplicate JSON key"):
        read_input("hotpot.ds.get", io.StringIO('{"key":"a","key":"b"}'))
    terminal = make_envelope(
        "hotpot.ds.list", {"database": "/data/store", "datasets": ["spin"]}
    )
    with pytest.raises(CommunicationError, match="does not accept"):
        read_input("hotpot.ds.get", io.StringIO(json.dumps(terminal)))


def test_structure_channel_remains_machine_data_and_is_consumable() -> None:
    envelope = make_envelope(
        "hotpot.ds.get", _payload(["hp-spin[001]-1"]), channel="structure"
    )
    result = read_input("hotpot.ds.export", io.StringIO(json.dumps(envelope)))
    assert result["bindings"]["selectors"] == ["hp-spin[001]-1"]
