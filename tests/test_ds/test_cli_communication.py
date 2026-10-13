from __future__ import annotations

import hashlib
import io
import json
from importlib.resources import files

import pytest

from hotpot.ds._communication import (
    CommunicationError,
    consume_envelope,
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
            "release_id": "spin.v1",
            "version": "1",
            "version_serial": "001",
            "database": "/data/datasets.sqlite3",
        },
        "items": [{"id": value} for value in ids],
    }


def test_protocol_map_is_an_installed_package_resource() -> None:
    resource = files("hotpot.ds").joinpath("cli_communication.json")
    assert resource.is_file()
    assert json.loads(resource.read_text(encoding="utf-8"))["envelope_keys"] == [
        "key",
        "channel",
        "protocol_sha256",
        "payload",
    ]


def test_protocol_fingerprint_is_canonical_sha256_without_version_field() -> None:
    protocol = load_protocol()
    canonical = json.dumps(
        protocol,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert protocol_sha256() == hashlib.sha256(canonical).hexdigest()
    assert len(protocol_sha256()) == 64
    assert "protocol_version" not in protocol
    assert protocol["envelope_keys"] == [
        "key",
        "channel",
        "protocol_sha256",
        "payload",
    ]


def test_emit_is_one_utf8_json_document_with_exact_envelope_keys() -> None:
    stream = io.StringIO()
    emit_envelope("hotpot.ds.subset", _payload(["hp-spin[001]-0"]), stream=stream)

    envelope = json.loads(stream.getvalue())
    assert set(envelope) == {"key", "channel", "protocol_sha256", "payload"}
    assert envelope["key"] == "hotpot.ds.subset"
    assert envelope["channel"] == "default"
    assert stream.getvalue().count("\n") == 1


def test_safe_path_supports_dotted_fields_and_list_traversal() -> None:
    document = {"payload": {"items": [{"id": "a"}, {"id": "b"}]}}
    assert extract_path(document, "payload.items[].id") == ["a", "b"]
    assert extract_path(document, "payload.items") == document["payload"]["items"]

    with pytest.raises(CommunicationError, match="invalid communication path"):
        extract_path(document, "payload.items[0].id")
    with pytest.raises(CommunicationError, match="missing"):
        extract_path(document, "payload.unknown")


def test_consume_resolves_contract_paths_and_constants() -> None:
    envelope = make_envelope(
        "hotpot.ds.search", _payload(["hp-spin[001]-3"]), channel="smarts"
    )
    resolved = consume_envelope(envelope, "hotpot.ds.get")

    assert resolved["contract"] == "hotpot.ds.data_collection"
    assert resolved["producer_channel"] == "smarts"
    assert resolved["bindings"] == {
        "dataset": _payload([])["dataset"],
        "selectors": ["hp-spin[001]-3"],
    }


def test_get_accepts_zero_items_but_export_requires_one() -> None:
    envelope = make_envelope("hotpot.ds.subset", _payload([]))
    assert consume_envelope(envelope, "hotpot.ds.get")["bindings"]["selectors"] == []
    with pytest.raises(CommunicationError, match="at least 1"):
        consume_envelope(envelope, "hotpot.ds.export")


def test_duplicate_policy_preserves_get_and_rejects_export() -> None:
    envelope = make_envelope("hotpot.ds.subset", _payload(["same", "same"]))
    assert consume_envelope(envelope, "hotpot.ds.get")["bindings"]["selectors"] == [
        "same",
        "same",
    ]
    with pytest.raises(CommunicationError, match="duplicate"):
        consume_envelope(envelope, "hotpot.ds.export")


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda value: value.update(extra=True), "unknown or missing keys"),
        (
            lambda value: value.update(protocol_sha256="0" * 64),
            "fingerprint does not match",
        ),
        (lambda value: value.update(key="hotpot.ds.unknown"), "unknown producer"),
        (lambda value: value.update(channel="unknown"), "unknown channel"),
    ],
)
def test_consume_strictly_rejects_invalid_envelopes(mutation, message: str) -> None:
    envelope = make_envelope("hotpot.ds.subset", _payload(["hp-spin[001]-0"]))
    mutation(envelope)
    with pytest.raises(CommunicationError, match=message):
        consume_envelope(envelope, "hotpot.ds.get")


def test_terminal_output_and_malformed_input_are_rejected() -> None:
    terminal = make_envelope("hotpot.ds.list", ["spin"])
    with pytest.raises(CommunicationError, match="terminal"):
        consume_envelope(terminal, "hotpot.ds.get")
    with pytest.raises(CommunicationError, match="one JSON document"):
        read_input("hotpot.ds.get", io.StringIO("{}\n{}"))
    with pytest.raises(CommunicationError, match="JSON object"):
        read_input("hotpot.ds.get", io.StringIO("[]"))
    with pytest.raises(CommunicationError, match="duplicate JSON key"):
        read_input(
            "hotpot.ds.get",
            io.StringIO('{"key":"a","key":"b"}'),
        )
    with pytest.raises(CommunicationError, match="non-finite"):
        read_input("hotpot.ds.get", io.StringIO('{"value":NaN}'))


def test_binding_types_are_checked_on_output_and_input() -> None:
    malformed = _payload(["valid"])
    malformed["items"] = [{"id": 2}]
    with pytest.raises(CommunicationError, match="items of binding"):
        make_envelope("hotpot.ds.subset", malformed)

    malformed = _payload(["valid"])
    del malformed["dataset"]["release_id"]
    with pytest.raises(CommunicationError, match="unknown or missing properties"):
        make_envelope("hotpot.ds.subset", malformed)


def test_get_structure_channel_is_a_consumable_data_collection() -> None:
    envelope = make_envelope(
        "hotpot.ds.get", _payload(["hp-spin[001]-1"]), channel="structure"
    )
    result = read_input("hotpot.ds.export", io.StringIO(json.dumps(envelope)))
    assert result["bindings"]["selectors"] == ["hp-spin[001]-1"]
