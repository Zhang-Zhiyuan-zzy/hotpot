"""Private, declarative stdin/stdout communication for ``hotpot ds``."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, TextIO


_PROTOCOL_FILE = Path(__file__).with_name("cli_communication.json")
_REQUIRED_ENVELOPE_KEYS = ("key", "channel", "protocol_sha256", "payload")
_SEGMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\[\])?$")
_JSON_TYPES = {
    "array": list,
    "boolean": bool,
    "integer": int,
    "null": type(None),
    "number": (int, float),
    "object": dict,
    "string": str,
}


class CommunicationError(ValueError):
    """A CLI document does not conform to the one current communication map."""


def _reject_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise CommunicationError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise CommunicationError(f"non-finite JSON number {value!r} is not allowed")


def _load_json(stream_or_text: Any) -> Any:
    try:
        if hasattr(stream_or_text, "read"):
            return json.load(
                stream_or_text,
                object_pairs_hook=_reject_pairs,
                parse_constant=_reject_constant,
            )
        return json.loads(
            stream_or_text,
            object_pairs_hook=_reject_pairs,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as error:
        raise CommunicationError("communication input is not one JSON document") from error


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _has_json_type(value: Any, type_name: str) -> bool:
    return isinstance(value, _JSON_TYPES[type_name]) and not (
        type_name in {"integer", "number"} and isinstance(value, bool)
    )


def _validate_binding_value(name: str, value: Any, binding: Mapping[str, Any]) -> None:
    value_type = binding["type"]
    if not _has_json_type(value, value_type):
        raise CommunicationError(
            f"binding {name!r} must have JSON type {value_type!r}"
        )
    items_type = binding.get("items_type")
    if items_type is not None and any(
        not _has_json_type(item, items_type) for item in value
    ):
        raise CommunicationError(
            f"items of binding {name!r} must have JSON type {items_type!r}"
        )
    properties = binding.get("properties")
    if properties is not None:
        if set(value) != set(properties):
            raise CommunicationError(
                f"binding {name!r} has unknown or missing properties"
            )
        for property_name, property_type in properties.items():
            if not _has_json_type(value[property_name], property_type):
                raise CommunicationError(
                    f"property {property_name!r} of binding {name!r} must have "
                    f"JSON type {property_type!r}"
                )


@lru_cache(maxsize=1)
def load_protocol() -> dict[str, Any]:
    """Load the single communication map shipped with this source revision."""
    with _PROTOCOL_FILE.open(encoding="utf-8") as stream:
        protocol = _load_json(stream)
    if set(protocol) != {"envelope_keys", "contracts", "producers", "consumers"}:
        raise CommunicationError("communication map has unknown or missing sections")
    _validate_protocol(protocol)
    return protocol


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol["envelope_keys"] != list(_REQUIRED_ENVELOPE_KEYS):
        raise CommunicationError("communication map has an invalid envelope definition")

    contracts = protocol["contracts"]
    producers = protocol["producers"]
    consumers = protocol["consumers"]
    if not all(isinstance(section, dict) for section in (contracts, producers, consumers)):
        raise CommunicationError("communication map sections must be objects")

    for contract, specification in contracts.items():
        if not isinstance(contract, str) or not contract:
            raise CommunicationError("communication contract names must be non-empty strings")
        if not isinstance(specification, dict) or set(specification) != {"bindings"}:
            raise CommunicationError(f"malformed communication contract {contract!r}")
        bindings = specification["bindings"]
        if not isinstance(bindings, dict):
            raise CommunicationError(f"bindings for {contract!r} must be an object")
        for name, binding in bindings.items():
            if not isinstance(name, str) or not name:
                raise CommunicationError("binding names must be non-empty strings")
            if not isinstance(binding, dict):
                raise CommunicationError(f"malformed binding {name!r}")
            operation = {"path", "constant"}.intersection(binding)
            allowed = operation | {"type", "items_type", "properties"}
            if len(operation) != 1 or "type" not in binding or set(binding) - allowed:
                raise CommunicationError(f"malformed binding {name!r}")
            if "path" in binding:
                path = binding["path"]
                if (
                    not isinstance(path, str)
                    or not path
                    or any(
                        _SEGMENT.fullmatch(segment) is None
                        for segment in path.split(".")
                    )
                ):
                    raise CommunicationError(f"invalid communication path {path!r}")
            value_type = binding["type"]
            if value_type not in _JSON_TYPES:
                raise CommunicationError(f"binding {name!r} has an unknown type")
            items_type = binding.get("items_type")
            if items_type is not None and (
                value_type != "array" or items_type not in _JSON_TYPES
            ):
                raise CommunicationError(f"binding {name!r} has an invalid item type")
            properties = binding.get("properties")
            if properties is not None and (
                value_type != "object"
                or not isinstance(properties, dict)
                or any(
                    not isinstance(key, str) or value not in _JSON_TYPES
                    for key, value in properties.items()
                )
            ):
                raise CommunicationError(f"binding {name!r} has invalid properties")
            if "constant" in binding:
                _validate_binding_value(name, binding["constant"], binding)

    for key, channels in producers.items():
        if not isinstance(key, str) or not key or not isinstance(channels, dict):
            raise CommunicationError("producer definitions are malformed")
        if not channels:
            raise CommunicationError(f"producer {key!r} has no channels")
        for channel, contract in channels.items():
            if not isinstance(channel, str) or not channel:
                raise CommunicationError(f"producer {key!r} has an invalid channel")
            if contract is not None and contract not in contracts:
                raise CommunicationError(
                    f"producer {key!r} references unknown contract {contract!r}"
                )

    for consumer, definition in consumers.items():
        if (
            not isinstance(consumer, str)
            or not consumer
            or not isinstance(definition, dict)
            or set(definition) != {"accepts"}
            or not isinstance(definition["accepts"], dict)
        ):
            raise CommunicationError("consumer definitions are malformed")
        for contract, policy in definition["accepts"].items():
            if contract not in contracts:
                raise CommunicationError(
                    f"consumer {consumer!r} accepts unknown contract {contract!r}"
                )
            if (
                not isinstance(policy, dict)
                or set(policy) != {"cardinality", "duplicates"}
                or policy["duplicates"] not in {"preserve", "reject"}
            ):
                raise CommunicationError(f"consumer policy for {consumer!r} is malformed")
            cardinality = policy["cardinality"]
            if (
                not isinstance(cardinality, dict)
                or set(cardinality) != {"field", "minimum", "maximum"}
            ):
                raise CommunicationError(
                    f"consumer cardinality for {consumer!r} is malformed"
                )
            field = cardinality["field"]
            minimum = cardinality["minimum"]
            maximum = cardinality["maximum"]
            if field not in contracts[contract]["bindings"]:
                raise CommunicationError(
                    f"consumer {consumer!r} cardinality field is not a binding"
                )
            if contracts[contract]["bindings"][field]["type"] != "array":
                raise CommunicationError(
                    f"consumer {consumer!r} cardinality field is not array-typed"
                )
            if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 0:
                raise CommunicationError(
                    f"consumer {consumer!r} cardinality minimum is invalid"
                )
            if maximum is not None and (
                not isinstance(maximum, int)
                or isinstance(maximum, bool)
                or maximum < minimum
            ):
                raise CommunicationError(
                    f"consumer {consumer!r} cardinality maximum is invalid"
                )


@lru_cache(maxsize=1)
def protocol_sha256() -> str:
    """Return the SHA-256 fingerprint of the canonical communication map."""
    return hashlib.sha256(_canonical_json(load_protocol())).hexdigest()


def _producer_contract(key: str, channel: str) -> str | None:
    producers = load_protocol()["producers"]
    if key not in producers:
        raise CommunicationError(f"unknown producer key {key!r}")
    channels = producers[key]
    if channel not in channels:
        raise CommunicationError(
            f"unknown channel {channel!r} for producer {key!r}"
        )
    return channels[channel]


def extract_path(document: Mapping[str, Any], path: str) -> Any:
    """Resolve a safe dotted path, with ``[]`` denoting list traversal."""
    segments = path.split(".")
    if not path or any(_SEGMENT.fullmatch(segment) is None for segment in segments):
        raise CommunicationError(f"invalid communication path {path!r}")

    nodes: list[Any] = [document]
    expanded = False
    for segment in segments:
        traverse = segment.endswith("[]")
        name = segment[:-2] if traverse else segment
        next_nodes: list[Any] = []
        for node in nodes:
            if not isinstance(node, Mapping) or name not in node:
                raise CommunicationError(f"communication path {path!r} is missing")
            value = node[name]
            if traverse:
                if not isinstance(value, list):
                    raise CommunicationError(
                        f"communication path {path!r} requires a list at {name!r}"
                    )
                next_nodes.extend(value)
            else:
                next_nodes.append(value)
        expanded = expanded or traverse
        nodes = next_nodes
    if expanded:
        return nodes
    return nodes[0]


def _resolve_bindings(envelope: Mapping[str, Any], contract: str) -> dict[str, Any]:
    contracts = load_protocol()["contracts"]
    if contract not in contracts:
        raise CommunicationError(f"unknown communication contract {contract!r}")
    specification = contracts[contract]
    if set(specification) != {"bindings"}:
        raise CommunicationError(f"malformed communication contract {contract!r}")

    resolved: dict[str, Any] = {}
    for name, binding in specification["bindings"].items():
        if "path" in binding:
            value = extract_path(envelope, binding["path"])
        else:
            value = binding["constant"]
        _validate_binding_value(name, value, binding)
        resolved[name] = value
    return resolved


def _validate_envelope(envelope: Mapping[str, Any]) -> None:
    if set(envelope) != set(load_protocol()["envelope_keys"]):
        raise CommunicationError("envelope has unknown or missing keys")
    if not isinstance(envelope["key"], str) or not envelope["key"]:
        raise CommunicationError("envelope key must be a non-empty string")
    if not isinstance(envelope["channel"], str) or not envelope["channel"]:
        raise CommunicationError("envelope channel must be a non-empty string")
    if envelope["protocol_sha256"] != protocol_sha256():
        raise CommunicationError("communication protocol fingerprint does not match")
    _producer_contract(envelope["key"], envelope["channel"])


def make_envelope(
    key: str, payload: Any, *, channel: str = "default"
) -> dict[str, Any]:
    """Build and validate one current-protocol output envelope."""
    contract = _producer_contract(key, channel)
    envelope = {
        "key": key,
        "channel": channel,
        "protocol_sha256": protocol_sha256(),
        "payload": payload,
    }
    if contract is not None:
        _resolve_bindings(envelope, contract)
    return envelope


def encode_envelope(key: str, payload: Any, *, channel: str = "default") -> bytes:
    """Encode one envelope as one compact UTF-8 JSON document."""
    return _canonical_json(make_envelope(key, payload, channel=channel)) + b"\n"


def emit_envelope(
    key: str,
    payload: Any,
    *,
    channel: str = "default",
    stream: TextIO | None = None,
) -> None:
    """Write exactly one UTF-8 JSON envelope to a text or buffered stream."""
    output = sys.stdout if stream is None else stream
    document = encode_envelope(key, payload, channel=channel)
    buffer = getattr(output, "buffer", None)
    if buffer is not None:
        buffer.write(document)
        buffer.flush()
    else:
        output.write(document.decode("utf-8"))
        output.flush()


def consume_envelope(
    envelope: Mapping[str, Any], consumer_key: str
) -> dict[str, Any]:
    """Resolve an input envelope into config-defined, consumer-ready bindings."""
    if not isinstance(envelope, Mapping):
        raise CommunicationError("communication input must be a JSON object")
    _validate_envelope(envelope)

    contract = _producer_contract(envelope["key"], envelope["channel"])
    if contract is None:
        raise CommunicationError("producer output is terminal and cannot be consumed")
    consumers = load_protocol()["consumers"]
    if consumer_key not in consumers:
        raise CommunicationError(f"unknown consumer key {consumer_key!r}")
    accepted = consumers[consumer_key]["accepts"]
    if contract not in accepted:
        raise CommunicationError(
            f"consumer {consumer_key!r} does not accept contract {contract!r}"
        )

    bindings = _resolve_bindings(envelope, contract)
    policy = accepted[contract]
    cardinality = policy["cardinality"]
    field = cardinality["field"]
    values = bindings.get(field)
    if not isinstance(values, list):
        raise CommunicationError(f"cardinality field {field!r} is not a list")
    if len(values) < cardinality["minimum"]:
        raise CommunicationError(
            f"consumer {consumer_key!r} requires at least "
            f"{cardinality['minimum']} input item(s)"
        )
    maximum = cardinality["maximum"]
    if maximum is not None and len(values) > maximum:
        raise CommunicationError(
            f"consumer {consumer_key!r} accepts at most {maximum} input item(s)"
        )
    if policy["duplicates"] == "reject":
        encoded = [_canonical_json(value) for value in values]
        if len(encoded) != len(set(encoded)):
            raise CommunicationError(
                f"consumer {consumer_key!r} does not accept duplicate input items"
            )

    return {
        "contract": contract,
        "producer_key": envelope["key"],
        "producer_channel": envelope["channel"],
        "bindings": bindings,
    }


def read_input(consumer_key: str, stream: TextIO | None = None) -> dict[str, Any]:
    """Read one JSON document from stdin and resolve it for a consumer."""
    source = sys.stdin if stream is None else stream
    raw = source.read()
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise CommunicationError("communication input is not UTF-8") from error
    envelope = _load_json(raw)
    return consume_envelope(envelope, consumer_key)


__all__ = [
    "CommunicationError",
    "consume_envelope",
    "emit_envelope",
    "encode_envelope",
    "extract_path",
    "load_protocol",
    "make_envelope",
    "protocol_sha256",
    "read_input",
]
