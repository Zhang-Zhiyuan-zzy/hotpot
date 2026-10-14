"""Closed, hash-bound stdin/stdout protocol for :mod:`hotpot.ds.cli`."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, TextIO


_PACKAGE_ROOT = Path(__file__).resolve().parent
_PROTOCOL_FILE = _PACKAGE_ROOT / "cli_communication.json"
_ENVELOPE_KEYS = ("key", "channel", "protocol_sha256", "payload")
_HEX = re.compile(r"^[0-9a-f]{64}$")
_SEGMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\[\])?$")


class CommunicationError(ValueError):
    """One protocol document, schema, or route is not current and valid."""


def _reject_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise CommunicationError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise CommunicationError(f"non-finite JSON number {value!r} is not allowed")


def _load_json(source: Any) -> Any:
    try:
        options = {
            "object_pairs_hook": _reject_pairs,
            "parse_constant": _reject_constant,
        }
        return (
            json.load(source, **options)
            if hasattr(source, "read")
            else json.loads(source, **options)
        )
    except json.JSONDecodeError as error:
        raise CommunicationError("communication input is not one JSON document") from error


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CommunicationError("communication value is not canonical JSON") from error


def _schema_file(reference: Mapping[str, Any]) -> Path:
    if set(reference) != {"path", "pointer", "sha256"}:
        raise CommunicationError("schema reference has unknown or missing fields")
    relative = reference["path"]
    if (
        not isinstance(relative, str)
        or not relative
        or "\\" in relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
    ):
        raise CommunicationError("schema path is not a safe package-relative path")
    path = (_PACKAGE_ROOT / relative).resolve()
    try:
        path.relative_to(_PACKAGE_ROOT)
    except ValueError as error:
        raise CommunicationError("schema path escapes the package") from error
    digest = reference["sha256"]
    if not isinstance(digest, str) or _HEX.fullmatch(digest) is None:
        raise CommunicationError("schema digest must be lowercase SHA-256")
    try:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise CommunicationError(f"cannot read communication schema {relative!r}") from error
    if actual != digest:
        raise CommunicationError(f"communication schema digest changed: {relative}")
    return path


def _resolve_pointer(document: Any, pointer: str) -> Any:
    if pointer == "#":
        return document
    if not isinstance(pointer, str) or not pointer.startswith("#/"):
        raise CommunicationError("schema pointer must be a local JSON pointer")
    value = document
    for raw in pointer[2:].split("/"):
        part = raw.replace("~1", "/").replace("~0", "~")
        if not isinstance(value, Mapping) or part not in value:
            raise CommunicationError(f"schema pointer {pointer!r} does not resolve")
        value = value[part]
    return value


@lru_cache(maxsize=1)
def load_protocol() -> dict[str, Any]:
    """Load and validate the one protocol map shipped by this source tree."""

    try:
        protocol = _load_json(_PROTOCOL_FILE.read_text(encoding="utf-8"))
    except OSError as error:
        raise CommunicationError("cannot read the communication map") from error
    if set(protocol) != {"envelope_keys", "contracts", "producers", "consumers"}:
        raise CommunicationError("communication map has unknown or missing sections")
    if protocol["envelope_keys"] != list(_ENVELOPE_KEYS):
        raise CommunicationError("communication map has an invalid envelope")
    contracts = protocol["contracts"]
    producers = protocol["producers"]
    consumers = protocol["consumers"]
    if not all(isinstance(value, dict) for value in (contracts, producers, consumers)):
        raise CommunicationError("communication map sections must be objects")
    for name, contract in contracts.items():
        if not isinstance(name, str) or not name or not isinstance(contract, dict):
            raise CommunicationError("communication contracts are malformed")
        if set(contract) != {"schema", "bindings"} or not isinstance(contract["bindings"], dict):
            raise CommunicationError(f"contract {name!r} is malformed")
        path = _schema_file(contract["schema"])
        schema_document = _load_json(path.read_text(encoding="utf-8"))
        schema = _resolve_pointer(schema_document, contract["schema"]["pointer"])
        if not isinstance(schema, Mapping):
            raise CommunicationError(f"schema for {name!r} is not an object")
        for binding, specification in contract["bindings"].items():
            if (
                not isinstance(binding, str)
                or not binding
                or not isinstance(specification, Mapping)
                or set(specification) != {"path"}
                or not isinstance(specification["path"], str)
                or any(
                    _SEGMENT.fullmatch(part) is None
                    for part in specification["path"].split(".")
                )
            ):
                raise CommunicationError(f"contract {name!r} has a malformed binding")
    for key, channels in producers.items():
        if not isinstance(key, str) or not key or not isinstance(channels, dict) or not channels:
            raise CommunicationError("communication producer is malformed")
        for channel, contract in channels.items():
            if not isinstance(channel, str) or not channel or contract not in contracts:
                raise CommunicationError(f"producer {key!r} has an invalid route")
    for key, consumer in consumers.items():
        if (
            not isinstance(key, str)
            or not key
            or not isinstance(consumer, dict)
            or set(consumer) != {"accepts"}
            or not isinstance(consumer["accepts"], dict)
        ):
            raise CommunicationError("communication consumer is malformed")
        for contract, policy in consumer["accepts"].items():
            if contract not in contracts or set(policy) != {"cardinality", "duplicates"}:
                raise CommunicationError(f"consumer {key!r} has an invalid contract")
            cardinality = policy["cardinality"]
            if (
                policy["duplicates"] not in {"preserve", "reject"}
                or set(cardinality) != {"field", "minimum", "maximum"}
                or cardinality["field"] not in contracts[contract]["bindings"]
                or not isinstance(cardinality["minimum"], int)
                or isinstance(cardinality["minimum"], bool)
                or cardinality["minimum"] < 0
                or (
                    cardinality["maximum"] is not None
                    and (
                        not isinstance(cardinality["maximum"], int)
                        or isinstance(cardinality["maximum"], bool)
                        or cardinality["maximum"] < cardinality["minimum"]
                    )
                )
            ):
                raise CommunicationError(f"consumer {key!r} has an invalid policy")
    return protocol


@lru_cache(maxsize=1)
def protocol_sha256() -> str:
    """Return the digest of the map, including every bound schema digest."""

    return hashlib.sha256(_canonical_bytes(load_protocol())).hexdigest()


def _contract_schema(contract_name: str) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    contract = load_protocol()["contracts"][contract_name]
    path = _schema_file(contract["schema"])
    document = _load_json(path.read_text(encoding="utf-8"))
    return document, _resolve_pointer(document, contract["schema"]["pointer"])


def _json_type(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return type(value) is bool
    if expected == "null":
        return value is None
    raise CommunicationError(f"unsupported JSON Schema type {expected!r}")


def _validate_schema(
    value: Any,
    schema: Mapping[str, Any],
    document: Mapping[str, Any],
    location: str,
) -> None:
    if "$ref" in schema:
        _validate_schema(value, _resolve_pointer(document, schema["$ref"]), document, location)
    for candidate in schema.get("allOf", []):
        _validate_schema(value, candidate, document, location)
    if "anyOf" in schema:
        for candidate in schema["anyOf"]:
            try:
                _validate_schema(value, candidate, document, location)
                break
            except CommunicationError:
                continue
        else:
            raise CommunicationError(f"{location} does not match any allowed schema")
    if "const" in schema and value != schema["const"]:
        raise CommunicationError(f"{location} must equal {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        raise CommunicationError(f"{location} is not an allowed value")
    declared_type = schema.get("type")
    if isinstance(declared_type, list):
        if not any(_json_type(value, expected) for expected in declared_type):
            raise CommunicationError(f"{location} has the wrong JSON type")
    elif declared_type is not None and not _json_type(value, declared_type):
        raise CommunicationError(f"{location} must have JSON type {declared_type}")
    if isinstance(value, Mapping):
        required = schema.get("required", [])
        missing = [name for name in required if name not in value]
        if missing:
            raise CommunicationError(f"{location} is missing fields: {', '.join(missing)}")
        properties = schema.get("properties", {})
        patterns = schema.get("patternProperties", {})
        if schema.get("additionalProperties") is False:
            unknown = sorted(
                name
                for name in set(value) - set(properties)
                if not any(re.fullmatch(pattern, name) for pattern in patterns)
            )
            if unknown:
                raise CommunicationError(f"{location} has unknown fields: {', '.join(unknown)}")
        for name, child in properties.items():
            if name in value:
                _validate_schema(value[name], child, document, f"{location}.{name}")
        for name, child_value in value.items():
            for pattern, child_schema in patterns.items():
                if re.fullmatch(pattern, name):
                    _validate_schema(child_value, child_schema, document, f"{location}.{name}")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            raise CommunicationError(f"{location} has too few items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise CommunicationError(f"{location} has too many items")
        if schema.get("uniqueItems") and len({_canonical_bytes(item) for item in value}) != len(value):
            raise CommunicationError(f"{location} contains duplicate items")
        if "items" in schema:
            for index, item in enumerate(value):
                _validate_schema(item, schema["items"], document, f"{location}[{index}]")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            raise CommunicationError(f"{location} is too short")
        if "pattern" in schema and re.fullmatch(schema["pattern"], value) is None:
            raise CommunicationError(f"{location} does not match its pattern")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise CommunicationError(f"{location} is below its minimum")


def _producer_contract(key: str, channel: str) -> str:
    producers = load_protocol()["producers"]
    if key not in producers:
        raise CommunicationError(f"unknown producer key {key!r}")
    if channel not in producers[key]:
        raise CommunicationError(f"unknown channel {channel!r} for producer {key!r}")
    return producers[key][channel]


def extract_path(document: Mapping[str, Any], path: str) -> Any:
    """Resolve a declared dotted path; ``[]`` traverses one JSON array."""

    segments = path.split(".")
    if not path or any(_SEGMENT.fullmatch(segment) is None for segment in segments):
        raise CommunicationError(f"invalid communication path {path!r}")
    nodes: list[Any] = [document]
    expanded = False
    for segment in segments:
        traverse = segment.endswith("[]")
        name = segment[:-2] if traverse else segment
        following: list[Any] = []
        for node in nodes:
            if not isinstance(node, Mapping) or name not in node:
                raise CommunicationError(f"communication path {path!r} is missing")
            child = node[name]
            if traverse:
                if not isinstance(child, list):
                    raise CommunicationError(f"communication path {path!r} requires an array")
                following.extend(child)
            else:
                following.append(child)
        nodes = following
        expanded = expanded or traverse
    return nodes if expanded else nodes[0]


def _resolve_bindings(envelope: Mapping[str, Any], contract: str) -> dict[str, Any]:
    return {
        name: extract_path(envelope, specification["path"])
        for name, specification in load_protocol()["contracts"][contract]["bindings"].items()
    }


def make_envelope(key: str, payload: Any, *, channel: str = "default") -> dict[str, Any]:
    """Create one validated current-protocol envelope."""

    contract = _producer_contract(key, channel)
    document, schema = _contract_schema(contract)
    _validate_schema(payload, schema, document, "payload")
    envelope = {
        "key": key,
        "channel": channel,
        "protocol_sha256": protocol_sha256(),
        "payload": payload,
    }
    _canonical_bytes(envelope)
    return envelope


def emit_envelope(
    key: str,
    payload: Any,
    *,
    channel: str = "default",
    stream: TextIO | None = None,
) -> dict[str, Any]:
    """Write exactly one validated JSON envelope followed by one newline."""

    envelope = make_envelope(key, payload, channel=channel)
    target = stream if stream is not None else sys.stdout
    target.write(_canonical_bytes(envelope).decode("utf-8") + "\n")
    return envelope


def read_input(consumer: str, stream: TextIO | None = None) -> dict[str, Any]:
    """Read one envelope and resolve the consumer's declared bindings."""

    if consumer not in load_protocol()["consumers"]:
        raise CommunicationError(f"unknown communication consumer {consumer!r}")
    envelope = _load_json(stream if stream is not None else sys.stdin)
    if not isinstance(envelope, Mapping) or set(envelope) != set(_ENVELOPE_KEYS):
        raise CommunicationError("input envelope has unknown or missing keys")
    if envelope["protocol_sha256"] != protocol_sha256():
        raise CommunicationError("communication protocol fingerprint does not match")
    if not isinstance(envelope["key"], str) or not isinstance(envelope["channel"], str):
        raise CommunicationError("input key and channel must be strings")
    contract = _producer_contract(envelope["key"], envelope["channel"])
    accepted = load_protocol()["consumers"][consumer]["accepts"]
    if contract not in accepted:
        raise CommunicationError(
            f"consumer {consumer!r} does not accept producer contract {contract!r}"
        )
    document, schema = _contract_schema(contract)
    _validate_schema(envelope["payload"], schema, document, "payload")
    bindings = _resolve_bindings(envelope, contract)
    policy = accepted[contract]
    values = bindings[policy["cardinality"]["field"]]
    if not isinstance(values, list):
        raise CommunicationError("consumer cardinality binding must be an array")
    minimum = policy["cardinality"]["minimum"]
    maximum = policy["cardinality"]["maximum"]
    if len(values) < minimum or (maximum is not None and len(values) > maximum):
        raise CommunicationError("communication input cardinality is not allowed")
    if policy["duplicates"] == "reject" and len({_canonical_bytes(item) for item in values}) != len(values):
        raise CommunicationError("communication input contains duplicate selections")
    return {"envelope": dict(envelope), "contract": contract, "bindings": bindings}


__all__ = [
    "CommunicationError",
    "emit_envelope",
    "extract_path",
    "load_protocol",
    "make_envelope",
    "protocol_sha256",
    "read_input",
]
