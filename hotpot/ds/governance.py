"""Private Hotpot governance trust boundary.

The public dataset API deliberately does not re-export this module.  A
``Hotpot`` benchmark attestation is effective only after this module verifies
it against the tracked genesis public key and a complete, canonical governance
registry snapshot.  Private key material is never loaded from package files,
Bundle files, environment variables, or the dataset store.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional


ROOT_SCHEMA = "hotpot.ds.governance-root/current"
REGISTRY_SCHEMA = "hotpot.ds.governance-registry/current"
KEY_EVENT_SCHEMA = "hotpot.ds.governance-key/current"
ATTESTATION_SCHEMA = "hotpot.ds.attestation/current"
SIGNATURE_SCHEMA = "hotpot.ds.ed25519/current"
KEY_EVENT_DOMAIN = "hotpot.ds.governance-key"
ATTESTATION_DOMAIN = "hotpot.ds.attestation"

_ROOT_FIELDS = {
    "schema",
    "signature_schema",
    "pinned_genesis_key_id",
    "public_key",
    "created_at",
}
_REGISTRY_FIELDS = {
    "schema",
    "signature_schema",
    "pinned_genesis_key_id",
    "revision",
    "keys",
    "registry_digest",
}
_KEY_FIELDS = {
    "key_id",
    "public_key",
    "predecessor_key_id",
    "activated_at",
    "activation",
    "revocation",
}
_AUTHORIZATION_FIELDS = {"signer_key_id", "payload_digest", "signature"}
_REVOCATION_FIELDS = _AUTHORIZATION_FIELDS | {"effective_at", "reason"}
_ATTESTATION_PAYLOAD_FIELDS = {
    "domain",
    "schema",
    "dataset",
    "release",
    "case_id",
    "decision",
    "reason",
    "actor",
    "event_id",
    "created_at",
    "supersedes",
    "key_id",
}
_ATTESTATION_FIELDS = _ATTESTATION_PAYLOAD_FIELDS | {
    "authority",
    "approved",
    "signature_schema",
    "payload_digest",
    "signature",
    "status",
    "attestation_id",
}
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]*$")


class GovernanceError(ValueError):
    """Base error for an invalid governance object."""


class GovernanceUnavailableError(GovernanceError):
    """No vetted Ed25519 verification backend is installed."""


class GovernanceVerificationError(GovernanceError):
    """A registry does not verify to the tracked genesis root."""


class GovernanceAuthorizationError(GovernanceError):
    """A caller attempted to use the private Hotpot signing path."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise GovernanceVerificationError(
            "governance data must be canonical JSON"
        ) from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _json_copy(value: Any) -> Any:
    return json.loads(_canonical_bytes(value).decode("utf-8"))


def _exact_fields(value: Any, expected: set[str], context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GovernanceVerificationError(f"{context} must be a JSON object")
    actual = set(value)
    if actual != expected:
        raise GovernanceVerificationError(
            f"{context} fields differ from the current schema; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return value


def _identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise GovernanceVerificationError(f"{field} is not a portable identifier")
    return value


def _text(value: Any, field: str, *, optional: bool = False) -> Optional[str]:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value:
        raise GovernanceVerificationError(f"{field} must be a non-empty string")
    return value


def _timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise GovernanceVerificationError(f"{field} must be an RFC 3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise GovernanceVerificationError(
            f"{field} must be an RFC 3339 UTC timestamp"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise GovernanceVerificationError(f"{field} must use UTC")
    return parsed


def _decoded(value: Any, field: str, length: int) -> bytes:
    if not isinstance(value, str):
        raise GovernanceVerificationError(f"{field} must be base64 text")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise GovernanceVerificationError(f"{field} must be valid base64") from exc
    if len(decoded) != length:
        raise GovernanceVerificationError(f"{field} must encode {length} bytes")
    return decoded


def _digest_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise GovernanceVerificationError(f"{field} must be a SHA-256 digest")
    return value


def _verify_ed25519(public_key: bytes, signature: bytes, payload: bytes) -> bool:
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey,
        )
    except ImportError:
        try:
            from nacl.exceptions import BadSignatureError
            from nacl.signing import VerifyKey
        except ImportError as exc:
            raise GovernanceUnavailableError(
                "Hotpot governance verification requires cryptography or PyNaCl"
            ) from exc
        try:
            VerifyKey(public_key).verify(payload, signature)
        except (BadSignatureError, ValueError):
            return False
        return True
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(signature, payload)
    except (InvalidSignature, ValueError):
        return False
    return True


@dataclass(frozen=True)
class PinnedRoot:
    schema: str
    signature_schema: str
    pinned_genesis_key_id: str
    public_key: str
    created_at: str
    digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "signature_schema": self.signature_schema,
            "pinned_genesis_key_id": self.pinned_genesis_key_id,
            "public_key": self.public_key,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class GovernanceKey:
    key_id: str
    public_key: str
    predecessor_key_id: Optional[str]
    activated_at: str
    revoked_at: Optional[str]


class VerifiedRegistry:
    """Immutable registry snapshot already verified to the pinned root."""

    __slots__ = ("revision", "digest", "pinned_genesis_key_id", "_snapshot", "_keys")

    def __init__(
        self,
        snapshot: Mapping[str, Any],
        keys: Mapping[str, GovernanceKey],
    ) -> None:
        copied = _json_copy(snapshot)
        self.revision = int(copied["revision"])
        self.digest = str(copied["registry_digest"])
        self.pinned_genesis_key_id = str(copied["pinned_genesis_key_id"])
        self._snapshot = copied
        self._keys = MappingProxyType(dict(keys))

    @property
    def keys(self) -> Mapping[str, GovernanceKey]:
        return self._keys

    def to_dict(self) -> dict[str, Any]:
        return _json_copy(self._snapshot)

    def key_is_active(self, key_id: str, at: str) -> bool:
        key = self._keys.get(key_id)
        if key is None:
            return False
        moment = _timestamp(at, "attestation.created_at")
        if moment < _timestamp(key.activated_at, "key.activated_at"):
            return False
        return key.revoked_at is None or moment < _timestamp(
            key.revoked_at, "key.revocation.effective_at"
        )


def load_pinned_root() -> PinnedRoot:
    """Load and validate the package-tracked, public-only genesis root."""

    path = Path(__file__).with_name("governance_root.json")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GovernanceVerificationError(
            "the tracked Hotpot governance root is unavailable or invalid"
        ) from exc
    root = _exact_fields(value, _ROOT_FIELDS, "governance root")
    if root["schema"] != ROOT_SCHEMA:
        raise GovernanceVerificationError("unsupported governance root schema")
    if root["signature_schema"] != SIGNATURE_SCHEMA:
        raise GovernanceVerificationError("unsupported governance signature schema")
    key_id = _identifier(root["pinned_genesis_key_id"], "pinned_genesis_key_id")
    _decoded(root["public_key"], "governance root public_key", 32)
    _timestamp(root["created_at"], "governance root created_at")
    return PinnedRoot(
        schema=ROOT_SCHEMA,
        signature_schema=SIGNATURE_SCHEMA,
        pinned_genesis_key_id=key_id,
        public_key=str(root["public_key"]),
        created_at=str(root["created_at"]),
        digest=_digest(root),
    )


def default_registry_snapshot() -> dict[str, Any]:
    """Return the canonical genesis-only registry used before the first rotation."""

    root = load_pinned_root()
    unsigned = {
        "schema": REGISTRY_SCHEMA,
        "signature_schema": SIGNATURE_SCHEMA,
        "pinned_genesis_key_id": root.pinned_genesis_key_id,
        "revision": 0,
        "keys": [
            {
                "key_id": root.pinned_genesis_key_id,
                "public_key": root.public_key,
                "predecessor_key_id": None,
                "activated_at": root.created_at,
                "activation": None,
                "revocation": None,
            }
        ],
    }
    return {**unsigned, "registry_digest": _digest(unsigned)}


def _activation_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "domain": KEY_EVENT_DOMAIN,
        "schema": KEY_EVENT_SCHEMA,
        "action": "activate",
        "key_id": record["key_id"],
        "public_key": record["public_key"],
        "predecessor_key_id": record["predecessor_key_id"],
        "effective_at": record["activated_at"],
    }


def _revocation_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    revocation = record["revocation"]
    return {
        "domain": KEY_EVENT_DOMAIN,
        "schema": KEY_EVENT_SCHEMA,
        "action": "revoke",
        "key_id": record["key_id"],
        "effective_at": revocation["effective_at"],
        "reason": revocation["reason"],
        "signer_key_id": revocation["signer_key_id"],
    }


def _verify_authorization(
    authorization: Mapping[str, Any],
    payload: Mapping[str, Any],
    public_key: str,
    context: str,
) -> None:
    _exact_fields(authorization, _AUTHORIZATION_FIELDS, context)
    digest = _digest(payload)
    if _digest_text(authorization["payload_digest"], f"{context}.payload_digest") != digest:
        raise GovernanceVerificationError(f"{context} payload digest does not match")
    signature = _decoded(authorization["signature"], f"{context}.signature", 64)
    if not _verify_ed25519(
        _decoded(public_key, f"{context}.public_key", 32),
        signature,
        _canonical_bytes(payload),
    ):
        raise GovernanceVerificationError(f"{context} signature is invalid")


def verify_registry(snapshot: Mapping[str, Any]) -> VerifiedRegistry:
    """Verify a complete registry snapshot back to the package trust root.

    A genesis-only registry needs no signature and therefore remains readable
    without a crypto backend.  Any rotation or revocation requires a vetted
    Ed25519 backend and fails closed when none is installed.
    """

    registry = _exact_fields(snapshot, _REGISTRY_FIELDS, "governance registry")
    if registry["schema"] != REGISTRY_SCHEMA:
        raise GovernanceVerificationError("unsupported governance registry schema")
    if registry["signature_schema"] != SIGNATURE_SCHEMA:
        raise GovernanceVerificationError("unsupported governance signature schema")
    root = load_pinned_root()
    if registry["pinned_genesis_key_id"] != root.pinned_genesis_key_id:
        raise GovernanceVerificationError(
            "governance registry does not name the pinned genesis key"
        )
    if type(registry["revision"]) is not int or registry["revision"] < 0:
        raise GovernanceVerificationError("governance registry revision must be nonnegative")
    if not isinstance(registry["keys"], list) or not registry["keys"]:
        raise GovernanceVerificationError("governance registry keys must be a non-empty list")

    unsigned = {key: value for key, value in registry.items() if key != "registry_digest"}
    expected_digest = _digest(unsigned)
    if _digest_text(registry["registry_digest"], "registry_digest") != expected_digest:
        raise GovernanceVerificationError("governance registry digest does not match")

    records: list[Mapping[str, Any]] = []
    raw_by_id: dict[str, Mapping[str, Any]] = {}
    verified: dict[str, GovernanceKey] = {}
    signed_events = 0
    for index, raw in enumerate(registry["keys"]):
        record = _exact_fields(raw, _KEY_FIELDS, f"governance key[{index}]")
        key_id = _identifier(record["key_id"], f"governance key[{index}].key_id")
        if key_id in raw_by_id:
            raise GovernanceVerificationError(f"duplicate governance key {key_id!r}")
        _decoded(record["public_key"], f"governance key {key_id}.public_key", 32)
        activated = _timestamp(record["activated_at"], f"governance key {key_id}.activated_at")
        predecessor = record["predecessor_key_id"]
        if index == 0:
            if (
                key_id != root.pinned_genesis_key_id
                or record["public_key"] != root.public_key
                or predecessor is not None
                or record["activated_at"] != root.created_at
                or record["activation"] is not None
            ):
                raise GovernanceVerificationError(
                    "the first registry key is not the tracked genesis root"
                )
        else:
            predecessor = _identifier(
                predecessor, f"governance key {key_id}.predecessor_key_id"
            )
            previous = records[-1]
            if predecessor != previous["key_id"]:
                raise GovernanceVerificationError("governance rotation chain is not linear")
            predecessor_revocation = previous["revocation"]
            if predecessor_revocation is not None and activated > _timestamp(
                predecessor_revocation["effective_at"],
                f"governance key {predecessor}.revocation.effective_at",
            ):
                raise GovernanceVerificationError(
                    "a revoked predecessor cannot authorize a later rotation"
                )
            authorization = _exact_fields(
                record["activation"],
                _AUTHORIZATION_FIELDS,
                f"governance key {key_id}.activation",
            )
            if authorization["signer_key_id"] != predecessor:
                raise GovernanceVerificationError(
                    "a rotated key must be authorized by its predecessor"
                )
            _verify_authorization(
                authorization,
                _activation_payload(record),
                str(previous["public_key"]),
                f"governance key {key_id}.activation",
            )
            signed_events += 1
        records.append(record)
        raw_by_id[key_id] = record

    position = {record["key_id"]: index for index, record in enumerate(records)}
    for record in records:
        revocation = record["revocation"]
        revoked_at: Optional[str] = None
        if revocation is not None:
            _exact_fields(
                revocation,
                _REVOCATION_FIELDS,
                f"governance key {record['key_id']}.revocation",
            )
            revoked_at = str(revocation["effective_at"])
            effective = _timestamp(
                revoked_at, f"governance key {record['key_id']}.revocation.effective_at"
            )
            if effective < _timestamp(
                record["activated_at"], f"governance key {record['key_id']}.activated_at"
            ):
                raise GovernanceVerificationError("a key cannot be revoked before activation")
            _text(
                revocation["reason"],
                f"governance key {record['key_id']}.revocation.reason",
            )
            signer_id = _identifier(
                revocation["signer_key_id"],
                f"governance key {record['key_id']}.revocation.signer_key_id",
            )
            if signer_id not in raw_by_id or position[signer_id] < position[record["key_id"]]:
                raise GovernanceVerificationError(
                    "revocation must be signed by the key or one of its successors"
                )
            signer = raw_by_id[signer_id]
            if effective < _timestamp(
                signer["activated_at"], f"governance key {signer_id}.activated_at"
            ):
                raise GovernanceVerificationError("revocation signer was not active")
            signer_revocation = signer["revocation"]
            if signer_revocation is not None and effective > _timestamp(
                signer_revocation["effective_at"],
                f"governance key {signer_id}.revocation.effective_at",
            ):
                raise GovernanceVerificationError("revocation signer was already revoked")
            authorization = {
                key: revocation[key] for key in _AUTHORIZATION_FIELDS
            }
            _verify_authorization(
                authorization,
                _revocation_payload(record),
                str(signer["public_key"]),
                f"governance key {record['key_id']}.revocation",
            )
            signed_events += 1
        verified[str(record["key_id"])] = GovernanceKey(
            key_id=str(record["key_id"]),
            public_key=str(record["public_key"]),
            predecessor_key_id=record["predecessor_key_id"],
            activated_at=str(record["activated_at"]),
            revoked_at=revoked_at,
        )

    if registry["revision"] != signed_events:
        raise GovernanceVerificationError(
            "governance registry revision does not match its signed event history"
        )
    return VerifiedRegistry(registry, verified)


def _attestation_payload(attestation: Mapping[str, Any]) -> dict[str, Any]:
    return {key: attestation[key] for key in _ATTESTATION_PAYLOAD_FIELDS}


def verify_hotpot_attestation(
    attestation: Mapping[str, Any],
    dataset: str,
    release: str,
    case_id: str,
    registry: Mapping[str, Any] | VerifiedRegistry,
) -> tuple[bool, str]:
    """Verify one Hotpot attestation in its exact scientific context."""

    try:
        item = _exact_fields(attestation, _ATTESTATION_FIELDS, "Hotpot attestation")
        trusted = verify_registry(
            registry.to_dict() if isinstance(registry, VerifiedRegistry) else registry
        )
        if item["authority"] != "Hotpot":
            return False, "authority_not_hotpot"
        if item["domain"] != ATTESTATION_DOMAIN or item["schema"] != ATTESTATION_SCHEMA:
            return False, "unsupported_attestation_schema"
        if item["signature_schema"] != SIGNATURE_SCHEMA:
            return False, "unsupported_signature_schema"
        if (
            item["dataset"] != dataset
            or item["release"] != release
            or item["case_id"] != case_id
        ):
            return False, "context_mismatch"
        if type(item["approved"]) is not bool:
            return False, "invalid_decision"
        decision = "approve" if item["approved"] else "reject"
        if item["decision"] != decision:
            return False, "invalid_decision"
        _identifier(item["dataset"], "dataset")
        _identifier(item["release"], "release")
        _identifier(item["case_id"], "case_id")
        _identifier(item["attestation_id"], "attestation_id")
        _identifier(item["event_id"], "event_id")
        _identifier(item["key_id"], "key_id")
        _text(item["actor"], "actor")
        _text(item["reason"], "reason", optional=True)
        if item["supersedes"] is not None:
            _identifier(item["supersedes"], "supersedes")
        if item["status"] != "active":
            return False, "attestation_not_active"
        created_at = str(item["created_at"])
        _timestamp(created_at, "created_at")
        key = trusted.keys.get(str(item["key_id"]))
        if key is None:
            return False, "untrusted_key"
        if not trusted.key_is_active(key.key_id, created_at):
            return False, "key_not_active"
        payload = _attestation_payload(item)
        if _digest_text(item["payload_digest"], "payload_digest") != _digest(payload):
            return False, "payload_digest_mismatch"
        signature = _decoded(item["signature"], "signature", 64)
        if not _verify_ed25519(
            _decoded(key.public_key, "public_key", 32),
            signature,
            _canonical_bytes(payload),
        ):
            return False, "signature_invalid"
    except GovernanceUnavailableError:
        raise
    except GovernanceVerificationError:
        return False, "invalid_attestation"
    return True, "verified"


_SIGNER_CAPABILITY = object()


class _AuthenticatedSigner:
    """Unserializable signer wrapper created only by trusted owner code."""

    __slots__ = ("key_id", "_sign", "_capability")

    def __init__(
        self,
        key_id: str,
        sign: Callable[[bytes], bytes],
        *,
        capability: object,
    ) -> None:
        if capability is not _SIGNER_CAPABILITY:
            raise GovernanceAuthorizationError(
                "Hotpot signers require an authenticated governance capability"
            )
        self.key_id = key_id
        self._sign = sign
        self._capability = capability

    def sign(self, payload: bytes) -> bytes:
        return self._sign(payload)


def _sign_hotpot_attestation(
    case: Any,
    approved: bool,
    *,
    actor: str,
    reason: Optional[str] = None,
    signer: Any,
) -> str:
    """Create and commit a Hotpot attestation through the private owner path."""

    from .public.case import Case

    if not isinstance(case, Case):
        raise GovernanceAuthorizationError("Hotpot signing requires a Bundle-owned Case")
    if (
        not isinstance(signer, _AuthenticatedSigner)
        or signer._capability is not _SIGNER_CAPABILITY
    ):
        raise GovernanceAuthorizationError(
            "Hotpot signing requires an authenticated governance signer"
        )
    if type(approved) is not bool:
        raise GovernanceAuthorizationError("Hotpot approval must be boolean")
    owner = case._owner
    context_method = getattr(owner, "_hotpot_governance_context", None)
    if context_method is None:
        raise GovernanceAuthorizationError(
            "the Case owner has no authenticated Hotpot governance context"
        )
    context = context_method()
    if not isinstance(context, Mapping):
        raise GovernanceAuthorizationError("invalid Hotpot governance context")
    dataset = _identifier(context.get("dataset"), "governance context dataset")
    release = _identifier(context.get("release"), "governance context release")
    registry = context.get("registry")
    trusted = verify_registry(
        registry.to_dict() if isinstance(registry, VerifiedRegistry) else registry
    )
    if signer.key_id not in trusted.keys:
        raise GovernanceAuthorizationError("signer key is not in the trusted registry")

    now = datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )
    if not trusted.key_is_active(signer.key_id, now):
        raise GovernanceAuthorizationError("signer key is not active")
    facts = case._facts()
    attestations = facts.setdefault("attestations", [])
    supersedes = None
    for previous in attestations:
        if (
            previous.get("authority") == "Hotpot"
            and previous.get("actor") == actor
            and previous.get("status", "active") == "active"
        ):
            supersedes = previous.get("attestation_id")

    attestation_id = owner._new_fact_identifier(case.id, "attestation")
    event_id = owner._new_fact_identifier(case.id, "governance-event")
    payload = {
        "domain": ATTESTATION_DOMAIN,
        "schema": ATTESTATION_SCHEMA,
        "dataset": dataset,
        "release": release,
        "case_id": case.id,
        "decision": "approve" if bool(approved) else "reject",
        "reason": reason,
        "actor": actor,
        "event_id": event_id,
        "created_at": now,
        "supersedes": supersedes,
        "key_id": signer.key_id,
    }
    signature = signer.sign(_canonical_bytes(payload))
    if not isinstance(signature, bytes) or len(signature) != 64:
        raise GovernanceAuthorizationError("governance signer returned an invalid signature")
    item = {
        **payload,
        "authority": "Hotpot",
        "approved": bool(approved),
        "signature_schema": SIGNATURE_SCHEMA,
        "payload_digest": _digest(payload),
        "signature": base64.b64encode(signature).decode("ascii"),
        "status": "active",
        "attestation_id": attestation_id,
    }
    valid, verification_reason = verify_hotpot_attestation(
        item, dataset, release, case.id, trusted
    )
    if not valid:
        raise GovernanceAuthorizationError(
            f"new Hotpot signature did not verify: {verification_reason}"
        )
    if supersedes is not None:
        for previous in attestations:
            if previous.get("attestation_id") == supersedes:
                previous["status"] = "superseded"
                previous["superseded_by"] = attestation_id
                break
    attestations.append(item)
    case._commit(
        facts,
        "benchmark.hotpot_attest",
        {
            "attestation_id": attestation_id,
            "event_id": event_id,
            "approved": bool(approved),
            "actor": actor,
            "key_id": signer.key_id,
            "registry_digest": trusted.digest,
        },
    )
    return attestation_id


__all__ = [
    "GovernanceAuthorizationError",
    "GovernanceError",
    "GovernanceUnavailableError",
    "GovernanceVerificationError",
    "GovernanceKey",
    "PinnedRoot",
    "VerifiedRegistry",
    "load_pinned_root",
    "verify_registry",
    "verify_hotpot_attestation",
]
