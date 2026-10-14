from __future__ import annotations

import base64
import hashlib
import json
from copy import deepcopy

import pytest

from hotpot.ds import governance
from hotpot.ds.public.case import Case


def _canonical(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _genesis_registry(root=None):
    root = root or governance.load_pinned_root()
    unsigned = {
        "schema": governance.REGISTRY_SCHEMA,
        "signature_schema": governance.SIGNATURE_SCHEMA,
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


def _attestation(dataset="spin", release="spin-001", case_id="case-1", **updates):
    payload = {
        "domain": governance.ATTESTATION_DOMAIN,
        "schema": governance.ATTESTATION_SCHEMA,
        "dataset": dataset,
        "release": release,
        "case_id": case_id,
        "decision": "approve",
        "reason": "owner-reviewed",
        "actor": "dataset-owner",
        "event_id": "event-1",
        "created_at": "2026-10-14T01:00:00Z",
        "supersedes": None,
        "key_id": governance.load_pinned_root().pinned_genesis_key_id,
    }
    payload.update({key: value for key, value in updates.items() if key in payload})
    item = {
        **payload,
        "authority": "Hotpot",
        "approved": payload["decision"] == "approve",
        "signature_schema": governance.SIGNATURE_SCHEMA,
        "payload_digest": _digest(payload),
        "signature": base64.b64encode(b"\0" * 64).decode("ascii"),
        "status": "active",
        "attestation_id": "attestation-1",
    }
    item.update({key: value for key, value in updates.items() if key not in payload})
    return item


def test_pinned_root_has_stable_public_only_digest():
    root = governance.load_pinned_root()

    assert root.pinned_genesis_key_id == "hotpot-genesis-2026-10"
    assert root.digest == "fe92b156bc1f66fd08d3eb78addfdad900d2378b98dae3d908988e2cb3dcef48"
    assert set(root.to_dict()) == {
        "schema",
        "signature_schema",
        "pinned_genesis_key_id",
        "public_key",
        "created_at",
    }
    assert len(base64.b64decode(root.public_key, validate=True)) == 32


def test_genesis_only_registry_has_no_signed_event_to_verify(monkeypatch):
    monkeypatch.setattr(
        governance,
        "_verify_ed25519",
        lambda *_: pytest.fail("genesis registry must not invoke Ed25519"),
    )

    registry = governance.verify_registry(_genesis_registry())

    assert registry.revision == 0
    assert registry.digest == _genesis_registry()["registry_digest"]
    assert tuple(registry.keys) == (
        governance.load_pinned_root().pinned_genesis_key_id,
    )
    assert governance.default_registry_snapshot() == _genesis_registry()


def test_registry_cannot_replace_pinned_root_with_self_declared_key():
    snapshot = _genesis_registry()
    snapshot["keys"][0]["key_id"] = "caller-key"
    snapshot["keys"][0]["public_key"] = base64.b64encode(b"x" * 32).decode()
    unsigned = {key: value for key, value in snapshot.items() if key != "registry_digest"}
    snapshot["registry_digest"] = _digest(unsigned)

    with pytest.raises(governance.GovernanceVerificationError, match="genesis"):
        governance.verify_registry(snapshot)


def test_context_replay_is_rejected_before_signature_verification(monkeypatch):
    monkeypatch.setattr(
        governance,
        "_verify_ed25519",
        lambda *_: pytest.fail("context mismatch must be rejected before crypto"),
    )

    valid, reason = governance.verify_hotpot_attestation(
        _attestation(), "oxidation", "spin-001", "case-1", _genesis_registry()
    )

    assert (valid, reason) == (False, "context_mismatch")


def test_unverified_hotpot_declaration_has_no_governance_effect(monkeypatch):
    monkeypatch.setattr(governance, "_verify_ed25519", lambda *_: False)

    valid, reason = governance.verify_hotpot_attestation(
        _attestation(), "spin", "spin-001", "case-1", _genesis_registry()
    )

    assert valid is False
    assert reason == "signature_invalid"


def test_unknown_attestation_key_is_not_promoted_to_trust(monkeypatch):
    monkeypatch.setattr(
        governance,
        "_verify_ed25519",
        lambda *_: pytest.fail("unknown key must be rejected before crypto"),
    )
    item = _attestation(key_id="caller-key")

    valid, reason = governance.verify_hotpot_attestation(
        item, "spin", "spin-001", "case-1", _genesis_registry()
    )

    assert (valid, reason) == (False, "untrusted_key")


def test_caller_constructed_verified_registry_cannot_inject_a_key(monkeypatch):
    monkeypatch.setattr(
        governance,
        "_verify_ed25519",
        lambda *_: pytest.fail("unknown key must be rejected before crypto"),
    )
    snapshot = _genesis_registry()
    injected = governance.VerifiedRegistry(
        snapshot,
        {
            "caller-key": governance.GovernanceKey(
                key_id="caller-key",
                public_key=base64.b64encode(b"x" * 32).decode(),
                predecessor_key_id=None,
                activated_at="2026-01-01T00:00:00Z",
                revoked_at=None,
            )
        },
    )
    item = _attestation(key_id="caller-key")

    valid, reason = governance.verify_hotpot_attestation(
        item, "spin", "spin-001", "case-1", injected
    )

    assert (valid, reason) == (False, "untrusted_key")


def test_ordinary_signer_object_cannot_enter_private_signing_path():
    class CallerSigner:
        key_id = governance.load_pinned_root().pinned_genesis_key_id

        @staticmethod
        def sign(payload):
            return b"\0" * 64

    case = Case(object(), "case-1")

    with pytest.raises(governance.GovernanceAuthorizationError, match="authenticated"):
        governance._sign_hotpot_attestation(
            case,
            True,
            actor="caller",
            signer=CallerSigner(),
        )


def test_rotation_revocation_attestation_and_private_signing(monkeypatch):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    root_private = Ed25519PrivateKey.generate()
    child_private = Ed25519PrivateKey.generate()

    def public_text(private):
        raw = private.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        return base64.b64encode(raw).decode("ascii")

    root = governance.PinnedRoot(
        schema=governance.ROOT_SCHEMA,
        signature_schema=governance.SIGNATURE_SCHEMA,
        pinned_genesis_key_id="test-root",
        public_key=public_text(root_private),
        created_at="2026-01-01T00:00:00Z",
        digest="0" * 64,
    )
    monkeypatch.setattr(governance, "load_pinned_root", lambda: root)

    root_record = {
        "key_id": "test-root",
        "public_key": root.public_key,
        "predecessor_key_id": None,
        "activated_at": root.created_at,
        "activation": None,
        "revocation": None,
    }
    child_record = {
        "key_id": "test-child",
        "public_key": public_text(child_private),
        "predecessor_key_id": "test-root",
        "activated_at": "2026-02-01T00:00:00Z",
        "activation": None,
        "revocation": None,
    }
    activation_payload = governance._activation_payload(child_record)
    child_record["activation"] = {
        "signer_key_id": "test-root",
        "payload_digest": _digest(activation_payload),
        "signature": base64.b64encode(
            root_private.sign(_canonical(activation_payload))
        ).decode("ascii"),
    }
    root_record["revocation"] = {
        "effective_at": "2026-02-01T00:00:00Z",
        "reason": "scheduled rotation",
        "signer_key_id": "test-child",
        "payload_digest": "0" * 64,
        "signature": "",
    }
    revocation_payload = governance._revocation_payload(root_record)
    root_record["revocation"].update(
        {
            "payload_digest": _digest(revocation_payload),
            "signature": base64.b64encode(
                child_private.sign(_canonical(revocation_payload))
            ).decode("ascii"),
        }
    )
    unsigned = {
        "schema": governance.REGISTRY_SCHEMA,
        "signature_schema": governance.SIGNATURE_SCHEMA,
        "pinned_genesis_key_id": "test-root",
        "revision": 2,
        "keys": [root_record, child_record],
    }
    snapshot = {**unsigned, "registry_digest": _digest(unsigned)}
    registry = governance.verify_registry(snapshot)

    assert registry.key_is_active("test-root", "2026-01-31T23:59:59Z")
    assert not registry.key_is_active("test-root", "2026-02-01T00:00:00Z")
    assert registry.key_is_active("test-child", "2026-02-01T00:00:00Z")

    class Owner:
        is_synchronized = False

        def __init__(self):
            self.facts = {"attestations": []}
            self.counter = 0

        def _case_payload(self, case_id):
            return self.facts

        def _new_fact_identifier(self, case_id, kind):
            self.counter += 1
            return f"{kind}-{self.counter}"

        def _hotpot_governance_context(self):
            return {
                "dataset": "spin",
                "release": "spin-001",
                "registry": registry,
            }

        def _mutate_case(self, case_id, action, mutation):
            facts = deepcopy(self.facts)
            details, result = mutation(facts)
            self.facts = facts
            self.action = action
            self.details = details
            return result

    owner = Owner()
    case = Case(owner, "case-1")
    signer = governance._AuthenticatedSigner(
        "test-child",
        child_private.sign,
        capability=governance._SIGNER_CAPABILITY,
    )

    attestation_id = governance._sign_hotpot_attestation(
        case,
        True,
        actor="hotpot-owner",
        reason="reviewed",
        signer=signer,
    )
    item = owner.facts["attestations"][0]

    assert attestation_id == item["attestation_id"]
    assert owner.action == "benchmark.hotpot_attest"
    assert governance.verify_hotpot_attestation(
        item, "spin", "spin-001", "case-1", registry
    ) == (True, "verified")
    assert governance.verify_hotpot_attestation(
        item, "oxidation", "spin-001", "case-1", registry
    ) == (False, "context_mismatch")

    tampered = deepcopy(item)
    signature = bytearray(base64.b64decode(tampered["signature"], validate=True))
    signature[0] ^= 1
    tampered["signature"] = base64.b64encode(signature).decode("ascii")
    assert governance.verify_hotpot_attestation(
        tampered, "spin", "spin-001", "case-1", registry
    ) == (False, "signature_invalid")
