from __future__ import annotations

import copy
import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import hotpot.ds.public.certificate as certificate_module

from hotpot.ds.public.case import UnsynchronizedBundleError
from hotpot.ds.public.certificate import (
    CERTIFICATE_SCHEMA,
    CertificateIntegrityError,
    CertificatePathError,
    certificate_digest,
    load_certificate_envelope,
    verify_bundle_certificate,
    write_bundle_certificate,
)
from hotpot.ds.gates._checker import CHECKER_PROTOCOL_DIGEST
from hotpot.ds.public._implementation import runtime_policy


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest_json(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return _digest_bytes(encoded)


@dataclass
class _Generation:
    root: Path

    @property
    def id(self) -> str:
        return "generation-001"

    @property
    def digest(self) -> str:
        return _digest_bytes(b"logical generation")

    @property
    def manifest_path(self) -> Path:
        return (
            self.root
            / "assessments"
            / "generations"
            / self.id
            / "generation.json"
        )

    @property
    def manifest_digest(self) -> str:
        return _digest_bytes(self.manifest_path.read_bytes())

    @property
    def manifest(self) -> dict:
        return {"runtime_policy": runtime_policy()}

    @property
    def summary(self) -> dict:
        return {"case_count": 1, "archive_class": {"ordinary": 1}}

    @contextmanager
    def lease(self):
        yield self


class _State:
    def __init__(self, root: Path, policy: dict):
        self.root = root
        self.ds_name = "demo"
        self.manifest = {
            "dataset_id": "demo",
            "release_id": "demo-r1",
            "version": "1.0",
            "release_serial": 1,
        }
        self.policy = policy
        self.target_context = {
            "canonical_path": str(root.parent / "datasets.sqlite3"),
            "store_state": "absent",
            "store_uuid": None,
            "schema_id": "hotpot-ds-store/current",
            "dataset_state": "absent",
            "dataset_policy_revision": None,
            "dataset_policy_digest": None,
        }
        self.status = {
            "source_revision": 2,
            "source_digest": _digest_bytes(b"source state"),
            "policy_revision": 1,
            "policy_digest": _digest_json(policy),
        }


class _Bundle:
    def __init__(self, root: Path, *, assets: list[dict], policy: dict | None = None):
        self.root = root
        self.name = "demo"
        self._state = _State(
            root,
            policy
            or {
                "schema": "hotpot-ds-policy/current",
                "runtime_policy": runtime_policy(),
            },
        )
        self._current_generation = _Generation(root)
        self.is_synchronized = True
        self._assets = copy.deepcopy(assets)

    def _scientific_asset_inventory(self) -> list[dict]:
        return copy.deepcopy(self._assets)

    def snapshot(self) -> dict:
        return {
            "name": self.name,
            "is_synchronized": self.is_synchronized,
            "manifest": copy.deepcopy(self._state.manifest),
            "status": copy.deepcopy(self._state.status),
            "policy": copy.deepcopy(self._state.policy),
            "target_context": copy.deepcopy(self._state.target_context),
            "assets": copy.deepcopy(self._assets),
        }


def _bundle_root(tmp_path: Path) -> Path:
    root = tmp_path / "bundle"
    (root / "cases").mkdir(parents=True)
    (root / "cases" / "facts.sqlite3").write_bytes(b"canonical facts")
    generation = root / "assessments" / "generations" / "generation-001"
    generation.mkdir(parents=True)
    (generation / "generation.json").write_text(
        json.dumps({"schema": "test-generation", "files": []}),
        encoding="utf-8",
    )
    (root / "structures").mkdir()
    (root / "structures" / "case-1.smi").write_text("O\n", encoding="utf-8")
    return root


def _present_asset(root: Path) -> dict:
    path = root / "structures" / "case-1.smi"
    content = path.read_bytes()
    return {
        "case_id": "case-1",
        "declaration_id": "structure-1",
        "path": "structures/case-1.smi",
        "state": "present",
        "sha256": _digest_bytes(content),
        "size": len(content),
    }


def _bundle(tmp_path: Path) -> _Bundle:
    root = _bundle_root(tmp_path)
    return _Bundle(root, assets=[_present_asset(root)])


def test_writer_rejects_unsynchronized_bundle(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    bundle.is_synchronized = False

    with pytest.raises(UnsynchronizedBundleError, match="synchronize"):
        write_bundle_certificate(bundle)

    assert not (bundle.root / "certificate.json").exists()


def test_root_certificate_round_trip_and_digest(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)

    path = write_bundle_certificate(bundle)
    envelope = load_certificate_envelope(path)

    assert path == bundle.root / "certificate.json"
    assert envelope["schema"] == CERTIFICATE_SCHEMA
    assert envelope["certificate"]["dataset"] == "demo"
    assert verify_bundle_certificate(bundle) == envelope
    assert certificate_digest(path) == _digest_bytes(path.read_bytes())

    alternate = bundle.root / "alternate.json"
    alternate.write_bytes(path.read_bytes())
    with pytest.raises(CertificatePathError):
        load_certificate_envelope(alternate)


def test_certificate_rejects_tampered_frozen_runtime_policy(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    path = write_bundle_certificate(bundle)
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["certificate"]["runtime_policy"]["runtime"]["python"][
        "version"
    ] = "tampered"
    envelope["envelope_sha256"] = _digest_json(envelope["certificate"])
    path.write_text(json.dumps(envelope), encoding="utf-8")

    with pytest.raises(CertificateIntegrityError, match="runtime policy"):
        load_certificate_envelope(path)


@pytest.mark.parametrize("target", ["facts", "generation", "asset"])
def test_bound_positive_file_tampering_is_rejected(
    tmp_path: Path, target: str
) -> None:
    bundle = _bundle(tmp_path)
    write_bundle_certificate(bundle)
    paths = {
        "facts": bundle.root / "cases" / "facts.sqlite3",
        "generation": bundle._current_generation.manifest_path,
        "asset": bundle.root / "structures" / "case-1.smi",
    }

    paths[target].write_bytes(paths[target].read_bytes() + b"tampered")

    with pytest.raises(CertificateIntegrityError):
        verify_bundle_certificate(bundle)


def test_envelope_tampering_is_rejected_before_bundle_association(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    path = write_bundle_certificate(bundle)
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["certificate"]["summary"]["case_count"] = 99
    path.write_text(json.dumps(envelope), encoding="utf-8")

    with pytest.raises(CertificateIntegrityError, match="envelope digest"):
        load_certificate_envelope(path)


def test_certified_missing_asset_must_remain_absent(tmp_path: Path) -> None:
    root = _bundle_root(tmp_path)
    missing = {
        "case_id": "case-1",
        "declaration_id": "evidence-1",
        "path": "evidence/missing.txt",
        "state": "missing",
        "sha256": None,
        "size": None,
    }
    bundle = _Bundle(root, assets=[_present_asset(root), missing])
    write_bundle_certificate(bundle)
    (root / "evidence").mkdir()
    (root / "evidence" / "missing.txt").write_text("appeared", encoding="utf-8")

    with pytest.raises(CertificateIntegrityError, match="appeared"):
        verify_bundle_certificate(bundle)


def test_custom_checker_source_is_certificate_bound(tmp_path: Path) -> None:
    root = _bundle_root(tmp_path)
    source = "def checker(case, bundle):\n    return True, 'ok'\n"
    source_digest = _digest_bytes(source.encode("utf-8"))
    protocol_digest = CHECKER_PROTOCOL_DIGEST
    checker_path = root / "policy" / "checkers" / f"{source_digest}.py"
    checker_path.parent.mkdir(parents=True)
    checker_path.write_text(source, encoding="utf-8")
    policy = {
        "schema": "hotpot-ds-policy/current",
        "runtime_policy": runtime_policy(),
        "gates": [
            {
                "checks": [
                    {
                        "kind": "custom",
                        "source": source,
                        "source_sha256": source_digest,
                        "entrypoint": "checker",
                        "checker_protocol_digest": protocol_digest,
                    }
                ]
            }
        ],
    }
    bundle = _Bundle(root, assets=[_present_asset(root)], policy=policy)

    path = write_bundle_certificate(bundle)
    envelope = load_certificate_envelope(path)
    assert envelope["certificate"]["checker_artifacts"] == [
        {
            "path": f"policy/checkers/{source_digest}.py",
            "sha256": source_digest,
            "size": len(source.encode("utf-8")),
            "entrypoint": "checker",
            "checker_protocol_digest": protocol_digest,
        }
    ]

    checker_path.write_text(source + "# changed\n", encoding="utf-8")
    with pytest.raises(CertificateIntegrityError):
        verify_bundle_certificate(bundle)


def test_certification_does_not_mutate_bundle_state(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    before = bundle.snapshot()

    path = write_bundle_certificate(bundle)
    verify_bundle_certificate(bundle, path)

    assert bundle.snapshot() == before
