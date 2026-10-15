from __future__ import annotations

from pathlib import Path

import pytest

from hotpot.ds.public._implementation import (
    PolicyProtocolError,
    _clear_runtime_policy_cache,
    _digest_document,
    _semantic_source_digest,
    parser_binding_for_format,
    require_current_runtime_policy,
    runtime_policy,
    runtime_policy_digest,
    runtime_policy_is_current,
    validate_frozen_runtime_policy,
)


def test_runtime_policy_is_a_complete_canonical_self_digest() -> None:
    policy = runtime_policy()

    assert policy["digest"] == _digest_document(policy) == runtime_policy_digest()
    assert validate_frozen_runtime_policy(dict(reversed(list(policy.items())))) == policy
    assert runtime_policy_is_current(policy)
    require_current_runtime_policy(policy)

    assessment_paths = {
        item["path"] for item in policy["assessment"]["source"]["files"]
    }
    parser_paths = {
        item["path"]
        for item in policy["parsers"]["entries"][0]["source"]["files"]
    }
    assert "ds/gates/_base.py" in assessment_paths
    assert "ds/public/_implementation.py" in assessment_paths
    assert "cheminfo/_io/_io.py" in parser_paths
    assert "cheminfo/crystal.py" in parser_paths

    identities = {
        item["name"]: item for item in policy["runtime"]["distributions"]
    }
    assert identities["cryptography"]["installed"] is True
    assert identities["hotpot-zzy"]["version"]
    assert identities["openbabel"]["version"]
    assert policy["runtime"]["python"]["implementation"]
    assert policy["runtime"]["python"]["version"]
    assert policy["runtime"]["sqlite"]["version"]


def test_runtime_policy_is_cached_and_returned_as_a_defensive_copy() -> None:
    _clear_runtime_policy_cache()
    digest = runtime_policy_digest()
    first = runtime_policy()
    first["assessment"]["source"]["files"].clear()
    first["runtime"]["distributions"][0]["version"] = "changed"

    second = runtime_policy()
    assert second["digest"] == digest
    assert second["assessment"]["source"]["files"]
    assert second["runtime"]["distributions"][0]["version"] != "changed"


def test_frozen_validation_rejects_tampering() -> None:
    tampered = runtime_policy()
    tampered["runtime"]["python"]["version"] = "tampered"

    with pytest.raises(PolicyProtocolError, match="runtime policy digest"):
        validate_frozen_runtime_policy(tampered)
    assert runtime_policy_is_current(tampered) is False

    nested = runtime_policy()
    nested["parsers"]["entries"][0]["source"]["files"][0]["sha256"] = "0" * 64
    nested["digest"] = _digest_document(nested)
    with pytest.raises(PolicyProtocolError, match="parser source digest"):
        validate_frozen_runtime_policy(nested)


def test_structurally_valid_frozen_policy_can_be_non_current() -> None:
    frozen = runtime_policy()
    frozen["runtime"]["python"]["version"] = "0.0.0-frozen"
    frozen["digest"] = _digest_document(frozen)

    assert validate_frozen_runtime_policy(frozen) == frozen
    assert runtime_policy_is_current(frozen) is False
    with pytest.raises(PolicyProtocolError, match="owner policy migration"):
        require_current_runtime_policy(frozen)


def test_parser_registry_binds_protocol_source_and_distributions() -> None:
    binding = parser_binding_for_format(".SMILES")

    assert binding is not None
    assert binding["name"] == "hotpot.Molecule.read_one/openbabel"
    assert binding["protocol"]["entrypoint"] == "hotpot.Molecule.read_one"
    assert binding["protocol"]["digest"] == _digest_document(binding["protocol"])
    assert binding["source"]["digest"] == _digest_document(binding["source"])
    assert binding["digest"] == _digest_document(binding)
    assert {item["name"] for item in binding["distributions"]} >= {
        "hotpot-zzy",
        "openbabel",
        "numpy",
        "rdkit",
    }
    assert parser_binding_for_format("not-configured") is None

    binding["formats"].clear()
    assert parser_binding_for_format("smi")["formats"]


def test_parser_binding_can_be_read_from_a_valid_frozen_policy() -> None:
    frozen = runtime_policy()
    frozen["runtime"]["python"]["version"] = "0.0.0-frozen"
    frozen["digest"] = _digest_document(frozen)

    binding = parser_binding_for_format("xyz", frozen)
    assert binding is not None
    assert "xyz" in binding["formats"]


def test_semantic_source_digest_ignores_unselected_operational_code(
    tmp_path: Path,
) -> None:
    module = tmp_path / "component.py"
    module.write_text(
        "def semantic():\n    return 1\n\ndef operational():\n    return 1\n",
        encoding="utf-8",
    )
    first = _semantic_source_digest(module, ("semantic",))
    module.write_text(
        "def semantic():\n    return 1\n\ndef operational():\n    return 2\n",
        encoding="utf-8",
    )
    assert _semantic_source_digest(module, ("semantic",)) == first
    module.write_text(
        "def semantic():\n    return 2\n\ndef operational():\n    return 2\n",
        encoding="utf-8",
    )
    assert _semantic_source_digest(module, ("semantic",)) != first
