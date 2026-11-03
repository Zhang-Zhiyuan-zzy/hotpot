"""Artifact operation contracts and the single packaged resource owner."""

from dataclasses import fields
from importlib import resources
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hotpot.cheminfo.adapters.inference import artifacts as effects
from hotpot.cheminfo.operations.inference import artifacts
from hotpot.cheminfo.operations.inference.artifacts.manifest import (
    artifact_is_complete,
    bundle_sha256,
    load_manifest,
    verify_artifact,
)
from hotpot.cheminfo.operations.inference.artifacts.registry import ARTIFACTS, ArtifactSpec
from hotpot.cheminfo.operations.inference.artifacts.resolver import ModelArtifact, ModelSource
from tests.model_artifacts.test_resolver import _write_pointer


def test_existing_artifact_contracts_have_one_operation_owner():
    owner = "hotpot.cheminfo.operations.inference.artifacts"

    assert ArtifactSpec.__module__ == f"{owner}.registry"
    assert tuple(field.name for field in fields(ArtifactSpec)) == (
        "name", "manifest_path", "environment_variable",
    )
    assert ModelArtifact.__module__ == f"{owner}.resolver"
    assert ModelSource.__module__ == f"{owner}.resolver"
    assert tuple((source.name, source.value) for source in ModelSource) == (
        ("AUTO", "auto"), ("LOCAL", "local"), ("HUB", "hub"),
    )
    assert artifacts.ModelArtifact is ModelArtifact


@pytest.mark.parametrize("name", ("mca", "cbond"))
def test_registry_uses_packaged_pointer_and_readme(name):
    resource = resources.files(artifacts).joinpath("data", name)
    spec = ARTIFACTS[name]

    assert spec.manifest_path == Path(str(resource.joinpath("manifest.json")))
    assert resource.joinpath("README.md").is_file()
    assert load_manifest(spec.manifest_path)["artifact"]["id"] == name
    assert spec.environment_variable == f"HOTPOT_{name.upper()}_MODEL_DIR"


def test_configured_environment_directory_precedes_packaged_and_hub(tmp_path, monkeypatch):
    pointer = _write_pointer(tmp_path)
    (pointer.parent / "model.onnx").write_bytes(b"model")
    configured = tmp_path / "configured"
    monkeypatch.setenv("EXAMPLE_MODEL_DIR", str(configured))

    assert ModelArtifact(pointer, "EXAMPLE_MODEL_DIR", source="hub").resolve() == configured


def test_explicit_directory_precedes_environment_directory(tmp_path, monkeypatch):
    pointer = _write_pointer(tmp_path)
    monkeypatch.setenv("EXAMPLE_MODEL_DIR", str(tmp_path / "configured"))
    explicit = tmp_path / "explicit"

    assert ModelArtifact(pointer, "EXAMPLE_MODEL_DIR", explicit, "hub").resolve() == explicit


def test_explicit_source_precedes_environment_and_offline_mode(tmp_path, monkeypatch):
    pointer = _write_pointer(tmp_path)
    monkeypatch.setenv("HOTPOT_MODEL_SOURCE", "local")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    assert ModelArtifact(pointer, "EXAMPLE_MODEL_DIR", source="hub").source is ModelSource.HUB


@pytest.mark.parametrize("offline", ("1", "on", "TRUE", "Yes"))
def test_offline_spellings_select_local_when_source_is_unset(tmp_path, monkeypatch, offline):
    pointer = _write_pointer(tmp_path)
    monkeypatch.delenv("HOTPOT_MODEL_SOURCE", raising=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", offline)

    assert ModelArtifact(pointer, "EXAMPLE_MODEL_DIR").source is ModelSource.LOCAL


def test_environment_source_precedes_offline_mode(tmp_path, monkeypatch):
    pointer = _write_pointer(tmp_path)
    monkeypatch.setenv("HOTPOT_MODEL_SOURCE", "hub")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    assert ModelArtifact(pointer, "EXAMPLE_MODEL_DIR").source is ModelSource.HUB


@pytest.mark.parametrize("source", tuple(ModelSource))
def test_complete_materialized_cache_precedes_download(tmp_path, monkeypatch, source):
    pointer = _write_pointer(tmp_path)
    monkeypatch.setenv("HOTPOT_MODEL_HOME", str(tmp_path / "cache"))
    bundle = tmp_path / "cache" / "artifacts" / "example" / "1" / ("1" * 40) / "example" / "v1"
    bundle.mkdir(parents=True)
    (bundle / "model.onnx").write_bytes(b"model")

    def reject_download(**kwargs):
        pytest.fail("complete materialized cache must precede download")

    monkeypatch.setattr(effects, "snapshot_download", reject_download)

    assert ModelArtifact(pointer, "EXAMPLE_MODEL_DIR", source=source).resolve() == bundle


def test_hub_policy_bypasses_complete_packaged_directory(tmp_path, monkeypatch):
    pointer = _write_pointer(tmp_path)
    (pointer.parent / "model.onnx").write_bytes(b"model")
    monkeypatch.setenv("HOTPOT_MODEL_HOME", str(tmp_path / "cache"))
    calls = []
    monkeypatch.setattr(effects, "snapshot_download", lambda **kwargs: calls.append(kwargs))

    resolved = ModelArtifact(pointer, "EXAMPLE_MODEL_DIR", source="hub").resolve()

    assert resolved != pointer.parent
    assert len(calls) == 1
    assert calls[0]["local_files_only"] is False


def _with_external_files(tmp_path):
    pointer = _write_pointer(tmp_path)
    (pointer.parent / "model.onnx").write_bytes(b"model")
    # Deliberately create in reverse lexical order; bundle hashing uses names.
    second = pointer.parent / "part-b.data"
    first = pointer.parent / "part-a.data"
    second.write_bytes(b"second")
    first.write_bytes(b"first")
    manifest = load_manifest(pointer)
    manifest["models"]["default"]["external_data"] = {
        "glob": "part-*.data", "count": 2,
        "sha256": bundle_sha256([first, second]),
    }
    pointer.write_text(json.dumps(manifest), encoding="utf-8")
    return pointer, manifest, first, second


def test_external_completeness_does_not_replace_hash_verification(tmp_path):
    pointer, manifest, first, _ = _with_external_files(tmp_path)
    verify_artifact(pointer.parent, manifest)
    first.write_bytes(b"corrupt")

    assert artifact_is_complete(pointer.parent, manifest) is True
    with pytest.raises(RuntimeError) as error:
        verify_artifact(pointer.parent, manifest)
    assert str(error.value) == f"SHA-256 mismatch for external-data bundle in {pointer.parent}"


def test_external_count_preserves_exact_completeness_and_verification_errors(tmp_path):
    pointer, manifest, first, _ = _with_external_files(tmp_path)
    first.unlink()

    assert artifact_is_complete(pointer.parent, manifest) is False
    with pytest.raises(RuntimeError) as error:
        verify_artifact(pointer.parent, manifest)
    assert str(error.value) == f"Incomplete external-data bundle in {pointer.parent}"


def test_missing_model_preserves_exact_verification_error(tmp_path):
    pointer = _write_pointer(tmp_path)

    with pytest.raises(FileNotFoundError) as error:
        verify_artifact(pointer.parent, load_manifest(pointer))
    assert str(error.value) == f"Model artifact not found: {pointer.parent / 'model.onnx'}"


def test_pinned_download_preserves_external_patterns_and_arguments(tmp_path, monkeypatch):
    pointer, _, _, _ = _with_external_files(tmp_path)
    monkeypatch.setenv("HOTPOT_MODEL_HOME", str(tmp_path / "cache"))
    observed = {}
    monkeypatch.setattr(effects, "snapshot_download", lambda **kwargs: observed.update(kwargs))

    resolved = ModelArtifact(pointer, "EXAMPLE_MODEL_DIR", source="hub").resolve()

    assert observed == {
        "repo_id": "owner/models",
        "revision": "1" * 40,
        "allow_patterns": ["example/v1/manifest.json", "example/v1/model.onnx", "example/v1/part-*.data"],
        "cache_dir": tmp_path / "cache",
        "local_dir": resolved.parents[1],
        "local_files_only": False,
    }


def test_only_named_missing_local_error_is_translated_with_its_cause(tmp_path, monkeypatch):
    pointer = _write_pointer(tmp_path)
    monkeypatch.setenv("HOTPOT_MODEL_HOME", str(tmp_path / "cache"))
    original = effects.LocalEntryNotFoundError("missing")

    def missing(**kwargs):
        raise original

    monkeypatch.setattr(effects, "snapshot_download", missing)
    with pytest.raises(artifacts.ModelArtifactUnavailable) as error:
        ModelArtifact(pointer, "EXAMPLE_MODEL_DIR", source="local").resolve()

    assert str(error.value) == "example model v1 is not cached; run 'hotpot models install example' while online"
    assert error.value.__cause__ is original


def test_other_download_errors_propagate_unchanged(tmp_path, monkeypatch):
    pointer = _write_pointer(tmp_path)
    monkeypatch.setenv("HOTPOT_MODEL_HOME", str(tmp_path / "cache"))
    original = PermissionError("unavailable")

    def failed(**kwargs):
        raise original

    monkeypatch.setattr(effects, "snapshot_download", failed)
    with pytest.raises(PermissionError) as error:
        ModelArtifact(pointer, "EXAMPLE_MODEL_DIR", source="hub").resolve()

    assert error.value is original


def test_mca_model_store_uses_registered_pointer(tmp_path, monkeypatch):
    from hotpot.cheminfo.AImodels.mca import model_store

    pointer = _write_pointer(tmp_path)
    calls = []

    def resolve(pointer_path, environment_variable, **options):
        calls.append((pointer_path, environment_variable, options))
        return SimpleNamespace(resolve=lambda: pointer.parent)

    monkeypatch.setattr(model_store, "ModelArtifact", resolve)
    store = model_store.ModelStore(verify=False, model_source="local")

    assert store.model_dir == pointer.parent
    assert calls == [(ARTIFACTS["mca"].manifest_path, "HOTPOT_MCA_MODEL_DIR", {
        "model_dir": None, "source": "local",
    })]


def test_cbond_runtime_uses_registered_pointer(tmp_path, monkeypatch):
    from hotpot.cheminfo.AImodels.cbond import runtime

    manifest = {"models": {"graph": {"file": "graph.onnx"}, "cbond": {"file": "head.onnx"}}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    calls = []

    def resolve(pointer_path, environment_variable, **options):
        calls.append((pointer_path, environment_variable, options))
        return SimpleNamespace(resolve=lambda: tmp_path)

    monkeypatch.setattr(runtime, "ModelArtifact", resolve)
    monkeypatch.setattr(runtime.ort, "SessionOptions", SimpleNamespace)
    monkeypatch.setattr(runtime.ort, "InferenceSession", lambda *args, **kwargs: SimpleNamespace(
        get_providers=lambda: ["CPUExecutionProvider"],
    ))
    predictor = runtime.CBondRuntime(device="cpu", verify=False, model_source="local")

    assert predictor.model_dir == tmp_path
    assert calls == [(ARTIFACTS["cbond"].manifest_path, "HOTPOT_CBOND_MODEL_DIR", {
        "model_dir": None, "source": "local",
    })]
