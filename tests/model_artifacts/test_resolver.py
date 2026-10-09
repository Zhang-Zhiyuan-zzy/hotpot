import hashlib
import json
from pathlib import Path

import pytest

from hotpot import __main__ as hotpot_main
from hotpot.cheminfo.AImodels.artifacts.errors import ModelArtifactUnavailable
from hotpot.cheminfo.AImodels.artifacts.manifest import verify_artifact
from hotpot.cheminfo.AImodels.artifacts.resolver import ModelArtifact
from hotpot.cheminfo.AImodels.artifacts import resolver as resolver_module


def _write_pointer(tmp_path: Path) -> Path:
    pointer = tmp_path / "package" / "manifest.json"
    pointer.parent.mkdir()
    pointer.write_text(
        json.dumps(
            {
                "artifact": {
                    "id": "example",
                    "version": "1",
                    "repo_id": "owner/models",
                    "revision": "1" * 40,
                    "subfolder": "example/v1",
                },
                "models": {
                    "default": {
                        "file": "model.onnx",
                        "sha256": hashlib.sha256(b"model").hexdigest(),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return pointer


def test_complete_packaged_artifact_precedes_hub(tmp_path, monkeypatch):
    pointer = _write_pointer(tmp_path)
    (pointer.parent / "model.onnx").write_bytes(b"model")

    def reject_download(**kwargs):
        raise AssertionError("complete packaged artifacts must not be downloaded")

    monkeypatch.setattr(resolver_module, "snapshot_download", reject_download)

    assert ModelArtifact(pointer, "EXAMPLE_MODEL_DIR").resolve() == pointer.parent


def test_explicit_model_directory_has_highest_precedence(tmp_path, monkeypatch):
    pointer = _write_pointer(tmp_path)
    explicit = tmp_path / "explicit"

    def reject_download(**kwargs):
        raise AssertionError("explicit model directories must not be downloaded")

    monkeypatch.setattr(resolver_module, "snapshot_download", reject_download)

    assert ModelArtifact(
        pointer,
        "EXAMPLE_MODEL_DIR",
        model_dir=explicit,
        source="hub",
    ).resolve() == explicit


def test_hub_resolution_is_pinned_and_scoped(tmp_path, monkeypatch):
    pointer = _write_pointer(tmp_path)
    observed = {}

    def record_download(**kwargs):
        observed.update(kwargs)
        return str(kwargs["local_dir"])

    monkeypatch.setattr(resolver_module, "snapshot_download", record_download)
    monkeypatch.setenv("HOTPOT_MODEL_HOME", str(tmp_path / "cache"))

    resolved = ModelArtifact(
        pointer,
        "EXAMPLE_MODEL_DIR",
        source="hub",
    ).resolve()

    assert resolved == (
        tmp_path
        / "cache"
        / "artifacts"
        / "example"
        / "1"
        / ("1" * 40)
        / "example"
        / "v1"
    )
    assert observed["repo_id"] == "owner/models"
    assert observed["revision"] == "1" * 40
    assert observed["allow_patterns"] == [
        "example/v1/manifest.json",
        "example/v1/model.onnx",
    ]
    assert observed["local_files_only"] is False
    assert observed["local_dir"] == resolved.parents[1]


def test_local_policy_reports_a_missing_cache(tmp_path, monkeypatch):
    pointer = _write_pointer(tmp_path)

    def missing_cache(**kwargs):
        raise resolver_module.LocalEntryNotFoundError("missing")

    monkeypatch.setattr(resolver_module, "snapshot_download", missing_cache)

    with pytest.raises(ModelArtifactUnavailable, match="hotpot models install example"):
        ModelArtifact(pointer, "EXAMPLE_MODEL_DIR", source="local").resolve()


def test_standard_hugging_face_offline_toggle_selects_local_mode(tmp_path, monkeypatch):
    pointer = _write_pointer(tmp_path)
    observed = {}

    def record_download(**kwargs):
        observed.update(kwargs)
        return str(kwargs["local_dir"])

    monkeypatch.setattr(resolver_module, "snapshot_download", record_download)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HOTPOT_MODEL_HOME", str(tmp_path / "cache"))

    ModelArtifact(pointer, "EXAMPLE_MODEL_DIR").resolve()

    assert observed["local_files_only"] is True


def test_sha256_verification_accepts_exact_file_and_rejects_changes(tmp_path):
    pointer = _write_pointer(tmp_path)
    model_path = pointer.parent / "model.onnx"
    model_path.write_bytes(b"model")
    manifest = json.loads(pointer.read_text(encoding="utf-8"))

    verify_artifact(pointer.parent, manifest)
    model_path.write_bytes(b"changed")

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        verify_artifact(pointer.parent, manifest)


def test_top_level_models_list_reports_pinned_artifacts(capsys):
    assert hotpot_main.main(["models", "list"]) == 0

    output = capsys.readouterr().out
    assert "mca\tv1\tZhang-Zhiyuan-zzy/hotpot-models@" in output
    assert "cbond\tv1\tZhang-Zhiyuan-zzy/hotpot-models@" in output
