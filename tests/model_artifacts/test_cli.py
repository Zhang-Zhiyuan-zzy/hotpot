"""Preserved model command output and local missing-artifact behavior."""

import argparse
from types import SimpleNamespace

import pytest

from hotpot.commands import models
from hotpot.cheminfo.operations.inference.artifacts.errors import ModelArtifactUnavailable
from hotpot.cheminfo.operations.inference.artifacts.resolver import ModelSource


@pytest.mark.parametrize("action,label,source", (
    ("status", "installed", ModelSource.LOCAL),
    ("verify", "verified", ModelSource.LOCAL),
    ("install", "installed", ModelSource.HUB),
))
def test_model_action_preserves_source_output_and_verification(
    tmp_path, monkeypatch, capsys, action, label, source,
):
    observed = []
    manifest = {"models": {}}

    def resolver(name, selected_source):
        observed.append((name, selected_source))
        return SimpleNamespace(resolve=lambda: tmp_path)

    monkeypatch.setattr(models, "_resolver", resolver)
    monkeypatch.setattr(models, "load_manifest", lambda path: manifest)
    verified = []
    monkeypatch.setattr(models, "verify_artifact", lambda path, value: verified.append((path, value)))
    args = argparse.Namespace(models_action=action, models=["mca"], all=False)

    assert models.run(args) == 0
    assert capsys.readouterr().out == f"mca\t{label}\t{tmp_path}\n"
    assert observed == [("mca", source)]
    assert verified == ([] if action == "status" else [(tmp_path, manifest)])


@pytest.mark.parametrize("action", ("status", "verify"))
def test_missing_local_model_is_reported_and_independent_model_continues(
    tmp_path, monkeypatch, capsys, action,
):
    def resolver(name, source):
        if name == "mca":
            raise ModelArtifactUnavailable("missing")
        return SimpleNamespace(resolve=lambda: tmp_path)

    monkeypatch.setattr(models, "_resolver", resolver)
    monkeypatch.setattr(models, "load_manifest", lambda path: {})
    monkeypatch.setattr(models, "verify_artifact", lambda path, manifest: None)
    args = argparse.Namespace(models_action=action, models=["mca", "cbond"], all=False)

    assert models.run(args) == (1 if action == "verify" else 0)
    label = "verified" if action == "verify" else "installed"
    assert capsys.readouterr().out == f"mca\tmissing\ncbond\t{label}\t{tmp_path}\n"


def test_install_failure_propagates_without_success_output(monkeypatch, capsys):
    original = ModelArtifactUnavailable("not cached")

    def resolver(name, source):
        raise original

    monkeypatch.setattr(models, "_resolver", resolver)
    args = argparse.Namespace(models_action="install", models=["mca"], all=False)

    with pytest.raises(ModelArtifactUnavailable) as error:
        models.run(args)
    assert error.value is original
    assert capsys.readouterr().out == ""


def test_unknown_model_keeps_the_existing_error():
    args = argparse.Namespace(models_action="status", models=["missing"], all=False)

    with pytest.raises(ValueError) as error:
        models.run(args)
    assert str(error.value) == "Unknown model artifact: missing"
