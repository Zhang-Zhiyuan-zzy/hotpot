"""CLI for installing and verifying external Hotpot model artifacts."""

from __future__ import annotations

import argparse

from .errors import ModelArtifactUnavailable
from .manifest import load_manifest, verify_artifact
from .registry import ARTIFACTS
from .resolver import ModelArtifact, ModelSource


def add_arguments(parser: argparse.ArgumentParser) -> None:
    actions = parser.add_subparsers(dest="models_action", required=True)
    actions.add_parser("list", help="list pinned model artifacts")

    for action, help_text in (
        ("status", "show whether model artifacts are cached"),
        ("install", "download and verify model artifacts"),
        ("verify", "verify cached model artifacts against SHA-256"),
    ):
        action_parser = actions.add_parser(action, help=help_text)
        action_parser.add_argument(
            "models",
            nargs="*",
            metavar="MODEL",
            help="model name; all models are selected when omitted",
        )
        action_parser.add_argument(
            "--all",
            action="store_true",
            help="select every registered model",
        )


def _selected_models(args: argparse.Namespace) -> tuple[str, ...]:
    selected = tuple(ARTIFACTS) if args.all or not args.models else tuple(args.models)
    unknown = tuple(name for name in selected if name not in ARTIFACTS)
    if unknown:
        raise ValueError(f"Unknown model artifact: {', '.join(unknown)}")
    return selected


def _resolver(name: str, source: ModelSource) -> ModelArtifact:
    spec = ARTIFACTS[name]
    return ModelArtifact(
        spec.manifest_path,
        spec.environment_variable,
        source=source,
    )


def run(args: argparse.Namespace) -> int:
    if args.models_action == "list":
        for name, spec in ARTIFACTS.items():
            artifact = load_manifest(spec.manifest_path)["artifact"]
            print(
                f"{name}\tv{artifact['version']}\t"
                f"{artifact['repo_id']}@{artifact['revision']}"
            )
        return 0

    missing = False
    for name in _selected_models(args):
        if args.models_action == "install":
            model_dir = _resolver(name, ModelSource.HUB).resolve()
            verify_artifact(model_dir, load_manifest(model_dir / "manifest.json"))
            print(f"{name}\tinstalled\t{model_dir}")
            continue

        try:
            model_dir = _resolver(name, ModelSource.LOCAL).resolve()
        except ModelArtifactUnavailable:
            print(f"{name}\tmissing")
            missing = True
            continue

        if args.models_action == "verify":
            verify_artifact(model_dir, load_manifest(model_dir / "manifest.json"))
            print(f"{name}\tverified\t{model_dir}")
        else:
            print(f"{name}\tinstalled\t{model_dir}")
    return 1 if missing and args.models_action == "verify" else 0
