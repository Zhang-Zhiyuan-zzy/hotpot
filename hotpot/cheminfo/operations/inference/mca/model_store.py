"""Resolve and verify MCA ONNX artifacts."""

from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
from hotpot.cheminfo.adapters.inference.all_atom_model import materialize_all_atom_model
from hotpot.cheminfo.adapters.inference.runtime import MCARuntime, create_cuda_runtime_from_artifact

from hotpot.cheminfo.operations.inference.artifacts import (
    ModelArtifact,
    load_manifest,
    verify_artifact,
)
from hotpot.cheminfo.operations.inference.artifacts.registry import ARTIFACTS


class ModelStore:
    def __init__(self, model_dir=None, verify: bool = True, model_source=None):
        pointer = ARTIFACTS["mca"].manifest_path
        self.model_dir = ModelArtifact(
            pointer,
            "HOTPOT_MCA_MODEL_DIR",
            model_dir=model_dir,
            source=model_source,
        ).resolve()
        self.manifest = load_manifest(self.model_dir / "manifest.json")
        self.verify = verify

    def resolve(self, variant: str | None = None):
        selected = variant or self.manifest["default_variant"]
        entry = self.manifest["models"][selected]
        path = self.model_dir / entry["file"]
        if self.verify:
            verify_artifact(self.model_dir, self.manifest)
        return path, selected


@dataclass(frozen=True)
class ResolvedModelArtifact:
    session_path: Path
    variant: str
    manifest: Mapping[str, object]


@contextmanager
def resolved_model_artifact(
    *,
    model_dir=None,
    variant: str | None = None,
    verify_model: bool = True,
    model_source=None,
):
    """Resolve, verify, and materialize one model artifact for session creation."""

    store = ModelStore(
        model_dir=model_dir,
        verify=verify_model,
        model_source=model_source,
    )
    path, selected_variant = store.resolve(variant)
    external_data = store.manifest["models"][selected_variant].get("external_data")
    external_digest = (
        external_data["sha256"]
        if verify_model and external_data is not None
        else None
    )
    with materialize_all_atom_model(path, external_digest) as session_path:
        yield ResolvedModelArtifact(
            session_path=Path(session_path),
            variant=selected_variant,
            manifest=store.manifest,
        )


def create_cpu_runtime(
    *,
    model_dir=None,
    variant: str | None = None,
    verify_model: bool = True,
    model_source=None,
    cpu_threads: int | None = None,
) -> MCARuntime:
    with resolved_model_artifact(
        model_dir=model_dir,
        variant=variant,
        verify_model=verify_model,
        model_source=model_source,
    ) as artifact:
        return MCARuntime(
            artifact.session_path,
            backend="cpu",
            variant=artifact.variant,
            manifest=artifact.manifest,
            cpu_threads=cpu_threads,
        )


def create_cuda_runtime(
    *,
    model_dir=None,
    variant: str | None = None,
    verify_model: bool = True,
    model_source=None,
    device_id: int = 0,
) -> MCARuntime:
    with resolved_model_artifact(
        model_dir=model_dir,
        variant=variant,
        verify_model=verify_model,
        model_source=model_source,
    ) as artifact:
        return create_cuda_runtime_from_artifact(
            session_path=artifact.session_path,
            variant=artifact.variant,
            max_atoms=int(artifact.manifest["max_atoms"]),
            device_id=device_id,
        )
