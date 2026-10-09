"""Resolve packaged, cached, and Hugging Face model artifacts."""

from __future__ import annotations

from enum import Enum
import os
from pathlib import Path
from typing import Optional, Union

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

from .errors import ModelArtifactUnavailable
from .manifest import artifact_is_complete, load_manifest


PathLike = Union[str, os.PathLike]


class ModelSource(str, Enum):
    AUTO = "auto"
    LOCAL = "local"
    HUB = "hub"


class ModelArtifact:
    """Resolve one model bundle without changing its inference semantics."""

    def __init__(
        self,
        pointer_manifest: PathLike,
        model_env_var: str,
        model_dir: Optional[PathLike] = None,
        source: Optional[Union[str, ModelSource]] = None,
    ):
        self.pointer_path = Path(pointer_manifest)
        self.pointer = load_manifest(self.pointer_path)
        self.packaged_dir = self.pointer_path.parent
        configured = model_dir or os.environ.get(model_env_var)
        self.configured_dir = Path(configured).expanduser() if configured else None
        selected_source = source or os.environ.get("HOTPOT_MODEL_SOURCE")
        if selected_source is None:
            offline = os.environ.get("HF_HUB_OFFLINE", "").lower()
            selected_source = "local" if offline in {"1", "on", "true", "yes"} else "auto"
        self.source = ModelSource(selected_source)

    def resolve(self) -> Path:
        if self.configured_dir is not None:
            return self.configured_dir

        if self.source is not ModelSource.HUB and artifact_is_complete(
            self.packaged_dir, self.pointer
        ):
            return self.packaged_dir

        artifact = self.pointer["artifact"]
        patterns = [f"{artifact['subfolder']}/manifest.json"]
        for entry in self.pointer["models"].values():
            patterns.append(f"{artifact['subfolder']}/{entry['file']}")
            external = entry.get("external_data")
            if external is not None:
                patterns.append(f"{artifact['subfolder']}/{external['glob']}")
        cache_dir = Path(
            os.environ.get(
                "HOTPOT_MODEL_HOME",
                Path.home() / ".cache" / "hotpot" / "models",
            )
        ).expanduser()
        materialized_root = (
            cache_dir
            / "artifacts"
            / artifact["id"]
            / artifact["version"]
            / artifact["revision"]
        )
        materialized_bundle = materialized_root / artifact["subfolder"]
        if artifact_is_complete(materialized_bundle, self.pointer):
            return materialized_bundle
        try:
            snapshot_download(
                repo_id=artifact["repo_id"],
                revision=artifact["revision"],
                allow_patterns=patterns,
                cache_dir=cache_dir,
                local_dir=materialized_root,
                local_files_only=self.source is ModelSource.LOCAL,
            )
        except LocalEntryNotFoundError as exc:
            raise ModelArtifactUnavailable(
                f"{artifact['id']} model v{artifact['version']} is not cached; "
                f"run 'hotpot models install {artifact['id']}' while online"
            ) from exc
        return materialized_bundle
