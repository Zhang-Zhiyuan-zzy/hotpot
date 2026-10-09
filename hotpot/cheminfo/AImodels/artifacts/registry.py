"""Registry of model pointer manifests shipped with Hotpot."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactSpec:
    name: str
    manifest_path: Path
    environment_variable: str


_AIMODELS_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS = {
    "mca": ArtifactSpec(
        "mca",
        _AIMODELS_DIR / "mca" / "models" / "manifest.json",
        "HOTPOT_MCA_MODEL_DIR",
    ),
    "cbond": ArtifactSpec(
        "cbond",
        _AIMODELS_DIR / "cbond" / "onnx" / "manifest.json",
        "HOTPOT_CBOND_MODEL_DIR",
    ),
}
