"""Registry of model pointer manifests shipped with Hotpot."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactSpec:
    name: str
    manifest_path: Path
    environment_variable: str


_ARTIFACT_DATA_DIR = Path(__file__).resolve().parent / "data"
ARTIFACTS = {
    "mca": ArtifactSpec(
        "mca",
        _ARTIFACT_DATA_DIR / "mca" / "manifest.json",
        "HOTPOT_MCA_MODEL_DIR",
    ),
    "cbond": ArtifactSpec(
        "cbond",
        _ARTIFACT_DATA_DIR / "cbond" / "manifest.json",
        "HOTPOT_CBOND_MODEL_DIR",
    ),
}
