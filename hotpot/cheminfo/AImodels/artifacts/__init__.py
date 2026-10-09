"""Versioned model-artifact resolution for Hotpot inference backends."""

from .errors import ModelArtifactUnavailable
from .manifest import load_manifest, verify_artifact
from .resolver import ModelArtifact, ModelSource

__all__ = [
    "ModelArtifact",
    "ModelArtifactUnavailable",
    "ModelSource",
    "load_manifest",
    "verify_artifact",
]
