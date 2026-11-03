"""Errors raised while resolving external inference artifacts."""


class ModelArtifactUnavailable(FileNotFoundError):
    """A pinned model artifact is absent from all permitted sources."""
