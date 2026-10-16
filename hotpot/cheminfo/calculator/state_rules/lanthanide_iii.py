"""OPTIONAL-005 ordinary lanthanide(III) oxidation-state prior."""

from __future__ import annotations

from functools import lru_cache

from .oxidation_priors import (
    _OxidationPriorArtifact,
    _OxidationStatePriorRule,
    _load_oxidation_prior_artifact,
)

__all__ = ["LanthanideIIIPriorRule"]


_RESOURCE = "data/rules/lanthanide_iii_prior_v1.json"


@lru_cache(maxsize=1)
def _artifact() -> _OxidationPriorArtifact:
    return _load_oxidation_prior_artifact(_RESOURCE, "OPTIONAL-005")


class LanthanideIIIPriorRule(_OxidationStatePriorRule):
    """Retain reviewed Ln alternatives and rank ordinary +3 candidates."""

    def __init__(self) -> None:
        super().__init__(_artifact())
