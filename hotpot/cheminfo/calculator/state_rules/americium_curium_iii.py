"""OPTIONAL-006 ordinary americium/curium(III) prior."""

from __future__ import annotations

from functools import lru_cache

from .oxidation_priors import (
    _OxidationPriorArtifact,
    _OxidationStatePriorRule,
    _load_oxidation_prior_artifact,
)

__all__ = ["AmericiumCuriumIIIPriorRule"]


_RESOURCE = "data/rules/americium_curium_iii_prior_v1.json"


@lru_cache(maxsize=1)
def _artifact() -> _OxidationPriorArtifact:
    return _load_oxidation_prior_artifact(_RESOURCE, "OPTIONAL-006")


class AmericiumCuriumIIIPriorRule(_OxidationStatePriorRule):
    """Propose +3 only inside the reviewed ordinary Am/Cm donor domain."""

    def __init__(self) -> None:
        super().__init__(_artifact())
