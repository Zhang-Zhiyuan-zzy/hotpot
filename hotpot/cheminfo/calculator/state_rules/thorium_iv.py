"""OPTIONAL-007 ordinary thorium(IV) prior."""

from __future__ import annotations

from functools import lru_cache

from .oxidation_priors import (
    _OxidationPriorArtifact,
    _OxidationStatePriorRule,
    _load_oxidation_prior_artifact,
)

__all__ = ["ThoriumIVPriorRule"]


_RESOURCE = "data/rules/thorium_iv_prior_v1.json"


@lru_cache(maxsize=1)
def _artifact() -> _OxidationPriorArtifact:
    return _load_oxidation_prior_artifact(_RESOURCE, "OPTIONAL-007")


class ThoriumIVPriorRule(_OxidationStatePriorRule):
    """Propose +4 only inside the reviewed ordinary Th donor domain."""

    def __init__(self) -> None:
        super().__init__(_artifact())
