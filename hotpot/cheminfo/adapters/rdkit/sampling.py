"""Native RDKit sampling of an already prepared molecular transport."""

from __future__ import annotations

from ...rdWrappers import native


def generate(transport, *, options, constraints=None):
    return native._backend().generate(transport, options, constraints)
