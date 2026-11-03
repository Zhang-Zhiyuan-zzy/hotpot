"""Immutable owner/dependency binding shared by snapshots and source lifecycle."""

from __future__ import annotations

from dataclasses import dataclass


__all__ = ()


@dataclass(frozen=True, eq=False)
class _SourceIdentity:
    owner: object
    dependencies: tuple[tuple[str, str], ...]
