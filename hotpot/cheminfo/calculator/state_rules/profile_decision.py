"""Decision primitives frozen by the coordination-screening profile."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Optional, Tuple

from .profiles import _ScientificStateProfileArtifact

__all__ = []


@dataclass(frozen=True)
class _AssumptionSeverityPolicy:
    """Versioned categories ordered from most to least severe."""

    identity: str
    ordered_categories: Tuple[str, ...]

    def vector(
        self,
        assumption_ids: Tuple[str, ...],
    ) -> Optional[Tuple[int, ...]]:
        categories = tuple(
            assumption_id.partition(":")[0]
            for assumption_id in assumption_ids
        )
        known = frozenset(self.ordered_categories)
        if any(not category or category not in known for category in categories):
            return None
        return tuple(
            categories.count(category)
            for category in self.ordered_categories
        )


def _assumption_severity_policy(
    profile: _ScientificStateProfileArtifact,
) -> _AssumptionSeverityPolicy:
    ordered = tuple(
        category
        for category, _ in sorted(
            profile.assumption_severity,
            key=lambda item: (-item[1], item[0]),
        )
    )
    encoded = json.dumps(
        {
            "profile_digest": profile.content_digest,
            "ordered_categories": ordered,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return _AssumptionSeverityPolicy(
        hashlib.sha256(encoded).hexdigest(),
        ordered,
    )
