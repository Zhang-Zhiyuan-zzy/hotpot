"""Private aromatic/Kekulé connection to the sole built-in rule.

``_evaluate_aromatic_assignment(snapshot, *, kekule)`` consumes the existing
immutable ``_BondingSnapshot`` and returns ``_AromaticAssignment | None``.
Rows, ordered atom/bond UIDs, source binding and selected topology keep their
snapshot meanings. Bond orders are dimensionless; coordinates are angstrom.
The existing rule's ring order, H convention and modeling limits are retained.

A proposal is complete for its snapshot, including nonaromatic rows. ``None``
means that this rule did not produce a complete assignment; it is not an empty
or successful proposal. Exceptions propagate. The operation retains preparation,
source validation, unresolved-state policy and atomic commit/rollback.

This explicit function binding adds no wrapper, registry or alternate algorithm.
It is private; ordinary Molecule methods remain the public entry points.
"""

from ..rules.chemistry.bonding.kekule import (
    _calculate_aromatic_assignment as _evaluate_aromatic_assignment,
)


__all__ = ()
