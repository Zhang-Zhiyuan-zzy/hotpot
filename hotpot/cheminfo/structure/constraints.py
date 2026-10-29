"""Private persisted angle/torsion constraints keyed by ordered member UIDs."""

from dataclasses import dataclass


__all__ = ()


def _updated_entries(current, updates):
    entries = dict.fromkeys(current)
    for members, value in updates:
        if value:
            entries[members] = None
        else:
            entries.pop(members, None)
    return tuple(entries)


@dataclass(frozen=True)
class _ConstraintSet:
    angles: tuple[tuple[int, int, int], ...] = ()
    torsions: tuple[tuple[int, int, int, int], ...] = ()

    def updated(self, *, angles=(), torsions=()):
        return type(self)(
            _updated_entries(self.angles, angles),
            _updated_entries(self.torsions, torsions),
        )

    def subset(self, members):
        return type(self)(
            tuple(entry for entry in self.angles if all(uid in members for uid in entry)),
            tuple(entry for entry in self.torsions if all(uid in members for uid in entry)),
        )
