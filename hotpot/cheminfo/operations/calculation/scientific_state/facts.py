"""Prepare an owned working copy for immutable scientific-state capture."""

from __future__ import annotations

from hotpot.cheminfo.adapters.structure.scientific_state import _extract_molecular_facts
from hotpot.cheminfo.structure.molecule import Molecule
from hotpot.cheminfo.operations.chemistry import _prepare_hydrogens, _require_non_graph_hydrogens
from hotpot.cheminfo.structure.snapshots import MolecularFacts
from hotpot.cheminfo.structure._dependencies import GROUP_INDEX, _FactStatus
from hotpot.cheminfo.structure._fields import FIELD_INDEX, FIELD_PRESENCE_BITS

__all__ = ["extract_molecular_facts"]


def extract_molecular_facts(mol: Molecule) -> MolecularFacts:
    """Extract immutable chemical facts without mutating ``mol``."""
    working_copy = mol.copy()
    _prepare_hydrogens(working_copy)
    table = working_copy._atom_table
    field, group = FIELD_INDEX['implicit_hydrogens'], GROUP_INDEX['IMPLICIT_H']
    known_rows = tuple(row for row in range(len(table))
                       if table._group_status[row, group] == _FactStatus.CURRENT
                       and int(table._presence[row]) & FIELD_PRESENCE_BITS[field])
    _require_non_graph_hydrogens(working_copy, rows=known_rows)
    return _extract_molecular_facts(working_copy)
