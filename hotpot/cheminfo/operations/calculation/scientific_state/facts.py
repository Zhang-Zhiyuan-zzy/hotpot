"""Prepare an owned working copy for immutable scientific-state capture."""

from __future__ import annotations

from hotpot.cheminfo.adapters.structure.scientific_state import _extract_molecular_facts
from hotpot.cheminfo.core import Molecule
from hotpot.cheminfo.operations.chemistry import _prepare_hydrogens
from hotpot.cheminfo.structure.snapshots import MolecularFacts

__all__ = ["extract_molecular_facts"]


def extract_molecular_facts(mol: Molecule) -> MolecularFacts:
    """Extract immutable chemical facts without mutating ``mol``."""
    working_copy = mol.copy()
    _prepare_hydrogens(working_copy)
    return _extract_molecular_facts(working_copy)
