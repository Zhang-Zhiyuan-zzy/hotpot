"""Public chemistry facade without external toolkit initialization."""

from .__version__ import __version__
from .cheminfo import (
    Molecule, Atom, Bond, BondKind, Crystal, MolReader, MolWriter, read_mol,
    MolBundle, to_pyg_dataset, is_molecule_input, to_hotpot_mol,
    Fragment, EdgeShoulder, AtomLink, AlkylGraft, BondAdding, AtomReplace,
    RingWedge, AssembleFactory, Searcher, Substructure, Hit, Hits,
    Query, QueryAtom, QueryBond, SmartsSemantics, SmartsSyntaxError,
    UnsupportedSmartsError,
)


__all__ = (
    "Molecule", "Atom", "Bond", "BondKind", "Crystal", "MolReader", "MolWriter",
    "read_mol", "MolBundle", "to_pyg_dataset", "is_molecule_input", "to_hotpot_mol",
    "Fragment", "EdgeShoulder", "AtomLink", "AlkylGraft", "BondAdding",
    "AtomReplace", "RingWedge", "AssembleFactory", "Searcher", "Substructure",
    "Hit", "Hits", "Query", "QueryAtom", "QueryBond", "SmartsSemantics",
    "SmartsSyntaxError", "UnsupportedSmartsError", "version",
)


def version():
    """Return the package's single declared version."""
    return __version__
