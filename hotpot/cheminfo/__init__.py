"""
python v3.9.0
@Project: hotpot0.5.0
@File   : __init__.py
@Auther : Zhiyuan Zhang
@Data   : 2024/6/1
@Time   : 17:23
"""
from .core import Molecule, Atom, Bond, BondKind
from .structure.crystal import Crystal
from .operations.io import MolReader, MolWriter, read_mol
from .bundle import MolBundle, to_pyg_dataset
from .operations.conversion import is_molecule_input, to_hotpot_mol
from .operations.assembly import (
    Fragment, EdgeShoulder, AtomLink, AlkylGraft, BondAdding, AtomReplace,
    RingWedge, AssembleFactory,
)

from .operations.search import Searcher, Substructure, Hit, Hits, Query, QueryAtom, QueryBond
from .operations.search import SmartsSemantics, SmartsSyntaxError, UnsupportedSmartsError

__all__ = (
    "Molecule", "Atom", "Bond", "BondKind", "Crystal", "MolReader", "MolWriter",
    "read_mol", "MolBundle", "to_pyg_dataset", "is_molecule_input", "to_hotpot_mol",
    "Fragment", "EdgeShoulder", "AtomLink", "AlkylGraft", "BondAdding",
    "AtomReplace", "RingWedge", "AssembleFactory", "Searcher", "Substructure",
    "Hit", "Hits", "Query", "QueryAtom", "QueryBond", "SmartsSemantics",
    "SmartsSyntaxError", "UnsupportedSmartsError",
)
