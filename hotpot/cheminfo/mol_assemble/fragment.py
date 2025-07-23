# -*- coding: utf-8 -*-
"""
===========================================================
 Project   : hotpot
 File      : mol_assemble
 Created   : 2025/5/19 14:36
 Author    : zhang
 Python    : 
-----------------------------------------------------------
 Description
 ----------------------------------------------------------
    Module for Common Molecular Fragment Actions (extension for hotpot-zzy)

    This module defines reusable functions and classes for manipulating molecular structures by
    attaching or grafting molecular fragments onto parent molecules, as part of a molecular
    generation or editing pipeline. It is designed as an extension to the hotpot-zzy package—see
    https://github.com/Zhang-Zhiyuan-zzy/hotpot—for integration with its molecular modeling framework.

    Core Concepts and Dependencies (from hotpot-zzy):
    - `Molecule`: Core class representing a chemical structure with accessible atoms, bonds, and methods such as
        `add_component`, `add_bonds`, `remove_atoms`, and `remove_bonds`.
    - Atoms and bonds are manipulated via their attributes (e.g., `atom.idx`, `atom.neighbours`, `bond.bond_order`)
        and linking methods (e.g., `atom.link_with`).
    - `Searcher`: Utility for locating potential attachment sites ('hits') in a parent molecule.

    Key Components:
    ----------------

    1. atom_link_atom_action(mol, hit, frag, action_points):
        - Action function to form a single bond between an atom in the parent molecule (`mol`, at index from `hit`)
          and an atom in the fragment (`frag`, at index from `action_points`). Uses hotpot's low-level atom
          linkage method for rapid attachment.

    2. shoulder_bond_action(mol, hit, frag, action_points):
        - More complex action for replacing a bond in the parent molecule: attaches a molecular fragment to the parent by
          removing a bond (between two atoms specified in `hit`), inserting a new fragment, and remapping connectivity.
          Handles bond order preservation and atom bookkeeping using hotpot-zzy's molecule methods.

    3. Fragment class:
        - Encapsulates a fragment molecule, its action points, a searcher (for finding graft locations in parents),
          and an action function (as above). Provides the `graft(mol)` method to generate modified molecules by applying
          the action at all valid sites located by the searcher.

    Hotpot-specific logic is used throughout:
    - Atom and bond manipulation leverages the hotpot `Molecule` API.
    - Atom and bond indexing strictly adheres to the molecule's internal representations.
    - Grafting and bond insertion/removal align with hotpot's design of molecular graph editing.

    Typical Usage:
    --------------
    - Define fragments and strategies for connecting them to parent molecules.
    - Use custom or hotpot-provided searchers to determine locations for molecular modification.
    - Apply `Fragment.graft` with a given molecule to produce a set of new structures.

    Note: These utilities expect valid `Molecule` and `Searcher` objects from the hotpot-zzy package
    and are not standalone.

===========================================================
"""
from typing import Optional, Iterable

from hotpot.cheminfo.core import Molecule
from hotpot.cheminfo.search import Searcher
from hotpot.cheminfo.mol_assemble.action_func import ActionFuncTemplate


class Fragment:
    """
    Represents a molecular fragment, its potential action sites, and the logic for attaching it to a target molecule.

    This class encapsulates:
        - The fragment molecule.
        - An action point specification (list of atom indices).
        - An action function (e.g. `atom_link_atom_action` or `shoulder_bond_action`).
        - A `Searcher` to locate valid grafting sites in a parent molecule.

    Args:
        mol (Molecule): The molecular fragment.
        searcher (Searcher): An object, typically from hotpot-zzy, for identifying valid graft sites in molecules.
        action_points (Iterable[int]): Indices in `mol` serving as connection points.
        action_func (Callable): Function for performing the graft (must follow standard action signature).

    Methods:
        graft(mol): For every valid hit found by `searcher`, applies `action_func` to graft the fragment.

    Example:
        frag = Fragment(some_mol, my_searcher, [0], atom_link_atom_action)
        new_mols = frag.graft(parent_mol)
    """
    def __init__(
            self,
            mol: Molecule,
            searcher: Searcher,
            action_points: Iterable[int],
            action_func: Optional[ActionFuncTemplate],
    ):
        self.frag = mol
        self.searcher = searcher
        self.action_points = list(action_points)
        self.action_func = action_func

    def __repr__(self):
        return f"{self.__class__.__name__}(frag={self.frag.smiles}, searcher={self.searcher}, action_points={self.action_points})"

    def graft(self, frame: Molecule) -> dict[str, Molecule]:
        """
        Apply the fragment to all valid grafting sites in the input molecule.

        Uses the associated `Searcher` to find sites in `mol`, then applies the
        fragment's `action_func` at each site, returning the set of all resulting molecules.

        Args:
            frame (Molecule): The parent molecule to which the fragment may be grafted.

        Returns:
            list[Molecule]: List of modified molecules generated by grafting the fragment at each hit site.

        Note:
            Each molecule is copied before modification; original `mol` is not altered.
        """
        frame.calc_implicit_hydrogens()
        hits = self.searcher.search(frame)
        hits.get_hit = False

        grafted_mol = []
        for hit in hits:
            gen_mol = self.action_func(frame.copy(), list(hit), self.frag.copy(), self.action_points)
            gen_mol.calc_implicit_hydrogens()
            grafted_mol.append((gen_mol.smiles, gen_mol))

        return dict(grafted_mol)
