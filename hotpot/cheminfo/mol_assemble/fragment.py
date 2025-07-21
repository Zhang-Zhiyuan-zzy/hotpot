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
from typing import Union, Optional, Iterable, Callable, Literal

from hotpot.cheminfo.core import Molecule
from hotpot.cheminfo.search import Searcher, Hit


########## Definition of action functions ###################
_default_actions = {}  # register default action func

# Types template
DefaultAction = Literal['atom_link', 'bond_shoulder']
ActionFuncTemplate = Callable[[Molecule, list[int], Molecule, list[int]], Molecule]
ActionFuncTemplate = Union[ActionFuncTemplate, DefaultAction]


def actions_register(register_key: str):
    """
    Register an action function as the default action function.
    The default action functions can be invoked by a str-key in `Fragment` initialization.
    """
    def register(act_func: ActionFuncTemplate):
        _default_actions[register_key] = act_func
        return act_func
    return register

# Define common action function
@actions_register('AtomLink')
def atom_link_atom_action(
        mol: Molecule,
        hit: list[int],
        frag: Molecule,
        action_points: list[int]
):
    """
    Links a single atom in the parent molecule to a single atom in the fragment by creating a single bond.

    This function takes two molecules—a parent molecule (`mol`) and a fragment (`frag`)—and connects the atom
    specified by `hit[0]` in the parent molecule to the atom specified by `action_points[0]` in the fragment,
    using a single chemical bond. Both `hit` and `action_points` must refer to exactly one atom each.

    Args:
        mol (Molecule): The parent molecule to which the fragment will be attached.
        hit (list[int]): A list containing a single integer, the index of the atom in `mol` to be linked.
        frag (Molecule): The fragment molecule to link to the parent.
        action_points (list[int]): A list containing a single integer, the index of the atom in `frag` to be linked.

    Returns:
        Molecule: The modified parent molecule with the fragment attached by a single bond.

    Raises:
        AssertionError: If `hit` or `action_points` does not contain exactly one element.

    Example:
        new_molecule = atom_link_atom_action(mol, [3], frag, [0])
        # Links atom 3 in mol with atom 0 in frag by a single bond.
    """
    assert len(hit) == 1
    assert len(action_points) == 1

    mol_atom = mol.atoms[hit[0]]
    frag_atom = frag.atoms[action_points[0]]

    mol_atom.link_with(frag_atom)
    return mol

@actions_register('EdgeShoulder')
def shoulder_bond_action(
        mol: Molecule,
        hit: list[int],
        frag: Molecule,
        action_points: list[int]
):
    """
    Insert a fragment into the parent molecule by replacing a bond between two atoms.

    This function removes the bond between the two atoms specified by `hit` in the parent molecule,
    adds the fragment (with its atoms), and reconnects bonds to preserve molecular structure.
    Bonds and bond orders are preserved as appropriate.

    Args:
        mol (Molecule): The parent molecule (from hotpot-zzy).
        hit (list[int]): List of two indices, specifying the atoms in `mol` whose bond will be replaced.
        frag (Molecule): The fragment molecule to insert.
        action_points (list[int]): List of two indices, specifying the atoms in `frag` used to attach
            to the parent molecule.

    Returns:
        Molecule: The modified parent molecule with the fragment inserted, previously connected atoms removed.

    Raises:
        AssertionError: If `hit` or `action_points` do not contain exactly two elements.

    Details:
        - Uses hotpot-zzy `add_component`, `add_bonds`, `remove_bonds`, and `remove_atoms`.
        - Ensures correct atom re-indexing and bond order preservation during insertion.
    """
    assert len(hit) == 2
    assert len(action_points) == 2

    # update action points after add the frag as a component
    ap1, ap2 = action_points
    ap1 += len(mol.atoms)
    ap2 += len(mol.atoms)

    mol.add_component(frag)

    # Get the atoms in the replaced bond
    ma1, ma2 = mol.atoms[hit[0]], mol.atoms[hit[1]]

    # Recording the original linking net for replaced bond end (atoms)
    ma1_neigh_idx = [a.idx for a in ma1.neighbours]
    ma2_neigh_idx = [a.idx for a in ma2.neighbours]
    assert ma2.idx in ma1_neigh_idx
    assert ma1.idx in ma2_neigh_idx
    # Remove redundant link between ma1-ma2
    ma2_neigh_idx.remove(ma1.idx)
    ma1_neigh_idx.remove(ma2.idx)

    # Break all bonds with ma1 and ma2
    ma1_bonds = [mol.bond(ma1.idx, ma1n_idx) for ma1n_idx in ma1_neigh_idx]
    ma2_bonds = [mol.bond(ma2.idx, ma2n_idx) for ma2n_idx in ma2_neigh_idx]
    ma1_ma2_bond = [mol.bond(ma1.idx, ma2.idx)]

    # Recording the bond order for rebuilding below
    ma1_bond_order = [b.bond_order for b in ma1_bonds]
    ma2_bond_order = [b.bond_order for b in ma2_bonds]

    # Removing bond
    mol.remove_bonds(ma1_bonds + ma2_bonds + ma1_ma2_bond)

    # Build new link to the atoms in the fragment
    bond_ap1_info = [(ap1, ma1n_idx, ma1_bo) for ma1n_idx, ma1_bo in zip(ma1_neigh_idx, ma1_bond_order)]
    bond_ap2_info = [(ap2, ma2n_idx, ma2_bo) for ma2n_idx, ma2_bo in zip(ma2_neigh_idx, ma2_bond_order)]
    mol.add_bonds(bond_ap1_info + bond_ap2_info)

    # Remove old bond atoms
    mol.remove_atoms([ma1, ma2])

    return mol

@actions_register('BondAdd')
def bond_order_add(
        mol: Molecule,
        hit: list[int],
        frag: Molecule,
        action_points: list[int]
):
    assert len(hit) == 2

    bond = mol.bond(hit[0], hit[1])
    atom1, atom2 = bond.atoms
    hc1, hc2 = atom1.implicit_hydrogens, atom2.implicit_hydrogens
    assert hc1 > 0 and hc2 > 0

    bond.bond_order = bond.bond_order + 1
    assert hc1 - atom1.implicit_hydrogens == 1
    assert hc2 - atom2.implicit_hydrogens == 1

    return mol


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

    def graft(self, mol: Molecule) -> dict[str, Molecule]:
        """
        Apply the fragment to all valid grafting sites in the input molecule.

        Uses the associated `Searcher` to find sites in `mol`, then applies the
        fragment's `action_func` at each site, returning the set of all resulting molecules.

        Args:
            mol (Molecule): The parent molecule to which the fragment may be grafted.

        Returns:
            list[Molecule]: List of modified molecules generated by grafting the fragment at each hit site.

        Note:
            Each molecule is copied before modification; original `mol` is not altered.
        """
        hits = self.searcher.search(mol)
        hits.get_hit = False

        grafted_mol = []
        for hit in hits:
            gen_mol = self.action_func(mol.copy(), list(hit), self.frag.copy(), self.action_points)
            grafted_mol.append((gen_mol.smiles, gen_mol))

        return dict(grafted_mol)
