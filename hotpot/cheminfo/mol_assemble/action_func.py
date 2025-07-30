# -*- coding: utf-8 -*-
"""
===========================================================
 Python    : v3.9.0
 Project   : hotpot
 File      : action_func
 Created   : 2025/7/19 16:25
 Author    : Zhiyuan Zhang
 Python    : 
-----------------------------------------------------------
 Description
 ----------------------------------------------------------
 
===========================================================
"""
from typing import Literal, Callable, Union

from hotpot.cheminfo.core import Molecule

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

    # Check the aromatic in "reaction sites"
    aa1_is_aromatic = mol.atoms[ap1].is_aromatic or ma1.is_aromatic
    aa2_is_aromatic = mol.atoms[ap2].is_aromatic or ma2.is_aromatic

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

    # Reassign the aromatics
    mol.atoms[ap1].is_aromatic = aa1_is_aromatic
    mol.atoms[ap2].is_aromatic = aa2_is_aromatic

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
    assert atom1.implicit_hydrogens >= 0
    assert atom2.implicit_hydrogens >= 0

    return mol

@actions_register('AtomReplace')
def atom_replace(
        mol: Molecule,
        hit: list[int],
        frag: Molecule,
        action_points: list[int]
):
    assert len(hit) == 1
    assert len(frag.atoms) == 1

    mol.atoms[hit[0]].atomic_number = frag.atoms[0].atomic_number
    return mol

@actions_register('RingWedge')
def ring_wedge(
        mol: Molecule,
        hit: list[int],
        frag: Molecule,
        action_points: list[int]
):
    assert len(hit) == 1
    assert len(action_points) == 1

    ma = mol.atoms[hit[0]]
    heavy_neighbours = ma.heavy_neighbours
    assert len(heavy_neighbours) == 1
    h_neigh = heavy_neighbours[0]

    mol.remove_atom(ma)

    ap_after_link = action_points[0] + len(mol.atoms)

    ap_atom_before_link = frag.atoms[action_points[0]]
    h_neigh.link_with(ap_atom_before_link)

    assert (check_value := mol.atoms[ap_after_link].atomic_number) == ap_atom_before_link.atomic_number

    ap_atom_after_link = mol.replace_atom(ap_after_link, ma)
    assert ap_atom_after_link.atomic_number == check_value

    return mol

