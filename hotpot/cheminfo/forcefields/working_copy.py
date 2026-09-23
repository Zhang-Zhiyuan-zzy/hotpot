"""Transactional molecule working copies for force-field workflows."""

from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple, TYPE_CHECKING, TypedDict, cast

import networkx as nx
import numpy as np
from openbabel import openbabel as ob

from .coordination import _iter_metal_donor_pairs
from .topology import (
    _atom_identity,
    _atom_index_map,
    _bond_endpoint_indices,
    _bond_identity,
)


if TYPE_CHECKING:
    from ..core import Angle, Atom, AtomPair, Bond, BondKind, Molecule, Ring, Torsion


__all__ = ()


_NEUTRAL_DONOR_ATOMIC_NUMBERS = frozenset({7, 8, 15, 16, 33, 34})


class _BondAttributePayload(TypedDict):
    bond_order: float
    constraint: bool
    id: int
    bond_kind: "BondKind"
    bond_direction: Optional[str]
    bond_source: Optional[str]
    bond_source_metadata: Mapping[str, object]


@dataclass(frozen=True)
class _WorkingCopyCommit:
    original_atom_attrs: Tuple[np.ndarray, ...]
    added_atom_attrs: Tuple[np.ndarray, ...]
    added_bonds: Tuple[Tuple[int, int, _BondAttributePayload], ...]
    conformer_state: Mapping[str, object]
    conformer_index: int


@dataclass(frozen=True)
class _MoleculeCommitSnapshot:
    atoms: Tuple["Atom", ...]
    bonds: Tuple["Bond", ...]
    atom_state: Tuple[
        Tuple["Atom", np.ndarray, list["Atom"], list["Bond"]], ...
    ]
    graph: nx.Graph
    row_to_index: Optional[dict[int, int]]
    angles: list["Angle"]
    torsions: list["Torsion"]
    rings: list["Ring"]
    cycle_basis_rings: list["Ring"]
    ring_indices_cache: dict[
        Tuple[
            bool,
            Optional[int],
            Optional[int],
            Optional[Tuple[Tuple[int, ...], Tuple[Tuple[int, int], ...]]],
        ],
        Tuple[Tuple[int, ...], ...],
    ]
    ligand_rings: Optional[list["Ring"]]
    ligand_cycle_basis_rings: Optional[list["Ring"]]
    ligand_rings_signature: Optional[
        Tuple[Tuple[int, ...], Tuple[Tuple[int, int], ...]]
    ]
    obmol: Optional[ob.OBMol]
    atom_pair_items: Tuple[Tuple[frozenset["Atom"], "AtomPair"], ...]
    conformer_state: Mapping[str, object]
    conformer_index: int


def _copy_molecule_metadata(source_mol: "Molecule", target_mol: "Molecule") -> None:
    target_mol.charge = source_mol.charge
    target_mol.properties = dict(source_mol.properties)
    target_mol._model = source_mol._model
    target_mol._environ = source_mol._environ
    target_mol._crystal = source_mol._crystal


def _recalculate_neutral_donor_valence(
    mol: "Molecule",
    donor_indices: set[int],
) -> None:
    """Infer neutral donor hydrogens from the metal-free ligand skeleton."""
    for donor_index in donor_indices:
        donor = mol.atoms[donor_index]
        if (
            donor.formal_charge == 0
            and donor.atomic_number in _NEUTRAL_DONOR_ATOMIC_NUMBERS
        ):
            donor.valence = donor.get_valence()
            donor.calc_implicit_hydrogens()


def _hydrogenated_working_copy(
    mol: "Molecule",
    *,
    add_hydrogens: bool,
    seed: Optional[int] = None,
) -> "Molecule":
    """Copy ``mol`` and infer H atoms against its ligand covalent skeleton."""
    working_mol = copy(mol)
    _copy_molecule_metadata(mol, working_mol)
    original_atom_count = len(working_mol.atoms)
    if add_hydrogens:
        if working_mol.has_metal:
            donor_indices = {
                donor.idx
                for _, donor in _iter_metal_donor_pairs(working_mol)
            }
            working_mol.hide_metal_ligand_bonds(clear_conformers=False)
            _recalculate_neutral_donor_valence(working_mol, donor_indices)
            working_mol.add_hydrogens(
                rm_polar_hs=False,
                rng=np.random.default_rng(seed),
            )
            working_mol.recover_hided_metal_ligand_bonds(clear_conformers=False)
        else:
            working_mol.add_hydrogens(
                rm_polar_hs=False,
                rng=np.random.default_rng(seed),
            )
    used_ids = {
        int(atom.id) for atom in working_mol.atoms[:original_atom_count]
    }
    next_id = max(used_ids, default=-1) + 1
    for atom in working_mol.atoms[original_atom_count:]:
        while next_id in used_ids:
            next_id += 1
        atom.id = next_id
        used_ids.add(next_id)
        next_id += 1
    return working_mol


def _make_worker_mol(mol: "Molecule") -> "Molecule":
    """Return a structure-only clone with private positional IDs for a worker."""
    worker_mol = copy(mol)
    worker_mol.charge = mol.charge
    worker_mol.refresh_atom_id()
    return worker_mol


def _prepare_working_copy_commit(
    original_mol: "Molecule",
    working_mol: "Molecule",
) -> _WorkingCopyCommit:
    original_atoms = tuple(original_mol._atoms)
    working_atoms = tuple(working_mol.atoms)
    original_atom_count = len(original_atoms)
    if len(working_atoms) < original_atom_count:
        raise ValueError("The working copy removed an original atom")

    for original_atom, working_atom in zip(
        original_atoms,
        working_atoms[:original_atom_count],
    ):
        if _atom_identity(original_atom) != _atom_identity(working_atom):
            raise ValueError("The working copy changed an original atom identity")

    original_atom_indices = _atom_index_map(original_atoms)
    working_atom_indices = _atom_index_map(working_atoms)
    original_bonds = {
        _bond_identity(bond, original_atom_indices)
        for bond in original_mol.bonds
    }
    working_original_bonds = set()
    working_bond_keys = set()
    added_bonds: list[Tuple[int, int, _BondAttributePayload]] = []
    for bond in working_mol.bonds:
        if (
            id(bond.atom1) not in working_atom_indices
            or id(bond.atom2) not in working_atom_indices
        ):
            raise ValueError("The working copy contains a bond to an external atom")
        first, second = _bond_endpoint_indices(bond, working_atom_indices)
        key = tuple(sorted((first, second)))
        if first == second or key in working_bond_keys:
            raise ValueError("The working copy contains an invalid duplicate bond")
        working_bond_keys.add(key)
        if first < original_atom_count and second < original_atom_count:
            working_original_bonds.add(
                _bond_identity(bond, working_atom_indices)
            )
        else:
            attributes = cast(_BondAttributePayload, deepcopy(bond.attr_dict))
            added_bonds.append((first, second, attributes))

    if working_original_bonds != original_bonds:
        raise ValueError("The working copy changed the original bond topology")

    return _WorkingCopyCommit(
        original_atom_attrs=tuple(
            np.array(atom.attrs, copy=True)
            for atom in working_atoms[:original_atom_count]
        ),
        added_atom_attrs=tuple(
            np.array(atom.attrs, copy=True)
            for atom in working_atoms[original_atom_count:]
        ),
        added_bonds=tuple(added_bonds),
        conformer_state=deepcopy(working_mol._conformers.__dict__),
        conformer_index=working_mol._conformers_index,
    )


def _snapshot_molecule_for_commit(mol: "Molecule") -> _MoleculeCommitSnapshot:
    return _MoleculeCommitSnapshot(
        atoms=tuple(mol._atoms),
        bonds=tuple(mol._bonds),
        atom_state=tuple(
            (atom, atom.attrs, atom._neighbours, atom._bonds)
            for atom in mol._atoms
        ),
        graph=mol._graph,
        row_to_index=mol._row2idx,
        angles=mol._angles,
        torsions=mol._torsions,
        rings=mol._rings,
        cycle_basis_rings=mol._cycle_basis_rings,
        ring_indices_cache=mol._ring_indices_cache,
        ligand_rings=mol._ligand_rings,
        ligand_cycle_basis_rings=mol._ligand_cycle_basis_rings,
        ligand_rings_signature=mol._ligand_rings_signature,
        obmol=mol._obmol,
        atom_pair_items=tuple(mol._atom_pairs.items()),
        conformer_state=dict(mol._conformers.__dict__),
        conformer_index=mol._conformers_index,
    )


def _restore_failed_commit(
    mol: "Molecule",
    snapshot: _MoleculeCommitSnapshot,
) -> None:
    mol._atoms[:] = snapshot.atoms
    mol._bonds[:] = snapshot.bonds
    for atom, attrs, neighbours, bonds in snapshot.atom_state:
        object.__setattr__(atom, "attrs", attrs)
        object.__setattr__(atom, "_neighbours", neighbours)
        object.__setattr__(atom, "_bonds", bonds)
    mol._graph = snapshot.graph
    mol._row2idx = snapshot.row_to_index
    mol._angles = snapshot.angles
    mol._torsions = snapshot.torsions
    mol._rings = snapshot.rings
    mol._cycle_basis_rings = snapshot.cycle_basis_rings
    mol._ring_indices_cache = snapshot.ring_indices_cache
    mol._ligand_rings = snapshot.ligand_rings
    mol._ligand_cycle_basis_rings = snapshot.ligand_cycle_basis_rings
    mol._ligand_rings_signature = snapshot.ligand_rings_signature
    mol._obmol = snapshot.obmol
    dict.clear(mol._atom_pairs)
    dict.update(mol._atom_pairs, snapshot.atom_pair_items)
    mol._conformers.__dict__.clear()
    mol._conformers.__dict__.update(snapshot.conformer_state)
    mol._conformers_index = snapshot.conformer_index


def _commit_working_copy(
    original_mol: "Molecule",
    working_mol: "Molecule",
) -> None:
    """Commit accepted geometry while preserving caller-owned object identities."""
    payload = _prepare_working_copy_commit(original_mol, working_mol)
    snapshot = _snapshot_molecule_for_commit(original_mol)
    try:
        for atom, attrs in zip(original_mol._atoms, payload.original_atom_attrs):
            atom.attrs = attrs
        for attrs in payload.added_atom_attrs:
            original_mol._create_atom_from_array(attrs)
        for first, second, attributes in payload.added_bonds:
            original_mol._add_bond(first, second, **attributes)

        original_mol._update_graph(clear_conformers=False)
        original_mol._row2idx = None
        original_mol._atom_pairs.update_pairs()
        original_mol._conformers.__dict__.clear()
        original_mol._conformers.__dict__.update(payload.conformer_state)
        original_mol._conformers_index = payload.conformer_index
    except BaseException:
        _restore_failed_commit(original_mol, snapshot)
        raise
