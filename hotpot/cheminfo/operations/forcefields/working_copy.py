"""Transactional molecule working copies for force-field workflows."""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, replace
from typing import Mapping, Optional, TYPE_CHECKING

import numpy as np

from hotpot.cheminfo.structure._stereo import _get_stereo_payload
from hotpot.cheminfo.structure._identity import _capture_source, _check_source, _complete_bonds
from hotpot.cheminfo.adapters.structure.forcefields import _iter_metal_donor_pairs
from hotpot.cheminfo.adapters.structure.forcefields import (
    _atom_identity,
    _atom_index_map,
    _bond_endpoint_indices,
    _bond_identity,
)


if TYPE_CHECKING:
    from hotpot.cheminfo.structure.stereo import StereoPayload
    from hotpot.cheminfo.structure._builder import MoleculeRecord
    from hotpot.cheminfo.structure._source_identity import _SourceIdentity


__all__ = ()


_NEUTRAL_DONOR_ATOMIC_NUMBERS = frozenset({7, 8, 15, 16, 33, 34})
_FF_SOURCE_DOMAINS = (
    "topology", "geometry", "cell", "constraints", "source_frames", "forcefield_inputs",
)


def _capture_working_source(mol: "Molecule") -> tuple:
    from hotpot.cheminfo.operations.chemistry import _require_aromatic_representation, _require_implicit_hydrogens
    with mol._mutation_lock:
        _require_implicit_hydrogens(mol)
        _require_aromatic_representation(mol)
        return (
            _capture_source(mol, _FF_SOURCE_DOMAINS),
            tuple((bond, bond._bond_source_metadata) for bond in _complete_bonds(mol)),
        )


def _check_working_source(mol: "Molecule", source: tuple) -> None:
    _check_source(mol, source[0])
    if any(bond._bond_source_metadata is not metadata for bond, metadata in source[1]):
        raise RuntimeError("Molecular Bond metadata changed; recalculate before committing")


@dataclass(frozen=True)
class _WorkingCopyCommit:
    source: tuple
    record: "MoleculeRecord"
    bond_metadata: tuple
    atom_bindings: tuple
    bond_bindings: tuple
    dependency_state: tuple
    stereo_payload: "StereoPayload"


@dataclass(frozen=True)
class _MoleculeCommitSnapshot:
    state: Mapping[str, object]
    member_state: tuple
    conformer_state: Mapping[str, object]
    membership: tuple
    cache_state: tuple


def _copy_molecule_metadata(source_mol: "Molecule", target_mol: "Molecule") -> None:
    # copy() already remaps declarations and cell; annotation values retain
    # their existing user-owned lifetime, with a separate mapping container.
    target_mol.properties = dict(source_mol.properties)


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
            donor.set_valence_to_default()
            donor.calc_implicit_hydrogens()


def _hydrogenate_ligand_projection(
    working_mol: "Molecule",
    donor_indices: set[int],
    seed: Optional[int],
) -> None:
    from hotpot.cheminfo.operations.topology_queries import _project_molecule
    from hotpot.cheminfo.structure._commit import _FactTransaction, _commit_topology
    from hotpot.cheminfo.structure._dependencies import _mark_topology_dirty, _remap_dependency_state
    from hotpot.cheminfo.structure._fields import ATOM_SOURCE_CODES
    from hotpot.cheminfo.structure._selection import _capture_topology_selection
    from .acceptance import _topology_checks
    from hotpot.cheminfo.adapters.structure.forcefields import capture_topology

    with working_mol._mutation_lock:
        source = _capture_source(working_mol, _FF_SOURCE_DOMAINS)
        original_atoms = tuple(working_mol._atoms)
        original_bonds = _complete_bonds(working_mol)
        original_record, original_metadata = working_mol._capture_factual_record()
        selection = _capture_topology_selection(working_mol, ligand_skeleton=True)
        selected_rows = {row: target for target, row in enumerate(
            row for row, active in enumerate(selection.bond_mask) if active)}
        ligand_mol = _project_molecule(working_mol, type(working_mol), selection=selection)
    _copy_molecule_metadata(working_mol, ligand_mol)
    projected_atoms = tuple(ligand_mol._atoms)
    projected_bonds = _complete_bonds(ligand_mol)
    reference = capture_topology(ligand_mol)
    _recalculate_neutral_donor_valence(ligand_mol, donor_indices)
    ligand_mol.add_hydrogens(rm_polar_hs=False, rng=np.random.default_rng(seed))
    if (len(ligand_mol._atoms) < len(projected_atoms)
            or any(first is not second for first, second in
                   zip(projected_atoms, ligand_mol._atoms))
            or len(ligand_mol._bonds) < len(projected_bonds)
            or any(first is not second for first, second in
                   zip(projected_bonds, ligand_mol._bonds))
            or any(not check.passed for check in _topology_checks(ligand_mol, reference))):
        raise ValueError("The hydrogen working projection changed its retained topology")

    projected_record, projected_metadata = ligand_mol._capture_factual_record()
    bond_rows = tuple(
        (projected_record, projected_metadata, selected_rows[row])
        if row in selected_rows else (original_record, original_metadata, row)
        for row in range(len(original_bonds))) + tuple(
        (projected_record, projected_metadata, row)
        for row in range(len(projected_bonds), len(ligand_mol._bonds)))
    columns = ("bond_endpoints", "bond_kinds", "bond_directions", "bond_sources",
               "bond_metadata", "bond_aromatic")
    record = replace(projected_record,
        source_frames=original_record.source_frames,
        bond_values=b"".join(record.bond_values[row * 24:(row + 1) * 24]
                             for record, _, row in bond_rows),
        **{name: tuple(getattr(record, name)[row] for record, _, row in bond_rows)
           for name in columns})
    metadata = tuple(values[row] for _, values, row in bond_rows)
    added_atom_count = record.atom_count - len(original_atoms)
    added_bond_count = len(record.bond_endpoints) - len(original_bonds)
    with _FactTransaction(working_mol, producer=ATOM_SOURCE_CODES["forcefield"]):
        _check_source(working_mol, source)
        final_uids = tuple(atom._uid for atom in original_atoms) + tuple(range(
            working_mol._next_atom_uid, working_mol._next_atom_uid + added_atom_count))
        dependency_state = _remap_dependency_state(
            ligand_mol._atom_table, tuple(atom._uid for atom in ligand_mol._atoms), final_uids)
        _commit_topology(working_mol, source, record,
            atom_bindings=original_atoms + (None,) * added_atom_count,
            bond_bindings=original_bonds + (None,) * added_bond_count,
            dependency_state=dependency_state, bond_metadata=metadata)
        reinstated_uids = tuple({atom._uid for row, bond in enumerate(original_bonds)
                                 if row not in selected_rows for atom in bond._atoms})
        if reinstated_uids:
            _mark_topology_dirty(working_mol, reinstated_uids)


def _hydrogenated_working_copy(
    mol: "Molecule",
    *,
    add_hydrogens: bool,
    seed: Optional[int] = None,
    source: tuple | None = None,
) -> "Molecule":
    """Copy ``mol`` and infer H atoms against its ligand covalent skeleton."""
    with mol._mutation_lock:
        if source is not None:
            _check_working_source(mol, source)
        working_mol = copy(mol)
        _copy_molecule_metadata(mol, working_mol)
    original_atom_count = len(working_mol.atoms)
    if add_hydrogens:
        if working_mol.has_metal:
            donor_indices = {
                donor.idx
                for _, donor in _iter_metal_donor_pairs(working_mol)
            }
            _hydrogenate_ligand_projection(working_mol, donor_indices, seed)
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
    worker_mol.properties = {}
    worker_mol.refresh_atom_id()
    return worker_mol


def _prepare_working_copy_commit(
    original_mol: "Molecule",
    working_mol: "Molecule",
    source: tuple,
) -> _WorkingCopyCommit:
    from hotpot.cheminfo.structure._dependencies import _remap_dependency_state

    original_atoms = tuple(original_mol._atoms)
    working_atoms = tuple(working_mol._atoms)
    original_count = len(original_atoms)
    if len(working_atoms) < original_count:
        raise ValueError("The working copy removed an original atom")
    # copy() allocates a fresh 0..N-1 UID range in source row order. Source
    # labels/chemistry can coincide, so they cannot identify this retained prefix.
    if any(atom.mol is not working_mol or atom._uid_owner is not working_mol._owner_token
           or atom._uid != row or working_mol._topology_index.get(id(atom), (None,))[0] != row
           for row, atom in enumerate(working_atoms[:original_count])):
        raise ValueError("The working copy changed an original atom membership")
    if any(_atom_identity(first) != _atom_identity(second)
           for first, second in zip(original_atoms, working_atoms)):
        raise ValueError("The working copy changed an original atom identity")

    original_indices = _atom_index_map(original_atoms)
    working_indices = _atom_index_map(working_atoms)
    original_bonds = _complete_bonds(original_mol)
    working_bonds = _complete_bonds(working_mol)
    if len(working_bonds) < len(original_bonds):
        raise ValueError("The working copy removed an original bond")
    seen = set()
    for row, bond in enumerate(working_bonds):
        if id(bond.atom1) not in working_indices or id(bond.atom2) not in working_indices:
            raise ValueError("The working copy contains a bond to an external atom")
        first, second = _bond_endpoint_indices(bond, working_indices)
        key = tuple(sorted((first, second)))
        if first == second or key in seen:
            raise ValueError("The working copy contains an invalid duplicate bond")
        seen.add(key)
        if row >= len(original_bonds) and first < original_count and second < original_count:
            raise ValueError("The working copy changed the original bond topology")
    for row, (original, working) in enumerate(zip(original_bonds, working_bonds)):
        if (working.mol is not working_mol or working._uid_owner is not working_mol._owner_token
                or working._uid != row
                or _bond_endpoint_indices(original, original_indices)
                   != _bond_endpoint_indices(working, working_indices)
                or _bond_identity(original, original_indices) != _bond_identity(working, working_indices)):
            raise ValueError("The working copy changed the original bond topology or order")
    bond_bindings = original_bonds + (None,) * (len(working_bonds) - len(original_bonds))

    record, metadata = working_mol._capture_factual_record()
    original_record = original_mol._capture_factual_record()[0]
    # The retained Atom prefix gives both records the same canonical source rows.
    # Whole-copy deliberately drops generated history; accepted publication
    # restores that history ahead of the newly generated working-copy batches.
    previous_results = original_record.conformer_results
    if len(working_atoms) != original_count:
        previous_results = tuple(
            (payload, marker + (() if "frame_representation" in marker
                                else ("frame_representation",)))
            for payload, marker in previous_results)
    sources = (original_record.source_frames
               + record.source_frames[len(original_record.source_frames):])
    cursor, selected = (0, None) if record.frame_selection is None else record.frame_selection
    if cursor >= len(record.source_frames) and record.conformer_results:
        cursor += (len(sources) - len(record.source_frames)
                   + sum(len(payload.batch.coordinates) for payload, _ in previous_results))
    record = replace(record, source_frames=sources,
                     conformer_results=previous_results + record.conformer_results,
                     frame_selection=(cursor, selected) if sources or previous_results
                     or record.conformer_results or selected is not None else None)
    final_uids = tuple(atom._uid for atom in original_atoms) + tuple(range(
        original_mol._next_atom_uid,
        original_mol._next_atom_uid + len(working_atoms) - original_count,
    ))
    dependency_state = _remap_dependency_state(
        working_mol._atom_table, tuple(atom._uid for atom in working_atoms), final_uids,
    )
    return _WorkingCopyCommit(
        source, record, metadata,
        original_atoms + (None,) * (len(working_atoms) - original_count),
        tuple(bond_bindings), dependency_state, _get_stereo_payload(working_mol),
    )


def _snapshot_molecule_for_commit(mol: "Molecule") -> _MoleculeCommitSnapshot:
    # The installer swaps complete prepared owners. These old owners remain
    # task-local undo data and are never exposed as a second active store.
    members = tuple(mol._atoms) + _complete_bonds(mol)
    return _MoleculeCommitSnapshot(
        state=dict(mol.__dict__),
        member_state=tuple((member, dict(member.__dict__)) for member in members),
        conformer_state=dict(mol._conformers.__dict__),
        membership=tuple((values, tuple(values)) for values in (
            mol._atoms, mol._bonds,
        )),
        cache_state=(),
    )


def _restore_failed_commit(
    mol: "Molecule",
    snapshot: _MoleculeCommitSnapshot,
) -> None:
    atom_high_water, bond_high_water = mol._next_atom_uid, mol._next_bond_uid
    # Snapshot dicts already own their buckets; undo must not rebuild them.
    object.__setattr__(mol, "__dict__", snapshot.state)
    mol._next_atom_uid = max(mol._next_atom_uid, atom_high_water)
    mol._next_bond_uid = max(mol._next_bond_uid, bond_high_water)
    for members, values in snapshot.membership:
        members[:] = values
    for member, state in snapshot.member_state:
        object.__setattr__(member, "__dict__", state)
    for value, previous in snapshot.cache_state:
        value.clear()
        if isinstance(value, dict):
            value.update(previous)
        else:
            value.extend(previous)
    object.__setattr__(mol._conformers, "__dict__", snapshot.conformer_state)


def _commit_working_copy(
    original_mol: "Molecule",
    working_mol: "Molecule",
    *,
    source: tuple | None = None,
) -> None:
    """Install one accepted factual batch, retaining all original handles."""
    from hotpot.cheminfo.structure._commit import _FactTransaction, _commit_stereo, _commit_topology
    from hotpot.cheminfo.structure._fields import ATOM_SOURCE_CODES

    with original_mol._mutation_lock, working_mol._mutation_lock:
        if source is None:
            source = _capture_working_source(original_mol)
        _check_working_source(original_mol, source)
        payload = _prepare_working_copy_commit(original_mol, working_mol, source)
        _check_working_source(original_mol, source)
        snapshot = _snapshot_molecule_for_commit(original_mol)
        try:
            with _FactTransaction(original_mol, producer=ATOM_SOURCE_CODES["forcefield"]):
                _commit_topology(
                    original_mol, payload.source[0], payload.record,
                    atom_bindings=payload.atom_bindings, bond_bindings=payload.bond_bindings,
                    dependency_state=payload.dependency_state, bond_metadata=payload.bond_metadata,
                )
                _commit_stereo(original_mol, _capture_source(original_mol), payload.stereo_payload)
        except BaseException:
            _restore_failed_commit(original_mol, snapshot)
            raise
