"""Named topology transformations over the sole carrier primitives."""

import numpy as np

from ..structure._commit import _FactTransaction, _commit_topology
from ..structure._dependencies import _dependency_state
from ..structure._fields import ATOM_SOURCE_CODES as _ATOM_SOURCE_CODES
from ..structure._identity import _capture_source
from ..structure.conformer import _capture_frame_records


__all__ = ()


def _capture_component(component):
    with component._mutation_lock:
        incoming, metadata = component._capture_factual_record(include_frames=False)
        frames = _capture_frame_records(component, include_generated=False)
        dependencies = _dependency_state(component._atom_table)
        uids = tuple(atom._uid for atom in component._atoms)
        return incoming, metadata, frames, dependencies, uids


def _add_component(mol, component):
    return _merge_component(mol, _capture_component(component))


def _merge_component(mol, captured):
    from dataclasses import replace
    from .chemistry import _remap_stereo_transport
    from ..structure.conformer import _remap_observations
    incoming, incoming_metadata, incoming_frames, incoming_dependencies, incoming_uids = captured
    with _FactTransaction(mol, producer=_ATOM_SOURCE_CODES['explicit']) as transaction:
        transaction.touch_topology()
        current, current_metadata = mol._capture_factual_record()
        offset = current.atom_count
        incoming_sources, _, incoming_selection = incoming_frames
        if offset:
            mapping = {row: offset + row for row in range(incoming.atom_count)}
            observations = {}
            shifted = []
            for frame in incoming_sources:
                if frame.observations is not None and id(frame.observations) not in observations:
                    observations[id(frame.observations)] = _remap_observations(frame.observations, mapping)
                shifted.append(replace(frame, atom_uids=tuple(mapping[uid] for uid in frame.atom_uids),
                    observations=observations.get(id(frame.observations))))
            incoming_sources = tuple(shifted)
        stale_ensembles = tuple((payload,
            marker + (() if 'frame_representation' in marker else ('frame_representation',)))
            for payload, marker in current.conformer_results)
        from ..structure._crystal import _merge_crystal_records
        cell = _merge_crystal_records(current.cell, incoming.cell, offset, incoming.atom_count)
        charge = incoming.charge if not offset else (
            (True, current.charge[1] + incoming.charge[1], 'component_merge')
            if current.charge[0] and incoming.charge[0] else (False, None, None))
        spin = incoming.spin_mult if not offset else (False, None, None)
        payload = incoming.stereo_payload
        if offset:
            if payload is not None:
                payload = _remap_stereo_transport(payload,
                            {row: offset + row for row in range(incoming.atom_count)})
            if payload is None:
                payload = current.stereo_payload
            elif current.stereo_payload is not None:
                payload = type(payload)(current.stereo_payload.schema_version,
                                        current.stereo_payload.records + payload.records,
                                        current.stereo_payload.atom_facts + payload.atom_facts,
                                        current.stereo_payload.bond_facts + payload.bond_facts)
        record = replace(current, atom_count=offset + incoming.atom_count,
            atom_values=current.atom_values + incoming.atom_values,
            atom_presence=current.atom_presence + incoming.atom_presence,
            atom_sources=current.atom_sources + incoming.atom_sources,
            atom_frac_valences=current.atom_frac_valences + tuple((offset + row, value) for row, value in incoming.atom_frac_valences),
            bond_endpoints=current.bond_endpoints + tuple((offset + a, offset + b) for a,b in incoming.bond_endpoints),
            bond_values=current.bond_values + incoming.bond_values,
            bond_kinds=current.bond_kinds + incoming.bond_kinds,
            bond_aromatic=current.bond_aromatic + incoming.bond_aromatic,
            bond_directions=current.bond_directions + incoming.bond_directions,
            bond_sources=current.bond_sources + incoming.bond_sources,
            bond_metadata=current.bond_metadata + incoming.bond_metadata,
            charge=charge, spin_mult=spin, cell=cell,
            partial_charge_result=(incoming.partial_charge_result if not offset else
                _merged_partial_charge_record(current, incoming.atom_count)),
            stereo_payload=payload,
            source_frames=current.source_frames + incoming_sources,
            conformer_results=stale_ensembles,
            frame_selection=(0, None) if offset else incoming_selection,
            angle_constraints=current.angle_constraints + tuple(tuple(offset + row for row in item) for item in incoming.angle_constraints),
            torsion_constraints=current.torsion_constraints + tuple(tuple(offset + row for row in item) for item in incoming.torsion_constraints),
            pair_wiberg_bond_orders=current.pair_wiberg_bond_orders + tuple(
                (offset + first, offset + second, value)
                for first, second, value in incoming.pair_wiberg_bond_orders))
        old_atoms = tuple(mol._atoms)
        old_bonds = tuple(mol._bonds)
        source_status, source_seeds = incoming_dependencies
        uid_map = dict(zip(incoming_uids, range(mol._next_atom_uid, mol._next_atom_uid + incoming.atom_count)))
        target_status, target_seeds = _dependency_state(mol._atom_table)
        dependencies = (np.vstack((target_status, source_status)),
            tuple(frozenset(target) | frozenset(uid_map[uid] for uid in source if uid in uid_map)
                  for target, source in zip(target_seeds, source_seeds)))
        # Bindings follow the record's concatenated complete Bond order.
        _commit_topology(mol, _capture_source(mol), record,
            atom_bindings=old_atoms + (None,) * incoming.atom_count,
            bond_bindings=old_bonds + (None,) * len(incoming.bond_endpoints),
            dependency_state=dependencies, bond_metadata=current_metadata + incoming_metadata)


def _merged_partial_charge_record(current, added_count):
    from dataclasses import replace
    if current.partial_charge_result is None:
        return None
    payload, marker = current.partial_charge_result
    if "charge_representation" not in marker:
        marker += ("charge_representation",)
    return (replace(payload, atom_uids=tuple(range(current.atom_count + added_count)),
                    values=payload.values + (None,) * added_count,
                    provenance=payload.provenance + (_ATOM_SOURCE_CODES["default"],) * added_count), marker)


def _replace_atom(mol, original_atom_idx, new_atom):
    original = mol._replace_atom(original_atom_idx, new_atom)
    mol._update_topology()
    mol.calc_implicit_hydrogens()
    return original


def _link_atoms(mol, assign_bond_order):
    from .chemistry import _link_atoms as perceive_links
    with _FactTransaction(mol, producer=_ATOM_SOURCE_CODES['explicit']) as transaction:
        transaction.touch_topology()
        transaction.touch_dependencies()
        perceive_links(mol)
        mol._update_topology(clear_conformers=False)
        if assign_bond_order:
            mol.assign_bond_order()


def _link_atom(atom, other):
    assert atom.mol is not other.mol, (
        "The `link_with` method should only implement in two atoms from different molecules, "
        "this atom and other atom come from a same molecule")
    with other.mol._mutation_lock:
        other_row = other.idx
        captured = _capture_component(other.mol)
    mol = atom.mol
    with _FactTransaction(mol, producer=_ATOM_SOURCE_CODES['explicit']) as transaction:
        transaction.touch_topology()
        transaction.touch_dependencies()
        other_clone_idx = other_row + len(mol.atoms)
        _merge_component(mol, captured)
        mol.add_bond(atom, mol.atoms[other_clone_idx])


def _add_atom(atom, new_atom, bond_order, atom_attrs, bond_attrs):
    added = atom._add_atom(new_atom, bond_order, atom_attrs, bond_attrs)
    atom.mol._update_topology()
    return added


def _remove_hydrogens(mol, *, polar_only=False):
    from dataclasses import replace
    from .chemistry import _set_stereo_payload
    from ..structure._stereo import _get_stereo_payload

    with _FactTransaction(mol, producer=_ATOM_SOURCE_CODES['explicit']):
        selected = tuple(atom for atom in mol._atoms
                         if (atom.is_polar_hydrogen if polar_only else atom.is_hydrogen))
        affected = {neighbour._uid for atom in selected for neighbour in atom.neighbours}
        mol.remove_atoms(selected)
        payload = _get_stereo_payload(mol)
        if payload is None:
            return
        facts = tuple(
            replace(fact, current_hydrogens=replace(
                fact.current_hydrogens, implicit_hydrogens=None))
            if (0 <= fact.atom < len(mol._atoms)
                and mol._atoms[fact.atom]._uid in affected
                and fact.current_hydrogens is not None) else fact
            for fact in payload.atom_facts
        )
        if facts != payload.atom_facts:
            _set_stereo_payload(mol, replace(payload, atom_facts=facts))
