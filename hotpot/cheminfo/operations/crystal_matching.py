"""Read-only relations between complete, already placed molecular units."""

from dataclasses import replace
import warnings

import numpy as np

from .crystal import _full_image, _image_entry, _transform_crystal_record
from .crystal_generation import _arrays, _capture_input, _source_descriptors
from ..adapters.native.substructure.matching import _NativeTarget
from ..adapters.structure.crystal import _pack_matching_query, _pack_matching_target
from ..rules.chemistry.crystal_matching import (
    _complete_mapping, _coordinate_domains, _matching_unavailable, _pair_record,
)
from ..rules.mathematics import symmetry as affine
from ..rules.mathematics.cell import _to_fractional
from ..rules.mathematics.periodic import _finite_component_lifts, _lattice_images, _periodic_pairs, _wrap_fractional
from ..structure._builder import _freeze_value
from ..structure._fields import ATOM_SOURCE_CODES, FIELD_INDEX
from ..structure.symmetry import SymmUnit


_XYZ = tuple(FIELD_INDEX[name] for name in ('x', 'y', 'z'))


def _positive_length(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f'CRYSTAL-CONTEXT-INTEGRITY: {name} requires a positive finite Angstrom value')
    return value


def _translation(group, image):
    return SymmUnit.from_affine(affine._IDENTITY, image, context=group)


def _finite_units(crystal):
    molecule = crystal._molecule
    if molecule is None:
        raise ValueError('CRYSTAL-UNSUPPORTED-PROJECTION: relation query requires contained Molecule facts')
    with molecule._mutation_lock:
        record = _capture_input(crystal, molecule)
        units, unavailable = [], []
        for unit, (rows, lifts, unknown, inconsistent) in enumerate(_finite_component_lifts(
                record.atom_count, record.bond_endpoints, record.cell[4])):
            if lifts is None:
                unavailable.append((rows, f'unknown_edges={unknown}, nonzero_cycle_edges={inconsistent}'))
                continue
            partial, metadata = molecule._capture_factual_record(rows=rows, include_frames=False)
            values, presence, sources = _arrays(partial)
            values, sources = values.copy(), sources.copy()
            values[:, _XYZ] += np.asarray(lifts, dtype=float) @ np.asarray(record.cell[1])
            for row, image in enumerate(lifts):
                if any(image):
                    sources[row, _XYZ] = ATOM_SOURCE_CODES['geometry']
            context = (record.cell[0], record.cell[1], record.cell[2],
                       partial.cell[3], ((0, 0, 0),) * len(partial.bond_endpoints),
                       'independent', None, None, None)
            partial = replace(partial, atom_values=values.tobytes(), atom_sources=sources.tobytes(),
                cell=context, bond_metadata=tuple(_freeze_value(item) for item in metadata),
                partial_charge_result=None, source_frames=(), conformer_results=(),
                frame_selection=None, pair_wiberg_bond_orders=())
            units.append({'unit': unit, 'rows': rows, 'lifts': lifts, 'record': partial,
                          'coordinates': values[:, _XYZ].copy()})
    return record, tuple(units), tuple(unavailable)


def _warn_unavailable(entries, code='CRYSTAL-MOLECULAR-LIFT-UNAVAILABLE'):
    for subjects, reason in entries:
        warnings.warn(f'{code}: subjects={subjects}; {reason}', RuntimeWarning, stacklevel=4)


def _place_unit(unit, operation, *, anchor=True):
    record = unit['record']
    group, matrix = record.cell[2], record.cell[1]
    fractional = _to_fractional(unit['coordinates'], matrix)
    if anchor:
        shift = _wrap_fractional(operation.operate(fractional[:1]))[1][0]
        operation = _translation(group, tuple(-value for value in shift)) @ operation
    coordinates = operation.operate(fractional) @ np.asarray(matrix)
    reason = _matching_unavailable(record)
    transformed = None
    if reason is None:
        try:
            transformed = _transform_crystal_record(record, operation)
        except ValueError as exc:
            # This exact local action limitation is already a supported contract.
            if not str(exc).startswith('CRYSTAL-UNSUPPORTED-PROJECTION:'):
                raise
            reason = str(exc)
    keys = tuple(_image_entry(row, operation @ _translation(group, lift))
                 for row, lift in zip(unit['rows'], unit['lifts']))
    return {'unit': unit['unit'], 'rows': unit['rows'], 'action': operation,
            'keys': keys, 'coordinates': coordinates, 'record': transformed, 'reason': reason}


def _shift_placement(placement, image, matrix):
    if not any(image):
        return placement
    translation = _translation(placement['action'].context, image)
    action = translation @ placement['action']
    coordinates = placement['coordinates'] + np.asarray(image) @ np.asarray(matrix)
    record = placement['record']
    if record is not None:
        values, _, sources = _arrays(record)
        values, sources = values.copy(), sources.copy()
        values[:, _XYZ] = coordinates
        sources[:, _XYZ] = ATOM_SOURCE_CODES['geometry']
        record = replace(record, atom_values=values.tobytes(), atom_sources=sources.tobytes())
    keys = tuple(_image_entry(row, translation @ member)
                 for row, member in (_full_image(key, action.context) for key in placement['keys']))
    return {**placement, 'action': action, 'keys': keys, 'coordinates': coordinates, 'record': record}


def _overlap_images(first, second, matrix, tolerance):
    left, right = first['coordinates'], second['coordinates']
    left_centre, right_centre = left.mean(axis=0), right.mean(axis=0)
    radius = (np.linalg.norm(left - left_centre, axis=1).max()
              + np.linalg.norm(right - right_centre, axis=1).max() + tolerance)
    delta = _to_fractional(right_centre - left_centre, matrix)
    return _lattice_images(delta, matrix, radius)


def _placed_matches(first, second, tolerance, target):
    if first['reason'] is not None or second['reason'] is not None:
        return (), first['reason'] or second['reason']
    left, right = first['record'], second['record']
    if left.atom_count != right.atom_count or len(left.bond_endpoints) != len(right.bond_endpoints):
        return (), None
    domains = _coordinate_domains(left, right, tolerance)
    if any(not domain for domain in domains):
        return (), None
    record = _pair_record(left, right)
    matches, unresolved = [], None
    for mapping in target.iter_mappings(_pack_matching_query(left, domains)):
        valid, reason = _complete_mapping(record, mapping, left.atom_count, tolerance)
        if valid:
            distances = np.linalg.norm(first['coordinates'] - second['coordinates'][
                [mapping[row] for row in range(left.atom_count)]], axis=1)
            matches.append((tuple(mapping[row] for row in range(left.atom_count)), float(distances.max())))
        elif valid is None:
            unresolved = reason
    return tuple(matches), unresolved


def _same_molecule_data(crystal, tolerance):
    record, units, unavailable = _finite_units(crystal)
    group, matrix = record.cell[2], record.cell[1]
    placements = tuple(_place_unit(unit, operation) for unit in units for operation in group.symm_units)
    candidates, unresolved = [], [('CRYSTAL-MOLECULAR-LIFT-UNAVAILABLE', rows, reason)
                                      for rows, reason in unavailable]
    targets = {}
    if placements:
        centres = np.array([item['coordinates'].mean(axis=0) for item in placements])
        radii = tuple(float(np.linalg.norm(item['coordinates'] - centre, axis=1).max())
                      for item, centre in zip(placements, centres))
        neighbors = _periodic_pairs(centres, matrix, 2 * max(radii) + tolerance)
    else:
        neighbors = ()
    for first_index, second_index, image, distance in neighbors:
        first, second = placements[first_index], placements[second_index]
        if (len(first['rows']) != len(second['rows'])
                or distance > radii[first_index] + radii[second_index] + tolerance):
            continue
        if second['record'] is not None and second_index not in targets:
            targets[second_index] = _NativeTarget(_pack_matching_target(second['record']))
        shifted = _shift_placement(second, image, matrix)
        matches, reason = _placed_matches(first, shifted, tolerance, targets.get(second_index))
        if reason is not None:
            unresolved.append(('CRYSTAL-UNSUPPORTED-PROJECTION', (first['keys'], shifted['keys']), reason))
        for mapping, distance in matches:
            pairs = tuple((first['keys'][row], shifted['keys'][target]) for row, target in enumerate(mapping))
            candidates.append((~shifted['action'] @ first['action'], pairs, distance))
    return tuple(candidates), tuple(dict.fromkeys(unresolved)), record, units, placements


def _same_molecule_candidates(crystal, tolerance):
    tolerance = _positive_length(tolerance, 'tolerance')
    candidates, unresolved, *_ = _same_molecule_data(crystal, tolerance)
    for code, subjects, reason in unresolved:
        _warn_unavailable(((subjects, reason),), code)
    return candidates


def _explicit_unit(crystal, molecule, current):
    other = molecule.crystal
    if other is None:
        raise ValueError('CRYSTAL-UNSUPPORTED-PROJECTION: selected finite unit needs an explicit cell frame')
    selected, units, unavailable = _finite_units(other)
    if unavailable or len(units) != 1 or len(units[0]['rows']) != selected.atom_count:
        raise ValueError('CRYSTAL-MOLECULAR-LIFT-UNAVAILABLE: selected content is not one complete finite unit')
    if selected.cell[1] != current.cell[1]:
        raise ValueError('CRYSTAL-IMAGE-MAP-INTEGRITY: selected unit uses a different active cell frame')
    if selected.cell[2] != current.cell[2]:
        origin = selected.cell[6]
        sites, bonds = _source_descriptors(current)
        if (origin is None or origin[:4] != (current.cell[1], current.cell[2], sites, bonds)
                or origin[4] != affine._IDENTITY):
            raise ValueError('CRYSTAL-IMAGE-MAP-INTEGRITY: selected generated unit has no current source-frame binding')
        origins = selected.cell[7]
        if origins is None or any(entry is None for entry in origins):
            raise ValueError('CRYSTAL-IMAGE-MAP-INTEGRITY: selected generated unit has unknown required origins')
        current_xyz = _arrays(current)[0][:, _XYZ]
        current_fractional = _to_fractional(current_xyz, current.cell[1])
        selected_xyz = _arrays(selected)[0][:, _XYZ]
        for row, keys in enumerate(origins):
            for source_row, operation in (_full_image(key, current.cell[2]) for key in keys):
                point = operation.operate(current_fractional[source_row]) @ np.asarray(current.cell[1])
                if np.linalg.norm(point - selected_xyz[row]) > 1e-10:
                    raise ValueError('CRYSTAL-IMAGE-MAP-INTEGRITY: selected generated unit no longer maps current source coordinates')
    unit = units[0]
    local = unit['record']
    # The caller's explicit association/binding supplies this active group frame.
    return {**unit, 'record': replace(local, cell=(*local.cell[:2], current.cell[2], *local.cell[3:]))}


def _molecular_symmetry(crystal, molecule, tolerance):
    tolerance = _positive_length(tolerance, 'tolerance')
    current, units, unavailable = _finite_units(crystal)
    if molecule is None or molecule is crystal._molecule:
        if unavailable or len(units) != 1 or len(units[0]['rows']) != current.atom_count:
            raise ValueError('CRYSTAL-MOLECULAR-LIFT-UNAVAILABLE: choose one complete finite molecular unit')
        unit = units[0]
    else:
        unit = _explicit_unit(crystal, molecule, current)
    group, matrix = current.cell[2], current.cell[1]
    original = _place_unit(unit, group.identity(), anchor=False)
    if original['reason'] is not None:
        _warn_unavailable(((original['keys'], original['reason']),), 'CRYSTAL-UNSUPPORTED-PROJECTION')
        return ()
    target = _NativeTarget(_pack_matching_target(original['record']))
    operations, unresolved = [], []
    for representative in group.symm_units:
        placed = _place_unit(unit, representative, anchor=False)
        for image in _overlap_images(original, placed, matrix, tolerance):
            shifted = _shift_placement(placed, image, matrix)
            operation = shifted['action']
            if operation.R == affine._IDENTITY and any(operation.t):
                continue
            matches, reason = _placed_matches(original, shifted, tolerance, target)
            if matches:
                operations.append(operation)
            elif reason is not None:
                unresolved.append((shifted['keys'], reason))
    operations = tuple(dict.fromkeys(operations))
    values = set(operations)
    if (group.identity() not in values or any(~operation not in values for operation in operations)
            or any(left @ right not in values for left in operations for right in operations)):
        warnings.warn('CRYSTAL-STABILIZER-NONCLOSURE: matched operations do not form a complete subgroup',
                      RuntimeWarning, stacklevel=3)
    _warn_unavailable(unresolved, 'CRYSTAL-UNSUPPORTED-PROJECTION')
    return operations
