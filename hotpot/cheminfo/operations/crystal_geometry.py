"""Read-only periodic geometry observations over captured chemical facts."""

import operator
import warnings

import numpy as np

from .crystal_generation import _arrays, _capture_input
from .crystal_matching import _positive_length
from ..rules.mathematics.geometry.linking import _polygon_linking
from ..structure._crystal import _image
from ..structure._fields import FIELD_INDEX


_XYZ = tuple(FIELD_INDEX[name] for name in ('x', 'y', 'z'))


def _cycle_input(record, cycle):
    cycle = tuple((operator.index(row), _image(image)) for row, image in cycle)
    if any(not 0 <= row < record.atom_count for row, _ in cycle):
        raise ValueError('CRYSTAL-IMAGE-MAP-INTEGRITY: cycle Atom row outside current membership')
    return cycle


def _directed_images(record):
    result = {}
    for (first, second), image in zip(record.bond_endpoints, record.cell[4]):
        result[first, second] = image
        result[second, first] = None if image is None else tuple(-value for value in image)
    return result


def _cycle_geometry(record, cycle, directed):
    """Check real edges and a zero-voltage lift; never invent a closing edge."""
    if len(cycle) < 3:
        return None, (('reason', 'fewer_than_three_vertices'),)
    for index, ((first, start), (second, end)) in enumerate(zip(cycle, cycle[1:] + cycle[:1])):
        edge = first, second
        subjects = (('edge', (cycle[index], cycle[(index + 1) % len(cycle)])),)
        if edge not in directed:
            return None, (('reason', 'open_cycle'), *subjects)
        expected = directed[edge]
        if expected is None:
            return None, (('reason', 'unknown_bond_image'), *subjects)
        supplied = tuple(last - initial for initial, last in zip(start, end))
        if supplied != expected:
            reason = 'nonzero_cycle_voltage' if index == len(cycle) - 1 else 'cycle_edge_image_mismatch'
            return None, (('reason', reason), *subjects, ('expected_image', expected), ('supplied_image', supplied))
    coordinates = _arrays(record)[0][:, _XYZ]
    return tuple(tuple(coordinates[row] + np.asarray(image) @ np.asarray(record.cell[1]))
                 for row, image in cycle), ()


def _link_cycles(record, first, second, tolerance):
    directed = _directed_images(record)
    polygons = []
    for index, cycle in enumerate((first, second)):
        coordinates, issue = _cycle_geometry(record, cycle, directed)
        if issue:
            return None, (('curve_index', index), *issue)
        polygons.append(coordinates)
    return _polygon_linking(*polygons, tolerance=tolerance)


def _emit_findings(findings):
    for code, subjects, measurements in findings:
        warnings.warn(f'{code}: subjects={subjects}; measurements={measurements}', RuntimeWarning, stacklevel=4)


def _linking_number(crystal, first_cycle, second_cycle, tolerance):
    tolerance = _positive_length(tolerance, 'tolerance')
    molecule = crystal._molecule
    if molecule is None:
        raise ValueError('CRYSTAL-UNSUPPORTED-PROJECTION: linking requires contained Molecule facts')
    with molecule._mutation_lock:
        record = _capture_input(crystal, molecule, require_group=False)
        first, second = (_cycle_input(record, cycle) for cycle in (first_cycle, second_cycle))
    value, measurements = _link_cycles(record, first, second, tolerance)
    if value is None:
        _emit_findings((('CRYSTAL-LINKING-UNAVAILABLE', (first, second), measurements),))
    elif value:
        _emit_findings((('CRYSTAL-CLOSED-LOOP-LINK', (first, second),
                         (('linking_number', value), *measurements)),))
    return value


def _shift_key(key, image):
    return key[0], key[1], tuple(a + b for a, b in zip(key[2], image))


def _pair_key(first, second, representatives):
    """One common translation and unordered-pair canonicalization, without wrapping sites."""
    left, right = (first[0], representatives[first[1]]), (second[0], representatives[second[1]])
    delta = tuple(b - a for a, b in zip(first[2], second[2]))
    forward, backward = (*left, *right, delta), (*right, *left, tuple(-value for value in delta))
    return min(forward, backward)


def _pair_subjects(first, second, representatives):
    token = _pair_key(first, second, representatives)
    operations = tuple(representatives)
    return ((token[0], operations[token[1]], (0, 0, 0)),
            (token[2], operations[token[3]], token[4]))


def _lift_rows(rows, directed):
    images = [(0, 0, 0)]
    for first, second in zip(rows[:-1], rows[1:]):
        image = directed[first, second]
        if image is None:
            return None
        images.append(tuple(a + b for a, b in zip(images[-1], image)))
    return tuple(zip(rows, images))


def _placed_geometry(record, cycles):
    from .crystal import _image_entry
    from .crystal_matching import _translation
    from ..rules.mathematics.cell import _to_fractional

    group, matrix = record.cell[2], np.asarray(record.cell[1])
    from ..rules.mathematics.symmetry import _IDENTITY
    source = _arrays(record)[0][:, _XYZ]
    fractional = _to_fractional(source, matrix)
    operations = group.symm_units
    placed = tuple(source + np.asarray(operation.t, dtype=float) @ matrix
                   if operation.R == _IDENTITY else operation.operate(fractional) @ matrix
                   for operation in operations)
    coordinates = np.stack(placed, axis=1).reshape(-1, 3)
    keys = tuple(_image_entry(row, operation) for row in range(record.atom_count) for operation in operations)
    bonds, rings, unavailable = [], [], []
    for edge, ((first, second), image) in enumerate(zip(record.bond_endpoints, record.cell[4])):
        if image is None:
            unavailable.append(('CRYSTAL-UNSUPPORTED-PROJECTION', (edge, first, second),
                                (('reason', 'unknown_bond_image'),)))
            continue
        for copy, operation in enumerate(operations):
            endpoint = operation @ _translation(group, image)
            bonds.append({'keys': (_image_entry(first, operation), _image_entry(second, endpoint)),
                          'coordinates': np.array((placed[copy][first],
                              placed[copy][second] + (np.asarray(operation.R) @ image) @ matrix))})
    directed = _directed_images(record)
    for rows in cycles:
        cycle = _lift_rows(rows, directed)
        points, issue = (None, (('reason', 'unknown_bond_image'),)) if cycle is None else _cycle_geometry(record, cycle, directed)
        if issue:
            unavailable.append(('CRYSTAL-LINKING-UNAVAILABLE', (tuple(rows),), issue))
            continue
        for copy, operation in enumerate(operations):
            rings.append({'keys': tuple(_image_entry(row, operation @ _translation(group, image)) for row, image in cycle),
                          'coordinates': np.array([placed[copy][row] + (np.asarray(operation.R) @ image) @ matrix
                                                   for row, image in cycle])})
    return coordinates, keys, tuple(bonds), tuple(rings), tuple(unavailable)


def _relation_context(candidates, units, placements, unavailable, group):
    from .crystal import _full_image, _image_entry
    from .crystal_matching import _translation

    representatives = {operation: index for index, operation in enumerate(group.symm_units)}
    members = {row: (unit['unit'], lift) for unit in units for row, lift in zip(unit['rows'], unit['lifts'])}
    reasons = {row: reason for code, rows, reason in unavailable
               if code == 'CRYSTAL-MOLECULAR-LIFT-UNAVAILABLE' for row in rows}
    placed_reasons = {(row, key[1]): placement['reason'] for placement in placements
                      if placement['reason'] is not None for row, key in zip(placement['rows'], placement['keys'])}

    def placement_key(key):
        if key[0] not in members:
            return None
        unit, lift = members[key[0]]
        _, operation = _full_image(key, group)
        return _image_entry(unit, operation @ _translation(group, tuple(-value for value in lift)))

    related = set()
    for _, pairs, _ in candidates:
        first, second = (placement_key(key) for key in pairs[0])
        related.add(_pair_key(first, second, representatives))

    unresolved = {}
    for code, pair, reason in unavailable:
        if code == 'CRYSTAL-UNSUPPORTED-PROJECTION':
            first, second = (placement_key(keys[0]) for keys in pair)
            unresolved[_pair_key(first, second, representatives)] = reason

    def relation(first, second):
        left, right = placement_key(first), placement_key(second)
        if left is not None and right is not None:
            signature = _pair_key(left, right, representatives)
            if signature in related:
                return 'complete_coincident_unit', None
            if left == right:
                return 'same_finite_unit', None
            if signature in unresolved:
                return 'unresolved', unresolved[signature]
        reason = (reasons.get(first[0]) or reasons.get(second[0])
                  or placed_reasons.get((first[0], first[1])) or placed_reasons.get((second[0], second[1])))
        return ('unresolved', reason) if reason is not None else ('no_complete_coincident_relation', None)
    return representatives, relation


def _contact_findings(record, coordinates, keys, bonds, representatives, relation, coincidence, cutoff):
    from ..rules.mathematics.periodic import _periodic_pairs

    bonded = {_pair_key(*bond['keys'], representatives) for bond in bonds}
    findings = []
    for first, second, image, distance in _periodic_pairs(coordinates, record.cell[1], max(coincidence, cutoff),
                                                        include_self_images=True):
        left, right = keys[first], _shift_key(keys[second], image)
        token = _pair_key(left, right, representatives)
        state, reason = relation(left, right)
        if state == 'complete_coincident_unit':
            continue
        subjects = _pair_subjects(left, right, representatives)
        measurements = (('distance', distance), ('unit', 'angstrom'), ('relation', state))
        if reason is not None:
            measurements += (('reason', reason),)
        if distance <= coincidence:
            findings.append(('CRYSTAL-COINCIDENT-IMAGES', subjects,
                             (*measurements, ('coincidence_tolerance', coincidence))))
        if distance < cutoff and token not in bonded:
            findings.append(('CRYSTAL-CROWDING', subjects, (*measurements, ('close_contact_cutoff', cutoff))))
    return tuple(findings)


def _bounds(item):
    points = item['coordinates']
    centre = points.mean(axis=0)
    return centre, float(np.linalg.norm(points - centre, axis=1).max())


def _nearby_objects(first, second, matrix, *, same=False, padding=0.):
    """Reuse one prepared periodic centre search; retain full common image shifts."""
    from ..rules.mathematics.periodic import _periodic_pairs

    if not first or not second:
        return ()
    items = first if same else (*first, *second)
    bounds = tuple(_bounds(item) for item in items)
    centres = np.array([centre for centre, _ in bounds])
    maximum = max(radius for _, radius in bounds)
    search = 2 * maximum + padding
    result = []
    for left, right, image, distance in _periodic_pairs(centres, matrix, search, include_self_images=same):
        if not same and not (left < len(first) <= right):
            continue
        if distance <= bounds[left][1] + bounds[right][1] + padding:
            result.append((left, right if same else right - len(first), image))
    return tuple(result)


def _translated(item, image, matrix):
    return {'keys': tuple(_shift_key(key, image) for key in item['keys']),
            'coordinates': item['coordinates'] + np.asarray(image) @ np.asarray(matrix)}


def _canonical_subjects(ring, other):
    anchor = ring['keys'][0][2]
    shift = tuple(-value for value in anchor)
    return (tuple(_shift_key(key, shift) for key in ring['keys']),
            tuple(_shift_key(key, shift) for key in other['keys']))


def _surface_measurements(value):
    from dataclasses import fields

    closest = value.closest_boundary_edge
    return (('state', value.state.value), ('features', tuple(sorted(item.value for item in value.features))),
            ('indeterminacy_causes', tuple(sorted(item.value for item in value.indeterminacy_causes))),
            ('surface_model', None if value.surface_model is None else value.surface_model.value),
            ('intersection_points', tuple(point.coordinates for point in value.intersection_points)),
            ('closest_boundary_edge', None if closest is None else (closest.edge_index, closest.distance)),
            ('surface_evidence', tuple((field.name, getattr(value.surface_evidence, field.name))
                                       for field in fields(value.surface_evidence))))


def _surface_findings(record, rings, bonds, representatives):
    from ..rules.mathematics.geometry.object import Cycle, Segment
    from ..rules.mathematics.geometry.relation import PiercingState, iter_segment_cycle_screenings
    from ..rules.mathematics.geometry.settings import DEFAULT_GEOMETRY_SETTINGS

    if not rings or not bonds:
        return ()
    settings = DEFAULT_GEOMETRY_SETTINGS.tolerance
    scale = max(float(np.max(np.abs(item['coordinates']))) for item in (*rings, *bonds))
    padding = settings.aabb_padding_factor * (settings.absolute_length + settings.relative_length * scale
              + settings.machine_epsilon_factor * np.finfo(float).eps * scale)
    grouped = {}
    for ring, bond, image in _nearby_objects(rings, bonds, record.cell[1], padding=padding):
        shifted = _translated(bonds[bond], image, record.cell[1])
        if set(rings[ring]['keys']).intersection(shifted['keys']):
            continue
        grouped.setdefault(ring, []).append(shifted)
    cycle_edges = {_pair_key(first, second, representatives)
                   for ring in rings for first, second in zip(ring['keys'], ring['keys'][1:] + ring['keys'][:1])}
    findings = []
    for index, segments in grouped.items():
        ring = rings[index]
        results = iter_segment_cycle_screenings((Segment(*item['coordinates']) for item in segments),
                                                Cycle(ring['coordinates']))
        for item, screening in zip(segments, results):
            if screening.state is PiercingState.DOES_NOT_PIERCE:
                continue
            measurements = _surface_measurements(screening.relation)
            subjects = _canonical_subjects(ring, item)
            findings.append(('CRYSTAL-PERIODIC-BOND-RING', subjects, measurements))
            if (screening.state is PiercingState.PIERCES
                    and _pair_key(*item['keys'], representatives) in cycle_edges):
                findings.append(('CRYSTAL-INTERPENETRATION', subjects, measurements))
    return tuple(findings)


def _loop_findings(record, rings, relation, tolerance):
    findings = []
    for first, second, image in _nearby_objects(rings, rings, record.cell[1], same=True, padding=tolerance):
        left, right = rings[first], _translated(rings[second], image, record.cell[1])
        if set(left['keys']).intersection(right['keys']):
            continue
        if relation(left['keys'][0], right['keys'][0])[0] == 'complete_coincident_unit':
            continue
        value, measurements = _polygon_linking(left['coordinates'], right['coordinates'], tolerance=tolerance)
        if value is None:
            findings.append(('CRYSTAL-LINKING-UNAVAILABLE', _canonical_subjects(left, right), measurements))
        elif value:
            findings.append(('CRYSTAL-CLOSED-LOOP-LINK', _canonical_subjects(left, right),
                             (('linking_number', value), *measurements)))
    return tuple(findings)


def _geometry_findings(crystal, coincidence_tolerance, close_contact_cutoff, ring_scope, max_ring_size):
    from .crystal_matching import _same_molecule_data
    from .topology_queries import _ring_cycles

    coincidence = _positive_length(coincidence_tolerance, 'coincidence_tolerance')
    cutoff = _positive_length(close_contact_cutoff, 'close_contact_cutoff')
    if ring_scope not in ('full_graph', 'ligand_skeleton'):
        raise ValueError('ring_scope must be full_graph or ligand_skeleton')
    molecule = crystal._molecule
    if molecule is None:
        raise ValueError('CRYSTAL-UNSUPPORTED-PROJECTION: geometry requires contained Molecule facts')
    with molecule._mutation_lock:
        candidates, unavailable, record, units, placements = _same_molecule_data(crystal, coincidence)
        cycles = _ring_cycles(molecule, ligand_skeleton=ring_scope == 'ligand_skeleton', max_size=max_ring_size)
    representatives, relation = _relation_context(candidates, units, placements, unavailable, record.cell[2])
    coordinates, keys, bonds, rings, limitations = _placed_geometry(record, cycles)
    findings = [('CRYSTAL-POSSIBLE-SAME-MOLECULE', pairs,
                 (('operation', operation), ('maximum_distance', distance), ('unit', 'angstrom')))
                for operation, pairs, distance in candidates]
    findings.extend((code, subjects, (('reason', reason),)) for code, subjects, reason in unavailable)
    findings.extend(limitations)
    findings.extend(_contact_findings(record, coordinates, keys, bonds, representatives, relation, coincidence, cutoff))
    findings.extend(_surface_findings(record, rings, bonds, representatives))
    findings.extend(_loop_findings(record, rings, relation, coincidence))
    findings = tuple(dict.fromkeys(findings))
    _emit_findings(findings)
    return findings
