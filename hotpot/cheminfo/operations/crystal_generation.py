"""Whole Crystal projections over captured facts and exact image arithmetic."""

from dataclasses import replace
from itertools import product
from math import prod
import operator
import warnings

import numpy as np

from .crystal import (
    _coordinates, _full_image, _image_entry, _transform_crystal_record,
)
from ..rules.mathematics import symmetry as affine
from ..rules.mathematics.cell import _to_cartesian, _to_fractional
from ..rules.mathematics.periodic import (
    _finite_component_lifts, _image_after_wrap, _supercell_edges, _wrap_fractional,
)
from ..rules.chemistry.stereochemistry.algebra import _remap_stereo_payload
from ..structure._builder import _freeze_value
from ..structure._fields import (
    ATOM_FIELDS, ATOM_SOURCE_CODES, DERIVED_FIELDS, FIELD_INDEX, FIELD_PRESENCE_BITS,
)
from ..structure.molecule import Molecule
from ..structure.stereo import StereoPayload
from ..structure.symmetry import SpaceGroup, SymmUnit


_XYZ = tuple(FIELD_INDEX[name] for name in ('x', 'y', 'z'))


def _arrays(record):
    return (np.frombuffer(record.atom_values, dtype='<f8').reshape(-1, len(ATOM_FIELDS)),
            np.frombuffer(record.atom_presence, dtype='<u4'),
            np.frombuffer(record.atom_sources, dtype='u1').reshape(-1, len(ATOM_FIELDS)))


def _capture_input(crystal, molecule, *, require_group=True):
    if molecule is None:
        raise ValueError('CRYSTAL-UNSUPPORTED-PROJECTION: generation requires contained Molecule facts')
    if crystal._molecule is not molecule or molecule._crystal is not crystal:
        raise RuntimeError('CRYSTAL-CONTEXT-INTEGRITY: active Crystal association changed')
    _coordinates(molecule)
    record, metadata = molecule._capture_factual_record(include_frames=False)
    if require_group and record.cell[2] is None:
        raise ValueError('CRYSTAL-CONTEXT-INTEGRITY: cell projection requires a known active group')
    return replace(record, bond_metadata=tuple(_freeze_value(item) for item in metadata))


def _warn_omitted_results(molecule, record):
    omitted = []
    if molecule._mca_result is not None:
        omitted.append('MCA')
    if record.partial_charge_result is not None:
        omitted.append('partial charge')
    if molecule._source_frames or molecule._conformer_results or molecule._frame_result is not None:
        omitted.append('frame/calculation')
    if record.pair_wiberg_bond_orders:
        omitted.append('Wiberg')
    if omitted:
        warnings.warn('CRYSTAL-UNSUPPORTED-PROJECTION: replication omits live '
                      + ', '.join(omitted) + ' results; original observations remain at the source',
                      RuntimeWarning, stacklevel=3)


def _translate_origin(origin, entry, image):
    if entry is None or not any(image):
        return entry
    frame = origin[4]
    if frame is None:
        raise ValueError('CRYSTAL-UNSUPPORTED-PROJECTION: exact active-to-source basis map is unavailable')
    translation = SymmUnit.from_affine(affine._IDENTITY, affine._mv(frame, image), context=origin[1])
    return tuple(_image_entry(row, translation @ member)
                 for row, member in (_full_image(key, origin[1]) for key in entry))


def _wrap_record(record):
    values, presence, sources = _arrays(record)
    fractional, shifts = _wrap_fractional(_to_fractional(values[:, _XYZ], record.cell[1]))
    coordinates = _to_cartesian(fractional, record.cell[1])
    output, codes = values.copy(), sources.copy()
    changed = values[:, _XYZ] != coordinates
    output[:, _XYZ] = coordinates
    for axis, column in enumerate(_XYZ):
        codes[changed[:, axis], column] = ATOM_SOURCE_CODES['geometry']
    cell = record.cell
    images = tuple(None if image is None else _image_after_wrap(
        affine._IDENTITY, image, shifts[first], shifts[second])
        for (first, second), image in zip(record.bond_endpoints, cell[4]))
    origin, atom_origins, bond_origins = cell[6:]
    if origin is not None:
        atom_origins = tuple(_translate_origin(origin, entry, tuple(-value for value in shift))
                             for entry, shift in zip(atom_origins, shifts))
        bond_origins = tuple(_translate_origin(origin, entry,
            tuple(-value for value in shifts[first]))
            for entry, (first, _) in zip(bond_origins, record.bond_endpoints))
    destination = (*cell[:4], images, cell[5], origin, atom_origins, bond_origins)
    return replace(record, atom_values=output.tobytes(), atom_sources=codes.tobytes(), cell=destination), shifts


def _source_descriptors(record):
    values, presence, sources = _arrays(record)
    fractional = dict(record.atom_frac_valences)
    payload = record.stereo_payload
    atom_facts = [[] for _ in range(record.atom_count)]
    if payload is not None:
        for fact in payload.atom_facts:
            def hydrogen(value):
                return None if value is None else (
                    value.implicit_hydrogens, value.bracket_explicit_hydrogens, value.no_implicit)
            atom_facts[fact.atom].append((fact.isotope, fact.radical_electrons,
                hydrogen(fact.current_hydrogens), hydrogen(fact.source_hydrogens), fact.source_namespace))
    columns = tuple(index for index, name in enumerate(ATOM_FIELDS)
                    if index not in _XYZ and name not in DERIVED_FIELDS)
    sites = tuple((tuple((ATOM_FIELDS[column], float(values[row, column]), int(sources[row, column]))
                          for column in columns if int(presence[row]) & FIELD_PRESENCE_BITS[column]),
                   fractional.get(row), tuple(atom_facts[row])) for row in range(record.atom_count))
    bond_values = np.frombuffer(record.bond_values, dtype='<f8').reshape(-1, 3)
    bonds = tuple((endpoints, tuple(map(float, raw)), kind, aromatic, direction, source)
                  for endpoints, raw, kind, aromatic, direction, source in zip(
                      record.bond_endpoints, bond_values, record.bond_kinds,
                      record.bond_aromatic, record.bond_directions, record.bond_sources))
    return sites, bonds


def _concat_stereo(payloads):
    present = tuple(payload for payload in payloads if payload is not None)
    if not present:
        return None
    return StereoPayload(present[0].schema_version,
        tuple(record for payload in present for record in payload.records),
        tuple(fact for payload in present for fact in payload.atom_facts),
        tuple(fact for payload in present for fact in payload.bond_facts))


def _declared_classes(record, operations):
    count = len(operations)
    parent = list(range(record.atom_count * count))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def identify(first, second):
        first, second = root(first), root(second)
        parent[max(first, second)] = min(first, second)

    group = record.cell[2]
    indices = {operation: index for index, operation in enumerate(operations)}
    coordinates = _to_fractional(_arrays(record)[0][:, _XYZ], record.cell[1])
    for entry in record.cell[3][0]:
        members = tuple(_full_image(key, group) for key in entry)
        placed = np.array([operation.operate(coordinates[row]) for row, operation in members])
        differences = _to_cartesian(placed - placed[0], record.cell[1])
        if np.any(np.linalg.norm(differences, axis=1) > 1e-10):
            raise ValueError('CRYSTAL-IMAGE-MAP-INTEGRITY: declared identified images do not coincide')
        for action in operations:
            nodes = tuple(row * count + indices[(action @ operation).reduce_to_cell()[0]]
                          for row, operation in members)
            for node in nodes[1:]:
                identify(nodes[0], node)
    classes = {}
    for node in range(len(parent)):
        classes.setdefault(root(node), []).append(node)
    return tuple(map(tuple, classes.values()))


def _cell_projection(record):
    cell = record.cell
    operations = cell[2].symm_units
    count = len(operations)
    projected, shifts = zip(*(_wrap_record(_transform_crystal_record(record, operation))
                             for operation in operations))
    arrays = tuple(_arrays(item) for item in projected)
    values = np.stack([item[0] for item in arrays], axis=1).reshape(-1, len(ATOM_FIELDS))
    presence = np.stack([item[1] for item in arrays], axis=1).reshape(-1)
    sources = np.stack([item[2] for item in arrays], axis=1).reshape(-1, len(ATOM_FIELDS))
    endpoints = tuple((first * count + copy, second * count + copy)
                     for first, second in record.bond_endpoints for copy in range(count))
    images = tuple(item.cell[4][edge] for edge in range(len(record.bond_endpoints)) for item in projected)
    if cell[5] == 'independent':
        sites, bonds = _source_descriptors(record)
        origin = (cell[1], cell[2], sites, bonds, affine._IDENTITY)
        atom_origins = tuple((_image_entry(row, SymmUnit.from_affine(operation.R,
            tuple(value - shift for value, shift in zip(operation.t, shifts[copy][row])), context=cell[2])),)
            for row in range(record.atom_count) for copy, operation in enumerate(operations))
        bond_origins = tuple((_image_entry(edge, SymmUnit.from_affine(operation.R,
            tuple(value - shift for value, shift in zip(operation.t, shifts[copy][first])), context=cell[2])),)
            for edge, (first, _) in enumerate(record.bond_endpoints)
            for copy, operation in enumerate(operations))
    else:
        origin = cell[6]
        atom_origins = tuple(item.cell[7][row] for row in range(record.atom_count) for item in projected)
        bond_origins = tuple(item.cell[8][edge] for edge in range(len(record.bond_endpoints)) for item in projected)
    relations = ((), tuple(value for value in cell[3][1] for _ in operations),
                 tuple(value for value in cell[3][2] for _ in operations))
    destination_cell = (cell[0], cell[1], SpaceGroup(1), relations, images,
                        'cell' if cell[5] == 'independent' else cell[5], origin, atom_origins, bond_origins)
    stereo = _concat_stereo(None if item.stereo_payload is None else
        _remap_stereo_payload(item.stereo_payload, {row: row * count + copy for row in range(record.atom_count)})
        for copy, item in enumerate(projected))
    def replicated(entries):
        return tuple(tuple(row * count + copy for row in entry) for entry in entries for copy in range(count))
    classes = _declared_classes(record, operations)
    targets = {node: target for target, entry in enumerate(classes) for node in entry}
    identity = operations.index(cell[2].identity())
    single_whole = (len(classes) == record.atom_count and len({
        targets[row * count + identity] for row in range(record.atom_count)}) == record.atom_count)
    charge = ((False, None, None) if single_whole else
              (True, record.charge[1] * count, 'crystal_generation') if record.charge[0] and count > 1 else record.charge)
    result = replace(record, atom_count=record.atom_count * count,
        atom_values=values.tobytes(), atom_presence=presence.tobytes(), atom_sources=sources.tobytes(),
        atom_frac_valences=tuple((row * count + copy, value) for row, value in record.atom_frac_valences for copy in range(count)),
        bond_endpoints=endpoints, bond_values=b''.join(record.bond_values[edge * 24:(edge + 1) * 24]
            for edge in range(len(record.bond_endpoints)) for _ in operations),
        bond_kinds=tuple(value for value in record.bond_kinds for _ in operations),
        bond_directions=tuple(value for value in record.bond_directions for _ in operations),
        bond_sources=tuple(value for value in record.bond_sources for _ in operations),
        bond_metadata=tuple(value for value in record.bond_metadata for _ in operations),
        bond_aromatic=tuple(value for value in record.bond_aromatic for _ in operations),
        charge=charge, spin_mult=record.spin_mult if count == 1 else (False, None, None),
        cell=destination_cell, stereo_payload=stereo, partial_charge_result=None,
        source_frames=(), conformer_results=(), frame_selection=None,
        angle_constraints=replicated(record.angle_constraints), torsion_constraints=replicated(record.torsion_constraints),
        pair_wiberg_bond_orders=())
    if any(len(entry) > 1 for entry in classes):
        from ..rules.chemistry.crystal_identification import _identify_generated_record
        identified = _identify_generated_record(result, classes)
    else:
        identified = result
    return replace(identified, charge=record.charge, spin_mult=record.spin_mult) if single_whole else identified


def _unit_cell_crystal(crystal):
    molecule = crystal._molecule
    if molecule is None:
        raise ValueError('CRYSTAL-UNSUPPORTED-PROJECTION: cell projection requires contained Molecule facts')
    with molecule._mutation_lock:
        record = _capture_input(crystal, molecule)
        projected = _cell_projection(record)
        result = Molecule._build_molecule(projected)
        result.properties = dict(molecule.properties)
        _warn_omitted_results(molecule, record)
        return result.crystal


def _packet_lifts(record, rows):
    rows = tuple(dict.fromkeys(rows))
    mapping = {row: index for index, row in enumerate(rows)}
    edges, images = [], []
    for endpoints, image in zip(record.bond_endpoints, record.cell[4]):
        if all(row in mapping for row in endpoints):
            edges.append(tuple(mapping[row] for row in endpoints))
            images.append(image)
    components = _finite_component_lifts(len(rows), edges, images)
    if len(components) != 1 or components[0][1] is None:
        raise ValueError('CRYSTAL-UNSUPPORTED-PROJECTION: local stereo/constraint reference image is unavailable')
    return dict(zip(rows, components[0][1]))


def _copy_index(image, repeats):
    target = tuple(value % repeat for value, repeat in zip(image, repeats))
    return (target[0] * repeats[1] + target[1]) * repeats[2] + target[2]


def _packet_mappings(record, rows, repeats, copies):
    lifts = _packet_lifts(record, rows)
    count = len(copies)
    return tuple({row: row * count + _copy_index(tuple(a + b for a, b in zip(copy, lift)), repeats)
                  for row, lift in lifts.items()} for copy in copies)


def _expanded_stereo(record, repeats, copies):
    payload = record.stereo_payload
    if payload is None:
        return None
    count = len(copies)
    records, atom_facts, bond_facts = [], [], []
    for entry in payload.records:
        rows = (*entry.anchor.atom_indices, *(carrier.atom for carrier in entry.carriers if carrier.atom is not None))
        packet = StereoPayload(payload.schema_version, (entry,), (), ())
        for mapping in _packet_mappings(record, rows, repeats, copies):
            records.extend(_remap_stereo_payload(packet, mapping).records)
    for fact in payload.atom_facts:
        atom_facts.extend(replace(fact, atom=fact.atom * count + copy) for copy in range(count))
    for fact in payload.bond_facts:
        packet = StereoPayload(payload.schema_version, (), (), (fact,))
        for mapping in _packet_mappings(record, fact.atom_indices, repeats, copies):
            bond_facts.extend(_remap_stereo_payload(packet, mapping).bond_facts)
    return StereoPayload(payload.schema_version, tuple(records), tuple(atom_facts), tuple(bond_facts))


def _expand_record(record, repeats):
    copies = tuple(product(*(range(repeat) for repeat in repeats)))
    count = len(copies)
    cell = record.cell
    if count > 1 and any(image is None for image in cell[4]):
        raise ValueError('CRYSTAL-UNSUPPORTED-PROJECTION: expansion target cell needs known Bond images')
    if count == 1:
        return replace(record, cell=(*cell[:5], 'supercell', *cell[6:]))
    values, presence, sources = _arrays(record)
    output = np.repeat(values[:, None, :], count, axis=1)
    output[:, :, _XYZ] = values[:, None, _XYZ] + np.asarray(copies) @ np.asarray(cell[1])
    codes = np.repeat(sources[:, None, :], count, axis=1)
    for copy, image in enumerate(copies):
        if any(image):
            codes[:, copy, _XYZ] = ATOM_SOURCE_CODES['geometry']
    edges = _supercell_edges(record.atom_count, record.bond_endpoints, cell[4], repeats)
    origin = cell[6]
    atom_origins = tuple(_translate_origin(origin, entry, image) for entry in cell[7] for image in copies)
    bond_origins = tuple(_translate_origin(origin, cell[8][edge], image)
                         for _, _, _, edge, image in edges)
    frame = None if origin[4] is None else affine._mm(origin[4], tuple(
        tuple(repeats[row] if row == column else 0 for column in range(3)) for row in range(3)))
    origin = (*origin[:4], frame)
    relations = ((), tuple(value for value in cell[3][1] for _ in copies),
                 tuple(value for value in cell[3][2] for _ in copies))
    destination = (cell[0], tuple(tuple(value * repeat for value in row)
        for row, repeat in zip(cell[1], repeats)), SpaceGroup(1), relations,
        tuple(item[2] for item in edges), 'supercell', origin, atom_origins, bond_origins)
    def constraints(entries):
        return tuple(tuple(mapping[row] for row in entry) for entry in entries
                     for mapping in _packet_mappings(record, entry, repeats, copies))
    charge = (True, record.charge[1] * count, 'crystal_generation') if record.charge[0] else record.charge
    return replace(record, atom_count=record.atom_count * count, atom_values=output.tobytes(),
        atom_presence=np.repeat(presence, count).tobytes(), atom_sources=codes.tobytes(),
        atom_frac_valences=tuple((row * count + copy, value) for row, value in record.atom_frac_valences for copy in range(count)),
        bond_endpoints=tuple((first, second) for first, second, *_ in edges),
        bond_values=b''.join(record.bond_values[edge * 24:(edge + 1) * 24] for _, _, _, edge, _ in edges),
        bond_kinds=tuple(record.bond_kinds[edge] for _, _, _, edge, _ in edges),
        bond_directions=tuple(record.bond_directions[edge] for _, _, _, edge, _ in edges),
        bond_sources=tuple(record.bond_sources[edge] for _, _, _, edge, _ in edges),
        bond_metadata=tuple(record.bond_metadata[edge] for _, _, _, edge, _ in edges),
        bond_aromatic=tuple(record.bond_aromatic[edge] for _, _, _, edge, _ in edges),
        cell=destination, charge=charge, spin_mult=(False, None, None),
        stereo_payload=_expanded_stereo(record, repeats, copies),
        angle_constraints=constraints(record.angle_constraints), torsion_constraints=constraints(record.torsion_constraints))


def _expand_crystal(crystal, repeats):
    repeats = tuple(map(operator.index, repeats))
    if len(repeats) != 3 or any(repeat <= 0 for repeat in repeats):
        raise ValueError('CRYSTAL-CONTEXT-INTEGRITY: repeats require three positive integers')
    cell = _unit_cell_crystal(crystal)
    with cell.molecule._mutation_lock:
        record = _capture_input(cell, cell.molecule)
        expanded = _expand_record(record, repeats)
        result = Molecule._build_molecule(expanded)
        result.properties = dict(cell.molecule.properties)
        return result.crystal


def _finite_molecules(crystal):
    cell = _unit_cell_crystal(crystal)
    molecule = cell.molecule
    with molecule._mutation_lock:
        record = _capture_input(cell, cell.molecule)
        result = []
        for rows, lifts, unknown, inconsistent in _finite_component_lifts(
                record.atom_count, record.bond_endpoints, record.cell[4]):
            if lifts is None:
                warnings.warn('CRYSTAL-MOLECULAR-LIFT-UNAVAILABLE: '
                    f'rows={rows}, unknown_edges={unknown}, nonzero_cycle_edges={inconsistent}',
                    RuntimeWarning, stacklevel=3)
                continue
            partial, metadata = molecule._capture_factual_record(rows=rows, include_frames=False)
            values, presence, sources = _arrays(partial)
            values, sources = values.copy(), sources.copy()
            values[:, _XYZ] += np.asarray(lifts, dtype=float) @ np.asarray(cell.matrix)
            for row, lift in enumerate(lifts):
                if any(lift):
                    sources[row, _XYZ] = ATOM_SOURCE_CODES['geometry']
            context = partial.cell
            origin, atom_origins, bond_origins = context[6:]
            atom_origins = tuple(_translate_origin(origin, entry, image)
                                 for entry, image in zip(atom_origins, lifts))
            bond_origins = tuple(_translate_origin(origin, entry, lifts[first])
                for entry, (first, _) in zip(bond_origins, partial.bond_endpoints))
            context = (*context[:4], ((0, 0, 0),) * len(partial.bond_endpoints),
                       context[5], origin, atom_origins, bond_origins)
            partial = replace(partial, atom_values=values.tobytes(), atom_sources=sources.tobytes(), cell=context)
            item = Molecule._build_molecule(partial, bond_metadata=metadata)
            item.properties = dict(molecule.properties)
            result.append(item)
        return tuple(result)
