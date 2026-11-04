"""Current structural CIF persistence over captured Hotpot factual records."""

from dataclasses import replace
import json
import os
from pathlib import Path
import re
import warnings

import numpy as np

from ..adapters.cif import _read_cif, _write_cif
from ..reference.properties import _ELEMENT_SYMBOLS
from ..rules.mathematics import symmetry as affine
from ..rules.mathematics.cell import _cell_matrix, _matrix_to_params, _to_fractional, _validate_matrix
from ..structure._builder import _BOND_KINDS, _freeze_value, _make_molecule_record
from ..structure._cif_facts import _facts_from_data, _facts_to_data
from ..structure._crystal import _new_crystal, _read_crystal_record
from ..structure._fields import ATOM_FIELDS, FIELD_INDEX, FIELD_PRESENCE_BITS
from ..structure.molecule import Molecule
from ..structure.stereo import AtomTransportRecord, StereoPayload
from ..structure.symmetry import SpaceGroup, SymmUnit


_MISSING = (None, '?', '.')
_XYZ = tuple(FIELD_INDEX[name] for name in ('x', 'y', 'z'))
_COORDINATES = sum(FIELD_PRESENCE_BITS[index] for index in _XYZ)
_TYPES = {'S': ('single', 1.), 'D': ('double', 2.), 'T': ('triple', 3.), 'A': ('aromatic', 1.5)}
_TYPE_CODES = {kind: code for code, (kind, _) in _TYPES.items()}
_CELL_TAGS = tuple('_cell_' + name for name in ('length_a', 'length_b', 'length_c',
                                              'angle_alpha', 'angle_beta', 'angle_gamma'))


def _finding(reason):
    warnings.warn('CRYSTAL-UNSUPPORTED-PROJECTION: ' + reason, RuntimeWarning, stacklevel=3)


def _pick(tags, *names):
    found = [tags[name] for name in names if name in tags]
    if found and any(value != found[0] for value in found[1:]):
        raise ValueError('CRYSTAL-CONTEXT-INTEGRITY: conflicting CIF aliases ' + ', '.join(names))
    return found[0] if found else ()


def _scalar(tags, *names):
    values = _pick(tags, *names)
    if len(values) > 1:
        raise ValueError('CRYSTAL-CONTEXT-INTEGRITY: CIF scalar has multiple values: ' + names[0])
    return values[0] if values else None


def _number(value):
    if value in _MISSING:
        return None
    result = float(re.sub(r'\(\d+\)(?=$|[eE])', '', value))
    if not np.isfinite(result):
        raise ValueError('CRYSTAL-CONTEXT-INTEGRITY: CIF numeric geometry must be finite')
    return result


def _column(tags, count, *names):
    value = _pick(tags, *names)
    if not value:
        return (None,) * count
    if len(value) != count:
        raise ValueError('CRYSTAL-IMAGE-MAP-INTEGRITY: CIF columns must align with complete site/Bond rows')
    return value


def _group(tags):
    symbols = _pick(tags, '_space_group_symop_operation_xyz', '_symmetry_equiv_pos_as_xyz')
    number = _scalar(tags, '_space_group_it_number', '_symmetry_int_tables_number')
    number = None if number in _MISSING else int(number)
    if number is not None and not 1 <= number <= 230:
        raise ValueError('CRYSTAL-CONTEXT-INTEGRITY: CIF reference number must be in 1..230')
    if not symbols:
        if number is not None or any('space_group_name' in tag for tag in tags):
            _finding('CIF reference labels without complete verified operators leave the active group unknown')
        return None, {}, number
    group = SpaceGroup.from_operations(symbols)
    if number is not None:
        reference = SpaceGroup(number)
        if {(value.R, value.t) for value in group.symm_units} == {
                (value.R, value.t) for value in reference.symm_units}:
            group = SpaceGroup.from_operations(symbols, reference_number=number,
                                                to_reference=(affine._IDENTITY, affine._ZERO))
        else:
            _finding('CIF reference number does not verify the actual setting; explicit operators remain anonymous')
    ids = _pick(tags, '_space_group_symop_id', '_symmetry_equiv_pos_site_id') or tuple(
        str(index + 1) for index in range(len(symbols)))
    if len(ids) != len(symbols) or len(set(ids)) != len(ids):
        raise ValueError('CRYSTAL-IMAGE-MAP-INTEGRITY: CIF operator IDs must be unique and aligned')
    return group, {key: SymmUnit.from_xyz(symbol, context=group) for key, symbol in zip(ids, symbols)}, number


def _species(value, *, label=False):
    if value in _MISSING:
        return None, None
    pattern = (r'([A-Z][a-z]?)(?:[0-9][A-Za-z0-9_\'\-]*|_[A-Za-z0-9_]+)?' if label else
               r'([A-Z][a-z]?)(?:[0-9]*[+-]|[+-][0-9]+|\((?:[IVX]+|[0-9]+)[+-]?\))?')
    match = re.fullmatch(pattern, value)
    if match is None:
        raise ValueError('CRYSTAL-UNSUPPORTED-PROJECTION: unsupported composite/dummy CIF species ' + value)
    symbol = match[1]
    isotope = {'D': 2, 'T': 3}.get(symbol)
    symbol = 'H' if isotope is not None else symbol
    if symbol not in _ELEMENT_SYMBOLS or symbol == _ELEMENT_SYMBOLS[0]:
        raise ValueError('CRYSTAL-UNSUPPORTED-PROJECTION: unrepresentable CIF species ' + value)
    return _ELEMENT_SYMBOLS.index(symbol), isotope


def _bond_action(code, operators):
    if code in (None, '?'):
        return None
    if code == '.':
        return affine._IDENTITY, affine._ZERO
    match = re.fullmatch(r'([^_]+)_([0-9]{3})', code)
    if match is None or match[1] not in operators:
        raise ValueError('CRYSTAL-IMAGE-MAP-INTEGRITY: unknown CIF Bond symmetry code ' + code)
    operation = operators[match[1]]
    return operation.R, tuple(value + int(digit) - 5 for value, digit in zip(operation.t, match[2]))


def _bond_image(first, second, operators):
    left, right = _bond_action(first, operators), _bond_action(second, operators)
    if left is None or right is None:
        return None
    rotation, translation = affine._compose(affine._invert(left), right)
    if rotation != affine._IDENTITY or any(value.denominator != 1 for value in translation):
        raise ValueError('CRYSTAL-UNSUPPORTED-PROJECTION: CIF Bond has a nontranslational relative operation')
    return tuple(map(int, translation))


def _standard(tags, *, matrix=None):
    parameters = tuple(_number(_scalar(tags, tag)) for tag in _CELL_TAGS)
    if any(value is None for value in parameters):
        raise ValueError('CRYSTAL-CONTEXT-INTEGRITY: CIF requires six explicit finite cell parameters')
    conventional = _cell_matrix(parameters)
    oriented = _scalar(tags, '_hotpot_cell_matrix_json')
    matrix = _validate_matrix(json.loads(oriented) if oriented not in _MISSING else
                              conventional if matrix is None else matrix)
    if not np.allclose(matrix @ matrix.T, conventional @ conventional.T, rtol=1e-10, atol=1e-10):
        raise ValueError('CRYSTAL-CONTEXT-INTEGRITY: CIF oriented basis and standard metric disagree')
    group, operators, number = _group(tags)
    labels = _pick(tags, '_atom_site_label', '_atom_site_label_atom_id')
    count = len(labels)
    if len(set(labels)) != count or any(value in ('', '?', '.') for value in labels):
        raise ValueError('CRYSTAL-IMAGE-MAP-INTEGRITY: CIF site labels require unique nonmissing values')
    kinds = _column(tags, count, '_atom_site_type_symbol')
    explicit_types = '_atom_site_type_symbol' in tags
    species = tuple(_species(kind if explicit_types else label, label=not explicit_types)
                    for label, kind in zip(labels, kinds))
    fractional = tuple(_column(tags, count, '_atom_site_fract_' + axis) for axis in 'xyz')
    cartesian = tuple(_column(tags, count, '_atom_site_cartn_' + axis) for axis in 'xyz')
    transform = None
    if any('_atom_site_cartn_' + axis in tags for axis in 'xyz'):
        rows = [[_number(_scalar(tags, f'_atom_sites_fract_transf_matrix_{i}{j}',
                                f'_atom_sites_fract_transf_matrix{i}{j}')) for j in (1, 2, 3)] for i in (1, 2, 3)]
        vector = [_number(_scalar(tags, f'_atom_sites_fract_transf_vector_{i}',
                                  f'_atom_sites_fract_transf_vector{i}')) for i in (1, 2, 3)]
        if any(value is None for row in rows for value in row) or any(value is None for value in vector):
            raise ValueError('CRYSTAL-UNSUPPORTED-PROJECTION: Cartesian CIF sites require an explicit fractional transform')
        transform = np.asarray(rows), np.asarray(vector)
    values = np.full((count, len(ATOM_FIELDS)), np.nan)
    presence = np.zeros(count, dtype='<u4')
    sources = np.zeros(values.shape, dtype='u1')
    isotope_facts = []
    for row, (element, isotope) in enumerate(species):
        if element is not None:
            field = FIELD_INDEX['atomic_number']
            values[row, field] = element
            presence[row] |= FIELD_PRESENCE_BITS[field]
            sources[row, field] = 1
        if isotope is not None:
            isotope_facts.append(AtomTransportRecord(row, isotope, None, None, None, 'cif'))
        point = tuple(_number(column[row]) for column in fractional)
        xyz = np.asarray(point) @ matrix if all(value is not None for value in point) else None
        if transform is not None:
            point = tuple(_number(column[row]) for column in cartesian)
            transformed = None if any(value is None for value in point) else (
                np.asarray(point) @ transform[0].T + transform[1]) @ matrix
            if xyz is not None and transformed is not None and not np.allclose(xyz, transformed, rtol=0., atol=1e-10):
                raise ValueError('CRYSTAL-CONTEXT-INTEGRITY: fractional and transformed Cartesian CIF sites disagree')
            if xyz is None:
                xyz = transformed
        if xyz is not None:
            values[row, _XYZ] = xyz
            presence[row] |= _COORDINATES
            sources[row, _XYZ] = 1
        elif any(column[row] not in _MISSING for column in fractional):
            _finding(f'CIF site {labels[row]} has an incomplete fractional triplet; Cartesian facts remain absent')
    occupancies = tuple(_number(value) for value in _column(tags, count, '_atom_site_occupancy'))
    assemblies = _column(tags, count, '_atom_site_disorder_assembly')
    groups = _column(tags, count, '_atom_site_disorder_group')
    disorder = tuple(None if first in _MISSING and second in _MISSING else
                     (None if first in _MISSING else first, None if second in _MISSING else second)
                     for first, second in zip(assemblies, groups))
    firsts = _pick(tags, '_geom_bond_atom_site_label_1')
    seconds = _pick(tags, '_geom_bond_atom_site_label_2')
    if len(firsts) != len(seconds):
        raise ValueError('CRYSTAL-IMAGE-MAP-INTEGRITY: CIF Bond endpoint columns must align')
    bond_count = len(firsts)
    types = _column(tags, bond_count, '_geom_bond_type', '_ccdc_geom_bond_type')
    symmetry2 = _column(tags, bond_count, '_geom_bond_site_symmetry_2')
    symmetry1 = (_column(tags, bond_count, '_geom_bond_site_symmetry_1')
                 if '_geom_bond_site_symmetry_1' in tags else ('.',) * bond_count)
    hot_kinds = _column(tags, bond_count, '_geom_bond_hotpot_kind')
    hot_orders = _column(tags, bond_count, '_geom_bond_hotpot_order')
    hot_images = _column(tags, bond_count, '_geom_bond_hotpot_image')
    label_rows = {label: row for row, label in enumerate(labels)}
    edges, bond_values, bond_kinds, images = [], [], [], []
    seen = set()
    for first, second, kind, a, b, hot_kind, hot_order, hot_image in zip(
            firsts, seconds, types, symmetry1, symmetry2, hot_kinds, hot_orders, hot_images):
        if first not in label_rows or second not in label_rows:
            raise ValueError('CRYSTAL-IMAGE-MAP-INTEGRITY: CIF Bond names an absent site')
        edge = label_rows[first], label_rows[second]
        key = tuple(sorted(edge))
        if edge[0] == edge[1] or key in seen:
            raise ValueError('CRYSTAL-UNSUPPORTED-PROJECTION: self/parallel/repeated CIF quotient Bond is outside the simple graph')
        seen.add(key)
        known = None if kind in _MISSING else _TYPES.get(kind.upper())
        if kind not in _MISSING and known is None:
            raise ValueError('CRYSTAL-UNSUPPORTED-PROJECTION: unmapped CIF Bond type ' + kind)
        actual_kind = ('unknown' if known is None else known[0]) if hot_kind in _MISSING else hot_kind
        if actual_kind not in _BOND_KINDS or (known is not None and actual_kind != known[0]):
            raise ValueError('CRYSTAL-CONTEXT-INTEGRITY: standard and Hotpot Bond kinds disagree')
        order = _number(hot_order)
        if known is not None and order is not None and abs(known[1] - order) > 1e-10:
            raise ValueError('CRYSTAL-CONTEXT-INTEGRITY: standard and Hotpot Bond orders disagree')
        order = (np.nan if known is None else known[1]) if order is None else order
        image = _bond_image(a, b, operators)
        if hot_image not in _MISSING:
            supplied = json.loads(hot_image)
            from ..structure._crystal import _image
            supplied = _image(supplied)
            if image is not None and supplied != image:
                raise ValueError('CRYSTAL-IMAGE-MAP-INTEGRITY: standard and Hotpot Bond images disagree')
            image = supplied
        edges.append(edge)
        bond_values.append((order, 0., 0.))
        bond_kinds.append(actual_kind)
        images.append(image)
    record = _make_molecule_record(atom_values=values, atom_presence=presence, atom_sources=sources,
        atom_frac_valences=(), bond_endpoints=edges, bond_values=np.asarray(bond_values).reshape(-1, 3),
        bond_kinds=bond_kinds, bond_directions=(None,) * bond_count, bond_sources=('cif',) * bond_count,
        bond_metadata=(None,) * bond_count, charge=(False, None, None), spin_mult=(False, None, None),
        cell=('hotpot.crystal/1', matrix, group, ((), occupancies, disorder), tuple(images), 'independent', None, None, None),
        stereo_payload=StereoPayload(1, (), tuple(isotope_facts), ()), source_frames=(), conformer_results=(),
        frame_selection=None, angle_constraints=(), torsion_constraints=(), identifier=(False, None))
    return record, tuple(labels), number


def _arrays(record):
    return (np.frombuffer(record.atom_values, dtype='<f8').reshape(record.atom_count, len(ATOM_FIELDS)),
            np.frombuffer(record.atom_presence, dtype='<u4'))


def _same_group(first, second):
    if first is None or second is None:
        return first is second
    return {(value.R, value.t) for value in first.symm_units} == {
        (value.R, value.t) for value in second.symm_units}


def _compare(record, labels, standard, standard_labels, number):
    def require(condition, subject):
        if not condition:
            raise ValueError('CRYSTAL-CONTEXT-INTEGRITY: standard CIF and hotpot.cif/1 disagree: ' + subject)
    require(labels == standard_labels, 'ordered site labels')
    require(record.atom_count == standard.atom_count, 'site count')
    require(np.allclose(record.cell[1], standard.cell[1], rtol=1e-10, atol=1e-10), 'oriented cell')
    require(_same_group(record.cell[2], standard.cell[2]), 'complete operators')
    if number is not None:
        require(record.cell[2] is not None and record.cell[2].number == number, 'reference number')
    actual, mask = _arrays(record)
    supplied, supplied_mask = _arrays(standard)
    element = FIELD_INDEX['atomic_number']
    bit = FIELD_PRESENCE_BITS[element]
    require(np.array_equal(mask & bit, supplied_mask & bit), 'element presence')
    rows = (mask & bit) != 0
    require(np.array_equal(actual[rows, element], supplied[rows, element]), 'elements')
    complete, provided = (mask & _COORDINATES) == _COORDINATES, (supplied_mask & _COORDINATES) == _COORDINATES
    require(np.array_equal(complete, provided), 'Cartesian coordinate presence')
    require(np.allclose(actual[complete][:, _XYZ], supplied[provided][:, _XYZ], rtol=0., atol=1e-10), 'coordinates')
    for left, right in zip(record.cell[3][1], standard.cell[3][1]):
        require(left is right if left is None or right is None else abs(left - right) <= 1e-10, 'occupancy')
    normalize = lambda values: tuple(None if row is None else tuple(None if value is None else str(value) for value in row)
                                     for row in values)
    require(normalize(record.cell[3][2]) == normalize(standard.cell[3][2]), 'disorder')
    require(record.bond_endpoints == standard.bond_endpoints, 'ordered Bond endpoints')
    require(record.bond_kinds == standard.bond_kinds, 'Bond kinds')
    require(record.cell[4] == standard.cell[4], 'Bond images')
    first = np.frombuffer(record.bond_values, dtype='<f8').reshape(-1, 3)[:, 0]
    second = np.frombuffer(standard.bond_values, dtype='<f8').reshape(-1, 3)[:, 0]
    require(np.allclose(first, second, rtol=0., atol=1e-10, equal_nan=True), 'Bond orders')
    if standard.stereo_payload is not None:
        facts = () if record.stereo_payload is None else record.stereo_payload.atom_facts
        for fact in standard.stereo_payload.atom_facts:
            require(any(value.atom == fact.atom and value.isotope == fact.isotope for value in facts), 'D/T isotope')


def _source_text(source):
    if isinstance(source, os.PathLike):
        return Path(source).read_bytes().decode('utf-8')
    if '\n' in source or source.lstrip().lower().startswith(('data_', 'global_', '#')):
        return source
    return Path(source).read_bytes().decode('utf-8')


def _from_cif(crystal_type, source, block):
    text = _source_text(source)
    selected = block
    try:
        selected, tags = _read_cif(text, block)
        state = _scalar(tags, '_hotpot_state_json')
        if state not in _MISSING:
            record, labels, contained = _facts_from_data(json.loads(state))
            standard, standard_labels, number = _standard(tags, matrix=record.cell[1])
            _compare(record, labels, standard, standard_labels, number)
        else:
            record, labels, _ = _standard(tags)
            contained = bool(record.atom_count)
        archive = {'text': text, 'block': selected}
        if contained:
            molecule = Molecule._build_molecule(record)
            molecule.properties['cif_source'] = archive
            crystal = molecule.crystal
            if type(crystal) is not crystal_type:
                crystal.__class__ = crystal_type
            return crystal
        crystal = _new_crystal(record.cell, crystal_type=crystal_type)
        crystal._cif_source = archive
        return crystal
    except (ValueError, TypeError, KeyError, IndexError, RuntimeError) as exc:
        error = exc if isinstance(exc, ValueError) and str(exc).startswith('CRYSTAL-') else ValueError(
            'CRYSTAL-UNSUPPORTED-PROJECTION: CIF projection failed: ' + str(exc))
        error.cif_source, error.cif_block = text, selected
        if error is exc:
            raise
        raise error from exc


def _capture(crystal):
    molecule = crystal.molecule
    extra = tuple(key for key in crystal.__dict__ if key not in ('_state', '_molecule', '_cif_source'))
    crystal_omissions = ('Crystal extras=' + repr(extra),) if extra else ()
    if molecule is None:
        empty = Molecule()
        record = empty._capture_factual_record()[0]
        record = replace(record, cell=_read_crystal_record(crystal))
        omissions = ('archived CIF source',) if getattr(crystal, '_cif_source', None) else ()
        return record, False, omissions + crystal_omissions
    with molecule._mutation_lock:
        record, metadata = molecule._capture_factual_record()
        frozen_metadata = []
        for row, value in enumerate(metadata):
            try:
                frozen_metadata.append(_freeze_value(value))
            except TypeError as exc:
                raise ValueError(f'CRYSTAL-UNSUPPORTED-PROJECTION: CIF Bond[{row}] metadata has no closed value encoding: {exc}') from exc
        record = replace(record, bond_metadata=tuple(frozen_metadata))
        omissions = list(crystal_omissions)
        if record.source_frames:
            omissions.append('source_frames')
        if record.conformer_results:
            omissions.append('conformer_results')
        if record.frame_selection is not None and record.frame_selection[1] is not None:
            omissions.append('selected frame observation')
        if molecule._mca_result is not None:
            omissions.append('MCA attachment')
        if molecule.properties:
            omissions.append('properties keys=' + repr(tuple(molecule.properties)))
        baseline = set(Molecule().__dict__) | {'identifier'}
        extras = tuple(key for key in molecule.__dict__ if key not in baseline)
        if extras:
            omissions.append('Molecule extras=' + repr(extras))
        for row, atom in enumerate(molecule._atoms):
            extra = tuple(key for key in atom.__dict__ if key not in ('_mol', '_member_uid', '_uid_owner', '_constraint_values', 'frac_valence'))
            if extra:
                omissions.append(f'Atom[{row}] extras={extra!r}')
        for row, bond in enumerate(molecule._bonds):
            extra = tuple(key for key in bond.__dict__ if key not in (
                '_atoms', '_bonds', '_member_uid', '_uid_owner', '_attrs', '_bond_kind', '_bond_direction',
                '_display_facts', '_bond_source', '_bond_source_metadata', '_is_aromatic', '_constraint_values'))
            if extra:
                omissions.append(f'Bond[{row}] extras={extra!r}')
        record = replace(record, source_frames=(), conformer_results=(), frame_selection=None)
    return record, True, tuple(omissions)


def _format(value):
    return '?' if value is None or not np.isfinite(value) else format(float(value), '.17g')


def _to_cif(crystal):
    record, contained, omissions = _capture(crystal)
    for reason in omissions:
        _finding('CIF current-facts export omits ' + reason)
    values, presence = _arrays(record)
    for field in _XYZ:
        present = (presence & FIELD_PRESENCE_BITS[field]) != 0
        if not np.isfinite(values[present, field]).all():
            raise ValueError('CRYSTAL-CONTEXT-INTEGRITY: CIF export requires finite supplied Cartesian geometry')
    element = FIELD_INDEX['atomic_number']
    present = (presence & FIELD_PRESENCE_BITS[element]) != 0
    numbers = values[present, element]
    if np.any(~np.isfinite(numbers)) or np.any(numbers != np.floor(numbers)) or np.any(
            (numbers <= 0) | (numbers >= len(_ELEMENT_SYMBOLS))):
        raise ValueError('CRYSTAL-UNSUPPORTED-PROJECTION: CIF export requires representable supplied elements')
    if any(value is not None and not np.isfinite(value) for value in record.cell[3][1]):
        raise ValueError('CRYSTAL-UNSUPPORTED-PROJECTION: CIF export requires finite supplied occupancies')
    group = record.cell[2]
    isotope = {}
    if record.stereo_payload is not None:
        for fact in record.stereo_payload.atom_facts:
            if fact.isotope is not None:
                isotope[fact.atom] = fact.isotope
    symbols = [(_ELEMENT_SYMBOLS[int(values[row, FIELD_INDEX['atomic_number']])]
                if presence[row] & FIELD_PRESENCE_BITS[FIELD_INDEX['atomic_number']] else '?')
               for row in range(record.atom_count)]
    symbols = [('D' if isotope.get(row) == 2 else 'T' if isotope.get(row) == 3 else symbol)
               if symbol == 'H' else symbol for row, symbol in enumerate(symbols)]
    labels = tuple((symbol if symbol != '?' else 'X') + str(row + 1) for row, symbol in enumerate(symbols))
    state = _facts_to_data(record, labels, contained)
    pairs = list(zip(_CELL_TAGS, map(_format, _matrix_to_params(record.cell[1]))))
    pairs.extend((('_hotpot_cell_matrix_json', json.dumps(record.cell[1], separators=(',', ':'))),
                  ('_hotpot_state_json', json.dumps(state, separators=(',', ':'), allow_nan=False))))
    loops = []
    operators = () if group is None else group.symm_units
    if operators:
        loops.append((['_space_group_symop_id', '_space_group_symop_operation_xyz'],
                      [[str(index + 1), value.xyz] for index, value in enumerate(operators)]))
        if group.number is not None:
            pairs.append(('_space_group_IT_number', str(group.number)))
    site_rows = []
    for row, (label, symbol, occupancy, disorder) in enumerate(zip(labels, symbols, record.cell[3][1], record.cell[3][2])):
        complete = (presence[row] & _COORDINATES) == _COORDINATES
        point = _to_fractional(values[row, _XYZ], record.cell[1]) if complete else (None,) * 3
        disorder = (None, None) if disorder is None else disorder
        site_rows.append([label, symbol, *map(_format, point), _format(occupancy),
                          *('?' if item is None else str(item) for item in disorder)])
    if site_rows:
        loops.append((['_atom_site_label', '_atom_site_type_symbol', '_atom_site_fract_x', '_atom_site_fract_y',
                       '_atom_site_fract_z', '_atom_site_occupancy', '_atom_site_disorder_assembly',
                       '_atom_site_disorder_group'], site_rows))
    identity = next((str(index + 1) for index, operation in enumerate(operators)
                     if operation.R == affine._IDENTITY and not any(operation.t)), None)
    raw_bonds = np.frombuffer(record.bond_values, dtype='<f8').reshape(-1, 3)
    if np.isinf(raw_bonds[:, 0]).any():
        raise ValueError('CRYSTAL-UNSUPPORTED-PROJECTION: CIF export cannot represent infinite Bond orders')
    bond_rows = []
    for edge, ((first, second), kind, raw, image) in enumerate(zip(
            record.bond_endpoints, record.bond_kinds, raw_bonds, record.cell[4])):
        if image is None:
            code = '?'
        elif not any(image):
            code = '.'
        elif identity is not None and all(-5 <= value <= 4 for value in image):
            code = identity + '_' + ''.join(str(value + 5) for value in image)
        else:
            code = '?'
            _finding(f'CIF conventional Bond[{edge}] image code unavailable for {image!r}; exact Hotpot image retained')
        type_code = _TYPE_CODES.get(kind, '?')
        if type_code != '?' and (not np.isfinite(raw[0]) or abs(_TYPES[type_code][1] - raw[0]) > 1e-10):
            type_code = '?'
            _finding(f'CIF standard Bond[{edge}] type cannot express kind {kind} with order {raw[0]}; exact Hotpot facts retained')
        if kind not in _TYPE_CODES and kind != 'unknown':
            _finding(f'CIF standard Bond[{edge}] type cannot express {kind}; exact Hotpot facts retained')
        bond_rows.append([labels[first], labels[second], type_code, '.', code, kind, _format(raw[0]),
                          '?' if image is None else json.dumps(image, separators=(',', ':'))])
    if bond_rows:
        loops.append((['_geom_bond_atom_site_label_1', '_geom_bond_atom_site_label_2', '_geom_bond_type',
                       '_geom_bond_site_symmetry_1', '_geom_bond_site_symmetry_2', '_geom_bond_hotpot_kind',
                       '_geom_bond_hotpot_order', '_geom_bond_hotpot_image'], bond_rows))
    return _write_cif('hotpot', pairs, loops)
