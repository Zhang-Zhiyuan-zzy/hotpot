"""Closed current factual persistence; never a live-session/object loader."""

from dataclasses import fields

from ._builder import MoleculeRecord, _canonical_value, _freeze_value, _thaw_value
from ._crystal_codec import _crystal_from_data, _crystal_to_data, _fields
from ._results import PartialChargeSet
from ._stereo_codec import _stereo_from_data, _stereo_to_data


_BUFFERS = frozenset(('atom_values', 'atom_presence', 'atom_sources', 'bond_values'))
_OMITTED = frozenset(('source_frames', 'conformer_results', 'frame_selection'))
_RECORD_FIELDS = tuple(field.name for field in fields(MoleculeRecord) if field.name not in _OMITTED)
_CHARGE_FIELDS = tuple(field.name for field in fields(PartialChargeSet))


def _value_to_data(value):
    tag, *items = value
    if tag in ('bytes', 'numpy_scalar', 'ndarray'):
        items[-1] = items[-1].hex()
    elif tag == 'float':
        items[0] = items[0].hex()
    elif tag in ('list', 'tuple'):
        items[0] = [_value_to_data(item) for item in items[0]]
    elif tag == 'dict':
        items[0] = [[_value_to_data(key), _value_to_data(item)] for key, item in items[0]]
    return [tag, *items]


def _value_from_data(value):
    tag, *items = value
    if tag in ('bytes', 'numpy_scalar', 'ndarray'):
        items[-1] = bytes.fromhex(items[-1])
    elif tag == 'float':
        items[0] = float.fromhex(items[0])
    elif tag in ('list', 'tuple'):
        items[0] = tuple(_value_from_data(item) for item in items[0])
    elif tag == 'dict':
        items[0] = tuple((_value_from_data(key), _value_from_data(item)) for key, item in items[0])
    return _canonical_value((tag, *items))


def _facts_to_data(record, labels, contained):
    values = {}
    for name in _RECORD_FIELDS:
        value = getattr(record, name)
        if name in _BUFFERS:
            value = value.hex()
        elif name == 'cell':
            value = None if value is None else _crystal_to_data(value)
        elif name == 'stereo_payload':
            value = _stereo_to_data(value)
        elif name == 'partial_charge_result':
            value = None if value is None else {
                'payload': {field: _value_to_data(_freeze_value(getattr(value[0], field))) for field in _CHARGE_FIELDS},
                'invalidated_by': list(value[1])}
        else:
            value = _value_to_data(_freeze_value(value))
        values[name] = value
    return {'schema': 'hotpot.cif/1', 'labels': list(labels), 'contained': contained, 'record': values}


def _facts_from_data(data):
    schema, labels, contained, values = _fields(data, ('schema', 'labels', 'contained', 'record'))
    if schema != 'hotpot.cif/1' or type(contained) is not bool:
        raise ValueError('CRYSTAL-CONTEXT-INTEGRITY: current hotpot.cif/1 facts required')
    _fields(values, _RECORD_FIELDS)
    decoded = {}
    for name in _RECORD_FIELDS:
        value = values[name]
        if name in _BUFFERS:
            value = bytes.fromhex(value)
        elif name == 'cell':
            continue
        elif name == 'stereo_payload':
            value = _stereo_from_data(value, {row: row for row in range(decoded['atom_count'])},
                                      decoded['bond_endpoints'])
        elif name == 'partial_charge_result':
            if value is not None:
                payload, marker = _fields(value, ('payload', 'invalidated_by'))
                _fields(payload, _CHARGE_FIELDS)
                value = PartialChargeSet(**{key: _thaw_value(_value_from_data(payload[key])) for key in _CHARGE_FIELDS}), tuple(marker)
        else:
            value = _thaw_value(_value_from_data(value))
        decoded[name] = value
    decoded['cell'] = (None if values['cell'] is None else
                       _crystal_from_data(values['cell'], decoded['atom_count'], len(decoded['bond_endpoints'])))
    record = MoleculeRecord(**decoded, source_frames=(), conformer_results=(), frame_selection=None)
    if (len(labels) != record.atom_count or len(set(labels)) != len(labels)
            or any(type(label) is not str or label in ('', '?', '.') for label in labels)
            or (not contained and record.atom_count)):
        raise ValueError('CRYSTAL-IMAGE-MAP-INTEGRITY: CIF site labels must match complete current rows')
    return record, tuple(labels), contained
