"""The closed molecular Data codec. Torch imports occur only at conversion calls."""

import numpy as np
import warnings

SCHEMA = 'hotpot.molecule/2'
KINDS = ('unknown', 'zero', 'single', 'double', 'triple', 'aromatic', 'dative')
ATOM_COLUMNS = ('atomic_number', 'formal_charge', 'implicit_hydrogens', 'is_aromatic', 'id', 'x', 'y', 'z')
NODE_FIELDS = ('z', 'formal_charge', 'total_h', 'aromatic', 'atom_id')
EDGE_FIELDS = ('bond_key', 'bond_order', 'bond_kind', 'bond_direction', 'bond_aromatic', 'bond_id', 'bond_constraint')
INDEX_WIDTHS = {'edge_index': 2, 'angle_index': 3, 'torsion_index': 4, 'wiberg_index': 2}


def _portable(value, path):
    """Copy only the existing portable value domain, before external callbacks."""
    if value is None or type(value) in (str, bool, int, float):
        return value
    if type(value) is bytes:
        return {'type': 'bytes', 'hex': value.hex()}
    if isinstance(value, np.generic):
        return {'type': 'numpy_scalar', 'value': _portable(np.asarray(value), path)}
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in 'biufc':
            raise TypeError(f'pyg_unsupported_metadata {path}: dtype {value.dtype}')
        return {'type': 'array', 'dtype': value.dtype.str, 'shape': list(value.shape),
                'hex': value.tobytes(order='C').hex()}
    if type(value) in (list, tuple):
        return {'type': 'tuple' if type(value) is tuple else 'list',
                'items': [_portable(item, f'{path}[{i}]') for i, item in enumerate(value)]}
    if type(value) is dict:
        return {'type': 'dict', 'items': [(_portable(key, f'{path}.key'),
                  _portable(item, f'{path}[{key!r}]')) for key, item in value.items()]}
    raise TypeError(f'pyg_unsupported_metadata {path}: {type(value).__name__}')


def _portable_tensors(value, torch):
    if type(value) is dict and value.get('type') == 'array':
        dtype = np.dtype(value['dtype'])
        if dtype.itemsize in ({'b': (1,), 'i': (1, 2, 4, 8), 'u': (1, 2, 4, 8),
                              'f': (2, 4, 8), 'c': (8, 16)}[dtype.kind]):
            array = np.frombuffer(bytes.fromhex(value['hex']), dtype=dtype).reshape(value['shape'])
            return {'type': 'array', 'dtype': dtype.str, 'shape': list(value['shape']),
                    'tensor': torch.from_numpy(array.astype(dtype.newbyteorder('='), copy=True))}
        return dict(value)
    if type(value) is dict:
        return {key: _portable_tensors(item, torch) for key, item in value.items()}
    if type(value) in (tuple, list):
        return [_portable_tensors(item, torch) for item in value]
    return value


def _mapping(value, names, path):
    if type(value) is not dict:
        raise ValueError(f'pyg_schema {path}: requires a structured mapping')
    for name in names:
        if name not in value:
            raise ValueError(f'pyg_schema {path}.{name}: missing field')
    return value


def _unportable(value, path='metadata'):
    if value is None or type(value) in (str, bool, int, float, bytes):
        return value
    _mapping(value, ('type',), path)
    kind = value['type']
    if kind == 'bytes':
        _mapping(value, ('hex',), path)
        return bytes.fromhex(value['hex'])
    if kind == 'array':
        import torch
        _mapping(value, ('dtype', 'shape'), path)
        try:
            dtype = np.dtype(value['dtype'])
        except (TypeError, ValueError) as error:
            raise ValueError(f'pyg_schema {path}.dtype: invalid array dtype') from error
        if dtype.kind not in 'biufc':
            raise TypeError(f'pyg_unsupported_metadata {path}: dtype {dtype}')
        if 'tensor' in value:
            tensor = value['tensor']
            if not isinstance(tensor, torch.Tensor) or list(tensor.shape) != value['shape']:
                raise ValueError(f'pyg_schema {path}.tensor: declared shape mismatch')
            return tensor.detach().cpu().numpy().astype(dtype, copy=True)
        _mapping(value, ('hex',), path)
        return np.frombuffer(bytes.fromhex(value['hex']), dtype=dtype).reshape(value['shape']).copy()
    if kind == 'numpy_scalar':
        _mapping(value, ('value',), path)
        return _unportable(value['value'], path + '.value')[()]
    if kind not in ('dict', 'list', 'tuple'):
        raise ValueError(f'pyg_schema {path}.type: unsupported portable encoding {kind!r}')
    _mapping(value, ('items',), path)
    if kind == 'dict':
        return {_unportable(key, path + '.key'): _unportable(item, f'{path}[{i}]')
                for i, (key, item) in enumerate(value['items'])}
    items = [_unportable(item, f'{path}[{i}]') for i, item in enumerate(value['items'])]
    return tuple(items) if kind == 'tuple' else items


def _stereo(value, keys, edges):
    from ..structure._stereo_codec import _stereo_from_data
    try:
        return _stereo_from_data(value, keys, edges)
    except (ValueError, TypeError) as error:
        code = 'pyg_stereo_reference' if str(error).startswith('stereo_reference') else 'pyg_schema'
        raise type(error)(f'{code} hotpot_stereo: {error}') from error


def _subjects(uids, current, namespace):
    return {'namespace': namespace, 'ordinals': list(range(len(uids))),
            'atom_keys': [current.get(uid) for uid in uids]}


def _history_provenance(value, current):
    """Translate the codecs' named removed-subject evidence, never arbitrary ints."""
    if type(value) is dict:
        return {key: (_subjects(item, current, 'historical_source_subjects') if key == 'whole_system_subjects'
            else {'subjects': _subjects(item[0], current, 'historical_source_subjects'), 'coordinates': item[1]}
            if key == 'whole_system_coordinates' and type(item) in (tuple, list) and len(item) == 2
            else _history_provenance(item, current)) for key, item in value.items()}
    if type(value) in (tuple, list):
        return type(value)(_history_provenance(item, current) for item in value)
    return value


def _frame_observation(payload, current, namespace):
    if payload is None:
        return None
    from ..structure._builder import _thaw_canonical
    return {'subjects': _subjects(payload.atom_uids, current, namespace),
            'values': [_portable(_thaw_canonical(item), f'{namespace}.values[{i}]') for i, item in enumerate(payload.values)],
            'frame_indices': [list(row) for row in payload.frame_indices],
            'method': payload.method, 'method_version': payload.method_version,
            'parameters': _portable(_thaw_canonical(payload.parameters), namespace + '.parameters'),
            'units': _portable(_thaw_canonical(payload.units), namespace + '.units'),
            'provenance': _portable(_history_provenance(_thaw_canonical(payload.provenance), current), namespace + '.provenance')}


def capture_details(record, metadata, *, properties, atom_uids, source_frames, generated, selected_frame,
                    partial_charge, mca, mca_join):
    """Pure capture from values already selected under the domain owner lock."""
    from ..structure._builder import _thaw_canonical
    current = {uid: row for row, uid in enumerate(atom_uids)}
    annotations = {key: value for key, value in properties.items() if key != 'pyg_observations'}
    graph = {'molecule_present': True, 'charge_source': record.charge[2], 'spin_source': record.spin_mult[2],
             'identifier': _portable(record.identifier, 'identifier'),
             'annotations': _portable(annotations, 'properties')}
    bonds = [{'bond_key': row, 'source': record.bond_sources[row],
              'other_direction': record.bond_directions[row] if record.bond_directions[row] not in
                  (None, 'atom1_to_atom2', 'atom2_to_atom1') else None,
              'metadata': _portable(item, f'bonds[{row}].metadata')} for row, item in enumerate(metadata)]
    from ..structure._stereo_codec import _stereo_to_data
    stereo = _stereo_to_data(record.stereo_payload)
    _stereo(stereo, dict(enumerate(range(record.atom_count))), record.bond_endpoints)
    sources = []
    for i, frame in enumerate(source_frames):
        namespace = f'source_frame/{i}'
        sources.append({'subjects': _subjects(frame.atom_uids, current, namespace),
            'coordinates': frame.coordinates.copy(), 'observation': _frame_observation(frame.observations, current, namespace + '/observations'),
            'observation_index': frame.observation_index,
            'provenance': _portable(_history_provenance(_thaw_canonical(frame.provenance), current), namespace + '.provenance')})
    ensembles = []
    for i, attachment in enumerate(generated):
        payload = attachment.payload
        namespace = f'generated/{i}'
        ensembles.append({'subjects': _subjects(payload.atom_uids, current, namespace),
            'coordinates': payload.batch.coordinates.copy(), 'ordinals': list(payload.batch.ordinals),
            'observation': _frame_observation(payload.observations, current, namespace + '/observations'),
            'invalidated_by': list(attachment.invalidated_by)})
    frame_selection = None if selected_frame is None else {
        'observation': _frame_observation(selected_frame.payload[0], current, 'selected_frame'),
        'row': selected_frame.payload[1], 'invalidated_by': list(selected_frame.invalidated_by)}
    observations = []
    if partial_charge is not None:
        payload = partial_charge.payload
        observations.append({'kind': 'partial_charge', 'subjects': _subjects(payload.atom_uids, current, 'partial_charge'),
            'values': list(payload.values), 'method': payload.method, 'method_version': payload.method_version,
            'parameters': [list(item) for item in payload.parameters], 'units': payload.units,
            'provenance': list(payload.provenance), 'invalidated_by': list(partial_charge.invalidated_by)})
    if mca is not None:
        payload = mca.payload
        observations.append({'kind': 'mca', 'subjects': {'namespace': 'mca_original_rows',
            'ordinals': [item.atom_index for item in payload.atom_predictions],
            'atom_keys': [item.atom_index if mca_join else None for item in payload.atom_predictions]},
            'prediction': payload.to_dict(), 'method': 'mca', 'units': 'kJ/mol',
            'invalidated_by': list(mca.invalidated_by)})
    return {'hotpot_graph': graph, 'hotpot_bonds': bonds, 'hotpot_stereo': stereo,
            'hotpot_fractional_valence': [{'atom_key': row, 'value': value} for row, value in record.atom_frac_valences],
            'hotpot_frames': {'sources': sources, 'generated': ensembles, 'selected': frame_selection},
            'hotpot_observations': {'current_source': observations,
                'historical': _portable(properties.get('pyg_observations'), 'properties.pyg_observations')}}


def _tensors(value, torch):
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if type(value) is dict:
        if value.get('type') in ('array', 'numpy_scalar', 'list', 'tuple', 'dict', 'bytes'):
            return _portable_tensors(value, torch)
        return {key: _tensors(item, torch) for key, item in value.items()}
    if type(value) in (tuple, list):
        return [_tensors(item, torch) for item in value]
    return value


def encode(record, details, available):
    import torch
    from torch_geometric.data import Data
    from ..structure._fields import FIELD_INDEX, FIELD_PRESENCE_BITS
    n = record.atom_count
    values = np.frombuffer(record.atom_values, dtype='<f8').reshape(n, 19)
    masks = np.frombuffer(record.atom_presence, dtype='<u4')
    codes = np.frombuffer(record.atom_sources, dtype='u1').reshape(n, 19)
    indices = [FIELD_INDEX[name] for name in ATOM_COLUMNS]
    present = np.asarray([[bool(mask & FIELD_PRESENCE_BITS[index]) for index in indices] for mask in masks], dtype=bool).reshape(n, 8)
    def tensor(value, dtype):
        return torch.tensor(value, dtype=dtype)
    data = Data(num_nodes=n, hotpot_schema=[SCHEMA], atom_key=torch.arange(n),
                atom_present=tensor(present, torch.bool), atom_source=tensor(codes[:, indices].copy(), torch.uint8),
                representation_available=tensor(available, torch.bool).reshape(n, 2))
    for column, name in enumerate(NODE_FIELDS):
        raw = values[:, indices[column]]
        selected = np.where(present[:, column], raw, 0)
        data[name] = tensor(selected.astype(bool if name == 'aromatic' else np.int64), torch.bool if name == 'aromatic' else torch.int64)
    data.pos = tensor(values[:, 10:13].copy(), torch.float64)
    m = len(record.bond_endpoints)
    bond_values = np.frombuffer(record.bond_values, dtype='<f8').reshape(m, 3)
    data.edge_index = tensor(record.bond_endpoints, torch.int64).reshape(m, 2).T.contiguous()
    data.bond_key = torch.arange(m)
    data.bond_order = tensor(bond_values[:, 0].copy(), torch.float64)
    data.bond_kind = tensor([KINDS.index(kind) for kind in record.bond_kinds], torch.int64)
    data.bond_direction = tensor([1 if d == 'atom1_to_atom2' else -1 if d == 'atom2_to_atom1' else 0 for d in record.bond_directions], torch.int8)
    data.bond_aromatic = tensor(record.bond_aromatic, torch.bool)
    data.bond_id = tensor(bond_values[:, 2].astype(np.int64), torch.int64)
    data.bond_constraint = tensor(bond_values[:, 1].astype(bool), torch.bool)
    data.atom_constraint = tensor(values[:, 16:19].astype(bool), torch.bool)
    for name, entries in [('angle_index', record.angle_constraints), ('torsion_index', record.torsion_constraints)]:
        data[name] = tensor(entries, torch.int64).reshape(-1, INDEX_WIDTHS[name]).T.contiguous()
    pairs = record.pair_wiberg_bond_orders
    data.wiberg_index = tensor([item[:2] for item in pairs], torch.int64).reshape(-1, 2).T.contiguous()
    data.wiberg_value = tensor([item[2] for item in pairs], torch.float64)
    for name, declaration in [('net_charge', record.charge), ('spin_mult', record.spin_mult)]:
        data[name] = tensor([declaration[1] if declaration[0] else 0], torch.float64)
        data[name + '_present'] = tensor([declaration[0]], torch.bool)
    from ._pyg_crystal import encode_cell
    for name, value in encode_cell(record.cell, n, m).items():
        data[name] = value
    for name, value in details.items():
        data[name] = [_tensors(value, torch)]
    return data


def _plain(value, path):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().copy()
    if type(value) is dict:
        if value.get('type') in ('array', 'numpy_scalar', 'list', 'tuple', 'dict', 'bytes'):
            return _unportable(value)
        return {key: _plain(item, f'{path}.{key}') for key, item in value.items()}
    if type(value) in (list, tuple):
        return type(value)(_plain(item, f'{path}[{i}]') for i, item in enumerate(value))
    if value is None or type(value) in (str, bytes, int, float, bool):
        return value
    raise TypeError(f'pyg_unsupported_metadata {path}: {type(value).__name__}')


def _wrapper(data, name):
    value = data.get(name)
    if type(value) is not list or len(value) != 1:
        raise ValueError(f'pyg_schema {name}: requires one graph wrapper')
    return value[0]


def _array(data, name, dtype, shape):
    import torch
    value = data.get(name)
    if not isinstance(value, torch.Tensor) or value.dtype != dtype or tuple(value.shape) != shape:
        raise ValueError(f'pyg_schema {name}: requires {dtype} shape={shape}')
    return value.detach().cpu().numpy().copy()


def _check_edges(edges, n, field='edge_index'):
    seen = set()
    for first, second in edges:
        if first < 0 or second < 0 or first >= n or second >= n or first == second:
            raise ValueError(f'pyg_mapping {field}: invalid endpoints {(first, second)}')
        key = frozenset((first, second))
        if key in seen:
            raise ValueError(f'pyg_mapping {field}: parallel bond {(first, second)}')
        seen.add(key)


def _check_subjects(subjects, keys, path):
    _mapping(subjects, ('namespace', 'ordinals', 'atom_keys'), path)
    ordinals, members = subjects['ordinals'], subjects['atom_keys']
    if (type(subjects['namespace']) is not str or type(ordinals) is not list
            or type(members) is not list or len(ordinals) != len(members)
            or any(type(row) is not int for row in ordinals) or len(set(ordinals)) != len(ordinals)):
        raise ValueError(f'pyg_schema {path}: unique original ordinals must align with subjects')
    joined = [key for key in members if key is not None]
    if any(type(key) is not int for key in joined) or len(set(joined)) != len(joined):
        raise ValueError(f'pyg_mapping {path}: current atom joins must be unique integer keys')
    for key in joined:
        if key not in keys:
            raise ValueError(f'pyg_mapping {path}: missing atom_key={key}')


def _source_frames(value, keys):
    from ..structure.conformer import CoordinateFrame, EnergyForceResult, _remap_observations
    from ..structure._builder import _freeze_value
    _mapping(value, ('sources', 'generated', 'selected'), 'hotpot_frames')
    def observation(item):
        if item is None:
            return None
        _mapping(item, ('subjects', 'values', 'frame_indices', 'method', 'method_version',
                        'parameters', 'units', 'provenance'), 'frames.observations')
        _check_subjects(item['subjects'], keys, 'frames.observations.subjects')
        entries = item['subjects']['atom_keys']
        values = tuple(_unportable(v, f'frames.observations.values[{i}]') for i, v in enumerate(item['values']))
        indices = item['frame_indices']
        if len(values) != 8 or any(len(row) != 8 for row in indices):
            raise ValueError('pyg_schema frames.observations: requires eight observation slots')
        for row in indices:
            for index, value in zip(row, values):
                if index is None:
                    continue
                if value is None or (index != 'whole' and
                        (type(index) is not int or np.ndim(value) == 0 or not 0 <= index < len(value))):
                    raise ValueError('pyg_mapping frames.observations.frame_indices: unavailable observation row')
        for column, width in ((1, None), (3, 3)):
            if values[column] is not None:
                for row in indices:
                    index = row[column]
                    if index is not None:
                        shape = np.shape(values[column] if index == 'whole' else values[column][index])
                        if shape != ((len(entries),) if width is None else (len(entries), width)):
                            raise ValueError('pyg_mapping frames.observations.values: per-atom subject shape mismatch')
        payload = EnergyForceResult(tuple(range(len(entries))),
            tuple(_freeze_value(v) for v in values), tuple(tuple(row) for row in indices),
            item['method'], item['method_version'],
            _freeze_value(_unportable(item['parameters'], 'frames.observations.parameters')),
            _freeze_value(_unportable(item['units'], 'frames.observations.units')),
            _freeze_value(_unportable(item['provenance'], 'frames.observations.provenance')))
        mapping = {row: keys[key] for row, key in enumerate(entries) if key is not None}
        return _remap_observations(payload, mapping, fragment=len(mapping) != len(entries))
    sources = []
    for item in value['sources']:
        _mapping(item, ('subjects', 'coordinates', 'observation', 'observation_index', 'provenance'), 'source_frames')
        _check_subjects(item['subjects'], keys, 'source_frames.subjects')
        entries = item['subjects']['atom_keys']
        kept = [(row, keys[key]) for row, key in enumerate(entries) if key is not None]
        coordinates = _plain(item['coordinates'], 'frames.coordinates')
        if np.shape(coordinates) != (len(entries), 3):
            raise ValueError('pyg_schema source_frames.coordinates: subject shape mismatch')
        observed = observation(item['observation'])
        index = item['observation_index']
        if ((observed is None) != (index is None) or (index is not None and
                (type(index) is not int or not 0 <= index < len(observed.frame_indices)))):
            raise ValueError('pyg_mapping source_frames.observation_index: unavailable observation row')
        sources.append(CoordinateFrame(tuple(row for _, row in kept),
            coordinates[[row for row, _ in kept]], observed, index,
            _freeze_value(_unportable(item['provenance'], 'source_frames.provenance'))))
    for item in value['generated']:
        _mapping(item, ('subjects', 'coordinates', 'ordinals', 'observation', 'invalidated_by'), 'generated_frames')
        _check_subjects(item['subjects'], keys, 'generated_frames.subjects')
        coordinates = _plain(item['coordinates'], 'generated_frames.coordinates')
        if np.shape(coordinates) != (len(item['ordinals']), len(item['subjects']['atom_keys']), 3):
            raise ValueError('pyg_schema generated_frames.coordinates: frame/subject shape mismatch')
        observed = observation(item['observation'])
        if observed is not None and len(observed.frame_indices) != len(item['ordinals']):
            raise ValueError('pyg_mapping generated_frames.observations: frame selectors differ from ordinals')
    if value['selected'] is not None:
        item = _mapping(value['selected'], ('observation', 'row', 'invalidated_by'), 'selected_frame')
        observed = observation(item['observation'])
        if observed is None or type(item['row']) is not int or not 0 <= item['row'] < len(observed.frame_indices):
            raise ValueError('pyg_mapping selected_frame.row: unavailable observation row')
    return tuple(sources)


def _decode_schema(data):
    import torch
    from ..structure._builder import _make_molecule_record
    from ..structure._fields import FIELD_INDEX, FIELD_PRESENCE_BITS, ATOM_SOURCE_CODES
    n = data['num_nodes'] if 'num_nodes' in data else None
    if type(n) is not int or n < 0:
        raise ValueError('pyg_schema num_nodes: explicit nonnegative count required')
    atom_keys = _array(data, 'atom_key', torch.int64, (n,)).tolist()
    if len(set(atom_keys)) != n:
        raise ValueError('pyg_mapping atom_key: duplicate keys')
    key_rows = {key: row for row, key in enumerate(atom_keys)}
    present = _array(data, 'atom_present', torch.bool, (n, 8))
    codes = _array(data, 'atom_source', torch.uint8, (n, 8))
    if not np.isin(codes, tuple(ATOM_SOURCE_CODES.values())).all():
        raise ValueError('pyg_schema atom_source: unknown producer code')
    _array(data, 'representation_available', torch.bool, (n, 2))
    values = np.zeros((n, 19), dtype=np.float64)
    masks = np.zeros(n, dtype=np.uint32)
    sources = np.zeros((n, 19), dtype=np.uint8)
    for column, name in enumerate(NODE_FIELDS):
        values[:, FIELD_INDEX[ATOM_COLUMNS[column]]] = _array(data, name, torch.bool if name == 'aromatic' else torch.int64, (n,))
    values[:, 10:13] = _array(data, 'pos', torch.float64, (n, 3))
    for column, name in enumerate(ATOM_COLUMNS):
        index = FIELD_INDEX[name]
        masks |= np.where(present[:, column], FIELD_PRESENCE_BITS[index], 0).astype(np.uint32)
        sources[:, index] = codes[:, column]
    values[:, 16:19] = _array(data, 'atom_constraint', torch.bool, (n, 3))
    for i in range(16, 19):
        masks |= np.uint32(FIELD_PRESENCE_BITS[i])
        sources[:, i] = ATOM_SOURCE_CODES['explicit']
    edges_value = data.get('edge_index')
    m = edges_value.shape[1] if isinstance(edges_value, torch.Tensor) and edges_value.ndim == 2 else -1
    edges = [tuple(map(int, row)) for row in _array(data, 'edge_index', torch.int64, (2, m)).T]
    _check_edges(edges, n)
    bond_keys = _array(data, 'bond_key', torch.int64, (m,)).tolist()
    if len(set(bond_keys)) != m:
        raise ValueError('pyg_mapping bond_key: duplicate keys')
    bond_values = np.column_stack((_array(data, 'bond_order', torch.float64, (m,)),
        _array(data, 'bond_constraint', torch.bool, (m,)), _array(data, 'bond_id', torch.int64, (m,))))
    kind_codes = _array(data, 'bond_kind', torch.int64, (m,))
    if ((kind_codes < 0) | (kind_codes >= len(KINDS))).any():
        raise ValueError('pyg_schema bond_kind: unknown code')
    directions = _array(data, 'bond_direction', torch.int8, (m,))
    if not np.isin(directions, (-1, 0, 1)).all():
        raise ValueError('pyg_schema bond_direction: unknown code')
    aromatic = _array(data, 'bond_aromatic', torch.bool, (m,)).tolist()
    wrapped_bonds = _wrapper(data, 'hotpot_bonds')
    for item in wrapped_bonds:
        _mapping(item, ('bond_key', 'source', 'other_direction', 'metadata'), 'hotpot_bonds')
    metadata = {item['bond_key']: item for item in wrapped_bonds}
    if len(metadata) != len(wrapped_bonds) or set(metadata) != set(bond_keys):
        raise ValueError('pyg_mapping hotpot_bonds: bond keys differ from current edges')
    graph = _mapping(_wrapper(data, 'hotpot_graph'),
        ('annotations', 'identifier', 'charge_source', 'spin_source', 'molecule_present'), 'hotpot_graph')
    properties = _unportable(graph['annotations'], 'hotpot_graph.annotations')
    if type(properties) is not dict:
        properties = {'annotations': properties}
    frame_values = _wrapper(data, 'hotpot_frames')
    frames = _source_frames(frame_values, key_rows)
    observations = _mapping(_wrapper(data, 'hotpot_observations'),
        ('current_source', 'historical'), 'hotpot_observations')
    for item in observations['current_source']:
        _mapping(item, ('kind', 'subjects', 'invalidated_by'), 'observations')
        _check_subjects(item['subjects'], key_rows, 'observations.subjects')
        if item['kind'] == 'partial_charge':
            _mapping(item, ('values', 'method', 'method_version', 'parameters', 'units', 'provenance'), 'observations.partial_charge')
            if len(item['values']) != len(item['subjects']['ordinals']) or len(item['provenance']) != len(item['values']):
                raise ValueError('pyg_mapping observations.partial_charge: subject/value shape mismatch')
        elif item['kind'] == 'mca':
            _mapping(item, ('prediction', 'method', 'units'), 'observations.mca')
            prediction = _mapping(item['prediction'], ('atom_predictions', 'sites'), 'observations.mca.prediction')
            if [entry['atom_index'] for entry in prediction['atom_predictions']] != item['subjects']['ordinals']:
                raise ValueError('pyg_mapping observations.mca: prediction rows differ from subjects')
        else:
            raise ValueError(f"pyg_schema observations.kind: unknown observation {item['kind']!r}")
    fractional = _wrapper(data, 'hotpot_fractional_valence')
    for item in fractional:
        _mapping(item, ('atom_key', 'value'), 'hotpot_fractional_valence')
        if item['atom_key'] not in key_rows:
            raise ValueError(f"pyg_mapping fractional_valence: missing atom_key={item['atom_key']}")
    constraints = {}
    for name in ('angle_index', 'torsion_index', 'wiberg_index'):
        tensor = data.get(name)
        width = tensor.shape[1] if isinstance(tensor, torch.Tensor) and tensor.ndim == 2 else -1
        array = _array(data, name, torch.int64, (INDEX_WIDTHS[name], width))
        if ((array < 0) | (array >= n)).any():
            raise ValueError(f'pyg_mapping {name}: atom row outside graph')
        constraints[name] = tuple(tuple(map(int, row)) for row in array.T)
    wiberg = _array(data, 'wiberg_value', torch.float64, (len(constraints['wiberg_index']),))
    declarations = {}
    for name, source in [('net_charge', graph['charge_source']), ('spin_mult', graph['spin_source'])]:
        is_present = bool(_array(data, name + '_present', torch.bool, (1,))[0])
        value = float(_array(data, name, torch.float64, (1,))[0])
        declarations[name] = (is_present, value if is_present else None, source if is_present else None)
    stereo_payload = _stereo(_wrapper(data, 'hotpot_stereo'), key_rows, edges)
    identifier = tuple(_unportable(graph['identifier']))
    from ._pyg_crystal import decode_cell
    cell = decode_cell(data, key_rows, bond_keys)
    if type(graph['molecule_present']) is not bool:
        raise ValueError('pyg_schema hotpot_graph.molecule_present: requires a boolean')
    if not graph['molecule_present'] and (n or m or properties or frames
            or declarations['net_charge'][0] or declarations['spin_mult'][0] or identifier[0]
            or observations['current_source'] or observations['historical'] is not None):
        raise ValueError('pyg_mapping standalone Crystal: molecular content requires molecule_present=True')
    record = _make_molecule_record(atom_values=values, atom_presence=masks, atom_sources=sources,
        atom_frac_valences=tuple((key_rows[item['atom_key']], item['value']) for item in fractional),
        bond_endpoints=tuple(edges), bond_values=bond_values, bond_kinds=tuple(KINDS[i] for i in kind_codes),
        bond_directions=tuple('atom1_to_atom2' if direction == 1 else 'atom2_to_atom1' if direction == -1
            else metadata[key]['other_direction'] for key, direction in zip(bond_keys, directions)),
        bond_sources=tuple(metadata[key]['source'] for key in bond_keys),
        bond_metadata=tuple(_unportable(metadata[key]['metadata']) for key in bond_keys),
        bond_aromatic=tuple(aromatic), charge=declarations['net_charge'], spin_mult=declarations['spin_mult'],
        cell=cell,
        stereo_payload=stereo_payload,
        source_frames=frames, conformer_results=(), frame_selection=None,
        angle_constraints=constraints['angle_index'], torsion_constraints=constraints['torsion_index'],
        identifier=identifier,
        pair_wiberg_bond_orders=tuple((*pair, float(value)) for pair, value in zip(constraints['wiberg_index'], wiberg)))
    properties['pyg_observations'] = {'namespace': 'imported_hotpot.molecule/2',
        'initial_atom_keys': list(atom_keys), 'initial_bond_keys': list(bond_keys),
        'observations': _plain(observations, 'observations'), 'frames': _plain(frame_values, 'frames')}
    return record, properties, graph['molecule_present']


def _foreign(data, field_map):
    import torch
    from torch_geometric.data import Data
    if type(field_map) is not dict or set(field_map) != {'fields', 'edge_mode', 'coordinate_unit'}:
        raise ValueError('pyg_schema field_map: requires fields, edge_mode and coordinate_unit')
    selectors = field_map['fields']
    supported = set(NODE_FIELDS) | set(EDGE_FIELDS) | set(INDEX_WIDTHS) | {
        'atom_key', 'pos', 'atom_present', 'atom_source', 'representation_available', 'atom_constraint',
        'wiberg_value', 'net_charge', 'spin_mult', 'net_charge_present', 'spin_mult_present',
        'cell_matrix', 'cell_present', 'bond_image', 'bond_image_present',
        'occupancy', 'occupancy_present', 'crystal_context', 'annotations'}
    if type(selectors) is not dict or 'z' not in selectors or set(selectors) - supported:
        raise ValueError('pyg_schema field_map.fields: required z or unsupported field')
    selected, used = {}, {}
    for name, selector in selectors.items():
        if type(selector) is str:
            attr, column = selector, None
        elif (type(selector) in (tuple, list) and len(selector) == 2
                and type(selector[0]) is str and type(selector[1]) is int):
            attr, column = selector
        else:
            raise ValueError(f'pyg_schema field_map.fields.{name}: requires attribute or (attribute, column)')
        if attr not in data:
            raise ValueError(f'pyg_mapping {name}: missing attribute {attr}')
        value = data[attr]
        if column is not None:
            if not isinstance(value, torch.Tensor) or value.ndim != 2 or not 0 <= column < value.shape[1]:
                raise ValueError(f'pyg_mapping {name}: unavailable column {attr}[:,{column}]')
            value = value[:, column]
        selected[name] = value
        used.setdefault(attr, []).append(column)
    n = data['num_nodes'] if 'num_nodes' in data else len(selected['z'])
    if 'edge_index' not in selected and data.get('edge_index') is not None and data.edge_index.numel():
        raise ValueError('pyg_mapping edge_index: edges require an explicit selector')
    edges = selected.get('edge_index', torch.empty((2, 0), dtype=torch.int64))
    if not isinstance(edges, torch.Tensor) or edges.dtype != torch.int64 or edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError('pyg_schema edge_index: requires int64 2xM')
    m = edges.shape[1]
    def supplied(name, default, dtype, shape):
        value = selected.get(name, default)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f'pyg_schema {name}: shape {shape} required')
        source = value.detach().cpu()
        result = source.to(dtype=dtype)
        if (name in selected and dtype in (torch.int64, torch.int8, torch.uint8, torch.bool)
                and source.reshape(-1).tolist() != result.reshape(-1).tolist()):
            raise ValueError(f'pyg_mapping {name}: conversion to {dtype} would change mapped values')
        return result.clone()
    normalized = Data(num_nodes=n, hotpot_schema=[SCHEMA])
    normalized.atom_key = supplied('atom_key', torch.arange(n), torch.int64, (n,))
    for name in NODE_FIELDS:
        normalized[name] = supplied(name, torch.zeros(n), torch.bool if name == 'aromatic' else torch.int64, (n,))
    normalized.pos = supplied('pos', torch.zeros(n, 3), torch.float64, (n, 3))
    factors = {'angstrom': 1., 'nanometer': 10., 'bohr': 0.529177210903}
    if ('pos' in selected or 'cell_matrix' in selected) and field_map['coordinate_unit'] not in factors:
        raise ValueError('pyg_schema coordinate_unit: angstrom, nanometer or bohr required')
    if 'pos' in selected:
        normalized.pos *= factors[field_map['coordinate_unit']]
    present = torch.tensor([[name in selected for name in (*NODE_FIELDS, 'pos', 'pos', 'pos')]] * n, dtype=torch.bool).reshape(n, 8)
    normalized.atom_present = supplied('atom_present', present, torch.bool, (n, 8))
    normalized.atom_source = supplied('atom_source', normalized.atom_present.to(torch.uint8), torch.uint8, (n, 8))
    normalized.representation_available = supplied('representation_available', normalized.atom_present[:, [2, 3]], torch.bool, (n, 2))
    normalized.atom_constraint = supplied('atom_constraint', torch.zeros(n, 3), torch.bool, (n, 3))
    normalized.edge_index = edges.detach().cpu().clone()
    normalized.bond_order = supplied('bond_order', torch.zeros(m), torch.float64, (m,))
    kind = torch.tensor([{0.: 1, 1.: 2, 1.5: 5, 2.: 3, 3.: 4}.get(value, 0)
                         if 'bond_order' in selected else 0 for value in normalized.bond_order.tolist()])
    for name, default, dtype in [('bond_kind', kind, torch.int64), ('bond_direction', torch.zeros(m), torch.int8),
        ('bond_aromatic', torch.zeros(m), torch.bool), ('bond_id', torch.zeros(m), torch.int64),
        ('bond_constraint', torch.zeros(m), torch.bool), ('bond_key', torch.arange(m), torch.int64)]:
        normalized[name] = supplied(name, default, dtype, (m,))
    mode = field_map['edge_mode']
    keep = None
    if mode == 'paired':
        groups, order = {}, []
        endpoints = normalized.edge_index.T.tolist()
        for i, pair in enumerate(endpoints):
            if pair[0] == pair[1]:
                raise ValueError('pyg_mapping paired edge_index: self bond')
            key = frozenset(pair)
            if key not in groups:
                groups[key] = [];order.append(key)
            groups[key].append(i)
        keep = []
        for key in order:
            indices = groups[key]
            if len(indices) != 2 or endpoints[indices[0]] != endpoints[indices[1]][::-1]:
                raise ValueError('pyg_mapping paired edge_index: requires exactly two opposite arcs')
            first, second = indices
            for name in EDGE_FIELDS:
                if name == 'bond_key' and name not in selected:
                    continue
                actual = selected.get(name, normalized[name])
                if name == 'bond_direction':
                    equal = actual[first].item() == -actual[second].item()
                else:
                    equal = torch.equal(actual[first], actual[second])
                if not equal:
                    raise ValueError(f'pyg_mapping paired {name}: opposite arcs disagree')
            keep.append(first)
        normalized.edge_index = normalized.edge_index[:, keep]
        for name in EDGE_FIELDS:
            normalized[name] = normalized[name][keep]
        if 'bond_key' not in selected:
            normalized.bond_key = torch.arange(len(keep))
    elif mode != 'single':
        raise ValueError('pyg_schema edge_mode: single or paired required')
    _check_edges(normalized.edge_index.T.tolist(), n)
    for name, width in INDEX_WIDTHS.items():
        if name != 'edge_index':
            raw = selected.get(name, torch.empty((width, 0), dtype=torch.int64))
            normalized[name] = raw.detach().cpu().clone()
    normalized.wiberg_value = supplied('wiberg_value', torch.zeros(normalized.wiberg_index.shape[1]), torch.float64,
                                      (normalized.wiberg_index.shape[1],))
    for name in ('net_charge', 'spin_mult'):
        normalized[name] = supplied(name, torch.zeros(1), torch.float64, (1,))
        normalized[name + '_present'] = supplied(name + '_present', torch.tensor([name in selected]), torch.bool, (1,))
    from ._pyg_crystal import foreign_cell
    periodic = {}
    for name, dtype, shape in [('cell_matrix', torch.float64, (1, 3, 3)),
            ('cell_present', torch.bool, (1,)), ('bond_image', torch.int64, (m, 3)),
            ('bond_image_present', torch.bool, (m,)), ('occupancy', torch.float64, (n,)),
            ('occupancy_present', torch.bool, (n,))]:
        if name in selected:
            periodic[name] = supplied(name, None, dtype, shape)
    if 'cell_matrix' in periodic:
        periodic['cell_matrix'] *= factors[field_map['coordinate_unit']]
    if keep is not None:
        for first, second in (groups[key] for key in order):
            known = periodic.get('bond_image_present')
            if known is not None and known[first].item() != known[second].item():
                raise ValueError('pyg_mapping paired bond_image_present: opposite arcs disagree')
            if 'bond_image' in periodic and (known is None or known[first].item()):
                if not torch.equal(periodic['bond_image'][first], -periodic['bond_image'][second]):
                    raise ValueError('pyg_mapping paired bond_image: opposite arcs disagree')
        for name in ('bond_image', 'bond_image_present'):
            if name in periodic:
                periodic[name] = periodic[name][keep]
    if 'crystal_context' in selected:
        periodic['crystal_context'] = selected['crystal_context']
    for name, value in foreign_cell(periodic, n, normalized.edge_index.shape[1]).items():
        normalized[name] = value
    context = normalized.hotpot_crystal[0]
    if context is not None:
        keys = normalized.atom_key.tolist()
        context['relations']['equivalent_images'] = [[[keys[row], symbol, image] for row, symbol, image in entry]
                                                    for entry in context['relations']['equivalent_images']]
        context['current_atom_keys'] = keys
        context['current_bond_keys'] = normalized.bond_key.tolist()
    annotations = _plain(selected.get('annotations', {}), 'annotations')
    normalized.hotpot_graph = [{'molecule_present': True, 'charge_source': 'pyg_supplied' if 'net_charge' in selected else None,
        'spin_source': 'pyg_supplied' if 'spin_mult' in selected else None,
        'identifier': _portable((False, None), 'identifier'), 'annotations': _tensors(_portable(annotations, 'annotations'), torch)}]
    normalized.hotpot_bonds = [[{'bond_key': key, 'source': None, 'other_direction': None,
        'metadata': _portable({}, 'bond.metadata')} for key in normalized.bond_key.tolist()]]
    normalized.hotpot_stereo = [None]
    normalized.hotpot_fractional_valence = [[]]
    normalized.hotpot_frames = [{'sources': [], 'generated': [], 'selected': None}]
    normalized.hotpot_observations = [{'current_source': [], 'historical': None}]
    for name in data.keys():
        if name == 'num_nodes':
            continue
        columns = used.get(name)
        if columns is None:
            warnings.warn(f'pyg_unmapped_field {name}', RuntimeWarning, stacklevel=3)
        elif None not in columns and isinstance(data[name], torch.Tensor) and data[name].ndim == 2:
            for column in range(data[name].shape[1]):
                if column not in columns:
                    warnings.warn(f'pyg_unmapped_field {name}[:,{column}]', RuntimeWarning, stacklevel=3)
    return normalized


def decode(data, field_map):
    from torch_geometric.data import Data, Batch
    if not isinstance(data, Data):
        raise TypeError('pyg_schema data: a PyG Data object is required')
    if isinstance(data, Batch):
        if data.num_graphs != 1:
            raise ValueError('pyg_mapping Batch: split multiple graphs with to_data_list()')
        data = data.to_data_list()[0]
    if data.get('hotpot_schema') != [SCHEMA]:
        if field_map is None:
            raise ValueError('pyg_schema hotpot_schema: foreign Data requires field_map')
        data = _foreign(data, field_map)
    elif field_map is not None:
        raise ValueError('pyg_schema field_map: exact schema already declares its fields')
    return _decode_schema(data)
