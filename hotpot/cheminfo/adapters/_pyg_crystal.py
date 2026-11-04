"""Tensor fields and exact context for the sole current Crystal record."""

from copy import deepcopy

import numpy as np


CELL_FIELDS = ('cell_matrix', 'cell_present', 'bond_image', 'bond_image_present',
               'occupancy', 'occupancy_present', 'space_group_R', 'space_group_t',
               'hotpot_crystal')


def encode_cell(cell, atom_count, bond_count):
    import torch
    from ..structure._crystal_codec import _crystal_to_data

    present = cell is not None
    data = None if cell is None else _crystal_to_data(cell)
    matrix = np.zeros((3, 3)) if cell is None else cell[1]
    images = (None,) * bond_count if cell is None else cell[4]
    occupancies = (None,) * atom_count if cell is None else cell[3][1]
    operations = () if cell is None or cell[2] is None else cell[2].symm_units
    if data is not None:
        del data['matrix'], data['bond_images'], data['relations']['occupancies']
        data['current_atom_keys'] = list(range(atom_count))
        data['current_bond_keys'] = list(range(bond_count))
    return {
        'cell_matrix': torch.tensor(np.asarray(matrix)[None, :, :], dtype=torch.float64),
        'cell_present': torch.tensor([present], dtype=torch.bool),
        'bond_image': torch.tensor([image if image is not None else (0, 0, 0) for image in images],
                                    dtype=torch.int64).reshape(bond_count, 3),
        'bond_image_present': torch.tensor([image is not None for image in images], dtype=torch.bool),
        'occupancy': torch.tensor([value if value is not None else 0. for value in occupancies], dtype=torch.float64),
        'occupancy_present': torch.tensor([value is not None for value in occupancies], dtype=torch.bool),
        'space_group_R': torch.tensor([value.R for value in operations], dtype=torch.int64).reshape(-1, 3, 3),
        'space_group_t': torch.tensor([[float(item) for item in value.t] for value in operations],
                                      dtype=torch.float64).reshape(-1, 3),
        'hotpot_crystal': [data],
    }


def _join(values, original, current, name):
    if len(original) != len(values) or len(set(original)) != len(original) or set(original) != set(current):
        raise ValueError(f'pyg_mapping crystal.{name}: current transport keys differ from context')
    entries = dict(zip(original, values))
    return [entries[key] for key in current]


def decode_cell(data, key_rows, bond_keys):
    import torch
    from ._pyg_molecule import _array, _mapping, _wrapper
    from ..structure._crystal_codec import _crystal_from_data

    n, m = len(key_rows), len(bond_keys)
    matrix = _array(data, 'cell_matrix', torch.float64, (1, 3, 3))[0]
    present = bool(_array(data, 'cell_present', torch.bool, (1,))[0])
    images = _array(data, 'bond_image', torch.int64, (m, 3))
    image_present = _array(data, 'bond_image_present', torch.bool, (m,))
    occupancy = _array(data, 'occupancy', torch.float64, (n,))
    occupancy_present = _array(data, 'occupancy_present', torch.bool, (n,))
    metadata = _wrapper(data, 'hotpot_crystal')
    rotations, translations = data.get('space_group_R'), data.get('space_group_t')
    count = rotations.shape[0] if isinstance(rotations, torch.Tensor) and rotations.ndim == 3 else -1
    rotations = _array(data, 'space_group_R', torch.int64, (count, 3, 3))
    translations = _array(data, 'space_group_t', torch.float64, (count, 3))
    if not present:
        if metadata is not None or image_present.any() or occupancy_present.any() or count:
            raise ValueError('pyg_mapping crystal: periodic facts require an explicit cell')
        return None
    if type(metadata) is not dict:
        raise ValueError('pyg_schema hotpot_crystal: present cell requires its structured context')
    context = _mapping(deepcopy(metadata),
        ('current_atom_keys', 'current_bond_keys', 'relations', 'image_map', 'bond_origins'), 'hotpot_crystal')
    atom_keys, original_bonds = context.pop('current_atom_keys'), context.pop('current_bond_keys')
    relations = _mapping(context['relations'], ('disorder', 'equivalent_images'), 'hotpot_crystal.relations')
    for entry in relations['equivalent_images']:
        for key, _, _ in entry:
            if key not in key_rows:
                raise ValueError(f'pyg_mapping crystal.equivalent_images: missing atom_key={key}')
    relations['disorder'] = _join(relations['disorder'], atom_keys, key_rows, 'disorder')
    relations['equivalent_images'] = [[[key_rows[row], symbol, image] for row, symbol, image in entry]
                                      for entry in relations['equivalent_images']]
    if context['image_map'] is not None:
        context['image_map'] = _join(context['image_map'], atom_keys, key_rows, 'image_map')
    elif len(atom_keys) != n or set(atom_keys) != set(key_rows):
        raise ValueError('pyg_mapping crystal.current_atom_keys: current membership differs')
    if context['bond_origins'] is not None:
        context['bond_origins'] = _join(context['bond_origins'], original_bonds, bond_keys, 'bond_origins')
    elif len(original_bonds) != m or set(original_bonds) != set(bond_keys):
        raise ValueError('pyg_mapping crystal.current_bond_keys: current membership differs')
    context['matrix'] = matrix.tolist()
    context['bond_images'] = [image.tolist() if known else None for image, known in zip(images, image_present)]
    relations['occupancies'] = [float(value) if known else None for value, known in zip(occupancy, occupancy_present)]
    cell = _crystal_from_data(context, n, m)
    operations = () if cell[2] is None else cell[2].symm_units
    expected_r = np.asarray([value.R for value in operations], dtype=np.int64).reshape(-1, 3, 3)
    expected_t = np.asarray([[float(item) for item in value.t] for value in operations], dtype=np.float64).reshape(-1, 3)
    if not np.array_equal(rotations, expected_r) or not np.array_equal(translations, expected_t):
        raise ValueError('pyg_mapping crystal.space_group: numeric views disagree with exact group context')
    return cell


def foreign_cell(selected, atom_count, bond_count):
    """Normalize only explicitly selected foreign periodic facts, without inference."""
    import torch
    from ..structure._crystal import _canonical_crystal_record
    from ..structure._crystal_codec import _crystal_from_data

    def array(name, dtype, shape):
        value = selected[name]
        if not isinstance(value, torch.Tensor) or value.dtype != dtype or tuple(value.shape) != shape:
            raise ValueError(f'pyg_schema field_map.{name}: requires {dtype} {shape}')
        return value.detach().cpu().numpy()

    has_context = 'crystal_context' in selected
    context = None
    if has_context:
        context = _crystal_from_data(selected['crystal_context'], atom_count, bond_count)
    matrix = array('cell_matrix', torch.float64, (1, 3, 3))[0] if 'cell_matrix' in selected else None
    is_present = (bool(array('cell_present', torch.bool, (1,))[0]) if 'cell_present' in selected
                  else has_context or matrix is not None)
    if not is_present:
        if has_context or any(name in selected for name in ('cell_matrix', 'bond_image', 'bond_image_present',
                                                            'occupancy', 'occupancy_present')):
            raise ValueError('pyg_mapping field_map: periodic fields conflict with absent cell')
        return encode_cell(None, atom_count, bond_count)
    if matrix is None and context is None:
        raise ValueError('pyg_mapping field_map.cell_matrix: present cell needs its oriented basis')
    images = None
    if 'bond_image' in selected:
        values = array('bond_image', torch.int64, (bond_count, 3))
        known = (array('bond_image_present', torch.bool, (bond_count,)) if 'bond_image_present' in selected
                 else np.ones(bond_count, dtype=bool))
        images = tuple(tuple(map(int, image)) if present else None for image, present in zip(values, known))
    elif 'bond_image_present' in selected:
        raise ValueError('pyg_mapping field_map.bond_image_present: image values are missing')
    occupancy = None
    if 'occupancy' in selected:
        values = array('occupancy', torch.float64, (atom_count,))
        known = (array('occupancy_present', torch.bool, (atom_count,)) if 'occupancy_present' in selected
                 else np.ones(atom_count, dtype=bool))
        occupancy = tuple(float(value) if present else None for value, present in zip(values, known))
    elif 'occupancy_present' in selected:
        raise ValueError('pyg_mapping field_map.occupancy_present: values are missing')
    if context is not None:
        if ((matrix is not None and not np.array_equal(matrix, context[1]))
                or (images is not None and images != context[4])
                or (occupancy is not None and occupancy != context[3][1])):
            raise ValueError('pyg_mapping field_map.crystal_context: selected fields contradict exact context')
        return encode_cell(context, atom_count, bond_count)
    context = _canonical_crystal_record(('hotpot.crystal/1', matrix, None,
        ((), (None,) * atom_count if occupancy is None else occupancy, (None,) * atom_count),
        (None,) * bond_count if images is None else images, 'independent', None, None, None), atom_count, bond_count)
    return encode_cell(context, atom_count, bond_count)
