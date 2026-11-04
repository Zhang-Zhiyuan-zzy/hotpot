"""Field-only packing for the existing native whole-graph matcher."""

from collections import Counter

import numpy as np

from ...structure._fields import ATOM_FIELDS, FIELD_INDEX


__all__ = ()


def _matching_atoms(record):
    values = np.frombuffer(record.atom_values, dtype='<f8').reshape(-1, len(ATOM_FIELDS))
    degree = Counter(row for edge in record.bond_endpoints for row in edge)
    return tuple({'atomic_number': float(values[row, FIELD_INDEX['atomic_number']]),
                  '_crystal_degree': degree[row], '_crystal_row': row}
                 for row in range(record.atom_count))


def _matching_bonds(record):
    values = np.frombuffer(record.bond_values, dtype='<f8').reshape(-1, 3)
    return tuple({'fields': {'bond_order': float(value[0]), '_crystal_aromatic': aromatic},
                  'bond_kind': kind, 'is_aromatic': aromatic, 'object': None}
                 for value, kind, aromatic in zip(values, record.bond_kinds, record.bond_aromatic))


def _matching_graph(record):
    ordered = tuple(sorted((tuple(sorted(edge)), row) for row, edge in enumerate(record.bond_endpoints)))
    bonds = _matching_bonds(record)
    return tuple(edge for edge, _ in ordered), tuple(bonds[row] for _, row in ordered)


def _pack_matching_target(record):
    edges, bonds = _matching_graph(record)
    return {'keys': tuple(range(record.atom_count)), 'edges': edges,
            'atoms': tuple({'fields': fields, 'full': {}, 'ligand': {}, 'object': None}
                           for fields in _matching_atoms(record)),
            'bonds': bonds}


def _pack_matching_query(record, domains):
    edges, records = _matching_graph(record)
    nodes = tuple({'preflight': (), 'leaves': (
        ('native_set', 'atomic_number', (fields['atomic_number'],)),
        ('native_set', '_crystal_degree', (fields['_crystal_degree'],)),
        ('native_set', '_crystal_row', tuple(domain)))}
        for fields, domain in zip(_matching_atoms(record), domains))
    bonds = tuple({'preflight': (), 'leaves': tuple(
        ('native_set', name, (value,)) for name, value in item['fields'].items()),
        'exclude_aromatic': not item['is_aromatic'], 'bond_kind': item['bond_kind']}
        for item in records)
    return {'keys': tuple(range(record.atom_count)), 'edges': edges,
            'nodes': nodes, 'bonds': bonds}
