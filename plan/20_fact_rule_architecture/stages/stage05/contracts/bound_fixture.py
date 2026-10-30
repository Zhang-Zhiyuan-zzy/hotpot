"""Rebuild the reporter's B20 factual input through the one current builder.

This is a bounded Stage5 fixture recipe, not a production chemistry adapter.
Modern presence/provenance below describes this candidate import; B20 had no
such metadata. Only the recorded reset-bond-order recipe establishes an old
eager-inferred H count. Other observed counts enter as supplied raw-array facts.
"""

from dataclasses import replace

import numpy as np

from hotpot import Molecule
from hotpot.cheminfo.structure._fields import (
    ATOM_SOURCE_CODES, FACT_FIELDS, FIELD_INDEX,
)


def build_baseline_case(case):
    prepared = case['prepared_facts']
    count = prepared['atom_count']
    values = bytes.fromhex(prepared['atom_values_float64_hex'])
    entries = prepared['bonds']
    supplied = tuple(FIELD_INDEX[field] for field in FACT_FIELDS)
    presence = np.full(count, sum(1 << index for index in supplied), dtype='<u4')
    sources = np.zeros((count, 19), dtype='u1')
    sources[:, supplied] = ATOM_SOURCE_CODES['legacy_array']
    inferred = case['case']['reset_bond_orders']
    if inferred:
        sources[:, FIELD_INDEX['implicit_hydrogens']] = ATOM_SOURCE_CODES['implicit_hydrogens']
    empty = Molecule()
    record, _ = empty._capture_factual_record()
    metadata = tuple(entry['attributes']['bond_source_metadata'] for entry in entries)
    record = replace(record, atom_count=count, atom_values=values,
        atom_presence=presence.tobytes(), atom_sources=sources.tobytes(),
        bond_endpoints=tuple(tuple(entry['ends']) for entry in entries),
        bond_values=bytes.fromhex(prepared['bond_values_float64_hex']),
        bond_partitions=(0,) * len(entries),
        bond_kinds=tuple(entry['attributes']['bond_kind'] for entry in entries),
        bond_directions=tuple(entry['attributes']['bond_direction'] for entry in entries),
        bond_sources=tuple(entry['attributes']['bond_source'] for entry in entries),
        bond_metadata=(('none',),) * len(entries),
        bond_aromatic=tuple(entry['is_aromatic'] for entry in entries),
        charge=(True, prepared['charge'], 'legacy_array'))
    molecule = Molecule._build_molecule(record, bond_metadata=metadata)
    classification = {
        'baseline_metadata': prepared['atom_presence_sources'],
        'candidate_supplied_fields': list(FACT_FIELDS),
        'candidate_default_producer': 'legacy_array',
        'candidate_hydrogen_producer': 'implicit_hydrogens' if inferred else 'legacy_array',
        'hydrogen_recipe_evidence': (
            'recorded reset_bond_orders=True: old eager Bond setters produced the all-single pre-assignment H counts'
            if inferred else 'supplied observed B20 raw H counts; no B20 provenance claim'),
        'retired_P4_slots': 'not stored; current projections compute electron configuration and valence',
    }
    return molecule, classification
