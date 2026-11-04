"""Closed pure Level-1 stereo projection shared by structural formats."""

from dataclasses import fields
from enum import Enum

from ._builder import _stereo
from ._crystal_codec import _fields
from .stereo import (
    AtomTransportRecord, BondTransportRecord, HydrogenTransportState,
    StereoAnchor, StereoAnchorKind, StereoCarrier, StereoCarrierKind,
    StereoGroupRef, StereoGroupKind, StereoPayload, StereoRecord,
)


def _stereo_to_data(payload):
    def structured(value):
        if isinstance(value, Enum):
            return value.value
        if hasattr(type(value), '__dataclass_fields__'):
            return {field.name: structured(getattr(value, field.name)) for field in fields(value)}
        if type(value) is tuple:
            return [structured(item) for item in value]
        return value
    return structured(_stereo(payload))


def _stereo_from_data(value, keys, edges):
    if value is None:
        return None
    version, records, atoms, bonds = _fields(value, ('schema_version', 'records', 'atom_facts', 'bond_facts'))
    edge_set = {frozenset(edge) for edge in edges}

    def row(key):
        if key not in keys:
            raise ValueError(f'stereo_reference atom_key={key}')
        return keys[key]

    def hydrogen(item):
        if item is None:
            return None
        return HydrogenTransportState(*_fields(item,
            ('implicit_hydrogens', 'bracket_explicit_hydrogens', 'no_implicit')))

    decoded = []
    for item in records:
        anchor, carriers, descriptor, group, namespace, code = _fields(item,
            ('anchor', 'carriers', 'descriptor', 'group', 'source_namespace', 'source_code'))
        kind, members = _fields(anchor, ('kind', 'atom_indices'))
        kind, members = StereoAnchorKind(kind), tuple(row(key) for key in members)
        pairs = ((members,) if kind is StereoAnchorKind.BOND else
                 tuple(zip(members, members[1:])) if kind is StereoAnchorKind.AXIS else ())
        if any(frozenset(pair) not in edge_set for pair in pairs):
            raise ValueError(f'stereo_reference anchor={members}: missing bond')
        decoded_carriers = []
        for entry in carriers:
            carrier_kind, atom = _fields(entry, ('kind', 'atom'))
            carrier_kind = StereoCarrierKind(carrier_kind)
            decoded_carriers.append(StereoCarrier(carrier_kind,
                row(atom) if carrier_kind is StereoCarrierKind.ATOM else atom))
        if group is not None:
            scope, group_kind, identifier = _fields(group, ('scope', 'kind', 'id'))
            group = StereoGroupRef(scope, StereoGroupKind(group_kind), identifier)
        decoded.append(StereoRecord(StereoAnchor(kind, members), tuple(decoded_carriers), descriptor,
                                   group, namespace, code))
    atom_facts = []
    for item in atoms:
        atom, isotope, radical, current, source, namespace = _fields(item,
            ('atom', 'isotope', 'radical_electrons', 'current_hydrogens', 'source_hydrogens', 'source_namespace'))
        atom_facts.append(AtomTransportRecord(row(atom), isotope, radical, hydrogen(current), hydrogen(source), namespace))
    bond_facts = []
    for item in bonds:
        members, direction = _fields(item, ('atom_indices', 'display_direction'))
        members = tuple(row(key) for key in members)
        if frozenset(members) not in edge_set:
            raise ValueError(f'stereo_reference display={members}: missing bond')
        bond_facts.append(BondTransportRecord(members, direction))
    return _stereo(StereoPayload(version, tuple(decoded), tuple(atom_facts), tuple(bond_facts)))
