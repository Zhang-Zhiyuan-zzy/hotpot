"""Pure explicit identification of a fully generated immutable P1 record."""

from collections import Counter
from dataclasses import replace

import numpy as np

from ...structure._fields import ATOM_FIELDS, FIELD_INDEX, FIELD_PRESENCE_BITS
from ...structure.stereo import StereoAnchorKind, StereoCarrierKind
from .stereochemistry.algebra import (
    _parse_level2_descriptor,
    _remap_stereo_payload,
    _render_level2_descriptor,
    _transform_level2_descriptor,
    _transform_supported_record,
    _validate_gate2_double_bond_carriers,
    _validate_stereo_anchor,
)


__all__ = ()


def _unsupported(reason):
    raise ValueError(f"CRYSTAL-UNSUPPORTED-PROJECTION: {reason}")


def _unique(values):
    return tuple(dict.fromkeys(values))


def _direction_key(direction, reversed_endpoints):
    if direction in ("1->2", "2->1"):
        return ("known", ("2->1" if direction == "1->2" else "1->2")
                if reversed_endpoints else direction)
    if direction is None:
        return ("none",)
    # Opaque meanings match only in the same explicitly mapped orientation.
    return ("opaque", reversed_endpoints, direction)


def _display_key(fact, mapping):
    endpoints = tuple(mapping[row] for row in fact.atom_indices)
    if fact.display_direction in (None, "NONE"):
        return (fact.display_direction,)
    return (fact.display_direction, endpoints)


def _current_atom_facts(payload, count):
    values = [[set() for _ in range(5)] for _ in range(count)]
    for fact in payload.atom_facts:
        hydrogen = fact.current_hydrogens
        current = (fact.isotope, fact.radical_electrons,
                   None if hydrogen is None else hydrogen.implicit_hydrogens,
                   None if hydrogen is None else hydrogen.bracket_explicit_hydrogens,
                   None if hydrogen is None else hydrogen.no_implicit)
        for column, value in zip(values[fact.atom], current):
            if value is not None:
                column.add(value)
    return values


def _stereo_support(record):
    return set(record.anchor.atom_indices) | {
        carrier.atom for carrier in record.carriers
        if carrier.kind is StereoCarrierKind.ATOM
    }


def _carrier_key(carrier):
    return carrier.kind.value, -1 if carrier.atom is None else carrier.atom


def _stereo_key(record):
    """Normalize supported ordering through the existing descriptor algebra."""
    parsed = _parse_level2_descriptor(record.descriptor)
    if parsed.issues:
        if record.descriptor.startswith("hotpot.stereo/2/"):
            _unsupported("identified stereo has an unsupported Level-2 descriptor")
        return ("opaque", record.anchor, record.carriers, record.descriptor,
                record.group, record.source_namespace, record.source_code)
    anchor = record.anchor
    indices = anchor.atom_indices
    canonical_indices = min(indices, indices[::-1])
    canonical_anchor = replace(anchor, atom_indices=canonical_indices)
    if parsed.descriptor.family.value in ("DB", "AX"):
        first, second = record.carriers[:2], record.carriers[2:]
        if canonical_indices != indices:
            first, second = second, first
        carriers = tuple(sorted(first, key=_carrier_key)) + tuple(sorted(second, key=_carrier_key))
    else:
        carriers = tuple(sorted(record.carriers, key=_carrier_key))
    if parsed.descriptor.family.value in ("DB", "AX"):
        issues = (*_validate_stereo_anchor(anchor, family=parsed.descriptor.family),
                  *_validate_gate2_double_bond_carriers(record.carriers))
        if issues:
            _unsupported("identified endpoint stereo has no supported projection: "
                         + "; ".join(issue.message for issue in issues))
        target = replace(record, anchor=canonical_anchor, carriers=carriers)
        transformed = _transform_supported_record(parsed.descriptor, record, target)
    else:
        transformed = _transform_level2_descriptor(
            parsed.descriptor, record.carriers, carriers,
            source_anchor=anchor, reordered_anchor=canonical_anchor,
        )
    if transformed.issues:
        _unsupported("identified stereo ordering has no supported projection: "
                     + "; ".join(issue.message for issue in transformed.issues))
    return ("level2", canonical_anchor, carriers,
            _render_level2_descriptor(transformed.descriptor), record.group)


def _validate_stereo(payload, mapped, mapping, classes, affected):
    """Compare current facts and mapped meaning, retaining every occurrence."""
    atom_facts = _current_atom_facts(payload, len(mapping))
    for members in classes:
        if len(members) == 1:
            continue
        for row in members:
            if any(len(column) > 1 for column in atom_facts[row]):
                _unsupported(f"identified Atom {row} has contradictory current isotope/radical/hydrogen facts")
            if atom_facts[row] != atom_facts[members[0]]:
                _unsupported(f"identified Atoms {members} disagree in current isotope/radical/hydrogen facts")
    signatures = [set() for _ in mapping]
    anchor_meanings = {}
    for original, projected in zip(payload.records, mapped.records):
        support = _stereo_support(original)
        if not support & affected:
            continue
        if len({mapping[row] for row in support}) != len(support):
            _unsupported("identification collapses distinct stereo references")
        key = _stereo_key(projected)
        indices = projected.anchor.atom_indices
        identity = projected.anchor.kind, min(indices, indices[::-1])
        if identity in anchor_meanings and anchor_meanings[identity] != key:
            _unsupported(f"identified stereo at {identity[1]} has incompatible orientation or meaning")
        anchor_meanings[identity] = key
        for row in support:
            signatures[row].add(key)
    for members in classes:
        if len(members) > 1 and any(signatures[row] != signatures[members[0]] for row in members[1:]):
            _unsupported(f"identified Atoms {members} have incomplete or incompatible mapped stereo")


def _mapped_constraints(entries, mapping, classes, label):
    signatures = [set() for _ in mapping]
    projected = []
    for entry in entries:
        mapped = tuple(mapping[row] for row in entry)
        if len(set(mapped)) != len(mapped):
            _unsupported(f"identification collapses distinct {label} constraint sites")
        canonical = min(mapped, mapped[::-1])
        for row in entry:
            signatures[row].add(canonical)
        projected.append(mapped)
    for members in classes:
        if len(members) > 1 and any(signatures[row] != signatures[members[0]] for row in members[1:]):
            _unsupported(f"identified Atoms {members} disagree in mapped {label} constraints")
    return _unique(projected)


def _identify_generated_record(record, classes):
    """Identify only supplied complete classes after chemistry-map validation.

    Inputs are a prevalidated full generated P1 record with frozen Bond metadata,
    complete origin maps and no live results/frames, and a stable sorted partition.
    The caller has already closed explicit equivalences under the group action.
    """
    if all(len(members) == 1 for members in classes):
        return record
    mapping = {row: target for target, members in enumerate(classes) for row in members}
    chosen = tuple(members[0] for members in classes)
    affected = {row for members in classes if len(members) > 1 for row in members}
    values = np.frombuffer(record.atom_values, dtype="<f8").reshape(record.atom_count, len(ATOM_FIELDS))
    presence = np.frombuffer(record.atom_presence, dtype="<u4")
    sources = np.frombuffer(record.atom_sources, dtype="u1").reshape(values.shape)
    xyz = tuple(FIELD_INDEX[name] for name in ("x", "y", "z"))
    columns = tuple(column for column in range(len(ATOM_FIELDS)) if column not in xyz)
    scalar_mask = sum(FIELD_PRESENCE_BITS[column] for column in columns)
    fractional_valences = dict(record.atom_frac_valences)
    equivalents, occupancies, disorder = record.cell[3]
    for members in classes:
        first = members[0]
        for row in members[1:]:
            if np.linalg.norm(values[row, xyz] - values[first, xyz]) > 1e-10:
                _unsupported(f"identified Atoms {members} exceed the 1e-10 Angstrom coordinate residual")
            if int(presence[row]) & scalar_mask != int(presence[first]) & scalar_mask:
                _unsupported(f"identified Atoms {members} disagree in supplied scalar presence")
            supplied = tuple(column for column in columns if int(presence[first]) & FIELD_PRESENCE_BITS[column])
            if not np.array_equal(values[row, supplied], values[first, supplied]):
                _unsupported(f"identified Atoms {members} disagree in supplied scalar facts or axis constraints")
            if ((row in fractional_valences, fractional_valences.get(row))
                    != (first in fractional_valences, fractional_valences.get(first))):
                _unsupported(f"identified Atoms {members} disagree in fractional valence")
            if (occupancies[row], disorder[row]) != (occupancies[first], disorder[first]):
                _unsupported(f"identified Atoms {members} disagree in occupancy/disorder")

    payload = record.stereo_payload
    mapped_payload = None if payload is None else _remap_stereo_payload(payload, mapping)
    if payload is not None:
        _validate_stereo(payload, mapped_payload, mapping, classes, affected)
    display = {}
    if payload is not None:
        for fact in payload.bond_facts:
            display.setdefault(frozenset(fact.atom_indices), set()).add(_display_key(fact, mapping))

    bond_values = np.frombuffer(record.bond_values, dtype="<f8").reshape(-1, 3)
    edge_groups = {}
    incidence = [Counter() for _ in mapping]
    for edge, (first, second) in enumerate(record.bond_endpoints):
        new_first, new_second = mapping[first], mapping[second]
        if new_first == new_second:
            _unsupported(f"identification creates unsupported self quotient Bond {edge}")
        image = record.cell[4][edge]
        if image is None and (first in affected or second in affected):
            _unsupported(f"required image is unknown for Bond {edge} incident to identified Atoms")
        reversed_endpoints = new_first > new_second
        canonical_image = (None if image is None else
                           tuple(-value for value in image) if reversed_endpoints else image)
        endpoints = tuple(sorted((new_first, new_second)))
        meaning = (tuple(bond_values[edge]), record.bond_kinds[edge],
                   record.bond_aromatic[edge], _direction_key(record.bond_directions[edge], reversed_endpoints),
                   record.bond_metadata[edge], frozenset(display.get(frozenset((first, second)), ())))
        signature = endpoints, canonical_image, meaning
        incidence[first][signature] += 1
        incidence[second][signature] += 1
        edge_groups.setdefault(endpoints, []).append((edge, canonical_image, meaning))
    for members in classes:
        if len(members) > 1 and any(incidence[row] != incidence[members[0]] for row in members[1:]):
            _unsupported(f"identified Atoms {members} have incomplete or incompatible mapped incident edges")

    bond_rows = []
    bond_classes = []
    for entries in edge_groups.values():
        first, image, meaning = entries[0]
        original_incidence = Counter(row for edge, _, _ in entries for row in record.bond_endpoints[edge])
        if any(count > 1 for count in original_incidence.values()):
            _unsupported("identification creates parallel quotient Bonds at one actual site alias")
        original_pairs = set()
        for edge, other_image, other_meaning in entries:
            pair = frozenset(record.bond_endpoints[edge])
            if pair in original_pairs or other_image != image or other_meaning != meaning:
                _unsupported(f"identification creates unsupported parallel or incompatible quotient Bonds at {mapping[record.bond_endpoints[first][0]], mapping[record.bond_endpoints[first][1]]}")
            original_pairs.add(pair)
        bond_rows.append(first)
        bond_classes.append(tuple(edge for edge, _, _ in entries))
    # Groups were inserted in source edge order, so first occurrences stay stable.
    endpoints = tuple(tuple(mapping[row] for row in record.bond_endpoints[edge]) for edge in bond_rows)
    angles = _mapped_constraints(record.angle_constraints, mapping, classes, "angle")
    torsions = _mapped_constraints(record.torsion_constraints, mapping, classes, "torsion")

    charge = record.charge
    if charge[0]:
        charge_column = FIELD_INDEX["formal_charge"]
        charge_bit = FIELD_PRESENCE_BITS[charge_column]
        if not np.all((presence & charge_bit) != 0) or sum(int(value) for value in values[:, charge_column]) != charge[1]:
            _unsupported("changed total charge requires complete supplied formal charges agreeing with the premerge declared total")
        charge = True, sum(int(values[row, charge_column]) for row in chosen), charge[2]

    def origins(groups, rows):
        merged = []
        for members in groups:
            entries = tuple(rows[row] for row in members)
            if all(entry is None for entry in entries):
                merged.append(None)
            elif any(entry is None for entry in entries):
                _unsupported("identified origin aliases mix known and unknown bindings and have no complete projection")
            else:
                merged.append(_unique(key for entry in entries for key in entry))
        return tuple(merged)

    active_equivalents = _unique(
        keys for entry in equivalents
        if len(keys := _unique((mapping[row], operation, image) for row, operation, image in entry)) > 1
    )
    cell = (*record.cell[:3], (active_equivalents, tuple(occupancies[row] for row in chosen),
                              tuple(disorder[row] for row in chosen)),
            tuple(record.cell[4][edge] for edge in bond_rows), *record.cell[5:7],
            origins(classes, record.cell[7]), origins(bond_classes, record.cell[8]))
    return replace(
        record, atom_count=len(chosen), atom_values=values[list(chosen)].tobytes(),
        atom_presence=presence[list(chosen)].tobytes(), atom_sources=sources[list(chosen)].tobytes(),
        atom_frac_valences=tuple((mapping[row], value) for row, value in record.atom_frac_valences if row in chosen),
        bond_endpoints=endpoints, bond_values=bond_values[bond_rows].tobytes(),
        bond_kinds=tuple(record.bond_kinds[edge] for edge in bond_rows),
        bond_directions=tuple(record.bond_directions[edge] for edge in bond_rows),
        bond_sources=tuple(record.bond_sources[edge] for edge in bond_rows),
        bond_metadata=tuple(record.bond_metadata[edge] for edge in bond_rows),
        bond_aromatic=tuple(record.bond_aromatic[edge] for edge in bond_rows),
        charge=charge, spin_mult=(False, None, None), cell=cell, stereo_payload=mapped_payload,
        angle_constraints=angles, torsion_constraints=torsions,
    )
