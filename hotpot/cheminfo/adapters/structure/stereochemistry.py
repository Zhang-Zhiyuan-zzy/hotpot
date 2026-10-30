"""Field-selective stereo capture and packing; callers hold the owner scope."""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from struct import pack

from ...structure._identity import _capture_source, _check_source
from ...structure.snapshots import _StereoSnapshot

__all__ = ()


def _capture_stereo_snapshot(mol):
    """Capture once and keep transient identity referents outside the snapshot."""

    atoms = tuple(mol._atoms)
    partitions = tuple(tuple(bonds) for bonds in (
        mol._bonds, mol._hided_metal_bonds, mol._hided_covalent_bonds,
    ))
    endpoints = tuple((bond.atom1, bond.atom2) for bonds in partitions for bond in bonds)
    snapshot = _StereoSnapshot(
        payload=mol._stereo_payload,
        atom_keys=tuple(id(atom) for atom in atoms),
        bonds=tuple(
            (id(first), id(second), float(bond.bond_order), bond.bond_kind.value,
             bond.bond_direction)
            for bond, (first, second) in zip(
                (bond for bonds in partitions for bond in bonds), endpoints,
            )
        ),
    )
    return snapshot, _capture_source(mol, ()), (atoms, partitions, endpoints)


def _same_stereo_raw_value(first, second):
    """Compare consumed raw schema/value/order, including float sign and type."""

    if type(first) is not type(second):
        return False
    if isinstance(first, float):
        return pack("!d", first) == pack("!d", second)
    if isinstance(first, tuple):
        return len(first) == len(second) and all(
            _same_stereo_raw_value(left, right) for left, right in zip(first, second)
        )
    if is_dataclass(first):
        return all(_same_stereo_raw_value(getattr(first, field.name), getattr(second, field.name))
                   for field in fields(first))
    return first == second


def _check_stereo_capture(mol, capture):
    """Check the existing owner binding and only the already-consumed raw facts."""

    snapshot, source, (atoms, partitions, endpoints) = capture
    _check_source(mol, source)
    unchanged = (
        _same_stereo_raw_value(snapshot.payload, mol._stereo_payload)
        and len(atoms) == len(mol._atoms)
        and all(first is second for first, second in zip(atoms, mol._atoms))
        and all(
            len(previous) == len(current)
            and all(first is second for first, second in zip(previous, current))
            for previous, current in zip(partitions, (
                mol._bonds, mol._hided_metal_bonds, mol._hided_covalent_bonds,
            ))
        )
    )
    if unchanged:
        for bond, previous_endpoints, row in zip(
            (bond for bonds in partitions for bond in bonds), endpoints, snapshot.bonds,
        ):
            first, second = bond.atom1, bond.atom2
            if (
                first is not previous_endpoints[0]
                or second is not previous_endpoints[1]
                or not _same_stereo_raw_value(row, (
                    id(first), id(second), float(bond.bond_order), bond.bond_kind.value,
                    bond.bond_direction,
                ))
            ):
                unchanged = False
                break
    if not unchanged:
        raise RuntimeError("Captured stereochemistry facts changed; retry the operation")


def _stereo_prepared_fields(mol, *, fields):
    """Read only requested scalars after the operation has prepared this owner."""

    return tuple(tuple(getattr(atom, field) for field in fields) for atom in mol._atoms)


def _pack_gate6_target(graph):
    """Materialize the exact ordinary target dictionary once per native target."""

    def materialize(value):
        if isinstance(value, Mapping):
            return {key: materialize(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(materialize(item) for item in value)
        return value

    return materialize(graph.target_spec)
