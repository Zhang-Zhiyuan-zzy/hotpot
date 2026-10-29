"""Field-selective contiguous molecule buffers for native consumers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum, IntFlag
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..core import Molecule


class NativeBondKindCode(IntEnum):
    """Toolkit-neutral bond-kind codes shared with native views."""

    SINGLE = 1
    DOUBLE = 2
    TRIPLE = 3
    AROMATIC = 4
    ZERO = 5
    DATIVE = 6
    UNKNOWN = 7


class NativeMoleculeField(Enum):
    """Independently selectable fields in the common molecule payload."""

    ATOMIC_NUMBERS = "atomic_numbers"
    FORMAL_CHARGES = "formal_charges"
    PARTIAL_CHARGES = "partial_charges"
    COORDINATES = "coordinates"
    ATOM_AROMATIC = "atom_aromatic"
    BOND_INDICES = "bond_indices"
    BOND_ORDERS = "bond_orders"
    BOND_KINDS = "bond_kinds"
    BOND_AROMATIC = "bond_aromatic"
    UNIT_CELL = "unit_cell"
    STEREO_RECORDS = "stereo_records"
    ATOM_TRANSPORT = "atom_transport"
    BOND_TRANSPORT = "bond_transport"


class NativeStereoAnchorKindCode(IntEnum):
    """Fixed-width stereo-anchor codes shared with the native ABI."""

    ATOM = 1
    BOND = 2
    AXIS = 3


class NativeStereoCarrierKindCode(IntEnum):
    """Fixed-width stereo-carrier codes shared with the native ABI."""

    ATOM = 1
    IMPLICIT_H = 2
    LONE_PAIR = 3
    VACANCY = 4


class NativeStereoGroupKindCode(IntEnum):
    """Fixed-width enhanced-stereo group codes shared with the native ABI."""

    NONE = 0
    ABS = 1
    AND = 2
    OR = 3


class NativeAtomTransportFieldCode(IntFlag):
    """Presence bits for atom-local optional transport facts."""

    ISOTOPE = 1 << 0
    RADICAL_ELECTRONS = 1 << 1


class NativeHydrogenTransportFieldCode(IntFlag):
    """Presence bits for one optional hydrogen transport state."""

    IMPLICIT = 1 << 0
    BRACKET_EXPLICIT = 1 << 1
    NO_IMPLICIT = 1 << 2


@dataclass(frozen=True)
class NativeUtf8Table:
    """Owned UTF-8 bytes and stable views for one ordered string column."""

    data: NDArray[np.uint8]
    offsets: NDArray[np.uint64]
    lengths: NDArray[np.uint64]
    present: NDArray[np.uint8]


@dataclass(frozen=True)
class NativeStereoRecordBuffers:
    """Flat, order-preserving native transport for Level-1 stereo records."""

    schema_version: int
    anchor_kinds: NDArray[np.uint32]
    anchor_offsets: NDArray[np.uint64]
    anchor_counts: NDArray[np.uint64]
    anchor_indices: NDArray[np.uint64]
    carrier_offsets: NDArray[np.uint64]
    carrier_counts: NDArray[np.uint64]
    carrier_kinds: NDArray[np.uint32]
    carrier_has_atoms: NDArray[np.uint32]
    carrier_atoms: NDArray[np.uint64]
    group_kinds: NDArray[np.uint32]
    group_ids: NDArray[np.int64]
    group_present: NDArray[np.uint32]
    group_scopes: NativeUtf8Table
    descriptors: NativeUtf8Table
    source_namespaces: NativeUtf8Table
    source_codes: NativeUtf8Table


@dataclass(frozen=True)
class NativeAtomTransportBuffers:
    """Flat, presence-preserving native transport for atom facts."""

    schema_version: int
    atoms: NDArray[np.uint64]
    presence_masks: NDArray[np.uint32]
    isotopes: NDArray[np.int32]
    radical_electrons: NDArray[np.int32]
    has_current_hydrogens: NDArray[np.uint32]
    current_hydrogen_presence_masks: NDArray[np.uint32]
    current_implicit_hydrogens: NDArray[np.int32]
    current_bracket_explicit_hydrogens: NDArray[np.int32]
    current_no_implicit: NDArray[np.uint32]
    has_source_hydrogens: NDArray[np.uint32]
    source_hydrogen_presence_masks: NDArray[np.uint32]
    source_implicit_hydrogens: NDArray[np.int32]
    source_bracket_explicit_hydrogens: NDArray[np.int32]
    source_no_implicit: NDArray[np.uint32]
    source_namespaces: NativeUtf8Table


@dataclass(frozen=True)
class NativeBondTransportBuffers:
    """Flat native transport for ordered bond-local display facts."""

    schema_version: int
    atom_indices: NDArray[np.uint64]
    display_directions: NativeUtf8Table


@dataclass(frozen=True)
class NativeMoleculeBuffers:
    """Contiguous values selected from one Hotpot molecule."""

    schema_version: int
    atom_count: int
    bond_count: int
    atomic_numbers: NDArray[np.int32] | None = None
    formal_charges: NDArray[np.int32] | None = None
    partial_charges: NDArray[np.float64] | None = None
    coordinates: NDArray[np.float64] | None = None
    atom_aromatic: NDArray[np.uint8] | None = None
    bond_indices: NDArray[np.int32] | None = None
    bond_orders: NDArray[np.float64] | None = None
    bond_kinds: NDArray[np.uint8] | None = None
    bond_aromatic: NDArray[np.uint8] | None = None
    unit_cell: NDArray[np.float64] | None = None
    stereo_records: NativeStereoRecordBuffers | None = None
    atom_transport: NativeAtomTransportBuffers | None = None
    bond_transport: NativeBondTransportBuffers | None = None


_BOND_KIND_CODES = {
    "single": NativeBondKindCode.SINGLE,
    "double": NativeBondKindCode.DOUBLE,
    "triple": NativeBondKindCode.TRIPLE,
    "aromatic": NativeBondKindCode.AROMATIC,
    "zero": NativeBondKindCode.ZERO,
    "dative": NativeBondKindCode.DATIVE,
    "unknown": NativeBondKindCode.UNKNOWN,
}

MOLECULE_BUFFER_SCHEMA_VERSION = 1


def _empty_matrix(columns: int, dtype: np.dtype) -> np.ndarray:
    return np.empty((0, columns), dtype=dtype)


def _array(values, dtype) -> np.ndarray:
    return np.ascontiguousarray(tuple(values), dtype=dtype)


def _pack_utf8(values: tuple[str | None, ...]) -> NativeUtf8Table:
    encoded = tuple(
        None if value is None else value.encode("utf-8") for value in values
    )
    data = bytearray(b"\0")
    offsets = []
    lengths = []
    present = []
    for value in encoded:
        if value is None:
            offsets.append(0)
            lengths.append(0)
            present.append(0)
        else:
            offsets.append(len(data))
            lengths.append(len(value))
            present.append(1)
            data.extend(value)

    return NativeUtf8Table(
        data=np.frombuffer(bytes(data), dtype=np.uint8).copy(),
        offsets=_array(offsets, np.uint64),
        lengths=_array(lengths, np.uint64),
        present=_array(present, np.uint8),
    )


def _pack_stereo_records(payload) -> NativeStereoRecordBuffers:
    from ..stereo import (
        StereoAnchorKind,
        StereoCarrierKind,
        StereoGroupKind,
    )

    anchor_kind_codes = {
        StereoAnchorKind.ATOM: NativeStereoAnchorKindCode.ATOM,
        StereoAnchorKind.BOND: NativeStereoAnchorKindCode.BOND,
        StereoAnchorKind.AXIS: NativeStereoAnchorKindCode.AXIS,
    }
    carrier_kind_codes = {
        StereoCarrierKind.ATOM: NativeStereoCarrierKindCode.ATOM,
        StereoCarrierKind.IMPLICIT_H: NativeStereoCarrierKindCode.IMPLICIT_H,
        StereoCarrierKind.LONE_PAIR: NativeStereoCarrierKindCode.LONE_PAIR,
        StereoCarrierKind.VACANCY: NativeStereoCarrierKindCode.VACANCY,
    }
    group_kind_codes = {
        StereoGroupKind.ABS: NativeStereoGroupKindCode.ABS,
        StereoGroupKind.AND: NativeStereoGroupKindCode.AND,
        StereoGroupKind.OR: NativeStereoGroupKindCode.OR,
    }

    anchor_indices = []
    carrier_kinds = []
    carrier_has_atoms = []
    carrier_atoms = []
    anchor_offsets = []
    carrier_offsets = []
    for record in payload.records:
        anchor_offsets.append(len(anchor_indices))
        carrier_offsets.append(len(carrier_kinds))
        anchor_indices.extend(record.anchor.atom_indices)
        for carrier in record.carriers:
            carrier_kinds.append(carrier_kind_codes[carrier.kind])
            carrier_has_atoms.append(carrier.atom is not None)
            carrier_atoms.append(0 if carrier.atom is None else carrier.atom)

    groups = tuple(record.group for record in payload.records)
    return NativeStereoRecordBuffers(
        schema_version=payload.schema_version,
        anchor_kinds=_array(
            (anchor_kind_codes[record.anchor.kind] for record in payload.records),
            np.uint32,
        ),
        anchor_offsets=_array(anchor_offsets, np.uint64),
        anchor_counts=_array(
            (len(record.anchor.atom_indices) for record in payload.records),
            np.uint64,
        ),
        anchor_indices=_array(anchor_indices, np.uint64),
        carrier_offsets=_array(carrier_offsets, np.uint64),
        carrier_counts=_array(
            (len(record.carriers) for record in payload.records), np.uint64
        ),
        carrier_kinds=_array(carrier_kinds, np.uint32),
        carrier_has_atoms=_array(carrier_has_atoms, np.uint32),
        carrier_atoms=_array(carrier_atoms, np.uint64),
        group_kinds=_array(
            (
                NativeStereoGroupKindCode.NONE
                if group is None
                else group_kind_codes[group.kind]
                for group in groups
            ),
            np.uint32,
        ),
        group_ids=_array(
            (0 if group is None else group.id for group in groups), np.int64
        ),
        group_present=_array((group is not None for group in groups), np.uint32),
        group_scopes=_pack_utf8(
            tuple(None if group is None else group.scope for group in groups)
        ),
        descriptors=_pack_utf8(
            tuple(record.descriptor for record in payload.records)
        ),
        source_namespaces=_pack_utf8(
            tuple(record.source_namespace for record in payload.records)
        ),
        source_codes=_pack_utf8(
            tuple(record.source_code for record in payload.records)
        ),
    )


def _pack_hydrogen_state(state):
    if state is None:
        return 0, 0, 0, 0

    presence_mask = 0
    if state.implicit_hydrogens is not None:
        presence_mask |= NativeHydrogenTransportFieldCode.IMPLICIT
    if state.bracket_explicit_hydrogens is not None:
        presence_mask |= NativeHydrogenTransportFieldCode.BRACKET_EXPLICIT
    if state.no_implicit is not None:
        presence_mask |= NativeHydrogenTransportFieldCode.NO_IMPLICIT
    return (
        presence_mask,
        0 if state.implicit_hydrogens is None else state.implicit_hydrogens,
        0
        if state.bracket_explicit_hydrogens is None
        else state.bracket_explicit_hydrogens,
        0 if state.no_implicit is None else state.no_implicit,
    )


def _pack_atom_transport(payload, atoms) -> NativeAtomTransportBuffers:
    facts = tuple(payload.atom_facts)
    transported_atoms = {fact.atom for fact in facts}
    missing_atoms = tuple(
        (row, atom)
        for row, atom in enumerate(atoms)
        if row not in transported_atoms
    )
    missing_count = len(missing_atoms)
    atom_rows = tuple(fact.atom for fact in facts) + tuple(
        row for row, _ in missing_atoms
    )
    presence_masks = tuple(
        (
            NativeAtomTransportFieldCode.ISOTOPE
            if fact.isotope is not None
            else 0
        )
        | (
            NativeAtomTransportFieldCode.RADICAL_ELECTRONS
            if fact.radical_electrons is not None
            else 0
        )
        for fact in facts
    ) + (0,) * missing_count
    isotopes = tuple(
        0 if fact.isotope is None else fact.isotope for fact in facts
    ) + (0,) * missing_count
    radical_electrons = tuple(
        0 if fact.radical_electrons is None else fact.radical_electrons
        for fact in facts
    ) + (0,) * missing_count
    has_current_hydrogens = tuple(
        fact.current_hydrogens is not None for fact in facts
    ) + (True,) * missing_count
    has_source_hydrogens = tuple(
        fact.source_hydrogens is not None for fact in facts
    ) + (False,) * missing_count
    current_hydrogens = tuple(
        _pack_hydrogen_state(fact.current_hydrogens)
        for fact in facts
    ) + tuple(
        (
            NativeHydrogenTransportFieldCode.IMPLICIT,
            atom.implicit_hydrogens,
            0,
            0,
        )
        for _, atom in missing_atoms
    )
    source_hydrogens = tuple(
        _pack_hydrogen_state(fact.source_hydrogens)
        for fact in facts
    ) + ((0, 0, 0, 0),) * missing_count
    return NativeAtomTransportBuffers(
        schema_version=payload.schema_version,
        atoms=_array(atom_rows, np.uint64),
        presence_masks=_array(presence_masks, np.uint32),
        isotopes=_array(isotopes, np.int32),
        radical_electrons=_array(radical_electrons, np.int32),
        has_current_hydrogens=_array(has_current_hydrogens, np.uint32),
        current_hydrogen_presence_masks=_array(
            (state[0] for state in current_hydrogens), np.uint32
        ),
        current_implicit_hydrogens=_array(
            (state[1] for state in current_hydrogens), np.int32
        ),
        current_bracket_explicit_hydrogens=_array(
            (state[2] for state in current_hydrogens), np.int32
        ),
        current_no_implicit=_array(
            (state[3] for state in current_hydrogens), np.uint32
        ),
        has_source_hydrogens=_array(has_source_hydrogens, np.uint32),
        source_hydrogen_presence_masks=_array(
            (state[0] for state in source_hydrogens), np.uint32
        ),
        source_implicit_hydrogens=_array(
            (state[1] for state in source_hydrogens), np.int32
        ),
        source_bracket_explicit_hydrogens=_array(
            (state[2] for state in source_hydrogens), np.int32
        ),
        source_no_implicit=_array(
            (state[3] for state in source_hydrogens), np.uint32
        ),
        source_namespaces=_pack_utf8(
            tuple(fact.source_namespace for fact in facts)
            + (None,) * missing_count
        ),
    )


def _pack_bond_transport(payload) -> NativeBondTransportBuffers:
    return NativeBondTransportBuffers(
        schema_version=payload.schema_version,
        atom_indices=_array(
            (fact.atom_indices for fact in payload.bond_facts), np.uint64
        ).reshape((-1, 2)),
        display_directions=_pack_utf8(
            tuple(fact.display_direction for fact in payload.bond_facts)
        ),
    )


def pack_molecule(
    mol: Molecule,
    fields: frozenset[NativeMoleculeField],
) -> NativeMoleculeBuffers:
    """Copy only the requested molecule fields into contiguous arrays."""
    with mol._mutation_lock:

        atoms = tuple(mol.atoms)
        bonds = tuple(mol.bonds)
        atom_rows = (
            {id(atom): row for row, atom in enumerate(atoms)}
            if NativeMoleculeField.BOND_INDICES in fields
            else None
        )

        def atom_values(field, name, dtype, codec):
            if field not in fields:
                return None
            values = mol._atom_table.column(name)
            if codec is float:
                return values
            return np.fromiter((codec(value) for value in values), dtype=dtype, count=len(atoms))

        def bond_values(field, values, dtype):
            if field not in fields:
                return None
            return np.ascontiguousarray(
                np.fromiter(values, dtype=dtype, count=len(bonds))
            )

        atomic_numbers = atom_values(
            NativeMoleculeField.ATOMIC_NUMBERS,
            "atomic_number", np.int32, int,
        )
        formal_charges = atom_values(
            NativeMoleculeField.FORMAL_CHARGES,
            "formal_charge", np.int32, int,
        )
        partial_charges = atom_values(
            NativeMoleculeField.PARTIAL_CHARGES,
            "partial_charge", np.float64, float,
        )
        coordinates = None
        if NativeMoleculeField.COORDINATES in fields:
            coordinates = mol._atom_table.coordinates()
        atom_aromatic = atom_values(
            NativeMoleculeField.ATOM_AROMATIC,
            "is_aromatic", np.uint8, bool,
        )

        bond_indices = None
        if NativeMoleculeField.BOND_INDICES in fields:
            def directed_endpoints(bond):
                atom1 = atom_rows[id(bond.atom1)]
                atom2 = atom_rows[id(bond.atom2)]
                if (
                    bond.bond_kind.value == "dative"
                    and bond.bond_direction == "atom2_to_atom1"
                ):
                    return atom2, atom1
                return atom1, atom2

            bond_indices = (
                np.ascontiguousarray(
                    [directed_endpoints(bond) for bond in bonds],
                    dtype=np.int32,
                )
                if bonds
                else _empty_matrix(2, np.dtype(np.int32))
            )
        bond_orders = bond_values(
            NativeMoleculeField.BOND_ORDERS,
            (bond.bond_order for bond in bonds),
            np.float64,
        )
        bond_kinds = bond_values(
            NativeMoleculeField.BOND_KINDS,
            (_BOND_KIND_CODES[bond.bond_kind.value] for bond in bonds),
            np.uint8,
        )
        bond_aromatic = bond_values(
            NativeMoleculeField.BOND_AROMATIC,
            (
                bond.bond_kind.value == "aromatic" or bond.is_aromatic
                for bond in bonds
            ),
            np.uint8,
        )

        unit_cell = None
        if NativeMoleculeField.UNIT_CELL in fields and mol.crystal is not None:
            crystal = mol.crystal
            unit_cell = np.ascontiguousarray(
                (
                    crystal.a,
                    crystal.b,
                    crystal.c,
                    crystal.alpha,
                    crystal.beta,
                    crystal.gamma,
                ),
                dtype=np.float64,
            )

        stereo_records = None
        atom_transport = None
        bond_transport = None
        stereo_fields = fields & frozenset(
            {
                NativeMoleculeField.STEREO_RECORDS,
                NativeMoleculeField.ATOM_TRANSPORT,
                NativeMoleculeField.BOND_TRANSPORT,
            }
        )
        if stereo_fields:
            from ..stereo import get_stereo_payload

            payload = get_stereo_payload(mol)
            if NativeMoleculeField.STEREO_RECORDS in fields:
                stereo_records = _pack_stereo_records(payload)
            if NativeMoleculeField.ATOM_TRANSPORT in fields:
                atom_transport = _pack_atom_transport(payload, atoms)
            if NativeMoleculeField.BOND_TRANSPORT in fields:
                bond_transport = _pack_bond_transport(payload)

        return NativeMoleculeBuffers(
            schema_version=MOLECULE_BUFFER_SCHEMA_VERSION,
            atom_count=len(atoms),
            bond_count=len(bonds),
            atomic_numbers=atomic_numbers,
            formal_charges=formal_charges,
            partial_charges=partial_charges,
            coordinates=coordinates,
            atom_aromatic=atom_aromatic,
            bond_indices=bond_indices,
            bond_orders=bond_orders,
            bond_kinds=bond_kinds,
            bond_aromatic=bond_aromatic,
            unit_cell=unit_cell,
            stereo_records=stereo_records,
            atom_transport=atom_transport,
            bond_transport=bond_transport,
        )
