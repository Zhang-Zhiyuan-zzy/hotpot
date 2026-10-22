"""Exact field-selective native packing for Level-1 stereo transport."""

from __future__ import annotations

from dataclasses import fields as dataclass_fields

import numpy as np

from hotpot import read_mol
from hotpot.cheminfo._native.buffers import (
    NativeAtomTransportFieldCode,
    NativeAtomTransportBuffers,
    NativeBondTransportBuffers,
    NativeHydrogenTransportFieldCode,
    NativeMoleculeField,
    NativeStereoAnchorKindCode,
    NativeStereoCarrierKindCode,
    NativeStereoGroupKindCode,
    NativeStereoRecordBuffers,
    NativeUtf8Table,
    pack_molecule,
)
from hotpot.cheminfo.stereo import (
    AtomTransportRecord,
    BondTransportRecord,
    HydrogenTransportState,
    StereoAnchor,
    StereoAnchorKind,
    StereoCarrier,
    StereoCarrierKind,
    StereoGroupKind,
    StereoGroupRef,
    StereoPayload,
    StereoRecord,
    set_stereo_payload,
)


def _payload() -> StereoPayload:
    return StereoPayload(
        schema_version=7,
        records=(
            StereoRecord(
                anchor=StereoAnchor(StereoAnchorKind.ATOM, (2,)),
                carriers=(
                    StereoCarrier(StereoCarrierKind.ATOM, 3),
                    StereoCarrier(StereoCarrierKind.IMPLICIT_H),
                ),
                descriptor="顺时针",
                group=StereoGroupRef("范围α", StereoGroupKind.AND, 0),
                source_namespace="工具/甲",
                source_code="代码β",
            ),
            StereoRecord(
                anchor=StereoAnchor(StereoAnchorKind.BOND, (3, 1)),
                carriers=(
                    StereoCarrier(StereoCarrierKind.VACANCY),
                    StereoCarrier(StereoCarrierKind.ATOM, 0),
                ),
                descriptor="",
                group=None,
                source_namespace="",
                source_code="未知/γ",
            ),
        ),
        atom_facts=(
            AtomTransportRecord(
                atom=3,
                isotope=0,
                radical_electrons=None,
                current_hydrogens=HydrogenTransportState(0, None, False),
                source_hydrogens=None,
                source_namespace="",
            ),
            AtomTransportRecord(
                atom=0,
                isotope=None,
                radical_electrons=0,
                current_hydrogens=None,
                source_hydrogens=HydrogenTransportState(None, 0, True),
                source_namespace=None,
            ),
        ),
        bond_facts=(
            BondTransportRecord((3, 1), ""),
            BondTransportRecord((0, 2), None),
            BondTransportRecord((2, 0), "向上↑"),
        ),
    )


def _decode(table: NativeUtf8Table) -> tuple[str | None, ...]:
    raw = table.data.tobytes()
    return tuple(
        None
        if not present
        else raw[int(offset) : int(offset + length)].decode("utf-8")
        for offset, length, present in zip(
            table.offsets, table.lengths, table.present
        )
    )


def _assert_all_arrays_are_contiguous(dto) -> None:
    for field in dataclass_fields(dto):
        value = getattr(dto, field.name)
        if isinstance(value, np.ndarray):
            assert value.flags.c_contiguous, field.name
        elif isinstance(value, NativeUtf8Table):
            _assert_all_arrays_are_contiguous(value)


def test_level1_tables_are_independently_selected_and_present_empty_is_distinct():
    mol = read_mol("CCCC")
    set_stereo_payload(mol, _payload())

    stereo_only = pack_molecule(
        mol, frozenset({NativeMoleculeField.STEREO_RECORDS})
    )
    atom_only = pack_molecule(
        mol, frozenset({NativeMoleculeField.ATOM_TRANSPORT})
    )
    bond_only = pack_molecule(
        mol, frozenset({NativeMoleculeField.BOND_TRANSPORT})
    )

    assert isinstance(stereo_only.stereo_records, NativeStereoRecordBuffers)
    assert stereo_only.atom_transport is None
    assert stereo_only.bond_transport is None
    assert isinstance(atom_only.atom_transport, NativeAtomTransportBuffers)
    assert atom_only.stereo_records is None
    assert atom_only.bond_transport is None
    assert isinstance(bond_only.bond_transport, NativeBondTransportBuffers)
    assert bond_only.stereo_records is None
    assert bond_only.atom_transport is None

    assert _decode(stereo_only.stereo_records.descriptors) == ("顺时针", "")
    assert _decode(stereo_only.stereo_records.group_scopes) == ("范围α", None)
    assert _decode(atom_only.atom_transport.source_namespaces) == ("", None)
    assert _decode(bond_only.bond_transport.display_directions) == (
        "",
        None,
        "向上↑",
    )


def test_stereo_record_packing_preserves_order_endpoints_groups_and_utf8():
    mol = read_mol("CCCC")
    set_stereo_payload(mol, _payload())

    records = pack_molecule(
        mol, frozenset({NativeMoleculeField.STEREO_RECORDS})
    ).stereo_records

    assert records.schema_version == 7
    assert records.anchor_kinds.dtype == np.uint32
    assert records.anchor_offsets.dtype == np.uint64
    assert records.anchor_counts.dtype == np.uint64
    assert records.anchor_indices.dtype == np.uint64
    assert records.carrier_kinds.dtype == np.uint32
    assert records.carrier_has_atoms.dtype == np.uint32
    assert records.carrier_atoms.dtype == np.uint64
    assert records.group_kinds.dtype == np.uint32
    assert records.group_ids.dtype == np.int64
    assert records.group_present.dtype == np.uint32
    assert records.anchor_kinds.tolist() == [
        NativeStereoAnchorKindCode.ATOM,
        NativeStereoAnchorKindCode.BOND,
    ]
    assert records.anchor_offsets.tolist() == [0, 1]
    assert records.anchor_counts.tolist() == [1, 2]
    assert records.anchor_indices.tolist() == [2, 3, 1]
    assert records.carrier_offsets.tolist() == [0, 2]
    assert records.carrier_counts.tolist() == [2, 2]
    assert records.carrier_kinds.tolist() == [
        NativeStereoCarrierKindCode.ATOM,
        NativeStereoCarrierKindCode.IMPLICIT_H,
        NativeStereoCarrierKindCode.VACANCY,
        NativeStereoCarrierKindCode.ATOM,
    ]
    assert records.carrier_has_atoms.tolist() == [1, 0, 0, 1]
    assert records.carrier_atoms.tolist() == [3, 0, 0, 0]
    assert records.group_kinds.tolist() == [
        NativeStereoGroupKindCode.AND,
        NativeStereoGroupKindCode.NONE,
    ]
    assert records.group_ids.tolist() == [0, 0]
    assert records.group_present.tolist() == [1, 0]
    assert _decode(records.source_namespaces) == ("工具/甲", "")
    assert _decode(records.source_codes) == ("代码β", "未知/γ")
    _assert_all_arrays_are_contiguous(records)


def test_axis_anchor_packing_preserves_variable_length_atom_sequence():
    assert NativeStereoAnchorKindCode.ATOM == 1
    assert NativeStereoAnchorKindCode.BOND == 2
    assert NativeStereoAnchorKindCode.AXIS == 3

    mol = read_mol("CCCCC")
    set_stereo_payload(
        mol,
        StereoPayload(
            schema_version=1,
            records=(
                StereoRecord(
                    anchor=StereoAnchor(
                        StereoAnchorKind.AXIS,
                        (4, 2, 1, 3),
                    ),
                    carriers=(),
                    descriptor="hotpot.stereo/2/AX/even",
                    group=None,
                    source_namespace="hotpot",
                    source_code="AX",
                ),
            ),
            atom_facts=(),
            bond_facts=(),
        ),
    )

    records = pack_molecule(
        mol, frozenset({NativeMoleculeField.STEREO_RECORDS})
    ).stereo_records

    assert records.anchor_kinds.tolist() == [
        NativeStereoAnchorKindCode.AXIS
    ]
    assert records.anchor_offsets.tolist() == [0]
    assert records.anchor_counts.tolist() == [4]
    assert records.anchor_indices.tolist() == [4, 2, 1, 3]


def test_atom_transport_packing_preserves_explicit_zero_and_dual_hydrogen_state():
    mol = read_mol("CCCC")
    set_stereo_payload(mol, _payload())

    atoms = pack_molecule(
        mol, frozenset({NativeMoleculeField.ATOM_TRANSPORT})
    ).atom_transport

    assert atoms.atoms.dtype == np.uint64
    assert atoms.presence_masks.dtype == np.uint32
    assert atoms.isotopes.dtype == np.int32
    assert atoms.radical_electrons.dtype == np.int32
    assert atoms.atoms.tolist() == [3, 0]
    assert atoms.presence_masks.tolist() == [
        NativeAtomTransportFieldCode.ISOTOPE,
        NativeAtomTransportFieldCode.RADICAL_ELECTRONS,
    ]
    assert atoms.isotopes.tolist() == [0, 0]
    assert atoms.radical_electrons.tolist() == [0, 0]
    assert atoms.has_current_hydrogens.tolist() == [1, 0]
    assert atoms.current_hydrogen_presence_masks.tolist() == [
        NativeHydrogenTransportFieldCode.IMPLICIT
        | NativeHydrogenTransportFieldCode.NO_IMPLICIT,
        0,
    ]
    assert atoms.current_implicit_hydrogens.tolist() == [0, 0]
    assert atoms.current_no_implicit.tolist() == [0, 0]
    assert atoms.has_source_hydrogens.tolist() == [0, 1]
    assert atoms.source_hydrogen_presence_masks.tolist() == [
        0,
        NativeHydrogenTransportFieldCode.BRACKET_EXPLICIT
        | NativeHydrogenTransportFieldCode.NO_IMPLICIT,
    ]
    assert atoms.source_bracket_explicit_hydrogens.tolist() == [0, 0]
    assert atoms.source_no_implicit.tolist() == [0, 1]
    _assert_all_arrays_are_contiguous(atoms)


def test_bond_transport_preserves_ordered_endpoints_and_empty_tables_have_shapes():
    mol = read_mol("CCCC")
    set_stereo_payload(mol, _payload())

    bonds = pack_molecule(
        mol, frozenset({NativeMoleculeField.BOND_TRANSPORT})
    ).bond_transport
    assert bonds.atom_indices.dtype == np.uint64
    assert bonds.atom_indices.shape == (3, 2)
    assert bonds.atom_indices.tolist() == [[3, 1], [0, 2], [2, 0]]
    _assert_all_arrays_are_contiguous(bonds)

    empty_mol = read_mol("C")
    empty = pack_molecule(
        empty_mol,
        frozenset(
            {
                NativeMoleculeField.STEREO_RECORDS,
                NativeMoleculeField.ATOM_TRANSPORT,
                NativeMoleculeField.BOND_TRANSPORT,
            }
        ),
    )
    assert empty.stereo_records.anchor_kinds.shape == (0,)
    assert empty.stereo_records.anchor_indices.shape == (0,)
    assert empty.atom_transport.atoms.shape == (0,)
    assert empty.bond_transport.atom_indices.shape == (0, 2)
    assert empty.stereo_records.descriptors.data.shape == (1,)
    assert empty.stereo_records.descriptors.offsets.shape == (0,)
    assert empty.stereo_records.descriptors.present.shape == (0,)
    _assert_all_arrays_are_contiguous(empty.stereo_records)
    _assert_all_arrays_are_contiguous(empty.atom_transport)
    _assert_all_arrays_are_contiguous(empty.bond_transport)
