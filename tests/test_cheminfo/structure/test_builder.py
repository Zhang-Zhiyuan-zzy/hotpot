"""Immutable complete import records and factual carrier installation contracts."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, fields, replace
import builtins
import importlib.util
import pickle
import sys

import numpy as np
import pytest

from hotpot.cheminfo.structure._builder import (
    MoleculeRecord, _freeze_value, _make_molecule_record, _thaw_value,
)
import hotpot.cheminfo.structure._builder as builder
from hotpot.cheminfo.stereo import (
    AtomTransportRecord, BondTransportRecord, HydrogenTransportState,
    StereoAnchor, StereoAnchorKind, StereoCarrier, StereoCarrierKind,
    StereoGroupKind, StereoGroupRef, StereoPayload, StereoRecord,
)


def _inputs():
    values = np.arange(4 * 19, dtype=np.float64).reshape(4, 19) + 0.125
    values[0, 15] = 2**40 + 0.125
    return dict(
        atom_values=values, atom_presence=np.array([0, 1, 0x7FFFF, 0], dtype=np.uint32),
        atom_sources=(np.arange(4 * 19).reshape(4, 19) % 10).astype(np.uint8),
        atom_isotopes=((0, None), (1, 0), (2, 13)), atom_frac_valences=((0, 1.75),),
        bond_endpoints=((2, 0), (0, 1), (3, 2)),
        bond_values=np.array([[1.25, 0.25, 2**40 + 0.125], [0, 0, 0], [-4.5, 2.5, -2.25]]),
        bond_partitions=(0, 1, 2), bond_kinds=("dative", "zero", "unknown"),
        bond_directions=("wedge", None, "raw direction"), bond_sources=(None, "rdkit", "source"),
        bond_metadata=({}, None, {"raw": [1, (2, b"three")]}),
        charge=(True, 0, None), spin_mult=(False, None, None), cell=None,
        stereo_payload=None, legacy_frame_values=(None,) * 9, legacy_frame_index=-1,
        angle_constraints=((2, 0, 1),), torsion_constraints=((3, 2, 0, 1),),
        identifier=(False, None),
    )


def _record(**changes):
    return _make_molecule_record(**(_inputs() | changes))


def _atoms(record):
    return np.frombuffer(record.atom_values, dtype="<f8").reshape(record.atom_count, 19)


def test_module_load_has_empty_exports_and_no_science_import(monkeypatch):
    original_import = builtins.__import__

    def checked_import(name, *args, **kwargs):
        if name.rsplit(".", 1)[-1] in ("core", "stereo"):
            raise AssertionError(f"Record module loaded scientific dependency: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked_import)
    name = "hotpot.cheminfo.structure._record_loading_probe"
    spec = importlib.util.spec_from_file_location(name, builder.__file__)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    assert module.__all__ == ()


def test_complete_record_preserves_raw_width_order_and_presence():
    inputs = _inputs()
    record = _make_molecule_record(**inputs)
    np.testing.assert_array_equal(_atoms(record), inputs["atom_values"])
    np.testing.assert_array_equal(np.frombuffer(record.bond_values, dtype="<f8").reshape(-1, 3), inputs["bond_values"])
    assert record.bond_endpoints == ((2, 0), (0, 1), (3, 2))
    assert record.bond_partitions == (0, 1, 2)
    assert record.atom_isotopes == ((0, None), (1, 0), (2, 13))
    assert record.atom_frac_valences == ((0, 1.75),)
    assert record.charge == (True, 0, None)
    assert record.spin_mult == (False, None, None)
    assert np.frombuffer(record.atom_presence, dtype="<u4").tolist() == [0, 1, 0x7FFFF, 0]
    np.testing.assert_array_equal(np.frombuffer(record.atom_sources, dtype="u1").reshape(4, 19), inputs["atom_sources"])
    assert record.angle_constraints == ((2, 0, 1),)
    assert record.torsion_constraints == ((3, 2, 0, 1),)
    assert record.legacy_frame_index == -1
    assert record.identifier == (False, None)
    assert _record(identifier=(True, None)).identifier == (True, None)
    assert not any("owner" in field.name or "uid" in field.name for field in fields(record))


def test_buffers_and_nested_inputs_are_isolated_and_frozen():
    inputs = _inputs()
    nested = {"list": [np.arange(6).reshape(2, 3), {"x": [1]}]}
    inputs["bond_metadata"] = (nested, None, {})
    record = _make_molecule_record(**inputs)
    saved = record.atom_values
    saved_bonds = record.bond_values
    inputs["atom_values"][:] = -1
    inputs["atom_presence"][:] = 1
    inputs["atom_sources"][:] = 0
    inputs["bond_values"][:] = -1
    nested["list"][0][:] = -1
    nested["list"][1]["x"].append(2)
    assert record.atom_values == saved
    assert record.bond_values == saved_bonds
    assert _thaw_value(record.bond_metadata[0])["list"][1] == {"x": [1]}
    matrix = _atoms(record)
    assert not matrix.flags.writeable
    with pytest.raises(ValueError):
        matrix.flags.writeable = True
    matrix.shape = (76,)
    matrix.dtype = np.uint8
    assert _atoms(record).shape == (4, 19)
    with pytest.raises(FrozenInstanceError):
        record.atom_count = 0
    mutable = bytearray(saved)
    reconstructed = replace(record, atom_values=mutable)
    mutable[0] ^= 255
    assert reconstructed.atom_values == saved


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_pickle_reconstructs_through_normal_constructor_and_deepcopy(protocol):
    record = _record()
    restored = pickle.loads(pickle.dumps(record, protocol=protocol))
    assert type(restored) is MoleculeRecord
    assert restored is not record
    for field in fields(record):
        assert getattr(restored, field.name) == getattr(record, field.name)
    assert deepcopy(record) is record
    assert restored.__reduce__()[0] is MoleculeRecord
    with pytest.raises(ValueError):
        _atoms(restored).flags.writeable = True


def test_nested_metadata_and_nine_legacy_slots_keep_types_shapes_and_dtypes():
    array = np.arange(12, dtype=">i2").reshape(3, 4)[:, ::-1]
    frames = (None, 0.0, np.array(0.0), np.float32(0), [1, (2,)],
              array, {"x": np.array([], dtype="<f4").reshape(0, 2)}, b"raw", "text")
    record = _record(legacy_frame_values=frames)
    decoded = tuple(map(_thaw_value, record.legacy_frame_values))
    assert decoded[0] is None
    assert type(decoded[1]) is float
    assert type(decoded[2]) is np.ndarray and decoded[2].shape == ()
    assert type(decoded[3]) is np.float32
    assert type(decoded[4]) is list and type(decoded[4][1]) is tuple
    assert decoded[5].dtype.str == ">i2"
    np.testing.assert_array_equal(decoded[5], array)
    assert decoded[6]["x"].shape == (0, 2)
    assert decoded[6]["x"].dtype.str == "<f4"
    assert decoded[7:] == (b"raw", "text")
    decoded[4].append(9)
    decoded[5].shape = (12,)
    assert len(_thaw_value(record.legacy_frame_values[4])) == 2
    assert _thaw_value(record.legacy_frame_values[5]).shape == (3, 4)
    assert _thaw_value(record.legacy_frame_values[5]) is not _thaw_value(record.legacy_frame_values[5])
    frames[5][:] = -1
    assert _thaw_value(record.legacy_frame_values[5]).tobytes() != frames[5].tobytes()


def test_bond_metadata_none_empty_mapping_and_nested_containers_remain_distinct():
    record = _record(bond_metadata=(None, {}, {"nested": [1, (None, {"raw": b"x"})]}))
    assert tuple(map(_thaw_value, record.bond_metadata)) == (
        None, {}, {"nested": [1, (None, {"raw": b"x"})]},
    )


@pytest.mark.parametrize("value", [True, -3, float("nan"), float("inf"),
                                      np.bool_(False), np.int64(-7), np.float32(-0.5),
                                      np.longdouble("1.0000000000000000001")])
def test_molecular_declarations_keep_real_scalar_domain(value):
    record = _record(charge=(True, value, None), spin_mult=(True, value, "raw"))
    expected = value.item() if isinstance(value, np.generic) else value
    assert type(record.charge[1]) is type(expected)
    assert record.charge[1] == expected or np.isnan(expected)
    assert record.spin_mult[2] == "raw"


@pytest.mark.parametrize("value", [1j, np.complex64(1), np.complex128(1), np.array(1.0)])
def test_molecular_declarations_reject_complex_and_array_values(value):
    with pytest.raises(TypeError):
        _record(charge=(True, value, None))
    with pytest.raises(TypeError):
        _record(spin_mult=(True, value, None))


@pytest.mark.parametrize("value", [0, "raw", b"raw", [], (), np.array([1])])
def test_bond_metadata_requires_mapping_or_none_at_factory_and_bytes_boundary(value):
    with pytest.raises(TypeError):
        _record(bond_metadata=(None, {}, value))
    with pytest.raises(TypeError):
        replace(_record(), bond_metadata=(_freeze_value(None), _freeze_value({}), _freeze_value(value)))


@pytest.mark.parametrize("dtype", ["bool", "<i8", ">f8", "complex128", "S3", "U3", "M8[ns]", "V0"])
def test_numpy_codec_keeps_dtype_and_bytes(dtype):
    array = np.zeros((2, 1), dtype=dtype)
    result = _thaw_value(_freeze_value(array))
    assert result.dtype == array.dtype
    assert result.shape == array.shape
    assert result.tobytes() == array.tobytes()
    scalar = array[0, 0]
    result_scalar = _thaw_value(_freeze_value(scalar))
    assert type(result_scalar) is type(scalar)
    assert result_scalar.dtype == scalar.dtype
    assert result_scalar.tobytes() == scalar.tobytes()


def test_stereo_reuses_schema_and_isolates_nested_sequences():
    anchor_rows, carriers, bond_rows = [2, 0], [StereoCarrier(StereoCarrierKind.ATOM, 1)], [2, 0]
    records = [StereoRecord(
        StereoAnchor(StereoAnchorKind.BOND, anchor_rows), carriers, "raw descriptor",
        StereoGroupRef("scope", StereoGroupKind.AND, 9), "source", "code",
    )]
    hydrogens = HydrogenTransportState(None, 0, False)
    facts = [AtomTransportRecord(0, 13, 1, hydrogens, hydrogens, "source")]
    payload = StereoPayload(23, records, facts, [BondTransportRecord(bond_rows, "raw")])
    record = _record(stereo_payload=payload)
    stored = record.stereo_payload
    assert type(stored) is StereoPayload
    assert [field.name for field in fields(stored)] == ["schema_version", "records", "atom_facts", "bond_facts"]
    assert stored.schema_version == 23
    assert type(stored.records) is tuple
    assert type(stored.records[0]) is StereoRecord
    assert type(stored.records[0].carriers) is tuple
    anchor_rows[0] = 3
    carriers.clear()
    bond_rows.reverse()
    records.clear()
    facts.clear()
    assert stored.records[0].anchor.atom_indices == (2, 0)
    assert stored.records[0].carriers == (StereoCarrier(StereoCarrierKind.ATOM, 1),)
    assert stored.bond_facts[0].atom_indices == (2, 0)
    restored = pickle.loads(pickle.dumps(record))
    assert restored.stereo_payload == stored


@pytest.mark.parametrize("value", [object(), np.array([object()]), np.zeros(2, dtype=[("x", "i4")]),
                                      np.dtype("i4"), np.zeros(2, dtype=np.dtype("i4", metadata={"unit": "x"}))])
def test_codec_rejects_opaque_and_nonlossless_dtype_values(value):
    with pytest.raises(TypeError):
        _freeze_value(value)


@pytest.mark.parametrize("changes", [
    {"atom_values": np.zeros((4, 18))}, {"atom_presence": np.zeros((4, 1), dtype=int)},
    {"atom_sources": np.zeros((4, 18), dtype=int)}, {"atom_presence": np.array([-1, 0, 0, 0])},
    {"atom_presence": np.array([1 << 19, 0, 0, 0])}, {"atom_sources": np.full((4, 19), 256)},
    {"atom_sources": np.ones((4, 19))}, {"bond_values": np.zeros((3, 2))},
    {"bond_endpoints": ((0, 4),) * 3}, {"bond_endpoints": ((0,),) * 3},
    {"bond_partitions": (0, 1, 3)}, {"bond_kinds": ("single", "double", "invalid")},
    {"bond_metadata": ({},)}, {"bond_metadata": ({}, {}, object())},
    {"atom_isotopes": ((0, 1), (0, 2))}, {"atom_frac_valences": ((4, 1.5),)},
    {"atom_frac_valences": ((0, 1.5), (0, 2.5))},
    {"charge": (False, 0, None)}, {"spin_mult": (True, 1, [])},
    {"cell": (1, 2, 3)}, {"legacy_frame_values": (None,) * 8},
    {"legacy_frame_index": None}, {"angle_constraints": ((0, 1),)},
    {"torsion_constraints": ((0, 1, 2, 4),)}, {"identifier": (False, "x")},
])
def test_bad_import_is_rejected_without_mutating_inputs(changes):
    inputs = _inputs() | changes
    values = inputs["atom_values"].copy()
    bonds = inputs["bond_values"].copy()
    with pytest.raises((TypeError, ValueError)):
        _make_molecule_record(**inputs)
    np.testing.assert_array_equal(inputs["atom_values"], values)
    np.testing.assert_array_equal(inputs["bond_values"], bonds)


@pytest.mark.parametrize("changes", [
    {"atom_values": b"bad"}, {"atom_presence": b"bad"}, {"atom_sources": bytes([255]) * 76},
    {"bond_values": b"bad"}, {"bond_metadata": (("opaque", 1),) * 3},
    {"legacy_frame_values": (("ndarray", "<i8", (2,), b""),) * 9},
])
def test_canonical_constructor_revalidates_buffers_and_encoded_values(changes):
    record = _record()
    with pytest.raises((TypeError, ValueError)):
        replace(record, **changes)


def test_empty_record_and_transport_only_self_parallel_bonds():
    record = _record(
        atom_values=np.empty((0, 19)), atom_presence=np.empty(0, dtype=np.uint32),
        atom_sources=np.empty((0, 19), dtype=np.uint8), atom_isotopes=(), atom_frac_valences=(),
        bond_endpoints=(), bond_values=np.empty((0, 3)), bond_partitions=(), bond_kinds=(),
        bond_directions=(), bond_sources=(), bond_metadata=(), angle_constraints=(), torsion_constraints=(),
    )
    assert record.atom_count == 0 and record.atom_values == b""
    assert _record(bond_endpoints=((0, 0), (0, 1), (0, 1))).bond_endpoints == ((0, 0), (0, 1), (0, 1))


def _typed_stereo_payload():
    hydrogens = HydrogenTransportState(None, 0, False)
    group = StereoGroupRef("scope", StereoGroupKind.OR, 2)
    record = StereoRecord(
        StereoAnchor(StereoAnchorKind.ATOM, (0,)),
        (StereoCarrier(StereoCarrierKind.ATOM, 1),),
        "uninterpreted descriptor", group, "namespace", "code",
    )
    atom = AtomTransportRecord(0, None, 1, hydrogens, hydrogens, None)
    bond = BondTransportRecord((0, 1), "wedge")
    return StereoPayload(23, (record,), (atom,), (bond,))


def _bad_stereo_payloads():
    payload = _typed_stereo_payload()
    record, atom, bond = payload.records[0], payload.atom_facts[0], payload.bond_facts[0]
    cases = [
        replace(payload, schema_version="1"),
        replace(payload, records=(bond,)),
        replace(payload, atom_facts=(record,)),
        replace(payload, bond_facts=(atom,)),
        replace(payload, records={}),
    ]
    for changes in (
            {"anchor": record.carriers[0]}, {"carriers": (record.anchor,)},
            {"carriers": {}}, {"group": atom.current_hydrogens},
            {"descriptor": b"raw"}, {"source_namespace": None}, {"source_code": 1},
            {"anchor": replace(record.anchor, kind=StereoCarrierKind.ATOM)},
            {"anchor": replace(record.anchor, atom_indices=(0, "1"))},
            {"anchor": replace(record.anchor, atom_indices="0")},
            {"carriers": (replace(record.carriers[0], kind=StereoAnchorKind.ATOM),)},
            {"carriers": (replace(record.carriers[0], atom="1"),)},
            {"group": replace(record.group, scope=0)},
            {"group": replace(record.group, kind=StereoAnchorKind.ATOM)},
            {"group": replace(record.group, id="2")},
    ):
        cases.append(replace(payload, records=(replace(record, **changes),)))
    for changes in (
            {"atom": "0"}, {"isotope": "13"}, {"radical_electrons": 1.0},
            {"current_hydrogens": record.group}, {"source_hydrogens": record.group},
            {"source_namespace": b"namespace"},
            {"current_hydrogens": replace(atom.current_hydrogens, implicit_hydrogens="1")},
            {"current_hydrogens": replace(atom.current_hydrogens, bracket_explicit_hydrogens=1.0)},
            {"current_hydrogens": replace(atom.current_hydrogens, no_implicit=1)},
    ):
        cases.append(replace(payload, atom_facts=(replace(atom, **changes),)))
    for changes in ({"atom_indices": (0,)}, {"atom_indices": (0, "1")}, {"display_direction": 1}):
        cases.append(replace(payload, bond_facts=(replace(bond, **changes),)))
    return cases


@pytest.mark.parametrize("payload", _bad_stereo_payloads())
def test_stereo_rejects_misplaced_nodes_and_wrong_member_types(payload):
    with pytest.raises((TypeError, ValueError)):
        _record(stereo_payload=payload)
    with pytest.raises((TypeError, ValueError)):
        replace(_record(), stereo_payload=payload)
    # A hostile current-version pickle still passes through constructor validation.
    unvalidated = _record()
    object.__setattr__(unvalidated, "stereo_payload", payload)
    with pytest.raises((TypeError, ValueError)):
        pickle.loads(pickle.dumps(unvalidated))


def test_stereo_schema_acceptance_does_not_interpret_scientific_members():
    payload = _typed_stereo_payload()
    record = payload.records[0]
    payload = replace(
        payload, schema_version=-7,
        records=(replace(
            record, anchor=replace(record.anchor, atom_indices=(-3, -3, 900)),
            carriers=(StereoCarrier(StereoCarrierKind.IMPLICIT_H, -9),),
            group=replace(record.group, id=-2), descriptor="opaque unsupported descriptor",
        ),),
    )
    assert _record(stereo_payload=payload).stereo_payload == payload


@pytest.mark.parametrize("shape,data", [
    ((1,) * 65, bytes(8)), ((2**64, 0), b""), ((0, 2**62), b""),
])
def test_canonical_array_header_must_be_decodable_at_constructor_and_thaw(shape, data):
    encoded = ("ndarray", "<f8", shape, data)
    with pytest.raises((TypeError, ValueError, OverflowError)):
        replace(_record(), legacy_frame_values=(encoded,) + (_freeze_value(None),) * 8)
    with pytest.raises((TypeError, ValueError, OverflowError)):
        _thaw_value(encoded)


def test_factual_builder_installs_every_raw_fact_and_builds_topology_once(monkeypatch):
    from hotpot import Atom, Molecule
    import importlib
    core = importlib.import_module(Molecule.__module__)
    calls = []
    original = core._build_topology_index
    def counted(*args, **kwargs):
        calls.append(len(args[0]))
        return original(*args, **kwargs)
    def forbidden(*args, **kwargs):
        raise AssertionError('factual builder ran chemistry')
    monkeypatch.setattr(core, '_build_topology_index', counted)
    monkeypatch.setattr(Atom, 'get_valence', forbidden)
    monkeypatch.setattr(Atom, '_calc_implicit_hydrogens', forbidden)
    monkeypatch.setattr(Molecule, 'assign_bond_order', forbidden)
    monkeypatch.setattr(Molecule, 'determine_rings_aromatic', forbidden)
    record = _record()
    mol = Molecule._build_molecule(record)
    assert calls == [4]
    assert mol.atom_matrix.tobytes() == record.atom_values
    assert mol._atom_table.presence().tobytes() == record.atom_presence
    assert mol._atom_table.sources().tobytes() == record.atom_sources
    complete = mol._bonds + mol._hided_metal_bonds + mol._hided_covalent_bonds
    assert b''.join(bond.attrs.tobytes() for bond in complete) == record.bond_values
    assert tuple((bond.atom1.idx, bond.atom2.idx) for bond in complete) == record.bond_endpoints
    assert (len(mol._bonds), len(mol._hided_metal_bonds), len(mol._hided_covalent_bonds)) == (1, 1, 1)
    assert mol.atoms[0].isotope is None and mol.atoms[1].isotope == 0
    assert mol.atoms[2].isotope == 13 and mol.atoms[0].frac_valence == 1.75
    assert 'isotope' not in mol.atoms[3].__dict__
    assert [tuple(atom.idx for atom in view.atoms) for view in mol._angles if view.constraint] == [(2, 0, 1)]
    assert [tuple(atom.idx for atom in view.atoms) for view in mol._torsions if view.constraint] == [(3, 2, 0, 1)]


def test_same_target_factual_installer_preserves_owner_lock_and_burned_uid_ranges():
    from hotpot import Molecule
    mol = Molecule()
    owner, lock = mol._owner_token, mol._mutation_lock
    mol._next_atom_uid, mol._next_bond_uid = 10, 20
    assert mol._install_molecule(_record()) is mol
    assert mol._owner_token is owner and mol._mutation_lock is lock
    assert [atom._uid for atom in mol.atoms] == [10, 11, 12, 13]
    assert [bond._uid for bond in mol._bonds + mol._hided_bonds] == [20, 21, 22]
    assert mol._charge_present and mol.charge == 0
    assert not mol._spin_present


def test_factual_installer_preserves_all_nine_frame_domains_and_selection():
    from hotpot import Molecule
    frames = (np.arange(24.).reshape(2, 4, 3), np.array([3., 4.]),
              np.arange(8.).reshape(2, 4), 1.25, np.arange(24.).reshape(2, 4, 3),
              np.float32(0.5), 3, np.array(0.75), None)
    mol = Molecule._build_molecule(_record(legacy_frame_values=frames, legacy_frame_index=-1))
    assert mol._conformers_index == -1
    for name, expected in zip(mol.conformers._attrs, frames):
        observed = getattr(mol.conformers, '_' + name)
        if expected is None:
            assert observed is None
        else:
            np.testing.assert_array_equal(observed, expected)
            if isinstance(expected, np.ndarray):
                assert not np.shares_memory(observed, expected)


def test_installer_rolls_back_after_published_topology_and_does_not_reuse_uids(monkeypatch):
    from hotpot import Molecule
    mol = Molecule()
    mol.identifier = 'before'
    index, table = mol._topology_index, mol._atom_table
    original = Molecule._update_topology
    def fail(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise KeyboardInterrupt('after topology publication')
    monkeypatch.setattr(Molecule, '_update_topology', fail)
    with pytest.raises(KeyboardInterrupt):
        mol._install_molecule(_record(identifier=(True, 'candidate')))
    assert not mol.atoms and not mol.bonds
    assert mol._topology_index is index and mol._atom_table is table
    assert mol.identifier == 'before'
    assert mol._next_atom_uid == 4 and mol._next_bond_uid == 3


def test_opaque_metadata_channel_requires_complete_mutually_exclusive_values():
    from hotpot import Molecule
    record = _record(bond_metadata=(None,) * 3)
    with pytest.raises(ValueError, match='complete'):
        Molecule._build_molecule(record, bond_metadata=({},))
    with pytest.raises(ValueError, match='exclude record metadata'):
        Molecule._build_molecule(_record(), bond_metadata=({}, {}, {}))
    marker = object()
    values = ({'opaque': marker}, {}, {})
    mol = Molecule._build_molecule(record, bond_metadata=deepcopy(values))
    assert mol.bonds[0]._bond_source_metadata['opaque'] is not marker
    clone = mol.copy()
    assert clone.bonds[0]._bond_source_metadata['opaque'] is not mol.bonds[0]._bond_source_metadata['opaque']
