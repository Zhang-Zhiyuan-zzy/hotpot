"""Complete force-field publication and field-selective model consumers."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from hotpot import Atom, Molecule, read_mol
from hotpot.cheminfo.AImodels.data_extract import extract_atom_attrs
from hotpot.cheminfo.forcefields import working_copy
from hotpot.cheminfo.structure import _commit
from hotpot.cheminfo.structure._dependencies import _dependency_state
from hotpot.cheminfo.structure._fields import ATOM_FIELDS, ATOM_PRESENCE_ALL


def _molecule():
    mol = Molecule()
    mol.create_atom(atomic_number=6, id=10)
    mol.create_atom(atomic_number=8, id=20)
    mol.add_bond(0, 1, bond_order=1)
    return mol


@pytest.mark.parametrize("count", [0, 1, 4])
def test_model_extraction_preserves_raw_features_and_requests_selected_fields(monkeypatch, count):
    mol = Molecule()
    rows = np.arange(count * 19, dtype=np.float64).reshape(count, 19) + 0.25
    rows[:, 0] = np.resize([6.5, 30.5, 8.5, 7.5], count)
    for row in rows:
        Atom(mol=mol, attrs_array=row, update_electron_config=False)
    expected = np.empty((count, 16), dtype=np.float64)
    expected[:, :15] = rows[:, :15]
    expected[:, 15] = [atom.is_metal for atom in mol.atoms]

    def forbidden(*args, **kwargs):
        raise AssertionError("selected export must not construct full rows or matrices")

    monkeypatch.setattr(Atom, "attrs", property(forbidden))
    monkeypatch.setattr(type(mol._atom_table), "matrix", forbidden)
    monkeypatch.setattr(type(mol._atom_table), "row", forbidden)
    extracted = extract_atom_attrs(mol)
    assert extracted["x_names"] == ATOM_FIELDS[:15] + ("is_metal",)
    np.testing.assert_array_equal(extracted["x"], expected)
    atomic_numbers = extract_atom_attrs(mol, atomic_number_only=True)
    assert atomic_numbers["x"].dtype == np.int32
    np.testing.assert_array_equal(atomic_numbers["x"], rows[:, 0].astype(np.int32))


@pytest.mark.parametrize("field,value", [
    ("x", 9), ("formal_charge", 1), ("partial_charge", 0.25),
    ("valence", 5), ("s", 1), ("frac_valence", 1.75), ("isotope", 13),
])
def test_forcefield_rejects_input_changes_since_working_copy_creation(field, value):
    mol = _molecule()
    source = working_copy._capture_working_source(mol)
    working = working_copy._hydrogenated_working_copy(mol, add_hydrogens=False, source=source)
    working.coordinates = np.ones((2, 3))
    setattr(mol.atoms[0], field, value)
    previous = mol.atom_matrix
    with pytest.raises(RuntimeError, match="changed"):
        working_copy._commit_working_copy(mol, working, source=source)
    np.testing.assert_array_equal(mol.atom_matrix, previous)


def test_forcefield_rejects_changed_input_before_making_a_clone(monkeypatch):
    mol = _molecule()
    source = working_copy._capture_working_source(mol)
    mol.atoms[0].x = 4

    def forbidden(*args, **kwargs):
        raise AssertionError("stale source must be checked before copying")

    monkeypatch.setattr(working_copy, "copy", forbidden)
    with pytest.raises(RuntimeError, match="changed"):
        working_copy._hydrogenated_working_copy(mol, add_hydrogens=False, source=source)


def test_forcefield_rejects_replaced_working_members_with_identical_source_labels():
    mol = Molecule()
    for x in (1, 2):
        mol.create_atom(atomic_number=6, x=x)
    source = working_copy._capture_working_source(mol)
    working = working_copy._hydrogenated_working_copy(mol, add_hydrogens=False, source=source)
    working.remove_atom(working.atoms[0])
    working.create_atom(atomic_number=6, x=9)
    previous = mol.atom_matrix
    assert [atom.id for atom in working.atoms] == [atom.id for atom in mol.atoms]
    with pytest.raises(ValueError, match="original atom membership"):
        working_copy._commit_working_copy(mol, working, source=source)
    np.testing.assert_array_equal(mol.atom_matrix, previous)


class _MetadataValue:
    def __init__(self, value):
        self.value = value


def test_forcefield_accepts_opaque_metadata_and_rejects_replaced_metadata_owner():
    mol = _molecule()
    bond = mol.bonds[0]
    bond._set_bond_metadata(bond_kind=bond.bond_kind,
                            bond_source_metadata={"opaque": _MetadataValue(3)})
    source = working_copy._capture_working_source(mol)
    working = working_copy._hydrogenated_working_copy(mol, add_hydrogens=False, source=source)
    working.coordinates = np.full((2, 3), 2)
    working_copy._commit_working_copy(mol, working, source=source)
    assert mol.bonds[0] is bond
    assert bond.bond_source_metadata["opaque"].value == 3
    source = working_copy._capture_working_source(mol)
    working = working_copy._hydrogenated_working_copy(mol, add_hydrogens=False, source=source)
    bond._set_bond_metadata(bond_kind=bond.bond_kind,
                            bond_source_metadata={"opaque": _MetadataValue(4)})
    with pytest.raises(RuntimeError, match="metadata changed"):
        working_copy._commit_working_copy(mol, working, source=source)
    assert bond.bond_source_metadata["opaque"].value == 4


def test_forcefield_preserves_heterogeneous_metadata_and_all_frame_values():
    mol = _molecule()
    atoms, bond = tuple(mol.atoms), mol.bonds[0]
    source = working_copy._capture_working_source(mol)
    working = working_copy._hydrogenated_working_copy(mol, add_hydrogens=False, source=source)
    table = working._atom_table
    table._presence[:2] = [1, ATOM_PRESENCE_ALL]
    table._sources[:2] = np.arange(38, dtype=np.uint8).reshape(2, 19) % 10
    expected_presence, expected_sources = table.presence(), table.sources()
    frames = (
        np.arange(12.0).reshape(2, 2, 3), np.array([-1., -2.]),
        np.array([[0.1, -0.1], [0.2, -0.2]]), np.float64(3),
        np.arange(12.0).reshape(2, 2, 3), 4.5, np.int32(1),
        np.array([4.]), np.array([5., 6.]),
    )
    working._install_frame_state(frames, -1)
    working_copy._commit_working_copy(mol, working, source=source)
    assert all(current is previous for current, previous in zip(mol.atoms, atoms))
    assert mol.bonds[0] is bond
    np.testing.assert_array_equal(mol._atom_table.presence(), expected_presence)
    np.testing.assert_array_equal(mol._atom_table.sources(), expected_sources)
    observed, index = mol._capture_frame_state()
    assert index == -1
    for actual, expected in zip(observed, frames):
        np.testing.assert_array_equal(actual, expected)


def test_forcefield_interrupted_publication_restores_owner_metadata_frames_and_status(monkeypatch):
    mol = _molecule()
    mol.conformer_add(np.ones((2, 3)), -1)
    source = working_copy._capture_working_source(mol)
    working = working_copy._hydrogenated_working_copy(mol, add_hydrogens=False, source=source)
    hydrogen = working.create_atom(atomic_number=1)
    working.add_bond(working.atoms[0], hydrogen)
    working.coordinates = working.coordinates + 1
    table, index = mol._atom_table, mol._topology_index
    atoms, bonds = tuple(mol.atoms), tuple(mol.bonds)
    expected = mol.atom_matrix
    expected_presence, expected_sources = table.presence(), table.sources()
    expected_status = _dependency_state(table)
    frames = deepcopy(mol.conformers[0])
    high_water = mol._next_atom_uid

    commit_stereo = _commit._commit_stereo

    def interrupt(owner, source, payload):
        commit_stereo(owner, source, payload)
        raise KeyboardInterrupt("interrupted after factual publication")

    monkeypatch.setattr(_commit, "_commit_stereo", interrupt)
    with pytest.raises(KeyboardInterrupt, match="after factual publication"):
        working_copy._commit_working_copy(mol, working, source=source)
    assert mol._atom_table is table
    assert mol._topology_index is index
    assert tuple(mol.atoms) == atoms and tuple(mol.bonds) == bonds
    assert mol._next_atom_uid > high_water
    np.testing.assert_array_equal(mol.atom_matrix, expected)
    np.testing.assert_array_equal(table.presence(), expected_presence)
    np.testing.assert_array_equal(table.sources(), expected_sources)
    np.testing.assert_array_equal(_dependency_state(table)[0], expected_status[0])
    assert _dependency_state(table)[1] == expected_status[1]
    for field, value in frames.items():
        np.testing.assert_array_equal(mol.conformers[0][field], value)


def test_native_selected_geometry_failure_restores_its_topology_selection():
    from hotpot.cheminfo.forcefields.native_adapters import apply_native_selected_structure
    from hotpot.cheminfo.forcefields.native_packing import pack_complex_session_input

    mol = read_mol("[Eu]N", "smi")
    session = pack_complex_session_input(mol)
    bond, previous = mol.bonds[0], mol.atom_matrix
    result = SimpleNamespace(
        final_active_coordination_mask=np.zeros(1, dtype=np.uint8),
        selected_coordinates=np.ones((len(mol.atoms) + 1, 3)),
    )
    with pytest.raises(AssertionError):
        apply_native_selected_structure(mol, session, result)
    assert mol.bonds == [bond] and mol._hided_metal_bonds == []
    np.testing.assert_array_equal(mol.atom_matrix, previous)
