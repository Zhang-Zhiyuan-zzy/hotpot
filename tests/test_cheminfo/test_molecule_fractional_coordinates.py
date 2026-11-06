"""Paired fractional access preserves the sole Cartesian fact owner."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import pickle
from threading import Event

import numpy as np
import pytest

from hotpot import Crystal, Molecule, SpaceGroup
from hotpot.cheminfo.structure._commit import _FactTransaction, _native_fail_after
from hotpot.cheminfo.structure._dependencies import _dependency_state
from hotpot.cheminfo.structure._identity import _capture_source, _check_source


# A skew row basis rotated by 90 degrees about z; expected Cartesian values
# below are independent sums of its three basis rows, not a conversion oracle.
MATRIX = np.array(((0., 2., 0.), (-3., 1., 0.), (-.75, .5, 4.)))
FRACTIONAL = np.array(((.25, -.5, 1.25), (1.5, .125, -.25)))
CARTESIAN = np.array(((.5625, .625, 5.), (-.1875, 3., -1.)))


def _molecule():
    mol = Molecule()
    for number, xyz in zip((6, 8), CARTESIAN):
        mol.create_atom(atomic_number=number, coordinates=xyz,
                        formal_charge=0, implicit_hydrogens=0)
    mol.add_bond(0, 1)
    Crystal.from_matrix(MATRIX, molecule=mol, space_group=SpaceGroup(1))
    return mol


def _fact_state(mol):
    record = mol._capture_factual_record()[0]
    status, seeds = _dependency_state(mol._atom_table)
    return (record.atom_values, record.atom_presence, record.atom_sources,
            record.cell, status.tobytes(), seeds)


def _select_energy(mol):
    mol.conformer_add(mol.coordinates, energy=3.)
    mol.conformer_load(0)
    return mol._frame_result, mol._source_frames


def test_analytic_oriented_skew_projection_is_unwrapped_isolated_and_read_only():
    mol = _molecule()
    source = _capture_source(mol, ('geometry', 'cell'))
    before, fields = _fact_state(mol), set(vars(mol))
    first, second = mol.fractional_coordinates, mol.fractional_coordinates
    np.testing.assert_array_equal(mol.coordinates, CARTESIAN)
    np.testing.assert_allclose(first, FRACTIONAL, rtol=0, atol=1e-10)
    assert first.shape == (2, 3) and first.dtype == np.dtype(np.float64)
    assert not np.shares_memory(first, second)
    assert not np.shares_memory(first, mol.coordinates)
    first[:] = 99
    np.testing.assert_allclose(mol.fractional_coordinates, FRACTIONAL, rtol=0, atol=1e-10)
    assert _fact_state(mol) == before and set(vars(mol)) == fields
    _check_source(mol, source)


def test_fractional_assignment_commits_only_cartesian_facts_and_isolates_input():
    mol = _molecule()
    cell, table, fields = mol.crystal, mol._atom_table, set(vars(mol))
    before = cell._state
    supplied = FRACTIONAL * 2
    mol.fractional_coordinates = supplied
    np.testing.assert_array_equal(mol.coordinates, CARTESIAN * 2)
    np.testing.assert_array_equal([atom.coordinates for atom in mol.atoms], CARTESIAN * 2)
    supplied[:] = 999
    np.testing.assert_allclose(mol.fractional_coordinates, FRACTIONAL * 2, rtol=0, atol=1e-10)
    assert mol._atom_table is table and mol.crystal is cell and cell._state is before
    assert set(vars(mol)) == fields


@pytest.mark.parametrize('writer', ('molecule', 'atom', 'batch'))
def test_fractional_reads_follow_actual_cartesian_writers(writer):
    mol = _molecule()
    if writer == 'molecule':
        mol.coordinates = CARTESIAN * 2
    elif writer == 'atom':
        for atom, xyz in zip(mol.atoms, CARTESIAN * 2):
            atom.coordinates = xyz
    else:
        mol.updata_atoms(coordinates=CARTESIAN * 2)
    np.testing.assert_allclose(mol.fractional_coordinates, FRACTIONAL * 2, rtol=0, atol=1e-10)


def test_fractional_reads_follow_both_cell_preservation_modes_and_scalar_changes():
    mol = _molecule()
    cell = mol.crystal
    cell.set_cell(MATRIX * 2)
    np.testing.assert_array_equal(mol.coordinates, CARTESIAN)
    np.testing.assert_allclose(mol.fractional_coordinates, FRACTIONAL / 2, rtol=0, atol=1e-10)
    cell.set_cell(MATRIX, preserve='fractional')
    np.testing.assert_allclose(mol.coordinates, CARTESIAN / 2, rtol=0, atol=1e-10)
    np.testing.assert_allclose(mol.fractional_coordinates, FRACTIONAL / 2, rtol=0, atol=1e-10)
    cell.a = 4.
    np.testing.assert_allclose(cell.matrix[0], (0., 4., 0.), rtol=0, atol=1e-10)
    expected = FRACTIONAL / 2
    expected[:, 0] /= 2
    np.testing.assert_allclose(mol.fractional_coordinates, expected, rtol=0, atol=1e-10)
    np.testing.assert_allclose(mol.coordinates, CARTESIAN / 2, rtol=0, atol=1e-10)


def test_fractional_access_uses_replacement_crystal_and_its_current_basis():
    mol = _molecule()
    old = mol.crystal
    new = mol.create_crystal(4., 8., 16., 90., 90., 90.)
    assert old.molecule is None and mol.crystal is new and new.molecule is mol
    np.testing.assert_allclose(mol.fractional_coordinates, CARTESIAN / (4, 8, 16), rtol=0, atol=1e-10)
    mol.fractional_coordinates = FRACTIONAL
    np.testing.assert_allclose(mol.coordinates, FRACTIONAL * (4, 8, 16), rtol=0, atol=1e-10)
    np.testing.assert_array_equal(old.matrix, MATRIX)


@pytest.mark.parametrize('operation', ('read', 'write'))
def test_fractional_access_without_crystal_is_a_named_local_failure(operation):
    mol = Molecule()
    mol.create_atom(atomic_number=6, coordinates=(0., 0., 0.))
    before = _fact_state(mol)
    with pytest.raises(ValueError, match='CRYSTAL-CONTEXT-INTEGRITY:.*associated Crystal'):
        if operation == 'read':
            _ = mol.fractional_coordinates
        else:
            mol.fractional_coordinates = np.zeros((1, 3))
    assert mol.crystal is None and _fact_state(mol) == before
    np.testing.assert_array_equal(mol.coordinates, ((0., 0., 0.),))


@pytest.mark.parametrize('supplied', ({}, {'x': 0.}, {'x': 0., 'y': 0.}))
def test_fractional_getter_requires_supplied_xyz_and_setter_can_supply_them(supplied):
    mol = Molecule()
    mol.create_atom(atomic_number=6, **supplied)
    Crystal.from_matrix(MATRIX, molecule=mol)
    before = _fact_state(mol)
    with pytest.raises(ValueError, match='complete supplied Cartesian coordinates'):
        _ = mol.fractional_coordinates
    assert _fact_state(mol) == before
    mol.fractional_coordinates = FRACTIONAL[:1]
    np.testing.assert_array_equal(mol.coordinates, CARTESIAN[:1])
    np.testing.assert_allclose(mol.fractional_coordinates, FRACTIONAL[:1], rtol=0, atol=1e-10)


def test_empty_paired_molecule_retains_empty_coordinate_shape():
    mol = Molecule()
    Crystal.from_matrix(MATRIX, molecule=mol)
    assert mol.coordinates.shape == mol.fractional_coordinates.shape == (0, 3)
    mol.fractional_coordinates = np.empty((0, 3))
    assert mol.coordinates.shape == mol.fractional_coordinates.shape == (0, 3)
    assert not mol.atoms


def test_fractional_assignment_invalidates_geometry_results_and_preserves_source_frames():
    mol = _molecule()
    selected, frames = _select_energy(mol)
    geometry, cell = _capture_source(mol, ('geometry',)), _capture_source(mol, ('cell',))
    mol.fractional_coordinates = FRACTIONAL * 2
    assert mol._source_frames is frames
    np.testing.assert_array_equal(mol.conformer_get(0)['coordinates'], CARTESIAN)
    assert selected.invalidated_by == ()
    assert mol._frame_result.invalidated_by == ('geometry',)
    with pytest.raises(RuntimeError, match='frame_observation_stale'):
        _ = mol.energy
    with pytest.raises(RuntimeError, match='geometry'):
        _check_source(mol, geometry)
    _check_source(mol, cell)


@pytest.mark.parametrize('boundary', ('value', 'result_invalidation'))
def test_fractional_native_publication_failure_restores_raw_facts_and_selected_result(boundary):
    mol = _molecule()
    selected, frames = _select_energy(mol)
    before, state = _fact_state(mol), mol.crystal._state
    with _native_fail_after(boundary), pytest.raises(KeyboardInterrupt, match='native fact store'):
        mol.fractional_coordinates = FRACTIONAL * 2
    assert _fact_state(mol) == before and mol.crystal._state is state
    assert mol._frame_result is selected and mol._source_frames is frames
    assert mol.energy == 3.
    np.testing.assert_allclose(mol.fractional_coordinates, FRACTIONAL, rtol=0, atol=1e-10)


def test_outer_failure_restores_fractional_access_cell_association_and_result_currentness():
    mol = _molecule()
    selected, frames = _select_energy(mol)
    cell, before, state = mol.crystal, _fact_state(mol), mol.crystal._state
    with pytest.raises(KeyboardInterrupt, match='fractional rollback'):
        with _FactTransaction(mol, producer=1):
            cell.set_cell(MATRIX * 2, preserve='fractional')
            mol.fractional_coordinates = FRACTIONAL * 3
            replacement = mol.create_crystal(5., 6., 7., 90., 90., 90.)
            mol.fractional_coordinates = FRACTIONAL * 4
            raise KeyboardInterrupt('fractional rollback')
    assert mol.crystal is cell and cell.molecule is mol and cell._state is state
    assert replacement.molecule is None and _fact_state(mol) == before
    assert mol._frame_result is selected and mol._source_frames is frames and mol.energy == 3.
    np.testing.assert_allclose(mol.fractional_coordinates, FRACTIONAL, rtol=0, atol=1e-10)


@pytest.mark.parametrize('boundary', ('molecule_copy', 'crystal_copy', 'deepcopy', 'record'))
def test_fractional_projection_is_reconstructed_from_independent_copied_facts(boundary):
    mol = _molecule()
    if boundary == 'molecule_copy':
        restored = mol.copy()
    elif boundary == 'crystal_copy':
        restored = mol.crystal.copy().molecule
    elif boundary == 'deepcopy':
        restored, cell = deepcopy((mol, mol.crystal))
        assert restored.crystal is cell
    else:
        restored = Molecule._build_molecule(mol._capture_factual_record()[0])
    assert restored.crystal is not mol.crystal and restored.crystal.molecule is restored
    np.testing.assert_array_equal(restored.coordinates, CARTESIAN)
    np.testing.assert_allclose(restored.fractional_coordinates, FRACTIONAL, rtol=0, atol=1e-10)
    restored.crystal.set_cell(MATRIX * 2)
    restored.fractional_coordinates = FRACTIONAL
    np.testing.assert_array_equal(restored.coordinates, CARTESIAN * 2)
    np.testing.assert_array_equal(mol.coordinates, CARTESIAN)
    np.testing.assert_array_equal(mol.crystal.matrix, MATRIX)


@pytest.mark.parametrize('protocol', range(pickle.HIGHEST_PROTOCOL + 1))
@pytest.mark.parametrize('crystal_first', (False, True))
def test_combined_pickle_preserves_active_pair_and_rederives_fractional_values(protocol, crystal_first):
    mol = _molecule()
    cell = mol.crystal
    pair = (cell, mol) if crystal_first else (mol, cell)
    first, second, atom, bond = pickle.loads(pickle.dumps((*pair, mol.atoms[0], mol.bonds[0]), protocol=protocol))
    restored, restored_cell = (second, first) if crystal_first else (first, second)
    assert restored.crystal is restored_cell and restored_cell.molecule is restored
    assert atom is restored.atoms[0] and bond is restored.bonds[0]
    np.testing.assert_allclose(restored.fractional_coordinates, FRACTIONAL, rtol=0, atol=1e-10)
    atom.coordinates = CARTESIAN[0] * 2
    np.testing.assert_allclose(restored.fractional_coordinates[0], FRACTIONAL[0] * 2, rtol=0, atol=1e-10)
    np.testing.assert_array_equal(mol.coordinates, CARTESIAN)


def test_cif_roundtrip_uses_current_cartesian_and_oriented_cell_facts():
    pytest.importorskip('gemmi')
    mol = _molecule()
    mol.fractional_coordinates = FRACTIONAL * 2
    mol.crystal.set_cell(MATRIX * 2)
    restored = Crystal.from_cif(mol.crystal.to_cif()).molecule
    np.testing.assert_allclose(restored.crystal.matrix, MATRIX * 2, rtol=0, atol=1e-10)
    np.testing.assert_allclose(restored.coordinates, CARTESIAN * 2, rtol=0, atol=1e-10)
    np.testing.assert_allclose(restored.fractional_coordinates, FRACTIONAL, rtol=0, atol=1e-10)


@pytest.mark.parametrize('access', ('read', 'write'))
def test_fractional_access_serializes_cell_changes_across_conversion(monkeypatch, access):
    mol = _molecule()
    cell = mol.crystal
    converting, release, attempted, changed = Event(), Event(), Event(), Event()
    method = 'to_fractional' if access == 'read' else 'to_cartesian'
    convert = getattr(Crystal, method)

    def pause(current, values):
        converting.set()
        assert release.wait(5)
        return convert(current, values)

    def fractional_access():
        if access == 'read':
            return mol.fractional_coordinates
        mol.fractional_coordinates = FRACTIONAL * 2

    def change_cell():
        attempted.set()
        cell.set_cell(MATRIX * 2)
        changed.set()

    monkeypatch.setattr(Crystal, method, pause)
    with ThreadPoolExecutor(max_workers=2) as pool:
        future = pool.submit(fractional_access)
        assert converting.wait(5)
        writer = pool.submit(change_cell)
        try:
            assert attempted.wait(5)
            assert not changed.wait(.05)
        finally:
            release.set()
        result = future.result(timeout=5)
        writer.result(timeout=5)
    if access == 'read':
        np.testing.assert_allclose(result, FRACTIONAL, rtol=0, atol=1e-10)
        np.testing.assert_array_equal(mol.coordinates, CARTESIAN)
    else:
        np.testing.assert_array_equal(mol.coordinates, CARTESIAN * 2)
    np.testing.assert_allclose(mol.fractional_coordinates,
                               FRACTIONAL / 2 if access == 'read' else FRACTIONAL,
                               rtol=0, atol=1e-10)
