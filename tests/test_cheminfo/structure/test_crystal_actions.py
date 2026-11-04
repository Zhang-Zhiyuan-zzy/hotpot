"""Independent affine placements, periodic facts and result-owner boundaries."""

import builtins
from dataclasses import replace
import pickle

import numpy as np
import pytest

from hotpot import Crystal, Molecule, SpaceGroup, SymmUnit
from hotpot.cheminfo import stereo
from hotpot.cheminfo.structure._fields import FIELD_INDEX, FIELD_PRESENCE_BITS
from hotpot.cheminfo.structure._identity import _capture_source, _check_source
from hotpot.cheminfo.structure._results import (
    _capture_mca_source, _import_partial_charges, _select_mca_result,
)
from hotpot.cheminfo.operations.inference.mca.results import AtomPrediction, MoleculePrediction


I = ((1, 0, 0), (0, 1, 0), (0, 0, 1))


def molecule():
    value = Molecule()
    for number, xyz in ((6, (1., 2., 3.)), (7, (2., 1., 0.)), (8, (-1., 4., 1.))):
        value.create_atom(atomic_number=number, coordinates=xyz, formal_charge=0,
                          implicit_hydrogens=0)
    value.add_bond(0, 1, bond_order=1, bond_kind='dative', bond_direction='1->2')
    value.add_bond(1, 2, bond_order=2)
    value.charge = 0
    value.spin_mult = 1
    return value


def crystal(group=2):
    mol = molecule()
    return Crystal.from_matrix(((0., 5., 0.), (-6., 0., 0.), (0., 0., 7.)), molecule=mol,
        space_group=SpaceGroup(group), bond_images=((1, -2, 0), None))


def fields(mol):
    record = mol._capture_factual_record()[0]
    return (record.atom_values, record.atom_presence, record.atom_sources,
            record.bond_endpoints, record.bond_values, record.bond_kinds,
            record.bond_directions, record.charge, record.spin_mult, record.cell,
            record.stereo_payload)


@pytest.mark.parametrize('translation', [(0, 0, 0), (2, -3, 1), (-7, 5, -2)])
def test_full_unwrapped_translation_one_copy_same_cell_group_and_isolated_source(translation):
    cell = crystal(1)
    before, original = fields(cell.molecule), cell.molecule.coordinates
    operation = SymmUnit.from_affine(I, translation, context=cell.space_group)
    result = cell.operate(operation)
    np.testing.assert_allclose(result.molecule.coordinates,
                               original + np.asarray(translation) @ cell.matrix, atol=1e-10)
    assert len(result.molecule.atoms) == 3 and len(result.molecule.bonds) == 2
    np.testing.assert_array_equal(result.matrix, cell.matrix)
    assert result.space_group == cell.space_group and result.symm_units == cell.symm_units
    assert result.molecule is not cell.molecule and result.molecule.crystal is result
    assert result.bond_images == cell.bond_images
    assert fields(cell.molecule) == before
    result.molecule.atoms[0].x += 10
    np.testing.assert_array_equal(cell.molecule.coordinates, original)
    assert result.molecule.bonds[0].bond_kind == cell.molecule.bonds[0].bond_kind
    assert result.molecule.bonds[0].bond_direction == '1->2'
    assert result.molecule.charge == 0 and result.molecule.spin_mult == 1


def test_inversion_and_second_action_equal_full_same_group_composition():
    cell = crystal()
    first = SymmUnit.from_xyz('-x+2,-y-1,-z', context=cell.space_group)
    second = SymmUnit.from_xyz('-x,-y+3,-z+2', context=cell.space_group)
    before = fields(cell.molecule)
    result = cell.operate(first)
    expected = -cell.molecule.coordinates + np.asarray((2, -1, 0)) @ cell.matrix
    np.testing.assert_allclose(result.molecule.coordinates, expected, atol=1e-10)
    assert result.bond_images == ((-1, 2, 0), None)
    composed = cell.operate(second @ first)
    twice = result.operate(second)
    np.testing.assert_allclose(twice.molecule.coordinates, composed.molecule.coordinates, atol=1e-10)
    assert twice.bond_images == composed.bond_images
    assert twice.space_group == cell.space_group and twice.representation == 'independent'
    assert fields(cell.molecule) == before


def test_active_site_image_relations_are_conjugated_not_copied():
    cell = crystal()
    g = SymmUnit.from_xyz('-x+2,-y,-z', context=cell.space_group)
    h = SymmUnit.from_xyz('-x,-y,-z', context=cell.space_group)
    cell.set_site_relations({'equivalent_images': (((0, h, (1, 0, 0)),
        (1, cell.space_group.identity(), (0, 0, 0))),),
        'occupancies': (.5, .5, None), 'disorder': (('A', 1), ('A', 2), None)})
    result = cell.operate(g)
    entry = result.site_relations['equivalent_images'][0][0]
    full = SymmUnit.from_affine(entry[1].R, tuple(a + b for a, b in zip(entry[1].t, entry[2])),
                               context=cell.space_group)
    before = SymmUnit.from_affine(h.R, (1, 0, 0), context=cell.space_group)
    assert full == g @ before @ ~g
    assert result.site_relations['occupancies'] == (.5, .5, None)
    assert result.site_relations['disorder'] == (('A', 1), ('A', 2), None)


@pytest.mark.parametrize('active', [None, SpaceGroup(1)])
def test_unknown_or_wrong_context_and_absent_content_reject_locally(active):
    mol = molecule()
    cell = Crystal(5, 6, 7, 90, 90, 90, molecule=mol, space_group=active)
    before = fields(mol)
    with pytest.raises(ValueError, match='active known group'):
        cell.operate(SpaceGroup(2).identity())
    assert fields(mol) == before
    empty = Crystal(5, 6, 7, 90, 90, 90, space_group=SpaceGroup(1))
    with pytest.raises(ValueError, match='contained Molecule'):
        empty.operate(empty.space_group.identity())


@pytest.mark.parametrize('supplied', [{}, {'x': 1.}, {'x': 0., 'y': 0.}])
def test_missing_supplied_xyz_is_not_turned_into_default_coordinate_facts(supplied):
    mol = Molecule()
    mol.create_atom(atomic_number=6, **supplied)
    cell = Crystal(5, 5, 5, 90, 90, 90, molecule=mol, space_group=SpaceGroup(1))
    before = fields(mol)
    with pytest.raises(ValueError, match='complete supplied Cartesian'):
        cell.operate(cell.space_group.identity())
    assert fields(mol) == before


def tetrahedron():
    mol = Molecule()
    for number, xyz in zip((6, 9, 17, 35, 53),
                          ((0, 0, 0), (1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))):
        mol.create_atom(atomic_number=number, coordinates=xyz, implicit_hydrogens=0)
    for row in range(1, 5):
        mol.add_bond(0, row, bond_order=1)
    record = stereo.StereoRecord(stereo.StereoAnchor(stereo.StereoAnchorKind.ATOM, (0,)),
        tuple(stereo.StereoCarrier(stereo.StereoCarrierKind.ATOM, row) for row in range(1, 5)),
        'hotpot.stereo/2/TH/even', None, 'hotpot.stereo/2', 'hotpot.stereo/2/TH/even')
    stereo.set_stereo_payload(mol, stereo.StereoPayload(1, (record,), (), ()))
    return Crystal(8, 8, 8, 90, 90, 90, molecule=mol, space_group=SpaceGroup(2))


def test_mirror_transforms_real_coordinate_orientation_and_chooses_no_new_chemical_state():
    cell = tetrahedron()
    initial = stereo.get_stereo_payload(cell.molecule)
    initial_xyz = cell.molecule.coordinates
    g = SymmUnit.from_xyz('-x,-y,-z', context=cell.space_group)
    mirrored = cell.operate(g)
    result = stereo.get_stereo_payload(mirrored.molecule)
    assert result.records[0].descriptor == 'hotpot.stereo/2/TH/odd'
    assert result.records[0].carriers == initial.records[0].carriers
    assert result.records[0].source_code == initial.records[0].source_code
    def signed_volume(xyz):
        a, b, c, d = xyz[1:]
        return np.dot(a - d, np.cross(b - d, c - d))
    assert signed_volume(mirrored.molecule.coordinates) == pytest.approx(-signed_volume(initial_xyz))
    twice = mirrored.operate(g)
    assert stereo.get_stereo_payload(twice.molecule) == initial
    np.testing.assert_allclose(twice.molecule.coordinates, initial_xyz, atol=1e-10)
    assert stereo.get_stereo_payload(cell.molecule) == initial


def test_improper_foreign_stereo_rejects_before_any_source_publication():
    cell = tetrahedron()
    payload = stereo.get_stereo_payload(cell.molecule)
    stereo.set_stereo_payload(cell.molecule, replace(payload, records=(replace(payload.records[0],
        descriptor='CHI_TETRAHEDRAL_CW', source_namespace='rdkit-stereo/1', source_code='CHI_TETRAHEDRAL_CW'),)))
    before = fields(cell.molecule)
    with pytest.raises(ValueError, match='CRYSTAL-UNSUPPORTED-PROJECTION'):
        cell.operate(SymmUnit.from_xyz('-x,-y,-z', context=cell.space_group))
    assert fields(cell.molecule) == before


def test_selected_results_stay_historical_and_source_results_remain_current():
    cell = crystal(1)
    mol = cell.molecule
    _import_partial_charges(mol, (0., -.2, .2), producer=4, method='fixture', method_version='1')
    prediction = MoleculePrediction('fixture', 0, (), 'fixture',
        tuple(AtomPrediction(row, atom.symbol, float(row)) for row, atom in enumerate(mol.atoms)))
    _select_mca_result(mol, prediction, _capture_mca_source(mol), tuple(atom._uid for atom in mol.atoms))
    source = _capture_source(mol)
    result = cell.operate(SymmUnit.from_xyz('x+1,y,z', context=cell.space_group)).molecule
    assert result._mca_result.payload is prediction
    assert result._partial_charge_result.payload.values == (0., -.2, .2)
    with pytest.raises(RuntimeError, match='geometry'):
        _ = result.atoms[0].partial_charge
    with pytest.raises(RuntimeError, match='geometry'):
        _ = result.atoms[0].mca
    assert mol.atoms[0].partial_charge == 0 and mol.atoms[1].mca == 1
    _check_source(mol, source)


def test_signed_axis_permutation_transports_constraint_values_presence_and_sources():
    mol = molecule()
    mol.atoms[0].x_constraint = True
    previous = mol._capture_factual_record()[0]
    cell = Crystal(5, 5, 7, 90, 90, 90, molecule=mol, space_group=SpaceGroup(75))
    g = SymmUnit.from_xyz('-y,x,z', context=cell.space_group)
    out = cell.operate(g).molecule
    np.testing.assert_array_equal(tuple(getattr(out.atoms[0], field) for field in ('x_constraint', 'y_constraint', 'z_constraint')), (False, True, False))
    assert tuple(getattr(mol.atoms[0], field) for field in ('x_constraint', 'y_constraint', 'z_constraint')) == (True, False, False)
    observed = out._capture_factual_record()[0]
    old_mask = np.frombuffer(previous.atom_presence, dtype='<u4')[0]
    new_mask = np.frombuffer(observed.atom_presence, dtype='<u4')[0]
    old_codes = np.frombuffer(previous.atom_sources, dtype='u1').reshape(3, 19)[0]
    new_codes = np.frombuffer(observed.atom_sources, dtype='u1').reshape(3, 19)[0]
    for target, source in zip((16, 17, 18), (17, 16, 18)):
        assert bool(new_mask & FIELD_PRESENCE_BITS[target]) == bool(old_mask & FIELD_PRESENCE_BITS[source])
        assert new_codes[target] == old_codes[source]
    np.testing.assert_allclose(out.coordinates, mol.coordinates @ np.array(((0, 1, 0), (-1, 0, 0), (0, 0, 1))), atol=1e-10)


@pytest.mark.parametrize('constraint,accepted', [((True, False, False), False),
                                               ((True, True, False), True),
                                               ((True, True, True), True),
                                               ((False, False, False), True)])
def test_general_rotation_respects_actual_cartesian_constraint_subspace(constraint, accepted):
    mol = molecule()
    for field, value in zip(('x_constraint', 'y_constraint', 'z_constraint'), constraint):
        setattr(mol.atoms[0], field, value)
    cell = Crystal(5, 5, 7, 90, 90, 120, molecule=mol, space_group=SpaceGroup(168))
    g = SymmUnit.from_xyz('x-y,x,z', context=cell.space_group)
    before = fields(mol)
    if accepted:
        output = cell.operate(g)
        np.testing.assert_array_equal(tuple(getattr(output.molecule.atoms[0], field) for field in ('x_constraint', 'y_constraint', 'z_constraint')), constraint)
    else:
        with pytest.raises(ValueError, match='not Cartesian-axis aligned'):
            cell.operate(g)
    assert fields(mol) == before


def generated():
    source = crystal(1)
    record = source.molecule._capture_factual_record()[0]
    origin = (record.cell[1], source.space_group, ((6,), (7,), (8,)), ((0, 1), (1, 2)),
              ((2, 0, 0), (0, 3, 0), (0, 0, 2)))
    basis = np.diag((2, 3, 2)) @ source.matrix
    atom_map = tuple(((row, source.space_group.identity(), (0, 0, 0)),) for row in range(3))
    bond_map = tuple(((row, source.space_group.identity(), (0, 0, 0)),) for row in range(2))
    cell = ('hotpot.crystal/1', tuple(map(tuple, basis)), SpaceGroup(1), None,
            record.cell[4], 'supercell', origin, atom_map, bond_map)
    return Molecule._build_molecule(replace(record, cell=cell)).crystal


def test_generated_active_translation_uses_explicit_exact_source_frame_map_and_keeps_carries():
    cell = generated()
    g = SymmUnit.from_xyz('x+1,y-2,z+3', context=cell.space_group)
    output = cell.operate(g)
    for row, entry in enumerate(output.image_map):
        assert entry == ((row, cell.image_map[row][0][1], (2, -6, 6)),)
    assert output._state[5][4] == ((2, 0, 0), (0, 3, 0), (0, 0, 2))
    np.testing.assert_allclose(output.molecule.coordinates,
        cell.molecule.coordinates + np.array((1, -2, 3)) @ cell.matrix, atol=1e-10)
    restored = pickle.loads(pickle.dumps(output))
    assert restored.image_map == output.image_map
    assert restored.molecule.crystal is restored


def test_cell_edit_retains_historical_origins_and_marks_exact_frame_unavailable_atomically():
    cell = generated()
    before_map = cell.image_map
    cell.set_cell(cell.matrix * 1.1)
    assert cell.image_map == before_map and cell._state[5][4] is None
    before = fields(cell.molecule)
    with pytest.raises(ValueError, match='exact active-to-source basis map is unavailable'):
        cell.operate(SymmUnit.from_xyz('x+1,y,z', context=cell.space_group))
    assert fields(cell.molecule) == before
    assert cell.operate(cell.space_group.identity()).image_map == before_map


def test_inconsistent_exact_source_basis_map_and_obsolete_origin_shape_reject():
    cell = generated()
    record = cell.molecule._capture_factual_record()[0]
    origin = record.cell[6]
    with pytest.raises(ValueError, match='basis map disagrees'):
        replace(record, cell=(*record.cell[:6], (*origin[:4], I), *record.cell[7:]))
    with pytest.raises(ValueError):
        replace(record, cell=(*record.cell[:6], origin[:4], *record.cell[7:]))


def test_action_path_never_uses_prohibited_toolkit_or_gemmi(monkeypatch):
    cell = tetrahedron()
    original = builtins.__import__
    def no_toolkit(name, *args, **kwargs):
        if name.split('.')[0] in ('openbabel', 'rdkit', 'ccdc', 'gemmi'):
            raise AssertionError('prohibited action dependency: ' + name)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', no_toolkit)
    g = SymmUnit.from_xyz('-x+1,-y,-z', context=cell.space_group)
    out = cell.operate(g).operate(~g)
    np.testing.assert_allclose(out.molecule.coordinates, cell.molecule.coordinates, atol=1e-10)



def test_known_native_absence_tokens_are_silently_invariant_without_toolkit_calls():
    cell = crystal()
    records = tuple(stereo.StereoRecord(
        stereo.StereoAnchor(stereo.StereoAnchorKind.ATOM, (row,)), (),
        'CHI_UNSPECIFIED', None, 'rdkit-2026.09.1', 'CHI_UNSPECIFIED') for row in range(3))
    records += tuple(stereo.StereoRecord(
        stereo.StereoAnchor(stereo.StereoAnchorKind.BOND, pair), (),
        'STEREONONE', None, 'rdkit-2026.09.1', 'STEREONONE') for pair in ((0, 1), (1, 2)))
    payload = stereo.StereoPayload(1, records, (), tuple(
        stereo.BondTransportRecord(pair, 'NONE') for pair in ((0, 1), (1, 2))))
    stereo.set_stereo_payload(cell.molecule, payload)
    result = cell.operate(SymmUnit.from_xyz('-x,-y,-z', context=cell.space_group))
    assert stereo.get_stereo_payload(result.molecule) == payload
    np.testing.assert_allclose(result.molecule.coordinates, -cell.molecule.coordinates, atol=1e-10)



def test_selected_source_frame_is_retained_with_explicit_geometry_invalidation():
    cell = crystal(1)
    mol = cell.molecule
    mol.conformer_add(mol.coordinates, energy=4.25)
    mol.conformer_load(0)
    selection = mol._frame_result
    source = _capture_source(mol)
    result = cell.operate(SymmUnit.from_xyz('x+1,y,z', context=cell.space_group)).molecule
    assert result._frame_result is not None
    assert 'geometry' in result._frame_result.invalidated_by
    assert result.conformer_get(0)['energy'] == 4.25
    with pytest.raises(RuntimeError, match='geometry'):
        _ = result.energy
    assert mol._frame_result is selection and mol.energy == 4.25
    _check_source(mol, source)


def test_all_unknown_row_origins_do_not_require_a_known_frame_for_unrelated_placement():
    cell = generated()
    record = cell.molecule._capture_factual_record()[0]
    origin = (*record.cell[6][:4], None)
    unknown = (*record.cell[:6], origin, (None,) * 3, (None,) * 2)
    cell = Molecule._build_molecule(replace(record, cell=unknown)).crystal
    before = fields(cell.molecule)
    result = cell.operate(SymmUnit.from_xyz('x+1,y-2,z', context=cell.space_group))
    assert result.image_map == (None,) * 3
    assert result.molecule._capture_factual_record()[0].cell[8] == (None,) * 2
    np.testing.assert_allclose(result.molecule.coordinates,
        cell.molecule.coordinates + np.array((1, -2, 0)) @ cell.matrix, atol=1e-10)
    assert fields(cell.molecule) == before
