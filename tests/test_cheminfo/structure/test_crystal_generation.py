"""Independent chemical/image witnesses for cell filling and finite expansion."""

import builtins
from dataclasses import replace
from fractions import Fraction
from itertools import product
import pickle

import numpy as np
import pytest

from hotpot import Crystal, Molecule, SpaceGroup, SymmUnit
from hotpot.cheminfo import stereo
from hotpot.cheminfo.structure._identity import _capture_source, _check_source
from hotpot.cheminfo.structure._results import _capture_mca_source, _import_partial_charges, _select_mca_result
from hotpot.cheminfo.operations.inference.mca.results import AtomPrediction, MoleculePrediction


I = ((1, 0, 0), (0, 1, 0), (0, 0, 1))


def molecule(points, numbers=None, edges=()):
    value = Molecule()
    numbers = (6,) * len(points) if numbers is None else numbers
    for number, point in zip(numbers, points):
        value.create_atom(atomic_number=number, coordinates=point, formal_charge=0, implicit_hydrogens=0)
    for first, second in edges:
        value.add_bond(first, second, bond_order=1)
    value.charge = 0
    value.spin_mult = 1
    return value


def finite_ring():
    value = molecule(((4.5, 1., 1.), (.5, 1., 1.), (1., 2., 1.)),
                     edges=((0, 1), (0, 2), (1, 2)))
    return Crystal(5, 6, 7, 90, 90, 90, molecule=value, space_group=SpaceGroup(1),
                   bond_images=((1, 0, 0), (1, 0, 0), (0, 0, 0)))


def state(value):
    record = value.molecule._capture_factual_record()[0]
    return (record.atom_values, record.atom_presence, record.atom_sources, record.cell,
            record.bond_endpoints, record.bond_values, record.stereo_payload,
            record.angle_constraints, record.torsion_constraints, record.charge, record.spin_mult)


def relation(group, entries, count):
    return {'equivalent_images': entries, 'occupancies': (None,) * count, 'disorder': (None,) * count}


def symmetric_triatom():
    value = molecule(((0., 0., 0.), (1., 0., 0.), (-1., 0., 0.)), (6, 8, 8), ((0, 1), (0, 2)))
    group = SpaceGroup(2)
    identity = group.identity()
    inversion = SymmUnit.from_xyz('-x,-y,-z', context=group)
    cell = Crystal(5, 5, 5, 90, 90, 90, molecule=value, space_group=group)
    entries = (((0, identity, (0, 0, 0)), (0, inversion, (0, 0, 0))),
               ((1, identity, (0, 0, 0)), (2, inversion, (0, 0, 0))))
    cell.set_site_relations(relation(group, entries, 3))
    return cell


def test_p1_wrap_transports_directed_images_and_preserves_every_original_fact():
    value = molecule(((-.5, 1., 1.), (5.5, 2., 1.)), (6, 7), ((0, 1),))
    value.bonds[0]._set_bond_metadata(bond_kind='dative', bond_direction='1->2', bond_source='fixture',
                                    bond_source_metadata={'labels': ['original']})
    cell = Crystal(5, 6, 7, 90, 90, 90, molecule=value, space_group=SpaceGroup(1), bond_images=((2, -1, 0),))
    before = state(cell)
    result = cell.unit_cell()
    assert result.space_group == SpaceGroup(1) and result.representation == 'cell'
    assert len(result.molecule.atoms) == 2 and result.bond_images == ((4, -1, 0),)
    np.testing.assert_allclose(result.to_fractional(result.molecule.coordinates), ((.9, 1 / 6, 1 / 7), (.1, 2 / 6, 1 / 7)), atol=1e-10)
    old = value.coordinates[1] + np.array((2, -1, 0)) @ cell.matrix - value.coordinates[0]
    actual = result.molecule.coordinates[1] + np.array(result.bond_images[0]) @ result.matrix - result.molecule.coordinates[0]
    np.testing.assert_allclose(actual, old, rtol=0, atol=1e-10)
    assert result.molecule.bonds[0].bond_direction == '1->2'
    result.molecule.bonds[0].bond_source_metadata['labels'].append('output')
    assert value.bonds[0].bond_source_metadata == {'labels': ['original']}
    assert state(cell) == before
    assert result.image_map[0][0][2] == (1, 0, 0)
    assert result.image_map[1][0][2] == (-1, 0, 0)


@pytest.mark.parametrize('number', (1, 2, 4, 5, 19, 75, 146))
def test_cell_representatives_and_second_projection_preserve_exact_active_population(number):
    group = SpaceGroup(number)
    matrix = (((5., 0., 0.), (-2.5, 5 * np.sqrt(3) / 2, 0.), (0., 0., 7.))
              if number == 146 else np.diag((5., 5., 7.)))
    points = np.array(((.137, .241, .319), (.367, .129, .473))) @ matrix
    cell = Crystal.from_matrix(matrix, molecule=molecule(points, (6, 7)), space_group=group)
    before = state(cell)
    result = cell.unit_cell()
    expected = np.array([operation.operate(point) % 1 for point in ((.137, .241, .319), (.367, .129, .473))
                         for operation in group.symm_units]) @ matrix
    np.testing.assert_allclose(result.molecule.coordinates, expected, rtol=0, atol=1e-10)
    assert len(result.molecule.atoms) == 2 * group.cell_order
    assert all(len(entry) == 1 for entry in result.image_map)
    assert result.space_group == SpaceGroup(1)
    again = result.unit_cell()
    assert len(again.molecule.atoms) == len(result.molecule.atoms)
    assert again.image_map == result.image_map
    np.testing.assert_allclose(again.molecule.coordinates, result.molecule.coordinates, atol=1e-10)
    assert state(cell) == before


def test_numeric_special_position_does_not_authorize_implicit_deduplication():
    cell = Crystal(5, 5, 5, 90, 90, 90, molecule=molecule(((0., 0., 0.),)), space_group=SpaceGroup(2))
    result = cell.unit_cell()
    assert len(result.molecule.atoms) == 2
    np.testing.assert_allclose(result.molecule.coordinates, 0, atol=1e-10)
    assert result.molecule.atoms[0] is not result.molecule.atoms[1]
    assert result.image_map[0] != result.image_map[1]


def test_explicit_isolated_site_stabilizer_retains_all_operator_aliases():
    group = SpaceGroup(2)
    cell = Crystal(5, 5, 5, 90, 90, 90, molecule=molecule(((0., 0., 0.),)), space_group=group)
    cell.set_site_relations(relation(group, (((0, group.identity(), (0, 0, 0)),
        (0, SymmUnit.from_xyz('-x,-y,-z', context=group), (0, 0, 0))),), 1))
    before = state(cell)
    result = cell.unit_cell()
    assert len(result.molecule.atoms) == 1 and len(result.image_map[0]) == 2
    assert result.molecule.charge == 0 and result.molecule.spin_mult == 1
    assert state(cell) == before


def test_complete_supplied_graph_identification_merges_only_its_declared_aliases():
    cell = symmetric_triatom()
    before = state(cell)
    result = cell.unit_cell()
    assert len(result.molecule.atoms) == 3 and len(result.molecule.bonds) == 2
    assert tuple(atom.atomic_number for atom in result.molecule.atoms) == (6, 8, 8)
    assert all(len(entry) == 2 for entry in result.image_map)
    record = result.molecule._capture_factual_record()[0]
    assert all(len(entry) == 2 for entry in record.cell[8])
    assert len(result.unit_molecules()) == 1
    assert result.molecule.spin_mult == cell.molecule.spin_mult
    assert state(cell) == before


def test_partial_identification_cannot_increase_the_identified_atoms_incidence():
    cell = symmetric_triatom()
    relations = dict(cell.site_relations)
    relations['equivalent_images'] = relations['equivalent_images'][:1]
    cell.set_site_relations(relations)
    before = state(cell)
    with pytest.raises(ValueError, match='CRYSTAL-'):
        cell.unit_cell()
    assert state(cell) == before


def test_identification_rechecks_actual_coordinate_edit_instead_of_trusting_old_assertion():
    cell = symmetric_triatom()
    cell.molecule.atoms[1].y = .1
    before = state(cell)
    with pytest.raises(ValueError, match='do not coincide'):
        cell.unit_cell()
    assert state(cell) == before


@pytest.mark.parametrize('repeats', ((2, 3, 2), (3, 1, 2), (1, 2, 1)))
def test_anisotropic_supercell_uses_actual_target_copy_and_retains_displacements(repeats):
    source = finite_ring()
    before = state(source)
    unit = source.unit_cell()
    result = source.expand(repeats)
    copies = tuple(product(*(range(count) for count in repeats)))
    count = len(copies)
    assert len(result.molecule.atoms) == 3 * count and len(result.molecule.bonds) == 3 * count
    assert result.representation == 'supercell' and result.space_group == SpaceGroup(1)
    np.testing.assert_allclose(result.matrix, np.diag(repeats) @ source.matrix, atol=1e-10)
    expected = np.array([point + np.array(image) @ source.matrix for point in unit.molecule.coordinates for image in copies])
    np.testing.assert_allclose(result.molecule.coordinates, expected, atol=1e-10)
    old = unit.molecule._capture_factual_record()[0]
    new = result.molecule._capture_factual_record()[0]
    for edge, ((first, second), image) in enumerate(zip(old.bond_endpoints, old.cell[4])):
        displacement = unit.molecule.coordinates[second] + np.array(image) @ unit.matrix - unit.molecule.coordinates[first]
        for copy in range(count):
            row = edge * count + copy
            a, b = new.bond_endpoints[row]
            actual = result.molecule.coordinates[b] + np.array(new.cell[4][row]) @ result.matrix - result.molecule.coordinates[a]
            np.testing.assert_allclose(actual, displacement, rtol=0, atol=1e-10)
    assert new.cell[6][4] == tuple(tuple(Fraction(repeats[row] if row == column else 0)
                                      for column in range(3)) for row in range(3))
    assert new.charge == (True, 0, 'crystal_generation') and not new.spin_mult[0]
    assert state(source) == before


def test_finite_crossing_ring_is_unwrapped_as_one_molecule_and_never_cut_at_face():
    source = finite_ring()
    before = state(source)
    (result,) = source.unit_molecules()
    np.testing.assert_allclose(result.coordinates, ((4.5, 1, 1), (5.5, 1, 1), (6, 2, 1)), atol=1e-10)
    assert len(result.bonds) == 3 and result.crystal.bond_images == ((0, 0, 0),) * 3
    assert result.crystal.space_group == SpaceGroup(1) and result.crystal.molecule is result
    result.atoms[0].x += 2
    assert state(source) == before


def test_nonzero_cycle_stays_periodic_while_independent_finite_components_continue():
    source = finite_ring()
    record = source.molecule._capture_factual_record()[0]
    cell = (*record.cell[:4], ((1, 0, 0), (0, 0, 0), (0, 0, 0)), *record.cell[5:])
    source = Molecule._build_molecule(replace(record, cell=cell)).crystal
    source.molecule.create_atom(atomic_number=8, coordinates=(2., 3., 1.))
    before = state(source)
    with pytest.warns(RuntimeWarning, match='CRYSTAL-MOLECULAR-LIFT-UNAVAILABLE'):
        result = source.unit_molecules()
    assert len(result) == 1 and len(result[0].atoms) == 1 and result[0].atoms[0].atomic_number == 8
    assert len(source.molecule.bonds) == 3 and state(source) == before


def test_unknown_images_do_not_become_finite_or_guessed_expansion_connections():
    value = molecule(((1., 1., 1.), (2., 1., 1.)), edges=((0, 1),))
    source = Crystal(5, 6, 7, 90, 90, 90, molecule=value, space_group=SpaceGroup(1), bond_images=(None,))
    before = state(source)
    assert source.unit_cell().bond_images == (None,)
    assert source.expand((1, 1, 1)).bond_images == (None,)
    with pytest.raises(ValueError, match='known Bond images'):
        source.expand((2, 1, 1))
    with pytest.warns(RuntimeWarning, match='unknown_edges='):
        assert source.unit_molecules() == ()
    assert state(source) == before


@pytest.mark.parametrize('repeats', ((0, 1, 1), (-1, 2, 1), (1, 1), (1, 1, 1, 1)))
def test_repeat_domain_rejects_locally_without_a_source_change(repeats):
    source = finite_ring()
    before = state(source)
    with pytest.raises(ValueError, match='three positive integers'):
        source.expand(repeats)
    assert state(source) == before


def test_unknown_group_and_absent_supplied_coordinates_remain_unavailable():
    value = molecule(((1., 1., 1.),))
    source = Crystal(5, 6, 7, 90, 90, 90, molecule=value)
    with pytest.raises(ValueError, match='known active group'):
        source.unit_cell()
    missing = Molecule()
    missing.create_atom(atomic_number=6)
    source = Crystal(5, 6, 7, 90, 90, 90, molecule=missing, space_group=SpaceGroup(1))
    with pytest.raises(ValueError, match='supplied Cartesian'):
        source.unit_cell()
    assert not (int(np.frombuffer(missing._capture_factual_record()[0].atom_presence, dtype='<u4')[0]) & (7 << 10))


def test_replication_omits_live_model_results_without_changing_original_observations():
    source = finite_ring()
    value = source.molecule
    _import_partial_charges(value, (0., -.2, .2), producer=4, method='fixture', method_version='1')
    prediction = MoleculePrediction('fixture', 0, (), 'fixture',
        tuple(AtomPrediction(row, atom.symbol, float(row)) for row, atom in enumerate(value.atoms)))
    _select_mca_result(value, prediction, _capture_mca_source(value), tuple(atom._uid for atom in value.atoms))
    binding = _capture_source(value)
    with pytest.warns(RuntimeWarning, match='MCA, partial charge'):
        result = source.expand((2, 1, 1)).molecule
    assert result._mca_result is None and result._partial_charge_result is None
    assert value.atoms[0].partial_charge == 0 and value.atoms[1].mca == 1
    assert value._mca_result.payload is prediction
    _check_source(value, binding)


def test_physical_cell_change_removes_only_needed_historical_basis_projection():
    source = finite_ring().unit_cell()
    source.set_cell(source.matrix * 2)
    before = state(source)
    assert source.unit_cell().space_group == SpaceGroup(1)
    with pytest.raises(ValueError, match='basis map is unavailable'):
        source.expand((2, 1, 1))
    assert state(source) == before


def test_all_unknown_row_origins_do_not_require_unavailable_frame_conversion():
    source = finite_ring().unit_cell()
    record = source.molecule._capture_factual_record()[0]
    context = record.cell
    origin = (*context[6][:4], None)
    context = (*context[:6], origin, (None,) * record.atom_count, (None,) * len(record.bond_endpoints))
    source = Molecule._build_molecule(replace(record, cell=context)).crystal
    result = source.expand((2, 1, 1))
    assert result.image_map == (None,) * 6
    assert result.molecule._capture_factual_record()[0].cell[8] == (None,) * 6


def test_crossing_angle_torsion_and_stereo_references_follow_whole_image_packets():
    value = molecule(((4.5, 1, 1), (.5, 1, 1), (1, 2, 1), (2, 2, 2)),
                     edges=((0, 1), (1, 2), (2, 3)))
    value.angles[0].constraint = True
    value.torsions[0].constraint = True
    ledger = stereo.HydrogenTransportState(0, 0, False)
    payload = stereo.StereoPayload(1, (), tuple(stereo.AtomTransportRecord(row, None, None, ledger, ledger, 'fixture')
        for row in range(4)), (stereo.BondTransportRecord((0, 1), 'NONE'),))
    stereo.set_stereo_payload(value, payload)
    source = Crystal(5, 6, 7, 90, 90, 90, molecule=value, space_group=SpaceGroup(1),
                     bond_images=((1, 0, 0), (0, 0, 0), (0, 0, 0)))
    result = source.expand((2, 1, 1)).molecule
    record = result._capture_factual_record()[0]
    assert record.angle_constraints == ((0, 3, 5), (1, 2, 4))
    assert record.torsion_constraints == ((0, 3, 5, 7), (1, 2, 4, 6))
    assert tuple(fact.atom_indices for fact in record.stereo_payload.bond_facts) == ((0, 3), (1, 2))
    assert all(tuple(pair) in record.bond_endpoints for pair in ((0, 3), (1, 2)))


def test_improper_cell_copy_preserves_hotpot_orientation_without_inference():
    from tests.test_cheminfo.structure.test_crystal_actions import tetrahedron
    source = tetrahedron()
    result = source.unit_cell()
    descriptors = tuple(item.descriptor for item in stereo.get_stereo_payload(result.molecule).records)
    assert set(descriptors) == {'hotpot.stereo/2/TH/even', 'hotpot.stereo/2/TH/odd'}
    assert len(descriptors) == 2 and len(result.molecule.atoms) == 10


@pytest.mark.parametrize('protocol', (0, 4, 5))
def test_generated_combined_pickle_keeps_aliases_origin_history_and_independent_facts(protocol):
    source = finite_ring().expand((2, 1, 1))
    restored, value, atom, bond = pickle.loads(pickle.dumps((source, source.molecule, source.molecule.atoms[0],
                                                          source.molecule.bonds[0]), protocol=protocol))
    assert restored.molecule is value and value.crystal is restored
    assert value.atoms[0] is atom and value.bonds[0] is bond
    assert restored.image_map == source.image_map and restored.bond_images == source.bond_images
    atom.x += 1
    assert value.coordinates[0, 0] != source.molecule.coordinates[0, 0]


def test_crystal_generation_calls_no_external_molecule_api(monkeypatch):
    source = finite_ring()
    original = builtins.__import__
    def no_toolkit(name, *args, **kwargs):
        if name.split('.')[0] in ('rdkit', 'openbabel', 'ccdc', 'pybel'):
            raise AssertionError(f'prohibited Crystal dependency: {name}')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', no_toolkit)
    assert len(source.unit_cell().molecule.atoms) == 3
    assert len(source.expand((2, 3, 2)).molecule.atoms) == 36
    assert len(source.unit_molecules()) == 1


def test_rotated_triclinic_projection_and_expansion_use_the_full_basis():
    matrix = np.array(((0., 5., 0.), (-4., 1., 0.), (1., .5, 7.)))
    fractional = np.array(((-.1, 1.2, .3), (.2, -.1, .4)))
    source = Crystal.from_matrix(matrix, molecule=molecule(fractional @ matrix, edges=((0, 1),)),
        space_group=SpaceGroup(1), bond_images=((1, -2, 0),))
    projected = source.unit_cell()
    np.testing.assert_allclose(projected.molecule.coordinates, (fractional % 1) @ matrix, atol=1e-10)
    expected_vector = (fractional[1] + np.array((1, -2, 0)) - fractional[0]) @ matrix
    actual_vector = projected.molecule.coordinates[1] + np.array(projected.bond_images[0]) @ matrix - projected.molecule.coordinates[0]
    np.testing.assert_allclose(actual_vector, expected_vector, atol=1e-10)
    expanded = source.expand((2, 3, 1))
    np.testing.assert_allclose(expanded.matrix, np.diag((2, 3, 1)) @ matrix, atol=1e-10)
    assert len(expanded.molecule.bonds) == 6


def test_bijective_identification_preserves_whole_charge_spin_without_redundant_atom_declarations():
    value = Molecule()
    value.create_atom(atomic_number=6, coordinates=(0., 0., 0.))
    value.charge = 1
    value.spin_mult = 2
    group = SpaceGroup(2)
    source = Crystal(5, 5, 5, 90, 90, 90, molecule=value, space_group=group)
    source.set_site_relations(relation(group, (((0, group.identity(), (0, 0, 0)),
        (0, SymmUnit.from_xyz('-x,-y,-z', context=group), (0, 0, 0))),), 1))
    result = source.unit_cell().molecule
    assert len(result.atoms) == 1 and result.charge == 1 and result.spin_mult == 2
    assert result._capture_factual_record()[0].charge == value._capture_factual_record()[0].charge


def test_repeated_supercells_preserve_full_source_origins_without_reapplying_historical_symmetry():
    source = finite_ring()
    first = source.expand((2, 1, 1))
    nested = first.expand((3, 1, 1))
    direct = source.expand((6, 1, 1))
    assert len(nested.molecule.atoms) == len(direct.molecule.atoms) == 18
    np.testing.assert_allclose(nested.matrix, direct.matrix, atol=1e-10)
    assert nested.molecule._capture_factual_record()[0].cell[6][4] == ((6, 0, 0), (0, 1, 0), (0, 0, 1))
    def origin_positions(value):
        return sorted((entry[0][0], tuple(map(float, entry[0][1].t)), entry[0][2], tuple(point))
                      for entry, point in zip(value.image_map, value.molecule.coordinates))
    for actual, expected in zip(origin_positions(nested), origin_positions(direct)):
        assert actual[:3] == expected[:3]
        np.testing.assert_allclose(actual[3], expected[3], atol=1e-10)


def test_empty_known_cell_has_no_fabricated_sites_edges_or_finite_molecules():
    source = Crystal(5, 6, 7, 90, 90, 90, molecule=Molecule(), space_group=SpaceGroup(2))
    projected = source.unit_cell()
    assert len(projected.molecule.atoms) == len(projected.molecule.bonds) == 0
    assert projected.image_map == () and source.unit_molecules() == ()
    assert len(source.expand((2, 3, 2)).molecule.atoms) == 0


def test_generated_row_source_keys_follow_current_order_across_original_uid_gaps():
    value = molecule(((1., 1., 1.), (2., 1., 1.), (3., 1., 1.)))
    value.remove_atom(1)
    value.create_atom(atomic_number=8, coordinates=(4., 1., 1.))
    assert tuple(atom._uid for atom in value.atoms) != (0, 1, 2)
    source = Crystal(5, 6, 7, 90, 90, 90, molecule=value, space_group=SpaceGroup(2))
    result = source.unit_cell()
    assert tuple(entry[0][0] for entry in result.image_map) == (0, 0, 1, 1, 2, 2)
    assert tuple(atom.atomic_number for atom in result.molecule.atoms) == (6, 6, 6, 6, 8, 8)


def test_rational_primitive_historical_axes_compose_exactly_during_supercell_expansion():
    group = SpaceGroup(5)
    source = Crystal(5, 6, 7, 90, 90, 90, molecule=molecule(((1., 1., 1.),)), space_group=group).unit_cell()
    record = source.molecule._capture_factual_record()[0]
    frame = ((Fraction(1, 2), Fraction(1, 2), 0),
             (Fraction(1, 2), Fraction(-1, 2), 0), (0, 0, -1))
    basis = (np.asarray(record.cell[6][0]).T @ np.asarray(frame, dtype=float)).T
    origin = (*record.cell[6][:4], frame)
    context = (record.cell[0], tuple(map(tuple, basis)), *record.cell[2:6], origin, *record.cell[7:])
    primitive = Molecule._build_molecule(replace(record, cell=context)).crystal
    result = primitive.expand((2, 1, 1))
    assert result.molecule._capture_factual_record()[0].cell[6][4] == (
        (1, Fraction(1, 2), 0), (1, Fraction(-1, 2), 0), (0, 0, -1))
    assert len(result.molecule.atoms) == 8
    for entry in result.image_map:
        for _, operation, image in entry:
            assert SymmUnit.from_affine(operation.R, tuple(value + shift for value, shift in zip(operation.t, image)), context=group) in group
