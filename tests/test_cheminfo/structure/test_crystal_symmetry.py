"""Independent exact and analytic crystallographic contracts for Stage15B."""

import copy
from fractions import Fraction as Q
import json
import pickle
import subprocess
import sys

import numpy as np
import pytest

from hotpot import SpaceGroup, SymmUnit
from hotpot.cheminfo.rules.mathematics.symmetry import _det, _inverse, _mv

I = ((1, 0, 0), (0, 1, 0), (0, 0, 1))


def _reference():
    from importlib.resources import files
    return json.loads(files('hotpot.cheminfo.reference').joinpath('data/space_groups.json').read_text())


def test_abstract_affine_surface_and_public_identity():
    import hotpot.cheminfo as cheminfo
    import hotpot.cheminfo.structure as structure
    assert structure.SpaceGroup is cheminfo.SpaceGroup is SpaceGroup
    assert structure.SymmUnit is cheminfo.SymmUnit is SymmUnit
    with pytest.raises(TypeError):
        SymmUnit()


@pytest.mark.parametrize('number', range(1, 231))
def test_complete_reference_groups_and_all_pair_carries(number):
    entry = _reference()['groups'][number - 1]
    group = SpaceGroup(number)
    operations = group.symm_units
    assert group.number == number
    assert group.hermann_mauguin == entry['hermann_mauguin']
    assert group.hall == entry['hall']
    assert group.cell_order == entry['cell_order']
    assert group.point_group_order == entry['point_group_order']
    raw = {(operation.R, operation.t) for operation in operations}
    assert len(raw) == len(operations)
    assert group.identity() in group
    assert abs(_det(group.translation_basis)) == Q(1, len(group.centering_translations))
    inverse_basis = _inverse(group.translation_basis)
    for vector in group.centering_translations:
        assert all(value.denominator == 1 for value in _mv(inverse_basis, vector))
    for index, first in enumerate(operations):
        assert SymmUnit.from_xyz(first.xyz, context=group) == first
        assert first @ ~first == group.identity()
        assert ~first @ first == group.identity()
        lifted = group.lift(index, (2, -3, 5))
        assert lifted.reduce_to_cell() == (first, (2, -3, 5))
        assert lifted != first
        for second in operations:
            # This direct coordinate definition is independent of the product
            # lookup and separately asserts that full carries are not reduced.
            expected_R = tuple(tuple(sum(first.R[i][k] * second.R[k][j] for k in range(3))
                                     for j in range(3)) for i in range(3))
            expected_t = tuple(sum(first.R[i][k] * second.t[k] for k in range(3)) + first.t[i]
                               for i in range(3))
            result = first @ second
            assert result.R == expected_R and result.t == expected_t
            representative, carry = result.reduce_to_cell()
            assert (representative.R, representative.t) in raw
            assert result.t == tuple(value + image for value, image in zip(representative.t, carry))


def test_catalogue_dimensions_are_setting_specific_not_a_new_group():
    table = _reference()
    assert len(table['groups']) == 230
    assert sum(group['cell_order'] for group in table['groups']) == 4425
    pairs = {(index, tuple(translation)) for group in table['groups']
             for index, translation, _ in group['operations']}
    assert len(pairs) == 786


def test_screw_glide_and_off_origin_rotation_have_distinct_full_powers():
    screw_group = SpaceGroup(4)
    screw = SymmUnit.from_xyz('-x,y+1/2,-z', context=screw_group)
    assert screw.kind == 'screw'
    assert (screw ** 2).R == I and (screw ** 2).t == (0, 1, 0)
    assert screw ** -1 == ~screw
    assert screw ** 0 == screw_group.identity()
    assert screw ** 11 == screw ** 5 @ screw ** 6
    with pytest.raises(ValueError, match='outside'):
        SymmUnit.from_xyz('-x,y,-z', context=screw_group)
    glide_group = SpaceGroup(7)
    glide = SymmUnit.from_xyz('x,-y,z+1/2', context=glide_group)
    assert glide.kind == 'glide'
    assert (glide ** 2).t == (0, 0, 1)
    rotated = SpaceGroup(3).change_setting(I, (Q(1, 7), 0, Q(2, 9)))
    turn = next(value for value in rotated.symm_units if value.R != I)
    assert any(turn.t) and turn.kind == 'rotation'
    assert turn ** 2 == rotated.identity()
    assert rotated.hermann_mauguin is None and rotated.hall is None


def test_full_elements_support_arbitrary_rational_origins_and_integer_lifts():
    group = SpaceGroup(2).change_setting(I, (Q(1, 101), Q(-2, 103), Q(3, 107)))
    inversion = next(value for value in group.symm_units if value.R != I)
    value = group.lift(1, (10**25, -10**30, 7))
    assert SymmUnit.from_xyz(value.xyz, context=group) == value
    assert inversion.kind == 'inversion'
    assert inversion @ inversion == group.identity()
    assert value.t[0] > 10**24


def test_centered_to_primitive_setting_changes_quotient_size_and_returns():
    group = SpaceGroup(225)
    P = ((0, Q(1, 2), Q(1, 2)), (Q(1, 2), 0, Q(1, 2)), (Q(1, 2), Q(1, 2), 0))
    primitive = group.change_setting(P, (0, 0, 0))
    assert group.cell_order == 192 and group.point_group_order == 48
    assert primitive.cell_order == primitive.point_group_order == 48
    assert primitive.centering_translations == ((0, 0, 0),)
    assert primitive.change_setting(_inverse(P), (0, 0, 0)) == group
    assert primitive.setting == (225, P, (0, 0, 0))
    rebuilt = SpaceGroup.from_operations(((value.R, value.t) for value in primitive.symm_units),
                                        reference_number=225, to_reference=(P, (0, 0, 0)))
    assert rebuilt == primitive


def test_supercell_explicit_full_lattice_cosets_and_setting_rejection():
    group = SpaceGroup(1)
    supercell = group.change_setting(((2, 0, 0), (0, 3, 0), (0, 0, 2)), (0, 0, 0))
    assert supercell.cell_order == 12 and supercell.point_group_order == 1
    assert abs(_det(supercell.translation_basis)) == Q(1, 12)
    assert supercell.change_setting(((Q(1, 2), 0, 0), (0, Q(1, 3), 0), (0, 0, Q(1, 2))),
                                    (0, 0, 0)) == group
    with pytest.raises(ValueError, match='sublattice'):
        group.change_setting(((Q(1, 2), 0, 0), (0, 1, 0), (0, 0, 1)), (0, 0, 0))
    with pytest.raises(ValueError, match='cell Z'):
        SpaceGroup(221).change_setting(((2, 0, 0), (0, 1, 0), (0, 0, 1)), (0, 0, 0))


def test_complete_group_input_validates_actual_membership_closure_and_reference():
    anonymous = SpaceGroup.from_operations(('x,y,z', '-x,-y,-z'))
    assert anonymous.number is None and anonymous.hall is None
    assert anonymous.point_group_order == 2
    for operations in [('-x,-y,-z',), ('x,y,z', '-y,x,z'), ('x,y,z', 'x+1/3,y,z')]:
        with pytest.raises(ValueError, match='CRYSTAL-CONTEXT-INTEGRITY'):
            SpaceGroup.from_operations(operations)
    with pytest.raises(ValueError, match='reference'):
        SpaceGroup.from_operations(('x,y,z',), reference_number=1)
    with pytest.raises(ValueError, match='reference'):
        SpaceGroup.from_operations(('x,y,z',), reference_number=2, to_reference=(I, (0, 0, 0)))
    with pytest.raises(ValueError, match='settings'):
        SpaceGroup(1).identity() @ SpaceGroup(2).identity()


def test_exact_parser_affine_domain_and_numeric_application_are_independent():
    group = SpaceGroup(4)
    operation = SymmUnit.from_xyz('-x,y+3/2,-z-2', context=group)
    source = np.array(((.2, .4, .6), (-.8, .1, 1.2)))
    expected = np.array(((-.2, 1.9, -2.6), (.8, 1.6, -3.2)))
    result = operation.operate(source)
    np.testing.assert_allclose(result, expected, atol=1e-14)
    assert not np.shares_memory(result, source)
    for symbol in ('x*y,y,z', '__import__("os"),y,z', 'x+0.5,y,z'):
        with pytest.raises(ValueError, match='CRYSTAL-CONTEXT-INTEGRITY'):
            SymmUnit.from_xyz(symbol, context=group)
    with pytest.raises(TypeError, match='exact'):
        SymmUnit.from_affine(I, (.5, 0, 0), context=group)


def test_values_are_immutable_hashable_and_share_only_immutable_context():
    group = SpaceGroup(19)
    value = group.lift(2, (1, -3, 2))
    with pytest.raises(AttributeError):
        group._number = 1
    with pytest.raises(AttributeError):
        value._t = (0, 0, 0)
    assert copy.deepcopy(group) is group
    assert copy.deepcopy(value) is value
    restored_group, restored_value = pickle.loads(pickle.dumps((group, value)))
    assert restored_value.context is restored_group
    assert restored_group == group and restored_value == value
    assert hash(restored_group) == hash(group)
    assert hash(restored_value) == hash(value)


def test_exact_domain_import_and_work_need_no_prohibited_python_package(tmp_path):
    script = '''
import sys
class Block:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('rdkit', 'openbabel', 'ccdc'):
            raise AssertionError('prohibited import '+fullname)
sys.meta_path.insert(0, Block())
from hotpot import SpaceGroup, SymmUnit
assert (SymmUnit.from_xyz('-x,y+1/2,-z',context=SpaceGroup(4))**2).t==(0,1,0)
assert not any(k.split('.')[0] in ('rdkit','openbabel','ccdc') for k in sys.modules)
'''
    result = subprocess.run([sys.executable, '-c', script], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr



def test_immutable_values_reject_constructor_reentry_and_slot_deletion():
    group = SpaceGroup(4)
    operation = group.lift(1, (2, -3, 5))
    members = {group: 'group', operation: 'operation'}
    original_group, original_operation = pickle.loads(pickle.dumps((group, operation)))
    with pytest.raises(AttributeError, match='immutable'):
        SpaceGroup.__init__(group, 1)
    with pytest.raises(AttributeError, match='immutable'):
        operation.__init__(I, (0, 0, 0), SpaceGroup(1))
    for value, slots in ((group, ('_operations', '_number', '_setting', '_metadata')),
                         (operation, ('_R', '_t', '_context'))):
        for slot in slots:
            with pytest.raises(AttributeError, match='immutable'):
                delattr(value, slot)
    assert group == original_group and operation == original_operation
    assert members[group] == 'group' and members[operation] == 'operation'
    assert operation.context is group
    assert copy.copy(group) is copy.deepcopy(group) is group
    assert copy.copy(operation) is copy.deepcopy(operation) is operation


def test_runtime_attribution_names_packaged_original_licences():
    from importlib.resources import files
    owner = files('hotpot.cheminfo.reference').joinpath('data')
    for attribution in _reference()['sources']['copyright_and_licences'].values():
        target = attribution.split('see ')[-1]
        assert owner.joinpath(target).is_file()
