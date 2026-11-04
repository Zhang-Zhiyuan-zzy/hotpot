"""Analytic kind distinctions; translations are invariant, not raw offset tests."""

from fractions import Fraction

import pytest

from hotpot import SpaceGroup, SymmUnit


@pytest.mark.parametrize('number,symbol,kind', [
    (1, 'x,y,z', 'identity'),
    (1, 'x+1,y-2,z+3', 'translation'),
    (3, '-x,y,-z', 'rotation'),
    (4, '-x,y+1/2,-z', 'screw'),
    (6, 'x,-y,z', 'reflection'),
    (7, 'x,-y,z+1/2', 'glide'),
    (2, '-x,-y,-z', 'inversion'),
    (81, '-y,x,-z', 'rotoinversion'),
])
def test_analytic_eight_full_affine_kinds(number, symbol, kind):
    operation = SymmUnit.from_xyz(symbol, context=SpaceGroup(number))
    assert operation.kind == kind


def test_off_origin_mirror_is_reflection_and_fourfold_screw_retains_pitch():
    identity = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    shifted = SpaceGroup(6).change_setting(identity, (0, Fraction(-1, 6), 0))
    mirror = SymmUnit.from_xyz('x,-y+1/3,z', context=shifted)
    assert mirror.kind == 'reflection' and mirror ** 2 == shifted.identity()
    screw = SymmUnit.from_xyz('-y,x,z+1/4', context=SpaceGroup(76))
    assert screw.kind == 'screw' and (screw ** 4).t == (0, 0, 1)
