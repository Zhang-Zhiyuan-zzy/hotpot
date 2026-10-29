"""Freeze the existing row/angle convention during mathematical extraction."""

import numpy as np

from hotpot.cheminfo.crystal import Crystal


def test_oblique_cell_preserves_alpha_between_the_first_two_rows():
    matrix = np.array(((1., 0., 0.), (.5, np.sqrt(3.) / 2., 0.), (0., 0., 2.)))
    cell = Crystal.from_matrix(matrix)
    np.testing.assert_allclose(
        (cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma),
        (1., 1., 2., 60., 90., 90.), atol=1e-12)
    assert not hasattr(cell, 'matrix')


def test_orthogonal_cell_has_independent_lengths_and_degree_angles():
    cell = Crystal.from_matrix(np.diag((2., 3., 5.)))
    assert (cell.a, cell.b, cell.c) == (2., 3., 5.)
    assert (cell.alpha, cell.beta, cell.gamma) == (90., 90., 90.)
