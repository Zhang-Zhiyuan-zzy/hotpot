"""Standard crystallographic angles and orientation replace the old scalar loss."""

import numpy as np

from hotpot.cheminfo.structure.crystal import Crystal


def test_oblique_cell_uses_standard_gamma_between_the_first_two_rows():
    matrix = np.array(((1., 0., 0.), (.5, np.sqrt(3.) / 2., 0.), (0., 0., 2.)))
    cell = Crystal.from_matrix(matrix)
    np.testing.assert_allclose(
        (cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma),
        (1., 1., 2., 90., 90., 60.), atol=1e-12)
    np.testing.assert_array_equal(cell.matrix, matrix)


def test_orthogonal_cell_has_independent_lengths_and_degree_angles():
    cell = Crystal.from_matrix(np.diag((2., 3., 5.)))
    assert (cell.a, cell.b, cell.c) == (2., 3., 5.)
    assert (cell.alpha, cell.beta, cell.gamma) == (90., 90., 90.)


def test_from_matrix_keeps_subclass_and_full_basis_without_scalar_reconstruction():
    class SpecializedCrystal(Crystal):
        @staticmethod
        def _matrix_to_params(matrix):
            return *np.diag(matrix), 80., 85., 95.

    cell = SpecializedCrystal.from_matrix(np.diag((2., 3., 5.)))
    assert type(cell) is SpecializedCrystal
    assert (cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma) == (2., 3., 5., 90., 90., 90.)
