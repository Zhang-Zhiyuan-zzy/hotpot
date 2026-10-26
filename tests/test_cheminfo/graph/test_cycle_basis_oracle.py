"""Self-validation for the independent Plan 19 cycle-basis oracle."""

from .cycle_basis_oracle import (
    all_simple_cycle_bases,
    canonical_cycle,
    cycle_rank,
    deterministic_paton_basis,
    gf2_rank,
    simple_cycles,
)


TREE = ((0, 1), (1, 2), (1, 3))
TRIANGLE = ((0, 1), (1, 2), (2, 0))
SQUARE_WITH_DIAGONAL = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (0, 2),
)
COMPLETE_4 = tuple(
    (first, second)
    for first in range(4)
    for second in range(first + 1, 4)
)
DISCONNECTED_TRIANGLES = TRIANGLE + ((4, 5), (5, 6), (6, 4))


def test_cycle_keys_ignore_rotation_and_direction():
    assert canonical_cycle((0, 3, 2, 1)) == (0, 1, 2, 3)
    assert canonical_cycle((2, 3, 0, 1)) == (0, 1, 2, 3)


def test_gf2_rank_is_independent_of_input_order_and_dependent_vectors():
    assert gf2_rank((0b011, 0b101, 0b110)) == 2
    assert gf2_rank((0b110, 0b011, 0b101)) == 2
    assert gf2_rank(()) == 0


def test_acyclic_graph_has_one_empty_basis():
    nodes = range(4)
    assert simple_cycles(nodes, TREE) == ()
    assert cycle_rank(nodes, TREE) == 0
    assert deterministic_paton_basis(nodes, TREE) == ()
    assert all_simple_cycle_bases(nodes, TREE) == ((),)


def test_triangle_has_one_cycle_and_one_basis():
    nodes = range(3)
    expected = (((0, 1, 2),),)
    assert simple_cycles(nodes, TRIANGLE) == ((0, 1, 2),)
    assert cycle_rank(nodes, TRIANGLE) == 1
    assert deterministic_paton_basis(nodes, TRIANGLE) == ((0, 1, 2),)
    assert all_simple_cycle_bases(nodes, TRIANGLE) == expected


def test_square_with_diagonal_has_three_distinct_simple_cycle_bases():
    nodes = range(4)
    assert simple_cycles(nodes, SQUARE_WITH_DIAGONAL) == (
        (0, 1, 2),
        (0, 2, 3),
        (0, 1, 2, 3),
    )
    assert cycle_rank(nodes, SQUARE_WITH_DIAGONAL) == 2
    assert deterministic_paton_basis(nodes, SQUARE_WITH_DIAGONAL) == (
        (0, 2, 3),
        (0, 1, 2),
    )
    assert len(all_simple_cycle_bases(nodes, SQUARE_WITH_DIAGONAL)) == 3


def test_complete_four_has_all_seven_nonzero_cycle_vectors_and_28_bases():
    nodes = range(4)
    cycles = simple_cycles(nodes, COMPLETE_4)
    assert len(cycles) == 7
    assert cycle_rank(nodes, COMPLETE_4) == 3
    assert deterministic_paton_basis(nodes, COMPLETE_4) == (
        (0, 2, 3),
        (1, 2, 3),
        (0, 1, 3),
    )
    assert len(all_simple_cycle_bases(nodes, COMPLETE_4)) == 28


def test_disconnected_components_are_complete_and_visit_high_component_first():
    nodes = range(7)
    assert cycle_rank(nodes, DISCONNECTED_TRIANGLES) == 2
    assert deterministic_paton_basis(nodes, DISCONNECTED_TRIANGLES) == (
        (4, 5, 6),
        (0, 1, 2),
    )
    assert all_simple_cycle_bases(nodes, DISCONNECTED_TRIANGLES) == (
        ((0, 1, 2), (4, 5, 6)),
    )


def test_oracle_is_invariant_to_edge_order_and_orientation():
    shuffled = tuple(
        tuple(reversed(edge)) for edge in reversed(SQUARE_WITH_DIAGONAL)
    )
    nodes = range(4)
    assert simple_cycles(nodes, shuffled) == simple_cycles(
        nodes, SQUARE_WITH_DIAGONAL
    )
    assert deterministic_paton_basis(nodes, shuffled) == deterministic_paton_basis(
        nodes, SQUARE_WITH_DIAGONAL
    )
    assert all_simple_cycle_bases(nodes, shuffled) == all_simple_cycle_bases(
        nodes, SQUARE_WITH_DIAGONAL
    )
