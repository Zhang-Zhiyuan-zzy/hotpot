"""Parity contracts for the production native search predicate program."""

import networkx as nx
import pytest

import hotpot as hp
from hotpot.cheminfo.core import Atom, Bond, BondKind
from hotpot.cheminfo.search import native
from hotpot.cheminfo.search.search import QueryAtom, Substructure


def _mapping_key(mapping):
    return tuple(sorted(mapping.items()))


def _query_graph(substructure):
    graph = nx.Graph()
    graph.add_nodes_from(
        (atom.idx, {"qa": atom}) for atom in substructure.query_atoms
    )
    graph.add_edges_from(
        (bond.a1idx, bond.a2idx, {"qb": bond})
        for bond in substructure.query_bonds
    )
    return graph


def _legacy_raw_mappings(substructure, molecule):
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        molecule.atom_bond_graph,
        _query_graph(substructure),
        node_match=lambda target, query: query["qa"].match(target["atom"]),
        edge_match=lambda target, query: query["qb"].match(target["bond"]),
    )
    return tuple(
        _mapping_key(mapping)
        for mapping in matcher.subgraph_monomorphisms_iter()
    )


def _native_raw_mappings(problem):
    return tuple(
        _mapping_key(mapping)
        for mapping in native._NativeMappingSource(
            problem
        ).subgraph_monomorphisms_iter()
    )


def _assert_native_parity(substructure, molecule, *, expected_count=None):
    legacy = _legacy_raw_mappings(substructure, molecule)
    problem = native._compile_problem(substructure, molecule)
    first = _native_raw_mappings(problem)
    second = _native_raw_mappings(problem)

    assert first == second
    assert len(first) == len(set(first))
    assert len(legacy) == len(set(legacy))
    assert set(first) == set(legacy)
    if expected_count is not None:
        assert len(legacy) == expected_count
    return problem, first


@pytest.mark.parametrize(
    ("smarts", "smiles"),
    (
        ("[*]", "CCO"),
        ("[#6]", "CCO"),
        ("[N+]", "C[NH3+]"),
        ("[a]", "c1ccccc1"),
        ("[NG1]", "[Na+]"),
        ("[Ln]", "[Eu]"),
        ("[An]", "[Am]"),
        ("[M]", "[Fe]"),
        ("[NG3-8]", "[Fe]"),
        ("[NP3-5]", "Cl"),
        ("[D2]", "CCC"),
        ("[X4]", "C"),
        ("[v3]", "N"),
        ("[R]", "c1ccccc1"),
        ("[R2]", "c1ccc2ccccc2c1"),
        ("[r0]", "CC"),
        ("[r6]", "c1ccccc1"),
        ("[H2]", "C=C"),
        ("[C;D1]", "CC"),
        ("[C,N]", "CN"),
        ("[!M]", "C[Fe]"),
    ),
)
def test_every_non_recursive_typed_atom_operation_matches_legacy(smarts, smiles):
    substructure = Substructure.from_smarts(smarts)
    molecule = hp.read_mol(smiles, "smi")

    _, mappings = _assert_native_parity(substructure, molecule)

    assert mappings


@pytest.mark.parametrize(
    ("smarts", "smiles"),
    (
        ("C-C", "CCC"),
        ("C=O", "CC=O"),
        ("C#N", "CC#N"),
        ("c:c", "c1ccccc1"),
        ("C~C", "C=C"),
        ("C=,:C", "C=C"),
        ("CC", "CC"),
    ),
)
def test_typed_and_explicit_bond_operations_match_legacy(smarts, smiles):
    substructure = Substructure.from_smarts(smarts)
    molecule = hp.read_mol(smiles, "smi")

    _, mappings = _assert_native_parity(substructure, molecule)

    assert mappings


def test_substructure_from_molecule_set_constraints_match_legacy():
    substructure = Substructure.from_mol(hp.read_mol("CO", "smi"))
    molecule = hp.read_mol("CCO", "smi")

    _assert_native_parity(substructure, molecule, expected_count=1)


@pytest.mark.parametrize(
    ("attrs", "expected_count"),
    (
        ({"symbol": {"C"}}, 2),
        ({"attribute_that_does_not_exist": {1}}, 0),
        ({"atomic_number": set()}, 3),
        ({"attribute_that_does_not_exist": set()}, 0),
    ),
)
def test_named_host_sets_and_empty_or_missing_attributes_match_legacy(
    attrs, expected_count
):
    substructure = Substructure()
    substructure.add_atom(QueryAtom(**attrs))

    _assert_native_parity(
        substructure,
        hp.read_mol("CCO", "smi"),
        expected_count=expected_count,
    )


def test_named_bond_host_set_matches_legacy_python_membership():
    substructure = Substructure()
    substructure.add_atom(QueryAtom(atomic_number={6}))
    substructure.add_atom(QueryAtom(atomic_number={6}))
    substructure.add_bond(0, 1, bond_kind={BondKind.DOUBLE})

    _assert_native_parity(
        substructure,
        hp.read_mol("CC=C", "smi"),
        expected_count=2,
    )


@pytest.mark.parametrize(
    ("guard", "target_smiles"),
    (
        ({"_smarts_exclude_aromatic": True}, "c1ccccc1"),
        ({"_smarts_bond_kind": BondKind.SINGLE}, "C=C"),
    ),
)
def test_programmatic_bond_guards_match_legacy(guard, target_smiles):
    substructure = Substructure()
    substructure.add_atom(QueryAtom(atomic_number={6}))
    substructure.add_atom(QueryAtom(atomic_number={6}))
    substructure.add_bond(0, 1, **guard)

    _assert_native_parity(
        substructure,
        hp.read_mol(target_smiles, "smi"),
        expected_count=0,
    )


def _bound_tertiary_amine():
    molecule = hp.Molecule()
    for atomic_number in (7, 6, 6, 6, 29):
        molecule.create_atom(
            atomic_number=atomic_number,
            implicit_hydrogens=0,
        )
    for carbon_index in (1, 2, 3):
        molecule.add_bond(
            0,
            carbon_index,
            1.0,
            bond_kind=BondKind.SINGLE,
        )
    molecule.add_bond(
        0,
        4,
        1.0,
        bond_kind=BondKind.DATIVE,
        bond_direction="atom2_to_atom1",
    )
    return molecule


def _ethylenediamine_chelate():
    molecule = hp.Molecule()
    for atomic_number, implicit_hydrogens in (
        (29, 0),
        (7, 1),
        (6, 2),
        (6, 2),
        (7, 1),
    ):
        molecule.create_atom(
            atomic_number=atomic_number,
            implicit_hydrogens=implicit_hydrogens,
        )
    for first, second in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 0)):
        molecule.add_bond(first, second, 1.0)
    return molecule


@pytest.mark.parametrize(
    ("semantics", "smarts", "expected_count"),
    (
        (hp.SmartsSemantics.FULL_GRAPH, "[N;D4;X4;v4]", 1),
        (hp.SmartsSemantics.LIGAND_SKELETON, "[N;D3;X3;v3]", 1),
        (hp.SmartsSemantics.FULL_GRAPH, "[N;D3;X3;v3]", 0),
        (hp.SmartsSemantics.LIGAND_SKELETON, "[N;D4;X4;v4]", 0),
    ),
)
def test_full_graph_and_ligand_skeleton_derived_values_match_legacy(
    semantics, smarts, expected_count
):
    substructure = Substructure.from_smarts(smarts, semantics=semantics)

    _assert_native_parity(
        substructure,
        _bound_tertiary_amine(),
        expected_count=expected_count,
    )


@pytest.mark.parametrize(
    ("semantics", "smarts", "expected_count"),
    (
        (hp.SmartsSemantics.FULL_GRAPH, "[M;r5]", 1),
        (hp.SmartsSemantics.LIGAND_SKELETON, "[M;r5]", 0),
        (hp.SmartsSemantics.FULL_GRAPH, "[N;r5]", 2),
        (hp.SmartsSemantics.LIGAND_SKELETON, "[N;r0]", 2),
    ),
)
def test_full_graph_and_ligand_ring_views_match_legacy(
    semantics, smarts, expected_count
):
    _assert_native_parity(
        Substructure.from_smarts(smarts, semantics=semantics),
        _ethylenediamine_chelate(),
        expected_count=expected_count,
    )


@pytest.mark.parametrize(
    ("smarts", "smiles", "expected_count"),
    (
        ("[#6;$([#6]=[#8])]", "CC(=O)N", 1),
        ("[$([N]-C)]", "CN", 1),
        ("[N;!$([N]-C)]", "N.C", 1),
    ),
)
def test_recursive_typed_predicates_are_anchored_and_match_legacy(
    smarts, smiles, expected_count
):
    _assert_native_parity(
        Substructure.from_smarts(smarts),
        hp.read_mol(smiles, "smi"),
        expected_count=expected_count,
    )


@pytest.mark.parametrize(
    ("smiles", "expected_count"),
    (("N.O", 1), ("N", 0)),
)
def test_recursive_disconnected_components_share_anchored_injectivity(
    smiles, expected_count
):
    _assert_native_parity(
        Substructure.from_smarts("[$([N].O)]"),
        hp.read_mol(smiles, "smi"),
        expected_count=expected_count,
    )


@pytest.mark.parametrize(
    ("semantics", "expected_count"),
    (
        (hp.SmartsSemantics.FULL_GRAPH, 1),
        (hp.SmartsSemantics.LIGAND_SKELETON, 0),
    ),
)
def test_recursive_query_inherits_ring_semantics(semantics, expected_count):
    _assert_native_parity(
        Substructure.from_smarts("[$([M;r5])]", semantics=semantics),
        _ethylenediamine_chelate(),
        expected_count=expected_count,
    )


def test_opaque_atom_callable_ignores_keyword_and_receives_whole_atom():
    molecule = hp.read_mol("CO", "smi")
    seen = []

    def is_carbon(atom):
        seen.append(atom)
        return isinstance(atom, Atom) and atom.atomic_number == 6

    substructure = Substructure()
    substructure.add_atom(
        QueryAtom(attribute_that_does_not_exist=is_carbon)
    )
    legacy = _legacy_raw_mappings(substructure, molecule)
    seen.clear()

    problem = native._compile_problem(substructure, molecule)
    assert seen == []
    first = _native_raw_mappings(problem)
    assert set(first) == set(legacy)
    assert seen and {id(atom) for atom in seen} <= {
        id(atom) for atom in molecule.atoms
    }

    seen.clear()
    second = _native_raw_mappings(problem)
    assert second == first
    assert seen and all(isinstance(atom, Atom) for atom in seen)


def test_opaque_bond_callable_ignores_keyword_and_receives_whole_bond():
    molecule = hp.read_mol("CC=C", "smi")
    seen = []

    def is_double(bond):
        seen.append(bond)
        return isinstance(bond, Bond) and bond.bond_order == 2

    substructure = Substructure()
    substructure.add_atom(QueryAtom(atomic_number={6}))
    substructure.add_atom(QueryAtom(atomic_number={6}))
    substructure.add_bond(
        0,
        1,
        attribute_that_does_not_exist=is_double,
    )
    legacy = _legacy_raw_mappings(substructure, molecule)
    seen.clear()

    problem = native._compile_problem(substructure, molecule)
    assert seen == []
    first = _native_raw_mappings(problem)
    assert set(first) == set(legacy)
    assert seen and {id(bond) for bond in seen} <= {
        id(bond) for bond in molecule.bonds
    }

    seen.clear()
    assert _native_raw_mappings(problem) == first
    assert seen and all(isinstance(bond, Bond) for bond in seen)


def test_mixed_native_and_opaque_constraints_short_circuit_in_insertion_order():
    typed_carbon = Substructure.from_smarts(
        "[#6]"
    ).query_atoms[0].kwargs["predicate"]
    calls = []

    def should_not_run(atom):
        calls.append(atom)
        raise AssertionError("short-circuited opaque leaf was evaluated")

    substructure = Substructure()
    substructure.add_atom(
        QueryAtom(predicate=typed_carbon, opaque_leaf=should_not_run)
    )
    problem = native._compile_problem(
        substructure,
        hp.read_mol("O", "smi"),
    )

    assert _native_raw_mappings(problem) == ()
    assert calls == []


def test_missing_set_attribute_preflight_precedes_an_earlier_callable():
    calls = []

    def should_not_run(atom):
        calls.append(atom)
        raise AssertionError("callable ran before missing-attribute preflight")

    substructure = Substructure()
    substructure.add_atom(
        QueryAtom(
            opaque_leaf=should_not_run,
            attribute_that_does_not_exist={1},
        )
    )
    problem = native._compile_problem(
        substructure,
        hp.read_mol("C", "smi"),
    )

    assert _native_raw_mappings(problem) == ()
    assert calls == []


def test_bond_guard_short_circuits_opaque_constraint():
    calls = []

    def should_not_run(bond):
        calls.append(bond)
        raise AssertionError("guarded opaque bond leaf was evaluated")

    substructure = Substructure()
    substructure.add_atom(QueryAtom(atomic_number={6}))
    substructure.add_atom(QueryAtom(atomic_number={6}))
    substructure.add_bond(
        0,
        1,
        _smarts_bond_kind=BondKind.SINGLE,
        opaque_leaf=should_not_run,
    )
    problem = native._compile_problem(
        substructure,
        hp.read_mol("C=C", "smi"),
    )

    assert _native_raw_mappings(problem) == ()
    assert calls == []


def test_callable_is_not_evaluated_while_problem_is_compiled():
    calls = []

    def accept(atom):
        calls.append(atom.idx)
        return True

    substructure = Substructure()
    substructure.add_atom(QueryAtom(opaque_leaf=accept))
    problem = native._compile_problem(
        substructure,
        hp.read_mol("CO", "smi"),
    )

    assert calls == []
    assert len(_native_raw_mappings(problem)) == 2
    assert calls == [0, 1]


def test_custom_host_attribute_is_resolved_lazily_during_traversal(
    monkeypatch,
):
    accesses = []

    def probe_attribute(atom):
        accesses.append(atom.idx)
        return atom.atomic_number

    monkeypatch.setattr(
        Atom,
        "native_search_probe_attribute",
        property(probe_attribute),
        raising=False,
    )
    substructure = Substructure()
    substructure.add_atom(
        QueryAtom(native_search_probe_attribute={6})
    )
    problem = native._compile_problem(
        substructure,
        hp.read_mol("CO", "smi"),
    )

    assert accesses == []
    assert len(_native_raw_mappings(problem)) == 1
    assert accesses


def test_topologically_unreachable_callable_leaf_is_never_evaluated():
    calls = []

    def unreachable(atom):
        calls.append(atom)
        raise AssertionError("topologically unreachable leaf was evaluated")

    substructure = Substructure()
    substructure.add_atom(QueryAtom())
    substructure.add_atom(QueryAtom(opaque_leaf=unreachable))
    substructure.add_atom(QueryAtom())
    substructure.add_bond(0, 1)
    substructure.add_bond(1, 2)
    problem = native._compile_problem(
        substructure,
        hp.read_mol("C", "smi"),
    )

    assert calls == []
    assert _native_raw_mappings(problem) == ()
    assert calls == []


class _ExpectedPredicateError(RuntimeError):
    pass


@pytest.mark.parametrize("leaf_kind", ("atom", "bond"))
def test_opaque_callable_exception_propagates_from_the_reached_leaf(leaf_kind):
    def fail(obj):
        raise _ExpectedPredicateError(f"{leaf_kind} predicate failed")

    substructure = Substructure()
    substructure.add_atom(
        QueryAtom(opaque_leaf=fail) if leaf_kind == "atom" else QueryAtom()
    )
    if leaf_kind == "bond":
        substructure.add_atom(QueryAtom())
        substructure.add_bond(0, 1, opaque_leaf=fail)
    molecule = hp.read_mol("CC", "smi")
    problem = native._compile_problem(substructure, molecule)
    mappings = native._NativeMappingSource(
        problem
    ).subgraph_monomorphisms_iter()

    with pytest.raises(
        _ExpectedPredicateError,
        match=rf"{leaf_kind} predicate failed",
    ):
        next(iter(mappings))


def test_opaque_callable_stays_inside_native_topology_traversal(monkeypatch):
    substructure = Substructure()
    substructure.add_atom(
        QueryAtom(opaque_leaf=lambda atom: atom.atomic_number == 6)
    )

    def reject_graph_matcher(*args, **kwargs):
        raise AssertionError("native predicate search constructed GraphMatcher")

    monkeypatch.setattr(
        nx.algorithms.isomorphism,
        "GraphMatcher",
        reject_graph_matcher,
    )
    problem = native._compile_problem(
        substructure,
        hp.read_mol("CO", "smi"),
    )

    assert len(_native_raw_mappings(problem)) == 1


def test_duplicate_query_atom_preserves_sparse_normalized_keys():
    substructure = Substructure()
    duplicate = QueryAtom(atomic_number={6})
    substructure.add_atom(duplicate)
    substructure.add_atom(duplicate)
    tail = substructure.add_atom(QueryAtom(atomic_number={6}))
    substructure.add_bond(duplicate, tail, bond_order={1})

    problem, _ = _assert_native_parity(
        substructure,
        hp.read_mol("CC", "smi"),
        expected_count=2,
    )

    assert tuple(problem.query_keys) == (0, 2)


def test_duplicate_query_edge_uses_the_last_predicate():
    substructure = Substructure()
    first = substructure.add_atom(QueryAtom(atomic_number={6}))
    second = substructure.add_atom(QueryAtom(atomic_number={6}))
    substructure.add_bond(first, second, bond_order={2})
    substructure.add_bond(first, second, bond_order={1})

    _assert_native_parity(
        substructure,
        hp.read_mol("CC", "smi"),
        expected_count=2,
    )
    _assert_native_parity(
        substructure,
        hp.read_mol("C=C", "smi"),
        expected_count=0,
    )


def test_empty_normalized_query_matches_legacy_once():
    problem, mappings = _assert_native_parity(
        Substructure(),
        hp.read_mol("CC", "smi"),
        expected_count=1,
    )

    assert tuple(problem.query_keys) == ()
    assert mappings == ((),)
