import hotpot as hp

from hotpot.cheminfo.core import Bond
from hotpot.cheminfo.search._predicate_ir import (
    _AtomExpression,
    _AtomOp,
    _BondExpression,
    _BondOp,
)
from hotpot.cheminfo.search.native import (
    _compile_problem,
    _compile_query,
    _pack_target,
)
from hotpot.cheminfo.search.query_ir import _query_object_requirements
from hotpot.cheminfo.search.semantics import SmartsSemantics


def _find_atom_expression(expression, op):
    if expression["op"] == op:
        return expression
    for child in expression["children"]:
        found = _find_atom_expression(child, op)
        if found is not None:
            return found
    return None


def test_query_compiler_reproduces_sparse_nodes_and_last_simple_edge():
    substructure = hp.Substructure()
    carbon = hp.QueryAtom(atomic_number={6})
    oxygen = hp.QueryAtom(atomic_number={8})
    substructure.add_atom(carbon)
    substructure.add_atom(carbon)
    substructure.add_atom(oxygen)
    substructure.add_bond(carbon, oxygen, bond_order={1})
    substructure.add_bond(oxygen, carbon, bond_order={2})
    substructure.add_bond(carbon, carbon, is_aromatic={False})

    compiled = _compile_query(substructure)

    assert set(compiled) == {"keys", "edges", "nodes", "bonds"}
    assert compiled["keys"] == (0, 2)
    assert compiled["edges"] == ((0, 0), (0, 2))
    assert compiled["nodes"][0]["leaves"] == (
        ("native_set", "atomic_number", (6,)),
    )
    assert compiled["bonds"][0]["leaves"] == (
        ("native_set", "is_aromatic", (False,)),
    )
    assert compiled["bonds"][1]["leaves"] == (
        ("native_set", "bond_order", (2,)),
    )


def test_constraint_order_preflight_and_host_set_identity_are_preserved():
    callback = lambda atom: atom.atomic_number == 6
    query_atom = hp.QueryAtom(
        atomic_number={6},
        custom_attr={"chosen"},
        is_aromatic=set(),
        predicate=callback,
    )
    substructure = hp.Substructure()
    substructure.add_atom(query_atom)

    program = _compile_query(substructure)["nodes"][0]

    assert program["preflight"] == (
        "atomic_number",
        "custom_attr",
        "is_aromatic",
    )
    assert program["leaves"][0] == (
        "native_set",
        "atomic_number",
        (6,),
    )
    assert program["leaves"][1] == (
        "host_set",
        "custom_attr",
        query_atom.kwargs["custom_attr"],
    )
    assert program["leaves"][1][2] is query_atom.kwargs["custom_attr"]
    assert program["leaves"][2] == ("native_set", "is_aromatic", ())
    assert program["leaves"][3] == ("callable", callback)


def test_unsafe_standard_set_values_remain_host_evaluated():
    query_atom = hp.QueryAtom(atomic_number={True}, is_aromatic={1})
    substructure = hp.Substructure()
    substructure.add_atom(query_atom)

    leaves = _compile_query(substructure)["nodes"][0]["leaves"]

    assert [leaf[0] for leaf in leaves] == ["host_set", "host_set"]


def test_integers_outside_signed_int64_remain_host_evaluated():
    outside_int64 = 1 << 80
    substructure = hp.Substructure()
    first = substructure.add_atom(
        hp.QueryAtom(atomic_number={outside_int64})
    )
    second = substructure.add_atom(hp.QueryAtom())
    substructure.add_bond(first, second, bond_order={-outside_int64})

    compiled = _compile_query(substructure)

    assert compiled["nodes"][0]["leaves"][0][0] == "host_set"
    assert compiled["bonds"][0]["leaves"][0][0] == "host_set"


def test_only_the_owner_specific_private_expression_type_is_lowered():
    bond_expression = _BondExpression(_BondOp.TRUE, "any bond")
    atom_expression = _AtomExpression(_AtomOp.TRUE, "any atom")
    substructure = hp.Substructure()
    first = substructure.add_atom(hp.QueryAtom(predicate=bond_expression))
    second = substructure.add_atom(hp.QueryAtom())
    substructure.add_bond(first, second, predicate=atom_expression)

    compiled = _compile_query(substructure)

    assert compiled["nodes"][0]["leaves"] == (
        ("callable", bond_expression),
    )
    assert compiled["bonds"][0]["leaves"] == (
        ("callable", atom_expression),
    )


def test_typed_atom_and_bond_expressions_encode_recursive_programs():
    substructure = hp.Substructure.from_smarts(
        "[#6;D1;$([#6]-[#8])]C"
    )

    compiled = _compile_query(substructure)
    atom_expression = compiled["nodes"][0]["leaves"][0][1]
    degree = _find_atom_expression(atom_expression, "degree_eq")
    recursive = _find_atom_expression(atom_expression, "recursive")

    assert degree["semantics"] == SmartsSemantics.FULL_GRAPH.value
    assert degree["value"] == 1
    assert set(recursive["value"]) == {"keys", "edges", "nodes", "bonds"}
    assert recursive["value"]["keys"] == (0, 1)
    assert recursive["value"]["edges"] == ((0, 1),)
    assert recursive["value"]["bonds"][0]["bond_kind"] == "single"
    assert compiled["bonds"][0]["leaves"][0][0] == "bond_expr"


def test_object_requirements_include_nested_recursive_queries():
    substructure = hp.Substructure.from_smarts("[$([#6])]C")
    outer_expression = substructure.query_atoms[0].kwargs["predicate"]
    recursive = next(
        expression
        for expression in _walk_atom_ir(outer_expression)
        if expression.op is _AtomOp.RECURSIVE
    )
    nested = recursive.value.substructure
    nested.query_atoms[0].kwargs["custom"] = {1}
    extra = nested.add_atom(hp.QueryAtom())
    nested.add_bond(
        nested.query_atoms[0],
        extra,
        predicate=lambda bond: bond.bond_order == 1,
    )

    requirements = _query_object_requirements(_compile_query(substructure))

    assert requirements == (True, True)


def _walk_atom_ir(expression):
    yield expression
    for child in expression.children:
        yield from _walk_atom_ir(child)


def test_target_packer_snapshots_fields_semantics_and_optional_handles():
    mol = hp.read_mol("C1CC1", "smi")
    packed = _pack_target(
        mol,
        needs_atom_objects=False,
        needs_bond_objects=True,
    )

    assert set(packed) == {"keys", "edges", "atoms", "bonds"}
    assert packed["keys"] == tuple(range(len(mol.atoms)))
    assert packed["edges"] == tuple(
        sorted(tuple(sorted((bond.a1idx, bond.a2idx))) for bond in mol.bonds)
    )
    first_atom = mol.atoms[0]
    atom_record = packed["atoms"][0]
    assert atom_record["fields"]["atomic_number"] == first_atom.atomic_number
    assert atom_record["fields"]["explicit_hydrogens"] == (
        first_atom.explicit_hydrogens
    )
    assert atom_record["full"] == _expected_semantics(
        first_atom, SmartsSemantics.FULL_GRAPH
    )
    assert atom_record["ligand"] == _expected_semantics(
        first_atom, SmartsSemantics.LIGAND_SKELETON
    )
    assert atom_record["object"] is None
    assert packed["bonds"][0]["object"] is mol.bonds[0]
    assert packed["bonds"][0]["bond_kind"] == mol.bonds[0].bond_kind.value


def _expected_semantics(atom, semantics):
    rings = semantics.rings_for(atom)
    return {
        "degree": semantics.degree(atom),
        "connectivity": semantics.connectivity(atom),
        "valence": semantics.valence(atom),
        "ring_count": len(rings),
        "ring_sizes": tuple(len(ring) for ring in rings),
    }


def test_target_packer_reproduces_sparse_nodes_and_last_simple_edge():
    mol = hp.read_mol("CC", "smi")
    first, second = mol.atoms
    mol._atoms.insert(1, first)
    replacement = Bond(first, second, bond_order=2)
    mol._bonds.append(replacement)
    mol._update_graph()

    packed = _pack_target(
        mol,
        needs_atom_objects=True,
        needs_bond_objects=True,
    )

    assert packed["keys"] == (0, 2)
    assert packed["edges"] == ((0, 2),)
    assert packed["bonds"][0]["fields"]["bond_order"] == 2
    assert packed["bonds"][0]["object"] is replacement


def test_target_and_query_are_resnapshotted_on_every_invocation():
    mol = hp.read_mol("C", "smi")
    substructure = hp.Substructure()
    substructure.add_atom(hp.QueryAtom(atomic_number={6}))
    first_query = _compile_query(substructure)
    first_target = _pack_target(
        mol,
        needs_atom_objects=False,
        needs_bond_objects=False,
    )

    substructure.query_atoms[0].kwargs["atomic_number"] = {7}
    mol.atoms[0].formal_charge = 1
    second_query = _compile_query(substructure)
    second_target = _pack_target(
        mol,
        needs_atom_objects=False,
        needs_bond_objects=False,
    )

    assert first_query["nodes"][0]["leaves"][0] == (
        "native_set",
        "atomic_number",
        (6,),
    )
    assert second_query["nodes"][0]["leaves"][0] == (
        "native_set",
        "atomic_number",
        (7,),
    )
    assert first_target["atoms"][0]["fields"]["formal_charge"] == 0
    assert second_target["atoms"][0]["fields"]["formal_charge"] == 1


def test_compile_problem_passes_fresh_specs_and_required_handles(monkeypatch):
    callback = lambda atom: atom.atomic_number == 6
    substructure = hp.Substructure()
    substructure.add_atom(hp.QueryAtom(any_name=callback))
    mol = hp.read_mol("C", "smi")
    captured = {}
    sentinel = object()

    class FakeNativeModule:
        @staticmethod
        def NativeProblem(query_spec, target_spec):
            captured["query"] = query_spec
            captured["target"] = target_spec
            return sentinel

    monkeypatch.setattr(
        "hotpot.cheminfo.search.native._native_module",
        lambda: FakeNativeModule,
    )

    result = _compile_problem(substructure, mol)

    assert result is sentinel
    assert captured["query"]["nodes"][0]["leaves"] == (
        ("callable", callback),
    )
    assert captured["target"]["atoms"][0]["object"] is mol.atoms[0]


def test_wildcard_native_problem_accepts_atomic_number_zero_dummy():
    mol = hp.Molecule()
    atom = mol.create_atom(atomic_number=0, implicit_hydrogens=0)
    substructure = hp.Substructure.from_smarts("[*]")

    assert substructure.query_atoms[0].match(atom)
    assert list(_compile_problem(substructure, mol).cursor()) == [(0,)]
