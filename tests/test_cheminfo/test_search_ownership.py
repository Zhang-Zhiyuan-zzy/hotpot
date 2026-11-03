"""Connected ownership and lazy-call contracts for the Stage8C search split."""

import gc
import subprocess
import sys
import textwrap
from types import SimpleNamespace
import weakref

import pytest

import hotpot as hp
from hotpot.cheminfo.structure.bond_kind import BondKind
from hotpot.cheminfo.structure.substructure import (
    SmartsSemantics,
    _AtomExpression,
    _AtomField,
    _AtomOp,
    _BondExpression,
    _BondOp,
)


def test_cold_facades_resolve_the_ten_physical_owners_and_contract_metadata():
    program = textwrap.dedent(
        """
        import dataclasses
        import importlib
        import pickle
        import sys

        import hotpot as hp
        import hotpot.cheminfo as cheminfo
        import hotpot.cheminfo.operations.search as search
        from hotpot.cheminfo.structure import substructure as contracts

        owners = {
            "hotpot.cheminfo.operations.search.search": (
                "Query", "QueryAtom", "QueryBond", "Substructure",
                "Searcher", "Hits", "Hit",
            ),
            "hotpot.cheminfo.structure.substructure": ("SmartsSemantics",),
            "hotpot.cheminfo.rules.chemistry.substructure.errors": (
                "SmartsSyntaxError", "UnsupportedSmartsError",
            ),
        }
        names = {name for members in owners.values() for name in members}
        assert len(names) == 10
        assert set(search.__all__) == names
        for module_name, members in owners.items():
            module = importlib.import_module(module_name)
            for name in members:
                canonical = getattr(module, name)
                assert canonical.__module__ == module_name
                assert canonical.__qualname__ == name
                assert getattr(hp, name) is canonical
                assert getattr(cheminfo, name) is canonical
                assert getattr(search, name) is canonical

        for member in contracts.SmartsSemantics:
            assert pickle.loads(pickle.dumps(member)) is member
        for name in ("_AtomField", "_AtomOp", "_BondOp", "_AtomExpression", "_BondExpression"):
            cls = getattr(contracts, name)
            assert cls.__module__ == "hotpot.cheminfo.structure.substructure"
            assert cls.__qualname__ == name
            assert pickle.loads(pickle.dumps(cls)) is cls
        for cls in (contracts._AtomExpression, contracts._BondExpression):
            assert cls.__dataclass_params__.frozen
            assert not cls.__dataclass_params__.eq
        atom = contracts._AtomExpression(contracts._AtomOp.TRUE, "any atom")
        bond = contracts._BondExpression(contracts._BondOp.TRUE, "any bond")
        assert atom.children == bond.children == ()
        assert atom.field is atom.value is atom.semantics is None
        assert atom.aromatic_states == frozenset((False, True))
        assert bond.kind is None
        assert repr(atom) == "any atom"
        assert repr(bond) == "any bond"
        assert not any(
            name == "hotpot.cheminfo.search" or name.startswith("hotpot.cheminfo.search.")
            for name in sys.modules
        )
        print("cold-search-owners-ok")
        """
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", program],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "cold-search-owners-ok"


def _direct_match_indices(query_atom, molecule):
    return {atom.idx for atom in molecule.atoms if query_atom.match(atom)}


def test_one_recursive_query_rechecks_all_anchors_across_molecules_and_mutation():
    query = hp.Substructure.from_smarts("[$([#6]=[#8])]")
    query_atom = query.query_atoms[0]
    two_carbonyls = hp.read_mol("CC(=O)N.C=O", "smi")
    formaldehyde = hp.read_mol("C=O", "smi")

    assert _direct_match_indices(query_atom, two_carbonyls) == {1, 4}
    assert _direct_match_indices(query_atom, formaldehyde) == {0}
    assert _direct_match_indices(query_atom, two_carbonyls) == {1, 4}

    amide_carbonyl_bond = next(
        bond for bond in two_carbonyls.bonds
        if {bond.a1idx, bond.a2idx} == {1, 2}
    )
    amide_carbonyl_bond.bond_order = 1
    assert _direct_match_indices(query_atom, two_carbonyls) == {4}
    assert _direct_match_indices(query_atom, formaldehyde) == {0}

    amide_carbonyl_bond.bond_order = 2
    assert _direct_match_indices(query_atom, two_carbonyls) == {1, 4}


def test_direct_recursive_calls_execute_fresh_tasks_and_release_the_target():
    query_atom = hp.Substructure.from_smarts("[$([#6]=[#8])]").query_atoms[0]
    recursive = query_atom.kwargs["predicate"].value
    observed = []

    def observe(atom):
        observed.append(atom.idx)
        return True

    recursive.substructure.query_atoms[0].kwargs["observe"] = observe
    molecule = hp.read_mol("C=O", "smi")
    for _ in range(2):
        previous_calls = len(observed)
        assert query_atom.match(molecule.atoms[0])
        assert len(observed) > previous_calls

    assert set(vars(recursive)) == {"smarts", "semantics", "substructure"}
    molecule_reference = weakref.ref(molecule)
    del molecule
    gc.collect()

    assert molecule_reference() is None
    assert query_atom.kwargs["predicate"].value is recursive


def test_direct_custom_atom_reads_only_reached_scalar_fields_and_keeps_exceptions():
    accessed = []
    reached_error = RuntimeError("reached atomic number")

    class Probe:
        group_value = 15

        @property
        def group(self):
            accessed.append("group")
            return self.group_value

        @property
        def explicit_hydrogens(self):
            accessed.append("explicit_hydrogens")
            return 0

        @property
        def implicit_hydrogens(self):
            accessed.append("implicit_hydrogens")
            return 3

        @property
        def atomic_number(self):
            accessed.append("atomic_number")
            raise reached_error

    expression = _AtomExpression(
        _AtomOp.OR,
        "(group15 & !H2) | period3",
        children=(
            _AtomExpression(
                _AtomOp.AND,
                "group15 & !H2",
                children=(
                    _AtomExpression(
                        _AtomOp.FIELD_IN, "group15", field=_AtomField.GROUP,
                        value=frozenset({15}),
                    ),
                    _AtomExpression(
                        _AtomOp.NOT, "!H2",
                        children=(_AtomExpression(_AtomOp.TOTAL_H_EQ, "H2", value=2),),
                    ),
                ),
            ),
            _AtomExpression(_AtomOp.PERIOD_IN, "period3", value=frozenset({3})),
        ),
    )
    probe = Probe()
    assert expression(probe)
    assert accessed == ["group", "explicit_hydrogens", "implicit_hydrogens"]

    accessed.clear()
    probe.group_value = 16
    with pytest.raises(RuntimeError) as error:
        expression(probe)
    assert error.value is reached_error
    assert accessed == ["group", "atomic_number"]


def test_direct_ring_size_comparison_reads_later_custom_rings_only_when_reached():
    sizes_read = []
    reached_error = RuntimeError("reached later ring")

    class Ring:
        def __init__(self, name, size):
            self.name = name
            self.size = size

        def __contains__(self, atom):
            return atom is probe

        def __len__(self):
            sizes_read.append(self.name)
            if self.name == "later":
                raise reached_error
            return self.size

    first_ring = Ring("first", 6)
    probe = SimpleNamespace(
        mol=SimpleNamespace(rings=(first_ring, Ring("later", 5))),
    )
    expression = _AtomExpression(
        _AtomOp.RING_SIZE_EQ, "r6", value=6, semantics=SmartsSemantics.FULL_GRAPH,
    )

    assert expression(probe)
    assert sizes_read == ["first"]

    first_ring.size = 5
    sizes_read.clear()
    with pytest.raises(RuntimeError) as error:
        expression(probe)
    assert error.value is reached_error
    assert sizes_read == ["first", "later"]


def test_direct_custom_bond_short_circuits_aromatic_access_and_keeps_exceptions():
    accessed = []
    reached_error = RuntimeError("reached aromatic flag")

    class Probe:
        kind = BondKind.SINGLE

        @property
        def bond_kind(self):
            accessed.append("bond_kind")
            return self.kind

        @property
        def is_aromatic(self):
            accessed.append("is_aromatic")
            raise reached_error

    expression = _BondExpression(
        _BondOp.OR,
        "single-or-aromatic",
        children=(
            _BondExpression(_BondOp.KIND_IS, "-", kind=BondKind.SINGLE),
            _BondExpression(_BondOp.AROMATIC, ":"),
        ),
    )
    probe = Probe()
    assert expression(probe)
    assert accessed == ["bond_kind"]

    probe.kind = BondKind.DOUBLE
    accessed.clear()
    with pytest.raises(RuntimeError) as error:
        expression(probe)
    assert error.value is reached_error
    assert accessed == ["bond_kind", "is_aromatic"]


def test_documented_acetamide_recursive_query_maps_the_amide_carbonyl_carbon():
    molecule = hp.read_mol("CC(=O)N", "smi")
    query = hp.Substructure.from_smarts("[#6;$([#6](=[#8])[#7])]")
    searcher = hp.Searcher(query)

    assert searcher.has_match(molecule)
    assert tuple(dict(mapping) for mapping in searcher.iter_mappings(molecule)) == ({0: 1},)
    carbonyl = molecule.atoms[1]
    assert carbonyl.atomic_number == 6
    assert sorted(atom.atomic_number for atom in carbonyl.neighbours) == [6, 7, 8]
    assert {
        frozenset((bond.a1idx, bond.a2idx)): bond.bond_order
        for bond in molecule.bonds
    } == {
        frozenset((0, 1)): 1,
        frozenset((1, 2)): 2,
        frozenset((1, 3)): 1,
    }


@pytest.mark.parametrize(
    ("smiles", "expected_hydrogens", "atom_sites", "bond_sites"),
    (
        ("CCO", (3, 2, 1), {0, 1}, {frozenset((0, 1))}),
        ("CC(=O)N", (3, 0, 0, 2), {0, 3}, set()),
    ),
)
def test_assembly_searchers_use_their_real_hydrogen_callable(
    smiles, expected_hydrogens, atom_sites, bond_sites,
):
    from hotpot.cheminfo.mol_assemble.assembler import AtomLink, EdgeShoulder

    molecule = hp.read_mol(smiles, "smi")
    molecule.calc_implicit_hydrogens()
    assert tuple(atom.implicit_hydrogens for atom in molecule.atoms) == expected_hydrogens
    assert {
        mapping[0] for mapping in AtomLink.searcher.iter_mappings(molecule)
    } == atom_sites

    hits = EdgeShoulder.searcher.search(molecule)
    hits.get_hit = False
    assert set(hits) == bond_sites


@pytest.mark.parametrize("hydrogen_source", [
    "inferred",
    pytest.param("toolkit", marks=pytest.mark.xfail(
        strict=True,
        reason="P20-D024: existing assembly does not consume imported H; Stage8E owns the transformation",
    )),
])
def test_atom_link_grafts_oxygen_onto_methane_as_methanol_without_changing_sources(hydrogen_source):
    from hotpot.cheminfo.mol_assemble.assembler import AtomLink

    if hydrogen_source == "toolkit":
        frame = hp.read_mol("C", "smi")
        fragment = hp.read_mol("O", "smi")
    else:
        frame, fragment = hp.Molecule(), hp.Molecule()
        frame.create_atom(atomic_number=6)
        fragment.create_atom(atomic_number=8)
    frame.calc_implicit_hydrogens()
    fragment.calc_implicit_hydrogens()
    assembler = AtomLink(fragment, action_points=(0,))

    products = assembler.graft(frame)

    assert len(products) == 1
    smiles, methanol = next(iter(products.items()))
    assert smiles == methanol.smiles == "CO"
    assert tuple(atom.atomic_number for atom in methanol.atoms) == (6, 8)
    assert tuple(atom.implicit_hydrogens for atom in methanol.atoms) == (3, 1)
    assert tuple(atom.formal_charge for atom in methanol.atoms) == (0, 0)
    assert len(methanol.bonds) == 1
    assert {methanol.bonds[0].a1idx, methanol.bonds[0].a2idx} == {0, 1}
    assert methanol.bonds[0].bond_order == 1
    assert methanol.bonds[0].bond_kind is BondKind.SINGLE
    assert len(frame.atoms) == len(fragment.atoms) == 1
    assert not frame.bonds and not fragment.bonds
    assert frame.atoms[0].atomic_number == 6
    assert frame.atoms[0].implicit_hydrogens == 4
    assert fragment.atoms[0].atomic_number == 8
    assert fragment.atoms[0].implicit_hydrogens == 2
