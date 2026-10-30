from copy import copy

import pytest

from hotpot.cheminfo import MolReader
from hotpot.cheminfo.core import Bond, Molecule
from hotpot.cheminfo.search._search_native import NativeTarget
from hotpot.cheminfo.search.native import _NativeTarget
from tests.test_cheminfo._stage5_fixtures import supplied_aromatic_flags
from hotpot.cheminfo.stereo import (
    AtomTransportRecord,
    StereoIssueCode,
    StereoPayload,
    get_stereo_payload,
    set_stereo_payload,
)
from hotpot.cheminfo.operations.stereochemistry import _gate6_projection
from hotpot.cheminfo.rules.chemistry.stereochemistry.graph import _gate6_query_spec

from hotpot.cheminfo.adapters.structure.stereochemistry import _pack_gate6_target


def _molecule(atomic_numbers, edges=(), *, implicit_hydrogens=()):
    molecule = Molecule()
    implicit_hydrogens = implicit_hydrogens or (0,) * len(atomic_numbers)
    for atomic_number, hydrogen_count in zip(
        atomic_numbers,
        implicit_hydrogens,
    ):
        molecule._create_atom(
            atomic_number=atomic_number,
            implicit_hydrogens=hydrogen_count,
        )
    for first, second, order in edges:
        molecule._add_bond(first, second, bond_order=order)
    molecule._update_topology(clear_conformers=False)
    return molecule


def _atom_fact(row, *, isotope=None, radical=None):
    return AtomTransportRecord(
        atom=row,
        isotope=isotope,
        radical_electrons=radical,
        current_hydrogens=None,
        source_hydrogens=None,
        source_namespace="test",
    )


def _fields(graph, key):
    position = graph.target_spec["keys"].index(key)
    return graph.target_spec["atoms"][position]["fields"]


def _bond_record(graph, edge):
    position = graph.target_spec["edges"].index(tuple(sorted(edge)))
    return graph.target_spec["bonds"][position]


def test_projection_unifies_active_and_both_hidden_bond_classes():
    molecule = _molecule(
        (30, 7, 8, 9),
        ((0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)),
    )
    molecule.hide_bonds(
        molecule.bonds[0],
        molecule.bonds[2],
        clear_conformers=False,
    )

    graph, issues = _gate6_projection(molecule)

    assert issues == ()
    assert len(molecule._hided_metal_bonds) == 1
    assert len(molecule._hided_covalent_bonds) == 1
    assert graph.target_spec["edges"] == ((0, 1), (1, 2), (2, 3))
    assert len(graph.target_spec["bonds"]) == 3


def test_virtual_hydrogen_has_protium_label_and_nonchemical_row_pin():
    molecule = _molecule((6, 1), ((0, 1, 1.0),), implicit_hydrogens=(1, 0))
    set_stereo_payload(
        molecule,
        StereoPayload(1, (), (_atom_fact(1, isotope=1),), ()),
    )

    graph, issues = _gate6_projection(molecule)

    assert issues == ()
    explicit = _fields(graph, 1)
    virtual = _fields(graph, 2)
    chemical_fields = (
        "atomic_number",
        "isotope",
        "formal_charge",
        "radical_electrons",
        "is_aromatic",
    )
    assert {
        key: explicit[key]
        for key in chemical_fields
    } == {
        key: virtual[key]
        for key in chemical_fields
    }
    assert explicit["_hotpot_stereo_row"] == 1
    assert virtual["_hotpot_stereo_row"] == 2
    assert graph.virtual_hydrogens == ((2,), ())


def test_query_can_pin_an_explicit_hydrogen_to_a_virtual_hydrogen():
    molecule = _molecule((6, 1), ((0, 1, 1.0),), implicit_hydrogens=(1, 0))
    graph, issues = _gate6_projection(molecule)
    assert issues == ()

    mappings = tuple(
        _NativeTarget(_pack_gate6_target(graph)).iter_mappings(
            _gate6_query_spec(graph, {0: 0, 1: 2})
        )
    )

    assert mappings == ({0: 0, 1: 2, 2: 1},)


def test_virtual_edges_are_lexicographically_sorted_with_real_edges():
    molecule = _molecule(
        (6, 6, 6),
        ((1, 2, 1.0),),
        implicit_hydrogens=(1, 0, 0),
    )

    graph, issues = _gate6_projection(molecule)

    assert issues == ()
    assert graph.target_spec["edges"] == ((0, 3), (1, 2))
    NativeTarget(_pack_gate6_target(graph))


def test_projection_uses_exact_atom_fact_and_molecule_scalar_labels():
    # A complete aromatic cycle supplies the label; an isolated aromatic Atom
    # is now an explicitly conflicting representation under Stage5 reads.
    molecule = _molecule((6,) * 6, tuple((row, (row + 1) % 6, 1.) for row in range(6)))
    molecule = supplied_aromatic_flags(molecule, (True,) * 6)
    molecule.atoms[0].formal_charge = -1
    set_stereo_payload(
        molecule,
        StereoPayload(1, (), (_atom_fact(0, isotope=13, radical=1),), ()),
    )

    graph, issues = _gate6_projection(molecule)

    assert issues == ()
    assert _fields(graph, 0) == {
        "atomic_number": 6,
        "isotope": 13,
        "formal_charge": -1,
        "radical_electrons": 1,
        "is_aromatic": True,
        "_hotpot_stereo_row": 0,
    }


def test_kekule_aromatic_cycle_normalizes_but_aromatic_biaryl_bridge_does_not():
    first_ring = tuple(
        (row, (row + 1) % 6, 1.0 if row % 2 == 0 else 2.0)
        for row in range(6)
    )
    second_ring = tuple(
        (row, 6 + ((row - 5) % 6), 1.0 if row % 2 == 0 else 2.0)
        for row in range(6, 12)
    )
    molecule = _molecule(
        (6,) * 12,
        first_ring + second_ring + ((0, 6, 1.0),),
    )
    molecule = supplied_aromatic_flags(molecule, (True,) * len(molecule.atoms))

    graph, issues = _gate6_projection(molecule)

    assert issues == ()
    for edge in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)):
        record = _bond_record(graph, edge)
        assert record["bond_kind"] == "aromatic"
        assert record["fields"] == {
            "bond_order": 1.5,
            "is_aromatic": True,
        }
    bridge = _bond_record(graph, (0, 6))
    assert bridge["bond_kind"] == "single"
    assert bridge["fields"] == {
        "bond_order": 1.0,
        "is_aromatic": False,
    }


@pytest.mark.parametrize(
    ("direction", "expected"),
    (
        ("atom1_to_atom2", frozenset({(0, 1)})),
        ("atom2_to_atom1", frozenset({(1, 0)})),
    ),
)
def test_dative_direction_is_preserved_separately(direction, expected):
    molecule = _molecule((7, 29))
    molecule._add_bond(
        0,
        1,
        bond_order=1.0,
        bond_kind="dative",
        bond_direction=direction,
    )

    graph, issues = _gate6_projection(molecule)

    assert issues == ()
    assert graph.directed_dative_edges == expected


def test_unknown_dative_direction_is_deterministically_unsupported():
    molecule = _molecule((7, 29))
    molecule._add_bond(0, 1, bond_kind="dative", bond_direction="sideways")

    graph, issues = _gate6_projection(molecule)

    assert graph is None
    assert len(issues) == 1
    assert issues[0].code is StereoIssueCode.UNSUPPORTED
    assert issues[0].atom_indices == (0, 1)
    assert "unknown direction" in issues[0].message


@pytest.mark.parametrize("malformation", ("parallel", "self_loop", "nonfinite"))
def test_malformed_represented_edges_are_unsupported(malformation):
    molecule = _molecule((6, 6), ((0, 1, 1.0),))
    if malformation == "parallel":
        molecule._hided_covalent_bonds.append(
            Bond(molecule.atoms[0], molecule.atoms[1], bond_order=2.0)
        )
    elif malformation == "self_loop":
        malformed = Bond(molecule.atoms[0], molecule.atoms[1], bond_order=1.0)
        malformed._atoms = (molecule.atoms[0], molecule.atoms[0])
        molecule._hided_covalent_bonds.append(malformed)
    else:
        # Inject malformed transport storage directly. Controlled setters now
        # restore failed eager chemistry rather than leaving a partial NaN row.
        molecule.bonds[0]._attrs[0] = float("nan")

    graph, issues = _gate6_projection(molecule)

    assert graph is None
    assert len(issues) == 1
    assert issues[0].code is StereoIssueCode.UNSUPPORTED
    expected = {
        "parallel": "parallel",
        "self_loop": "self-loop",
        "nonfinite": "non-finite",
    }
    assert expected[malformation] in issues[0].message


@pytest.mark.parametrize("fact_rows", ((0, 0), (2,)))
def test_duplicate_and_out_of_range_atom_facts_are_unsupported(fact_rows):
    molecule = _molecule((6,))
    set_stereo_payload(
        molecule,
        StereoPayload(1, (), tuple(_atom_fact(row) for row in fact_rows), ()),
    )

    graph, issues = _gate6_projection(molecule)

    assert graph is None
    assert len(issues) == 1
    assert issues[0].code is StereoIssueCode.UNSUPPORTED


def test_partly_explicit_reader_input_keeps_all_remaining_implicit_hydrogens():
    molecule = next(MolReader("[H]C", "smi"))
    carbon = next(atom for atom in molecule.atoms if atom.atomic_number == 6)
    carbon_row = carbon.idx

    graph, issues = _gate6_projection(molecule)

    assert issues == ()
    assert carbon.implicit_hydrogens == 3
    assert carbon.explicit_hydrogens == 1
    assert len(graph.target_spec["keys"]) == 5
    assert len(graph.virtual_hydrogens[carbon_row]) == 3


def test_fully_explicit_reader_input_creates_no_virtual_hydrogens():
    molecule = next(MolReader("[H]C([H])([H])[H]", "smi"))
    carbon = next(atom for atom in molecule.atoms if atom.atomic_number == 6)
    carbon_row = carbon.idx

    graph, issues = _gate6_projection(molecule)

    assert issues == ()
    assert carbon.implicit_hydrogens == 0
    assert carbon.explicit_hydrogens == 4
    assert len(graph.target_spec["keys"]) == 5
    assert graph.virtual_hydrogens[carbon_row] == ()


def test_negative_implicit_hydrogen_count_is_unsupported():
    molecule = _molecule((6,), implicit_hydrogens=(-1,))

    graph, issues = _gate6_projection(molecule)

    assert graph is None
    assert len(issues) == 1
    assert issues[0].code is StereoIssueCode.UNSUPPORTED
    assert issues[0].atom_indices == (0,)
    assert "negative implicit-hydrogen" in issues[0].message


def test_query_copies_exact_labels_and_pins_rows_for_real_native_target():
    molecule = _molecule((6, 6), ((0, 1, 1.0),))
    graph, issues = _gate6_projection(molecule)
    assert issues == ()
    query = _gate6_query_spec(graph, {0: 1})

    NativeTarget(_pack_gate6_target(graph))
    mappings = tuple(
        _NativeTarget(_pack_gate6_target(graph)).iter_mappings(query)
    )

    assert mappings == ({0: 1, 1: 0},)
    assert (
        "native_set",
        "_hotpot_stereo_row",
        (1,),
    ) in query["nodes"][0]["leaves"]
    assert all(
        field in {
            leaf[1]
            for leaf in query["nodes"][0]["leaves"]
        }
        for field in (
            "atomic_number",
            "isotope",
            "formal_charge",
            "radical_electrons",
            "is_aromatic",
        )
    )


def test_projection_is_deterministic_and_does_not_modify_the_molecule():
    molecule = _molecule(
        (6, 8),
        ((0, 1, 1.0),),
        implicit_hydrogens=(2, 0),
    )
    payload_before = get_stereo_payload(molecule)
    atoms_before = tuple(atom.attr_dict for atom in molecule.atoms)
    bonds_before = tuple(copy(bond.attr_dict) for bond in molecule.bonds)

    first = _gate6_projection(molecule)
    second = _gate6_projection(molecule)

    assert first == second
    assert get_stereo_payload(molecule) == payload_before
    assert tuple(atom.attr_dict for atom in molecule.atoms) == atoms_before
    assert tuple(bond.attr_dict for bond in molecule.bonds) == bonds_before
