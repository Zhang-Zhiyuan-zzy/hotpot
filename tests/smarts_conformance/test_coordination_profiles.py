"""Public SMARTS semantics profiles for coordination-aware graph matching."""

import pytest

import hotpot as hp
from hotpot.cheminfo.core import BondKind, Molecule
from hotpot.cheminfo.search import SmartsSemantics

from .coordination_cases import (
    COORDINATION_FIXTURES,
    ethylenediamine_chelate,
    metal_bound_ligand_ring,
    metal_star,
    tertiary_amine,
)


pytestmark = pytest.mark.smarts_core


def _matching_indices(molecule, smarts, semantics=None):
    if semantics is None:
        hits = molecule.search_substructure(smarts)
    else:
        hits = molecule.search_substructure(smarts, semantics=semantics)
    return {
        atom_index
        for hit in hits
        for atom_index in hit.atom_indices
    }


def _matching_atom_sets(molecule, smarts, semantics):
    return {
        frozenset(hit.atom_indices)
        for hit in molecule.search_substructure(smarts, semantics=semantics)
    }


def _molecule_graph_state(molecule):
    return (
        tuple(id(atom) for atom in molecule.atoms),
        tuple(
            (id(bond), bond.a1idx, bond.a2idx, bond.bond_order, bond.bond_kind)
            for bond in molecule.bonds
        ),
        frozenset(frozenset(edge) for edge in molecule.graph.edges),
        frozenset(
            frozenset(atom.idx for atom in ring.atoms)
            for ring in molecule.rings
        ),
    )


def _two_atom_metal_graph(bond_kind, bond_order, bond_direction=None):
    molecule = Molecule()
    molecule.create_atom(atomic_number=7, implicit_hydrogens=0)
    molecule.create_atom(atomic_number=29, implicit_hydrogens=0)
    molecule.add_bond(
        0,
        1,
        bond_order,
        bond_kind=bond_kind,
        bond_direction=bond_direction,
    )
    return molecule


def _bound_tertiary_amine(bond_kind, bond_order):
    molecule = Molecule()
    for atomic_number in (7, 6, 6, 6, 29):
        molecule.create_atom(atomic_number=atomic_number, implicit_hydrogens=0)
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
        bond_order,
        bond_kind=bond_kind,
        bond_direction=(
            "atom1_to_atom2" if bond_kind is BondKind.DATIVE else None
        ),
    )
    return molecule


def _dative_metal_star(connectivity):
    molecule = Molecule()
    molecule.create_atom(atomic_number=26, implicit_hydrogens=0)
    for ligand_index in range(1, connectivity + 1):
        molecule.create_atom(atomic_number=7, implicit_hydrogens=0)
        molecule.add_bond(
            0,
            ligand_index,
            1.0,
            bond_kind=BondKind.DATIVE,
            bond_direction="atom2_to_atom1",
        )
    return molecule


def test_smarts_semantics_is_public_and_full_graph_is_the_default():
    molecule = ethylenediamine_chelate()

    assert hp.SmartsSemantics is SmartsSemantics
    assert SmartsSemantics.FULL_GRAPH is not SmartsSemantics.LIGAND_SKELETON
    assert _matching_indices(molecule, "[M;r5]") == _matching_indices(
        molecule,
        "[M;r5]",
        SmartsSemantics.FULL_GRAPH,
    )


def test_string_semantics_name_equals_enum_and_invalid_name_is_rejected():
    molecule = ethylenediamine_chelate()

    assert _matching_indices(
        molecule,
        "[M;r5]",
        "ligand_skeleton",
    ) == _matching_indices(
        molecule,
        "[M;r5]",
        SmartsSemantics.LIGAND_SKELETON,
    )
    with pytest.raises(ValueError):
        molecule.search_substructure("[M;r5]", semantics="not_a_profile")
    with pytest.raises(ValueError):
        hp.Substructure.from_smarts("[M;r5]", semantics="not_a_profile")


def test_molecule_entry_point_applies_ring_profile_without_mutation():
    chelate = ethylenediamine_chelate()
    chelate_state = _molecule_graph_state(chelate)

    assert _matching_indices(chelate, "[M;r5]", SmartsSemantics.FULL_GRAPH) == {0}
    assert (
        _matching_indices(
            chelate,
            "[M;r5]",
            SmartsSemantics.LIGAND_SKELETON,
        )
        == set()
    )
    assert _molecule_graph_state(chelate) == chelate_state

    ligand_ring = metal_bound_ligand_ring()
    ligand_ring_state = _molecule_graph_state(ligand_ring)

    for semantics in SmartsSemantics:
        assert _matching_indices(ligand_ring, "[N;r6]", semantics) == {1}
        assert _matching_indices(ligand_ring, "[C;r6]", semantics) == set(range(2, 7))
    assert _molecule_graph_state(ligand_ring) == ligand_ring_state


def test_substructure_entry_point_applies_d_x_v_profile_to_bound_donor():
    molecule = tertiary_amine(1.0)
    full_query = hp.Substructure.from_smarts(
        "[N;D4;X4;v4]",
        semantics=SmartsSemantics.FULL_GRAPH,
    )
    ligand_query = hp.Substructure.from_smarts(
        "[N;D3;X3;v3]",
        semantics=SmartsSemantics.LIGAND_SKELETON,
    )

    assert {hit.atom_indices for hit in hp.Searcher(full_query).search(molecule)} == {
        frozenset((0,))
    }
    assert {hit.atom_indices for hit in hp.Searcher(ligand_query).search(molecule)} == {
        frozenset((0,))
    }
    assert (
        _matching_indices(
            molecule,
            "[N;D4;X4;v4]",
            SmartsSemantics.LIGAND_SKELETON,
        )
        == set()
    )
    assert (
        _matching_indices(
            molecule,
            "[N;D3;X3;v3]",
            SmartsSemantics.FULL_GRAPH,
        )
        == set()
    )


@pytest.mark.parametrize(
    ("bond_kind", "bond_order", "full_valence"),
    (
        (BondKind.UNKNOWN, 0.0, 3),
        (BondKind.DATIVE, 1.0, 4),
    ),
)
def test_bound_donor_valence_depends_on_profile_and_bond_representation(
    bond_kind, bond_order, full_valence
):
    molecule = _bound_tertiary_amine(bond_kind, bond_order)

    assert _matching_indices(
        molecule,
        f"[N;D4;X4;v{full_valence}]",
        SmartsSemantics.FULL_GRAPH,
    ) == {0}
    assert _matching_indices(
        molecule,
        "[N;D3;X3;v3]",
        SmartsSemantics.LIGAND_SKELETON,
    ) == {0}


@pytest.mark.parametrize(
    ("filename", "implicit_hydrogens", "connectivity", "valence"),
    (
        ("cu_trimethylamine_single.mol2", 0, 3, 3),
        ("cu_trimethylamine_single.sdf", 1, 4, 4),
    ),
)
def test_ligand_profile_preserves_source_implicit_hydrogen_perception(
    filename, implicit_hydrogens, connectivity, valence
):
    molecule = hp.read_mol(COORDINATION_FIXTURES / filename)
    donor = molecule.atoms[0]

    assert donor.implicit_hydrogens == implicit_hydrogens
    assert _matching_indices(
        molecule,
        f"[N;D4;X{connectivity + 1};v{valence + 1}]",
        SmartsSemantics.FULL_GRAPH,
    ) == {donor.idx}
    assert _matching_indices(
        molecule,
        f"[N;D3;X{connectivity};v{valence}]",
        SmartsSemantics.LIGAND_SKELETON,
    ) == {donor.idx}
    assert donor.implicit_hydrogens == implicit_hydrogens


@pytest.mark.parametrize("smarts", ("[M;D6]", "[M;X6]"))
def test_metal_connectivity_is_visible_in_both_profiles(smarts):
    molecule = metal_star(6)

    for semantics in SmartsSemantics:
        assert _matching_indices(molecule, smarts, semantics) == {0}


def test_dative_metal_star_keeps_connectivity_but_not_ligand_profile_valence():
    molecule = _dative_metal_star(6)

    assert _matching_indices(
        molecule,
        "[M;D6;X6;v6]",
        SmartsSemantics.FULL_GRAPH,
    ) == {0}
    assert _matching_indices(
        molecule,
        "[M;D6;X6;v0]",
        SmartsSemantics.LIGAND_SKELETON,
    ) == {0}
    assert _matching_indices(
        molecule,
        "[M;v6]",
        SmartsSemantics.LIGAND_SKELETON,
    ) == set()
    assert _matching_indices(
        molecule,
        "[M;v0]",
        SmartsSemantics.FULL_GRAPH,
    ) == set()


@pytest.mark.parametrize(
    ("bond_kind", "bond_order", "bond_direction", "matches_single"),
    (
        (BondKind.SINGLE, 1.0, None, True),
        (BondKind.DATIVE, 1.0, "atom1_to_atom2", False),
        (BondKind.UNKNOWN, 0.0, None, False),
        (BondKind.ZERO, 0.0, None, False),
    ),
)
def test_bond_kind_controls_explicit_and_implicit_single_bond_matching(
    bond_kind, bond_order, bond_direction, matches_single
):
    molecule = _two_atom_metal_graph(bond_kind, bond_order, bond_direction)
    expected = {frozenset((0, 1))} if matches_single else set()

    for semantics in SmartsSemantics:
        assert _matching_atom_sets(molecule, "[N]-[M]", semantics) == expected
        assert _matching_atom_sets(molecule, "[N][M]", semantics) == expected
        assert _matching_atom_sets(molecule, "[N]~[M]", semantics) == {
            frozenset((0, 1))
        }


def test_recursive_smarts_inherits_the_selected_semantics_profile():
    molecule = tertiary_amine(1.0)
    full_recursive = "[N;$([N;D4;X4;v4])]"
    ligand_recursive = "[N;$([N;D3;X3;v3])]"

    assert _matching_indices(
        molecule,
        full_recursive,
        SmartsSemantics.FULL_GRAPH,
    ) == {0}
    assert _matching_indices(
        molecule,
        full_recursive,
        SmartsSemantics.LIGAND_SKELETON,
    ) == set()
    assert _matching_indices(
        molecule,
        ligand_recursive,
        SmartsSemantics.FULL_GRAPH,
    ) == set()
    assert _matching_indices(
        molecule,
        ligand_recursive,
        SmartsSemantics.LIGAND_SKELETON,
    ) == {0}
