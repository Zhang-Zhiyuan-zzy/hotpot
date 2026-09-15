"""Coordination-aware integration fences for MCA site detection."""

import pytest

from mca.site_detection import find_nucleophilic_sites
from tests.smarts_conformance.coordination_cases import (
    build_graph,
    ethylenediamine_chelate,
    metal_star,
    tertiary_amine,
)


def _site_records(molecule):
    return tuple(
        (site.atom_index, molecule.atoms[site.atom_index].symbol, site.site_type)
        for site in find_nucleophilic_sites(molecule)
    )


def test_free_amine_is_a_site_but_metal_bound_donors_are_not():
    free = tertiary_amine()
    zero_order_bound = tertiary_amine(0.0)
    single_bound = tertiary_amine(1.0)

    assert _site_records(free) == ((0, "N", "Amine"),)
    assert find_nucleophilic_sites(zero_order_bound) == ()
    assert find_nucleophilic_sites(single_bound) == ()


def test_chelate_donors_are_outside_the_reliable_mca_site_domain():
    molecule = ethylenediamine_chelate()
    sites = find_nucleophilic_sites(molecule)
    nitrogen_sites = tuple(
        site for site in sites if molecule.atoms[site.atom_index].symbol == "N"
    )

    assert nitrogen_sites == ()
    assert molecule.atoms[1].implicit_hydrogens == 1
    assert len(molecule.atoms[1].neighbours) + molecule.atoms[1].implicit_hydrogens == 3


@pytest.mark.parametrize(
    "molecule_factory",
    (
        pytest.param(lambda: tertiary_amine(0.0), id="zero-order-bound-amine"),
        pytest.param(lambda: tertiary_amine(1.0), id="single-bound-amine"),
        pytest.param(ethylenediamine_chelate, id="chelate"),
        pytest.param(lambda: metal_star(6), id="six-connected-metal"),
    ),
)
def test_metal_centres_are_never_mca_sites(molecule_factory):
    molecule = molecule_factory()

    assert all(
        not molecule.atoms[site.atom_index].is_metal
        for site in find_nucleophilic_sites(molecule)
    )


def test_remote_site_indices_refer_to_the_original_hotpot_atoms():
    molecule = build_graph(
        [(29, 0), (7, 0), (6, 2), (16, 0)],
        [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)],
    )
    original_atoms = tuple(molecule.atoms)

    sites = find_nucleophilic_sites(molecule)
    assert tuple(molecule.atoms) == original_atoms
    assert find_nucleophilic_sites(molecule) == sites
    assert tuple(atom.idx for atom in molecule.atoms) == tuple(range(len(molecule.atoms)))
    assert tuple(molecule.atoms[site.atom_index] for site in sites) == (
        original_atoms[3],
    )
    assert tuple((site.atom_index, site.site_type) for site in sites) == (
        (3, "atom_with_lone_pair"),
    )


def test_site_detection_preserves_existing_atom_mca_values():
    molecule = ethylenediamine_chelate()
    expected = tuple(410.0 + atom.idx for atom in molecule.atoms)
    # Atom.mca is read-only; this is the same internal state populated by the
    # public calculator after inference, without loading a model in this test.
    for atom, value in zip(molecule.atoms, expected):
        object.__setattr__(atom, "_mca", value)

    find_nucleophilic_sites(molecule)

    assert tuple(atom.mca for atom in molecule.atoms) == expected
