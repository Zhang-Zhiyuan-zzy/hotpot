"""Single-source Molecule topology-index contracts."""

from __future__ import annotations

import pytest

from hotpot.cheminfo.core import Atom, Bond, Molecule


def _molecule(
    atom_count: int,
    edges: tuple[tuple[int, int], ...] = (),
) -> Molecule:
    molecule = Molecule()
    for _ in range(atom_count):
        molecule._create_atom(atomic_number=6)
    for first, second in edges:
        molecule._add_bond(first, second, bond_order=1.0)
    molecule._update_topology(clear_conformers=False)
    return molecule


def _index_signature(molecule: Molecule):
    return tuple(
        (
            atom.idx,
            tuple(
                (
                    bond.atom2 if bond.atom1 is atom else bond.atom1
                ).idx
                for bond in atom.bonds
            ),
        )
        for atom in molecule.atoms
    )


def test_direct_atom_construction_installs_an_isolated_o1_record():
    molecule = Molecule()
    atom = Atom(molecule, atomic_number=6)

    assert atom.idx == 0
    assert atom.connectivity == 0
    assert atom.bonds == []
    assert atom.neighbours == []
    assert molecule._topology_index == {id(atom): (0, ())}
    assert "_neighbours" not in atom.__dict__
    assert "_bonds" not in atom.__dict__


def test_rebuild_is_deterministic_and_independent_of_bond_insertion_order():
    edges = ((2, 3), (0, 2), (1, 2), (0, 1))
    first = _molecule(4, edges)
    second = _molecule(4, tuple(reversed(edges)))

    assert _index_signature(first) == _index_signature(second) == (
        (0, (1, 2)),
        (1, (0, 2)),
        (2, (0, 1, 3)),
        (3, (2,)),
    )

    first._topology_index = {}
    first._update_topology(clear_conformers=False)
    assert _index_signature(first) == _index_signature(second)


def test_atom_idx_does_not_scan_molecule_membership():
    molecule = _molecule(3, ((0, 1),))

    class NoIndexList(list):
        def index(self, *_args, **_kwargs):
            raise AssertionError("Atom.idx scanned Molecule._atoms")

    molecule._atoms = NoIndexList(molecule._atoms)

    assert molecule.atoms[2].idx == 2


def test_add_bonds_commits_topology_once(monkeypatch):
    molecule = _molecule(4)
    original = molecule._update_topology
    calls = []

    def record(clear_conformers=True):
        calls.append(clear_conformers)
        return original(clear_conformers)

    monkeypatch.setattr(molecule, "_update_topology", record)
    bonds = molecule.add_bonds(((0, 3, 1.0), (0, 2, 1.0), (0, 1, 1.0)))

    assert len(bonds) == 3
    assert calls == [True]
    assert [atom.idx for atom in molecule.atoms[0].neighbours] == [1, 2, 3]


def test_bond_lookup_uses_authoritative_endpoints_and_incident_ranges():
    molecule = _molecule(4, ((0, 1), (0, 2), (2, 3)))

    assert molecule.bond(2, 0) is molecule.bonds[1]
    with pytest.raises(KeyError):
        molecule.bond(1, 3)


def test_rebuild_rejects_duplicate_atoms_edges_and_hidden_membership():
    duplicate_atom = _molecule(1)
    duplicate_atom._atoms.append(duplicate_atom.atoms[0])
    with pytest.raises(ValueError, match="same Atom twice"):
        duplicate_atom._update_topology(clear_conformers=False)

    duplicate_edge = _molecule(2, ((0, 1),))
    duplicate_edge._bonds.append(
        Bond(duplicate_edge.atoms[1], duplicate_edge.atoms[0], bond_order=2.0)
    )
    with pytest.raises(ValueError, match="parallel active Bonds"):
        duplicate_edge._update_topology(clear_conformers=False)

    active_and_hidden = _molecule(2, ((0, 1),))
    active_and_hidden._hided_covalent_bonds.append(active_and_hidden.bonds[0])
    with pytest.raises(ValueError, match="both active and hidden"):
        active_and_hidden._update_topology(clear_conformers=False)


def test_remove_atom_reindexes_remaining_membership_and_incidence():
    molecule = _molecule(4, ((0, 1), (1, 2), (2, 3)))
    molecule.remove_atom(molecule.atoms[1])

    assert _index_signature(molecule) == (
        (0, ()),
        (1, (2,)),
        (2, (1,)),
    )
