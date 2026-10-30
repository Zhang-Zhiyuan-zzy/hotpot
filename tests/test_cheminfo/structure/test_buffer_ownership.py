"""Public projections and captured values cannot alter authoritative buffers."""

import numpy as np
import pytest

from hotpot import Molecule
from hotpot.cheminfo.structure._commit import _commit_atom_facts
from hotpot.cheminfo.structure._identity import _capture_source
from hotpot.cheminfo.structure._mutation import _normalize_atom_fields


def _molecule():
    mol = Molecule()
    mol.create_atom(atomic_number=6, coordinates=(1, 2, 3))
    mol.create_atom(atomic_number=8, coordinates=(4, 5, 6))
    mol.add_bond(0, 1)
    return mol


@pytest.mark.parametrize('project', [
    lambda mol: mol.atom_matrix,
    lambda mol: mol.coordinates,
    lambda mol: mol.atoms[0].attrs,
    lambda mol: mol.bonds[0].attrs,
])
def test_public_projection_is_an_independent_array(project):
    mol = _molecule()
    before = mol.atom_matrix.tobytes(), mol.bonds[0].attrs.tobytes()
    values = project(mol)
    values[:] = -99
    assert (mol.atom_matrix.tobytes(), mol.bonds[0].attrs.tobytes()) == before


def test_candidate_retains_only_immutable_input_bytes_and_installs_once():
    mol = _molecule()
    values = np.array([3., 7.])
    candidate = _normalize_atom_fields(_capture_source(mol), (0, 1),
        tuple(atom._uid for atom in mol.atoms), {'x': values}, producer=14)
    values[:] = 99
    _commit_atom_facts(mol, candidate)
    assert mol.coordinates[:, 0].tolist() == [3, 7]
    for result in candidate.values:
        with pytest.raises(ValueError):
            result.flags.writeable = True


def test_conformer_projection_does_not_expose_captured_numeric_owners():
    mol = _molecule()
    mol.conformer_add(mol.coordinates, energy=2)
    result = mol.conformers[0]
    result['coordinates'][:] = 99
    assert mol.conformers[0]['coordinates'].tolist() == [[1, 2, 3], [4, 5, 6]]


def test_stereo_capture_is_frozen_and_has_only_the_three_approved_fields():
    from dataclasses import FrozenInstanceError, fields
    from hotpot.cheminfo.operations.stereochemistry import _capture_stereo_owner
    from hotpot.cheminfo.stereo import get_stereo_payload

    mol = _molecule()
    snapshot, source, referents = _capture_stereo_owner(mol)
    assert tuple(field.name for field in fields(snapshot)) == ("payload", "atom_keys", "bonds")
    assert snapshot.payload is get_stereo_payload(mol)
    assert snapshot.atom_keys == tuple(id(atom) for atom in mol.atoms)
    assert all(type(key) is int for key in snapshot.atom_keys)
    assert all(type(row) is tuple and len(row) == 5 for row in snapshot.bonds)
    assert source.owner is mol._owner_token
    assert source.dependencies == ()
    assert len(referents) == 3
    with pytest.raises(FrozenInstanceError):
        snapshot.payload = None
    with pytest.raises(TypeError):
        snapshot.bonds[0][2] = 9.0


def test_stereo_graph_is_deeply_readonly_and_packed_values_are_independent():
    from hotpot.cheminfo.adapters.structure.stereochemistry import _pack_gate6_target
    from hotpot.cheminfo.operations.stereochemistry import _gate6_projection

    mol = _molecule()
    graph, issues = _gate6_projection(mol)
    assert issues == ()
    before = mol.atom_matrix.tobytes(), mol.bonds[0].attrs.tobytes()
    with pytest.raises(TypeError):
        graph.target_spec["keys"] = ()
    with pytest.raises(TypeError):
        graph.target_spec["atoms"][0]["fields"]["atomic_number"] = 99
    with pytest.raises(TypeError):
        graph.target_spec["atoms"][0]["full"]["degree"] = 99
    with pytest.raises(TypeError):
        graph.target_spec["bonds"][0]["fields"]["bond_order"] = 99.0
    packed = _pack_gate6_target(graph)
    packed["atoms"][0]["fields"]["atomic_number"] = 99
    packed["bonds"][0]["fields"]["bond_order"] = 99.0
    assert graph.target_spec["atoms"][0]["fields"]["atomic_number"] == 6
    assert graph.target_spec["bonds"][0]["fields"]["bond_order"] == 1.0
    assert (mol.atom_matrix.tobytes(), mol.bonds[0].attrs.tobytes()) == before
