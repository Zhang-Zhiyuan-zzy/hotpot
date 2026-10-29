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
