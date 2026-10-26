import numpy as np
import pytest

from hotpot import read_mol
from hotpot.cheminfo import stereo
from hotpot.cheminfo.forcefields import working_copy


def _payload(descriptor):
    hydrogen_state = stereo.HydrogenTransportState(1, 0, False)
    return stereo.StereoPayload(
        schema_version=1,
        records=(
            stereo.StereoRecord(
                anchor=stereo.StereoAnchor(
                    stereo.StereoAnchorKind.ATOM,
                    (0,),
                ),
                carriers=(
                    stereo.StereoCarrier(stereo.StereoCarrierKind.ATOM, 1),
                ),
                descriptor=descriptor,
                group=None,
                source_namespace="working-copy-test",
                source_code=descriptor,
            ),
        ),
        atom_facts=(
            stereo.AtomTransportRecord(
                atom=0,
                isotope=None,
                radical_electrons=None,
                current_hydrogens=hydrogen_state,
                source_hydrogens=hydrogen_state,
                source_namespace="working-copy-test",
            ),
        ),
        bond_facts=(),
    )


def _topology_index_state(molecule):
    return tuple(
        (
            row,
            tuple(id(bond) for bond in incident_bonds),
        )
        for atom in molecule._atoms
        for row, incident_bonds in (molecule._topology_index[id(atom)],)
    )


def test_successful_commit_transfers_the_exact_working_payload_reference():
    molecule = read_mol("CO")
    working = molecule.copy()
    original_payload = _payload("original")
    working_payload = _payload("working")
    stereo.set_stereo_payload(molecule, original_payload)
    stereo.set_stereo_payload(working, working_payload)
    properties = molecule.properties
    properties["stereo_payload"] = "user-owned-decoy"
    working.coordinates = working.coordinates + 1.0
    expected_coordinates = working.coordinates.copy()

    working_copy._commit_working_copy(molecule, working)

    assert stereo.get_stereo_payload(molecule) is working_payload
    assert stereo.get_stereo_payload(working) is working_payload
    assert original_payload.records[0].descriptor == "original"
    assert molecule.properties is properties
    assert molecule.properties == {"stereo_payload": "user-owned-decoy"}
    np.testing.assert_array_equal(molecule.coordinates, expected_coordinates)


def test_failed_commit_restores_exact_payload_reference_with_structure(monkeypatch):
    molecule = read_mol("CO")
    working = molecule.copy()
    original_payload = _payload("original")
    working_payload = _payload("working")
    stereo.set_stereo_payload(molecule, original_payload)
    stereo.set_stereo_payload(working, working_payload)
    properties = molecule.properties
    properties["stereo_payload"] = "user-owned-decoy"
    working.coordinates = working.coordinates + 1.0

    original_atoms = tuple(molecule.atoms)
    original_bonds = tuple(molecule.bonds)
    original_atom_attrs = tuple(atom.attrs for atom in molecule.atoms)
    original_topology_index = _topology_index_state(molecule)
    original_coordinates = molecule.coordinates.copy()
    failure_observed_working_payload = False

    def fail_after_stereo_assignment(atom_pairs):
        nonlocal failure_observed_working_payload
        failure_observed_working_payload = (
            stereo.get_stereo_payload(molecule) is working_payload
        )
        raise RuntimeError("injected failure after stereo assignment")

    monkeypatch.setattr(
        type(molecule.atom_pairs),
        "update_pairs",
        fail_after_stereo_assignment,
    )

    with pytest.raises(RuntimeError, match="failure after stereo assignment"):
        working_copy._commit_working_copy(molecule, working)

    assert failure_observed_working_payload
    assert stereo.get_stereo_payload(molecule) is original_payload
    assert tuple(molecule.atoms) == original_atoms
    assert tuple(molecule.bonds) == original_bonds
    assert all(
        atom.attrs is attrs
        for atom, attrs in zip(molecule.atoms, original_atom_attrs)
    )
    assert _topology_index_state(molecule) == original_topology_index
    assert molecule.properties is properties
    assert molecule.properties == {"stereo_payload": "user-owned-decoy"}
    np.testing.assert_array_equal(molecule.coordinates, original_coordinates)
