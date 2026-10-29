"""Owner bindings and requested dependency checks for staged source snapshots."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from hotpot import Molecule
from hotpot.cheminfo.stereo import AtomTransportRecord, HydrogenTransportState, get_stereo_payload
from hotpot.cheminfo.structure import _identity
from hotpot.cheminfo.structure._identity import _capture_source, _check_source


def _molecule():
    mol = Molecule()
    for row, number in enumerate((6, 7, 8, 1)):
        mol.create_atom(atomic_number=number, coordinates=(row, row / 2, 0))
    for first, second in ((0, 1), (1, 2), (2, 3)):
        mol.add_bond(first, second)
    return mol


def test_owner_only_binding_never_reads_or_hashes_molecular_contents(monkeypatch):
    mol = _molecule()

    def forbidden(*args):
        raise AssertionError("owner-only validation must not inspect molecular facts")

    monkeypatch.setattr(_identity, "_digest", forbidden)
    monkeypatch.setattr(_identity, "_complete_bonds", forbidden)
    source = _capture_source(mol)
    assert source.owner is mol._owner_token
    assert source.dependencies == ()
    mol.atoms[0].coordinates = (8, 9, 10)
    _check_source(mol, source)
    with pytest.raises(FrozenInstanceError):
        source.dependencies = (("geometry", "invented"),)


def test_foreign_owner_is_rejected_even_when_uids_and_content_match():
    mol = _molecule()
    other = _molecule()
    assert [atom._uid for atom in mol.atoms] == [atom._uid for atom in other.atoms]
    source = _capture_source(mol, ("topology", "geometry", "representation"))
    assert source.dependencies == _capture_source(other, ("topology", "geometry", "representation")).dependencies
    with pytest.raises(RuntimeError, match="owner"):
        _check_source(other, source)


def test_owner_validation_uses_object_identity_instead_of_equality():
    class EqualToken:
        def __eq__(self, other):
            return True

    mol = _molecule()
    other = _molecule()
    mol._owner_token = EqualToken()
    other._owner_token = EqualToken()
    assert mol._owner_token == other._owner_token
    with pytest.raises(RuntimeError, match="owner"):
        _check_source(other, _capture_source(mol))


def test_capture_hashes_only_requested_unique_domains(monkeypatch):
    mol = _molecule()
    seen = []
    digest = _identity._digest

    def count(domain, value):
        seen.append(domain)
        return digest(domain, value)

    monkeypatch.setattr(_identity, "_digest", count)
    source = _capture_source(mol, ("geometry", "cell", "geometry"))
    assert seen == ["geometry", "cell"]
    _check_source(mol, source)
    assert seen == ["geometry", "cell", "geometry", "cell"]


@pytest.mark.parametrize("domain", ["unknown", "revision", "GeometryIdentity"])
def test_unsupported_dependency_domains_are_explicit_errors(domain):
    with pytest.raises(ValueError, match="Unknown source dependency"):
        _capture_source(_molecule(), (domain,))


def test_coordinates_invalidate_geometry_without_inventing_representation_changes():
    mol = _molecule()
    representation = _capture_source(mol, ("representation",))
    geometry = _capture_source(mol, ("geometry",))
    coordinates = mol.coordinates.copy()
    mol.coordinates = coordinates + (1, 0, 0)
    _check_source(mol, representation)
    with pytest.raises(RuntimeError, match="'geometry'.*recalculate"):
        _check_source(mol, geometry)
    mol.coordinates = coordinates
    _check_source(mol, geometry)


def test_representation_change_preserves_topology_geometry_and_ignores_source_labels():
    mol = _molecule()
    geometry = _capture_source(mol, ("geometry", "topology"))
    representation = _capture_source(mol, ("representation",))
    mol.atoms[0].id = 999
    mol.bonds[0].id = 1001
    mol.atoms[0].partial_charge = 0.125
    mol.properties["mca"] = "annotation decoy"
    _check_source(mol, representation)
    mol.atoms[0].formal_charge = 1
    _check_source(mol, geometry)
    with pytest.raises(RuntimeError, match="'representation'"):
        _check_source(mol, representation)


@pytest.mark.parametrize("change", [
    lambda mol: setattr(mol, "charge", 1),
    lambda mol: setattr(mol, "spin_mult", 3),
    lambda mol: setattr(mol.atoms[0], "isotope", 13),
])
def test_explicit_molecular_declarations_and_legacy_isotope_are_representation_inputs(change):
    mol = _molecule()
    source = _capture_source(mol, ("representation",))
    change(mol)
    with pytest.raises(RuntimeError, match="representation"):
        _check_source(mol, source)


def test_selected_legacy_frame_spin_is_checked_without_inferred_default_spin():
    mol = _molecule()
    source = _capture_source(mol, ("representation",))
    mol._conformers._spin_mult = np.array([3])
    assert mol.spin_mult == 3
    with pytest.raises(RuntimeError, match="representation"):
        _check_source(mol, source)
    current = _capture_source(mol, ("representation",))
    mol._conformers._spin_mult[0] = 5
    with pytest.raises(RuntimeError, match="representation"):
        _check_source(mol, current)


def test_explicit_spin_declaration_takes_precedence_over_legacy_frame_results():
    mol = _molecule()
    mol.spin_mult = 3
    source = _capture_source(mol, ("representation",))
    mol._conformers._spin_mult = np.array([5])
    _check_source(mol, source)


def test_invalid_legacy_frame_spin_selection_preserves_the_existing_read_error():
    mol = _molecule()
    mol.conformer_add(mol.coordinates)
    mol._conformers._spin_mult = np.array([3])
    mol._conformers_index = 1
    with pytest.raises(IndexError, match="number of conformers"):
        _capture_source(mol, ("representation",))


def test_lossless_stereo_transport_and_source_hydrogen_provenance_affect_representation():
    mol = _molecule()
    source = _capture_source(mol, ("representation",))
    payload = get_stereo_payload(mol)
    record = AtomTransportRecord(
        atom=0, isotope=13, radical_electrons=1,
        current_hydrogens=HydrogenTransportState(0, 0, False),
        source_hydrogens=HydrogenTransportState(None, 1, True),
        source_namespace="test-source",
    )
    mol._stereo_payload = replace(payload, atom_facts=(record,))
    with pytest.raises(RuntimeError, match="representation"):
        _check_source(mol, source)


def test_bond_source_metadata_is_order_independent_but_nested_content_is_checked():
    mol = _molecule()
    bond = mol.bonds[0]
    bond._bond_source_metadata = {"kind": "source", "flags": {"raw": [1, np.int32(2)]}}
    source = _capture_source(mol, ("representation",))
    bond._bond_source_metadata = {"flags": {"raw": [1, 2]}, "kind": "source"}
    _check_source(mol, source)
    bond._bond_source_metadata["flags"]["raw"][1] = 3
    with pytest.raises(RuntimeError, match="representation"):
        _check_source(mol, source)
    bond._bond_source_metadata = {"unknown": object()}
    with pytest.raises(TypeError, match="Unsupported source identity value: object"):
        _capture_source(mol, ("representation",))


def test_topology_includes_hidden_bonds_and_their_ordered_endpoints():
    mol = _molecule()
    bond = mol.bonds[-1]
    mol.hide_bonds(bond, clear_conformers=False)
    source = _capture_source(mol, ("topology",))
    bond._atoms = (bond.atom2, bond.atom1)
    with pytest.raises(RuntimeError, match="topology"):
        _check_source(mol, source)


def test_atom_or_bond_membership_change_invalidates_topology():
    mol = _molecule()
    source = _capture_source(mol, ("topology",))
    mol.remove_bond(mol.bonds[0])
    with pytest.raises(RuntimeError, match="topology"):
        _check_source(mol, source)


def test_constraint_identity_reads_existing_true_views_without_creating_caches():
    mol = _molecule()
    geometry = _capture_source(mol, ("geometry",))
    constraints = _capture_source(mol, ("constraints",))
    mol.update_angles()
    mol.update_torsions()
    _check_source(mol, constraints)
    mol._angles[0].constraint = True
    with pytest.raises(RuntimeError, match="constraints"):
        _check_source(mol, constraints)
    _check_source(mol, geometry)
    snapshot = tuple(mol._angles), tuple(mol._torsions)
    _capture_source(mol, ("constraints",))
    assert snapshot == (tuple(mol._angles), tuple(mol._torsions))


@pytest.mark.parametrize("change", [
    lambda mol: setattr(mol.atoms[0], "x_constraint", True),
    lambda mol: setattr(mol.bonds[0], "constraint", True),
])
def test_atom_and_bond_constraints_invalidate_only_the_constraint_domain(change):
    mol = _molecule()
    source = _capture_source(mol, ("constraints",))
    representation = _capture_source(mol, ("representation",))
    change(mol)
    with pytest.raises(RuntimeError, match="constraints"):
        _check_source(mol, source)
    _check_source(mol, representation)


def test_cell_identity_uses_only_six_cell_parameters():
    mol = _molecule()
    mol.create_crystal(10, 11, 12, 90, 91, 92)
    source = _capture_source(mol, ("cell",))
    representation = _capture_source(mol, ("representation",))
    mol._crystal._obcell = object()
    _check_source(mol, source)
    mol._crystal.a = 20
    with pytest.raises(RuntimeError, match="cell"):
        _check_source(mol, source)
    _check_source(mol, representation)


def test_source_frames_track_ordered_coordinates_not_current_selection_or_result_arrays():
    mol = _molecule()
    mol.conformer_add(mol.coordinates)
    source = _capture_source(mol, ("source_frames",))
    mol._conformers_index = 9
    mol._conformers._energy = np.array([-12.0])
    mol._conformers._force = np.ones((1, len(mol.atoms), 3))
    _check_source(mol, source)
    mol._conformers._coordinates[0, 0, 0] += 1
    with pytest.raises(RuntimeError, match="source_frames"):
        _check_source(mol, source)


def test_module_has_no_public_exports():
    assert _identity.__all__ == ()
