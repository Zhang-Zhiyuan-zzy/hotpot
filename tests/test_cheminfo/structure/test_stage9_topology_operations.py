"""Whole named topology operations preserve chemical facts or rollback."""

from threading import Barrier, Thread

import pytest

from hotpot import Molecule
from hotpot.cheminfo.operations import chemistry, topology
from tests.test_cheminfo._stage5_fixtures import immutable_snapshot


def _molecule(number, charge):
    mol = Molecule()
    atom = mol.create_atom(atomic_number=number)
    mol.charge = charge
    return mol, atom


def _state(mol):
    record, metadata = mol._capture_factual_record()
    return immutable_snapshot(record), immutable_snapshot(metadata), tuple(mol.atoms), tuple(mol.bonds)


def test_link_with_copies_the_other_component_and_combines_declared_charge():
    first, carbon = _molecule(6, 1)
    second, oxygen = _molecule(8, -1)
    source = _state(second)

    assert carbon.link_with(oxygen) is None

    assert [atom.atomic_number for atom in first.atoms] == [6, 8]
    assert first.atoms[0] is carbon
    assert first.atoms[1] is not oxygen
    assert first.atoms[1].mol is first
    assert first.charge == 0
    assert len(first.bonds) == 1
    assert first.bonds[0].atoms == (carbon, first.atoms[1])
    assert first.bonds[0].bond_order == 1
    assert _state(second) == source


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_late_link_error_restores_the_destination_and_preserves_source(monkeypatch, error_type):
    first, carbon = _molecule(6, 1)
    second, oxygen = _molecule(8, -1)
    before, source = _state(first), _state(second)
    error = error_type("late bond publication")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(Molecule, "add_bond", fail)
    with pytest.raises(error_type) as caught:
        carbon.link_with(oxygen)
    assert caught.value is error
    assert _state(first) == before
    assert _state(second) == source


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_link_perception_and_order_assignment_publish_atomically(monkeypatch, error_type):
    mol, _ = _molecule(6, 0)
    mol.create_atom(atomic_number=8)
    before = _state(mol)
    error = error_type("late order assignment")

    def perceive(destination):
        destination.add_bond(0, 1)

    def fail(destination):
        raise error

    monkeypatch.setattr(chemistry, "_link_atoms", perceive)
    monkeypatch.setattr(Molecule, "assign_bond_order", fail)
    with pytest.raises(error_type) as caught:
        mol.link_atoms()
    assert caught.value is error
    assert _state(mol) == before


def test_reciprocal_component_links_do_not_hold_both_molecular_locks(monkeypatch):
    first, carbon = _molecule(6, 1)
    second, oxygen = _molecule(8, -1)
    ready = Barrier(2, timeout=5)
    capture = topology._capture_component
    errors = []

    def synchronized_capture(mol):
        result = capture(mol)
        ready.wait()
        return result

    def link(atom, other):
        try:
            atom.link_with(other)
        except BaseException as error:
            errors.append(error)

    monkeypatch.setattr(topology, "_capture_component", synchronized_capture)
    workers = [Thread(target=link, args=args, daemon=True)
               for args in ((carbon, oxygen), (oxygen, carbon))]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=8)
    assert not any(worker.is_alive() for worker in workers)
    assert not errors
    assert [atom.atomic_number for atom in first.atoms] == [6, 8]
    assert [atom.atomic_number for atom in second.atoms] == [8, 6]
    assert len(first.bonds) == len(second.bonds) == 1
    assert first.charge == second.charge == 0
