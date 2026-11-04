"""Isolated contract for the post-mainline aromatic/Kekulé rule binding."""

from dataclasses import FrozenInstanceError, replace
import inspect
import math

import pytest

from hotpot.cheminfo.operations._aromatic_protocol import _evaluate_aromatic_assignment
from hotpot.cheminfo.rules.chemistry.bonding import kekule as implementation
from hotpot.cheminfo.structure._source_identity import _SourceIdentity
from hotpot.cheminfo.structure.snapshots import _AromaticAssignment, _BondingSnapshot


def _ring(numbers, hydrogens, *, spatial=False):
    size = len(numbers)
    return _BondingSnapshot(
        _SourceIdentity(object(), (("bonding_inputs", "isolated-contract-fixture"),)),
        tuple(101 + 3 * row for row in range(size)),
        numbers,
        hydrogens,
        (False,) * size,
        tuple((201 + 5 * row, row, (row + 1) % size, 1.0, "single", False)
              for row in range(size)),
        tuple((math.cos(2 * math.pi * row / size),
               math.sin(2 * math.pi * row / size), 0.0) for row in range(size)),
        (tuple(range(size)),),
        (spatial,),
    )


def test_binding_is_the_existing_rule_without_wrapper_or_new_public_surface():
    from hotpot.cheminfo.operations import _aromatic_protocol

    assert _evaluate_aromatic_assignment is implementation._calculate_aromatic_assignment
    assert _aromatic_protocol.__all__ == ()
    parameters = inspect.signature(_evaluate_aromatic_assignment).parameters
    assert tuple(parameters) == ("snapshot", "kekule")
    assert parameters["kekule"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["kekule"].default is inspect.Parameter.empty


@pytest.mark.parametrize("spatial", (False, True))
@pytest.mark.parametrize("kekule", (False, True))
def test_benzene_has_complete_mapped_readonly_assignment(spatial, kekule):
    snapshot = _ring((6,) * 6, (1,) * 6, spatial=spatial)
    before = replace(snapshot)

    proposal = _evaluate_aromatic_assignment(snapshot, kekule=kekule)

    assert isinstance(proposal, _AromaticAssignment)
    assert snapshot == before
    assert proposal.source is snapshot.source
    assert proposal.atom_uids == snapshot.atom_uids
    assert proposal.bond_uids == tuple(bond[0] for bond in snapshot.bonds)
    assert proposal.covered_rows == tuple(range(6))
    assert proposal.atom_aromatic == (True,) * 6
    assert proposal.bond_aromatic == (True,) * 6
    assert proposal.method == ("kekule" if kekule else "aromaticity")
    if kekule:
        assert proposal.bond_orders.count(2.0) == 3
        assert all(proposal.bond_orders[row - 1] + proposal.bond_orders[row] == 3
                   for row in range(6))
        assert proposal.bond_kinds.count("double") == 3
    else:
        assert proposal.bond_orders == (1.0,) * 6
        assert proposal.bond_kinds == ("single",) * 6
    with pytest.raises(FrozenInstanceError):
        proposal.method = "replacement"


@pytest.mark.parametrize("kekule", (False, True))
@pytest.mark.parametrize("numbers,hydrogens", (
    ((8, 6, 6, 6, 6), (0, 1, 1, 1, 1)),  # furan: one O lone pair
    ((7, 6, 6, 6, 6, 6), (0, 1, 1, 1, 1, 1)),  # pyridine: one N electron
))
def test_named_heterocycle_fixtures_keep_complete_site_coverage(numbers, hydrogens, kekule):
    snapshot = _ring(numbers, hydrogens)
    proposal = _evaluate_aromatic_assignment(snapshot, kekule=kekule)

    assert proposal.atom_aromatic == (True,) * len(numbers)
    assert proposal.bond_aromatic == (True,) * len(numbers)
    assert proposal.atom_uids == snapshot.atom_uids
    assert proposal.bond_uids == tuple(bond[0] for bond in snapshot.bonds)
    assert proposal.covered_rows == tuple(range(len(numbers)))


@pytest.mark.parametrize("kekule", (False, True))
def test_four_pi_ring_has_a_defined_nonaromatic_assignment(kekule):
    snapshot = _ring((6,) * 4, (1,) * 4)
    proposal = _evaluate_aromatic_assignment(snapshot, kekule=kekule)

    assert proposal.atom_aromatic == (False,) * 4
    assert proposal.bond_aromatic == (False,) * 4
    assert proposal.bond_orders == (1.0,) * 4
    assert proposal.covered_rows == tuple(range(4))


@pytest.mark.parametrize("kekule", (False, True))
def test_missing_required_h_is_none_without_fabricated_partial_proposal(kekule):
    snapshot = _ring((6,) * 6, (None, 1, 1, 1, 1, 1))
    before = replace(snapshot)

    assert _evaluate_aromatic_assignment(snapshot, kekule=kekule) is None
    assert snapshot == before


def test_selected_topology_retains_nonring_zero_and_dative_bonds():
    base = _ring((6,) * 6, (1,) * 6)
    snapshot = replace(
        base,
        atom_uids=(*base.atom_uids, 601, 604, 607),
        atomic_numbers=(*base.atomic_numbers, 7, 26, 8),
        implicit_hydrogens=(*base.implicit_hydrogens, 0, 0, 0),
        aromatic=(*base.aromatic, False, False, False),
        coordinates=(*base.coordinates, (4., 0., 0.), (5., 0., 0.), (6., 0., 0.)),
        bonds=(*base.bonds, (701, 6, 7, 1., "dative", False),
               (706, 7, 8, 0., "zero", False)),
    )

    proposal = _evaluate_aromatic_assignment(snapshot, kekule=True)

    assert proposal.atom_uids == snapshot.atom_uids
    assert proposal.bond_uids == tuple(bond[0] for bond in snapshot.bonds)
    assert proposal.bond_orders[-2:] == (1., 0.)
    assert proposal.bond_kinds[-2:] == ("dative", "zero")
    assert proposal.bond_aromatic[-2:] == (False, False)
    assert proposal.atom_aromatic[-3:] == (False, False, False)


def test_rule_exception_is_preserved_without_fallback(monkeypatch):
    snapshot = _ring((6,) * 6, (1,) * 6)
    before = replace(snapshot)
    failure = RuntimeError("identified rule execution failure")

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(implementation, "_determine_ring_aromaticity", fail)
    with pytest.raises(RuntimeError) as caught:
        _evaluate_aromatic_assignment(snapshot, kekule=True)
    assert caught.value is failure
    assert snapshot == before
