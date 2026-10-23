from __future__ import annotations

import pickle

import numpy as np
import pytest

from hotpot.cheminfo.conformers import (
    ConformerBatch,
    ConformerSamplingError,
    ConformerSamplingRequest,
)
from hotpot.cheminfo.core import BondKind, Molecule

from .validation_providers import (
    FrozenCoordinateReplayAdapter,
    component_sha256,
    coordinate_sha256,
    freeze_rdkit_proposals,
    frozen_proposal_manifest,
    _worker_components,
)


class _DeterministicSampler:
    def sample(self, molecule, request):
        offset = float(request.seed or 0)
        rows = np.arange(len(molecule.atoms) * 3, dtype=float).reshape((-1, 3))
        return ConformerBatch((rows + offset)[None, :, :], (7,))


def _complex() -> Molecule:
    molecule = Molecule()
    molecule.create_atom(atomic_number=7, id=8)
    molecule.create_atom(atomic_number=6, id=9)
    molecule.create_atom(atomic_number=95, formal_charge=3, id=10)
    molecule.add_bond(0, 1, 1.0)
    molecule.add_bond(
        0,
        2,
        1.0,
        bond_kind=BondKind.DATIVE,
        bond_direction="atom1_to_atom2",
    )
    return molecule


def test_freeze_and_replay_use_exact_worker_component_rows() -> None:
    molecule = _complex()
    before = molecule.coordinates.copy()
    adapter = freeze_rdkit_proposals(
        molecule,
        seed=31,
        add_hydrogens=False,
        sampler=_DeterministicSampler(),
    )
    (component,) = _worker_components(
        molecule,
        add_hydrogens=False,
        seed=31,
    )

    batch = adapter.sample(
        component,
        ConformerSamplingRequest(conformer_count=1, seed=999),
    )

    assert batch.ordinals == (7,)
    assert batch.coordinates.shape == (1, len(component.atoms), 3)
    assert coordinate_sha256(batch.coordinates[0]) == (
        adapter.proposals[0].coordinate_sha256
    )
    assert component_sha256(component) == adapter.proposals[0].component_sha256
    np.testing.assert_array_equal(molecule.coordinates, before)


def test_replay_adapter_is_pickle_safe() -> None:
    adapter = freeze_rdkit_proposals(
        _complex(),
        seed=5,
        add_hydrogens=False,
        sampler=_DeterministicSampler(),
    )

    restored = pickle.loads(pickle.dumps(adapter))

    assert restored == adapter
    assert frozen_proposal_manifest(restored) == frozen_proposal_manifest(adapter)


def test_replay_rejects_a_different_component() -> None:
    adapter = freeze_rdkit_proposals(
        _complex(),
        seed=5,
        add_hydrogens=False,
        sampler=_DeterministicSampler(),
    )
    different = Molecule()
    different.create_atom(atomic_number=8, id=0)

    with pytest.raises(ConformerSamplingError, match="found 0"):
        adapter.sample(different, ConformerSamplingRequest(seed=5))


def test_replay_rejects_tampered_frozen_bytes() -> None:
    adapter = freeze_rdkit_proposals(
        _complex(),
        seed=5,
        add_hydrogens=False,
        sampler=_DeterministicSampler(),
    )
    proposal = adapter.proposals[0]
    tampered = proposal.__class__(
        component_sha256=proposal.component_sha256,
        coordinate_sha256=proposal.coordinate_sha256,
        coordinates=((999.0, 999.0, 999.0),) + proposal.coordinates[1:],
        ordinal=proposal.ordinal,
    )
    replay = FrozenCoordinateReplayAdapter((tampered,))
    (component,) = _worker_components(
        _complex(),
        add_hydrogens=False,
        seed=5,
    )

    with pytest.raises(ConformerSamplingError, match="bytes changed"):
        replay.sample(component, ConformerSamplingRequest(seed=5))
