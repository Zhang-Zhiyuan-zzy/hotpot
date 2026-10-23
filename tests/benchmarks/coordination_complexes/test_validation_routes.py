from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hotpot.cheminfo import rdWrappers
from hotpot.cheminfo.conformers import ConformerBatch
from hotpot.cheminfo.core import Molecule

from .configuration import SMOKE_SETTINGS
from .validation_providers import FrozenCoordinateReplayAdapter
from .validation_routes import (
    PROVIDER_STAGE_SEQUENCE,
    WHOLE_COMPLEX_STAGE_SEQUENCE,
    execute_provider_route,
    execute_whole_complex_route,
)


class _Sampler:
    def sample(self, molecule, request):
        return ConformerBatch(
            np.full((1, len(molecule.atoms), 3), float(request.seed)),
            (3,),
        )


def _complex() -> Molecule:
    molecule = Molecule()
    molecule.create_atom(atomic_number=7, id=2)
    molecule.create_atom(atomic_number=6, id=3)
    molecule.create_atom(atomic_number=95, formal_charge=3, id=4)
    molecule.add_bond(0, 1, 1.0)
    molecule.add_bond(0, 2, 1.0)
    return molecule


@pytest.mark.parametrize("route", ("O0", "F1", "F2"))
def test_provider_routes_use_one_identical_complete_workflow(route: str) -> None:
    calls = []

    def workflow(molecule, forcefield, **options):
        calls.append((molecule, forcefield, options))
        return "report"

    execution = execute_provider_route(
        _complex(),
        route,
        SMOKE_SETTINGS,
        4,
        workflow=workflow,
        freeze_sampler=_Sampler(),
    )

    assert execution.report == "report"
    assert execution.stage_sequence == PROVIDER_STAGE_SEQUENCE
    assert len(calls) == 1
    assert calls[0][1] == "UFF"
    assert calls[0][2]["seed"] == SMOKE_SETTINGS.seed + 4
    assert calls[0][2]["epochs"] == SMOKE_SETTINGS.epochs
    sampler = calls[0][2]["sampler"]
    if route == "F2":
        assert isinstance(sampler, FrozenCoordinateReplayAdapter)
        assert len(execution.proposal_manifest) == 1
    else:
        assert not isinstance(sampler, FrozenCoordinateReplayAdapter)
        assert execution.proposal_manifest == ()


@pytest.mark.parametrize("route", ("C1", "C2"))
def test_whole_complex_routes_preserve_proposal_until_stage3(route: str) -> None:
    observed = {}

    def generate(molecule, *, options, constraints):
        assert all(
            bond.bond_kind.value == "dative"
            for bond in molecule.bonds
            if bond.is_metal_ligand_bond
        )
        observed["constraints"] = constraints
        coordinates = np.arange(
            len(molecule.atoms) * 3,
            dtype=float,
        ).reshape((1, len(molecule.atoms), 3))
        return rdWrappers.ConformerBatch(
            status=rdWrappers.GenerationStatus.SUCCESS,
            coordinates=coordinates,
            records=(rdWrappers.ConformerRecord(0, 0),),
            engine="fixture",
            engine_version="1",
            diagnostic=None,
        )

    def optimize(molecule, forcefield, **options):
        observed["entry_coordinates"] = molecule.coordinates.copy()
        observed["forcefield"] = forcefield
        observed["options"] = options
        return "optimized"

    execution = execute_whole_complex_route(
        _complex(),
        route,
        SMOKE_SETTINGS,
        9,
        generate=generate,
        optimize=optimize,
        radius=lambda atomic_number: 1.0,
        add_hydrogens=False,
    )

    assert execution.report == "optimized"
    assert execution.stage_sequence == WHOLE_COMPLEX_STAGE_SEQUENCE
    assert execution.proposal is not None
    assert execution.proposal.mapping_preserved
    assert observed["entry_coordinates"].shape == (3, 3)
    assert observed["forcefield"] == "UFF"
    assert observed["options"]["add_hydrogens"] is False
    assert observed["options"]["trajectory_start"].value == "complex_untangling"
    if route == "C1":
        assert observed["constraints"] is None
        assert execution.proposal.constraint_policy is None
    else:
        constraints = observed["constraints"]
        np.testing.assert_allclose(
            constraints.metal_donor_bounds,
            np.array([[1.4, 3.0]]),
        )
        assert execution.proposal.constraint_policy["name"] == (
            "hotpot-native-placement-v1"
        )


def test_whole_complex_route_does_not_commit_generated_dative_metadata() -> None:
    source = _complex()
    original_kind = source.c_bonds[0].bond_kind

    def generate(molecule, *, options, constraints):
        return rdWrappers.ConformerBatch(
            status=rdWrappers.GenerationStatus.SUCCESS,
            coordinates=np.zeros((1, len(molecule.atoms), 3)),
            records=(rdWrappers.ConformerRecord(0, 0),),
            engine="fixture",
            engine_version="1",
            diagnostic=None,
        )

    execute_whole_complex_route(
        source,
        "C1",
        SMOKE_SETTINGS,
        1,
        generate=generate,
        optimize=lambda molecule, forcefield, **options: "optimized",
        radius=lambda atomic_number: 1.0,
        add_hydrogens=False,
    )

    assert source.c_bonds[0].bond_kind is original_kind
