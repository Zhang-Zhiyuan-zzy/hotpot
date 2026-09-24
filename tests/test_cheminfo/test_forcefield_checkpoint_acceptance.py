"""Tests for precomputed bond-ring evidence at force-field checkpoints."""

from inspect import Parameter, signature
from types import SimpleNamespace

import numpy as np
import pytest

from hotpot.cheminfo import geometry as geo
from hotpot.cheminfo.core import Molecule
from hotpot.cheminfo.forcefields import acceptance as acceptance_policy


def _carbon_bond() -> Molecule:
    mol = Molecule()
    mol.create_atom(atomic_number=6, coordinates=(0.0, 0.0, 0.0))
    mol.create_atom(atomic_number=6, coordinates=(1.52, 0.0, 0.0))
    mol.add_bond(0, 1, bond_order=1.0)
    mol.refresh_atom_id()
    return mol


def _empty_screening_report(
    *,
    ring_scope: geo.RingScope,
) -> geo.BondRingScreeningReport:
    return geo.BondRingScreeningReport(
        actionable_findings=(),
        ring_scope=ring_scope,
        max_ring_size=16,
        selected_ring_count=0,
        excluded_ring_count=0,
        candidate_pair_count=0,
        aabb_separated_pair_count=0,
        exact_pair_count=0,
        piercing_pair_count=0,
        does_not_pierce_pair_count=0,
        undetermined_pair_count=0,
        scan_complete=True,
    )


def _piercing_screening_report() -> geo.BondRingScreeningReport:
    finding = geo.BondRingFinding(
        target=SimpleNamespace(
            ring=SimpleNamespace(key=(2, 3, 4)),
            bond=SimpleNamespace(key=(0, 1)),
        ),
        relation=SimpleNamespace(state=geo.PiercingState.PIERCES),
    )
    return geo.BondRingScreeningReport(
        actionable_findings=(finding,),
        ring_scope="full_graph",
        max_ring_size=16,
        selected_ring_count=1,
        excluded_ring_count=0,
        candidate_pair_count=1,
        aabb_separated_pair_count=0,
        exact_pair_count=1,
        piercing_pair_count=1,
        does_not_pierce_pair_count=0,
        undetermined_pair_count=0,
        scan_complete=True,
    )


def test_checkpoint_acceptance_requires_and_reuses_precomputed_report(
    monkeypatch,
):
    report = _empty_screening_report(ring_scope="full_graph")

    def unexpected_scan(*args, **kwargs):
        raise AssertionError("checkpoint acceptance must not rescan geometry")

    monkeypatch.setattr(
        acceptance_policy.geo,
        "screen_bond_ring_relations",
        unexpected_scan,
    )

    result = acceptance_policy.evaluate_structure_acceptance_at_checkpoint(
        _carbon_bond(),
        bond_ring_report=report,
        level="standard",
    )

    report_parameter = signature(
        acceptance_policy.evaluate_structure_acceptance_at_checkpoint
    ).parameters["bond_ring_report"]
    assert report_parameter.default is Parameter.empty
    assert result.passed
    assert result.metrics["bond_ring_scope"] == "full_graph"
    assert result.metrics["bond_ring_scan_complete"] is True


@pytest.mark.parametrize("level", ("off", "basic"))
def test_checkpoint_bond_ring_gate_is_independent_of_quality_level(level):
    mol = _carbon_bond()

    result = acceptance_policy.evaluate_structure_acceptance_at_checkpoint(
        mol,
        bond_ring_report=_piercing_screening_report(),
        level=level,
    )

    assert not result.passed
    assert result.metrics["bond_ring_piercing_count"] == 1
    assert any(
        check.name == "bond_ring_piercing" for check in result.failures
    )


def test_public_acceptance_self_scans_once_with_existing_scope(monkeypatch):
    report = _empty_screening_report(ring_scope="ligand_skeleton")
    calls = []

    def scan_relations(mol, **options):
        calls.append((mol, options))
        return report

    monkeypatch.setattr(
        acceptance_policy.geo,
        "screen_bond_ring_relations",
        scan_relations,
    )
    mol = _carbon_bond()

    result = acceptance_policy.evaluate_structure_acceptance(
        mol,
        level="standard",
    )

    assert result.passed
    assert calls == [(
        mol,
        {"ring_scope": "ligand_skeleton", "max_ring_size": 16},
    )]


def test_public_acceptance_preserves_nonfinite_scan_short_circuit(monkeypatch):
    def unexpected_scan(*args, **kwargs):
        raise AssertionError("nonfinite coordinates must fail before ring scanning")

    monkeypatch.setattr(
        acceptance_policy.geo,
        "screen_bond_ring_relations",
        unexpected_scan,
    )
    mol = _carbon_bond()
    mol.atoms[1].coordinates = (np.nan, 0.0, 0.0)

    result = acceptance_policy.evaluate_structure_acceptance(
        mol,
        level="standard",
    )

    assert not result.passed
    assert any(check.name == "finite_coordinates" for check in result.failures)
