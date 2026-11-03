"""Geometry-only post-checks share the existing rules without FF requirements."""

from __future__ import annotations

import numpy as np
import pytest

from hotpot import Molecule
from hotpot.cheminfo.operations.forcefields import acceptance


def _molecule(distance=1.54):
    mol = Molecule()
    mol.create_atom(atomic_number=6)
    mol.create_atom(atomic_number=6)
    mol.add_bond(0, 1, bond_order=1.0)
    mol.coordinates = np.array(((0.0, 0.0, 0.0), (distance, 0.0, 0.0)))
    return mol


@pytest.mark.parametrize("level", ("off", "basic", "standard", "strict"))
def test_geometry_route_keeps_exact_geometry_checks_and_profiles_without_ff_report(level):
    mol = _molecule()
    geometry = acceptance._evaluate_geometry_acceptance(mol, level=level)
    public = acceptance.evaluate_structure_acceptance(mol, level=level)
    assert geometry.level == level
    assert geometry.checks == tuple(check for check in public.checks if check.name != "forcefield_report")
    assert geometry.metrics == public.metrics
    assert geometry.passed is True
    if level == "strict":
        assert public.passed is False
        assert [check.name for check in public.failures] == ["forcefield_report"]
    else:
        assert public.passed is True
    assert "_evaluate_geometry_acceptance" not in acceptance.__all__


@pytest.mark.parametrize("level, expected_scans", (("basic", 0), ("standard", 1), ("strict", 1)))
def test_geometry_route_uses_one_existing_pair_bond_and_ring_pass(monkeypatch, level, expected_scans):
    counts = {"pairs": 0, "bonds": 0, "rings": 0}

    def count(owner, name, key):
        operation = getattr(owner, name)

        def counted(*args, **kwargs):
            counts[key] += 1
            return operation(*args, **kwargs)

        monkeypatch.setattr(owner, name, counted)

    count(acceptance.geometry_adapter, "measure_atom_pair_distances", "pairs")
    count(acceptance, "capture_bond_acceptance_values", "bonds")
    count(acceptance.geometry_adapter, "screen_bond_ring_relations", "rings")

    def reject_ff_checks(*args, **kwargs):
        raise AssertionError("geometry-only post-checks must not fabricate or request FF evidence")

    monkeypatch.setattr(acceptance.acceptance_rules, "_forcefield_acceptance_checks", reject_ff_checks)
    report = acceptance._evaluate_geometry_acceptance(_molecule(), level=level)
    assert report.passed is True
    assert counts == {"pairs": 1, "bonds": 1, "rings": expected_scans}


def test_strict_geometry_still_reports_actual_short_bond_observation():
    report = acceptance._evaluate_geometry_acceptance(_molecule(0.1), level="strict")
    assert report.passed is False
    assert any(check.name == "short_bond" and not check.passed for check in report.checks)
    assert "forcefield_report" not in {check.name for check in report.checks}
    assert report.metrics["maximum_bond_length"] == pytest.approx(0.1)


def test_nonfinite_geometry_uses_existing_coordinate_failure_and_skips_scans(monkeypatch):
    mol = _molecule(np.nan)

    def unexpected_scan(*args, **kwargs):
        raise AssertionError("invalid coordinates must stop before pair or ring scans")

    monkeypatch.setattr(acceptance.geometry_adapter, "measure_atom_pair_distances", unexpected_scan)
    monkeypatch.setattr(acceptance.geometry_adapter, "screen_bond_ring_relations", unexpected_scan)
    report = acceptance._evaluate_geometry_acceptance(mol, level="strict")
    assert report.passed is False
    assert [check.name for check in report.failures] == ["finite_coordinates"]
    assert report.failures[0].atom_indices == (1,)
