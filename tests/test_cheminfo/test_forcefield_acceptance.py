from __future__ import annotations

import json
from copy import copy
from types import SimpleNamespace

import pytest

from hotpot.cheminfo.forcefields import utils as ff
from hotpot.cheminfo import geometry as geo
from hotpot.cheminfo.core import Molecule


def _carbon_bond() -> Molecule:
    mol = Molecule()
    mol.create_atom(atomic_number=6, coordinates=(0.0, 0.0, 0.0))
    mol.create_atom(atomic_number=6, coordinates=(1.52, 0.0, 0.0))
    mol.add_bond(0, 1, bond_order=1.0)
    mol.refresh_atom_id()
    return mol


def _bond_ring_report(
    state: geo.PiercingState,
    *,
    excluded_ring_count: int = 0,
):
    finding = SimpleNamespace(
        target=SimpleNamespace(
            ring=SimpleNamespace(key=(2, 3, 4)),
            bond=SimpleNamespace(key=(0, 1)),
        ),
        relation=SimpleNamespace(
            state=state,
            indeterminacy_causes=(
                geo.SegmentCycleIndeterminacy.NUMERIC_BAND,
            ),
        ),
    )
    piercings = (finding,) if state is geo.PiercingState.PIERCES else ()
    undetermined = (
        (finding,) if state is geo.PiercingState.UNDETERMINED else ()
    )
    return SimpleNamespace(
        piercings=piercings,
        undetermined=undetermined,
        piercing_pair_count=len(piercings),
        undetermined_pair_count=len(undetermined),
        selected_ring_count=1,
        excluded_ring_count=excluded_ring_count,
        max_ring_size=16,
        ring_scope="ligand_skeleton",
        scan_complete=True,
    )


def _successful_forcefield_evidence():
    return {
        "setup_succeeded": True,
        "converged": True,
        "epochs_completed": 6,
        "segment_epochs_completed": 6,
        "final_energy": -12.5,
        "energy_unit": "kJ/mol",
        "rms_gradient": 0.25,
        "max_gradient": 1.5,
        "exploded": False,
        "energy_changes": (1.0e-5, 2.0e-5, 3.0e-5, 4.0e-5, 5.0e-5),
        "max_displacements": (
            1.0e-5,
            2.0e-5,
            3.0e-5,
            4.0e-5,
            5.0e-5,
        ),
    }


def _clear_bond_ring_report():
    return SimpleNamespace(
        piercings=(),
        undetermined=(),
        piercing_pair_count=0,
        undetermined_pair_count=0,
        selected_ring_count=0,
        excluded_ring_count=0,
        max_ring_size=16,
        ring_scope="ligand_skeleton",
        scan_complete=True,
    )


def _serialized_check_contract(report):
    payload = json.loads(json.dumps(report.to_dict()))
    return [
        (
            check["name"],
            check["passed"],
            check["severity"],
            check["measured"],
            check["threshold"],
        )
        for check in payload["checks"]
    ], payload["metrics"]


@pytest.mark.parametrize(
    ("checks", "expected"),
    (
        ((), True),
        ((ff.AcceptanceCheck("passed", True),), True),
        ((ff.AcceptanceCheck("warning", False, severity="warning"),), True),
        ((ff.AcceptanceCheck("info", False, severity="info"),), True),
        ((ff.AcceptanceCheck("error", False, severity="error"),), False),
    ),
)
def test_acceptance_checks_pass_only_rejects_failed_errors(checks, expected):
    assert ff._acceptance_checks_pass(checks) is expected


def test_acceptance_levels_preserve_order_metrics_and_serialization(monkeypatch):
    scan_calls = []

    def scan_relations(*args, **options):
        scan_calls.append(options)
        return _clear_bond_ring_report()

    monkeypatch.setattr(ff.geo, "scan_bond_ring_relations", scan_relations)

    coordinate_checks = [
        ("coordinate_shape", True, "error", [2, 3], [2, 3]),
        ("finite_coordinates", True, "error", True, True),
        ("forcefield_setup", True, "error", True, True),
        ("finite_final_energy", True, "error", -12.5, "finite"),
        ("finite_rms_gradient", True, "error", 0.25, "finite"),
        ("finite_max_gradient", True, "error", 1.5, "finite"),
    ]
    backend_check = [
        ("backend_explosion", True, "error", False, False),
    ]
    basic_geometry_checks = [
        ("atom_overlap", True, "error", 0, 1.0e-3),
        ("atom_too_close", True, "error", 0, 0.40),
        ("bond_distance", True, "error", 1.52, [0.0, 30.0]),
    ]
    standard_geometry_checks = [
        ("atom_overlap", True, "error", 0, 1.0e-3),
        ("atom_too_close", True, "error", 0, [0.40, 0.50, 0.55]),
        ("bond_distance", True, "error", 1.52, [0.0, 30.0]),
        (
            "bond_length_ratio",
            True,
            "error",
            None,
            [[0.65, 1.45], [0.65, 1.60]],
        ),
        ("short_bond", True, "error", 0, [0.65, 0.65]),
        (
            "bond_ring_piercing",
            True,
            "error",
            "does_not_pierce",
            "does_not_pierce",
        ),
    ]
    expected_checks = {
        "off": coordinate_checks,
        "basic": coordinate_checks + backend_check + basic_geometry_checks,
        "standard": coordinate_checks + backend_check + [
            ("forcefield_convergence", True, "warning", True, True),
        ] + standard_geometry_checks,
        "strict": coordinate_checks + backend_check + [
            ("forcefield_convergence", True, "error", True, True),
            ("rms_gradient", True, "error", 0.25, 1.0),
            ("max_gradient", True, "error", 1.5, 5.0),
            ("energy_change", True, "error", 5.0e-5, 1.0e-4),
            ("max_displacement", True, "error", 5.0e-5, 1.0e-4),
            ("stability_observations", True, "error", 5, 5),
        ] + standard_geometry_checks,
    }
    expected_metrics = {
        "off": {
            "atom_count": 2,
            "bond_count": 1,
            "minimum_pair_distance": 1.52,
        },
        "basic": {
            "atom_count": 2,
            "bond_count": 1,
            "minimum_pair_distance": 1.52,
            "maximum_bond_length": 1.52,
        },
    }
    ring_metrics = {
        "bond_ring_piercing_count": 0,
        "bond_ring_undetermined_count": 0,
        "bond_ring_scan_complete": True,
        "bond_ring_selected_ring_count": 0,
        "bond_ring_excluded_ring_count": 0,
        "bond_ring_max_ring_size": 16,
        "bond_ring_scope": "ligand_skeleton",
        "coordination_environments": [],
    }
    expected_metrics["standard"] = expected_metrics["basic"] | ring_metrics
    expected_metrics["strict"] = expected_metrics["basic"] | ring_metrics

    for level in ("off", "basic", "standard", "strict"):
        report = ff.evaluate_structure_acceptance(
            _carbon_bond(),
            level=level,
            forcefield_report=_successful_forcefield_evidence(),
        )
        checks, metrics = _serialized_check_contract(report)

        assert report.passed
        assert checks == expected_checks[level]
        assert metrics == expected_metrics[level]

    assert scan_calls == [
        {"ring_scope": "ligand_skeleton", "max_ring_size": 16},
        {"ring_scope": "ligand_skeleton", "max_ring_size": 16},
    ]


def test_acceptance_report_is_owned_by_forcefields_and_serializable():
    report = ff.evaluate_structure_acceptance(_carbon_bond(), level="basic")

    assert isinstance(report, ff.ForceFieldValidationReport)
    assert report.passed
    assert ff.is_structure_accepted(_carbon_bond(), level="basic")
    json.dumps(report.to_dict())


def test_topology_reference_allows_only_appended_hydrogen():
    mol = _carbon_bond()
    reference = ff.capture_topology(mol, allow_added_hydrogens=True)
    candidate = copy(mol)
    hydrogen = candidate.create_atom(
        atomic_number=1,
        coordinates=(-1.0, 0.0, 0.0),
    )
    candidate.add_bond(candidate.atoms[0], hydrogen, bond_order=1.0)

    report = ff.evaluate_structure_acceptance(
        candidate,
        level="off",
        topology_reference=reference,
    )

    assert report.passed


def test_confirmed_bond_ring_piercing_fails_acceptance(monkeypatch):
    scan_options = {}

    def scan_relations(*args, **options):
        scan_options.update(options)
        return _bond_ring_report(geo.PiercingState.PIERCES)

    monkeypatch.setattr(
        ff.geo,
        "scan_bond_ring_relations",
        scan_relations,
    )

    report = ff.evaluate_structure_acceptance(
        _carbon_bond(),
        level="standard",
    )

    assert not report.passed
    assert any(
        check.name == "bond_ring_piercing" for check in report.failures
    )
    assert scan_options["max_ring_size"] == 16


def test_undetermined_bond_ring_relation_warns_without_rejection(monkeypatch):
    monkeypatch.setattr(
        ff.geo,
        "scan_bond_ring_relations",
        lambda *args, **kwargs: _bond_ring_report(
            geo.PiercingState.UNDETERMINED
        ),
    )

    report = ff.evaluate_structure_acceptance(
        _carbon_bond(),
        level="standard",
    )

    assert report.passed
    warning = next(
        check for check in report.warnings
        if check.name == "bond_ring_piercing"
    )
    assert warning.measured == ("numeric_band",)


def test_excluded_rings_are_reported_as_incomplete_policy_coverage(monkeypatch):
    monkeypatch.setattr(
        ff.geo,
        "scan_bond_ring_relations",
        lambda *args, **kwargs: _bond_ring_report(
            geo.PiercingState.DOES_NOT_PIERCE,
            excluded_ring_count=2,
        ),
    )

    report = ff.evaluate_structure_acceptance(
        _carbon_bond(),
        level="standard",
    )

    assert report.passed
    warning = next(
        check for check in report.warnings
        if check.name == "bond_ring_scope_coverage"
    )
    assert warning.measured == 2
    assert report.metrics["bond_ring_excluded_ring_count"] == 2


def test_complete_ring_scope_has_no_coverage_warning(monkeypatch):
    monkeypatch.setattr(
        ff.geo,
        "scan_bond_ring_relations",
        lambda *args, **kwargs: _bond_ring_report(
            geo.PiercingState.DOES_NOT_PIERCE,
            excluded_ring_count=0,
        ),
    )

    report = ff.evaluate_structure_acceptance(
        _carbon_bond(),
        level="standard",
    )

    assert report.passed
    assert not any(
        check.name == "bond_ring_scope_coverage" for check in report.warnings
    )
    assert report.metrics["bond_ring_excluded_ring_count"] == 0
