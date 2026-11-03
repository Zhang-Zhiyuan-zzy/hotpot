from __future__ import annotations

import json
from copy import copy
from types import SimpleNamespace

import pytest

from hotpot.cheminfo.forcefields import acceptance as acceptance_policy
from hotpot.cheminfo.forcefields import ff
from hotpot.cheminfo.rules.physics.forcefields import acceptance as acceptance_rules
from hotpot.cheminfo.rules.mathematics import geometry as geo
from hotpot.cheminfo.adapters.structure import geometry as geometry_adapter
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
    assert acceptance_rules._acceptance_checks_pass(checks) is expected


def test_acceptance_levels_preserve_order_metrics_and_serialization(monkeypatch):
    scan_calls = []

    def scan_relations(*args, **options):
        scan_calls.append(options)
        return _clear_bond_ring_report()

    monkeypatch.setattr(
        acceptance_policy.geometry_adapter,
        "screen_bond_ring_relations",
        scan_relations,
    )

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
        geometry_adapter,
        "screen_bond_ring_relations",
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
        geometry_adapter,
        "screen_bond_ring_relations",
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
        geometry_adapter,
        "screen_bond_ring_relations",
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
        geometry_adapter,
        "screen_bond_ring_relations",
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


@pytest.mark.parametrize('options', ({'level': 'unknown'}, {'forcefield_stage': 'unknown'}))
def test_invalid_policy_options_fail_before_carrier_reads(options):
    class UnavailableCarrier:
        def __getattribute__(self, name):
            raise AssertionError(f'Unexpected carrier read: {name}')

    with pytest.raises(ValueError):
        acceptance_policy.evaluate_structure_acceptance(UnavailableCarrier(), **options)


def test_malformed_coordinates_keep_topology_backend_diagnostics_without_capture(monkeypatch):
    import numpy as np
    from hotpot.cheminfo.structure.forcefields import AtomTopologySignature, TopologyReference

    class UnavailableAtom:
        def __getattribute__(self, name):
            raise AssertionError(f'Unexpected atom read: {name}')

    def unavailable(*args, **kwargs):
        raise AssertionError('Malformed coordinates must stop before geometry capture')

    mol = SimpleNamespace(atoms=(UnavailableAtom(),), coordinates=np.zeros((1, 2)), bonds=())
    reference = TopologyReference((AtomTopologySignature(0, 1, 6, 0), AtomTopologySignature(1, 2, 6, 0)), ())
    for name in ('capture_topology', 'capture_atom_pair_acceptance_values', 'capture_bond_acceptance_values',
                 'capture_coordination_acceptance_values'):
        monkeypatch.setattr(acceptance_policy, name, unavailable)
    monkeypatch.setattr(geometry_adapter, 'measure_atom_pair_distances', unavailable)
    monkeypatch.setattr(geometry_adapter, 'screen_bond_ring_relations', unavailable)
    result = acceptance_policy.evaluate_structure_acceptance(
        mol, level='standard', topology_reference=reference, forcefield_report={},
    )
    assert [c.name for c in result.checks] == [
        'coordinate_shape', 'finite_coordinates', 'topology_atom_count',
        'forcefield_setup', 'finite_final_energy', 'finite_rms_gradient',
        'finite_max_gradient', 'backend_explosion',
    ]
    assert result.checks[1].atom_indices == ()
    assert result.metrics == {'atom_count': 1, 'bond_count': 0}


def test_nonfinite_coordinate_section_reads_only_returned_row_labels():
    import numpy as np

    class UnavailableLabel:
        @property
        def idx(self):
            raise AssertionError('Finite atom labels must not be captured')

    atoms = (UnavailableLabel(), SimpleNamespace(idx=91), UnavailableLabel())
    checks, finite, metrics = acceptance_policy._coordinate_acceptance_section(
        SimpleNamespace(bonds=()), atoms,
        np.array([[0., 0., 0.], [np.nan, 0., 0.], [1., 0., 0.]]),
    )
    assert finite is False
    assert checks[1].atom_indices == (91,)
    assert metrics == {'atom_count': 3, 'bond_count': 0}


def test_off_measures_pairs_once_without_policy_capture(monkeypatch):
    calls = []

    def measure(mol, scope):
        calls.append((mol, scope))
        return (SimpleNamespace(measurement=SimpleNamespace(distance=0.25)),)

    def unavailable(*args, **kwargs):
        raise AssertionError('Off level must not capture pair policy values')

    monkeypatch.setattr(geometry_adapter, 'measure_atom_pair_distances', measure)
    monkeypatch.setattr(acceptance_policy, 'capture_atom_pair_acceptance_values', unavailable)
    mol = object()
    checks, metrics = acceptance_policy._atom_pair_distance_acceptance_section(
        mol, 'off', ff.StructureAcceptanceThresholds(),
    )
    assert checks == ()
    assert metrics == {'minimum_pair_distance': 0.25}
    assert calls == [(mol, 'all')]


def test_pair_capture_reads_only_consumed_radii():
    from hotpot.cheminfo.adapters.structure.forcefields import capture_atom_pair_acceptance_values

    class UnavailableRadius:
        @property
        def covalent_radius(self):
            raise AssertionError('This pair must not consume radii')

    def pair(keys, bonded, measured, atoms):
        return SimpleNamespace(
            measurement=SimpleNamespace(distance=measured),
            target=SimpleNamespace(
                first=SimpleNamespace(key=keys[0], atom=atoms[0]),
                second=SimpleNamespace(key=keys[1], atom=atoms[1]), bonded=bonded,
            ),
        )

    unread = UnavailableRadius()
    rows = (pair((0, 1), False, 0.001, (unread, unread)),
            pair((0, 2), True, 0.5, (unread, unread)),
            pair((1, 2), False, 0.5, (SimpleNamespace(covalent_radius=0.5), SimpleNamespace(covalent_radius=0.75))))
    prepared = capture_atom_pair_acceptance_values(rows, capture_radii=True, overlap_tolerance=0.001)
    assert prepared == (((0, 1), False, 0.001, None), ((0, 2), True, 0.5, None), ((1, 2), False, 0.5, 1.25))
    basic = capture_atom_pair_acceptance_values((pair((1, 2), False, 0.5, (unread, unread)),),
                                               capture_radii=False, overlap_tolerance=0.001)
    assert basic[0][-1] is None


def test_basic_bond_capture_avoids_radii_and_zero_radius_avoids_metal_flag():
    from hotpot.cheminfo.adapters.structure.forcefields import capture_bond_acceptance_values

    class Atom:
        def __init__(self, label, radius):
            self.idx = label
            self.radius = radius

        @property
        def covalent_radius(self):
            assert self.radius is not None, 'Basic must not capture radii'
            return self.radius

    class Bond:
        def __init__(self, first, second):
            self.atom1 = first
            self.atom2 = second

        @property
        def is_metal_ligand_bond(self):
            raise AssertionError('Unconsumed metal-ligand flag')

    atoms = (Atom(91, None), Atom(17, None))
    mol = SimpleNamespace(bonds=(Bond(*atoms),))
    assert capture_bond_acceptance_values(mol, atoms, capture_radii=False) == (
        (0, (0, 1), (91, 17), None, None),
    )
    atoms[0].radius = atoms[1].radius = 0.0
    assert capture_bond_acceptance_values(mol, atoms, capture_radii=True) == (
        (0, (0, 1), (91, 17), 0.0, None),
    )


@pytest.mark.parametrize('kind', ('single', 'dative', 'zero'))
def test_coordination_capture_keeps_element_classification_independent_of_bond_kind(kind):
    from hotpot.cheminfo.adapters.structure.forcefields import (
        _iter_metal_donor_pairs, capture_coordination_acceptance_values,
    )

    mol = Molecule()
    for number in (26, 7, 29, 6, 30):
        mol.create_atom(atomic_number=number)
    mol.add_bond(1, 0, bond_order=1.0, bond_kind=kind)
    mol.add_bond(0, 2, bond_order=1.0, bond_kind='dative')
    mol.add_bond(1, 3, bond_order=1.0, bond_kind='dative')
    assert tuple(bond.bond_kind.value for bond in mol.bonds) == (kind, 'dative', 'dative')
    assert tuple(_iter_metal_donor_pairs(mol)) == ((mol.atoms[0], mol.atoms[1]),)
    metals, pairs, labels = capture_coordination_acceptance_values(mol, tuple(mol.atoms))
    assert metals == (0, 2, 4)
    assert pairs == ((0, 1),)
    assert labels == tuple((row, mol.atoms[row].idx) for row in (0, 1, 2, 4))


def test_finite_coordinates_continue_after_topology_and_backend_failures(monkeypatch):
    from dataclasses import replace

    mol = _carbon_bond()
    reference = ff.capture_topology(mol)
    reference = replace(reference, atoms=(replace(reference.atoms[0], atom_id=-1), reference.atoms[1]))
    scans = []

    def scan(*args, **kwargs):
        scans.append(kwargs)
        return _clear_bond_ring_report()

    monkeypatch.setattr(geometry_adapter, 'screen_bond_ring_relations', scan)
    result = acceptance_policy.evaluate_structure_acceptance(
        mol, level='standard', topology_reference=reference, forcefield_report={},
    )
    names = [check.name for check in result.checks]
    assert names.index('topology_atom_identity') < names.index('forcefield_setup') < names.index('atom_overlap')
    assert names[-1] == 'bond_ring_piercing'
    assert result.passed is False
    assert scans == [{'ring_scope': 'ligand_skeleton', 'max_ring_size': 16}]


def test_operation_builds_fresh_mutable_coordination_metric_dictionaries():
    import numpy as np
    from hotpot.cheminfo.rules.physics.forcefields.contracts import _BondRingAcceptanceFacts

    atoms = (SimpleNamespace(idx=91, is_metal=True), SimpleNamespace(idx=17, is_metal=False))
    bond = SimpleNamespace(atom1=atoms[0], atom2=atoms[1], is_metal_ligand_bond=True)
    mol = SimpleNamespace(bonds=(bond,))
    facts = _BondRingAcceptanceFacts((), 'full_graph', 16, 0, 0, 0, 0, True)
    coordinates = np.array([[0., 0., 0.], [1., 0., 0.]])
    _, first = acceptance_policy._bond_ring_coordination_acceptance_section(mol, atoms, coordinates, facts)
    _, second = acceptance_policy._bond_ring_coordination_acceptance_section(mol, atoms, coordinates, facts)
    assert first is not second
    first_environment = first['coordination_environments'][0]
    second_environment = second['coordination_environments'][0]
    assert first_environment == second_environment == {
        'metal_index': 91, 'coordination_number': 1, 'donor_indices': (17,),
        'distances': (1.0,), 'angles': (),
    }
    assert first_environment is not second_environment
    first_environment['donor_indices'] = ()
    assert second_environment['donor_indices'] == (17,)
