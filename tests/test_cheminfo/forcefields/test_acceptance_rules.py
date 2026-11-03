"""Detached acceptance decisions, immutable outputs and numerical boundaries."""

import ast
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest

from hotpot.cheminfo.rules.mathematics import geometry as geo
from hotpot.cheminfo.rules.physics.forcefields import acceptance as rules
from hotpot.cheminfo.rules.physics.forcefields import contracts as rule_contracts
from hotpot.cheminfo.rules.physics.forcefields.contracts import (
    AcceptanceCheck,
    StructureAcceptanceThresholds,
    _BondRingAcceptanceFacts,
    _BondRingAcceptanceFinding,
)
from hotpot.cheminfo.structure.forcefields import (
    AtomTopologySignature,
    BondTopology,
    TopologyReference,
)


LIMITS = StructureAcceptanceThresholds()


def test_rule_modules_import_only_lower_values_mathematics_and_stdlib():
    for module in (rules, rule_contracts):
        tree = ast.parse(Path(module.__file__).read_text())
        imports = [
            (node.level, node.module)
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        ]
        assert all(
            name is None or not any(
                part in name.split('.')
                for part in ('adapters', 'core', 'native', 'workflows', 'obWrappers')
            )
            for _, name in imports
        )
        assert not any(isinstance(node, ast.ClassDef) for node in tree.body) or module is not rules


def test_coordinate_rules_return_rows_without_mutating_or_retaining_arrays():
    coordinates = np.array([[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0], [0.0, np.inf, 0.0]])
    coordinates.setflags(write=False)
    checks, finite = rules._coordinate_acceptance_checks(coordinates, 3)
    assert finite is False
    assert checks[1].atom_indices == (1, 2)
    assert checks[0].measured == (3, 3)
    with pytest.raises(FrozenInstanceError):
        checks[1].passed = True
    malformed, finite = rules._coordinate_acceptance_checks(np.zeros((3, 2)), 3)
    assert finite is False
    assert malformed[1].atom_indices == ()
    assert malformed[0].threshold == (3, 3)


def test_topology_count_and_identity_bond_and_hydrogen_rules():
    atom = AtomTopologySignature(0, 91, 6, 0)
    reference = TopologyReference((atom,), ())
    assert rules._topology_atom_count_checks(0, 1)[0].name == 'topology_atom_count'
    assert rules._topology_atom_count_checks(1, 1) == ()
    hydrogen = AtomTopologySignature(1, 22, 1, 0)
    bond = BondTopology((0, 1), 1.0, 'single')
    valid = TopologyReference((atom, hydrogen), (bond,))
    assert rules._topology_checks(valid, reference)[0].passed
    changed = replace(valid, atoms=(replace(atom, formal_charge=1), hydrogen))
    assert rules._topology_checks(changed, reference)[0].name == 'topology_atom_identity'
    for changed_atom in (replace(atom, atom_id=7), replace(atom, atomic_number=8)):
        changed = replace(valid, atoms=(changed_atom, hydrogen))
        assert rules._topology_checks(changed, reference)[0].atom_indices == (0,)
    no_additions = replace(reference, allow_added_hydrogens=False)
    assert [c.name for c in rules._topology_checks(valid, no_additions)] == [
        'topology_added_atoms', 'topology_added_bond',
    ]
    carbon = replace(hydrogen, atomic_number=6)
    assert rules._topology_checks(replace(valid, atoms=(atom, carbon)), reference)[0].measured == (6,)
    unbonded = replace(valid, bonds=())
    assert rules._topology_checks(unbonded, reference)[0].name == 'topology_added_hydrogen_degree'
    assert rules._topology_checks(replace(valid, bonds=()), valid)[0].name == 'topology_original_bond'
    assert rules._topology_checks(replace(valid, bonds=(replace(bond, bond_order=2.0),)), valid)[0].name == 'topology_original_bond'


def test_forcefield_none_empty_candidate_zero_and_severity_distinctions():
    assert rules._forcefield_acceptance_checks(None, 'standard', LIMITS, 'final') == ()
    assert rules._forcefield_acceptance_checks(None, 'strict', LIMITS, 'final')[0].name == 'forcefield_report'
    empty = rules._forcefield_acceptance_checks({}, 'basic', LIMITS, 'candidate')
    assert [(c.name, c.passed) for c in empty] == [
        ('forcefield_setup', False), ('finite_final_energy', False), ('backend_explosion', False),
    ]
    evidence = dict(setup_succeeded=True, final_energy=0.0, rms_gradient=0.0,
                    max_gradient=0.0, exploded=False, converged=False)
    candidate = rules._forcefield_acceptance_checks(evidence, 'strict', LIMITS, 'candidate')
    assert all(c.passed for c in candidate)
    assert [c.name for c in candidate] == ['forcefield_setup', 'finite_final_energy', 'backend_explosion']
    standard = rules._forcefield_acceptance_checks(evidence, 'standard', LIMITS, 'final')
    assert standard[-1].severity == 'warning'
    assert rules._acceptance_checks_pass(standard)
    strict = rules._forcefield_acceptance_checks(evidence, 'strict', LIMITS, 'final')
    assert next(c for c in strict if c.name == 'forcefield_convergence').severity == 'error'
    assert not rules._acceptance_checks_pass(strict)


def test_strict_history_uses_trailing_window_and_one_epoch_exception():
    evidence = dict(setup_succeeded=True, final_energy=0.0, rms_gradient=0.0,
                    max_gradient=0.0, exploded=False, converged=True,
                    epochs_completed=20, segment_epochs_completed=6,
                    energy_changes=(99.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    max_displacements=(99.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    checks = rules._forcefield_acceptance_checks(evidence, 'strict', LIMITS, 'final')
    assert all(c.passed for c in checks)
    assert checks[-1].measured == checks[-1].threshold == 5
    evidence.update(segment_epochs_completed=1, energy_changes=(), max_displacements=())
    checks = rules._forcefield_acceptance_checks(evidence, 'strict', LIMITS, 'final')
    assert all(c.passed for c in checks)
    assert checks[-1].measured == checks[-1].threshold == 0
    evidence.update(segment_epochs_completed=2)
    assert not rules._forcefield_acceptance_checks(evidence, 'strict', LIMITS, 'final')[-1].passed


def test_pair_rules_keep_overlap_inclusive_close_exclusive_and_sorted_override():
    pairs = (
        ((8, 9), False, LIMITS.overlap_tolerance, None),
        ((4, 5), True, 0.39, None),
        ((2, 3), False, 0.39, 2.0),
        ((6, 7), True, LIMITS.basic_minimum_distance, None),
        ((10, 11), False, 1.1, 2.0),
    )
    checks = rules._atom_pair_acceptance_checks(pairs, 'standard', LIMITS)
    assert [(c.name, c.atom_indices) for c in checks] == [
        ('atom_overlap', (8, 9)), ('atom_too_close', (2, 3)), ('atom_too_close', (4, 5)),
    ]
    assert checks[1].threshold == 1.1
    assert checks[2].threshold == LIMITS.basic_minimum_distance
    assert rules._atom_pair_acceptance_checks(pairs, 'off', LIMITS) == ()
    basic = rules._atom_pair_acceptance_checks((((0, 1), False, 0.4, None),), 'basic', LIMITS)
    assert all(c.passed for c in basic)


def test_bond_rules_keep_sequence_indices_labels_failure_order_and_maximum_scalar():
    coordinates = np.array([[0., 0., 0.], [0., 0., 0.], [40., 0., 0.]])
    coordinates.setflags(write=False)
    bonds = ((7, (0, 1), (91, 17), 2.0, False), (2, (0, 2), (91, 25), 0.0, None))
    checks, maximum = rules._bond_geometry_acceptance_checks(coordinates, bonds, 'standard', LIMITS)
    assert [(c.name, c.bond_indices) for c in checks] == [
        ('bond_distance', (7,)), ('short_bond', (7,)),
        ('bond_length_ratio', (7,)), ('bond_distance', (2,)),
    ]
    assert checks[0].atom_indices == (91, 17)
    assert maximum == 40.0
    basic, maximum = rules._bond_geometry_acceptance_checks(
        coordinates, ((7, (0, 2), (91, 25), None, None),), 'basic', LIMITS,
    )
    assert [c.name for c in basic] == ['bond_distance']
    assert maximum == 40.0
    empty, maximum = rules._bond_geometry_acceptance_checks(coordinates, (), 'strict', LIMITS)
    assert [c.name for c in empty] == ['bond_distance', 'bond_length_ratio', 'short_bond']
    assert all(c.passed for c in empty)
    assert maximum == 0.0


def test_coordination_outputs_are_detached_sorted_and_include_isolated_metals():
    coordinates = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 0.], [9., 9., 9.]])
    coordinates.setflags(write=False)
    result = rules._coordination_metrics(
        coordinates, (4, 0), ((0, 3), (0, 2), (0, 1)),
        ((0, 91), (1, 7), (2, 22), (3, 5), (4, 100)),
    )
    assert result == ((91, 3, (7, 22, 5), (1.0, 1.0, 0.0), (90.0,)), (100, 0, (), (), ()))
    coordinates.setflags(write=True)
    coordinates[:] = 99.0
    assert result[0][3:] == ((1.0, 1.0, 0.0), (90.0,))


def test_coordination_clips_roundoff_before_arccos(monkeypatch):
    original_dot = np.dot
    monkeypatch.setattr(rules.np, 'dot', lambda first, second: original_dot(first, second) * (1.0 + 1.e-12))
    result = rules._coordination_metrics(
        np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]]),
        (0,), ((0, 1), (0, 2)), ((0, 0), (1, 1), (2, 2)),
    )
    assert result[0][-1] == (0.0,)


def test_normalized_ring_checks_preserve_order_scope_warning_and_natural_keyerror():
    piercing = _BondRingAcceptanceFinding((8, 9, 10), (1, 2), geo.PiercingState.PIERCES, ())
    unresolved = _BondRingAcceptanceFinding(
        (4, 5, 6), (0, 1), geo.PiercingState.UNDETERMINED,
        (geo.SegmentCycleIndeterminacy.NUMERIC_BAND,),
    )
    facts = _BondRingAcceptanceFacts((unresolved, piercing), 'ligand_skeleton', 16, 2, 1, 1, 1, True)
    checks = rules._bond_ring_acceptance_checks(facts, (((0, 1), 9), ((1, 2), 3)))
    assert [(c.name, c.severity, c.bond_indices) for c in checks] == [
        ('bond_ring_piercing', 'error', (3,)), ('bond_ring_piercing', 'warning', (9,)),
        ('bond_ring_scope_coverage', 'warning', ()),
    ]
    assert checks[0].measured == (8, 9, 10)
    assert checks[1].measured == (geo.SegmentCycleIndeterminacy.NUMERIC_BAND.value,)
    with pytest.raises(KeyError):
        rules._bond_ring_acceptance_checks(facts, ())
    clear = replace(facts, actionable_findings=(), excluded_ring_count=0)
    assert rules._bond_ring_acceptance_checks(clear, ())[0].passed


def test_prepared_values_execute_without_any_carrier_capture(monkeypatch):
    from hotpot.cheminfo.adapters.structure import forcefields as capture

    def unavailable(*args, **kwargs):
        raise AssertionError('Pure rules must not capture carriers')

    for name in ('capture_topology', 'capture_atom_pair_acceptance_values',
                 'capture_bond_acceptance_values', 'capture_coordination_acceptance_values',
                 'capture_acceptance_bond_positions'):
        monkeypatch.setattr(capture, name, unavailable)
    coordinates = np.zeros((1, 3))
    reference = TopologyReference((), ())
    assert rules._topology_atom_count_checks(0, 0) == ()
    assert rules._topology_checks(reference, reference)[0].passed
    assert rules._forcefield_acceptance_checks(None, 'basic', LIMITS, 'candidate') == ()
    facts = _BondRingAcceptanceFacts((), 'full_graph', 16, 0, 0, 0, 0, True)
    assert rules._bond_ring_acceptance_checks(facts, ())[0].passed
    assert rules._coordinate_acceptance_checks(coordinates, 1)[1]
    assert rules._atom_pair_acceptance_checks((), 'basic', LIMITS)[0].passed
    assert rules._bond_geometry_acceptance_checks(coordinates, (), 'basic', LIMITS)[1] == 0.0
    assert rules._coordination_metrics(coordinates, (0,), (), ((0, 7),)) == ((7, 0, (), (), ()),)
