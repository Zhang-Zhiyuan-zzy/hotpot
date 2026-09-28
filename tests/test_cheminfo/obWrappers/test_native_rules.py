"""Scientific boundary tests for the native Open Babel rule planner."""

from __future__ import annotations

from math import sin

import numpy as np
import pytest

from hotpot.cheminfo.obWrappers import _ob_rules


PRE_BUILD = _ob_rules.RuleStage.PRE_BUILD
PRE_FORCEFIELD_SETUP = _ob_rules.RuleStage.PRE_FORCEFIELD_SETUP


def _atom(
    atomic_number: int,
    *,
    charge: int = 0,
    hybridization: int = 3,
    metal: bool = False,
):
    return _ob_rules.AtomSnapshot(
        atomic_number,
        charge,
        hybridization,
        metal,
    )


def _bond(
    begin: int,
    end: int,
    order: int = 1,
    *,
    aromatic: bool = False,
):
    return _ob_rules.BondSnapshot(begin, end, order, aromatic)


def _phosphorus_environment(
    *,
    phosphorus_charge: int = 0,
    phosphorus_hybridization: int = 5,
    double_bond_element: int = 8,
    double_bond_order: int = 2,
    single_neighbour_count: int = 3,
):
    atoms = [
        _atom(
            15,
            charge=phosphorus_charge,
            hybridization=phosphorus_hybridization,
        ),
        _atom(double_bond_element, charge=-1 if phosphorus_charge else 0),
    ]
    atoms.extend(_atom(6) for _ in range(single_neighbour_count))
    bonds = [_bond(0, 1, double_bond_order)]
    bonds.extend(_bond(0, index) for index in range(2, len(atoms)))
    return atoms, bonds


@pytest.mark.parametrize("chalcogen", (8, 16), ids=("phosphoryl", "thiophosphoryl"))
def test_neutral_tetracoordinate_pv_uses_tetrahedral_build_rule(chalcogen):
    atoms, bonds = _phosphorus_environment(double_bond_element=chalcogen)

    plan = _ob_rules.plan_build(atoms, bonds)

    assert plan.stage == PRE_BUILD
    assert len(plan.applications) == 1
    application = plan.applications[0]
    assert application.stage == PRE_BUILD
    assert application.atom_indices == [0]
    assert application.coordinate_changes == []
    assert len(application.hybridization_changes) == 1
    change = application.hybridization_changes[0]
    assert (change.atom_index, change.before, change.after) == (0, 5, 3)


@pytest.mark.parametrize(
    ("atoms", "bonds"),
    (
        (
            [_atom(15), _atom(6), _atom(6), _atom(6)],
            [_bond(0, 1), _bond(0, 2), _bond(0, 3)],
        ),
        (
            [_atom(15, charge=1), *(_atom(6) for _ in range(4))],
            [_bond(0, index) for index in range(1, 5)],
        ),
        (
            [_atom(15, hybridization=5), *(_atom(9) for _ in range(5))],
            [_bond(0, index) for index in range(1, 6)],
        ),
        (
            [
                _atom(15, charge=1),
                _atom(8, charge=-1),
                *(_atom(6) for _ in range(3)),
            ],
            [_bond(0, index) for index in range(1, 5)],
        ),
        _phosphorus_environment(double_bond_element=6),
    ),
    ids=(
        "trivalent_phosphorus",
        "phosphonium",
        "pentacoordinate_phosphorane",
        "charge_separated_phosphoryl",
        "phosphorus_carbon_double_bond",
    ),
)
def test_non_target_phosphorus_environments_do_not_change_build_plan(atoms, bonds):
    plan = _ob_rules.plan_build(atoms, bonds)

    assert plan.stage == PRE_BUILD
    assert plan.applications == []


def _degenerate_torsion_environment(*, degree: int = 3, metal: bool = False):
    atoms = [
        _atom(6),
        _atom(7 if not metal else 63, hybridization=3, metal=metal),
        _atom(6),
        _atom(6),
        _atom(6),
    ]
    coordinates = [
        (-1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
    ]
    bonds = [_bond(0, 1), _bond(1, 2), _bond(2, 3), _bond(1, 4)]
    if degree == 4:
        atoms.append(_atom(6))
        coordinates.append((0.0, 0.0, 1.0))
        bonds.append(_bond(1, 5))
    return atoms, bonds, coordinates


def _normalized_sine(coordinates, first: int, center: int, last: int) -> float:
    xyz = np.asarray(coordinates, dtype=float)
    left = xyz[first] - xyz[center]
    right = xyz[last] - xyz[center]
    return float(
        np.linalg.norm(np.cross(left, right))
        / (np.linalg.norm(left) * np.linalg.norm(right))
    )


def _coordinates_after(plan, coordinates):
    repaired = np.asarray(coordinates, dtype=float).copy()
    for application in plan.applications:
        for change in application.coordinate_changes:
            repaired[change.atom_index] = change.after
    return repaired


@pytest.mark.parametrize("degree", (3, 4))
def test_degenerate_branched_center_with_proper_torsion_is_repaired(degree):
    atoms, bonds, coordinates = _degenerate_torsion_environment(degree=degree)
    threshold = 1.0e-6
    repair_angle = 0.1

    plan = _ob_rules.plan_optimization(
        atoms,
        bonds,
        coordinates,
        threshold,
        repair_angle,
    )

    assert plan.stage == PRE_FORCEFIELD_SETUP
    assert len(plan.applications) == 1
    application = plan.applications[0]
    assert application.stage == PRE_FORCEFIELD_SETUP
    assert application.metric_before == pytest.approx(0.0, abs=1.0e-15)
    assert application.hybridization_changes == []
    assert application.coordinate_changes
    repaired = _coordinates_after(plan, coordinates)
    before = _normalized_sine(coordinates, 0, 1, 2)
    after = _normalized_sine(repaired, 0, 1, 2)
    assert after > before
    assert after > threshold
    assert after == pytest.approx(sin(repair_angle), rel=1.0e-6, abs=1.0e-9)

    edge_indices = [(0, 1), (1, 2), (2, 3), (1, 4)]
    if degree == 4:
        edge_indices.append((1, 5))
    original_lengths = sorted(
        np.linalg.norm(np.asarray(coordinates[a]) - np.asarray(coordinates[b]))
        for a, b in edge_indices
    )
    repaired_lengths = sorted(
        np.linalg.norm(repaired[a] - repaired[b])
        for a, b in edge_indices
    )
    assert repaired_lengths == pytest.approx(original_lengths, abs=1.0e-12)


def test_true_sp_center_with_proper_torsion_is_not_repaired():
    atoms = [
        _atom(7, hybridization=1),
        _atom(6, hybridization=1),
        _atom(6),
        _atom(6),
    ]
    bonds = [_bond(0, 1, 3), _bond(1, 2), _bond(2, 3)]
    coordinates = [
        (-1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
    ]

    plan = _ob_rules.plan_optimization(atoms, bonds, coordinates, 1.0e-6, 0.1)

    assert plan.applications == []


def test_degenerate_metal_center_is_not_repaired():
    atoms, bonds, coordinates = _degenerate_torsion_environment(metal=True)

    plan = _ob_rules.plan_optimization(atoms, bonds, coordinates, 1.0e-6, 0.1)

    assert plan.applications == []


def test_non_degenerate_center_is_not_repaired():
    atoms, bonds, coordinates = _degenerate_torsion_environment()
    coordinates[0] = (-0.5, 0.8660254037844386, 0.0)

    plan = _ob_rules.plan_optimization(atoms, bonds, coordinates, 1.0e-6, 0.1)

    assert plan.applications == []


def test_degenerate_center_without_proper_torsion_is_not_repaired():
    atoms = [_atom(6), _atom(7), _atom(6), _atom(6)]
    bonds = [_bond(0, 1), _bond(1, 2), _bond(1, 3)]
    coordinates = [
        (-1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    ]

    plan = _ob_rules.plan_optimization(atoms, bonds, coordinates, 1.0e-6, 0.1)

    assert plan.applications == []


def test_plans_are_deterministic():
    atoms, bonds, coordinates = _degenerate_torsion_environment(degree=4)

    first = _ob_rules.plan_optimization(atoms, bonds, coordinates, 1.0e-6, 0.1)
    second = _ob_rules.plan_optimization(atoms, bonds, coordinates, 1.0e-6, 0.1)

    first_changes = [
        (change.atom_index, tuple(change.before), tuple(change.after))
        for application in first.applications
        for change in application.coordinate_changes
    ]
    second_changes = [
        (change.atom_index, tuple(change.before), tuple(change.after))
        for application in second.applications
        for change in application.coordinate_changes
    ]
    assert first_changes == second_changes


def test_all_degenerate_pairs_at_one_center_are_repaired():
    atoms = [_atom(6) for _ in range(7)]
    bonds = [
        _bond(0, 1),
        _bond(0, 2),
        _bond(0, 3),
        _bond(0, 4),
        _bond(1, 5),
        _bond(3, 6),
    ]
    coordinates = [
        (0.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 1.0, 0.0),
        (-2.0, 0.0, 0.0),
        (0.0, -2.0, 0.0),
    ]

    plan = _ob_rules.plan_optimization(
        atoms,
        bonds,
        coordinates,
        1.0e-6,
        0.1,
    )
    repaired = _coordinates_after(plan, coordinates)

    assert len(plan.applications) == 2
    assert _normalized_sine(repaired, 1, 0, 2) > 1.0e-6
    assert _normalized_sine(repaired, 3, 0, 4) > 1.0e-6


def test_repair_angle_must_escape_the_singularity_threshold():
    atoms, bonds, coordinates = _degenerate_torsion_environment()

    with pytest.raises(ValueError, match="must move the normalized angle sine"):
        _ob_rules.plan_optimization(atoms, bonds, coordinates, 0.1, 0.01)


def test_rule_registry_is_stable_ordered_and_stage_filterable():
    first = _ob_rules.available_rules()
    second = _ob_rules.available_rules()

    def identity(rule):
        return rule.rule_id, rule.version, rule.stage, rule.priority

    assert [identity(rule) for rule in first] == [identity(rule) for rule in second]
    assert len({rule.rule_id for rule in first}) == len(first)
    assert all(rule.rule_id and rule.version for rule in first)
    for stage in (PRE_BUILD, PRE_FORCEFIELD_SETUP):
        stage_rules = _ob_rules.available_rules(stage)
        assert stage_rules
        assert all(rule.stage == stage for rule in stage_rules)
        assert [rule.priority for rule in stage_rules] == sorted(
            rule.priority for rule in stage_rules
        )
        assert [identity(rule) for rule in stage_rules] == [
            identity(rule) for rule in first if rule.stage == stage
        ]

    atoms, bonds = _phosphorus_environment()
    application = _ob_rules.plan_build(atoms, bonds).applications[0]
    matching_descriptors = [
        descriptor
        for descriptor in first
        if descriptor.rule_id == application.rule_id
    ]
    assert len(matching_descriptors) == 1
    descriptor = matching_descriptors[0]
    assert identity(descriptor) == (
        application.rule_id,
        application.version,
        application.stage,
        application.priority,
    )


def test_native_rule_records_are_read_only():
    atoms, bonds = _phosphorus_environment()
    plan = _ob_rules.plan_build(atoms, bonds)
    application = plan.applications[0]
    change = application.hybridization_changes[0]

    with pytest.raises(AttributeError):
        plan.applications = []
    with pytest.raises(AttributeError):
        application.priority = -1
    with pytest.raises(AttributeError):
        change.after = 5
