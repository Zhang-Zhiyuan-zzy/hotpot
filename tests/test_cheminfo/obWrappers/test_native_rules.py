"""Scientific boundary tests for rules applied to native ``OBMol`` values."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from math import sin

import numpy as np
import pytest

from hotpot import read_mol
from hotpot.cheminfo.core import Molecule
from hotpot.cheminfo.obWrappers import (
    RuleExecutionReport,
    RuleStage,
    available_rules,
    inspect_rules,
)


def _normalized_sine(
    coordinates: np.ndarray,
    first: int,
    center: int,
    last: int,
) -> float:
    left = coordinates[first] - coordinates[center]
    right = coordinates[last] - coordinates[center]
    return float(
        np.linalg.norm(np.cross(left, right))
        / (np.linalg.norm(left) * np.linalg.norm(right))
    )


def _coordinates_after(
    report: RuleExecutionReport,
    coordinates: np.ndarray,
) -> np.ndarray:
    repaired = coordinates.copy()
    for application in report.applications:
        for change in application.coordinate_changes:
            repaired[change.atom_index] = change.after
    return repaired


@pytest.mark.parametrize(
    "smiles",
    ("OP(=O)(O)O", "SP(=S)(SC)SC"),
    ids=("phosphoryl", "thiophosphoryl"),
)
def test_neutral_tetracoordinate_pv_uses_tetrahedral_build_rule(smiles):
    mol = read_mol(smiles)

    report = inspect_rules(mol, RuleStage.PRE_BUILD)

    assert len(report.applications) == 1
    application = report.applications[0]
    assert application.descriptor.rule_id == (
        "tetracoordinate_pentavalent_phosphorus_build"
    )
    assert application.atom_indices == (1,)
    assert application.coordinate_changes == ()
    assert len(application.hybridization_changes) == 1
    change = application.hybridization_changes[0]
    assert (change.atom_index, change.before, change.after) == (1, 5, 3)


@pytest.mark.parametrize(
    "smiles",
    (
        "P(C)(C)C",
        "[P+](C)(C)(C)C",
        "P(F)(F)(F)(F)F",
        "[P+](C)(C)(C)[O-]",
        "P(=C)(C)(C)C",
    ),
    ids=(
        "trivalent_phosphorus",
        "phosphonium",
        "pentacoordinate_phosphorane",
        "charge_separated_phosphoryl",
        "phosphorus_carbon_double_bond",
    ),
)
def test_non_target_phosphorus_environments_do_not_apply_build_rule(smiles):
    report = inspect_rules(read_mol(smiles), RuleStage.PRE_BUILD)

    assert report.applications == ()


def _degenerate_branched_molecule() -> tuple[Molecule, np.ndarray]:
    mol = read_mol("CCC(C)C")
    coordinates = np.asarray(
        [
            (-2.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        ]
    )
    mol.coordinates = coordinates
    return mol, coordinates


def test_degenerate_nonlinear_torsion_is_repaired_deterministically():
    outputs = []
    for _ in range(2):
        mol, coordinates = _degenerate_branched_molecule()
        report = inspect_rules(
            mol,
            RuleStage.PRE_FORCEFIELD_SETUP,
            repair_angle_radians=0.1,
        )
        repaired = _coordinates_after(report, coordinates)
        outputs.append(repaired)

        assert len(report.applications) == 1
        application = report.applications[0]
        assert application.descriptor.rule_id == "degenerate_nonlinear_torsion"
        assert application.metric_before == pytest.approx(0.0, abs=1.0e-15)
        assert _normalized_sine(repaired, 1, 2, 3) == pytest.approx(
            sin(0.1),
            rel=1.0e-6,
            abs=1.0e-9,
        )
        assert np.allclose(
            np.linalg.norm(repaired[[0, 1, 3, 4]] - repaired[2], axis=1),
            np.linalg.norm(
                coordinates[[0, 1, 3, 4]] - coordinates[2],
                axis=1,
            ),
        )

    assert np.array_equal(outputs[0], outputs[1])


@pytest.mark.parametrize(
    ("smiles", "coordinates"),
    (
        (
            "N#CCC",
            np.asarray(
                [(-1.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                 (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]
            ),
        ),
        (
            "C[Eu](C)C",
            np.asarray(
                [(-1.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                 (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
            ),
        ),
        (
            "CC(C)C",
            np.asarray(
                [(-1.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                 (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
            ),
        ),
    ),
    ids=("true_sp_center", "metal_center", "no_proper_torsion"),
)
def test_non_target_degenerate_centers_are_not_repaired(smiles, coordinates):
    mol = read_mol(smiles)
    mol.coordinates = coordinates

    report = inspect_rules(mol, RuleStage.PRE_FORCEFIELD_SETUP)

    assert report.applications == ()


def test_all_repairable_degenerate_pairs_at_one_center_are_reported():
    coordinates = np.asarray(
        [
            (0.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 1.0, 0.0),
            (-2.0, 0.0, 0.0),
            (0.0, -2.0, 0.0),
        ]
    )
    mol = Molecule()
    for coordinate in coordinates:
        mol.create_atom(atomic_number=6, coordinates=coordinate)
    for first, second in ((0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (3, 6)):
        mol.add_bond(first, second)

    report = inspect_rules(
        mol,
        RuleStage.PRE_FORCEFIELD_SETUP,
        repair_angle_radians=0.1,
    )
    repaired = _coordinates_after(report, coordinates)

    assert len(report.applications) == 2
    assert _normalized_sine(repaired, 1, 0, 2) > 1.0e-6
    assert _normalized_sine(repaired, 3, 0, 4) > 1.0e-6


def test_repair_angle_must_escape_the_singularity_threshold():
    mol, _ = _degenerate_branched_molecule()

    with pytest.raises(ValueError, match="must move the normalized angle sine"):
        inspect_rules(
            mol,
            RuleStage.PRE_FORCEFIELD_SETUP,
            singularity_threshold=0.1,
            repair_angle_radians=0.01,
        )


def test_rule_registry_is_stable_ordered_and_stage_filterable():
    first = available_rules()
    second = available_rules()

    assert first == second
    assert len({rule.rule_id for rule in first}) == len(first)
    assert all(rule.rule_id and rule.version for rule in first)
    for stage in RuleStage:
        stage_rules = available_rules(stage)
        assert stage_rules
        assert all(rule.stage is stage for rule in stage_rules)
        assert tuple(rule.priority for rule in stage_rules) == tuple(
            sorted(rule.priority for rule in stage_rules)
        )


def test_public_rule_records_are_read_only():
    report = inspect_rules(read_mol("OP(=O)(O)O"), RuleStage.PRE_BUILD)

    with pytest.raises(FrozenInstanceError):
        report.applications[0].metric_before = 1.0
