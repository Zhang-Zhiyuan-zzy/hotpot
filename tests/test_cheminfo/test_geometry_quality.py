import json
from copy import copy
from math import sqrt
from types import SimpleNamespace

import numpy as np
import pytest

from hotpot.cheminfo import geometry as geo
from hotpot.cheminfo.core import Molecule


def _molecule(coordinates, bonds=(), atomic_numbers=None):
    molecule = Molecule()
    if atomic_numbers is None:
        atomic_numbers = [6] * len(coordinates)
    for atomic_number, position in zip(atomic_numbers, coordinates):
        molecule.create_atom(
            atomic_number=atomic_number,
            coordinates=position,
        )
    for first, second in bonds:
        molecule.add_bond(first, second, bond_order=1.0)
    molecule.refresh_atom_id()
    return molecule


def _valid_carbon_bond():
    return _molecule(
        ((0.0, 0.0, 0.0), (1.52, 0.0, 0.0)),
        ((0, 1),),
    )


def _crossed_square():
    return _molecule(
        (
            (-1.0, -1.0, 0.0),
            (1.0, -1.0, 0.0),
            (1.0, 1.0, 0.0),
            (-1.0, 1.0, 0.0),
            (0.0, 0.0, -1.0),
            (0.0, 0.0, 1.0),
        ),
        ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5)),
    )


def _ring(coordinates):
    return _molecule(
        coordinates,
        tuple((index, (index + 1) % len(coordinates)) for index in range(len(coordinates))),
    )


def _ideal_complex(metal_atomic_number, donor_atomic_numbers, donor_coordinates):
    molecule = _molecule(
        ((0.0, 0.0, 0.0), *donor_coordinates),
        tuple((0, index) for index in range(1, len(donor_coordinates) + 1)),
        (metal_atomic_number, *donor_atomic_numbers),
    )
    return molecule


def test_off_level_still_rejects_bad_coordinate_shape_and_nonfinite_values():
    malformed = SimpleNamespace(
        atoms=(SimpleNamespace(coordinates=(0.0, 0.0, 0.0)),),
        bonds=(),
        coordinates=np.array((0.0, 0.0, 0.0)),
    )
    malformed_report = geo.evaluate_geometry_quality(malformed, level="off")
    assert not malformed_report.passed
    assert not next(
        check for check in malformed_report.checks
        if check.name == "coordinate_shape"
    ).passed

    molecule = _valid_carbon_bond()
    molecule.atoms[1].coordinates = (np.nan, 0.0, 0.0)
    nonfinite_report = geo.evaluate_geometry_quality(molecule, level="off")
    assert not nonfinite_report.passed
    finite_check = next(
        check for check in nonfinite_report.checks
        if check.name == "finite_coordinates"
    )
    assert finite_check.atom_indices == (1,)


def test_quality_partitions_overlap_from_too_close_pairs():
    molecule = _molecule(((0.0, 0.0, 0.0), (0.0005, 0.0, 0.0)))

    report = geo.evaluate_geometry_quality(molecule, level="basic")

    overlap_failures = [
        check for check in report.failures if check.name == "atom_overlap"
    ]
    close_failures = [
        check for check in report.failures if check.name == "atom_too_close"
    ]
    assert len(overlap_failures) == 1
    assert close_failures == []
    assert overlap_failures[0].measured == 0.0005
    assert overlap_failures[0].threshold == 0.001
    assert overlap_failures[0].atom_indices == (0, 1)


def test_basic_gate_rejects_an_exploded_explicit_bond():
    molecule = _molecule(
        ((0.0, 0.0, 0.0), (31.0, 0.0, 0.0)),
        ((0, 1),),
    )

    report = geo.evaluate_geometry_quality(molecule, level="basic")

    assert not report.passed
    failure = next(
        check for check in report.failures if check.name == "bond_distance"
    )
    assert failure.measured == 31.0
    assert failure.threshold == (0.0, 30.0)
    assert failure.atom_indices == (0, 1)
    assert failure.bond_indices == (0,)


def test_standard_gate_rejects_a_bond_crossing_a_ligand_ring():
    report = geo.evaluate_geometry_quality(_crossed_square(), level="standard")

    assert not report.passed
    failure = next(
        check for check in report.failures
        if check.name == "bond_ring_intersection"
    )
    assert failure.atom_indices == (4, 5)
    assert failure.bond_indices == (4,)


def test_standard_gate_accepts_a_sensible_small_molecule():
    molecule = _valid_carbon_bond()

    report = geo.evaluate_geometry_quality(molecule, level="standard")

    assert report.passed
    assert geo.is_geometry_reasonable(molecule, level="standard") == report.passed
    json.dumps(report.to_dict())


@pytest.mark.parametrize(
    "molecule",
    (
        _ring(tuple(
            (1.40 * np.cos(angle), 1.40 * np.sin(angle), 0.0)
            for angle in np.arange(6) * np.pi / 3.0
        )),
        _ring((
            (1.214, -0.700, 0.500),
            (0.000, -1.400, 0.000),
            (-1.214, -0.700, 0.500),
            (-1.214, 0.700, -0.500),
            (0.000, 1.400, 0.000),
            (1.214, 0.700, -0.500),
        )),
    ),
    ids=("benzene_skeleton", "cyclohexane_skeleton"),
)
def test_standard_gate_accepts_ideal_organic_ring_geometries(molecule):
    report = geo.evaluate_geometry_quality(molecule, level="standard")

    assert report.passed
    assert report.metrics["bond_ring_intersection_count"] == 0


@pytest.mark.parametrize(
    ("molecule", "expected_coordination_number"),
    (
        (
            _ideal_complex(30, (17, 17), ((-2.20, 0.0, 0.0), (2.20, 0.0, 0.0))),
            2,
        ),
        (
            _ideal_complex(
                30,
                (7, 7, 7, 7),
                tuple(
                    2.10 * np.asarray(vector) / sqrt(3.0)
                    for vector in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))
                ),
            ),
            4,
        ),
        (
            _ideal_complex(
                78,
                (17, 17, 17, 17),
                ((2.30, 0.0, 0.0), (0.0, 2.30, 0.0), (-2.30, 0.0, 0.0), (0.0, -2.30, 0.0)),
            ),
            4,
        ),
    ),
    ids=("zinc_chloride", "tetraamminezinc", "tetrachloroplatinum"),
)
def test_standard_gate_accepts_ideal_coordination_geometries(
    molecule,
    expected_coordination_number,
):
    report = geo.evaluate_geometry_quality(molecule, level="standard")

    assert report.passed
    (environment,) = report.metrics["coordination_environments"]
    assert environment["coordination_number"] == expected_coordination_number
    assert len(environment["donor_indices"]) == expected_coordination_number
    assert set(environment) == {
        "metal_index",
        "coordination_number",
        "donor_indices",
        "distances",
        "angles",
    }


def test_topology_reference_allows_only_appended_hydrogen_and_xh_bond():
    molecule = _valid_carbon_bond()
    reference = geo.capture_topology(molecule, allow_added_hydrogens=True)
    accepted = copy(molecule)
    hydrogen = accepted.create_atom(
        atomic_number=1,
        coordinates=(-1.0, 0.0, 0.0),
    )
    accepted.add_bond(accepted.atoms[0], hydrogen, bond_order=1.0)

    assert geo.evaluate_geometry_quality(
        accepted,
        level="off",
        topology_reference=reference,
    ).passed

    rejected = copy(molecule)
    oxygen = rejected.create_atom(
        atomic_number=8,
        coordinates=(-1.2, 0.0, 0.0),
    )
    rejected.add_bond(rejected.atoms[0], oxygen, bond_order=1.0)
    report = geo.evaluate_geometry_quality(
        rejected,
        level="off",
        topology_reference=reference,
    )
    assert not report.passed
    assert any(check.name == "topology_added_atoms" for check in report.failures)


def test_topology_reference_rejects_added_hydrogen_when_not_allowed():
    molecule = _valid_carbon_bond()
    reference = geo.capture_topology(molecule, allow_added_hydrogens=False)
    candidate = copy(molecule)
    hydrogen = candidate.create_atom(
        atomic_number=1,
        coordinates=(-1.0, 0.0, 0.0),
    )
    candidate.add_bond(candidate.atoms[0], hydrogen, bond_order=1.0)

    report = geo.evaluate_geometry_quality(
        candidate,
        level="off",
        topology_reference=reference,
    )

    assert not report.passed
    assert any(check.name == "topology_added_atoms" for check in report.failures)
    assert any(check.name == "topology_added_bond" for check in report.failures)


def test_topology_reference_rejects_original_bond_changes():
    molecule = _valid_carbon_bond()
    reference = geo.capture_topology(molecule)
    molecule.bonds[0].bond_order = 2.0

    report = geo.evaluate_geometry_quality(
        molecule,
        level="off",
        topology_reference=reference,
    )

    assert not report.passed
    assert any(check.name == "topology_original_bond" for check in report.failures)


def test_standard_warns_but_strict_fails_on_backend_nonconvergence():
    molecule = _valid_carbon_bond()
    forcefield_report = {
        "setup_succeeded": True,
        "converged": False,
        "final_energy": -10.0,
        "energy_unit": "kJ/mol",
        "rms_gradient": 2.0,
        "max_gradient": 6.0,
        "exploded": False,
    }

    standard = geo.evaluate_geometry_quality(
        molecule,
        level="standard",
        forcefield_report=forcefield_report,
    )
    strict = geo.evaluate_geometry_quality(
        molecule,
        level="strict",
        forcefield_report=forcefield_report,
    )

    assert standard.passed
    assert any(check.name == "forcefield_convergence" for check in standard.warnings)
    assert not strict.passed
    assert {
        check.name for check in strict.failures
    } >= {"forcefield_convergence", "rms_gradient", "max_gradient"}


def test_strict_gate_fails_closed_without_complete_forcefield_diagnostics():
    molecule = _valid_carbon_bond()

    missing_report = geo.evaluate_geometry_quality(molecule, level="strict")
    empty_report = geo.evaluate_geometry_quality(
        molecule,
        level="strict",
        forcefield_report={},
    )

    assert not missing_report.passed
    assert {check.name for check in missing_report.failures} == {
        "forcefield_report"
    }
    assert not empty_report.passed
    assert {
        "forcefield_setup",
        "finite_final_energy",
        "finite_rms_gradient",
        "finite_max_gradient",
        "backend_explosion",
        "forcefield_convergence",
        "energy_change",
        "max_displacement",
        "stability_observations",
    } <= {check.name for check in empty_report.failures}


def test_strict_gate_accepts_complete_stable_forcefield_diagnostics():
    molecule = _valid_carbon_bond()
    forcefield_report = {
        "setup_succeeded": True,
        "converged": True,
        "epochs_completed": 5,
        "final_energy": -10.0,
        "rms_gradient": 0.2,
        "max_gradient": 0.5,
        "exploded": False,
        "energy_changes": (1.0e-5,) * 5,
        "max_displacements": (1.0e-5,) * 5,
    }

    report = geo.evaluate_geometry_quality(
        molecule,
        level="strict",
        forcefield_report=forcefield_report,
    )

    assert report.passed


def test_strict_gate_accepts_a_definitively_converged_first_segment_frame():
    molecule = _valid_carbon_bond()
    forcefield_report = {
        "setup_succeeded": True,
        "converged": True,
        "epochs_completed": 1,
        "segment_epochs_completed": 1,
        "final_energy": -10.0,
        "rms_gradient": 0.2,
        "max_gradient": 0.5,
        "exploded": False,
        "energy_changes": (),
        "max_displacements": (),
    }

    report = geo.evaluate_geometry_quality(
        molecule,
        level="strict",
        forcefield_report=forcefield_report,
    )

    assert report.passed
    checks = {check.name: check for check in report.checks}
    assert checks["energy_change"].measured is None
    assert checks["max_displacement"].measured is None
    assert checks["stability_observations"].threshold == 0


def test_geometry_evaluation_does_not_change_structure_or_conformers():
    molecule = _crossed_square()
    molecule.conformer_add(molecule.coordinates.copy())
    coordinates = molecule.coordinates.copy()
    bonds = tuple(molecule.bonds)
    graph = molecule.graph
    graph_edges = tuple(molecule.graph.edges)
    conformers = molecule.conformers._coordinates.copy()
    rings_cache = molecule._rings
    ligand_rings_cache = molecule._ligand_rings
    ligand_rings_signature = molecule._ligand_rings_signature

    geo.evaluate_geometry_quality(molecule, level="standard")

    np.testing.assert_array_equal(molecule.coordinates, coordinates)
    assert tuple(molecule.bonds) == bonds
    assert molecule.graph is graph
    assert tuple(molecule.graph.edges) == graph_edges
    np.testing.assert_array_equal(molecule.conformers._coordinates, conformers)
    assert molecule._rings is rings_cache
    assert molecule._ligand_rings is ligand_rings_cache
    assert molecule._ligand_rings_signature is ligand_rings_signature
