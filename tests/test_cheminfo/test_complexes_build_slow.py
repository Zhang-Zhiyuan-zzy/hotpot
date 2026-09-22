import time

import numpy as np
import pytest
from openbabel import openbabel as ob

import hotpot as hp

README_LIGAND_SMILES = (
    "O=C(N(C)CCC)C(C=C1)=NC2=C1C=CC3=C2N=C("
    "C4=NC(C(C)(C)CCC5(C)C)=C5N=N4)C=C3"
)


@pytest.mark.slow
def test_readme_europium_complex_full_standard_workflow(record_testsuite_property):
    ligand = hp.read_mol(README_LIGAND_SMILES)
    pair = ligand.auto_pair_metal("Eu")
    original_atom_objects = {id(atom) for atom in pair.atoms}
    original_bonds = {
        (
            frozenset((bond.atom1.id, bond.atom2.id)),
            bond.bond_order,
            bond.bond_kind,
        )
        for bond in pair.bonds
    }

    started = time.monotonic()
    report = pair.build3d(
        seed=20260916,
        max_attempts=20,
        candidate_warmup_steps=500,
        candidate_score_steps=1000,
        best_candidate_refine_steps=3000,
        epochs=20,
        steps_per_epoch=500,
        quality_level="standard",
    )
    elapsed_seconds = time.monotonic() - started

    assert report.quality_report.passed
    assert report.quality_report.failures == ()
    assert report.quality_report.metrics["bond_ring_piercing_count"] == 0
    assert report.optimization is not None
    optimization = report.optimization
    assert optimization.setup_succeeded
    assert not optimization.exploded
    assert optimization.energy_unit == "kJ/mol"
    assert optimization.gradient_unit == "kJ/(mol*angstrom)"
    assert np.all(np.isfinite((
        optimization.final_energy,
        optimization.best_energy,
        optimization.rms_gradient,
        optimization.max_gradient,
    )))
    assert pair.energy == pytest.approx(optimization.best_energy)

    assert original_atom_objects <= {id(atom) for atom in pair.atoms}
    completed_bonds = {
        (
            frozenset((bond.atom1.id, bond.atom2.id)),
            bond.bond_order,
            bond.bond_kind,
        )
        for bond in pair.bonds
    }
    assert original_bonds <= completed_bonds

    (environment,) = report.quality_report.metrics["coordination_environments"]
    assert environment["coordination_number"] == 4
    donor_atomic_numbers = tuple(
        pair.atoms[index].atomic_number for index in environment["donor_indices"]
    )
    assert sorted(donor_atomic_numbers) == [7, 7, 7, 8]
    assert sorted(environment["distances"]) == pytest.approx(
        (2.24, 2.28, 2.31, 2.37),
        abs=0.15,
    )
    assert len(environment["angles"]) == 6
    assert np.all(np.isfinite(environment["angles"]))
    assert all(0.0 < angle < 180.0 for angle in environment["angles"])

    record_testsuite_property("openbabel_version", ob.OBReleaseVersion())
    record_testsuite_property("runtime_seconds", elapsed_seconds)
    record_testsuite_property("best_energy_kj_per_mol", optimization.best_energy)
    record_testsuite_property(
        "eu_donor_distances_angstrom",
        tuple(sorted(environment["distances"])),
    )
