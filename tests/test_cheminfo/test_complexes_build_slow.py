import numpy as np
import pytest

import hotpot as hp

README_LIGAND_SMILES = (
    "O=C(N(C)CCC)C(C=C1)=NC2=C1C=CC3=C2N=C("
    "C4=NC(C(C)(C)CCC5(C)C)=C5N=N4)C=C3"
)


@pytest.mark.slow
def test_readme_europium_complex_full_standard_workflow():
    ligand = hp.read_mol(README_LIGAND_SMILES)
    pair = ligand.auto_pair_metal("Eu")

    report = pair.build3d(
        seed=20260916,
        candidate_count=1,
        max_attempts=3,
        candidate_warmup_steps=500,
        candidate_score_steps=1000,
        best_candidate_refine_steps=3000,
        epochs=20,
        steps_per_epoch=500,
        quality_level="standard",
    )

    assert report.quality_report.passed
    assert report.optimization is not None
    assert np.isfinite(report.optimization.final_energy)
    assert report.optimization.best_energy == pytest.approx(902.8939, abs=1.0)

    (environment,) = report.quality_report.metrics["coordination_environments"]
    assert environment["coordination_number"] == 4
    donor_atomic_numbers = tuple(
        pair.atoms[index].atomic_number for index in environment["donor_indices"]
    )
    assert sorted(donor_atomic_numbers) == [7, 7, 7, 8]
    assert environment["distances"] == pytest.approx(
        (2.2431, 2.2809, 2.3062, 2.3596),
        abs=0.10,
    )
