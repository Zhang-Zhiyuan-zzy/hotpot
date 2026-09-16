from concurrent.futures import ThreadPoolExecutor

import numpy as np

from hotpot import read_mol
from hotpot.cheminfo import forcefields as ff


def _zinc_ethylene_diamine():
    molecule = read_mol("NCCN")
    zinc = molecule.create_atom(symbol="Zn")
    molecule.add_bond(zinc, molecule.atoms[0])
    molecule.add_bond(zinc, molecule.atoms[3])
    molecule.refresh_atom_id()
    return molecule


def test_real_openbabel_complex_proxy_and_full_system_optimization():
    molecule = _zinc_ethylene_diamine()
    original_bonds = {
        tuple(sorted((bond.a1idx, bond.a2idx))) for bond in molecule.bonds
    }

    report = ff.complexes_build(
        molecule,
        "MMFF94s",
        epochs=2,
        steps_per_epoch=20,
        candidate_count=1,
        max_attempts=3,
        candidate_warmup_steps=10,
        candidate_score_steps=10,
        best_candidate_refine_steps=10,
        quality_level="basic",
        timeout=30.0,
    )

    current_bonds = {
        tuple(sorted((bond.a1idx, bond.a2idx))) for bond in molecule.bonds
    }
    assert original_bonds <= current_bonds
    assert len(molecule.atoms) == 13
    assert len(molecule.hydrogens) == 8
    assert report.requested_forcefield == "MMFF94s"
    assert report.effective_forcefield == "UFF"
    assert report.optimization.energy_unit == "kJ/mol"
    assert np.isfinite(report.optimization.best_energy)
    assert report.quality_report.passed


def test_complex_worker_excludes_runtime_metadata_from_spawn_payload():
    molecule = _zinc_ethylene_diamine()

    def callback():
        return None

    molecule.properties["callback"] = callback
    molecule._model = callback

    report = ff.build_complex3d(
        molecule,
        candidate_count=1,
        max_attempts=3,
        candidate_warmup_steps=2,
        candidate_score_steps=2,
        best_candidate_refine_steps=2,
        timeout=30.0,
    )

    assert report.build.accepted_candidates == 1
    assert molecule.properties["callback"] is callback
    assert molecule._model is callback


def test_real_optimizer_preserves_custom_ids_and_existing_objects():
    molecule = read_mol("CCO", "smi")
    custom_ids = (101, 305, 902)
    for atom, atom_id in zip(molecule.atoms, custom_ids):
        atom.id = atom_id
    original_atoms = tuple(molecule.atoms)
    original_bonds = tuple(molecule.bonds)

    ff.optimize(
        molecule,
        "MMFF94s",
        epochs=1,
        steps_per_epoch=5,
        add_hydrogens=False,
        quality_level="off",
    )

    assert tuple(atom.id for atom in molecule.atoms) == custom_ids
    assert all(current is original for current, original in zip(molecule.atoms, original_atoms))
    assert all(current is original for current, original in zip(molecule.bonds, original_bonds))


def _optimize_ethanol(seed):
    molecule = read_mol("CCO", "smi")
    report = ff.build_and_optimize(
        molecule,
        epochs=1,
        steps_per_epoch=5,
        quality_level="basic",
        seed=seed,
    )
    return report.best_energy, molecule.coordinates


def test_openbabel_forcefield_singleton_is_serialized_across_threads():
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = tuple(executor.map(_optimize_ethanol, range(20)))

    assert all(np.isfinite(energy) for energy, _ in results)
    assert all(np.all(np.isfinite(coordinates)) for _, coordinates in results)


def test_real_openbabel_vdw_schedule_is_enabled_and_uses_final_potential():
    molecule = read_mol("CCO", "smi")

    report = ff.build_and_optimize(
        molecule,
        epochs=3,
        steps_per_epoch=5,
        quality_level="basic",
        increasing_vdw=True,
        vdw_cutoff_start=2.0,
        vdw_cutoff_end=8.0,
        save_movie=True,
        seed=41,
    )

    assert len(report.epoch_energies) == 3
    assert report.best_energy == min(report.epoch_energies)
    assert molecule.energy == report.best_energy
    assert np.all(np.isfinite(molecule.coordinates))


def test_real_first_epoch_convergence_can_pass_the_strict_gate():
    molecule = read_mol("CC", "smi")

    report = ff.build_and_optimize(
        molecule,
        forcefield="MMFF94s",
        epochs=1,
        steps_per_epoch=1000,
        quality_level="strict",
        seed=43,
    )

    assert report.converged
    assert report.quality_report.passed
    assert report.energy_changes == ()
    assert report.max_displacements == ()


def _build_zinc_amine(seed):
    molecule = read_mol("[Zn](N)", "smi")
    report = ff.build_complex3d(
        molecule,
        candidate_count=1,
        max_attempts=2,
        candidate_warmup_steps=2,
        candidate_score_steps=2,
        best_candidate_refine_steps=2,
        timeout=15.0,
        seed=seed,
    )
    return report.build.accepted_candidates, molecule.coordinates


def _distance_matrix(coordinates):
    return np.linalg.norm(
        coordinates[:, None, :] - coordinates[None, :, :],
        axis=2,
    )


def test_complex_build_workers_are_independent_when_called_concurrently():
    seeds = tuple(range(51, 71))
    with ThreadPoolExecutor(max_workers=8) as executor:
        first_results = tuple(executor.map(_build_zinc_amine, seeds))
    with ThreadPoolExecutor(max_workers=8) as executor:
        second_results = tuple(executor.map(_build_zinc_amine, seeds))

    for first, second in zip(first_results, second_results):
        assert first[0] == second[0] == 1
        assert np.all(np.isfinite(first[1]))
        assert np.all(np.isfinite(second[1]))
        np.testing.assert_allclose(
            _distance_matrix(first[1]),
            _distance_matrix(second[1]),
            rtol=0.0,
            atol=1e-12,
        )


def test_real_seeded_complex_build_is_reproducible():
    first_accepted, first_coordinates = _build_zinc_amine(71)
    second_accepted, second_coordinates = _build_zinc_amine(71)

    assert first_accepted == second_accepted == 1
    np.testing.assert_allclose(
        _distance_matrix(first_coordinates),
        _distance_matrix(second_coordinates),
        rtol=0.0,
        atol=1e-12,
    )
