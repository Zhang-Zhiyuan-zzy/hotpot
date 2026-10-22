from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import numpy as np
from rdkit import Chem

from mca.conformer import ensure_3d_conformer
from mca.featurizer import mol_to_unimolv2
from mca.pipeline import build_prediction
from mca.serialization import normalize_molecule, rdmol_from_dto
from mca.workers import (
    CPUPreprocessTask,
    FeatureTask,
    SiteDetectionTask,
    build_features_worker,
    cpu_preprocess_worker,
    detect_sites_worker,
)


def test_site_worker_returns_global_index_and_ordered_site_records():
    task = SiteDetectionTask(
        molecule_index=17,
        molecule=normalize_molecule("CC(=O)C"),
    )

    result = detect_sites_worker(task)

    assert result.molecule_index == 17
    assert result.sites == (
        (2, "Ketone"),
        (1, "double_bond"),
    )


def test_feature_worker_matches_the_existing_conformer_and_featurizer_path():
    dto = normalize_molecule(Chem.MolFromSmiles("C[C@H](O)F"))
    task = FeatureTask(
        molecule_index=23,
        molecule=dto,
        max_atoms=512,
        conformer_seed=73,
    )
    expected = mol_to_unimolv2(
        ensure_3d_conformer(rdmol_from_dto(dto), seed=73),
        max_atoms=512,
    )

    result = build_features_worker(task)

    assert result.molecule_index == 23
    assert result.feature.keys() == expected.keys()
    for name, array in result.feature.items():
        assert np.array_equal(array, expected[name]), name


def test_worker_entry_points_execute_under_spawn():
    dto = normalize_molecule("CCN")
    site_task = SiteDetectionTask(molecule_index=31, molecule=dto)
    feature_task = FeatureTask(
        molecule_index=31,
        molecule=dto,
        max_atoms=512,
        conformer_seed=42,
    )
    context = multiprocessing.get_context("spawn")

    with ProcessPoolExecutor(max_workers=1, mp_context=context) as executor:
        site_result = executor.submit(detect_sites_worker, site_task).result()
        feature_result = executor.submit(build_features_worker, feature_task).result()

    assert site_result.molecule_index == 31
    assert site_result.sites == ((2, "Amine"),)
    assert feature_result.molecule_index == 31
    assert feature_result.feature["atom_feat"].shape == (3, 8)


def test_fused_cpu_worker_preserves_site_order_and_feature_path():
    dto = normalize_molecule("CC(=O)C")
    result = cpu_preprocess_worker(
        CPUPreprocessTask(
            molecule_index=41,
            molecule=dto,
            sites=None,
            max_atoms=512,
            conformer_seed=42,
        )
    )

    assert result.molecule_index == 41
    assert result.molecule is dto
    assert result.sites == ((2, "Ketone"), (1, "double_bond"))
    assert result.feature["atom_feat"].shape == (4, 8)


def test_fused_cpu_worker_reuses_precomputed_sites_under_spawn():
    dto = normalize_molecule("CCN")
    task = CPUPreprocessTask(
        molecule_index=43,
        molecule=dto,
        sites=((0, "preserved"),),
        max_atoms=512,
        conformer_seed=42,
    )
    context = multiprocessing.get_context("spawn")

    with ProcessPoolExecutor(max_workers=1, mp_context=context) as executor:
        result = executor.submit(cpu_preprocess_worker, task).result()

    assert result.sites == ((0, "preserved"),)
    assert result.feature["atom_feat"].shape == (3, 8)


def test_shared_result_builder_joins_sites_to_the_same_atom_values():
    prediction = build_prediction(
        normalize_molecule("CCN"),
        ((2, "Amine"),),
        (101.0, 202.0, 303.0),
        "fp16",
    )

    site_indices = {site.atom_index for site in prediction.sites}
    assert [
        atom.atom_index in site_indices for atom in prediction.atom_predictions
    ] == [False, False, True]
    assert prediction.sites[0].mca_kj_mol == (
        prediction.atom_predictions[prediction.sites[0].atom_index].mca_kj_mol
    )
