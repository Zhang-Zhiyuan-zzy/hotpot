import importlib.util
from pathlib import Path
import pytest

from mca.result_types import AtomPrediction, MoleculePrediction, SitePrediction
from mca.resources import gpu_window_atom_budget


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "plan"
    / "15_mca_gpu_pipeline"
    / "validate_gpu_pipeline.py"
)
SPEC = importlib.util.spec_from_file_location("validate_gpu_pipeline", SCRIPT_PATH)
validation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validation)


def _prediction(*, site=True, site_type="amine", atom_mca=2.5, site_mca=2.5):
    sites = (
        (SitePrediction(1, "N", site_type, site_mca),)
        if site
        else ()
    )
    return MoleculePrediction(
        smiles="CN",
        formal_charge=0,
        sites=sites,
        model_variant="fp16",
        atom_predictions=(
            AtomPrediction(0, "C", 1.25),
            AtomPrediction(1, "N", atom_mca),
        ),
    )


def _joint_hash(prediction):
    return validation.build_validation_hashes([prediction])["joint_sha256"]


def test_historical_atom_and_site_hashes_are_unchanged():
    hashes = validation.build_validation_hashes([_prediction()])

    assert hashes["mca_sha256"] == (
        "a722883996ae7168467418a173f6a3314c988e6993869f14c90e3262439355ee"
    )
    assert hashes["sites_sha256"] == (
        "46c519027f794cf13e271f707f779466be19eb209ffb13bc71fa31bcbddf14de"
    )


def test_joint_hash_covers_site_boolean_type_and_atom_value():
    baseline = _joint_hash(_prediction())

    assert _joint_hash(_prediction(site=False)) != baseline
    assert _joint_hash(_prediction(site_type="amide")) != baseline
    assert _joint_hash(_prediction(atom_mca=2.75, site_mca=2.75)) != baseline


def test_joint_hash_rejects_site_mca_not_equal_to_atom_mca():
    with pytest.raises(RuntimeError, match="does not exactly match atom MCA"):
        _joint_hash(_prediction(site_mca=2.5000000000000004))


@pytest.mark.parametrize(
    ("sites", "message"),
    [
        (
            (
                SitePrediction(1, "N", "amine", 2.5),
                SitePrediction(1, "N", "amide", 2.5),
            ),
            "duplicate site atom index",
        ),
        ((SitePrediction(2, "N", "amine", 2.5),), "out-of-range"),
        ((SitePrediction(-1, "N", "amine", 2.5),), "out-of-range"),
    ],
)
def test_joint_hash_rejects_duplicate_and_out_of_range_sites(sites, message):
    prediction = _prediction()
    invalid = MoleculePrediction(
        smiles=prediction.smiles,
        formal_charge=prediction.formal_charge,
        sites=sites,
        model_variant=prediction.model_variant,
        atom_predictions=prediction.atom_predictions,
    )

    with pytest.raises(RuntimeError, match=message):
        _joint_hash(invalid)


def test_cross_capacity_requires_atom_count_strictly_above_budget():
    budget = gpu_window_atom_budget(2048, 16)

    assert budget == 8192
    assert validation.admission_capacity_cycle_count((6486,), budget) == 1
    assert validation.admission_capacity_cycle_count((8192,), budget) == 1
    assert validation.admission_capacity_cycle_count((8192, 1), budget) == 2
    assert validation.admission_capacity_cycle_count((4096, 4096, 1), budget) == 2
