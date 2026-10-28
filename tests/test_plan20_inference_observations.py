"""Failures and scientific values remain separate in inference observations."""

from importlib import import_module
import json
from types import SimpleNamespace

import numpy as np


_capture = import_module("plan.20_fact_rule_architecture.capture_inference_observations")


def _entry(count=3):
    return {"parameters": {"molecules": count, "smiles": ["CCO"], "seed": 42,
                           "backend": "cpu", "cpu_count": 1}}


def _prediction(value=1.0):
    payload = {"smiles": "CCO", "formal_charge": 0, "model_variant": "test",
               "atom_predictions": [{"atom_index": 2, "element": "O", "mca_kj_mol": value},
                                    {"atom_index": 0, "element": "C", "mca_kj_mol": value}],
               "sites": [{"atom_index": 2, "element": "O", "site_type": "O", "mca_kj_mol": value}]}
    return SimpleNamespace(to_dict=lambda: payload)


class _Predictor:
    cpu_ids = (0,)
    closed = False

    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        type(self).closed = True


def test_early_mca_failure_retains_returned_and_unreturned_inputs(tmp_path, monkeypatch):
    class Predictor(_Predictor):
        def iter_predict(self, inputs):
            next(inputs)
            yield _prediction()
            next(inputs)
            raise RuntimeError("second input failed")

    monkeypatch.setattr(_capture.importlib, "import_module", lambda name: SimpleNamespace(MCACpuPredictor=Predictor))
    result = _capture._mca(_entry(), "test_module", tmp_path)
    assert [r["status"] for r in result["molecules"]] == ["observed", "not_returned", "not_returned"]
    assert [r["input_consumed"] for r in result["molecules"]] == [True, True, False]
    assert result["molecules"][0]["is_Nuc_site"] == [True, False]
    assert result["molecules"][0]["ordered_atom_identity"] == [[2, "O"], [0, "C"]]
    assert result["errors"][0]["message"] == "second input failed"
    assert len(list((tmp_path / "molecules").glob("*.json"))) == 3
    assert Predictor.closed


def test_mca_extra_outputs_and_cleanup_errors_are_not_lost(tmp_path, monkeypatch):
    class Predictor(_Predictor):
        def iter_predict(self, inputs):
            next(inputs)
            yield _prediction()
            yield _prediction()

        def __exit__(self, *args):
            raise RuntimeError("cleanup failed")

    monkeypatch.setattr(_capture.importlib, "import_module", lambda name: SimpleNamespace(MCACpuPredictor=Predictor))
    result = _capture._mca(_entry(1), "test_module", tmp_path)
    assert len(result["unexpected_outputs"]) == 1
    assert [e["phase"] for e in result["errors"]] == ["output_completeness", "predictor_close"]
    assert result["molecules"][0]["status"] == "observed"


def test_nonfinite_mca_value_is_an_observation_not_execution_failure(tmp_path, monkeypatch):
    class Predictor(_Predictor):
        def iter_predict(self, inputs):
            next(inputs)
            yield _prediction(float("nan"))

    monkeypatch.setattr(_capture.importlib, "import_module", lambda name: SimpleNamespace(MCACpuPredictor=Predictor))
    result = _capture._mca(_entry(1), "test_module", tmp_path)
    assert not result["errors"]
    row = json.loads((tmp_path / "molecules/000000.json").read_text())
    assert row["status"] == "observed"
    assert row["prediction"]["atom_predictions"][0]["mca_kj_mol"] == {"nonfinite": "nan"}


def _runtime(tmp_path, predict):
    (tmp_path / "manifest.json").write_text("{}")
    session = SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"], get_provider_options=lambda: {})
    return SimpleNamespace(device="cpu", model_dir=tmp_path, graph_session=session, cbond_session=session,
                           embed_graph=lambda atoms, edges: np.zeros((4, 128)), predict=predict)


def test_cbond_failed_head_retains_graph_and_explicit_later_absences(tmp_path, monkeypatch):
    def fail(*args):
        raise RuntimeError("head unavailable")

    runtime = _runtime(tmp_path, fail)
    monkeypatch.setattr(_capture.importlib, "import_module", lambda name: SimpleNamespace(CBondRuntime=lambda **kwargs: runtime))
    result = _capture._cbond({"parameters": {"device": "cpu", "loops": 3}}, "test_module", tmp_path / "report")
    assert [r["status"] for r in result["calls"]] == ["execution_error", "not_run", "not_run"]
    assert result["calls"][0]["errors"][0]["phase"] == "cbond_prediction"
    assert set(result["calls"][0]["artifacts"]) == {"xg", "padded_rings"}
    assert (tmp_path / "report/call-000/xg.npy").exists()
    assert result["sessions"]["cbond"]["providers"] == ["CPUExecutionProvider"]


def test_cbond_nonfinite_predictions_are_saved_with_finite_monitor(tmp_path, monkeypatch):
    runtime = _runtime(tmp_path, lambda *args: np.asarray([np.inf, 0.0], dtype=np.float32))
    monkeypatch.setattr(_capture.importlib, "import_module", lambda name: SimpleNamespace(CBondRuntime=lambda **kwargs: runtime))
    result = _capture._cbond({"parameters": {"device": "cpu", "loops": 1}}, "test_module", tmp_path / "report")
    call = result["calls"][0]
    assert not result["errors"] and not call["errors"]
    assert call["status"] == "observed"
    assert not call["artifacts"]["predictions"]["all_finite"]
    assert np.isinf(np.load(tmp_path / "report/call-000/predictions.npy")[0])
