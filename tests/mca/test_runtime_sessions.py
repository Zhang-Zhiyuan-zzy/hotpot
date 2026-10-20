from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
from mca import runtime as runtime_module


class _Input:
    def __init__(self, name, type_="tensor(float)"):
        self.name = name
        self.type = type_


def _install_fake_model(monkeypatch, input_names=("atom_mask",)):
    captured = {"resolve_count": 0, "materialize_count": 0}

    class SessionOptions:
        pass

    class ExecutionMode:
        ORT_SEQUENTIAL = object()

    class Store:
        def __init__(self, **kwargs):
            self.manifest = {
                "max_atoms": 256,
                "models": {"fp16": {"external_data": None}},
            }

        def resolve(self, variant=None):
            captured["resolve_count"] += 1
            return "/model.onnx", variant or "fp16"

    class Session:
        def __init__(self, path, sess_options, providers):
            captured.update(
                path=path,
                options=sess_options,
                providers=providers,
            )
            self._providers = [
                provider[0] if isinstance(provider, tuple) else provider
                for provider in providers
            ]

        def get_providers(self):
            return self._providers

        def get_inputs(self):
            return [_Input(name) for name in input_names]

        def run(self, outputs, feed):
            captured.setdefault("feeds", []).append(feed)
            rows, width = feed["atom_mask"].shape
            return [np.zeros((rows, width), dtype=np.float32)]

    monkeypatch.setattr(runtime_module, "ModelStore", Store)
    def materialize(path, digest):
        captured["materialize_count"] += 1
        return nullcontext(path)

    monkeypatch.setattr(runtime_module, "materialize_all_atom_model", materialize)
    fake_ort = SimpleNamespace(
        ExecutionMode=ExecutionMode,
        InferenceSession=Session,
        SessionOptions=SessionOptions,
        get_available_providers=lambda: ["CPUExecutionProvider"],
    )
    monkeypatch.setattr(runtime_module, "_onnxruntime", lambda: fake_ort)
    captured["ort"] = fake_ort
    return captured


def test_cpu_session_has_explicit_bounded_sequential_threads(monkeypatch):
    captured = _install_fake_model(monkeypatch)
    monkeypatch.setattr(
        runtime_module.os,
        "sched_getaffinity",
        lambda pid: set(range(40)),
    )

    runtime = runtime_module.create_cpu_runtime()

    assert captured["providers"] == ["CPUExecutionProvider"]
    assert captured["options"].intra_op_num_threads == 16
    assert captured["options"].inter_op_num_threads == 1
    assert (
        captured["options"].execution_mode
        == captured["ort"].ExecutionMode.ORT_SEQUENTIAL
    )
    assert "atom_index" not in runtime.input_dtypes


def test_cuda_session_owns_explicit_device_and_single_ort_thread(monkeypatch):
    captured = _install_fake_model(monkeypatch)
    captured["ort"].get_available_providers = lambda: [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]

    runtime = runtime_module.create_cuda_runtime(device_id=3)

    assert captured["providers"] == [
        ("CUDAExecutionProvider", {"device_id": 3}),
        "CPUExecutionProvider",
    ]
    assert captured["options"].intra_op_num_threads == 1
    assert captured["options"].inter_op_num_threads == 1
    assert (
        captured["options"].execution_mode
        == captured["ort"].ExecutionMode.ORT_SEQUENTIAL
    )
    assert "CUDAExecutionProvider" in runtime.session.get_providers()
    assert captured["resolve_count"] == 1
    assert captured["materialize_count"] == 1


def test_cuda_session_from_resolved_artifact_skips_store_and_materialization(
    monkeypatch,
):
    captured = _install_fake_model(monkeypatch)
    captured["ort"].get_available_providers = lambda: [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    monkeypatch.setattr(
        runtime_module,
        "ModelStore",
        lambda **kwargs: pytest.fail("ModelStore must not be called"),
    )
    monkeypatch.setattr(
        runtime_module,
        "materialize_all_atom_model",
        lambda *args: pytest.fail("materialization must not be called"),
    )

    runtime = runtime_module.create_cuda_runtime_from_artifact(
        session_path="/resolved/all-atom.onnx",
        variant="fp16",
        max_atoms=512,
        device_id=2,
    )

    assert captured["path"] == "/resolved/all-atom.onnx"
    assert captured["providers"][0] == (
        "CUDAExecutionProvider",
        {"device_id": 2},
    )
    assert runtime.manifest == {"max_atoms": 512}


def test_runtime_rejects_selected_atom_model(monkeypatch):
    _install_fake_model(monkeypatch, input_names=("atom_mask", "atom_index"))

    with pytest.raises(RuntimeError, match="required all-atom layout"):
        runtime_module.create_cpu_runtime()


def test_runtime_keeps_onnx_batching_and_input_dtypes(monkeypatch):
    captured = _install_fake_model(monkeypatch)
    runtime = runtime_module.create_cpu_runtime()
    arrays = {"atom_mask": np.ones((5, 7), dtype=np.int32)}

    output = runtime.predict(arrays, batch_size=2)

    assert [feed["atom_mask"].shape[0] for feed in captured["feeds"]] == [2, 2, 1]
    assert all(feed["atom_mask"].dtype == np.float32 for feed in captured["feeds"])
    assert output.shape == (5, 7)
    assert output.dtype == np.float64
