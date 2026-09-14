"""ONNX Runtime session and batched site inference."""

from __future__ import annotations

import numpy as np
import onnxruntime as ort

from .model_store import ModelStore


ORT_DTYPES = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
}


class MCARuntime:
    def __init__(
        self,
        model_dir=None,
        device: str = "auto",
        variant: str | None = None,
        verify_model: bool = True,
    ):
        store = ModelStore(model_dir=model_dir, verify=verify_model)
        path, self.variant, providers, resolved_device = store.resolve(device, variant)
        self.manifest = store.manifest
        options = ort.SessionOptions()
        options.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=providers
        )
        if device == "cuda" and "CUDAExecutionProvider" not in self.session.get_providers():
            raise RuntimeError("The CUDA execution provider could not be initialized")
        self.device = (
            "cuda" if "CUDAExecutionProvider" in self.session.get_providers() else "cpu"
        )
        self.requested_device = resolved_device
        self.input_dtypes = {
            input_.name: ORT_DTYPES[input_.type] for input_ in self.session.get_inputs()
        }

    def predict(self, arrays, batch_size: int = 64):
        total = arrays["atom_index"].shape[0]
        predictions = []
        for start in range(0, total, batch_size):
            stop = min(start + batch_size, total)
            feed = {
                name: arrays[name][start:stop].astype(dtype, copy=False)
                for name, dtype in self.input_dtypes.items()
            }
            predictions.append(self.session.run(["mca_kj_mol"], feed)[0].reshape(-1))
        return np.concatenate(predictions).astype(np.float64, copy=False)
