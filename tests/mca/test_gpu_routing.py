import logging
from types import SimpleNamespace

import pytest

from hotpot.cheminfo.operations.inference.mca import gpu_backend as module
from hotpot.cheminfo.operations.inference.mca.gpu_backend import MCAGpuPredictor, MIN_GPU_SITE_COUNT
from hotpot.cheminfo.operations.inference.mca.pipeline import PreparedMolecule


def _prepared(value, site_count):
    molecule = SimpleNamespace(smiles=str(value), atom_count=1)
    sites = tuple((index, "test") for index in range(site_count))
    return PreparedMolecule(molecule=molecule, sites=sites)


def _patch_preparation(monkeypatch, site_counts):
    calls = []

    def prepare(value, position, allow_charged):
        calls.append((value, position, allow_charged))
        return _prepared(value, site_counts[value])

    monkeypatch.setattr(module, "prepare_molecule", prepare)
    return calls


class _CpuPredictor:
    instances = []

    def __init__(self, **options):
        self.options = options
        self.prepared = []
        self.closed = False
        self.__class__.instances.append(self)

    def _iter_predict_prepared(self, prepared):
        self.prepared.extend(prepared)
        return iter(("cpu-result",))

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


@pytest.fixture(autouse=True)
def _reset_cpu_predictors():
    _CpuPredictor.instances.clear()


def test_gpu_predictor_constructor_does_not_touch_cuda_affinity_or_memory(
    monkeypatch,
):
    def unexpected(*args, **kwargs):
        pytest.fail("GPU resources must be initialized lazily")

    monkeypatch.setattr(module, "cuda_provider_available", unexpected)
    monkeypatch.setattr(module, "build_gpu_resource_plan", unexpected)
    monkeypatch.setattr(module, "pin_current_process", unexpected)
    monkeypatch.setattr(module, "gpu_total_memories_bytes", unexpected)

    predictor = MCAGpuPredictor(gpu_count=1)

    assert predictor.resource_plan is None
    assert predictor._gpu_started is False
    predictor.close()


def test_less_than_minimum_sites_uses_temporary_cpu_without_starting_gpu(
    monkeypatch,
):
    calls = _patch_preparation(monkeypatch, {"few": MIN_GPU_SITE_COUNT - 1})
    monkeypatch.setattr(module, "MCACpuPredictor", _CpuPredictor)
    predictor = MCAGpuPredictor(gpu_count=1, allow_charged=True)
    monkeypatch.setattr(
        predictor,
        "_ensure_gpu_started",
        lambda: pytest.fail("GPU must not start below the site threshold"),
    )

    assert list(predictor.iter_predict(iter(("few",)))) == ["cpu-result"]

    assert calls == [("few", 0, True)]
    assert len(_CpuPredictor.instances) == 1
    cpu = _CpuPredictor.instances[0]
    assert [item.molecule.smiles for item in cpu.prepared] == ["few"]
    assert cpu.closed
    assert predictor._gpu_started is False


def test_exact_minimum_sites_starts_gpu_and_reuses_prepared_prefix(monkeypatch):
    calls = _patch_preparation(monkeypatch, {"boundary": MIN_GPU_SITE_COUNT})
    monkeypatch.setattr(
        module,
        "MCACpuPredictor",
        lambda **kwargs: pytest.fail("CPU must not be selected at the threshold"),
    )
    predictor = MCAGpuPredictor(gpu_count=1)
    started = []
    monkeypatch.setattr(predictor, "_ensure_gpu_started", lambda: started.append(True))
    observed = []

    def gpu_core(prepared):
        observed.extend(prepared)
        return iter(("gpu-result",))

    monkeypatch.setattr(predictor, "_iter_predict", gpu_core)

    assert list(predictor.iter_predict(iter(("boundary",)))) == ["gpu-result"]
    assert started == [True]
    assert calls == [("boundary", 0, False)]
    assert [item.molecule.smiles for item in observed] == ["boundary"]


def test_threshold_decision_does_not_consume_following_generator_item(
    monkeypatch,
):
    consumed = []

    def molecules():
        consumed.append("threshold")
        yield "threshold"
        consumed.append("following")
        yield "following"

    _patch_preparation(
        monkeypatch,
        {"threshold": MIN_GPU_SITE_COUNT, "following": 0},
    )
    predictor = MCAGpuPredictor(gpu_count=1)

    def start_gpu():
        assert consumed == ["threshold"]

    monkeypatch.setattr(predictor, "_ensure_gpu_started", start_gpu)

    def gpu_core(prepared):
        values = list(prepared)
        assert consumed == ["threshold", "following"]
        assert values[0].molecule.smiles == "threshold"
        assert values[1] == "following"
        return iter(("gpu-result",))

    monkeypatch.setattr(predictor, "_iter_predict", gpu_core)

    assert list(predictor.iter_predict(molecules())) == ["gpu-result"]


def test_cpu_routing_is_silent_by_default_and_descriptive_at_debug(
    monkeypatch,
    caplog,
):
    _patch_preparation(monkeypatch, {"few": 3})
    monkeypatch.setattr(module, "MCACpuPredictor", _CpuPredictor)

    caplog.set_level(logging.INFO, logger=module.__name__)
    predictor = MCAGpuPredictor(gpu_count=1)
    assert list(predictor.iter_predict(("few",))) == ["cpu-result"]
    assert not caplog.records

    caplog.clear()
    caplog.set_level(logging.DEBUG, logger=module.__name__)
    predictor = MCAGpuPredictor(gpu_count=1)
    assert list(predictor.iter_predict(("few",))) == ["cpu-result"]
    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 1
    assert "requested=gpu" in messages[0]
    assert "effective=cpu" in messages[0]
    assert "site_count=3" in messages[0]
    assert f"threshold={MIN_GPU_SITE_COUNT}" in messages[0]


def test_cuda_startup_error_at_threshold_is_not_hidden_by_cpu_fallback(
    monkeypatch,
):
    _patch_preparation(monkeypatch, {"boundary": MIN_GPU_SITE_COUNT})
    monkeypatch.setattr(
        module,
        "MCACpuPredictor",
        lambda **kwargs: pytest.fail("CUDA errors must not trigger CPU fallback"),
    )
    predictor = MCAGpuPredictor(gpu_count=1)

    def fail_startup():
        raise RuntimeError("CUDA startup failed")

    monkeypatch.setattr(predictor, "_ensure_gpu_started", fail_startup)

    with pytest.raises(RuntimeError, match="CUDA startup failed"):
        list(predictor.iter_predict(("boundary",)))


def test_gpu_routing_accepts_existing_dto_through_normal_preparation(monkeypatch):
    from hotpot.cheminfo.operations.inference.mca.serialization import normalize_molecule

    dto = normalize_molecule("CN")
    monkeypatch.setattr(module, "MCACpuPredictor", _CpuPredictor)
    predictor = MCAGpuPredictor(gpu_count=1)
    monkeypatch.setattr(
        predictor,
        "_ensure_gpu_started",
        lambda: pytest.fail("Single amine remains below the GPU site threshold"),
    )
    assert list(predictor.iter_predict((dto,))) == ["cpu-result"]
    prepared = _CpuPredictor.instances[0].prepared
    assert len(prepared) == 1
    assert prepared[0].molecule == dto
    assert prepared[0].sites == ((1, "Amine"),)


def test_gpu_python_predictor_keeps_fail_fast_domain_validation(monkeypatch):
    from hotpot.cheminfo.operations.inference.mca.serialization import normalize_molecule

    monkeypatch.setattr(module, "MCACpuPredictor", _CpuPredictor)
    predictor = MCAGpuPredictor(gpu_count=1)
    with pytest.raises(ValueError, match="outside the validated domain"):
        list(predictor.iter_predict((normalize_molecule("C[NH3+]"),)))
    assert _CpuPredictor.instances == []


def test_gpu_threshold_routing_keeps_following_dto_on_existing_path(monkeypatch):
    from hotpot.cheminfo.operations.inference.mca import pipeline
    from hotpot.cheminfo.operations.inference.mca.serialization import normalize_molecule

    first = normalize_molecule("CN")
    following = normalize_molecule("CO")
    sites = ((1, "Amine"),) * MIN_GPU_SITE_COUNT
    monkeypatch.setattr(pipeline, "detect_sites", lambda molecule: sites)
    monkeypatch.setattr(
        module,
        "MCACpuPredictor",
        lambda **kwargs: pytest.fail("GPU threshold must preserve GPU routing"),
    )
    predictor = MCAGpuPredictor(gpu_count=1)
    started = []
    monkeypatch.setattr(predictor, "_ensure_gpu_started", lambda: started.append(True))

    def gpu_core(molecules):
        prepared, remainder = list(molecules)
        assert prepared.molecule == first
        assert prepared.sites == sites
        assert remainder is following
        return iter(("gpu-result",))

    monkeypatch.setattr(predictor, "_iter_predict", gpu_core)
    assert list(predictor.iter_predict((first, following))) == ["gpu-result"]
    assert started == [True]
