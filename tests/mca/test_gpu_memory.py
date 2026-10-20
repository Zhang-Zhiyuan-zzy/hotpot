from types import SimpleNamespace

import pytest

from hotpot.cheminfo.AImodels.mca.gpu_memory import (
    GIB,
    MIB,
    MCAGpuOutOfMemoryError,
    automatic_gpu_batch_size,
    gpu_memory_bytes,
    gpu_total_memories_bytes,
    is_cuda_out_of_memory,
    next_lower_gpu_batch_size,
    parse_nvidia_smi_memory,
    resolve_gpu_batch_size,
    validate_gpu_batch_size,
)


@pytest.mark.parametrize(
    ("total_bytes", "expected"),
    (
        (8 * GIB - 1, 64),
        (8 * GIB, 128),
        (16 * GIB - 1, 128),
        (16 * GIB, 256),
        (24 * GIB - 1, 256),
        (24 * GIB, 512),
        (40 * GIB - 1, 512),
        (40 * GIB, 1024),
        (80 * GIB - 1, 1024),
        (80 * GIB, 2048),
    ),
)
def test_automatic_batch_uses_binary_total_vram_boundaries(total_bytes, expected):
    assert automatic_gpu_batch_size(total_bytes) == expected


def test_heterogeneous_gpus_use_the_smallest_automatic_tier():
    assert resolve_gpu_batch_size((80 * GIB, 39 * GIB, 48 * GIB)) == 512


def test_explicit_batch_overrides_memory_but_is_restricted_to_fixed_tiers():
    assert resolve_gpu_batch_size((7 * GIB,), 2048) == 2048
    with pytest.raises(ValueError, match="64, 128, 256, 512, 1024, 2048"):
        validate_gpu_batch_size(100)
    with pytest.raises(ValueError):
        validate_gpu_batch_size(64.0)


def test_total_memory_batch_query_is_ordered_and_called_once_per_gpu():
    calls = []

    def query(index):
        calls.append(index)
        return (index + 1) * GIB

    assert gpu_total_memories_bytes((3, 1), query) == (4 * GIB, 2 * GIB)
    assert calls == [3, 1]


def test_nvidia_smi_output_and_visible_logical_index_are_parsed(monkeypatch):
    calls = []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5,GPU-deadbeef")

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout="81920, 60000\n")

    monkeypatch.setattr(
        "hotpot.cheminfo.AImodels.mca.gpu_memory.subprocess.run", run
    )
    memory = gpu_memory_bytes(1)

    assert memory.total_bytes == 81920 * MIB
    assert memory.free_bytes == 60000 * MIB
    assert "--id=GPU-deadbeef" in calls[0][0]
    assert calls[0][1] == {"check": True, "capture_output": True, "text": True}


def test_malformed_nvidia_smi_output_is_not_silently_accepted():
    with pytest.raises(RuntimeError, match="Expected one"):
        parse_nvidia_smi_memory("1, 2\n3, 4\n", 0)
    with pytest.raises(RuntimeError, match="Malformed"):
        parse_nvidia_smi_memory("unknown, 4\n", 0)


def test_oom_error_reports_current_and_next_lower_tier_without_retry():
    error = MCAGpuOutOfMemoryError(
        gpu_index=2,
        batch_size=1024,
        operation="ONNX inference",
        total_memory_bytes=40 * GIB,
        free_memory_bytes=2 * GIB,
        variant="mecap",
    )

    assert error.gpu_index == 2
    assert error.batch_size == 1024
    assert error.next_batch_size == 512
    assert "gpu_batch_size=1024" in str(error)
    assert "gpu_batch_size=512" in str(error)
    assert "no automatic retry or CPU fallback" in str(error)


def test_lowest_oom_tier_says_no_lower_supported_batch():
    error = MCAGpuOutOfMemoryError(
        gpu_index=0,
        batch_size=64,
        operation="collation",
    )
    assert error.next_batch_size is None
    assert "lowest supported batch tier" in str(error)
    assert next_lower_gpu_batch_size(64) is None


def test_only_cuda_allocation_failures_are_classified_as_gpu_oom():
    assert is_cuda_out_of_memory(RuntimeError("CUDA out of memory"))
    assert is_cuda_out_of_memory(
        RuntimeError("CUDAExecutionProvider failed to allocate a buffer")
    )
    assert is_cuda_out_of_memory(
        RuntimeError("BFCArena failed to allocate memory for requested buffer")
    )
    assert not is_cuda_out_of_memory(MemoryError("out of memory"))
    assert not is_cuda_out_of_memory(RuntimeError("invalid CUDA graph"))
