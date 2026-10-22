import pytest

from hotpot.cheminfo.AImodels.mca.resources import (
    DEFAULT_CPU_COUNT,
    DEFAULT_INFER_CPU_PER_GPU,
    DEFAULT_SITE_CPU_PER_GPU,
    InsufficientCPUAffinityError,
    InsufficientCPUCountError,
    build_gpu_resource_plan,
    pin_current_process,
    pin_worker_from_queue,
    resolve_cpu_ids,
    resolve_gpu_indices,
    validate_gpu_resource_request,
)


def test_gpu_selection_defaults_to_logical_zero_and_preserves_explicit_order():
    assert resolve_gpu_indices() == (0,)
    assert resolve_gpu_indices(gpu_count=3) == (0, 1, 2)
    assert resolve_gpu_indices(gpu_indices=(3, 1)) == (3, 1)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"gpu_count": 2, "gpu_indices": (0, 1)},
        {"gpu_count": 0},
        {"gpu_indices": ()},
        {"gpu_indices": (1, 1)},
        {"gpu_indices": (-1,)},
    ),
)
def test_invalid_gpu_selection_is_rejected(kwargs):
    with pytest.raises(ValueError):
        resolve_gpu_indices(**kwargs)


def test_resource_plan_uses_approved_defaults_and_stable_per_gpu_blocks():
    affinity = tuple(range(101, 101 + 40))
    plan = build_gpu_resource_plan(
        gpu_indices=(4, 2),
        cpu_affinity=tuple(reversed(affinity)),
    )

    assert plan.site_cpu_per_gpu == DEFAULT_SITE_CPU_PER_GPU == 1
    assert plan.infer_cpu_per_gpu == DEFAULT_INFER_CPU_PER_GPU == 2
    assert plan.required_cpu_count == 6
    assert plan.assigned_cpu_ids == affinity[:6]
    assert plan.gpu_cpu_assignments[0].gpu_index == 4
    assert plan.gpu_cpu_assignments[0].site_cpu_ids == affinity[:1]
    assert plan.gpu_cpu_assignments[0].infer_cpu_ids == affinity[1:3]
    assert plan.gpu_cpu_assignments[1].gpu_index == 2
    assert plan.gpu_cpu_assignments[1].cpu_ids == affinity[3:6]
    assert tuple(item.cpu_id for item in plan.site_worker_assignments) == (
        affinity[:1] + affinity[3:4]
    )
    assert tuple(item.cpu_id for item in plan.infer_worker_assignments) == (
        affinity[1:3] + affinity[4:6]
    )


def test_resource_plan_reports_cpu_affinity_shortfall_before_startup():
    with pytest.raises(InsufficientCPUAffinityError) as caught:
        build_gpu_resource_plan(gpu_count=2, cpu_affinity=range(5))

    error = caught.value
    assert error.gpu_count == 2
    assert error.required_cpu_count == 6
    assert error.available_cpu_count == 5
    assert "1 site + 2 inference CPUs per GPU" in str(error)


def test_cpu_selection_defaults_to_eight_and_preserves_affinity_order():
    affinity = tuple(range(101, 121))

    assert DEFAULT_CPU_COUNT == 8
    assert resolve_cpu_ids(cpu_affinity=tuple(reversed(affinity))) == affinity[:8]
    assert resolve_cpu_ids(3, cpu_affinity=affinity) == affinity[:3]


def test_cpu_selection_reports_affinity_shortfall():
    with pytest.raises(InsufficientCPUCountError, match="requires 5 CPUs"):
        resolve_cpu_ids(5, cpu_affinity=(2, 4, 6, 8))


def test_custom_worker_counts_directly_determine_required_cpu_count():
    plan = build_gpu_resource_plan(
        gpu_count=3,
        site_cpu_per_gpu=2,
        infer_cpu_per_gpu=1,
        cpu_affinity=range(10, 30),
    )

    assert plan.required_cpu_count == 9
    assert plan.assigned_cpu_ids == tuple(range(10, 19))


def test_resource_request_validation_does_not_read_cpu_affinity(monkeypatch):
    monkeypatch.setattr(
        "hotpot.cheminfo.AImodels.mca.resources.current_cpu_affinity",
        lambda: pytest.fail("host affinity must remain lazy"),
    )

    assert validate_gpu_resource_request(
        gpu_indices=(3, 1),
        site_cpu_per_gpu=2,
        infer_cpu_per_gpu=4,
    ) == ((3, 1), 2, 4)


def test_pin_current_process_uses_exact_resolved_affinity(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "hotpot.cheminfo.AImodels.mca.resources.os.sched_setaffinity",
        lambda process, cpus: calls.append((process, cpus)),
    )

    pin_current_process(7)
    pin_current_process((8, 9))

    assert calls == [(0, {7}), (0, {8, 9})]


def test_pool_initializer_consumes_and_binds_exactly_one_cpu(monkeypatch):
    class Queue:
        def __init__(self):
            self.values = [12, 13]
            self.get_calls = 0

        def get(self):
            self.get_calls += 1
            return self.values.pop(0)

    queue = Queue()
    pinned = []
    monkeypatch.setattr(
        "hotpot.cheminfo.AImodels.mca.resources.pin_current_process",
        pinned.append,
    )

    pin_worker_from_queue(queue)

    assert queue.get_calls == 1
    assert queue.values == [13]
    assert pinned == [12]
