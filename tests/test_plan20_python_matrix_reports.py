"""Complete-work accounting for the supplementary Python matrix collector."""

from importlib import import_module
import json
from types import SimpleNamespace

import pytest


_matrix = import_module("plan.20_fact_rule_architecture.run_python_matrix_baseline")


def test_failed_first_pytest_group_keeps_second_group_and_all_elapsed_time(tmp_path, monkeypatch):
    preflight = tmp_path / "3.10/preflight"
    preflight.mkdir(parents=True)
    (preflight / "installed-wheel.json").write_text("{}")
    visited = []

    def run_group(args, manifest, group, *rest):
        visited.append(group["id"])
        return {"group": group["id"], "status": "error" if group["id"] == "first" else "completed",
                "seconds": 2.0 if group["id"] == "first" else 10.0, "peak_rss_bytes": 100,
                "requested_nodes": 1, "collected_nodes": 1, "collection_sha256": group["id"],
                "junit": {"tests": 1, "failures": int(group["id"] == "first"), "errors": 0, "skipped": 0}}

    monkeypatch.setattr(_matrix, "run_pytest", run_group)
    result = _matrix.attempt(SimpleNamespace(output=tmp_path, index=1),
                             {"policy": {"warmups": 1}, "groups": [{"id": "first"}, {"id": "second"}]},
                             "3.10", {})
    assert visited == ["first", "second"]
    assert result["status"] == "error"
    assert result["seconds"] == 12.0
    assert result["work"][0]["junit"]["failures"] == 1
    assert result["work"][1]["junit"]["failures"] == 0


def test_early_failure_is_in_total_work_but_not_the_completed_time_estimate():
    samples = [
        {"warmup": True, "status": "completed", "seconds": 100.0, "peak_rss_bytes": 1000},
        {"warmup": False, "status": "error", "seconds": 0.1, "peak_rss_bytes": 50},
        {"warmup": False, "status": "completed", "seconds": 10.0, "peak_rss_bytes": 100},
        {"warmup": False, "status": "not_run", "seconds": None},
    ]
    result = _matrix.summarize(samples)
    assert result["requested_samples"] == 3
    assert result["completed"] == 1
    assert result["errors"] == result["not_run"] == result["elapsed_unavailable"] == 1
    assert result["all_attempt_seconds"] == 10.1
    assert result["median_seconds"] == 10.0


def test_malformed_junit_retains_successful_process_time_as_a_technical_error(tmp_path, monkeypatch):
    def fake_process(command, directory, environment, *, name):
        (directory / "probe.json").write_text(json.dumps({
            "nodeids": ["tests/test_example.py::test_example"], "modules": {"hotpot": "/installed/hotpot/__init__.py"},
            "test_reports": [{"nodeid": "tests/test_example.py::test_example", "when": "call", "outcome": "passed"}],
        }))
        (directory / "junit.xml").write_text("<broken")
        return {"returncode": 0, "seconds": 20.0, "peak_rss_bytes": 1234}

    monkeypatch.setattr(_matrix, "process", fake_process)
    monkeypatch.setattr(_matrix._shared, "verify_loaded_package_files", lambda *args: {"verified": True})
    result = _matrix.run_pytest(
        SimpleNamespace(repository=tmp_path / "source", output=tmp_path),
        {"pytest_common_options": []},
        {"id": "contracts", "nodeids": ["tests/test_example.py::test_example"], "options": []},
        tmp_path / "attempt/contracts", {}, {},
    )
    assert result["returncode"] == 0
    assert result["status"] == "error"
    assert result["seconds"] == 20.0
    assert result["peak_rss_bytes"] == 1234
    assert any("report validation failed" in error for error in result["errors"])


@pytest.mark.parametrize("reports", [[], [{"nodeid": "tests/test_example.py::test_example", "when": "setup", "outcome": "passed"}]])
def test_collection_or_setup_without_execution_cannot_complete(tmp_path, monkeypatch, reports):
    node = "tests/test_example.py::test_example"

    def fake_process(command, directory, environment, *, name):
        (directory / "probe.json").write_text(json.dumps({
            "nodeids": [node], "modules": {"hotpot": "/installed/hotpot/__init__.py"},
            "test_reports": reports,
        }))
        (directory / "junit.xml").write_text('<testsuites><testsuite tests="0" failures="0" errors="0" skipped="0"/></testsuites>')
        return {"returncode": 0, "seconds": 0.1, "peak_rss_bytes": 100}

    monkeypatch.setattr(_matrix, "process", fake_process)
    monkeypatch.setattr(_matrix._shared, "verify_loaded_package_files", lambda *args: {"verified": True})
    result = _matrix.run_pytest(
        SimpleNamespace(repository=tmp_path / "source", output=tmp_path),
        {"pytest_common_options": []},
        {"id": "contracts", "nodeids": [node], "options": []},
        tmp_path / "attempt/contracts", {}, {},
    )
    assert result["status"] == "error"
    assert result["unexecuted_nodes"] == [node]
    assert "JUnit reports fewer tests than the frozen selection" in result["errors"]


def test_installed_matrix_child_cwd_and_pytest_options_are_isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("HOTPOT_TEST_WORKDIR", str(tmp_path / "source"))
    monkeypatch.setenv("PYTEST_ADDOPTS", "--collect-only")
    _, environment = _matrix.configured_environment(
        SimpleNamespace(output=tmp_path / "matrix"),
        {"policy": {"thread_environment": {}, "python_hash_seed": "0"}},
        {"versions": [{"version": "3.10", "python": "/env/bin/python", "wheel": str(tmp_path / "wheel.whl")}]},
        "3.10",
    )
    assert environment["HOTPOT_TEST_WORKDIR"] == str(tmp_path / "matrix/3.10")
    assert environment["PYTEST_ADDOPTS"] == ""


def test_worker_exit_failure_overrides_a_completed_report_without_losing_work():
    partial = {"status": "completed", "seconds": 20.0, "groups": [{"collected_nodes": 2408}]}
    execution = {"returncode": -11, "seconds": 23.0}
    result = _matrix.bind_worker_execution(partial, execution)
    assert result["status"] == "error"
    assert result["seconds"] == 20.0
    assert result["groups"] == partial["groups"]
    assert result["worker_process"] == execution
    assert partial["status"] == "completed"
