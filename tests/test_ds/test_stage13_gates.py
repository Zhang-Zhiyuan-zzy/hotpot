from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path

import pytest

from hotpot.ds.gates import BenchmarkGate, EvidenceGate, QualityGate
from hotpot.ds.gates._checker import (
    CHECKER_PROTOCOL_DIGEST,
    CheckerIsolationUnavailableError,
    evaluate_configured_check,
)
from hotpot.ds.gates._persistence import (
    default_gates,
    deserialize_gate_configs,
    materialize_checker_sources,
    serialize_gate_configs,
)


def _load_checker_module(tmp_path: Path, body: str):
    source_path = tmp_path / "checker_policy.py"
    source_path.write_text(body, encoding="utf-8")
    specification = importlib.util.spec_from_file_location(
        "stage13_test_checker_policy", source_path
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module, source_path.read_text(encoding="utf-8")


def test_public_exports_and_exact_constructor_shapes():
    import hotpot.ds.gates as gates

    assert gates.__all__ == ["EvidenceGate", "QualityGate", "BenchmarkGate"]
    def shape(value):
        return tuple(
            (parameter.name, parameter.kind, parameter.default)
            for parameter in inspect.signature(value).parameters.values()
        )

    assert shape(EvidenceGate) == (
        ("checks", inspect.Parameter.POSITIONAL_OR_KEYWORD, None),
        ("level_checks", inspect.Parameter.KEYWORD_ONLY, None),
        ("require_locator", inspect.Parameter.KEYWORD_ONLY, True),
    )
    assert shape(QualityGate) == (
        ("checks", inspect.Parameter.POSITIONAL_OR_KEYWORD, None),
        ("level_checks", inspect.Parameter.KEYWORD_ONLY, None),
        ("structure_checks", inspect.Parameter.KEYWORD_ONLY, None),
    )
    assert shape(BenchmarkGate) == (
        ("checklist", inspect.Parameter.POSITIONAL_OR_KEYWORD, None),
        ("direct_entry", inspect.Parameter.POSITIONAL_OR_KEYWORD, None),
        ("vetoes", inspect.Parameter.KEYWORD_ONLY, None),
        ("numeric_tolerances", inspect.Parameter.KEYWORD_ONLY, None),
    )


def test_default_policy_encodes_all_levels_and_benchmark_routes():
    evidence, quality, benchmark = default_gates()

    assert list(evidence.info["level_checks"]) == ["E0", "E1", "E2", "E3", "E4"]
    assert evidence.info["level_checks"]["E0"] == []
    assert "primary_source_verified" in evidence.info["level_checks"]["E3"]
    assert "independent_corroboration" in evidence.info["level_checks"]["E4"]
    assert list(quality.info["level_checks"]) == ["B0", "B1", "B2", "B3", "B4"]
    assert quality.info["level_checks"]["B0"] == []
    assert "case_complete" in quality.info["level_checks"]["B2"]
    assert "independent_corroboration" in quality.info["level_checks"]["B4"]
    assert benchmark.info["checklist"] == {
        "conflict_free": {"kind": "builtin", "enabled": True},
        "verified_primary_source": {"kind": "builtin", "enabled": True},
    }
    assert benchmark.info["direct_entry"] == {
        "hotpot_or_user_attestation": {"kind": "builtin", "enabled": True}
    }
    assert benchmark.info["vetoes"] == [
        "attestation_not_rejected",
        "conflict_free",
    ]


def test_digest_is_stable_and_mapping_order_independent():
    first = BenchmarkGate(
        checklist={"verified_primary_source": True, "conflict_free": False},
        direct_entry={"hotpot_or_user_attestation": True},
        vetoes=("attestation_not_rejected", "conflict_free"),
        numeric_tolerances={
            "spin": {"relative": 1, "absolute": 0.25},
            "oxidation": {"absolute": 0},
        },
    )
    second = BenchmarkGate(
        checklist={"conflict_free": False, "verified_primary_source": True},
        direct_entry={"hotpot_or_user_attestation": True},
        vetoes=("conflict_free", "attestation_not_rejected"),
        numeric_tolerances={
            "oxidation": {"absolute": 0.0},
            "spin": {"absolute": 0.25, "relative": 1.0},
        },
    )

    assert first.info == second.info
    assert first.digest == second.digest
    assert first.info["numeric_tolerances"]["oxidation"] == {
        "absolute": 0.0,
        "relative": 0.0,
    }
    assert len(first.digest) == 64
    int(first.digest, 16)


def test_gate_state_and_returned_info_are_immutable():
    gate = EvidenceGate()
    original_digest = gate.digest
    snapshot = gate.info
    snapshot["checks"]["explicit_report"]["enabled"] = False
    snapshot["level_checks"]["E4"].clear()

    assert gate.info["checks"]["explicit_report"]["enabled"] is True
    assert gate.info["level_checks"]["E4"]
    assert gate.digest == original_digest
    with pytest.raises(AttributeError, match="immutable"):
        gate._digest = "changed"


def test_custom_checker_captures_exact_source_and_protocol(tmp_path):
    module, source = _load_checker_module(
        tmp_path,
        "from __future__ import annotations\n"
        "import math\n\n"
        "def project_rule(case, bundle):\n"
        "    return (math.isfinite(case['value']), bundle['reason'])\n",
    )
    gate = BenchmarkGate(
        checklist={
            "verified_primary_source": True,
            "conflict_free": True,
            "project_rule": module.project_rule,
        }
    )
    definition = gate.info["checklist"]["project_rule"]

    assert definition["kind"] == "custom"
    assert definition["source"] == source
    assert definition["entrypoint"] == "project_rule"
    assert definition["imports"] == ["__future__", "math"]
    assert definition["checker_protocol_digest"] == CHECKER_PROTOCOL_DIGEST
    assert len(definition["source_sha256"]) == 64


@pytest.mark.parametrize(
    "factory, message",
    [
        (lambda: (lambda case, bundle: (True, "ok")), "top-level"),
        (lambda: _CallableChecker(), "Python function"),
    ],
)
def test_invalid_custom_checker_shapes_are_rejected(factory, message):
    with pytest.raises((TypeError, ValueError), match=message):
        BenchmarkGate(checklist={"bad": factory()})


class _CallableChecker:
    def __call__(self, case, bundle):
        return True, "callable instance"


def test_closure_and_source_unavailable_are_rejected():
    captured = True

    def closure(case, bundle):
        return captured, "captured"

    with pytest.raises(ValueError, match="top-level"):
        BenchmarkGate(checklist={"closure": closure})

    namespace = {}
    exec(
        compile(
            "def generated(case, bundle):\n    return True, 'generated'\n",
            "<generated-checker>",
            "exec",
        ),
        namespace,
    )
    with pytest.raises(ValueError, match="source is unavailable"):
        BenchmarkGate(checklist={"generated": namespace["generated"]})


def test_non_allowlisted_import_and_wrong_signature_are_rejected(tmp_path):
    forbidden, _ = _load_checker_module(
        tmp_path,
        "import os\n\ndef bad_import(case, bundle):\n    return True, os.name\n",
    )
    with pytest.raises(ValueError, match="not allowlisted: os"):
        BenchmarkGate(checklist={"bad_import": forbidden.bad_import})

    source_path = tmp_path / "checker_policy.py"
    source_path.write_text(
        "def wrong_name(item, bundle):\n    return True, 'wrong'\n",
        encoding="utf-8",
    )
    specification = importlib.util.spec_from_file_location(
        "stage13_wrong_signature", source_path
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    with pytest.raises(TypeError, match=r"checker\(case, bundle\)"):
        BenchmarkGate(checklist={"wrong": module.wrong_name})


def test_unknown_boolean_checker_and_invalid_tolerances_are_rejected():
    with pytest.raises(ValueError, match="unknown built-in checker"):
        BenchmarkGate(checklist={"typo": True})
    with pytest.raises(ValueError, match="finite and non-negative"):
        BenchmarkGate(numeric_tolerances={"spin": {"absolute": -1}})
    with pytest.raises(ValueError, match="only absolute and relative"):
        BenchmarkGate(numeric_tolerances={"spin": {"percent": 1}})


def test_current_config_roundtrip_preserves_info_and_digest(tmp_path):
    module, _ = _load_checker_module(
        tmp_path,
        "def project_rule(case, bundle):\n    return True, 'accepted'\n",
    )
    gates = (
        EvidenceGate(require_locator=False),
        QualityGate(),
        BenchmarkGate(
            checklist={"project_rule": module.project_rule},
            direct_entry={},
            vetoes=(),
            numeric_tolerances={"spin": {"relative": 0.01}},
        ),
    )

    encoded = serialize_gate_configs(reversed(gates))
    json.dumps(encoded, allow_nan=False)
    restored = deserialize_gate_configs(encoded)

    assert [gate.info for gate in restored] == [gate.info for gate in gates]
    assert [gate.digest for gate in restored] == [gate.digest for gate in gates]


def test_custom_sources_are_content_addressed_and_deduplicated(tmp_path):
    module, source = _load_checker_module(
        tmp_path,
        "def project_rule(case, bundle):\n    return True, 'accepted'\n",
    )
    gates = (
        EvidenceGate(checks={"project_rule": module.project_rule}),
        QualityGate(),
        BenchmarkGate(checklist={"project_rule": module.project_rule}),
    )

    paths = materialize_checker_sources(gates, tmp_path / "bundle")

    assert len(paths) == 1
    assert paths[0].parent == tmp_path / "bundle" / "policy" / "checkers"
    assert paths[0].read_text(encoding="utf-8") == source
    assert paths == materialize_checker_sources(gates, tmp_path / "bundle")


def test_custom_evaluation_fails_closed_without_os_isolation(tmp_path):
    module, _ = _load_checker_module(
        tmp_path,
        "def project_rule(case, bundle):\n    return True, 'accepted'\n",
    )
    gate = BenchmarkGate(checklist={"project_rule": module.project_rule})
    definition = dict(gate._checklist)["project_rule"]

    with pytest.raises(CheckerIsolationUnavailableError):
        evaluate_configured_check(definition, {}, {})

    class IsolatedRunner:
        os_isolation_verified = True

        def run(self, checker, case, bundle):
            assert checker["entrypoint"] == "project_rule"
            case["private_copy"] = True
            return True, "isolated result"

    case = {"value": 1}
    assert evaluate_configured_check(
        definition, case, {}, isolated_runner=IsolatedRunner()
    ) == (True, "isolated result")
    assert case == {"value": 1}


def test_builtin_evaluation_uses_strict_result_protocol():
    definition = dict(BenchmarkGate()._checklist)["conflict_free"]

    assert evaluate_configured_check(
        definition, {}, {}, builtin_result=(True, "no conflict")
    ) == (True, "no conflict")
    with pytest.raises(ValueError, match="exact bool"):
        evaluate_configured_check(definition, {}, {}, builtin_result=(1, "not bool"))
