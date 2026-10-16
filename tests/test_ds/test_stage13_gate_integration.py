from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil

import pytest

from hotpot.ds.gates import BenchmarkGate, EvidenceGate, QualityGate
from hotpot.ds.gates._checker import (
    CheckerIsolationUnavailableError,
    OSIsolatedCheckerRunner,
)
from hotpot.ds.gates._persistence import serialize_gate_configs
from hotpot.ds.public._assessment import assess_case
from hotpot.ds.public._bundle_state import BUNDLE_SCHEMA, BundleState
from hotpot.ds.public._generation import AssessmentGeneration
from hotpot.ds.public._implementation import runtime_policy


def _policy(gates) -> dict:
    return {
        "schema": "hotpot-ds-policy/current",
        "gates": serialize_gate_configs(gates),
        "runtime_policy": runtime_policy(),
    }


def _case(*, attestations=(), with_evidence: bool = False) -> dict:
    case = {
        "id": "case-1",
        "local_id": "case-1",
        "public_index": 0,
        "identifiers": {"sample": "one"},
        "metadata": {"supplier": "test"},
        "core_info": {"sample": "one"},
        "observations": [
            {
                "observation_id": "observation-1",
                "property": "spin",
                "value": 1,
            }
        ],
        "attestations": list(attestations),
    }
    if with_evidence:
        case["evidence"] = [
            {
                "evidence_id": "evidence-1",
                "target": "observation-1",
                "relation": "supports",
                "source_kind": "primary_literature",
                "scientific_work_id": "doi:10.1/example",
                "source_lineage_id": "result:one",
                "locator": "p. 1",
            }
        ]
        case["verifications"] = [
            {
                "verification_id": "verification-1",
                "evidence_id": "evidence-1",
                "method": "Human",
                "verifier": "test-reviewer",
            }
        ]
    return case


def _assess(case: dict, benchmark: BenchmarkGate, **values) -> dict:
    return assess_case(
        case,
        core_labels=["spin"],
        core_info=["sample"],
        required_structure_roles=[],
        asset_assessments=[],
        conflicting=False,
        evidence_policy=values.get("evidence", EvidenceGate()).info,
        quality_policy=values.get("quality", QualityGate()).info,
        benchmark_policy=benchmark.info,
        conflict_assessments=values.get("conflicts", ()),
        isolated_runner=values.get("runner"),
    )


def _checker_module(tmp_path: Path):
    path = tmp_path / "gate_checker.py"
    path.write_text(
        "def trusted_repository(case, bundle):\n"
        "    return bool(bundle['dataset']['ds_name']), 'repository accepted'\n",
        encoding="utf-8",
    )
    specification = importlib.util.spec_from_file_location("gate_checker", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_disabled_checks_and_configured_level_maps_are_authoritative() -> None:
    evidence = EvidenceGate(checks={"primary_source_verified": False})
    quality = QualityGate(checks={"reviewed": False})
    benchmark = BenchmarkGate(
        checklist={"verified_primary_source": False, "conflict_free": False},
        direct_entry={"hotpot_or_user_attestation": False},
        vetoes=(),
    )
    case = _case(
        attestations=[
            {
                "attestation_id": "approval-1",
                "authority": "User",
                "actor": "owner",
                "approved": True,
            }
        ],
        with_evidence=True,
    )

    result = _assess(case, benchmark, evidence=evidence, quality=quality)

    assert result["evidence_level"] == "E2"
    assert result["quality_level"] == "B1"
    assert result["benchmark"] is False
    assert result["gate_results"]["evidence"]["observation-1"][
        "primary_source_verified"
    ]["status"] == "disabled"
    assert result["gate_results"]["quality"]["reviewed"]["status"] == "disabled"


def test_direct_entry_and_veto_membership_follow_persisted_policy() -> None:
    case = _case(
        attestations=[
            {
                "attestation_id": "approval-1",
                "authority": "User",
                "actor": "approver",
                "approved": True,
            },
            {
                "attestation_id": "rejection-1",
                "authority": "User",
                "actor": "reviewer",
                "approved": False,
            },
        ]
    )

    assert _assess(case, BenchmarkGate())["benchmark_status"] == "rejected"
    permissive = BenchmarkGate(
        checklist={},
        direct_entry={"hotpot_or_user_attestation": True},
        vetoes=(),
    )
    assert _assess(case, permissive)["benchmark_status"] == "admitted"


def test_indeterminate_comparison_obeys_conflict_veto_configuration() -> None:
    case = _case(
        attestations=[
            {
                "attestation_id": "approval-1",
                "authority": "User",
                "actor": "owner",
                "approved": True,
            }
        ]
    )
    conflict = {
        "outcome": "indeterminate",
        "reason": "numeric_tolerance_not_configured",
    }

    assert _assess(case, BenchmarkGate(), conflicts=[conflict])[
        "benchmark_status"
    ] == "vetoed"
    no_conflict_veto = BenchmarkGate(
        checklist={},
        direct_entry={"hotpot_or_user_attestation": True},
        vetoes={"attestation_not_rejected"},
    )
    assert _assess(case, no_conflict_veto, conflicts=[conflict])["benchmark"] is True


def test_custom_checker_requires_isolation_and_errors_fail_closed(tmp_path: Path) -> None:
    module = _checker_module(tmp_path)
    gate = BenchmarkGate(
        checklist={},
        direct_entry={"trusted_repository": module.trusted_repository},
        vetoes=(),
    )

    with pytest.raises(CheckerIsolationUnavailableError):
        _assess(_case(), gate)

    class PassingRunner:
        os_isolation_verified = True

        def run(self, checker, case, bundle):
            return True, "isolated approval"

    assert _assess(_case(), gate, runner=PassingRunner())["benchmark"] is True

    class ThrowingRunner:
        os_isolation_verified = True

        def run(self, checker, case, bundle):
            raise RuntimeError("worker failed")

    result = _assess(_case(), gate, runner=ThrowingRunner())
    check = result["gate_results"]["benchmark_direct_entry"][
        "trusted_repository"
    ]
    assert result["benchmark"] is False
    assert check["status"] == "checker_error"
    assert check["reason"] == "checker_error"


def test_runtime_timeout_is_recorded_as_checker_error_by_assessment(
    tmp_path: Path,
) -> None:
    if shutil.which("bwrap") is None:
        pytest.skip("bubblewrap is not installed on this Linux host")
    path = tmp_path / "timeout_checker.py"
    path.write_text(
        "def never_returns(case, bundle):\n"
        "    while True:\n"
        "        pass\n",
        encoding="utf-8",
    )
    specification = importlib.util.spec_from_file_location("timeout_checker", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    gate = BenchmarkGate(
        checklist={},
        direct_entry={"timeout": module.never_returns},
        vetoes=(),
    )

    result = _assess(_case(), gate, runner=OSIsolatedCheckerRunner())

    check = result["gate_results"]["benchmark_direct_entry"]["timeout"]
    assert result["benchmark"] is False
    assert check["status"] == "checker_error"
    assert check["error_type"] == "CheckerExecutionError"


def test_generation_preserves_custom_definition_and_runner_boundary(
    tmp_path: Path,
) -> None:
    module = _checker_module(tmp_path)
    root = tmp_path / "bundle"
    intake = root / "evidence" / "intake" / "cases.jsonl"
    intake.parent.mkdir(parents=True)
    intake.write_text(json.dumps(_case()) + "\n", encoding="utf-8")
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "ds_name": "demo",
        "release_id": "demo-r1",
        "version": "1.0",
        "title": "Demo",
        "description": "Gate integration test",
        "license": "CC0-1.0",
        "access_policy": "open",
        "provenance": {"created_by": "test"},
        "core_labels": ["spin"],
        "core_info": ["sample"],
        "intake": {"format": "jsonl", "path": "evidence/intake/cases.jsonl"},
        "identity_schema": {
            "components": [
                {
                    "key": "sample",
                    "path": "core_info.sample",
                    "required": True,
                    "normalizer": "text",
                }
            ]
        },
        "required_structure_roles": [],
    }
    (root / "bundle.json").write_text(json.dumps(manifest), encoding="utf-8")
    policy = _policy(
        (
                EvidenceGate(),
                QualityGate(),
                BenchmarkGate(
                    checklist={},
                    direct_entry={"trusted_repository": module.trusted_repository},
                    vetoes=(),
                ),
        )
    )
    state = BundleState.build(
        root,
        "demo",
        policy_json=policy,
        target_context={"mode": "new", "store_id": None},
    )
    policy = state.policy
    from hotpot.ds.governance import default_registry_snapshot

    with state.synchronization_snapshot() as input_snapshot:
        with pytest.raises(CheckerIsolationUnavailableError):
            AssessmentGeneration.build(
                root,
                state,
                policy,
                default_registry_snapshot(),
                input_snapshot=input_snapshot,
            )

    class PassingRunner:
        os_isolation_verified = True

        def run(self, checker, case, bundle):
            assert checker["entrypoint"] == "trusted_repository"
            assert bundle["dataset"]["ds_name"] == "demo"
            return True, "isolated approval"

    with state.synchronization_snapshot() as input_snapshot:
        generation = AssessmentGeneration.build(
            root,
            state,
            policy,
            default_registry_snapshot(),
            input_snapshot=input_snapshot,
            isolated_runner=PassingRunner(),
        )
    assert generation.get_case("case-1")["benchmark"] is True


def test_generation_uses_persisted_numeric_tolerances(tmp_path: Path) -> None:
    from hotpot.ds.governance import default_registry_snapshot

    def build(name: str, tolerances: dict) -> dict:
        root = tmp_path / name
        case = _case(
            attestations=[
                {
                    "attestation_id": "approval-1",
                    "authority": "User",
                    "actor": "owner",
                    "approved": True,
                }
            ]
        )
        case["observations"].append(
            {
                "observation_id": "observation-2",
                "property": "spin",
                "value": 1.05,
            }
        )
        intake = root / "evidence" / "intake" / "cases.jsonl"
        intake.parent.mkdir(parents=True)
        intake.write_text(json.dumps(case) + "\n", encoding="utf-8")
        manifest = {
            "schema": BUNDLE_SCHEMA,
            "ds_name": name,
            "release_id": f"{name}-r1",
            "version": "1.0",
            "title": "Tolerance",
            "description": "Numeric tolerance test",
            "license": "CC0-1.0",
            "access_policy": "open",
            "provenance": {"created_by": "test"},
            "core_labels": ["spin"],
            "core_info": ["sample"],
            "intake": {
                "format": "jsonl",
                "path": "evidence/intake/cases.jsonl",
            },
            "identity_schema": {
                "components": [
                    {
                        "key": "sample",
                        "path": "core_info.sample",
                        "required": True,
                        "normalizer": "text",
                    }
                ]
            },
            "required_structure_roles": [],
        }
        (root / "bundle.json").write_text(json.dumps(manifest), encoding="utf-8")
        policy = _policy(
            (
                    EvidenceGate(),
                    QualityGate(),
                    BenchmarkGate(numeric_tolerances=tolerances),
            )
        )
        state = BundleState.build(
            root,
            name,
            policy_json=policy,
            target_context={"mode": "new", "store_id": None},
        )
        policy = state.policy
        with state.synchronization_snapshot() as input_snapshot:
            return AssessmentGeneration.build(
                root,
                state,
                policy,
                default_registry_snapshot(),
                input_snapshot=input_snapshot,
            ).get_case("case-1")

    assert build("strict", {})["benchmark_status"] == "vetoed"
    assert build("tolerant", {"spin": {"absolute": 0.1}})["benchmark"] is True
