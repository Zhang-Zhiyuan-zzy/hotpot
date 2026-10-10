"""Strict contract fence for lazy molecular-stage registration."""

from __future__ import annotations

import importlib
import subprocess
import sys
from types import ModuleType

import pytest


def _fresh_registry() -> ModuleType:
    registry = importlib.import_module("hotpot.pipeline.registry")
    return importlib.reload(registry)


def test_builtin_stage_names_are_explicit_and_stable() -> None:
    registry = _fresh_registry()

    assert registry.builtin_stage_names() == ("cbond", "ff", "xtb")


def test_registry_import_does_not_import_builtin_stage_modules() -> None:
    stage_modules = (
        "hotpot.cheminfo.AImodels.cbond.stage",
        "hotpot.cheminfo.forcefields.stage",
        "hotpot.plugins.xtb.stage",
    )
    script = "\n".join(
        (
            "import sys",
            "import hotpot.pipeline.registry",
            f"names = {stage_modules!r}",
            "raise SystemExit(int(any(name in sys.modules for name in names)))",
        )
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_unknown_stage_is_rejected_explicitly() -> None:
    registry = _fresh_registry()

    with pytest.raises(registry.UnknownStageError, match="not-registered"):
        registry.get_stage("not-registered")


def test_duplicate_registration_is_rejected() -> None:
    registry = _fresh_registry()

    with pytest.raises(registry.DuplicateStageError, match="xtb"):
        registry.register_stage(
            "xtb",
            "hotpot.plugins.xtb.stage:STAGE",
        )


def test_custom_stage_registration_is_lazy() -> None:
    registry = _fresh_registry()
    module_name = "hotpot_test_pipeline_stage_that_does_not_exist"

    registry.register_stage("deferred", f"{module_name}:STAGE")

    assert module_name not in sys.modules
    with pytest.raises(ModuleNotFoundError, match=module_name):
        registry.get_stage("deferred")


@pytest.mark.parametrize("name", ("cbond", "ff", "xtb"))
@pytest.mark.xfail(
    strict=True,
    reason="built-in stage adapters land in the next Phase 15 substep",
)
def test_builtin_stage_resolves_to_molecular_stage(name: str) -> None:
    from hotpot.pipeline.contracts import MolecularStage

    registry = _fresh_registry()
    stage = registry.get_stage(name)

    assert callable(stage.prepare)
    module_path = registry.stage_import_path(name).partition(":")[0]
    assert stage.__class__.__module__ == module_path
    assert getattr(MolecularStage, "_is_protocol", False)


@pytest.mark.xfail(
    strict=True,
    reason="the xTB stage adapter lands in the next Phase 15 substep",
)
def test_repeated_xtb_stages_prepare_independently() -> None:
    from hotpot.pipeline.contracts import StageSpec

    registry = _fresh_registry()
    stage = registry.get_stage("xtb")
    gfnff = stage.prepare(
        StageSpec("xtb", ("--method", "gfnff", "--task", "optimize"))
    )
    gfn2 = stage.prepare(
        StageSpec("xtb", ("--method", "gfn2", "--task", "optimize"))
    )

    assert registry.get_stage("xtb") is stage
    assert gfnff is not gfn2


def test_registry_errors_are_pipeline_definition_errors() -> None:
    from hotpot.pipeline.contracts import PipelineDefinitionError

    registry = _fresh_registry()

    assert issubclass(registry.UnknownStageError, PipelineDefinitionError)
    assert issubclass(registry.DuplicateStageError, PipelineDefinitionError)
