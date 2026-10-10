"""Explicit and lazy registry for molecular pipeline stages."""

from __future__ import annotations

import importlib
from typing import Dict, Tuple, cast

from .contracts import MolecularStage, PipelineDefinitionError


__all__ = [
    "DuplicateStageError",
    "UnknownStageError",
    "builtin_stage_names",
    "get_stage",
    "register_stage",
    "stage_import_path",
]


class UnknownStageError(PipelineDefinitionError):
    """Raised when a pipeline refers to an unregistered stage."""


class DuplicateStageError(PipelineDefinitionError):
    """Raised when registration would replace an existing stage."""


_BUILTIN_STAGE_IMPORT_PATHS = {
    "cbond": "hotpot.cheminfo.AImodels.cbond.stage:STAGE",
    "ff": "hotpot.cheminfo.forcefields.stage:STAGE",
    "xtb": "hotpot.plugins.xtb.stage:STAGE",
}
_stage_import_paths: Dict[str, str] = dict(_BUILTIN_STAGE_IMPORT_PATHS)
_stage_instances: Dict[str, MolecularStage] = {}


def builtin_stage_names() -> Tuple[str, ...]:
    """Return the ordered names of stages shipped with Hotpot."""

    return tuple(_BUILTIN_STAGE_IMPORT_PATHS)


def register_stage(name: str, import_path: str) -> None:
    """Register a lazy ``module:attribute`` stage import path."""

    if name in _stage_import_paths:
        raise DuplicateStageError(f"pipeline stage {name!r} is already registered")
    _stage_import_paths[name] = import_path


def stage_import_path(name: str) -> str:
    """Return the registered lazy import path for *name*."""

    try:
        return _stage_import_paths[name]
    except KeyError as error:
        raise UnknownStageError(
            f"pipeline stage {name!r} is not registered"
        ) from error


def get_stage(name: str) -> MolecularStage:
    """Resolve and cache one registered molecular stage instance."""

    if name in _stage_instances:
        return _stage_instances[name]

    import_path = stage_import_path(name)
    module_name, separator, attribute_name = import_path.partition(":")
    if not separator or not module_name or not attribute_name:
        raise PipelineDefinitionError(
            f"stage {name!r} has invalid import path {import_path!r}"
        )
    module = importlib.import_module(module_name)
    stage = cast(MolecularStage, getattr(module, attribute_name))
    if not isinstance(stage, MolecularStage):
        raise PipelineDefinitionError(
            f"registered object {import_path!r} is not a molecular stage"
        )
    _stage_instances[name] = stage
    return stage
