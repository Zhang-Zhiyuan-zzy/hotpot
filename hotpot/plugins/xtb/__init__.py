"""Public xTB plugin exports."""

from .adapter import (
    commit_xtb_coordinates,
    molecule_to_xtb_geometry,
    parse_xtb_artifacts,
    prepare_xtb_input,
)
from .backend import probe_xtb_backend
from .capabilities import validate_element_support
from .contracts import (
    GFNXTBMethod,
    XTBApplicabilityError,
    XTBArtifact,
    XTBBackendInfo,
    XTBError,
    XTBExecutableError,
    XTBGeometry,
    XTBExecutionError,
    XTBInputError,
    XTBMethod,
    XTBParsedResult,
    XTBRequest,
    XTBResultError,
    XTBRunReport,
    XTBTask,
)
from .core import XtbCalculator, xtb_batch_run
from .runner import run_xtb


__all__ = [
    "GFNXTBMethod",
    "XTBApplicabilityError",
    "XTBArtifact",
    "XTBBackendInfo",
    "XTBError",
    "XTBExecutableError",
    "XTBGeometry",
    "XTBExecutionError",
    "XTBInputError",
    "XTBMethod",
    "XTBParsedResult",
    "XTBRequest",
    "XTBResultError",
    "XTBRunReport",
    "XTBTask",
    "XtbCalculator",
    "commit_xtb_coordinates",
    "molecule_to_xtb_geometry",
    "parse_xtb_artifacts",
    "probe_xtb_backend",
    "prepare_xtb_input",
    "run_xtb",
    "validate_element_support",
    "xtb_batch_run",
]
