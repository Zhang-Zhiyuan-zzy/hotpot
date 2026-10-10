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
from .stream import (
    XTBStreamError,
    XTBStreamMetadata,
    XTBStreamProvenance,
    XTBStreamRecord,
    metadata_from_report,
    read_sdf_records,
    write_sdf_records,
)
from .workflow import run_gfn_xtb, run_gfnff


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
    "XTBStreamError",
    "XTBStreamMetadata",
    "XTBStreamProvenance",
    "XTBStreamRecord",
    "XTBTask",
    "XtbCalculator",
    "commit_xtb_coordinates",
    "molecule_to_xtb_geometry",
    "metadata_from_report",
    "parse_xtb_artifacts",
    "probe_xtb_backend",
    "prepare_xtb_input",
    "run_xtb",
    "run_gfn_xtb",
    "run_gfnff",
    "read_sdf_records",
    "validate_element_support",
    "write_sdf_records",
    "xtb_batch_run",
]
