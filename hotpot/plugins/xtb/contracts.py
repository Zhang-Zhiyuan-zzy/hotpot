"""Typed contracts for invoking an official xTB executable."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, Optional, Tuple

from hotpot.plugins._harness import NativeProcessResult, ProcessProvenance


__all__ = [
    "GFNXTBMethod",
    "XTBApplicabilityError",
    "XTBArtifact",
    "XTBBackendInfo",
    "XTBError",
    "XTBExecutableError",
    "XTBExecutionError",
    "XTBInputError",
    "XTBMethod",
    "XTBRequest",
    "XTBResultError",
    "XTBRunReport",
    "XTBTask",
]


class XTBMethod(str, Enum):
    """Numerical methods exposed by the official xTB command line."""

    GFNFF = "gfnff"
    GFN0_XTB = "gfn0"
    GFN1_XTB = "gfn1"
    GFN2_XTB = "gfn2"


class GFNXTBMethod(str, Enum):
    """Electronic GFN-xTB methods accepted by the GFN-xTB workflow."""

    GFN0_XTB = "gfn0"
    GFN1_XTB = "gfn1"
    GFN2_XTB = "gfn2"


class XTBTask(str, Enum):
    """Calculation tasks supported by the xTB runner."""

    SINGLEPOINT = "singlepoint"
    OPTIMIZE = "optimize"


@dataclass(frozen=True)
class XTBBackendInfo:
    """Identity and verified capability facts for one xTB executable."""

    executable: Path
    version: str
    revision: Optional[str]
    executable_sha256: str
    probe_result: NativeProcessResult
    gfn_xtb_max_atomic_number: int
    gfnff_max_atomic_number: Optional[int]


@dataclass(frozen=True)
class XTBArtifact:
    """Content-addressed file emitted by one xTB invocation."""

    name: str
    path: Path
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class XTBRequest:
    """Describe one low-level xTB invocation in an existing workspace."""

    backend_info: XTBBackendInfo
    method: XTBMethod
    task: XTBTask
    input_path: Path
    work_directory: Path
    charge: int
    unpaired_electrons: Optional[int]
    environment: Mapping[str, str]
    timeout_seconds: Optional[float] = None


@dataclass(frozen=True)
class XTBRunReport:
    """Immutable process facts and xTB artifact evidence from one run."""

    backend_info: XTBBackendInfo
    requested_method: XTBMethod
    effective_method: XTBMethod
    task: XTBTask
    charge: int
    unpaired_electrons: Optional[int]
    work_directory: Path
    argv: Tuple[str, ...]
    return_code: int
    stdout: str
    stderr: str
    elapsed_seconds: float
    process_succeeded: bool
    converged: bool
    artifacts: Mapping[str, XTBArtifact]
    provenance: ProcessProvenance
    energy_hartree: Optional[float] = None
    energy_unit: str = "hartree"
    gradient_norm: Optional[float] = None
    atom_order_verified: bool = False
    coordinates_committed: bool = False


class XTBError(RuntimeError):
    """Base class for explicit xTB plugin failures."""


class XTBExecutableError(XTBError):
    """Raised when an xTB executable cannot provide valid identity evidence."""

    def __init__(
        self,
        message: str,
        process_result: Optional[NativeProcessResult] = None,
    ) -> None:
        self.process_result = process_result
        super().__init__(message)


class XTBInputError(XTBError):
    """Raised when a request cannot be represented by the selected method."""


class XTBApplicabilityError(XTBInputError):
    """Raised when a backend cannot support an input element domain."""


class XTBExecutionError(XTBError):
    """Raised when the native process exits unsuccessfully."""

    def __init__(self, message: str, report: XTBRunReport) -> None:
        self.report = report
        super().__init__(message)


class XTBResultError(XTBError):
    """Raised when a nominally successful process lacks required evidence."""

    def __init__(self, message: str, report: XTBRunReport) -> None:
        self.report = report
        super().__init__(message)
