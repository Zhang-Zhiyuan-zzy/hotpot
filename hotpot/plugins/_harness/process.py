"""Run native processes while preserving their exact observable streams."""

from __future__ import annotations

import subprocess
from time import monotonic

from .contracts import NativeProcessResult, ProcessRequest


__all__ = ["ProcessTimeoutError", "run_process"]


class ProcessTimeoutError(TimeoutError):
    """Raised when a native process exceeds its requested timeout."""

    def __init__(self, request: ProcessRequest) -> None:
        self.request = request
        super().__init__(
            f"native process exceeded {request.timeout_seconds} seconds: "
            f"{request.argv!r}"
        )


def run_process(request: ProcessRequest) -> NativeProcessResult:
    """Execute one request without a shell and return process facts."""

    started_at = monotonic()
    try:
        completed = subprocess.run(
            request.argv,
            cwd=request.cwd,
            env=request.env,
            shell=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=request.timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise ProcessTimeoutError(request) from error

    return NativeProcessResult(
        argv=request.argv,
        cwd=request.cwd,
        return_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        elapsed_seconds=monotonic() - started_at,
    )
