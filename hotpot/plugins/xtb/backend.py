"""Resolve and probe xTB executable identity and stable capabilities."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Mapping, Optional, Union

from hotpot.plugins._harness import (
    ExecutableNotFoundError,
    ProcessRequest,
    ProcessTimeoutError,
    resolve_executable,
    run_process,
    sha256_file,
)

from .contracts import XTBBackendInfo, XTBExecutableError


__all__ = ["probe_xtb_backend"]


_ExecutablePath = Union[str, os.PathLike[str]]
_VERSION_PATTERN = re.compile(
    r"\bxtb\s+version\s+(\d+\.\d+\.\d+)(?:\s+\(([^)]+)\))?",
    re.IGNORECASE,
)
_STABLE_GFNFF_LIMITS = {"6.7.1": 86}


def _parse_version(
    stdout: str,
    stderr: str,
) -> Optional[tuple[str, Optional[str]]]:
    match = _VERSION_PATTERN.search(f"{stdout}\n{stderr}")
    return None if match is None else (match.group(1), match.group(2))


def probe_xtb_backend(
    executable: Optional[_ExecutablePath] = None,
    *,
    environment: Optional[Mapping[str, str]] = None,
    timeout_seconds: Optional[float] = 30.0,
) -> XTBBackendInfo:
    """Resolve xTB and return identity plus conservatively verified limits."""

    process_environment = dict(os.environ if environment is None else environment)
    try:
        executable_path = resolve_executable(
            "xtb",
            explicit=executable,
            env_var="HOTPOT_XTB_EXECUTABLE",
            env=process_environment,
        )
        process_result = run_process(
            ProcessRequest(
                argv=(str(executable_path), "--version"),
                cwd=executable_path.parent,
                env=process_environment,
                timeout_seconds=timeout_seconds,
            )
        )
    except (ExecutableNotFoundError, ProcessTimeoutError) as error:
        raise XTBExecutableError(str(error)) from error
    if process_result.return_code != 0:
        raise XTBExecutableError(
            f"xTB version probe exited with code {process_result.return_code}",
            process_result,
        )

    parsed_version = _parse_version(process_result.stdout, process_result.stderr)
    if parsed_version is None:
        raise XTBExecutableError(
            "xTB version probe did not report a parseable semantic version",
            process_result,
        )

    version, revision = parsed_version
    return XTBBackendInfo(
        executable=executable_path,
        version=version,
        revision=revision,
        executable_sha256=sha256_file(executable_path),
        probe_result=process_result,
        gfn_xtb_max_atomic_number=86,
        gfnff_max_atomic_number=_STABLE_GFNFF_LIMITS.get(version),
    )
