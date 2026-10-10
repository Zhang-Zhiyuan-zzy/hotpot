"""Native-process execution contracts for the external-process harness."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest


def test_process_uses_requested_cwd_environment_and_exact_streams(
    tmp_path: Path,
) -> None:
    from hotpot.plugins._harness.contracts import ProcessRequest
    from hotpot.plugins._harness.process import run_process

    script = (
        "from pathlib import Path; import os, sys; "
        "print(Path.cwd()); "
        "print(os.environ['HOTPOT_HARNESS_VALUE']); "
        "print('native warning', file=sys.stderr)"
    )
    env = dict(os.environ)
    env["HOTPOT_HARNESS_VALUE"] = "preserved value"
    request = ProcessRequest(
        argv=(sys.executable, "-c", script),
        cwd=tmp_path,
        env=env,
    )

    result = run_process(request)

    assert result.argv == request.argv
    assert result.cwd == tmp_path
    assert result.return_code == 0
    assert result.stdout == f"{tmp_path}\npreserved value\n"
    assert result.stderr == "native warning\n"
    assert result.elapsed_seconds >= 0.0


def test_process_timeout_is_explicit(tmp_path: Path) -> None:
    from hotpot.plugins._harness.contracts import ProcessRequest
    from hotpot.plugins._harness.process import ProcessTimeoutError, run_process

    request = ProcessRequest(
        argv=(sys.executable, "-c", "import time; time.sleep(10)"),
        cwd=tmp_path,
        env=dict(os.environ),
        timeout_seconds=0.05,
    )

    with pytest.raises(ProcessTimeoutError):
        run_process(request)
