"""Backend identity and version-probe contracts for the xTB plugin."""

from __future__ import annotations

import os
from importlib.util import find_spec
from pathlib import Path
from typing import Protocol

import pytest


PHASE10_BACKEND_API_AVAILABLE = all(
    find_spec(module_name) is not None
    for module_name in (
        "hotpot.plugins.xtb.backend",
        "hotpot.plugins.xtb.contracts",
    )
)
if PHASE10_BACKEND_API_AVAILABLE:
    from hotpot.plugins.xtb.backend import probe_xtb_backend
    from hotpot.plugins.xtb.contracts import XTBExecutableError


requires_phase10_backend = pytest.mark.xfail(
    not PHASE10_BACKEND_API_AVAILABLE,
    reason="Phase 10 xTB backend probe is not implemented yet",
    strict=True,
)


class FakeXTBFactory(Protocol):
    def __call__(
        self,
        scenario: str = "success",
        directory_name: str = "fake_xtb",
    ) -> Path: ...


@requires_phase10_backend
def test_probe_records_absolute_executable_and_parsed_capabilities(
    fake_xtb_factory: FakeXTBFactory,
) -> None:
    executable = fake_xtb_factory()

    backend_info = probe_xtb_backend(
        executable=executable,
        environment=dict(os.environ),
    )

    assert backend_info.executable == executable.resolve()
    assert backend_info.version == "6.7.1"
    assert backend_info.gfnff_max_atomic_number == 86


@requires_phase10_backend
def test_nonzero_version_probe_retains_native_process_evidence(
    fake_xtb_factory: FakeXTBFactory,
) -> None:
    executable = fake_xtb_factory("version_nonzero")

    with pytest.raises(XTBExecutableError) as error:
        probe_xtb_backend(
            executable=executable,
            environment=dict(os.environ),
        )

    assert error.value.process_result.return_code == 19
    assert "fake version probe failure" in error.value.process_result.stderr


@requires_phase10_backend
def test_malformed_version_probe_is_rejected_with_process_evidence(
    fake_xtb_factory: FakeXTBFactory,
) -> None:
    executable = fake_xtb_factory("version_malformed")

    with pytest.raises(XTBExecutableError) as error:
        probe_xtb_backend(
            executable=executable,
            environment=dict(os.environ),
        )

    assert error.value.process_result.return_code == 0
    assert "without a parseable version" in error.value.process_result.stdout
