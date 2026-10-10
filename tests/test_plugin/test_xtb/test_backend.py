"""Backend identity and version-probe contracts for the xTB plugin."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

import pytest

from hotpot.plugins._harness import sha256_file
from hotpot.plugins.xtb.backend import probe_xtb_backend
from hotpot.plugins.xtb.contracts import XTBExecutableError


class FakeXTBFactory(Protocol):
    def __call__(
        self,
        scenario: str = "success",
        directory_name: str = "fake_xtb",
    ) -> Path: ...


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
    assert backend_info.executable_sha256 == sha256_file(executable)
    assert backend_info.probe_result.argv == (
        str(executable.resolve()),
        "--version",
    )


def test_probe_honors_hotpot_xtb_executable_environment(
    fake_xtb_factory: FakeXTBFactory,
) -> None:
    executable = fake_xtb_factory()
    environment = dict(os.environ)
    environment["HOTPOT_XTB_EXECUTABLE"] = str(executable)

    backend_info = probe_xtb_backend(environment=environment)

    assert backend_info.executable == executable.resolve()


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
