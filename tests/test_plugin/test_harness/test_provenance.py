"""Artifact and process-provenance contracts for the external harness."""

from __future__ import annotations

from pathlib import Path

import pytest


pytestmark = pytest.mark.xfail(
    strict=True,
    reason="Phase 8 implements hotpot.plugins._harness",
)


def test_sha256_file_returns_the_content_digest(tmp_path: Path) -> None:
    from hotpot.plugins._harness.provenance import sha256_file

    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"abc")

    assert sha256_file(artifact) == (
        "ba7816bf8f01cfea414140de5dae2223"
        "b00361a396177a9cb410ff61f20015ad"
    )


def test_process_provenance_records_execution_and_artifact_lineage(
    tmp_path: Path,
) -> None:
    from hotpot.plugins._harness.contracts import NativeProcessResult
    from hotpot.plugins._harness.provenance import build_process_provenance

    executable = tmp_path / "bin" / "probe"
    executable.parent.mkdir()
    executable.write_bytes(b"executable")
    artifact = tmp_path / "result.dat"
    artifact.write_bytes(b"abc")
    result = NativeProcessResult(
        argv=(os_fspath(executable), "--version"),
        cwd=tmp_path,
        return_code=0,
        stdout="probe 1.0\n",
        stderr="",
        elapsed_seconds=0.125,
    )

    provenance = build_process_provenance(
        executable=executable,
        result=result,
        artifacts=(artifact,),
    )

    assert provenance.executable == executable.resolve()
    assert provenance.argv == result.argv
    assert provenance.cwd == result.cwd
    assert provenance.return_code == result.return_code
    assert provenance.elapsed_seconds == result.elapsed_seconds
    assert len(provenance.artifacts) == 1
    assert provenance.artifacts[0].path == artifact.resolve()
    assert provenance.artifacts[0].sha256 == (
        "ba7816bf8f01cfea414140de5dae2223"
        "b00361a396177a9cb410ff61f20015ad"
    )


def os_fspath(path: Path) -> str:
    """Return the platform path string without importing plugin code."""

    return str(path)
