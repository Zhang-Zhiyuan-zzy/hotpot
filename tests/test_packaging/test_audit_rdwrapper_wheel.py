from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest

AUDITOR = Path(__file__).with_name("audit_rdwrapper_wheel.py")
WRAPPER_ROOT = "hotpot/cheminfo/rdWrappers/"
VALID_MEMBERS = {
    f"{WRAPPER_ROOT}__init__.py",
    f"{WRAPPER_ROOT}contracts.py",
    f"{WRAPPER_ROOT}native.py",
    f"{WRAPPER_ROOT}sampler.py",
    f"{WRAPPER_ROOT}_rd_native.pyi",
    f"{WRAPPER_ROOT}_rd_native.cpython-311-x86_64-linux-gnu.so",
    f"{WRAPPER_ROOT}_native/THIRD_PARTY_NOTICES.md",
    f"{WRAPPER_ROOT}_native/licenses/COPYING.MPL2",
    f"{WRAPPER_ROOT}_native/licenses/LICENSE",
    f"{WRAPPER_ROOT}_native/licenses/LICENSE.md",
    f"{WRAPPER_ROOT}_native/licenses/LICENSE.txt",
    f"{WRAPPER_ROOT}_native/licenses/LICENSE_1_0.txt",
    f"{WRAPPER_ROOT}_native/licenses/THIRD_PARTY_NOTICES.md",
    f"{WRAPPER_ROOT}_native/licenses/license.txt",
    "hotpot_zzy-0.5.4.5.dist-info/licenses/LICENSE",
}


def write_wheel(path: Path, members: set[str]) -> Path:
    with ZipFile(path, "w") as archive:
        for member in sorted(members):
            archive.writestr(member, b"test")
    return path


def run_audit(*paths: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(AUDITOR), *(str(path) for path in paths)],
        text=True,
        capture_output=True,
        check=False,
    )


def test_accepts_exact_runtime_payload(tmp_path: Path):
    wheel = write_wheel(tmp_path / "valid.whl", VALID_MEMBERS)

    result = run_audit(wheel)

    assert result.returncode == 0, result.stderr
    assert "wheel content audit passed" in result.stdout


@pytest.mark.parametrize(
    "forbidden",
    [
        f"{WRAPPER_ROOT}CMakeLists.txt",
        f"{WRAPPER_ROOT}_native/include/hotpot/rdWrappers/c_api.h",
        f"{WRAPPER_ROOT}_native/src/core.hpp",
        "hotpot/cheminfo/_native/include/hotpot/native/array_view.hpp",
        "third_party/rdkit/rdkit.lock.json",
        "lib/libhotpot_rdwrappers.so",
        "lib/cmake/HotpotRDWrappers/HotpotRDWrappersConfig.cmake",
    ],
)
def test_rejects_build_and_sdk_members(tmp_path: Path, forbidden: str):
    wheel = write_wheel(tmp_path / "forbidden.whl", VALID_MEMBERS | {forbidden})

    result = run_audit(wheel)

    assert result.returncode == 1
    assert "wheel content audit failed" in result.stderr


def test_rejects_missing_required_license(tmp_path: Path):
    members = VALID_MEMBERS - {f"{WRAPPER_ROOT}_native/licenses/LICENSE_1_0.txt"}
    wheel = write_wheel(tmp_path / "missing.whl", members)

    result = run_audit(wheel)

    assert result.returncode == 1
    assert "missing required members" in result.stderr


@pytest.mark.parametrize("extension_count", [0, 2])
def test_rejects_wrong_native_extension_count(tmp_path: Path, extension_count: int):
    extension = f"{WRAPPER_ROOT}_rd_native.cpython-311-x86_64-linux-gnu.so"
    members = VALID_MEMBERS - {extension}
    if extension_count == 2:
        members |= {
            extension,
            f"{WRAPPER_ROOT}_rd_native.cpython-312-x86_64-linux-gnu.so",
        }
    wheel = write_wheel(tmp_path / "extensions.whl", members)

    result = run_audit(wheel)

    assert result.returncode == 1
    assert "expected exactly one rdWrapper native extension" in result.stderr


def test_rejects_multiple_wheel_arguments(tmp_path: Path):
    first = write_wheel(tmp_path / "first.whl", VALID_MEMBERS)
    second = write_wheel(tmp_path / "second.whl", VALID_MEMBERS)

    result = run_audit(first, second)

    assert result.returncode == 2
    assert "expected exactly one wheel path" in result.stderr
