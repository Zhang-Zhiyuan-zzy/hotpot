from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).parents[2]
WRAPPER_ROOT = ROOT / "hotpot/cheminfo/rdWrappers"
LOCK_ROOT = ROOT / "third_party/rdkit"


def test_setup_uses_one_locked_cmake_extension_path():
    setup_source = (ROOT / "setup.py").read_text(encoding="utf-8")

    assert "class LockedCMakeExtension(Extension)" in setup_source
    assert "class HotpotBuildExt(build_ext)" in setup_source
    assert (
        'LockedCMakeExtension("hotpot.cheminfo.rdWrappers._rd_native")' in setup_source
    )
    assert '"HOTPOT_RDWRAPPERS_BUILD_SDK": "OFF"' in setup_source
    assert '"HOTPOT_RDWRAPPERS_BUILD_PYTHON": "ON"' in setup_source
    assert '"HOTPOT_RDWRAPPERS_INSTALL_COMPONENT": "rdwrappers_python"' in setup_source
    assert "self.get_ext_fullpath(ext.name)" in setup_source
    assert setup_source.index("fetch_locked_sources.sh") < setup_source.index(
        'wrapper_root / "build_locked.sh"'
    )


def test_locked_driver_rejects_unknown_profile_before_configure(tmp_path: Path):
    environment = {
        "PATH": "/usr/bin:/bin",
        "HOTPOT_RDWRAPPERS_BUILD_PROFILE": "unsupported",
    }
    result = subprocess.run(
        [
            "bash",
            str(WRAPPER_ROOT / "build_locked.sh"),
            str(tmp_path / "cache"),
            str(tmp_path / "work"),
            sys.executable,
        ],
        text=True,
        capture_output=True,
        env=environment,
    )

    assert result.returncode != 0
    assert "unsupported rdWrappers build profile" in result.stderr
    assert not (tmp_path / "work/build/configure-output.txt").exists()


def test_locked_driver_profiles_are_exact_and_separate():
    driver = (WRAPPER_ROOT / "build_locked.sh").read_text(encoding="utf-8")
    lock = json.loads((LOCK_ROOT / "rdkit.lock.json").read_text())

    assert "probe)" in driver
    assert "manylinux_2_28_x86_64)" in driver
    assert "HOTPOT_RDWRAPPERS_TOOLCHAIN_PREFIX" in driver
    assert '"14.3.0"' in driver
    assert '"14.2.1"' in driver
    assert '"glibc 2.28"' in driver
    assert '"cmake version 4.4.3"' in driver
    assert '"1.13.2.git.kitware.jobserver-pipe-1"' in driver

    probe = lock["probe_toolchain"]
    release = lock["release_toolchain"]
    assert probe["compiler"]["cxx_version"] == "14.3.0"
    assert release["profile"] == "manylinux_2_28_x86_64"
    assert release["compiler"]["cxx_version"] == "14.2.1"
    assert release["glibc_baseline"] == "2.28"
    assert release["binutils"]["rpm"] == ("gcc-toolset-14-binutils-2.41-4.el8_10.1")
    assert release["ninja"]["cli_version"] == ("1.13.2.git.kitware.jobserver-pipe-1")
    assert release["container"]["digest"] == (
        "sha256:39df0042d5cc900b085aa25a0659368b42a0006c54c474299b785b44c1b4ff82"
    )


def test_locked_driver_does_not_resolve_python_for_sdk_only_build():
    driver = (WRAPPER_ROOT / "build_locked.sh").read_text(encoding="utf-8")

    conditional = driver.index('if [[ "$build_python" == ON ]]')
    resolution = driver.index('python=$(command -v "$python_command")')

    assert resolution > conditional


def test_sdist_manifest_contains_locked_native_build_inputs():
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    required_rules = (
        "include hotpot/cheminfo/rdWrappers/CMakeLists.txt",
        "include hotpot/cheminfo/rdWrappers/build_locked.sh",
        "recursive-include hotpot/cheminfo/rdWrappers/cmake *.cmake *.in",
        "recursive-include hotpot/cheminfo/rdWrappers/_native",
        "recursive-include hotpot/cheminfo/_native/include",
        "include third_party/rdkit/rdkit.lock.json",
        "include third_party/rdkit/fetch_locked_sources.sh",
        "recursive-include third_party/rdkit/probe",
    )

    for rule in required_rules:
        assert rule in manifest


def test_build_requirements_and_wheel_workflow_pin_release_profile():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    workflow = (ROOT / ".github/workflows/publish_pypi.yml").read_text(encoding="utf-8")

    assert '"ninja==1.13.2"' in pyproject
    assert (
        "quay.io/pypa/manylinux_2_28_x86_64@sha256:"
        "39df0042d5cc900b085aa25a0659368b42a0006c54c474299b785b44c1b4ff82" in workflow
    )
    assert "HOTPOT_RDWRAPPERS_BUILD_PROFILE=manylinux_2_28_x86_64" in workflow
    assert "auditwheel repair" in workflow
    assert "rdWrappers.runtime_info()" in workflow
    assert "rdWrappers.generate_conformers" in workflow
