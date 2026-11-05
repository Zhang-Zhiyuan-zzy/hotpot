from __future__ import annotations

import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


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
        check=False,
    )

    assert result.returncode != 0
    assert "unsupported rdWrappers build profile" in result.stderr
    assert not (tmp_path / "work/build/configure-output.txt").exists()


@pytest.mark.parametrize("profile", [None, "local"])
@pytest.mark.parametrize("compiler_overrides", [False, True])
def test_local_driver_reaches_offline_cache_without_probe_prefix(
    tmp_path: Path, profile: str | None, compiler_overrides: bool
):
    tools = tmp_path / "tools"
    tools.mkdir()
    for name in ("cmake", "ninja", "cc", "c++", "custom-cc", "custom-cxx"):
        command = tools / name
        command.write_text("#!/bin/sh\nexit 91\n", encoding="utf-8")
        command.chmod(0o755)
    environment = {"PATH": f"{tools}:/usr/bin:/bin"}
    if profile is not None:
        environment["HOTPOT_RDWRAPPERS_BUILD_PROFILE"] = profile
    if compiler_overrides:
        environment.update({"CC": "custom-cc", "CXX": str(tools / "custom-cxx")})

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
        check=False,
    )

    assert result.returncode != 0
    assert "offline cache is missing locked source:" in result.stderr
    assert "rdkit-Release_2026_09_1.tar.gz" in result.stderr
    assert not (tmp_path / "work/build/configure-output.txt").exists()


def test_locked_driver_preserves_pythonpath_for_cmake_launcher(tmp_path: Path):
    wrapper = tmp_path / "repository/hotpot/cheminfo/rdWrappers"
    wrapper.mkdir(parents=True)
    driver = wrapper / "build_locked.sh"
    driver.write_bytes((WRAPPER_ROOT / "build_locked.sh").read_bytes())
    lock = tmp_path / "repository/third_party/rdkit"
    lock.mkdir(parents=True)
    fetcher = lock / "fetch_locked_sources.sh"
    fetcher.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fetcher.chmod(0o755)

    cache = tmp_path / "cache"
    cache.mkdir()
    archives = (
        "rdkit-Release_2026_09_1.tar.gz",
        "boost_1_86_0.tar.bz2",
        "Catch2-v3.4.0.tar.gz",
        "better-enums-c35576bed0295689540b39873126129adfa0b4c8.tar.gz",
        "eigen-3.4.0.tar.gz",
        "RingDecomposerLib-v1.1.3_rdkit.tar.gz",
    )
    for name in archives:
        mode = "w:bz2" if name.endswith(".bz2") else "w:gz"
        with tarfile.open(cache / name, mode) as archive:
            directory = tarfile.TarInfo("source/")
            directory.type = tarfile.DIRTYPE
            archive.addfile(directory)

    tools = tmp_path / "tools"
    tools.mkdir()
    for name in ("ninja", "cc", "c++"):
        command = tools / name
        command.write_text("#!/bin/sh\nexit 91\n", encoding="utf-8")
        command.chmod(0o755)
    cmake = tools / "cmake"
    cmake.write_text(
        f"#!{sys.executable}\nfrom isolated_cmake import main\nmain()\n",
        encoding="utf-8",
    )
    cmake.chmod(0o755)
    modules = tmp_path / "isolated-site"
    modules.mkdir()
    calls = tmp_path / "cmake-calls.jsonl"
    (modules / "isolated_cmake.py").write_text(
        "import json, sys\n"
        "def main():\n"
        f"    with open({str(calls)!r}, 'a') as stream:\n"
        "        stream.write(json.dumps(sys.argv[1:]) + '\\n')\n",
        encoding="utf-8",
    )
    work = tmp_path / "work"

    result = subprocess.run(
        ["bash", str(driver), str(cache), str(work)],
        text=True,
        capture_output=True,
        env={
            "PATH": f"{tools}:/usr/bin:/bin",
            "PYTHONPATH": str(modules),
            "HOTPOT_RDWRAPPERS_BUILD_PYTHON": "OFF",
        },
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    configure, build = [json.loads(line) for line in calls.read_text().splitlines()]
    assert configure[:2] == ["-S", str(work / "source/rdkit")]
    assert "-DCMAKE_DISABLE_FIND_PACKAGE_Python=ON" in configure
    assert build == ["--build", str(work / "build"), "--target", "hotpot_rdwrappers"]


def test_explicit_probe_still_requires_toolchain_prefix(tmp_path: Path):
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
        env={"PATH": "/usr/bin:/bin", "HOTPOT_RDWRAPPERS_BUILD_PROFILE": "probe"},
        check=False,
    )

    assert result.returncode != 0
    assert "probe profile requires HOTPOT_RDWRAPPERS_TOOLCHAIN_PREFIX" in result.stderr
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize(
    ("variable", "label"), [("CC", "local C compiler"), ("CXX", "local C++ compiler")]
)
def test_local_driver_rejects_missing_explicit_compiler(
    tmp_path: Path, variable: str, label: str
):
    tools = tmp_path / "tools"
    tools.mkdir()
    for name in ("cmake", "ninja", "cc", "c++"):
        command = tools / name
        command.write_text("#!/bin/sh\nexit 91\n", encoding="utf-8")
        command.chmod(0o755)
    missing = str(tmp_path / "missing-compiler")

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
        env={"PATH": f"{tools}:/usr/bin:/bin", variable: missing},
        check=False,
    )

    assert result.returncode != 0
    assert f"{label} is unavailable: {missing}" in result.stderr
    assert not (tmp_path / "cache").exists()


def test_locked_driver_profiles_are_exact_and_separate():
    driver = (WRAPPER_ROOT / "build_locked.sh").read_text(encoding="utf-8")
    lock = json.loads((LOCK_ROOT / "rdkit.lock.json").read_text())

    assert "local)" in driver
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
        "recursive-include hotpot/cheminfo/adapters/native/include",
        "include third_party/rdkit/rdkit.lock.json",
        "include third_party/rdkit/fetch_locked_sources.sh",
        "recursive-include third_party/rdkit/probe",
    )

    for rule in required_rules:
        assert rule in manifest


def test_wheel_excludes_rdwrapper_sdk_and_build_inputs():
    with (ROOT / "pyproject.toml").open("rb") as stream:
        setuptools = tomllib.load(stream)["tool"]["setuptools"]

    package_data = setuptools["package-data"]
    assert package_data["hotpot.cheminfo.adapters.visualization"] == [
        "fonts/Arial/*.ttf", "draw.md"
    ]
    assert package_data["hotpot.cheminfo.rdWrappers"] == ["_rd_native.pyi"]
    assert package_data["hotpot.cheminfo.rdWrappers._native"] == [
        "THIRD_PARTY_NOTICES.md"
    ]

    excluded = setuptools["exclude-package-data"]
    assert excluded["hotpot.cheminfo.adapters.native.include.hotpot.native"] == ["*"]
    assert excluded["hotpot.cheminfo.rdWrappers"] == [
        "CMakeLists.txt",
        "build_locked.sh",
    ]
    assert excluded["hotpot.cheminfo.rdWrappers.cmake"] == ["*"]
    assert excluded["hotpot.cheminfo.rdWrappers._native"] == ["BUILDING.md"]
    assert excluded["hotpot.cheminfo.rdWrappers._native.include.hotpot.rdWrappers"] == [
        "*"
    ]
    assert excluded["hotpot.cheminfo.rdWrappers._native.src"] == ["*"]


def test_build_requirements_and_wheel_workflow_pin_release_profile():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    workflow = (ROOT / ".github/workflows/publish_pypi.yml").read_text(encoding="utf-8")

    assert '"cmake==4.4.3"' in pyproject
    assert '"ninja==1.13.2"' in pyproject
    assert (
        "quay.io/pypa/manylinux_2_28_x86_64@sha256:"
        "39df0042d5cc900b085aa25a0659368b42a0006c54c474299b785b44c1b4ff82" in workflow
    )
    assert "HOTPOT_RDWRAPPERS_BUILD_PROFILE=manylinux_2_28_x86_64" in workflow
    assert "docker; disable_host_mount: true" in workflow
    assert "auditwheel repair" in workflow
    assert "rdWrappers.runtime_info()" in workflow
    assert "rdWrappers.generate_conformers" in workflow
    assert "from rdkit import Chem, rdBase" in workflow
    assert "importlib.metadata.version('rdkit') == '2026.9.1'" in workflow
    assert (
        "python tests/test_packaging/audit_rdwrapper_wheel.py wheelhouse/*.whl"
        in workflow
    )
