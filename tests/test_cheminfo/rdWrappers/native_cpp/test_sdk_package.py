from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest


@pytest.mark.parametrize("glibcxx_abi", (0, 1))
@pytest.mark.external
def test_installed_sdk_find_package_link_and_run(
    tmp_path: Path,
    glibcxx_abi: int,
):
    prefix_value = os.environ.get("HOTPOT_RDWRAPPERS_TEST_SDK_PREFIX")
    if prefix_value is None:
        pytest.skip("HOTPOT_RDWRAPPERS_TEST_SDK_PREFIX is not set")
    prefix = Path(prefix_value)
    source = Path(__file__).with_name("sdk_consumer")
    build = tmp_path / f"build-abi{glibcxx_abi}"
    configure = [
        "cmake",
        "-S",
        str(source),
        "-B",
        str(build),
        f"-DCMAKE_PREFIX_PATH={prefix}",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DHOTPOT_CONSUMER_GLIBCXX_ABI={glibcxx_abi}",
    ]
    compiler = os.environ.get("HOTPOT_RDWRAPPERS_TEST_CXX")
    if compiler is not None:
        configure.append(f"-DCMAKE_CXX_COMPILER={compiler}")

    subprocess.run(
        configure,
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build), "--parallel", "2"],
        check=True,
    )
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(
        value
        for value in (
            str(prefix / "lib"),
            environment.get("LD_LIBRARY_PATH"),
        )
        if value
    )
    subprocess.run(
        [str(build / "hotpot_rdwrappers_sdk_consumer")],
        check=True,
        env=environment,
    )


def test_locked_build_declares_offline_source_only():
    wrapper_root = Path(__file__).parents[4] / "hotpot/cheminfo/rdWrappers"
    driver = (wrapper_root / "build_locked.sh").read_text(encoding="utf-8")
    cmake = (wrapper_root / "CMakeLists.txt").read_text(encoding="utf-8")

    assert 'HOTPOT_RDKIT_OFFLINE=1 "$lock_dir/fetch_locked_sources.sh"' in driver
    assert "HOTPOT_RDKIT_SOURCE_LOCK_VERIFIED=ON" in driver
    assert 'RDKit_RELEASENAME STREQUAL "2026.09.1"' in cmake
    assert "find_package(RDKit" not in cmake


def test_locked_build_resolves_python_only_for_python_artifacts():
    wrapper_root = Path(__file__).parents[4] / "hotpot/cheminfo/rdWrappers"
    driver = (wrapper_root / "build_locked.sh").read_text(encoding="utf-8")

    conditional = driver.index('if [[ "$build_python" == ON ]]')
    resolution = driver.index('python=$(command -v "$python_command")')
    assert resolution > conditional
