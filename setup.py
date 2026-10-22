"""Setuptools entry point and native-extension declarations.

Project metadata and dependency declarations live exclusively in
``pyproject.toml``.
"""

from importlib.util import find_spec
import os
from pathlib import Path
import shutil
import subprocess
import sys

from setuptools import Extension, setup
from pybind11.setup_helpers import Pybind11Extension, build_ext


REPOSITORY_ROOT = Path(__file__).resolve().parent


class LockedCMakeExtension(Extension):
    """Extension built only through the pinned native-RDKit driver."""

    def __init__(self, name: str):
        super().__init__(name, sources=[])


class HotpotBuildExt(build_ext):
    """Build pybind11 modules and the locked rdWrapper CMake module."""

    def build_extension(self, ext):
        if isinstance(ext, LockedCMakeExtension):
            self._build_locked_cmake_extension(ext)
            return
        super().build_extension(ext)

    def _build_locked_cmake_extension(self, ext: LockedCMakeExtension) -> None:
        wrapper_root = REPOSITORY_ROOT / "hotpot/cheminfo/rdWrappers"
        lock_root = REPOSITORY_ROOT / "third_party/rdkit"
        source_cache = (
            Path(
                os.environ.get(
                    "HOTPOT_RDKIT_SOURCE_CACHE",
                    str(Path(self.build_temp).resolve() / "rdkit-source-cache"),
                )
            )
            .expanduser()
            .resolve()
        )
        work_dir = (
            Path(self.build_temp).resolve()
            / f"rdwrapper-{sys.implementation.cache_tag}"
        )
        extension_path = Path(self.get_ext_fullpath(ext.name)).resolve()
        install_root = Path(self.build_lib).resolve()

        fetch_environment = os.environ.copy()
        subprocess.run(
            [
                "bash",
                str(lock_root / "fetch_locked_sources.sh"),
                str(source_cache),
            ],
            check=True,
            cwd=REPOSITORY_ROOT,
            env=fetch_environment,
        )

        if work_dir.exists():
            shutil.rmtree(work_dir)

        build_environment = os.environ.copy()
        build_environment.update(
            {
                "HOTPOT_RDKIT_OFFLINE": "1",
                "HOTPOT_RDWRAPPERS_BUILD_SDK": "OFF",
                "HOTPOT_RDWRAPPERS_BUILD_PYTHON": "ON",
                "HOTPOT_RDWRAPPERS_PYTHON_OUTPUT_DIR": str(extension_path.parent),
                "HOTPOT_RDWRAPPERS_INSTALL_PREFIX": str(install_root),
                "HOTPOT_RDWRAPPERS_INSTALL_COMPONENT": "rdwrappers_python",
            }
        )
        subprocess.run(
            [
                "bash",
                str(wrapper_root / "build_locked.sh"),
                str(source_cache),
                str(work_dir),
                sys.executable,
            ],
            check=True,
            cwd=REPOSITORY_ROOT,
            env=build_environment,
        )
        if not extension_path.is_file():
            raise RuntimeError(
                f"the locked rdWrapper build did not create {extension_path}"
            )


def _openbabel_build_paths() -> tuple[str, str]:
    """Locate headers and libraries shipped by the selected Open Babel wheel."""
    specification = find_spec("openbabel")
    if specification is None or specification.origin is None:
        raise RuntimeError(
            "Open Babel must be installed before building native extensions"
        )
    package_dir = Path(specification.origin).resolve().parent
    include_dir = package_dir / "include" / "openbabel3"
    library_dir = package_dir / "lib"
    if not include_dir.is_dir() or not library_dir.is_dir():
        raise RuntimeError(
            "the installed Open Babel package does not provide native "
            "headers and libraries"
        )
    return str(include_dir), str(library_dir)


openbabel_include_dir, openbabel_library_dir = _openbabel_build_paths()


ext_modules = [
    LockedCMakeExtension("hotpot.cheminfo.rdWrappers._rd_native"),
    Pybind11Extension(
        "hotpot.cheminfo.search._search_native",
        [
            "hotpot/cheminfo/search/_native/bindings.cpp",
            "hotpot/cheminfo/search/_native/graph_data.cpp",
            "hotpot/cheminfo/search/_native/matcher.cpp",
            "hotpot/cheminfo/search/_native/production.cpp",
        ],
        cxx_std=17,
    ),
    Pybind11Extension(
        "hotpot.cheminfo.geometry._geometry_native",
        [
            "hotpot/cheminfo/geometry/_native/batch.cpp",
            "hotpot/cheminfo/geometry/_native/bindings.cpp",
            "hotpot/cheminfo/geometry/_native/construction.cpp",
            "hotpot/cheminfo/geometry/_native/cycle_surface.cpp",
            "hotpot/cheminfo/geometry/_native/nonplanar_surface.cpp",
            "hotpot/cheminfo/geometry/_native/nonplanar_segment.cpp",
            "hotpot/cheminfo/geometry/_native/planar_predicates.cpp",
            "hotpot/cheminfo/geometry/_native/prepared_cycle.cpp",
            "hotpot/cheminfo/geometry/_native/primitives.cpp",
            "hotpot/cheminfo/geometry/_native/segment_cycle.cpp",
            "hotpot/cheminfo/geometry/_native/spatial.cpp",
            "hotpot/cheminfo/geometry/_native/triangle_predicates.cpp",
        ],
        cxx_std=17,
    ),
    Pybind11Extension(
        "hotpot.cheminfo.graph._relevant_cycles",
        [
            "hotpot/cheminfo/graph/_native/bindings.cpp",
            "hotpot/cheminfo/graph/_native/relevant_cycles.cpp",
        ],
        cxx_std=17,
    ),
    Pybind11Extension(
        "hotpot.cheminfo.obWrappers._ob_native",
        [
            "hotpot/cheminfo/forcefields/_native/bindings.cpp",
            "hotpot/cheminfo/forcefields/_native/contracts.cpp",
            "hotpot/cheminfo/forcefields/_native/coordination_stage.cpp",
            "hotpot/cheminfo/forcefields/_native/optimization_stage.cpp",
            "hotpot/cheminfo/forcefields/_native/placement_candidates.cpp",
            "hotpot/cheminfo/forcefields/_native/placement_engine.cpp",
            "hotpot/cheminfo/forcefields/_native/placement_policy.cpp",
            "hotpot/cheminfo/forcefields/_native/radii.cpp",
            "hotpot/cheminfo/forcefields/_native/session_optimization.cpp",
            "hotpot/cheminfo/forcefields/_native/stage_contracts.cpp",
            "hotpot/cheminfo/forcefields/_native/structure_session.cpp",
            "hotpot/cheminfo/forcefields/_native/target_selection.cpp",
            "hotpot/cheminfo/forcefields/_native/topology_workspace.cpp",
            "hotpot/cheminfo/forcefields/_native/trajectory.cpp",
            "hotpot/cheminfo/forcefields/_native/untangling_engine.cpp",
            "hotpot/cheminfo/forcefields/_native/workflow_stage.cpp",
            "hotpot/cheminfo/geometry/_native/batch.cpp",
            "hotpot/cheminfo/geometry/_native/construction.cpp",
            "hotpot/cheminfo/geometry/_native/cycle_surface.cpp",
            "hotpot/cheminfo/geometry/_native/nonplanar_surface.cpp",
            "hotpot/cheminfo/geometry/_native/nonplanar_segment.cpp",
            "hotpot/cheminfo/geometry/_native/planar_predicates.cpp",
            "hotpot/cheminfo/geometry/_native/prepared_cycle.cpp",
            "hotpot/cheminfo/geometry/_native/primitives.cpp",
            "hotpot/cheminfo/geometry/_native/segment_cycle.cpp",
            "hotpot/cheminfo/geometry/_native/spatial.cpp",
            "hotpot/cheminfo/geometry/_native/triangle_predicates.cpp",
            "hotpot/cheminfo/graph/_native/relevant_cycles.cpp",
            "hotpot/cheminfo/obWrappers/_native/native_bindings.cpp",
            "hotpot/cheminfo/obWrappers/_native/molecule_data.cpp",
            "hotpot/cheminfo/obWrappers/_native/openbabel_adapter.cpp",
            "hotpot/cheminfo/obWrappers/_native/native_engine.cpp",
            "hotpot/cheminfo/obWrappers/_native/optimization_checks.cpp",
            "hotpot/cheminfo/obWrappers/_native/optimization_controller.cpp",
            "hotpot/cheminfo/obWrappers/_native/optimization_operation.cpp",
            "hotpot/cheminfo/obWrappers/_native/registry.cpp",
            "hotpot/cheminfo/obWrappers/_native/phosphorus_builder.cpp",
            "hotpot/cheminfo/obWrappers/_native/degenerate_torsion.cpp",
        ],
        include_dirs=[openbabel_include_dir],
        library_dirs=[openbabel_library_dir],
        libraries=["openbabel", "dl"],
        define_macros=[("_GLIBCXX_USE_CXX11_ABI", "0")],
        extra_link_args=[
            "-Wl,-rpath,$ORIGIN/../../../openbabel/lib",
        ],
        cxx_std=17,
    ),
]


if __name__ == "__main__":
    setup(
        ext_modules=ext_modules,
        cmdclass={"build_ext": HotpotBuildExt},
        zip_safe=False,
    )
