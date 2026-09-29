"""Setuptools entry point and native-extension declarations.

Project metadata and dependency declarations live exclusively in
``pyproject.toml``.
"""

from importlib.util import find_spec
from pathlib import Path

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext


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
    Pybind11Extension(
        "hotpot.cheminfo.geometry._geometry_native",
        [
            "hotpot/cheminfo/geometry/_native/batch.cpp",
            "hotpot/cheminfo/geometry/_native/bindings.cpp",
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
            "hotpot/cheminfo/forcefields/_native/stage_contracts.cpp",
            "hotpot/cheminfo/forcefields/_native/structure_session.cpp",
            "hotpot/cheminfo/forcefields/_native/trajectory.cpp",
            "hotpot/cheminfo/obWrappers/_native/native_bindings.cpp",
            "hotpot/cheminfo/obWrappers/_native/molecule_data.cpp",
            "hotpot/cheminfo/obWrappers/_native/openbabel_adapter.cpp",
            "hotpot/cheminfo/obWrappers/_native/native_engine.cpp",
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
        cmdclass={"build_ext": build_ext},
        zip_safe=False,
    )
