"""Setuptools entry point and native-extension declaration.

Project metadata and dependency declarations live exclusively in
``pyproject.toml``.
"""

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext


ext_modules = [
    Pybind11Extension(
        "hotpot.cheminfo.graph._relevant_cycles",
        [
            "hotpot/cheminfo/graph/_native/bindings.cpp",
            "hotpot/cheminfo/graph/_native/relevant_cycles.cpp",
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
