"""Process-wide test initialization and explicit test-tier switches."""

# PyPI Open Babel 3.2.1 can crash if pybel loads before RDKit in one process.
# Hotpot enforces the same deterministic order in its package entry point.
from rdkit import Chem as _rdkit_chem  # noqa: F401


def pytest_addoption(parser):
    parser.addoption(
        "--run-external",
        action="store_true",
        default=False,
        help="run tests that launch separately installed software",
    )
    parser.addoption(
        "--run-optional",
        action="store_true",
        default=False,
        help="run tests that require optional Hotpot dependency groups",
    )
    parser.addoption(
        "--run-scientific-boundary",
        action="store_true",
        default=False,
        help="run governed scientific-boundary tests",
    )
    parser.addoption(
        "--run-high-compute",
        action="store_true",
        default=False,
        help="run tests that require substantial CPU, GPU, or wall time",
    )
    parser.addoption(
        "--run-model-artifacts",
        action="store_true",
        default=False,
        help="run inference examples that require installed model artifacts",
    )


def pytest_configure(config):
    markers = {
        "external": "requires separately installed software or a service",
        "optional_dependency": "requires an optional Hotpot dependency group",
        "scientific_boundary": "validates a governed scientific boundary",
        "high_compute": "requires substantial CPU, GPU, or wall time",
        "model_artifact": "requires an installed Hotpot model artifact",
    }
    for name, description in markers.items():
        config.addinivalue_line("markers", f"{name}: {description}")


def pytest_collection_modifyitems(config, items):
    import pytest

    for item in items:
        if "native_cpp" in item.path.parts:
            item.add_marker(pytest.mark.high_compute)
        if "mca" in item.path.parts and {
            "predictor",
            "cpu_backend",
        }.intersection(item.fixturenames):
            item.add_marker(pytest.mark.model_artifact)

    switches = {
        "external": "--run-external",
        "optional_dependency": "--run-optional",
        "scientific_boundary": "--run-scientific-boundary",
        "high_compute": "--run-high-compute",
        "model_artifact": "--run-model-artifacts",
    }
    for item in items:
        for marker, option in switches.items():
            if item.get_closest_marker(marker) and not config.getoption(option):
                item.add_marker(pytest.mark.skip(reason=f"requires {option}"))
