"""Process-wide test initialization for native chemistry extensions."""

# PyPI Open Babel 3.2.1 can crash if pybel loads before RDKit in one process.
# Hotpot enforces the same deterministic order in its package entry point.
from rdkit import Chem as _rdkit_chem  # noqa: F401


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run slow end-to-end chemistry tests",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: marks tests that exercise long-running chemistry workflows",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return

    import pytest

    skip_slow = pytest.mark.skip(reason="requires --run-slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
