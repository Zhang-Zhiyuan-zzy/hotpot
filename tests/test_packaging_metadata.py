from pathlib import Path
from runpy import run_path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


def _requirement_map(requirements):
    parsed = (Requirement(requirement) for requirement in requirements)
    return {
        canonicalize_name(requirement.name): str(requirement.specifier)
        for requirement in parsed
    }


def test_runtime_requirements_match_project_metadata():
    with (ROOT / "pyproject.toml").open("rb") as stream:
        project_dependencies = tomllib.load(stream)["project"]["dependencies"]
    requirements_dependencies = [
        line.strip()
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

    assert _requirement_map(requirements_dependencies) == _requirement_map(
        project_dependencies
    )


def test_package_version_matches_project_metadata():
    with (ROOT / "pyproject.toml").open("rb") as stream:
        project_version = tomllib.load(stream)["project"]["version"]

    package_version = run_path(ROOT / "hotpot" / "__version__.py")["__version__"]

    assert package_version == project_version


def test_setup_py_has_no_duplicate_project_metadata():
    setup_text = (ROOT / "setup.py").read_text(encoding="utf-8")

    assert "install_requires" not in setup_text
    assert "setuptools.setup(" not in setup_text


def test_conda_environment_uses_one_openbabel_distribution():
    environment_text = (ROOT / "environment.yml").read_text(encoding="utf-8")

    assert "openbabel=" not in environment_text
    assert '"-e .[dev]"' in environment_text


def test_expected_optional_dependency_groups_are_published():
    with (ROOT / "pyproject.toml").open("rb") as stream:
        extras = tomllib.load(stream)["project"]["optional-dependencies"]

    assert {
        "all",
        "complexformer",
        "datasets",
        "dev",
        "legacy-search",
        "onnx-export",
        "optimize",
    } <= extras.keys()
