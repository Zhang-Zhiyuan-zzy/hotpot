from importlib.util import find_spec
from pathlib import Path
from runpy import run_path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(find_spec("hotpot").origin).resolve().parent


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


def test_networkx_is_test_only():
    with (ROOT / "pyproject.toml").open("rb") as stream:
        project_dependencies = tomllib.load(stream)["project"]["dependencies"]
    runtime_requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    inference_requirements = (
        ROOT / "tests" / "requirements-inference.txt"
    ).read_text(encoding="utf-8")

    assert "networkx" not in _requirement_map(project_dependencies)
    assert "networkx" not in runtime_requirements.lower()
    assert "networkx" in inference_requirements.lower()


def test_inference_requirements_cover_runtime_dependencies():
    with (ROOT / "pyproject.toml").open("rb") as stream:
        project_dependencies = tomllib.load(stream)["project"]["dependencies"]
    inference_dependencies = [
        line.strip()
        for line in (ROOT / "tests" / "requirements-inference.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.startswith("#")
    ]

    assert _requirement_map(project_dependencies).items() <= _requirement_map(
        inference_dependencies
    ).items()


def test_package_version_matches_project_metadata():
    with (ROOT / "pyproject.toml").open("rb") as stream:
        project_version = tomllib.load(stream)["project"]["version"]

    package_version = run_path(PACKAGE_ROOT / "__version__.py")["__version__"]

    assert package_version == project_version


def test_setup_py_has_no_duplicate_project_metadata():
    setup_text = (ROOT / "setup.py").read_text(encoding="utf-8")

    assert "install_requires" not in setup_text
    assert "setuptools.setup(" not in setup_text


def test_conda_environment_uses_one_openbabel_distribution():
    environment_text = (ROOT / "environment.yml").read_text(encoding="utf-8")

    assert "openbabel=" not in environment_text
    assert "python=3.11" in environment_text
    assert '"-e .[dev]"' in environment_text


def test_openbabel_dependency_tracks_supported_python_versions():
    with (ROOT / "pyproject.toml").open("rb") as stream:
        dependencies = tomllib.load(stream)["project"]["dependencies"]

    requirements = {
        canonicalize_name(requirement.name): requirement
        for requirement in map(Requirement, dependencies)
    }

    current = requirements["openbabel"]
    assert str(current.specifier) == "<3.3,>=3.2.1"
    assert current.marker is None
    assert "openbabel-wheel" not in requirements


def test_python_and_rdkit_release_contract_is_exact():
    with (ROOT / "pyproject.toml").open("rb") as stream:
        metadata = tomllib.load(stream)

    project = metadata["project"]
    assert project["requires-python"] == ">=3.10,<3.15"
    assert {
        classifier.rsplit("::", 1)[-1].strip()
        for classifier in project["classifiers"]
        if classifier.startswith("Programming Language :: Python :: 3.")
    } == {"3.10", "3.11", "3.12", "3.13", "3.14"}

    dependencies = {
        canonicalize_name(requirement.name): requirement
        for requirement in map(Requirement, project["dependencies"])
    }
    rdkit = dependencies["rdkit"]
    assert str(rdkit.specifier) == "==2026.9.1"
    assert rdkit.marker is None

    inference_requirements = {
        canonicalize_name(requirement.name): requirement
        for requirement in map(
            Requirement,
            (
                line.strip()
                for line in (ROOT / "tests" / "requirements-inference.txt")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip() and not line.startswith("#")
            ),
        )
    }
    inference_rdkit = inference_requirements["rdkit"]
    assert str(inference_rdkit.specifier) == "==2026.9.1"
    assert inference_rdkit.marker is None


def test_expected_optional_dependency_groups_are_published():
    with (ROOT / "pyproject.toml").open("rb") as stream:
        extras = tomllib.load(stream)["project"]["optional-dependencies"]

    assert {
        "all",
        "complexformer",
        "datasets",
        "dev",
        "onnx-export",
        "optimize",
        "pymol",
    } <= extras.keys()

    assert "legacy-search" not in extras
    assert all(
        canonicalize_name(Requirement(requirement).name) != "sympy"
        for requirement in extras["all"]
    )


def test_pymol_compatibility_extra_selects_numpy_1x():
    with (ROOT / "pyproject.toml").open("rb") as stream:
        extras = tomllib.load(stream)["project"]["optional-dependencies"]

    requirements = tuple(map(Requirement, extras["pymol"]))

    assert len(requirements) == 1
    assert canonicalize_name(requirements[0].name) == "numpy"
    assert str(requirements[0].specifier) == "<2,>=1.26.4"
    assert requirements[0].marker is None


def test_production_model_weights_are_external():
    model_directories = (
        PACKAGE_ROOT / "cheminfo" / "operations" / "inference" / "artifacts" / "data" / "mca",
        PACKAGE_ROOT / "cheminfo" / "operations" / "inference" / "artifacts" / "data" / "cbond",
    )
    binary_suffixes = {".onnx", ".weight", ".data"}

    assert not [
        path
        for directory in model_directories
        for path in directory.iterdir()
        if path.suffix in binary_suffixes
    ]


def test_gaussian_runtime_configuration_is_packaged():
    with (ROOT / "pyproject.toml").open("rb") as stream:
        package_data = tomllib.load(stream)["tool"]["setuptools"]["package-data"]
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    assert package_data["hotpot.plugins.qm.gaussian"] == ["*.json"]
    assert "recursive-include hotpot/plugins/qm/gaussian *.json" in manifest


def test_draw_fonts_are_owned_by_the_discovered_font_package():
    with (ROOT / "pyproject.toml").open("rb") as stream:
        package_data = tomllib.load(stream)["tool"]["setuptools"]["package-data"]

    assert package_data["hotpot.cheminfo.adapters.visualization"] == [
        "fonts/Arial/*.ttf", "draw.md"
    ]


def test_artifact_pointers_are_owned_by_the_inference_operation_package():
    with (ROOT / "pyproject.toml").open("rb") as stream:
        package_data = tomllib.load(stream)["tool"]["setuptools"]["package-data"]
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    assert package_data["hotpot.cheminfo.operations.inference.artifacts"] == [
        "data/mca/*.json", "data/mca/*.md", "data/cbond/*.json", "data/cbond/*.md",
    ]
    for name in ("mca", "cbond"):
        for filename in ("manifest.json", "README.md"):
            path = f"hotpot/cheminfo/operations/inference/artifacts/data/{name}/{filename}"
            assert (ROOT / path).is_file()
            assert f"include {path}" in manifest


def test_inference_commands_and_companion_docs_follow_their_owners():
    with (ROOT / "pyproject.toml").open("rb") as stream:
        package_data = tomllib.load(stream)["tool"]["setuptools"]["package-data"]
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    assert package_data["hotpot.commands"] == ["ff.md", "mca.md", "cbond.md"]
    for name in ("mca", "cbond"):
        assert (PACKAGE_ROOT / "commands" / f"{name}.md").is_file()
        assert f"include hotpot/commands/{name}.md" in manifest
        assert "*.md" in package_data[f"hotpot.cheminfo.operations.inference.{name}"]
    assert "recursive-include tools/model_deployment/cbond *.py *.md" in manifest
    assert not (PACKAGE_ROOT / "cheminfo" / "AImodels").is_dir()
    assert not (PACKAGE_ROOT / "tools" / "model_deployment").exists()
