from pathlib import Path


ROOT = Path(__file__).parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"


def test_inference_workflow_builds_the_exact_wheels_it_tests():
    workflow = (WORKFLOWS / "inference_compatibility.yml").read_text(
        encoding="utf-8"
    )

    assert "pypa/cibuildwheel@v4.3.0" in workflow
    assert "HOTPOT_RDWRAPPERS_BUILD_PROFILE=manylinux_2_28_x86_64" in workflow
    assert "python -m build --wheel" not in workflow
    for python_tag in ("cp310", "cp311", "cp312", "cp313", "cp314"):
        assert python_tag in workflow
    assert "name: wheel-${{ matrix.python-tag }}" in workflow
    assert "HOTPOT_TEST_WHEEL" in workflow
    assert "HOTPOT_TEST_WORKDIR" in workflow


def test_coverage_has_one_maintained_workflow_implementation():
    workflow = (WORKFLOWS / "inference_compatibility.yml").read_text(
        encoding="utf-8"
    )

    assert not (WORKFLOWS / "test_push.yml").exists()
    assert "coverage:" in workflow
    assert "./tests/run_coverage.sh" in workflow
    assert "codecov/codecov-action@v5" in workflow


def test_tag_builds_do_not_publish_without_an_explicit_release_request():
    workflow = (WORKFLOWS / "publish_pypi.yml").read_text(encoding="utf-8")

    assert 'tags:\n      - "v*"' in workflow
    assert "github.event_name == 'workflow_dispatch' && inputs.publish" in workflow
    assert "pypa/gh-action-pypi-publish@v1.14.2" in workflow
    assert "Match a release tag to the project version" in workflow
