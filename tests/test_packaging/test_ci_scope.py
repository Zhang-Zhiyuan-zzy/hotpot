"""CI scheduling contracts exercised against real Git history and event files."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / ".github" / "scripts" / "ci_scope.py"


@pytest.fixture
def repository(tmp_path):
    environment = dict(os.environ, GIT_AUTHOR_NAME="CI Test", GIT_AUTHOR_EMAIL="ci@example.invalid",
                       GIT_COMMITTER_NAME="CI Test", GIT_COMMITTER_EMAIL="ci@example.invalid")

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, env=environment, text=True).strip()

    def commit(path, content):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        git("add", "--all")
        git("commit", "-qm", "fixture")
        return git("rev-parse", "HEAD")

    git("init", "-q")
    git("symbolic-ref", "HEAD", "refs/heads/main")
    base = commit("hotpot/model.py", "value = 1\n")
    return tmp_path, git, commit, base


def select(repository, event_name, event, full=False):
    directory = repository[0]
    payload = directory / "event.json"
    payload.write_text(json.dumps(event))
    output = directory / "output.txt"
    output.write_text("")
    environment = dict(os.environ, GITHUB_EVENT_NAME=event_name, GITHUB_EVENT_PATH=str(payload),
                       GITHUB_OUTPUT=str(output), HOTPOT_CI_FULL_MATRIX=str(full).lower())
    result = subprocess.run([sys.executable, str(SCRIPT)], cwd=directory, env=environment,
                            capture_output=True, text=True)
    values = dict(line.split("=", 1) for line in output.read_text().splitlines())
    return result, values


@pytest.mark.parametrize("path", ["README.md", "doc/usage.md", "assets/readme/example.png", "plan/example/goal.md"])
def test_documentation_push_is_lightweight(repository, path):
    directory, git, commit, base = repository
    head = commit(path, "Documentation\n")
    result, values = select(repository, "push", {"before": base, "after": head})
    assert result.returncode == 0, result.stderr
    assert values == {"core": "false", "python_tags": '["cp311"]'}


@pytest.mark.parametrize("path", ["hotpot/model.py", "hotpot/commands/ff.md", "tests/fixture.md",
                                  "plan/example/check.py", "environment.yml", ".github/workflows/ci.yml"])
def test_business_and_configuration_changes_keep_core_checks(repository, path):
    directory, git, commit, base = repository
    head = commit(path, "changed\n")
    result, values = select(repository, "push", {"before": base, "after": head})
    assert result.returncode == 0, result.stderr
    assert values["core"] == "true"
    assert json.loads(values["python_tags"]) == ["cp311"]


def test_renaming_code_to_documentation_cannot_hide_its_deletion(repository):
    directory, git, commit, base = repository
    git("mv", "hotpot/model.py", "README.md")
    git("commit", "-qm", "rename")
    result, values = select(repository, "push", {"before": base, "after": git("rev-parse", "HEAD")})
    assert result.returncode == 0, result.stderr
    assert values["core"] == "true"


@pytest.mark.parametrize("before", ["0" * 40, "f" * 40])
def test_new_or_rewritten_history_runs_core_checks(repository, before):
    result, values = select(repository, "push", {"before": before, "after": repository[3]})
    assert result.returncode == 0, result.stderr
    assert values["core"] == "true"


@pytest.mark.parametrize("event_name,full,expected", [
    ("workflow_dispatch", False, ["cp311"]),
    ("workflow_dispatch", True, ["cp310", "cp311", "cp312", "cp313", "cp314"]),
    ("schedule", True, ["cp310", "cp311", "cp312", "cp313", "cp314"]),
])
def test_requested_checks_always_execute_even_without_a_diff(repository, event_name, full, expected):
    result, values = select(repository, event_name, {}, full=full)
    assert result.returncode == 0, result.stderr
    assert values["core"] == "true"
    assert json.loads(values["python_tags"]) == expected


def test_release_tag_runs_full_matrix(repository):
    result, values = select(repository, "push", {"before": "0" * 40, "after": repository[3]}, full=True)
    assert result.returncode == 0, result.stderr
    assert values["core"] == "true"
    assert json.loads(values["python_tags"]) == ["cp310", "cp311", "cp312", "cp313", "cp314"]


def test_pull_request_compares_against_merge_base(repository):
    directory, git, commit, base = repository
    git("checkout", "-qb", "topic")
    head = commit("README.md", "Feature documentation\n")
    git("checkout", "-qb", "base", base)
    new_base = commit("hotpot/model.py", "value = 2\n")
    result, values = select(repository, "pull_request", {"pull_request": {
        "base": {"sha": new_base}, "head": {"sha": head},
    }})
    assert result.returncode == 0, result.stderr
    assert values["core"] == "false"


def test_lightweight_path_still_rejects_bad_patch_whitespace(repository):
    directory, git, commit, base = repository
    head = commit("README.md", "Trailing whitespace   \n")
    result, values = select(repository, "push", {"before": base, "after": head})
    assert result.returncode != 0
    assert values == {}
