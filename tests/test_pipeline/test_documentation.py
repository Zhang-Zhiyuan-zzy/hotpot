"""Execute the controlled-pipeline example from its Markdown source."""

from __future__ import annotations

import re
from pathlib import Path

import pytest


from hotpot import pipeline


GUIDE = Path(pipeline.__file__).resolve().with_name("README.md")
REFERENCE = (
    "tests/test_pipeline/test_documentation.py::"
    "test_pipeline_readme_python_example"
)


def test_pipeline_readme_python_example(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    text = GUIDE.read_text(encoding="utf-8")
    section = text[text.index(REFERENCE) + len(REFERENCE) :]
    match = re.search(r"```python\n(.*?)\n```", section, re.DOTALL)
    assert match is not None
    monkeypatch.chdir(tmp_path)

    exec(
        compile(match.group(1), str(GUIDE), "exec"),
        {"__name__": "__main__"},
    )

    assert capsys.readouterr().out == "succeeded\n"
    assert (tmp_path / "results/run-001/final.sdf").is_file()
