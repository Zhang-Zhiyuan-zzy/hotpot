"""Opt-in execution of examples from optional legacy research guides."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
GUIDE = ROOT / "doc" / "EvolutionAlgorithm.md"
REFERENCE = (
    "tests/documentation/test_legacy_examples.py::"
    "test_evolution_algorithm_tutorial"
)


@pytest.mark.optional_dependency
@pytest.mark.high_compute
def test_evolution_algorithm_tutorial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pandas = pytest.importorskip("pandas")
    datasets = pytest.importorskip("sklearn.datasets")
    pytest.importorskip("xgboost")
    text = GUIDE.read_text(encoding="utf-8")
    section = text[text.index(REFERENCE) + len(REFERENCE) :]
    match = re.search(r"```python\n(.*?)\n```", section, re.DOTALL)
    assert match is not None
    features, target = datasets.make_regression(
        n_samples=60,
        n_features=4,
        random_state=0,
    )
    pandas.DataFrame(
        np.column_stack((features, target)),
        columns=("x1", "x2", "x3", "x4", "target"),
    ).to_csv(tmp_path / "training.csv", index=False)
    monkeypatch.chdir(tmp_path)
    namespace: dict[str, object] = {"__name__": "__doc_example__"}

    exec(compile(match.group(1), str(GUIDE), "exec"), namespace)

    earth = namespace["earth"]
    assert earth.individuals
    assert all(np.isfinite(individual.fitness_) for individual in earth.individuals)
