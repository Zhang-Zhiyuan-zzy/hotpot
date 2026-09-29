"""Stable configuration contracts for coordination-complex benchmarks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class RunProfile(str, Enum):
    """Named runtime profiles with explicit scientific settings."""

    STANDARD = "standard"
    SMOKE = "smoke"


@dataclass(frozen=True)
class BenchmarkSuite:
    """A molecular corpus and the chemistry operation applied to it."""

    name: str
    input_path: Path
    expected_count: int
    metal: str = "Eu"
    title: str = "Eu coordination complexes from extractant ligands"

    def to_manifest(self) -> dict[str, object]:
        return {
            "name": self.name,
            "input_path": str(self.input_path),
            "expected_count": self.expected_count,
            "metal": self.metal,
            "title": self.title,
        }


@dataclass(frozen=True)
class BenchmarkSettings:
    """CBond and force-field settings recorded with every result."""

    cbond_threshold: float = -0.125
    epochs: int = 100
    steps_per_epoch: int = 100
    max_attempts: int = 50
    candidate_warmup_steps: int = 500
    candidate_score_steps: int = 1000
    best_candidate_refine_steps: int = 3000
    ligand_untangling_attempts: int = 20
    coordination_restoration_attempts: int = 20
    coordination_relaxation_steps: int = 100
    complex_untangling_attempts: int = 30
    timeout: float = 1000.0
    seed: int = 20260921
    perturb_sigma: float = 0.5
    quality_level: str = "standard"
    trajectory_start: str = "ligand_build"

    def to_manifest(self) -> Mapping[str, object]:
        return asdict(self)


STANDARD_SETTINGS = BenchmarkSettings()

SMOKE_SETTINGS = BenchmarkSettings(
    epochs=2,
    steps_per_epoch=10,
    max_attempts=1,
    candidate_warmup_steps=10,
    candidate_score_steps=10,
    best_candidate_refine_steps=20,
    ligand_untangling_attempts=2,
    coordination_restoration_attempts=2,
    coordination_relaxation_steps=10,
    complex_untangling_attempts=2,
    timeout=120.0,
)


BUILTIN_SUITES = {
    "extractants-eu-187": BenchmarkSuite(
        name="extractants-eu-187",
        input_path=REPOSITORY_ROOT / "molecules/extractant/extractants.smi",
        expected_count=187,
    ),
}


def settings_for_profile(profile: RunProfile) -> BenchmarkSettings:
    """Return the immutable settings attached to a named profile."""
    if profile is RunProfile.STANDARD:
        return STANDARD_SETTINGS
    return SMOKE_SETTINGS
