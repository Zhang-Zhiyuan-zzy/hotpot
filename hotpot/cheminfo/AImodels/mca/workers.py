"""Spawn-safe pure-CPU worker entry points for the MCA GPU pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from .pipeline import FeatureDict, build_features, detect_sites
from .serialization import MoleculeDTO


@dataclass(frozen=True, slots=True)
class SiteDetectionTask:
    molecule_index: int
    molecule: MoleculeDTO


@dataclass(frozen=True, slots=True)
class SiteDetectionResult:
    molecule_index: int
    sites: tuple[tuple[int, str], ...]


@dataclass(frozen=True, slots=True)
class FeatureTask:
    molecule_index: int
    molecule: MoleculeDTO
    max_atoms: int
    conformer_seed: int


@dataclass(frozen=True, slots=True)
class FeatureResult:
    molecule_index: int
    feature: FeatureDict


@dataclass(frozen=True, slots=True)
class CPUPreprocessTask:
    molecule_index: int
    molecule: MoleculeDTO
    sites: tuple[tuple[int, str], ...] | None
    max_atoms: int
    conformer_seed: int


@dataclass(frozen=True, slots=True)
class CPUPreprocessResult:
    molecule_index: int
    molecule: MoleculeDTO
    sites: tuple[tuple[int, str], ...]
    feature: FeatureDict


def detect_sites_worker(task: SiteDetectionTask) -> SiteDetectionResult:
    """Run ordered MCA site rules for one normalized molecule."""

    return SiteDetectionResult(
        molecule_index=task.molecule_index,
        sites=detect_sites(task.molecule),
    )


def build_features_worker(task: FeatureTask) -> FeatureResult:
    """Generate a deterministic conformer and Uni-Mol v2 features."""

    return FeatureResult(
        molecule_index=task.molecule_index,
        feature=build_features(
            task.molecule,
            task.max_atoms,
            task.conformer_seed,
        ),
    )


def cpu_preprocess_worker(task: CPUPreprocessTask) -> CPUPreprocessResult:
    """Run one molecule's ordered site, conformer, and feature stages."""

    sites = detect_sites(task.molecule) if task.sites is None else task.sites
    return CPUPreprocessResult(
        molecule_index=task.molecule_index,
        molecule=task.molecule,
        sites=sites,
        feature=build_features(
            task.molecule,
            task.max_atoms,
            task.conformer_seed,
        ),
    )


__all__ = [
    "CPUPreprocessResult",
    "CPUPreprocessTask",
    "FeatureDict",
    "FeatureResult",
    "FeatureTask",
    "SiteDetectionResult",
    "SiteDetectionTask",
    "build_features_worker",
    "cpu_preprocess_worker",
    "detect_sites_worker",
]
