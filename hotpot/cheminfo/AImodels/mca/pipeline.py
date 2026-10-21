"""Shared molecule stages and result assembly for MCA backends."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from hotpot.cheminfo.convert import is_molecule_input

from .conformer import ensure_3d_conformer
from .featurizer import mol_to_unimolv2
from .result_types import AtomPrediction, MoleculePrediction, SitePrediction
from .serialization import (
    MoleculeDTO,
    hotpot_mol_from_dto,
    normalize_molecule,
    rdmol_from_dto,
    validate_dto,
)
from .site_detection import find_nucleophilic_sites


SiteRecord = tuple[int, str]
FeatureDict = dict[str, np.ndarray]
DEFAULT_CONFORMER_SEED = 42


@dataclass(frozen=True, slots=True)
class PreparedMolecule:
    molecule: MoleculeDTO
    sites: tuple[SiteRecord, ...]


def is_single_molecule_input(value) -> bool:
    return is_molecule_input(value) or not isinstance(value, Iterable)


def detect_sites(molecule: MoleculeDTO) -> tuple[SiteRecord, ...]:
    sites = find_nucleophilic_sites(hotpot_mol_from_dto(molecule))
    return tuple((site.atom_index, site.site_type) for site in sites)


def prepare_molecule(value, position: int, allow_charged: bool) -> PreparedMolecule:
    molecule = normalize_molecule(value)
    validate_dto(molecule, position, allow_charged)
    return PreparedMolecule(molecule=molecule, sites=detect_sites(molecule))


def build_features(
    molecule: MoleculeDTO,
    max_atoms: int,
    conformer_seed: int,
) -> FeatureDict:
    conformer = ensure_3d_conformer(
        rdmol_from_dto(molecule),
        conformer_seed,
    )
    return mol_to_unimolv2(conformer, max_atoms)


def build_prediction(
    molecule: MoleculeDTO,
    sites: Sequence[SiteRecord],
    values: Sequence[float],
    model_variant: str,
) -> MoleculePrediction:
    atom_predictions = tuple(
        AtomPrediction(
            atom_index=atom_index,
            element=element,
            mca_kj_mol=float(values[atom_index]),
        )
        for atom_index, element in enumerate(molecule.atom_symbols)
    )
    site_predictions = tuple(
        SitePrediction(
            atom_index=atom_index,
            element=molecule.atom_symbols[atom_index],
            site_type=site_type,
            mca_kj_mol=float(values[atom_index]),
        )
        for atom_index, site_type in sites
    )
    return MoleculePrediction(
        smiles=molecule.smiles,
        formal_charge=molecule.formal_charge,
        sites=site_predictions,
        model_variant=model_variant,
        atom_predictions=atom_predictions,
    )


__all__ = [
    "FeatureDict",
    "PreparedMolecule",
    "SiteRecord",
    "DEFAULT_CONFORMER_SEED",
    "build_features",
    "build_prediction",
    "detect_sites",
    "is_single_molecule_input",
    "prepare_molecule",
]
