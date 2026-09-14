"""Stable public result types."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class SitePrediction:
    atom_index: int
    element: str
    site_type: str
    mca_kj_mol: float

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MoleculePrediction:
    smiles: str
    formal_charge: int
    sites: tuple[SitePrediction, ...]
    model_variant: str

    def to_dict(self):
        value = asdict(self)
        value["sites"] = [site.to_dict() for site in self.sites]
        return value
