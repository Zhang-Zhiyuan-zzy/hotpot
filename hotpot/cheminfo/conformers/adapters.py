"""Pickle-safe adapters for current conformer-generation backends."""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Callable

import numpy as np

from .contracts import (
    ConformerBatch,
    ConformerSamplingError,
    ConformerSamplingRequest,
)

if TYPE_CHECKING:
    from ..core import Molecule


__all__ = ("OpenBabelConformerAdapter", "RDKitConformerAdapter")


def _materialized_hydrogen_projection(molecule: "Molecule") -> "Molecule":
    """Project a fully explicit-H working copy into RDKit's graph-H model."""
    from ..stereo import (
        HydrogenTransportState,
        StereoAnchorKind,
        StereoCarrierKind,
        get_stereo_payload,
        set_stereo_payload,
    )

    projected = copy(molecule)
    for atom in projected.atoms:
        atom.implicit_hydrogens = 0

    payload = get_stereo_payload(projected)
    records = []
    for record in payload.records:
        if not any(
            carrier.kind is StereoCarrierKind.IMPLICIT_H
            for carrier in record.carriers
        ):
            records.append(record)
            continue
        if (
            record.anchor.kind is not StereoAnchorKind.ATOM
            or len(record.anchor.atom_indices) != 1
        ):
            raise ConformerSamplingError(
                "A materialized implicit-H stereo carrier requires one atom anchor"
            )
        anchor = projected.atoms[record.anchor.atom_indices[0]]
        represented_rows = {
            carrier.atom
            for carrier in record.carriers
            if carrier.kind is StereoCarrierKind.ATOM
        }
        implicit_count = sum(
            carrier.kind is StereoCarrierKind.IMPLICIT_H
            for carrier in record.carriers
        )
        hydrogen_rows = tuple(
            sorted(
                neighbour.idx
                for neighbour in anchor.neighbours
                if neighbour.is_hydrogen and neighbour.idx not in represented_rows
            )
        )
        if len(hydrogen_rows) != implicit_count:
            raise ConformerSamplingError(
                "Materialized implicit-H stereo carriers require one unrepresented "
                "graph-hydrogen neighbour each"
            )
        replacements = iter(hydrogen_rows)
        records.append(
            replace(
                record,
                carriers=tuple(
                    replace(
                        carrier,
                        kind=StereoCarrierKind.ATOM,
                        atom=next(replacements),
                    )
                    if carrier.kind is StereoCarrierKind.IMPLICIT_H
                    else carrier
                    for carrier in record.carriers
                ),
            )
        )

    current_hydrogens = HydrogenTransportState(
        implicit_hydrogens=0,
        bracket_explicit_hydrogens=0,
        no_implicit=True,
    )
    set_stereo_payload(
        projected,
        replace(
            payload,
            records=tuple(records),
            atom_facts=tuple(
                replace(fact, current_hydrogens=current_hydrogens)
                for fact in payload.atom_facts
            ),
        ),
    )
    return projected


@dataclass(frozen=True)
class OpenBabelConformerAdapter:
    """Adapt the existing single OBBuilder call to the private protocol."""

    _builder: Callable[["Molecule"], None] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def sample(
        self,
        molecule: "Molecule",
        request: ConformerSamplingRequest,
    ) -> ConformerBatch:
        del request
        builder = self._builder
        if builder is None:
            from ..forcefields.backend import _ob_build

            builder = _ob_build
        builder(molecule)
        coordinates = np.asarray(molecule.coordinates, dtype=float).copy()
        return ConformerBatch(coordinates[None, :, :], (0,))


@dataclass(frozen=True)
class RDKitConformerAdapter:
    """Adapt pinned native ETKDG to one deterministic workflow conformer."""

    materialized_hydrogens: bool = False

    def sample(
        self,
        molecule: "Molecule",
        request: ConformerSamplingRequest,
    ) -> ConformerBatch:
        from .. import rdWrappers

        try:
            input_molecule = (
                _materialized_hydrogen_projection(molecule)
                if self.materialized_hydrogens
                else molecule
            )
            batch = rdWrappers.generate_conformers(
                input_molecule,
                options=rdWrappers.ETKDGOptions(
                    conformer_count=1,
                    seed=request.seed,
                ),
            )
        except Exception as exc:
            raise ConformerSamplingError(
                f"RDKit conformer generation failed: {exc}"
            ) from exc
        if batch.status not in (
            rdWrappers.GenerationStatus.SUCCESS,
            rdWrappers.GenerationStatus.PARTIAL,
        ):
            detail = f": {batch.diagnostic}" if batch.diagnostic else ""
            raise ConformerSamplingError(
                "RDKit conformer generation returned "
                f"{batch.status.name}{detail}"
            )

        coordinates = np.asarray(batch.coordinates, dtype=float)
        if (
            coordinates.ndim != 3
            or coordinates.shape[1:] != (len(molecule.atoms), 3)
            or coordinates.shape[0] != len(batch.records)
            or coordinates.shape[0] == 0
            or not np.all(np.isfinite(coordinates))
        ):
            raise ConformerSamplingError(
                "RDKit conformer generation returned no valid mapped conformer"
            )

        selected_index, selected_record = min(
            enumerate(batch.records),
            key=lambda item: (item[1].ordinal, item[0]),
        )
        selected = coordinates[selected_index : selected_index + 1].copy()
        return ConformerBatch(selected, (selected_record.ordinal,))
