"""Execute the owner-approved Plan-17 force-field validation routes."""

from __future__ import annotations

from copy import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Literal, Mapping, Optional, TYPE_CHECKING

import numpy as np

from hotpot.cheminfo.conformers import (
    ConformerSampler,
    OpenBabelConformerAdapter,
    RDKitConformerAdapter,
)
from hotpot.cheminfo.core import BondKind
from hotpot.cheminfo.forcefields.coordination import _iter_metal_donor_pairs
from hotpot.cheminfo.forcefields.working_copy import (
    _copy_molecule_metadata,
    _hydrogenated_working_copy,
)

from .configuration import BenchmarkSettings
from .pipeline import _forcefield_options
from .validation_providers import (
    component_sha256,
    coordinate_sha256,
    freeze_rdkit_proposals,
    frozen_proposal_manifest,
    write_mapped_coordinate_proposal,
    write_frozen_proposals,
)


if TYPE_CHECKING:
    from hotpot.cheminfo.core import Molecule


ProviderRoute = Literal["O0", "F1", "F2"]
WholeComplexRoute = Literal["C1", "C2"]
ValidationRoute = Literal["O0", "F1", "F2", "C1", "C2"]

PROVIDER_STAGE_SEQUENCE = (
    "provider",
    "ligand_build",
    "placement",
    "coordination_restoration",
    "optimization",
    "acceptance",
)
WHOLE_COMPLEX_STAGE_SEQUENCE = (
    "whole_complex_provider",
    "complex_entry",
    "topology_checkpoint",
    "optimization",
    "acceptance",
)


@dataclass(frozen=True)
class ProposalRecord:
    """Coordinate proposal identity before downstream force-field work."""

    status: str
    atom_count: int
    coordinate_sha256: str
    generation_graph_sha256: str
    source_graph_sha256: str
    mapping_preserved: bool
    diagnostic: Optional[str] = None
    constraint_policy: Optional[Mapping[str, object]] = None


@dataclass(frozen=True)
class RouteExecution:
    """Successful in-memory route execution consumed by the evidence writer."""

    route: ValidationRoute
    molecule: "Molecule"
    report: object
    proposal: Optional[ProposalRecord]
    proposal_manifest: tuple[Mapping[str, object], ...]
    stage_sequence: tuple[str, ...]
    proposal_seconds: float
    forcefield_seconds: float
    elapsed_seconds: float
    proposal_artifact: Optional[Mapping[str, object]] = None


def _provider_sampler(
    route: ProviderRoute,
    molecule: "Molecule",
    *,
    seed: int,
    add_hydrogens: bool,
    freeze_sampler: Optional[ConformerSampler],
) -> tuple[ConformerSampler, tuple[Mapping[str, object], ...]]:
    if route == "O0":
        return OpenBabelConformerAdapter(), ()
    if route == "F1":
        return RDKitConformerAdapter(), ()
    replay = freeze_rdkit_proposals(
        molecule,
        seed=seed,
        add_hydrogens=add_hydrogens,
        sampler=freeze_sampler,
    )
    return replay, frozen_proposal_manifest(replay)


def execute_provider_route(
    molecule: "Molecule",
    route: ProviderRoute,
    settings: BenchmarkSettings,
    case_index: int,
    *,
    workflow: Optional[Callable[..., object]] = None,
    freeze_sampler: Optional[ConformerSampler] = None,
    proposal_artifact_path: Optional[Path] = None,
) -> RouteExecution:
    """Run O0/F1/F2 through the same private complete-complex workflow."""
    from hotpot.cheminfo import forcefields as ff
    from hotpot.cheminfo.forcefields.workflows import _complexes_build_workflow

    selected_workflow = workflow or _complexes_build_workflow
    options = _forcefield_options(ff, settings, case_index)
    options["add_hydrogens"] = True
    seed = settings.seed + case_index
    started = perf_counter()
    proposal_started = started
    sampler, proposal_manifest = _provider_sampler(
        route,
        molecule,
        seed=seed,
        add_hydrogens=True,
        freeze_sampler=freeze_sampler,
    )
    proposal_artifact = None
    if route == "F2" and proposal_artifact_path is not None:
        proposal_artifact = write_frozen_proposals(
            proposal_artifact_path,
            sampler,
        )
    proposal_seconds = perf_counter() - proposal_started
    forcefield_started = perf_counter()
    report = selected_workflow(
        molecule,
        "UFF",
        **options,
        sampler=sampler,
    )
    completed = perf_counter()
    return RouteExecution(
        route=route,
        molecule=molecule,
        report=report,
        proposal=None,
        proposal_manifest=proposal_manifest,
        stage_sequence=PROVIDER_STAGE_SEQUENCE,
        proposal_seconds=proposal_seconds,
        forcefield_seconds=completed - forcefield_started,
        elapsed_seconds=completed - started,
        proposal_artifact=proposal_artifact,
    )


def _dative_generation_graph(molecule: "Molecule") -> "Molecule":
    generation_graph = copy(molecule)
    _copy_molecule_metadata(molecule, generation_graph)
    for bond in generation_graph.bonds:
        if not bond.is_metal_ligand_bond:
            continue
        direction = "atom2_to_atom1" if bond.atom1.is_metal else "atom1_to_atom2"
        bond._set_bond_metadata(
            bond_kind=BondKind.DATIVE,
            bond_direction=direction,
            bond_source=bond.bond_source,
            bond_source_metadata=bond.bond_source_metadata,
        )
    return generation_graph


def _native_covalent_radius(atomic_number: int) -> float:
    from hotpot.cheminfo.forcefields.native import _native_module

    return float(_native_module().covalent_radius(atomic_number).angstrom)


def _coordination_constraints(
    generation_graph: "Molecule",
    *,
    radius: Callable[[int], float] = _native_covalent_radius,
):
    from hotpot.cheminfo import rdWrappers

    pairs = []
    bounds = []
    for metal, donor in _iter_metal_donor_pairs(generation_graph):
        pairs.append((int(donor.idx), int(metal.idx)))
        target = radius(int(metal.atomic_number)) + radius(int(donor.atomic_number))
        bounds.append((0.70 * target, 1.50 * target))
    return rdWrappers.CoordinationEmbeddingConstraints(
        coordination_pairs=np.asarray(pairs, dtype=np.int32).reshape((-1, 2)),
        metal_donor_bounds=np.asarray(bounds, dtype=np.float64).reshape((-1, 2)),
    )


def _generate_whole_complex(
    molecule: "Molecule",
    route: WholeComplexRoute,
    *,
    seed: int,
    generate: Optional[Callable[..., object]],
    radius: Callable[[int], float],
) -> ProposalRecord:
    from hotpot.cheminfo import rdWrappers

    generation_graph = _dative_generation_graph(molecule)
    source_graph_sha256 = component_sha256(molecule)
    generation_graph_sha256 = component_sha256(generation_graph)
    constraints = (
        None
        if route == "C1"
        else _coordination_constraints(generation_graph, radius=radius)
    )
    selected_generate = generate or rdWrappers.generate_conformers
    batch = selected_generate(
        generation_graph,
        options=rdWrappers.ETKDGOptions(conformer_count=1, seed=seed),
        constraints=constraints,
    )
    if batch.status not in (
        rdWrappers.GenerationStatus.SUCCESS,
        rdWrappers.GenerationStatus.PARTIAL,
    ):
        detail = f": {batch.diagnostic}" if batch.diagnostic else ""
        raise RuntimeError(
            f"{route} whole-complex ETKDG returned {batch.status.name}{detail}"
        )
    rows = np.asarray(batch.coordinates, dtype=np.float64)
    if rows.shape != (len(batch.records), len(molecule.atoms), 3) or not len(rows):
        raise RuntimeError(f"{route} returned invalid mapped coordinate rows")
    selected_index, selected_record = min(
        enumerate(batch.records),
        key=lambda item: (item[1].ordinal, item[0]),
    )
    coordinates = np.ascontiguousarray(rows[selected_index], dtype=np.float64)
    if not np.all(np.isfinite(coordinates)):
        raise RuntimeError(f"{route} returned non-finite proposal coordinates")
    molecule.coordinates = coordinates
    constraint_policy = None
    if constraints is not None:
        constraint_policy = {
            "name": "hotpot-native-placement-v1",
            "distance_unit": "angstrom",
            "coordination_distance_scale": 1.0,
            "coordination_distance_ratio_minimum": 0.70,
            "coordination_distance_ratio_maximum": 1.50,
            "coordination_pairs": constraints.coordination_pairs.tolist(),
            "metal_donor_bounds_angstrom": (constraints.metal_donor_bounds.tolist()),
        }
    return ProposalRecord(
        status=batch.status.name,
        atom_count=len(molecule.atoms),
        coordinate_sha256=coordinate_sha256(coordinates),
        generation_graph_sha256=generation_graph_sha256,
        source_graph_sha256=source_graph_sha256,
        mapping_preserved=coordinates.shape == (len(molecule.atoms), 3),
        diagnostic=batch.diagnostic,
        constraint_policy=constraint_policy,
    )


def execute_whole_complex_route(
    molecule: "Molecule",
    route: WholeComplexRoute,
    settings: BenchmarkSettings,
    case_index: int,
    *,
    generate: Optional[Callable[..., object]] = None,
    optimize: Optional[Callable[..., object]] = None,
    radius: Callable[[int], float] = _native_covalent_radius,
    add_hydrogens: bool = True,
    proposal_artifact_path: Optional[Path] = None,
) -> RouteExecution:
    """Generate C1/C2 coordinates, then enter the established Stage-3 flow."""
    from hotpot.cheminfo import forcefields as ff

    seed = settings.seed + case_index
    started = perf_counter()
    working = _hydrogenated_working_copy(
        molecule,
        add_hydrogens=add_hydrogens,
        seed=seed,
    )
    proposal = _generate_whole_complex(
        working,
        route,
        seed=seed,
        generate=generate,
        radius=radius,
    )
    proposal_artifact = None
    if proposal_artifact_path is not None:
        proposal_artifact = write_mapped_coordinate_proposal(
            proposal_artifact_path,
            route=route,
            proposal=asdict(proposal),
            coordinates=working.coordinates,
        )
    proposal_seconds = perf_counter() - started
    selected_optimize = optimize or ff.optimize_complex
    proposal_rows = np.ascontiguousarray(working.coordinates, dtype=np.float64)
    forcefield_started = perf_counter()
    report = selected_optimize(
        working,
        "UFF",
        epochs=settings.epochs,
        steps_per_epoch=settings.steps_per_epoch,
        complex_untangling_attempts=settings.complex_untangling_attempts,
        add_hydrogens=False,
        quality_level=settings.quality_level,
        seed=seed,
        perturb_sigma=settings.perturb_sigma,
        convergence_level=ff.ConvergenceLevel.FAST,
        save_movie=True,
        trajectory_start=ff.TrajectoryStart.COMPLEX_UNTANGLING,
        trajectory_path=None,
    )
    completed = perf_counter()
    if coordinate_sha256(proposal_rows) != proposal.coordinate_sha256:
        raise RuntimeError(f"{route} proposal changed before force-field entry")
    return RouteExecution(
        route=route,
        molecule=working,
        report=report,
        proposal=proposal,
        proposal_manifest=(),
        stage_sequence=WHOLE_COMPLEX_STAGE_SEQUENCE,
        proposal_seconds=proposal_seconds,
        forcefield_seconds=completed - forcefield_started,
        elapsed_seconds=completed - started,
        proposal_artifact=proposal_artifact,
    )


__all__ = (
    "PROVIDER_STAGE_SEQUENCE",
    "WHOLE_COMPLEX_STAGE_SEQUENCE",
    "ProposalRecord",
    "RouteExecution",
    "execute_provider_route",
    "execute_whole_complex_route",
)
