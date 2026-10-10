"""Controlled-pipeline adapter for one CBond inference path."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple, Union

from hotpot.cheminfo.convert import to_hotpot_mol
from hotpot.pipeline.contracts import (
    JSONValue,
    MolecularPayload,
    MolecularRecord,
    MolecularStage,
    PipelineDefinitionError,
    PreparedMolecularStage,
    StageContext,
    StageExecutionError,
    StageResult,
    StageSpec,
    StageStatus,
)

from . import apply
from .constants import DEFAULT_MAX_STATES


__all__ = ("STAGE", "get_stage")


class _StageArgumentParser(argparse.ArgumentParser):
    """Report CBond syntax failures through the pipeline definition contract."""

    def error(self, message: str) -> None:
        raise PipelineDefinitionError(f"cbond stage: {message}")


@dataclass(frozen=True)
class _PreparedCBondStage(PreparedMolecularStage):
    metal: Union[int, str]
    ligand: str
    input_format: Optional[str]
    threshold: Optional[float]
    first_threshold: Optional[float]
    greedy: bool
    all_structures: bool
    max_states: int
    device: str
    model_dir: Optional[str]
    model_source: Optional[str]

    def execute(
        self,
        payload: MolecularPayload,
        context: StageContext,
    ) -> StageResult:
        if payload.records:
            raise StageExecutionError(
                "cbond stage with a ligand argument must be the first molecular stage"
            )
        try:
            ligand_mol = to_hotpot_mol(self.ligand, fmt=self.input_format)
            runtime = apply.get_cbond_runtime(
                self.device,
                self.model_dir,
                self.model_source,
            )
            if self.all_structures:
                structures = tuple(
                    apply.build_all_possible_cbond(
                        ligand_mol,
                        self.metal,
                        threshold=self.threshold,
                        first_threshold=self.first_threshold,
                        greedy=self.greedy,
                        runtime=runtime,
                        max_states=self.max_states,
                        return_details=True,
                    )
                )
                if not structures:
                    raise ValueError(
                        "No coordination structure exceeded the requested threshold"
                    )
                return _all_structure_result(structures)

            result = apply.auto_build_cbond(
                ligand_mol,
                self.metal,
                threshold=self.threshold,
                first_threshold=self.first_threshold,
                greedy=self.greedy,
                runtime=runtime,
                return_details=True,
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise StageExecutionError(f"cbond stage failed: {error}") from error

        return StageResult(
            status=StageStatus.SUCCEEDED,
            payload=MolecularPayload((MolecularRecord(result.molecule),)),
            report={
                "cbond": {
                    "metal": self.metal,
                    "donor_indices": tuple(int(value) for value in result.donor_indices),
                    "path_probability": float(result.path_probability),
                    "steps": _steps(result.steps),
                }
            },
        )


@dataclass(frozen=True)
class _CBondStage(MolecularStage):
    def prepare(self, spec: StageSpec) -> PreparedMolecularStage:
        if spec.name != "cbond":
            raise PipelineDefinitionError(
                f"CBond stage cannot prepare stage {spec.name!r}"
            )
        args = _stage_parser().parse_args(spec.argv)
        metal = int(args.metal) if args.metal.isdecimal() else args.metal
        return _PreparedCBondStage(
            metal=metal,
            ligand=args.ligand,
            input_format=args.input_format,
            threshold=args.threshold,
            first_threshold=args.first_threshold,
            greedy=args.greedy,
            all_structures=args.all_structures,
            max_states=args.max_states,
            device=args.device,
            model_dir=args.model_dir,
            model_source=args.model_source,
        )


def _stage_parser() -> _StageArgumentParser:
    parser = _StageArgumentParser(add_help=False)
    parser.add_argument("metal")
    parser.add_argument("ligand")
    parser.add_argument("--input-format")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--first-threshold", type=float)
    parser.add_argument("--no-greedy", dest="greedy", action="store_false")
    parser.set_defaults(greedy=True)
    parser.add_argument("--all-structures", action="store_true")
    parser.add_argument("--max-states", type=_positive_int, default=DEFAULT_MAX_STATES)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--model-dir")
    parser.add_argument("--model-source", choices=("auto", "local", "hub"))
    return parser


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _steps(
    steps: Tuple[apply.CBondStep, ...],
) -> Tuple[Mapping[str, JSONValue], ...]:
    return tuple(
        {
            "atom_index": int(step.atom_index),
            "element": str(step.element),
            "score": float(step.score),
            "probability": float(step.probability),
        }
        for step in steps
    )


def _all_structure_result(
    structures: Tuple[apply.CBondStructureResult, ...],
) -> StageResult:
    reports = tuple(
        {
            "rank": rank,
            "donor_indices": tuple(int(value) for value in result.donor_indices),
            "probability": float(result.probability),
            "path_count": int(result.path_count),
            "steps": _steps(result.steps),
        }
        for rank, result in enumerate(structures, start=1)
    )
    return StageResult(
        status=StageStatus.SUCCEEDED,
        payload=MolecularPayload(
            tuple(MolecularRecord(result.molecule) for result in structures)
        ),
        report={"cbond": {"structures": reports}},
    )


STAGE = _CBondStage()


def get_stage() -> MolecularStage:
    """Return the registered CBond molecular stage."""
    return STAGE
