"""Command-line adapter for controlled molecular-stage pipelines."""

from __future__ import annotations

import argparse
import json
import sys
from importlib import resources
from pathlib import Path
from typing import Optional, Sequence, Tuple

from hotpot._cli import MarkdownDocumentationAction

from .contracts import (
    PipelineDefinitionError,
    PipelineExecutionError,
    PipelineStatus,
    StageSpec,
)
from .registry import stage_import_path
from .runner import run_pipeline


__all__ = (
    "add_arguments",
    "build_parser",
    "load_cli_documentation",
    "load_json_workflow",
    "main",
    "parse_inline_stages",
    "run",
)


class _PipelineArgumentParser(argparse.ArgumentParser):
    """Convert direct-module syntax errors into the pipeline error contract."""

    def error(self, message: str) -> None:
        raise PipelineDefinitionError(message)


def load_cli_documentation() -> str:
    """Return the packaged long-form pipeline documentation."""
    return (
        resources.files(__package__)
        .joinpath("cli_doc.md")
        .read_text(encoding="utf-8")
    )


def _validated_stage(name: str, argv: Sequence[str]) -> StageSpec:
    stage_import_path(name)
    return StageSpec(name=name, argv=tuple(argv))


def parse_inline_stages(tokens: Sequence[str]) -> Tuple[StageSpec, ...]:
    """Parse exact ``::`` argv separators without invoking a shell."""
    if not tokens:
        raise PipelineDefinitionError("An inline pipeline must contain a stage")

    groups = []
    current = []
    for token in tokens:
        if token == "::":
            if not current:
                raise PipelineDefinitionError("Pipeline stages cannot be empty")
            groups.append(tuple(current))
            current = []
        else:
            current.append(token)
    if not current:
        raise PipelineDefinitionError("Pipeline stages cannot be empty")
    groups.append(tuple(current))
    return tuple(_validated_stage(group[0], group[1:]) for group in groups)


def _unique_json_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise PipelineDefinitionError(f"Duplicate JSON key {key!r}")
        payload[key] = value
    return payload


def load_json_workflow(path: Path) -> Tuple[StageSpec, ...]:
    """Load one strict JSON workflow into the canonical stage sequence."""
    try:
        payload = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise PipelineDefinitionError(
            f"Could not read workflow JSON {path}: {error}"
        ) from error
    if not isinstance(payload, dict) or set(payload) != {"stages"}:
        raise PipelineDefinitionError(
            "Workflow JSON must be an object containing only 'stages'"
        )
    stages = payload["stages"]
    if not isinstance(stages, list) or not stages:
        raise PipelineDefinitionError("Workflow 'stages' must be a nonempty list")

    specs = []
    for index, stage in enumerate(stages):
        if not isinstance(stage, dict) or set(stage) != {"name", "argv"}:
            raise PipelineDefinitionError(
                f"Workflow stage {index} must contain only 'name' and 'argv'"
            )
        name = stage["name"]
        argv = stage["argv"]
        if not isinstance(name, str) or not name:
            raise PipelineDefinitionError(
                f"Workflow stage {index} name must be a nonempty string"
            )
        if not isinstance(argv, list) or any(
            not isinstance(value, str) for value in argv
        ):
            raise PipelineDefinitionError(
                f"Workflow stage {index} argv must be a list of strings"
            )
        specs.append(_validated_stage(name, argv))
    return tuple(specs)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the ``hotpot run`` public command arguments."""
    parser.add_argument(
        "--doc",
        action=MarkdownDocumentationAction,
        nargs=0,
        document_loader=load_cli_documentation,
        help="show detailed Markdown usage documentation and exit",
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        metavar="DIRECTORY",
        help="create this exact directory for ordered pipeline artifacts",
    )
    parser.add_argument(
        "pipeline_tokens",
        nargs="*",
        metavar="WORKFLOW.json/-- STAGE ...",
        help="JSON workflow file or inline stages separated by exact '::' tokens",
    )


def _pipeline_specs(tokens: Sequence[str]) -> Tuple[StageSpec, ...]:
    if not tokens:
        raise PipelineDefinitionError("A JSON or inline workflow is required")
    first = Path(tokens[0])
    if first.suffix.lower() == ".json":
        if len(tokens) != 1:
            raise PipelineDefinitionError(
                "JSON and inline workflow sources cannot be combined"
            )
        return load_json_workflow(first)
    return parse_inline_stages(tokens)


def run(args: argparse.Namespace) -> int:
    """Execute one parsed controlled molecular pipeline."""
    try:
        specs = _pipeline_specs(args.pipeline_tokens)
        result = run_pipeline(
            specs,
            results_directory=Path(args.results_dir),
            initial_payload=None,
        )
    except (FileExistsError, PipelineDefinitionError) as error:
        print(f"hotpot run: {error}", file=sys.stderr)
        return 2
    except PipelineExecutionError as error:
        print(f"hotpot run: {error}", file=sys.stderr)
        return 1
    return 0 if result.status is PipelineStatus.SUCCEEDED else 1


def build_parser() -> argparse.ArgumentParser:
    parser = _PipelineArgumentParser(
        description="Run a controlled sequence of molecular Hotpot stages"
    )
    add_arguments(parser)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    raw_args = tuple(sys.argv[1:] if argv is None else argv)
    stage_tail: Tuple[str, ...] = ()
    if "--" in raw_args:
        separator_index = raw_args.index("--")
        stage_tail = raw_args[separator_index + 1 :]
        raw_args = raw_args[:separator_index]
    try:
        args = build_parser().parse_args(raw_args)
    except PipelineDefinitionError as error:
        print(f"hotpot run: {error}", file=sys.stderr)
        return 2
    args.pipeline_tokens.extend(stage_tail)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
