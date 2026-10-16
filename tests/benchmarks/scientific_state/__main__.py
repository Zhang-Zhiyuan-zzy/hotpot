"""Command-line entry point for the scientific-state dry benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from .corpus import (
    load_corpus,
    load_structure_registry,
    load_topology_manifest,
)
from .runner import run_benchmark


def build_parser() -> argparse.ArgumentParser:
    """Build the opt-in pre-activation benchmark parser."""

    parser = argparse.ArgumentParser(
        description="Run the scientific-state pre-activation resolver baseline."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--smiles-corpus", type=Path)
    source.add_argument("--structure-registry", type=Path)
    source.add_argument("--topology-manifest", type=Path)
    parser.add_argument("--structure-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--no-replay",
        action="store_false",
        dest="replay",
        help="Skip the deterministic second resolution pass.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the selected corpus and persist the complete baseline report."""

    parser = build_parser()
    arguments = parser.parse_args(argv)
    uses_structure_root = (
        arguments.structure_registry is not None
        or arguments.topology_manifest is not None
    )
    if uses_structure_root:
        if arguments.structure_root is None:
            parser.error("file-backed corpora require --structure-root")
        if arguments.structure_registry is not None:
            corpus = load_structure_registry(
                arguments.structure_registry,
                arguments.structure_root,
            )
        else:
            corpus = load_topology_manifest(
                arguments.topology_manifest,
                arguments.structure_root,
            )
    else:
        if arguments.structure_root is not None:
            parser.error("--structure-root is valid only for file-backed corpora")
        corpus = load_corpus(arguments.smiles_corpus)
    summary = run_benchmark(
        corpus,
        arguments.output,
        replay=arguments.replay,
    )
    print(
        "scientific-state pre-activation baseline: "
        f"{summary['resolved_count']}/{summary['requested_count']} resolved; "
        f"{summary['replay_mismatch_count']} replay mismatches"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
