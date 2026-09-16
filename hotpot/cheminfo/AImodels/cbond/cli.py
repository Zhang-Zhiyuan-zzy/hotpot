"""Command-line interface for coordination-bond inference."""

from __future__ import annotations

import argparse
from importlib import resources
from pathlib import Path

from hotpot.cheminfo.convert import to_hotpot_mol

from .apply import auto_build_cbond, build_all_possible_cbond, get_cbond_runtime
from .constants import DEFAULT_CBOND_THRESHOLD, DEFAULT_MAX_STATES


def load_cli_documentation() -> str:
    return (
        resources.files(__package__).joinpath("cli_doc.md").read_text(encoding="utf-8")
    )


class _DocumentationAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        from rich.console import Console
        from rich.markdown import Markdown

        Console().print(Markdown(load_cli_documentation()))
        parser.exit()


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--doc",
        action=_DocumentationAction,
        nargs=0,
        help="show detailed Markdown usage documentation and exit",
    )
    parser.add_argument(
        "metal",
        help="metal element symbol or atomic number, such as Eu or 63",
    )
    parser.add_argument(
        "ligand",
        metavar="LIGAND_SMILES/FILE",
        help="ligand SMILES string or molecule file",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="write the result to this file instead of standard output",
    )
    parser.add_argument(
        "--input-format",
        help="input format override, such as smi, sdf, mol2, or cif",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_CBOND_THRESHOLD,
        help=(f"minimum raw CBond model score (default: {DEFAULT_CBOND_THRESHOLD})"),
    )
    parser.add_argument(
        "--no-greedy",
        dest="greedy",
        action="store_false",
        help="stop when the highest-scoring candidate is already connected",
    )
    parser.add_argument(
        "--all-structures",
        action="store_true",
        help="enumerate and rank every terminal coordination structure",
    )
    parser.add_argument(
        "--bond-detail",
        action="store_true",
        help="show atom indices, elements, and raw scores for selected bonds",
    )
    parser.add_argument(
        "--max-states",
        type=int,
        default=DEFAULT_MAX_STATES,
        help=(
            "maximum unique states explored by --all-structures "
            f"(default: {DEFAULT_MAX_STATES})"
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="inference device (default: auto)",
    )
    parser.add_argument(
        "--model-dir",
        help="directory containing the CBond ONNX release",
    )


def _parse_metal(value: str):
    return int(value) if value.isdecimal() else value


def _format_bond_detail(steps) -> str:
    rows = [(str(step.atom_index), step.element, f"{step.score:.5f}") for step in steps]
    headers = ("AtomIdx", "Atom", "Score")
    widths = [
        max([len(header), *(len(row[index]) for row in rows)])
        for index, header in enumerate(headers)
    ]

    def format_row(row):
        return "  ".join(
            value.ljust(width) for value, width in zip(row, widths)
        ).rstrip()

    return "\n".join(
        ("Cbond Detail:", format_row(headers), *(format_row(row) for row in rows))
    )


def format_single_result(result, bond_detail: bool = False) -> str:
    lines = [result.molecule.smiles]
    if bond_detail:
        lines.append(_format_bond_detail(result.steps))
        lines.append("-- End --")
    return "\n".join(lines)


def format_all_results(results, bond_detail: bool = False) -> str:
    results = list(results)
    if not results:
        return "No coordination structures exceeded the threshold.\n-- End --"

    blocks = []
    for rank, result in enumerate(results, start=1):
        lines = [
            f"{result.molecule.smiles}  --> Rank {rank}: Prob: {result.probability:.1%}"
        ]
        if bond_detail:
            lines.append(_format_bond_detail(result.steps))
        blocks.append("\n".join(lines))
    return "\n-----\n".join(blocks) + "\n-- End --"


def run(args: argparse.Namespace) -> int:
    ligand = to_hotpot_mol(args.ligand, fmt=args.input_format)
    runtime = get_cbond_runtime(args.device, args.model_dir)
    metal = _parse_metal(args.metal)

    if args.all_structures:
        results = build_all_possible_cbond(
            ligand,
            metal,
            threshold=args.threshold,
            greedy=args.greedy,
            runtime=runtime,
            max_states=args.max_states,
            return_details=True,
        )
        text = format_all_results(results, args.bond_detail)
    else:
        result = auto_build_cbond(
            ligand,
            metal,
            threshold=args.threshold,
            greedy=args.greedy,
            runtime=runtime,
            return_details=True,
        )
        text = format_single_result(result, args.bond_detail)

    if args.output:
        Path(args.output).write_text(f"{text}\n", encoding="utf-8")
    else:
        print(text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build metal-ligand coordination bonds with the CBond model"
    )
    add_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
