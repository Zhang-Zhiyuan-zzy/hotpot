"""Command-line interface for coordination-bond inference."""

from __future__ import annotations

import argparse
from pathlib import Path

from hotpot.cheminfo.convert import to_hotpot_mol

from .apply import auto_build_cbond, get_cbond_runtime


def add_arguments(parser: argparse.ArgumentParser) -> None:
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
        help="write the resulting SMILES to this file instead of standard output",
    )
    parser.add_argument(
        "--input-format",
        help="input format override, such as smi, sdf, mol2, or cif",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="minimum raw CBond model score (default: 0.0)",
    )
    parser.add_argument(
        "--no-greedy",
        dest="greedy",
        action="store_false",
        help="stop when the highest-scoring candidate is already connected",
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


def run(args: argparse.Namespace) -> int:
    ligand = to_hotpot_mol(args.ligand, fmt=args.input_format)
    runtime = get_cbond_runtime(args.device, args.model_dir)
    result, _ = auto_build_cbond(
        ligand,
        _parse_metal(args.metal),
        threshold=args.threshold,
        greedy=args.greedy,
        runtime=runtime,
    )
    smiles = result.smiles

    if args.output:
        Path(args.output).write_text(f"{smiles}\n", encoding="utf-8")
    else:
        print(smiles)
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
