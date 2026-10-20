"""Command-line interface for site-resolved MCA inference."""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from contextlib import ExitStack, nullcontext
from importlib import resources
from pathlib import Path

import numpy as np

from hotpot._cli import MarkdownDocumentationAction
from hotpot.cheminfo._io import MolReader

from .api import MCACpuPredictor, MCAGpuPredictor
from .gpu_memory import SUPPORTED_GPU_BATCH_SIZES
from .pipeline import DEFAULT_CONFORMER_SEED


def load_cli_documentation() -> str:
    return (
        resources.files(__package__)
        .joinpath("cli_doc.md")
        .read_text(encoding="utf-8")
    )


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--doc",
        action=MarkdownDocumentationAction,
        nargs=0,
        document_loader=load_cli_documentation,
        help="show detailed Markdown usage documentation and exit",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        metavar="SMILES/FILE",
        help="one or more SMILES strings or molecule files",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="write the text table to this file instead of standard output",
    )
    parser.add_argument(
        "--plot",
        metavar="IMAGE",
        help="draw MCA-coloured atoms to an image file",
    )
    parser.add_argument(
        "--all-site",
        action="store_true",
        help="colour every atom in the plot (default: detected nucleophilic sites only)",
    )
    parser.add_argument(
        "--input-format",
        help="input format override, such as smi, sdf, mol2, or cif",
    )
    parser.add_argument("--model-dir", help="directory containing the MCA ONNX release")
    parser.add_argument(
        "--model-source",
        choices=("auto", "local", "hub"),
        help="model source policy (default: HOTPOT_MODEL_SOURCE or auto)",
    )
    parser.add_argument(
        "--backend",
        choices=("cpu", "gpu"),
        required=True,
        help="explicit inference backend",
    )
    parser.add_argument("--variant", choices=("fp32", "fp16"))
    parser.add_argument(
        "--batch-size",
        type=int,
        help="CPU atom-row batch size (default: 64)",
    )
    gpu_selection = parser.add_mutually_exclusive_group()
    gpu_selection.add_argument(
        "--gpu-count",
        type=int,
        help="use logical GPUs 0 through N-1",
    )
    gpu_selection.add_argument(
        "--gpu-indices",
        type=lambda value: tuple(int(index) for index in value.split(",")),
        help="comma-separated logical GPU indices, such as 0,2",
    )
    parser.add_argument("--site-cpu-per-gpu", type=int)
    parser.add_argument("--infer-cpu-per-gpu", type=int)
    parser.add_argument(
        "--gpu-batch-size",
        type=int,
        choices=SUPPORTED_GPU_BATCH_SIZES,
        help="override the total-VRAM-derived GPU atom-row batch",
    )
    parser.add_argument(
        "--conformer-seed",
        type=int,
        default=DEFAULT_CONFORMER_SEED,
    )
    parser.add_argument(
        "--allow-charged",
        action="store_true",
        help="allow out-of-domain predictions for charged molecules",
    )


def _read_molecules(sources: list[str], input_format: str | None):
    found = False
    for source in sources:
        if os.path.isfile(source):
            for molecule in MolReader(source, fmt=input_format):
                found = True
                yield molecule
        else:
            found = True
            yield next(MolReader(source, fmt=input_format or "smi"))
    if not found:
        raise ValueError("No molecules were found in the input")


def _format_table(prediction) -> str:
    site_indices = {site.atom_index for site in prediction.sites}
    rows = [
        (
            str(atom.atom_index + 1),
            atom.element,
            f"{atom.mca_kj_mol:.2f}",
            str(atom.atom_index in site_indices),
        )
        for atom in prediction.atom_predictions
    ]
    headers = ("No.", "Atom", "MCA(kJ/mol)", "is_Nuc_site")
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]

    def format_row(row):
        return "  ".join(value.ljust(width) for value, width in zip(row, widths)).rstrip()

    return "\n".join((format_row(headers), *(format_row(row) for row in rows)))


def _plot_data(molecules, predictions, all_sites: bool):
    rd_molecules = []
    atom_values = []
    highlighted_values = []
    for molecule, prediction in zip(molecules, predictions):
        rd_mol = molecule.to_rdmol()
        values_by_index = {
            atom.atom_index: atom.mca_kj_mol
            for atom in prediction.atom_predictions
        }
        selected = (
            set(values_by_index)
            if all_sites
            else {site.atom_index for site in prediction.sites}
        )
        values = np.full(rd_mol.GetNumAtoms(), np.nan, dtype=float)
        for atom_index in selected:
            value = values_by_index[atom_index]
            values[atom_index] = value
            rd_mol.GetAtomWithIdx(atom_index).SetProp("atomNote", f"{value:.2f}")
            highlighted_values.append(value)
        rd_molecules.append(rd_mol)
        atom_values.append(values)
    return rd_molecules, atom_values, highlighted_values


def _plot_range(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    lower, upper = min(values), max(values)
    if lower == upper:
        padding = max(abs(lower) * 0.05, 1.0)
        return lower - padding, upper + padding
    return lower, upper


def save_plot(molecules, predictions, path: str, all_sites: bool) -> None:
    from hotpot.cheminfo.draw import draw_grid, draw_single_mol

    rd_molecules, atom_values, highlighted = _plot_data(
        molecules, predictions, all_sites
    )
    vmin, vmax = _plot_range(highlighted)
    highlight_options = {}
    if highlighted:
        highlight_options = {
            "cmap_name": "viridis",
            "vmin": vmin,
            "vmax": vmax,
            "threshold": -1.0,
            "colorbar": True,
        }

    if len(rd_molecules) == 1:
        draw_single_mol(
            rd_molecules[0],
            save_path=path,
            legend="MCA (kJ/mol)",
            atom_hl_values=atom_values[0] if highlighted else None,
            **highlight_options,
        )
        return

    draw_grid(
        rd_molecules,
        save_path=path,
        legends=[prediction.smiles for prediction in predictions],
        list_atom_values=atom_values if highlighted else None,
        **highlight_options,
    )


def run(args: argparse.Namespace) -> int:
    molecules = _read_molecules(args.inputs, args.input_format)
    predictor_options = {
        "model_dir": args.model_dir,
        "variant": args.variant,
        "conformer_seed": args.conformer_seed,
        "allow_charged": args.allow_charged,
        "model_source": args.model_source,
    }
    plot_molecules = []

    def tracked_molecules():
        for molecule in molecules:
            if args.plot:
                plot_molecules.append(molecule)
            yield molecule

    plot_predictions = []
    with ExitStack() as stack:
        if args.backend == "cpu":
            if any(
                value is not None
                for value in (
                    args.gpu_count,
                    args.gpu_indices,
                    args.gpu_batch_size,
                    args.site_cpu_per_gpu,
                    args.infer_cpu_per_gpu,
                )
            ):
                raise ValueError("GPU options require --backend gpu")
            cpu_options = (
                {} if args.batch_size is None else {"batch_size": args.batch_size}
            )
            predictor = MCACpuPredictor(**cpu_options, **predictor_options)
        else:
            if args.batch_size is not None:
                raise ValueError("--batch-size is CPU-only; use --gpu-batch-size")
            gpu_options = {
                name: value
                for name, value in (
                    ("site_cpu_per_gpu", args.site_cpu_per_gpu),
                    ("infer_cpu_per_gpu", args.infer_cpu_per_gpu),
                )
                if value is not None
            }
            predictor = MCAGpuPredictor(
                gpu_count=args.gpu_count,
                gpu_indices=args.gpu_indices,
                gpu_batch_size=args.gpu_batch_size,
                **gpu_options,
                **predictor_options,
            )
        stack.enter_context(predictor)
        output_context = (
            Path(args.output).open("w", encoding="utf-8")
            if args.output
            else nullcontext(sys.stdout)
        )
        stream = stack.enter_context(output_context)
        predictions = iter(predictor.iter_predict(tracked_molecules()))
        first = next(predictions, None)
        second = next(predictions, None)
        if first is not None and second is None:
            stream.write(f"{_format_table(first)}\n")
            if args.plot:
                plot_predictions.append(first)
        elif first is not None:
            for index, prediction in enumerate(
                itertools.chain((first, second), predictions),
                start=1,
            ):
                if index > 1:
                    stream.write("\n")
                stream.write(
                    f"Molecule {index}: {prediction.smiles}\n"
                    f"{_format_table(prediction)}\n"
                )
                if args.plot:
                    plot_predictions.append(prediction)

    if args.plot:
        save_plot(plot_molecules, plot_predictions, args.plot, args.all_site)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict site-resolved MCA values")
    add_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
