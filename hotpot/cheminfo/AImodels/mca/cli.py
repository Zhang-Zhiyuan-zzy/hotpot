"""Command-line interface for site-resolved MCA inference."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from hotpot.cheminfo._io import MolReader

from .api import MCAPredictor


def add_arguments(parser: argparse.ArgumentParser) -> None:
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
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="inference device (default: auto)",
    )
    parser.add_argument("--variant", choices=("fp32", "fp16"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--conformer-seed", type=int, default=42)
    parser.add_argument(
        "--allow-charged",
        action="store_true",
        help="allow out-of-domain predictions for charged molecules",
    )


def _read_molecules(sources: list[str], input_format: str | None):
    molecules = []
    for source in sources:
        if os.path.isfile(source):
            molecules.extend(MolReader(source, fmt=input_format))
        else:
            molecules.append(next(MolReader(source, fmt=input_format or "smi")))
    if not molecules:
        raise ValueError("No molecules were found in the input")
    return molecules


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


def format_predictions(predictions) -> str:
    predictions = list(predictions)
    if len(predictions) == 1:
        return _format_table(predictions[0])
    return "\n\n".join(
        f"Molecule {index}: {prediction.smiles}\n{_format_table(prediction)}"
        for index, prediction in enumerate(predictions, start=1)
    )


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
    predictor = MCAPredictor(
        model_dir=args.model_dir,
        device=args.device,
        variant=args.variant,
        batch_size=args.batch_size,
        conformer_seed=args.conformer_seed,
        allow_charged=args.allow_charged,
    )
    predictions = predictor.predict(molecules)
    text = format_predictions(predictions)

    if args.output:
        Path(args.output).write_text(f"{text}\n", encoding="utf-8")
    else:
        print(text)

    if args.plot:
        save_plot(molecules, predictions, args.plot, args.all_site)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict site-resolved MCA values")
    add_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
