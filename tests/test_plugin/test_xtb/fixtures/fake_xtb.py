#!/usr/bin/env python3
"""Deterministic xTB command double used by the public workflow tests."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def _read_scenario() -> str:
    config_path = Path(__file__).with_name("scenario.json")
    return json.loads(config_path.read_text(encoding="utf-8"))["scenario"]


def _record_arguments(arguments: list[str], *, version_probe: bool) -> None:
    record_path = (
        Path(__file__).with_name("version_argv.json")
        if version_probe
        else Path("argv.json")
    )
    record_path.write_text(json.dumps(arguments), encoding="utf-8")


def _input_path(arguments: list[str]) -> Path:
    for argument in arguments:
        path = Path(argument)
        if path.is_file() and path.suffix.lower() in {
            ".xyz",
            ".coord",
            ".sdf",
            ".mol",
        }:
            return path
    raise RuntimeError("fake xTB did not receive a supported molecular input")


def _read_xyz(path: Path) -> tuple[list[str], list[list[float]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    atom_count = int(lines[0])
    atom_lines = lines[2 : 2 + atom_count]
    symbols = []
    coordinates = []
    for line in atom_lines:
        symbol, x, y, z = line.split()[:4]
        symbols.append(symbol)
        coordinates.append([float(x), float(y), float(z)])
    return symbols, coordinates


def _read_sdf(path: Path) -> tuple[list[str], list[list[float]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    atom_count = int(lines[3][:3])
    symbols = []
    coordinates = []
    for line in lines[4 : 4 + atom_count]:
        symbols.append(line[31:34].strip())
        coordinates.append(
            [float(line[0:10]), float(line[10:20]), float(line[20:30])]
        )
    return symbols, coordinates


def _write_xyz(
    path: Path,
    symbols: list[str],
    coordinates: list[list[float]],
) -> None:
    lines = [str(len(symbols)), "fake xTB optimized geometry"]
    lines.extend(
        f"{symbol} {x:.12f} {y:.12f} {z:.12f}"
        for symbol, (x, y, z) in zip(symbols, coordinates)
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_sdf(
    path: Path,
    symbols: list[str],
    coordinates: list[list[float]],
) -> None:
    lines = [
        "fake xTB optimized geometry",
        "  fake xTB",
        "",
        f"{len(symbols):>3}{0:>3}  0  0  0  0            999 V2000",
    ]
    lines.extend(
        f"{x:>10.4f}{y:>10.4f}{z:>10.4f} "
        f"{symbol:<3} 0  0  0  0  0  0  0  0  0  0  0  0"
        for symbol, (x, y, z) in zip(symbols, coordinates)
    )
    lines.extend(("M  END", "$$$$"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_json(energy: float, gradient_norm: float, atom_count: int) -> None:
    payload = {
        "total energy": energy,
        "total energy unit": "Eh",
        "number of atoms": atom_count,
        "gradient norm": gradient_norm,
    }
    Path("xtbout.json").write_text(
        json.dumps(payload, allow_nan=True),
        encoding="utf-8",
    )


def _write_gfnff_json(energy: float, gradient_norm: float) -> None:
    payload = {
        "total energy": energy,
        "gradient norm": gradient_norm,
        "method": "GFN-FF",
        "xtb version": "6.7.1 (fake contract backend)",
    }
    Path("gfnff_lists.json").write_text(
        json.dumps(payload, allow_nan=True),
        encoding="utf-8",
    )


def main() -> int:
    arguments = sys.argv[1:]
    scenario = _read_scenario()
    if "--version" in arguments or "-v" in arguments:
        _record_arguments(arguments, version_probe=True)
        if scenario == "version_nonzero":
            print("fake version probe failure", file=sys.stderr)
            return 19
        if scenario == "version_malformed":
            print("fake backend without a parseable version")
            return 0
        print("xtb version 6.7.1 (fake contract backend)")
        return 0

    _record_arguments(arguments, version_probe=False)
    input_path = _input_path(arguments)
    symbols, coordinates = (
        _read_sdf(input_path)
        if input_path.suffix.lower() in {".sdf", ".mol"}
        else _read_xyz(input_path)
    )
    is_optimization = "--opt" in arguments
    is_gfnff = "--gfnff" in arguments

    if scenario == "nonzero":
        print("fake execution failure", file=sys.stderr)
        return 17

    if scenario != "missing_artifact":
        energy = math.nan if scenario == "nonfinite_energy" else -5.123456789
        gradient_norm = math.inf if scenario == "nonfinite_gradient" else 0.00001
        if is_gfnff:
            _write_gfnff_json(energy, gradient_norm)
        else:
            _write_json(energy, gradient_norm, len(symbols))

    if is_optimization:
        optimized_symbols = list(symbols)
        optimized_coordinates = [
            [coordinate + 0.1 for coordinate in atom_coordinates]
            for atom_coordinates in coordinates
        ]
        if scenario == "nonfinite_coordinates":
            optimized_coordinates[0][0] = math.inf
        elif scenario == "truncated_geometry":
            optimized_symbols.pop()
            optimized_coordinates.pop()
        elif scenario == "reordered_elements":
            optimized_symbols = list(reversed(optimized_symbols))

        if is_gfnff and input_path.suffix.lower() in {".sdf", ".mol"}:
            _write_sdf(
                Path(f"xtbopt{input_path.suffix.lower()}"),
                optimized_symbols,
                optimized_coordinates,
            )
        else:
            _write_xyz(
                Path("xtbopt.xyz"),
                optimized_symbols,
                optimized_coordinates,
            )
        Path("xtbopt.log").write_text(
            "fake xTB optimization trajectory\n",
            encoding="utf-8",
        )
        if scenario != "nonconvergence":
            Path(".xtboptok").touch()

    if scenario == "stderr_success":
        print("fake informational diagnostic", file=sys.stderr)

    if is_optimization and scenario == "nonconvergence":
        print("GEOMETRY OPTIMIZATION DID NOT CONVERGE")
    elif is_optimization:
        print("GEOMETRY OPTIMIZATION CONVERGED")
    print("TOTAL ENERGY      -5.123456789 Eh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
