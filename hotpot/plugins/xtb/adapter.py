"""Translate Hotpot molecules and validated xTB artifacts."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from hotpot.cheminfo.core import BondKind, Molecule

from .contracts import (
    XTBGeometry,
    XTBInputError,
    XTBMethod,
    XTBParsedResult,
    XTBResultError,
    XTBRunReport,
    XTBTask,
)


__all__ = [
    "commit_xtb_coordinates",
    "molecule_to_xtb_geometry",
    "parse_xtb_artifacts",
    "prepare_xtb_input",
]


_FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"
_GFNFF_FIELDS = {
    "total energy": re.compile(
        rf'(?:\A\s*\{{|,)\s*"total energy"\s*:\s*({_FLOAT_PATTERN})(?=\s*[,}}])'
    ),
    "gradient norm": re.compile(
        rf'(?:\A\s*\{{|,)\s*"gradient norm"\s*:\s*({_FLOAT_PATTERN})(?=\s*[,}}])'
    ),
}


def _validate_geometry(geometry: XTBGeometry) -> None:
    atom_count = len(geometry.symbols)
    if atom_count == 0:
        raise XTBInputError("xTB requires a non-empty geometry")
    if len(geometry.coordinates) != atom_count:
        raise XTBInputError("xTB symbol and coordinate counts differ")
    if any(not symbol for symbol in geometry.symbols):
        raise XTBInputError("xTB geometry contains an empty element symbol")
    if any(len(row) != 3 for row in geometry.coordinates):
        raise XTBInputError("xTB coordinates must have exactly three columns")
    if any(
        not math.isfinite(value)
        for row in geometry.coordinates
        for value in row
    ):
        raise XTBInputError("xTB geometry contains non-finite coordinates")
    if atom_count > 1 and len(set(geometry.coordinates)) == 1:
        raise XTBInputError("xTB geometry has all atoms at the same position")


def _require_complete_explicit_atoms(mol: Molecule) -> None:
    for atom in mol.heavy_atoms:
        explicit_hydrogen_count = len(atom.hydrogens)
        if atom.implicit_hydrogens and (
            explicit_hydrogen_count != atom.implicit_hydrogens
        ):
            raise XTBInputError(
                f"Atom {atom.idx} ({atom.symbol}) stores "
                f"{atom.implicit_hydrogens} implicit hydrogens but has "
                f"{explicit_hydrogen_count} explicit hydrogen atoms"
            )


def _write_xyz(geometry: XTBGeometry, path: Path) -> None:
    lines = (str(len(geometry.symbols)), "Hotpot xTB input")
    atom_lines = tuple(
        f"{symbol} {x:.17g} {y:.17g} {z:.17g}"
        for symbol, (x, y, z) in zip(
            geometry.symbols,
            geometry.coordinates,
        )
    )
    path.write_text("\n".join((*lines, *atom_lines)) + "\n", encoding="utf-8")


def _sdf_bond_order(bond_kind: BondKind, bond_order: float) -> int:
    if bond_kind is BondKind.AROMATIC or bond_order == 1.5:
        return 4
    rounded = int(round(bond_order))
    return rounded if rounded in {1, 2, 3} else 1


def _write_sdf(mol: Molecule, geometry: XTBGeometry, path: Path) -> None:
    atoms = tuple(mol.atoms)
    bonds = tuple(mol.bonds)
    if len(atoms) > 999 or len(bonds) > 999:
        raise XTBInputError("GFN-FF SDF input currently supports V2000 limits")

    lines = [
        "Hotpot GFN-FF input",
        "  Hotpot",
        "",
        f"{len(atoms):>3}{len(bonds):>3}  0  0  0  0            999 V2000",
    ]
    for symbol, (x, y, z) in zip(geometry.symbols, geometry.coordinates):
        lines.append(
            f"{x:>10.4f}{y:>10.4f}{z:>10.4f} "
            f"{symbol:<3} 0  0  0  0  0  0  0  0  0  0  0  0"
        )
    for bond in bonds:
        lines.append(
            f"{bond.a1idx + 1:>3}{bond.a2idx + 1:>3}"
            f"{_sdf_bond_order(bond.bond_kind, bond.bond_order):>3}"
            "  0  0  0  0"
        )

    charged_atoms = tuple(
        (index + 1, int(atom.formal_charge))
        for index, atom in enumerate(atoms)
        if atom.formal_charge
    )
    for offset in range(0, len(charged_atoms), 8):
        group = charged_atoms[offset : offset + 8]
        charge_fields = "".join(
            f"{atom_index:>4}{formal_charge:>4}"
            for atom_index, formal_charge in group
        )
        lines.append(f"M  CHG{len(group):>3}{charge_fields}")
    lines.extend(("M  END", "$$$$"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_xyz(path: Path) -> XTBGeometry:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError("XYZ must contain an atom-count line and a comment line")
    try:
        atom_count = int(lines[0].strip())
    except ValueError as error:
        raise ValueError("XYZ atom count is not an integer") from error
    if len(lines) != atom_count + 2:
        raise ValueError("XYZ atom count does not match its coordinate records")

    symbols = []
    coordinates = []
    for line in lines[2:]:
        fields = line.split()
        if len(fields) != 4:
            raise ValueError("XYZ atom records must contain symbol, x, y and z")
        symbols.append(fields[0])
        coordinates.append(tuple(float(value) for value in fields[1:]))
    return XTBGeometry(tuple(symbols), tuple(coordinates))


def _read_sdf(path: Path) -> XTBGeometry:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 4 or "V2000" not in lines[3]:
        raise ValueError("SDF optimization output must contain one V2000 record")
    try:
        atom_count = int(lines[3][:3])
        bond_count = int(lines[3][3:6])
    except ValueError as error:
        raise ValueError("SDF counts line is malformed") from error
    if len(lines) < 4 + atom_count + bond_count:
        raise ValueError("SDF atom or bond records are truncated")

    symbols = []
    coordinates = []
    for line in lines[4 : 4 + atom_count]:
        try:
            coordinate = (
                float(line[0:10]),
                float(line[10:20]),
                float(line[20:30]),
            )
        except ValueError as error:
            raise ValueError("SDF atom coordinates are malformed") from error
        symbol = line[31:34].strip()
        if not symbol:
            raise ValueError("SDF atom symbol is empty")
        symbols.append(symbol)
        coordinates.append(coordinate)
    return XTBGeometry(tuple(symbols), tuple(coordinates))


def _reject_nonstandard_json_number(value: str) -> float:
    raise ValueError(f"non-standard JSON number {value!r}")


def _require_finite_json(value: object) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ValueError("JSON result contains a non-finite number")
        return
    if isinstance(value, dict):
        for item in value.values():
            _require_finite_json(item)
        return
    if isinstance(value, list):
        for item in value:
            _require_finite_json(item)
        return
    raise ValueError("JSON result contains an unsupported value")


def _finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name!r} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name!r} is not finite")
    return result


def _optional_charges(
    values: object,
    atom_count: int,
) -> Optional[Tuple[float, ...]]:
    if values is None:
        return None
    if not isinstance(values, list) or len(values) != atom_count:
        raise ValueError("partial-charge count does not match the atom count")
    return tuple(
        _finite_number(value, "partial charge")
        for value in values
    )


def _read_charge_artifact(
    path: Path,
    atom_count: int,
) -> Tuple[float, ...]:
    charges = tuple(float(value) for value in path.read_text(encoding="utf-8").split())
    if len(charges) != atom_count:
        raise ValueError("charge artifact count does not match the atom count")
    if any(not math.isfinite(value) for value in charges):
        raise ValueError("charge artifact contains a non-finite number")
    return charges


def _parse_gfn_xtb_result(
    path: Path,
    atom_count: int,
) -> Tuple[float, Optional[float], Optional[Tuple[float, ...]]]:
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_nonstandard_json_number,
    )
    if not isinstance(payload, dict):
        raise ValueError("xtbout.json must contain one JSON object")
    _require_finite_json(payload)
    energy = _finite_number(payload.get("total energy"), "total energy")
    gradient_value = payload.get("gradient norm")
    gradient = (
        None
        if gradient_value is None
        else _finite_number(gradient_value, "gradient norm")
    )
    reported_count = payload.get("number of atoms")
    if reported_count is not None:
        if (
            isinstance(reported_count, bool)
            or not isinstance(reported_count, int)
            or reported_count != atom_count
        ):
            raise ValueError("reported atom count does not match the input geometry")
    charges = _optional_charges(payload.get("partial charges"), atom_count)
    return energy, gradient, charges


def _parse_gfnff_result(path: Path) -> Tuple[float, float]:
    values: Dict[str, float] = {}
    text = path.read_text(encoding="utf-8")
    for field_name, pattern in _GFNFF_FIELDS.items():
        matches = pattern.findall(text)
        if len(matches) > 1:
            raise ValueError(f"GFN-FF result repeats top-level field {field_name!r}")
        if matches:
            values[field_name] = float(
                matches[0].replace("D", "E").replace("d", "e")
            )
    missing = tuple(field for field in _GFNFF_FIELDS if field not in values)
    if missing:
        raise ValueError(f"GFN-FF result lacks top-level fields {missing!r}")
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("GFN-FF result contains a non-finite primary metric")
    return values["total energy"], values["gradient norm"]


def _artifact_path(report: XTBRunReport, name: str) -> Path:
    artifact = report.artifacts.get(name)
    if artifact is None:
        raise ValueError(f"required xTB artifact is absent: {name}")
    return artifact.path


def molecule_to_xtb_geometry(mol: Molecule) -> XTBGeometry:
    """Return a validated immutable xTB geometry without changing ``mol``."""
    _require_complete_explicit_atoms(mol)
    geometry = XTBGeometry(
        symbols=tuple(atom.symbol for atom in mol.atoms),
        coordinates=tuple(
            tuple(float(value) for value in atom.coordinates)
            for atom in mol.atoms
        ),
    )
    _validate_geometry(geometry)
    return geometry


def prepare_xtb_input(mol: Molecule, input_path: Path) -> XTBGeometry:
    """Write method-native input selected by the explicit file suffix."""
    geometry = molecule_to_xtb_geometry(mol)
    try:
        suffix = input_path.suffix.lower()
        if suffix == ".xyz":
            _write_xyz(geometry, input_path)
        elif suffix in {".sdf", ".mol"}:
            _write_sdf(mol, geometry, input_path)
        else:
            raise XTBInputError(
                "xTB input suffix must be .xyz, .sdf, or .mol"
            )
    except OSError as error:
        raise XTBInputError(f"Cannot write xTB input: {input_path}") from error
    return geometry


def parse_xtb_artifacts(
    report: XTBRunReport,
    expected_geometry: XTBGeometry,
) -> XTBParsedResult:
    """Parse complete finite xTB results without mutating a molecule."""
    _validate_geometry(expected_geometry)
    try:
        if not report.process_succeeded or not report.converged:
            raise ValueError("xTB report is not a converged successful run")
        if report.effective_method is XTBMethod.GFNFF:
            energy, gradient = _parse_gfnff_result(
                _artifact_path(report, "gfnff_lists.json")
            )
            charge_artifact = report.artifacts.get("gfnff_charges")
            charges = (
                None
                if charge_artifact is None
                else _read_charge_artifact(
                    charge_artifact.path,
                    len(expected_geometry.symbols),
                )
            )
        else:
            energy, gradient, charges = _parse_gfn_xtb_result(
                _artifact_path(report, "xtbout.json"),
                len(expected_geometry.symbols),
            )
            charge_artifact = report.artifacts.get("charges")
            if charges is None and charge_artifact is not None:
                charges = _read_charge_artifact(
                    charge_artifact.path,
                    len(expected_geometry.symbols),
                )

        optimized_geometry = None
        if report.task is XTBTask.OPTIMIZE:
            input_suffix = Path(report.argv[1]).suffix.lower()
            optimized_path = _artifact_path(
                report,
                f"xtbopt{input_suffix}",
            )
            optimized_geometry = (
                _read_sdf(optimized_path)
                if input_suffix in {".sdf", ".mol"}
                else _read_xyz(optimized_path)
            )
            _validate_geometry(optimized_geometry)
            if optimized_geometry.symbols != expected_geometry.symbols:
                raise ValueError("optimized XYZ changed the input atom order")
    except (OSError, UnicodeError, ValueError, XTBInputError) as error:
        raise XTBResultError(f"Invalid xTB result artifact: {error}", report) from error

    return XTBParsedResult(
        energy_hartree=energy,
        gradient_norm=gradient,
        partial_charges=charges,
        optimized_geometry=optimized_geometry,
        atom_order_verified=True,
    )


def commit_xtb_coordinates(mol: Molecule, result: XTBParsedResult) -> None:
    """Atomically apply a validated optimization geometry to ``mol``."""
    from hotpot.cheminfo.structure._commit import _FactTransaction
    from hotpot.cheminfo.structure._fields import ATOM_SOURCE_CODES

    with _FactTransaction(mol, producer=ATOM_SOURCE_CODES["xtb"]):
        geometry = result.optimized_geometry
        if geometry is None:
            raise XTBInputError("A single-point xTB result has no coordinates to commit")
        if not result.atom_order_verified:
            raise XTBInputError("xTB result atom order has not been verified")
        _validate_geometry(geometry)
        if geometry.symbols != tuple(atom.symbol for atom in mol.atoms):
            raise XTBInputError("xTB result atom order does not match the molecule")

        mol.coordinates = np.asarray(geometry.coordinates, dtype=float)
        mol._obmol = None
        mol._row2idx = None
