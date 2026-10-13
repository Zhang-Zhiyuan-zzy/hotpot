"""Private structure rendering and atomic dataset export helpers."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .public.structures import StructureAccessError, _FORMAT_ALIASES


# Export prefers coordinate-bearing sources so a requested geometry format never
# falls back to topology-only SMILES merely because it is also stored.
_SOURCE_PRIORITY = ("mol2", "sdf", "mol", "cif", "pdb", "xyz", "smi")
_COORDINATE_FORMATS = frozenset({"mol2", "sdf", "mol", "cif", "pdb", "xyz"})
_FORMAT_TOKEN = re.compile(r"[a-z0-9][a-z0-9_+.-]*\Z")


class ExportError(ValueError):
    """A requested structure cannot be rendered or exported safely."""


def normalize_format(fmt: str) -> str:
    """Return a safe, normalized filename/output-format token."""
    normalized = str(fmt).strip().lower().lstrip(".")
    normalized = _FORMAT_ALIASES.get(normalized, normalized)
    if not normalized or not _FORMAT_TOKEN.fullmatch(normalized):
        raise ExportError(f"invalid structure format {fmt!r}")
    return normalized


def _native_asset(structure):
    assets = [asset for asset in structure.assets if asset.format]
    if not assets:
        raise ExportError(f"structure {structure.structure_id!r} has no stored format")
    coordinate_assets = [
        asset for asset in assets if asset.metadata.get("role") == "coordinates"
    ]
    candidates = coordinate_assets or assets
    priority = {fmt: index for index, fmt in enumerate(_SOURCE_PRIORITY)}
    return min(
        candidates,
        key=lambda asset: (priority.get(asset.format, len(priority)), asset.asset_id),
    )


def _source_asset(structure, target_format: str):
    base_priorities = (
        tuple(fmt for fmt in _SOURCE_PRIORITY if fmt in _COORDINATE_FORMATS)
        if target_format == "xyz"
        else _SOURCE_PRIORITY
    )
    priorities = (
        (target_format,)
        + tuple(fmt for fmt in base_priorities if fmt != target_format)
        if target_format in base_priorities
        else base_priorities
    )
    for fmt in priorities:
        asset = structure.asset(fmt)
        if asset is not None:
            return asset
    if target_format == "xyz":
        detail = "no coordinate-bearing source format"
    else:
        detail = "no Hotpot-readable source format"
    raise ExportError(f"structure {structure.structure_id!r} has {detail}")


def _validate_output_format(fmt: str) -> None:
    try:
        from openbabel import openbabel as ob

        conversion = ob.OBConversion()
        supported = conversion.SetOutFormat(fmt)
    except Exception as error:
        raise ExportError(
            f"cannot initialize structure output format {fmt!r}: {error}"
        ) from error
    if not supported:
        raise ExportError(f"unsupported structure output format {fmt!r}")


def _validate_xyz_coordinates(molecule: Any, structure_id: str) -> None:
    import numpy as np

    coordinates = np.asarray(molecule.coordinates, dtype=float)
    if coordinates.shape != (len(molecule.atoms), 3) or not np.isfinite(coordinates).all():
        raise ExportError(f"structure {structure_id!r} has invalid coordinates")
    if len(coordinates) > 1 and np.allclose(coordinates, coordinates[0]):
        raise ExportError(
            f"structure {structure_id!r} has collapsed coordinates and cannot be converted to xyz"
        )


def _convert_structure(
    structure,
    target_format: str,
    data_id: str,
    *,
    source=None,
    validate_output: bool = True,
) -> tuple[str, str]:
    from hotpot.cheminfo.core import Molecule

    target_format = normalize_format(target_format)
    if validate_output:
        _validate_output_format(target_format)
    if source is None:
        source = _source_asset(structure, target_format)
    try:
        molecule = Molecule.read_one(source.text(), fmt=source.format)
        if target_format == "xyz":
            _validate_xyz_coordinates(molecule, structure.structure_id)
        if target_format == "cif" and molecule.crystal is None:
            raise ExportError(
                f"structure {structure.structure_id!r} has no unit cell "
                "and cannot be converted to cif"
            )
        rendered = (
            molecule.smiles + "\n"
            if target_format == "smi"
            else molecule.write(fmt=target_format, write_single=True)
        )
    except ExportError:
        raise
    except Exception as error:
        raise ExportError(
            f"cannot convert data {data_id!r} structure {structure.structure_id!r} "
            f"from {source.format!r} to {target_format!r}: {error}"
        ) from error
    if not isinstance(rendered, str) or not rendered.strip():
        raise ExportError(
            f"cannot convert data {data_id!r} structure {structure.structure_id!r} "
            f"from {source.format!r} to {target_format!r}: empty output"
        )
    return rendered, source.format


def _select_structure(
    data,
    *,
    structure_index: Optional[int],
    structure_role: Optional[str],
    structure_id: Optional[str],
):
    if structure_index is None and structure_role is None and structure_id is None:
        try:
            structure = data.structures._primary()
        except StructureAccessError as error:
            raise ExportError(str(error)) from error
        if structure is None:
            raise ExportError(f"data {data.id!r} has no structure")
        return structure

    structures = data.structures._rows()
    if not structures:
        raise ExportError(f"data {data.id!r} has no structure")
    if structure_index is not None:
        if structure_index < 0 or structure_index >= len(structures):
            raise ExportError(
                f"structure index {structure_index} is out of range for data {data.id!r}; "
                f"available indices: 0..{len(structures) - 1}"
            )
        return structures[structure_index]
    if structure_role is not None:
        matches = [
            structure for structure in structures if structure_role in structure.roles
        ]
        if not matches:
            available = sorted({role for structure in structures for role in structure.roles})
            raise ExportError(
                f"structure role {structure_role!r} was not found for data {data.id!r}; "
                f"available roles: {', '.join(available) or 'none'}"
            )
        if len(matches) > 1:
            ids = ", ".join(structure.structure_id for structure in matches)
            raise ExportError(
                f"structure role {structure_role!r} matches {len(matches)} structures "
                f"for data {data.id!r}: {ids}; use --structure-index or --structure-id"
            )
        return matches[0]

    matches = [structure for structure in structures if structure.structure_id == structure_id]
    if not matches:
        raise ExportError(
            f"structure ID {structure_id!r} was not found for data {data.id!r}"
        )
    return matches[0]


def render_structure_record(
    data,
    fmt: str,
    *,
    structure_index: Optional[int] = None,
    structure_role: Optional[str] = None,
    structure_id: Optional[str] = None,
) -> dict[str, Any]:
    """Render one structure and return its content with selection provenance."""
    target_format = normalize_format(fmt)
    structure = _select_structure(
        data,
        structure_index=structure_index,
        structure_role=structure_role,
        structure_id=structure_id,
    )
    direct = structure.asset(target_format)
    if direct is not None:
        try:
            rendered = direct.text()
        except StructureAccessError as error:
            raise ExportError(str(error)) from error
        if not rendered.strip():
            raise ExportError(
                f"data {data.id!r} structure {structure.structure_id!r} "
                f"contains an empty {target_format!r} asset"
            )
        source_format = direct.format
        converted = False
    else:
        rendered, source_format = _convert_structure(
            structure, target_format, data.id
        )
        converted = True
    selection_mode = next(
        (
            mode
            for mode, value in (
                ("index", structure_index),
                ("role", structure_role),
                ("id", structure_id),
            )
            if value is not None
        ),
        "primary",
    )
    return {
        "data_id": data.id,
        "structure_id": structure.structure_id,
        "structure_roles": list(structure.roles),
        "selection": {
            "mode": selection_mode,
            "structure_index": structure_index,
            "structure_role": structure_role,
            "structure_id": structure_id,
        },
        "format": target_format,
        "source_format": source_format,
        "converted": converted,
        "content": rendered,
    }


def _json_file(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _structure_rows(records: Sequence[Any]) -> list[tuple[Any, Any, int, int]]:
    rows = []
    for data in records:
        structures = data.structures._rows()
        if not structures:
            raise ExportError(f"data {data.id!r} has no structure")
        for structure_index, structure in enumerate(structures):
            rows.append((data, structure, structure_index, len(structures)))
    return rows


def export_records(
    dataset,
    output: Path,
    *,
    fmt: Optional[str],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Export selected records atomically without adding a public Python API."""
    records = tuple(dataset)
    if not records:
        raise ExportError("the export selection is empty")
    output = Path(output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"export target already exists: {output}")

    requested_format = normalize_format(fmt) if fmt is not None else None
    if requested_format is not None:
        _validate_output_format(requested_format)
    rows = _structure_rows(records)
    single = len(records) == 1 and len(rows) == 1

    plan = []
    for data, structure, structure_index, structure_count in rows:
        if requested_format is None:
            asset = _native_asset(structure)
            source_asset = None
            try:
                source_path = asset.path()
            except StructureAccessError as error:
                raise ExportError(str(error)) from error
            if source_path.stat().st_size == 0:
                raise ExportError(
                    f"structure {structure.structure_id!r} contains an empty asset"
                )
            source_format = asset.format
            output_format = normalize_format(source_format)
        else:
            source_path = None
            source_asset = _source_asset(structure, requested_format)
            source_format = source_asset.format
            output_format = requested_format

        if single:
            structure_name = f"struct.{output_format}"
            info_name = "info.json"
        else:
            stem = f"{data.index:06d}"
            if structure_count > 1:
                stem += f"-{structure_index:02d}"
            structure_name = f"struct-{stem}.{output_format}"
            info_name = f"info-{stem}.json"
        plan.append(
            {
                "data": data,
                "structure": structure,
                "source_path": source_path,
                "source_asset": source_asset,
                "source_format": source_format,
                "output_format": output_format,
                "structure_file": structure_name,
                "info_file": info_name,
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=str(output.parent))
    )
    files = []
    try:
        for item in plan:
            data = item["data"]
            structure = item["structure"]
            structure_path = temporary / item["structure_file"]
            if requested_format is None:
                shutil.copyfile(item["source_path"], structure_path)
            else:
                rendered, source_format = _convert_structure(
                    structure,
                    requested_format,
                    data.id,
                    source=item["source_asset"],
                    validate_output=False,
                )
                item["source_format"] = source_format
                structure_path.write_text(rendered, encoding="utf-8")

            file_info = {
                "data_id": data.id,
                "record_id": data.record_id,
                "structure_id": structure.structure_id,
                "structure_file": item["structure_file"],
                "info_file": item["info_file"],
                "source_format": item["source_format"],
                "output_format": item["output_format"],
            }
            info = dict(data.info)
            info["export"] = {
                **file_info,
                "roles": list(structure.roles),
                "converted": requested_format is not None,
            }
            _json_file(temporary / item["info_file"], info)
            files.append(file_info)

        summary = {
            "dataset": dataset.info,
            "selection": dict(selection),
            "record_count": len(records),
            "structure_count": len(rows),
            "requested_format": requested_format,
            "files": files,
        }
        if not single:
            _json_file(temporary / "summary.json", summary)
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return summary


__all__: list[str] = []
