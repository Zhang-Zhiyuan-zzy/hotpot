"""Private structure rendering and atomic export for current Dataset objects."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional

from .public.structures import StructureAccessError


_ALIASES = {"smiles": "smi"}
_SOURCE_PRIORITY = ("mol2", "sdf", "mol", "cif", "pdb", "xyz", "smi")
_TOKEN = re.compile(r"[a-z0-9][a-z0-9_+.-]*\Z")


class ExportError(ValueError):
    """A requested structure selection or conversion cannot be completed."""


def normalize_format(fmt: str) -> str:
    value = str(fmt).strip().lower().lstrip(".")
    value = _ALIASES.get(value, value)
    if not value or _TOKEN.fullmatch(value) is None:
        raise ExportError(f"invalid structure format {fmt!r}")
    return value


def _groups(data: Any) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    order = []
    for row in data.structures._rows():
        structure_id = str(row["structure_id"])
        if structure_id not in grouped:
            grouped[structure_id] = {
                "id": structure_id,
                "roles": set(),
                "primary": False,
                "rows": {},
            }
            order.append(structure_id)
        group = grouped[structure_id]
        group["roles"].add(str(row.get("role", "structure")))
        group["primary"] = group["primary"] or bool(row.get("primary"))
        group["rows"][normalize_format(row["format"])] = row
    return [grouped[structure_id] for structure_id in order]


def _select_group(
    data: Any,
    *,
    structure_index: Optional[int],
    structure_role: Optional[str],
    structure_id: Optional[str],
) -> dict[str, Any]:
    groups = _groups(data)
    if not groups:
        raise ExportError(f"data {data.id!r} has no structure")
    if structure_index is not None:
        if structure_index < 0 or structure_index >= len(groups):
            raise ExportError(f"structure index {structure_index} is out of range")
        return groups[structure_index]
    if structure_role is not None:
        matches = [group for group in groups if structure_role in group["roles"]]
        if len(matches) != 1:
            raise ExportError(
                f"structure role {structure_role!r} matched {len(matches)} structures"
            )
        return matches[0]
    if structure_id is not None:
        matches = [group for group in groups if group["id"] == structure_id]
        if len(matches) != 1:
            raise ExportError(f"structure ID {structure_id!r} was not found")
        return matches[0]
    if len(groups) == 1:
        return groups[0]
    primary = [group for group in groups if group["primary"]]
    if len(primary) != 1:
        raise ExportError(
            f"data {data.id!r} has multiple structures without exactly one primary"
        )
    return primary[0]


def _source_row(group: Mapping[str, Any], target: Optional[str] = None) -> Mapping[str, Any]:
    rows = group["rows"]
    if target is not None and target in rows:
        return rows[target]
    for fmt in _SOURCE_PRIORITY:
        if fmt in rows:
            return rows[fmt]
    return next(iter(rows.values()))


def _content(data: Any, row: Mapping[str, Any]) -> str:
    try:
        return data.structures._verified_path(dict(row)).read_text(encoding="utf-8")
    except (OSError, UnicodeError, StructureAccessError) as error:
        raise ExportError(str(error)) from error


def _convert(data: Any, group: Mapping[str, Any], target: str) -> tuple[str, str, bool]:
    target = normalize_format(target)
    row = _source_row(group, target)
    source = normalize_format(row["format"])
    if source == target:
        content = _content(data, row)
        if not content.strip():
            raise ExportError(f"structure {group['id']!r} contains an empty {target} asset")
        return content, source, False
    try:
        from hotpot import Molecule

        molecule = Molecule.read_one(str(data.structures._verified_path(dict(row))), source)
        content = (
            molecule.smiles + "\n"
            if target == "smi"
            else molecule.write(fmt=target, write_single=True)
        )
    except Exception as error:
        raise ExportError(
            f"cannot convert structure {group['id']!r} from {source!r} to {target!r}: {error}"
        ) from error
    if not isinstance(content, str) or not content.strip():
        raise ExportError(f"conversion of structure {group['id']!r} returned empty output")
    return content, source, True


def render_structure_record(
    data: Any,
    fmt: str,
    *,
    structure_index: Optional[int] = None,
    structure_role: Optional[str] = None,
    structure_id: Optional[str] = None,
) -> dict[str, Any]:
    """Render one selected structure with explicit selection provenance."""

    target = normalize_format(fmt)
    group = _select_group(
        data,
        structure_index=structure_index,
        structure_role=structure_role,
        structure_id=structure_id,
    )
    content, source, converted = _convert(data, group, target)
    mode = next(
        (
            name
            for name, value in (
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
        "structure_id": group["id"],
        "structure_roles": sorted(group["roles"]),
        "selection": {
            "mode": mode,
            "structure_index": structure_index,
            "structure_role": structure_role,
            "structure_id": structure_id,
        },
        "format": target,
        "source_format": source,
        "converted": converted,
        "content": content,
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def export_records(
    dataset: Any,
    output: Path,
    *,
    fmt: Optional[str],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Export one current Dataset selection into a new directory atomically."""

    records = tuple(dataset)
    if not records:
        raise ExportError("the export selection is empty")
    destination = Path(output).expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"export target already exists: {destination}")
    target = normalize_format(fmt) if fmt is not None else None
    rows = [(data, group) for data in records for group in _groups(data)]
    if not rows:
        raise ExportError("the export selection contains no structures")
    single = len(records) == 1 and len(rows) == 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=str(destination.parent))
    )
    files = []
    try:
        for data, group in rows:
            source_row = _source_row(group, target)
            output_format = target or normalize_format(source_row["format"])
            content, source_format, converted = _convert(data, group, output_format)
            if single:
                stem = ""
                structure_name = f"struct.{output_format}"
                info_name = "info.json"
            else:
                public_index = int(data._value["public_index"])
                group_index = _groups(data).index(group)
                stem = f"{public_index:06d}"
                if len(_groups(data)) > 1:
                    stem += f"-{group_index:02d}"
                structure_name = f"struct-{stem}.{output_format}"
                info_name = f"info-{stem}.json"
            (temporary / structure_name).write_text(content, encoding="utf-8")
            file_info = {
                "data_id": data.id,
                "record_id": data.record_id,
                "structure_id": group["id"],
                "structure_file": structure_name,
                "info_file": info_name,
                "source_format": source_format,
                "output_format": output_format,
            }
            info = data.info
            info["export"] = {
                **file_info,
                "roles": sorted(group["roles"]),
                "converted": converted,
            }
            _write_json(temporary / info_name, info)
            files.append(file_info)
        summary = {
            "dataset": dataset.info,
            "selection": dict(selection),
            "record_count": len(records),
            "structure_count": len(rows),
            "requested_format": target,
            "files": files,
        }
        if not single:
            _write_json(temporary / "summary.json", summary)
        temporary.rename(destination)
        return summary
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


__all__: list[str] = []
