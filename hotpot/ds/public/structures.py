"""Verified access to target-owned structure representations."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from .dataset import Data


class StructureAccessError(RuntimeError):
    pass


class AmbiguousStructureError(StructureAccessError):
    pass


class AssetIntegrityError(StructureAccessError):
    pass


class StructureUnavailableError(StructureAccessError):
    pass


class CrystalStructureUnavailableError(StructureUnavailableError):
    pass


_ALIASES = {"smiles": "smi"}
_LOAD_PRIORITY = ("mol2", "mol", "sdf", "smi", "xyz", "pdb", "cif")


class Structures:
    """Available formats and content for a Data item's selected structure."""

    def __init__(self, data: "Data") -> None:
        self._data = data

    def _rows(self) -> tuple[Dict[str, Any], ...]:
        return tuple(self._data._value.get("structures", []))

    def __call__(self) -> list[str]:
        return sorted(
            {
                _ALIASES.get(str(row["format"]).lower(), str(row["format"]).lower())
                for row in self._rows()
            }
        )

    def _selected_structure_id(self) -> Optional[str]:
        rows = self._rows()
        structure_ids = {str(row["structure_id"]) for row in rows}
        if not structure_ids:
            return None
        if len(structure_ids) == 1:
            return next(iter(structure_ids))
        primary_ids = {
            str(row["structure_id"])
            for row in rows
            if bool(row.get("primary"))
        }
        if len(primary_ids) != 1:
            raise AmbiguousStructureError(
                f"data {self._data.id} has multiple structures without exactly "
                "one primary structure"
            )
        return next(iter(primary_ids))

    def _row(self, fmt: str) -> Optional[Dict[str, Any]]:
        normalized = _ALIASES.get(fmt.lower().lstrip("."), fmt.lower().lstrip("."))
        selected = self._selected_structure_id()
        if selected is None:
            return None
        matches = [
            row
            for row in self._rows()
            if str(row["structure_id"]) == selected
            if _ALIASES.get(str(row["format"]).lower(), str(row["format"]).lower())
            == normalized
        ]
        if not matches:
            return None
        return matches[0]

    def _verified_path(self, row: Dict[str, Any]) -> Path:
        path = Path(row["path"])
        digest_value = str(row["sha256"])
        expected = (
            self._data.dataset._store.asset_root
            / "objects"
            / "sha256"
            / digest_value[:2]
            / digest_value
        )
        if path.absolute() != expected.absolute():
            raise AssetIntegrityError(
                f"structure is not stored at its target-owned content address: {path}"
            )
        if not path.is_file():
            raise AssetIntegrityError(f"target-owned structure is missing: {path}")
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
                size += len(block)
        if digest.hexdigest() != digest_value or size != row["size"]:
            raise AssetIntegrityError(f"target-owned structure fails integrity: {path}")
        return path

    def content(self, fmt: str) -> Optional[str]:
        row = self._row(fmt)
        if row is None:
            return None
        return self._verified_path(row).read_text(encoding="utf-8")

    def path(self, fmt: str) -> Optional[Path]:
        row = self._row(fmt)
        return None if row is None else self._verified_path(row)

    def molecule(self):
        from hotpot import Molecule

        for fmt in _LOAD_PRIORITY:
            row = self._row(fmt)
            if row is None:
                continue
            try:
                return Molecule.read_one(str(self._verified_path(row)), fmt)
            except Exception as error:
                raise StructureUnavailableError(
                    f"data {self._data.id} {fmt} representation cannot be loaded"
                ) from error
        raise StructureUnavailableError(
            f"data {self._data.id} has no supported molecular structure"
        )

    def __getattr__(self, name: str) -> Optional[str]:
        if name.startswith("_"):
            raise AttributeError(name)
        return self.content(name)

    def __repr__(self) -> str:
        return f"Structures(formats={self()!r})"


__all__ = [
    "AmbiguousStructureError",
    "AssetIntegrityError",
    "CrystalStructureUnavailableError",
    "StructureAccessError",
    "StructureUnavailableError",
    "Structures",
]
