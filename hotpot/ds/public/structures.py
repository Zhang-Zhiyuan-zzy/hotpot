"""User-facing access to structure assets stored by :mod:`hotpot.ds`."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

if TYPE_CHECKING:  # pragma: no cover - imports are only for static analysis
    from .dataset import Data


class StructureAccessError(RuntimeError):
    """Base error raised while resolving a governed structure asset."""


class AmbiguousStructureError(StructureAccessError):
    """A record has several structures but does not identify one as primary."""


class AssetIntegrityError(StructureAccessError):
    """A structure asset is missing, unsafe, or fails its stored hash check."""


class StructureUnavailableError(StructureAccessError):
    """A record has no molecular structure that Hotpot can load."""


class CrystalStructureUnavailableError(StructureUnavailableError):
    """A record does not contain loadable crystal/unit-cell information."""


_MEDIA_FORMATS = {
    "chemical/x-xyz": "xyz",
    "chemical/x-mol2": "mol2",
    "chemical/x-mdl-molfile": "mol",
    "chemical/x-mdl-sdfile": "sdf",
    "chemical/x-daylight-smiles": "smi",
    "chemical/smiles": "smi",
    "chemical/x-cif": "cif",
    "chemical/x-pdb": "pdb",
}
_FORMAT_ALIASES = {"smiles": "smi"}
_PRIMARY_ROLES = frozenset(
    {"primary", "primary_structure", "canonical", "canonical_structure", "structure"}
)


def _asset_format(media_type: str, uri: str, metadata: Dict[str, object]) -> str:
    declared = metadata.get("format")
    if isinstance(declared, str) and declared:
        return _FORMAT_ALIASES.get(declared.lower().lstrip("."), declared.lower().lstrip("."))
    if media_type in _MEDIA_FORMATS:
        return _MEDIA_FORMATS[media_type]
    suffixes = Path(urlparse(uri).path).suffixes
    if suffixes and suffixes[-1].lower() == ".gz" and len(suffixes) > 1:
        return suffixes[-2].lower().lstrip(".")
    return suffixes[-1].lower().lstrip(".") if suffixes else ""


@dataclass(frozen=True)
class _Asset:
    asset_id: str
    uri: str
    media_type: str
    content_sha256: Optional[str]
    metadata: Dict[str, object]
    roots: Tuple[str, ...]

    @property
    def format(self) -> str:
        return _asset_format(self.media_type, self.uri, self.metadata)

    def path(self) -> Path:
        if urlparse(self.uri).scheme:
            raise AssetIntegrityError(
                f"remote asset {self.asset_id!r} is not locally content-addressable"
            )
        if not self.content_sha256:
            raise AssetIntegrityError(f"asset {self.asset_id!r} has no SHA-256 digest")

        failures: List[str] = []
        for root_text in self.roots:
            root = Path(root_text).resolve()
            candidate = (root / self.uri).resolve()
            try:
                candidate.relative_to(root)
            except ValueError:
                failures.append(f"path escapes asset root {root}")
                continue
            if not candidate.is_file():
                failures.append(f"missing at {candidate}")
                continue
            digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
            if digest != self.content_sha256:
                failures.append(f"SHA-256 mismatch at {candidate}")
                continue
            return candidate

        detail = "; ".join(failures) if failures else "no registered asset root"
        raise AssetIntegrityError(f"cannot verify asset {self.asset_id!r}: {detail}")

    def text(self) -> str:
        return self.path().read_text(encoding="utf-8")


@dataclass(frozen=True)
class _Structure:
    structure_id: str
    entity_id: str
    coordinate_hash: Optional[str]
    atom_count: Optional[int]
    formal_charge: Optional[int]
    metadata: Dict[str, object]
    roles: Tuple[str, ...]
    assets: Tuple[_Asset, ...]

    @property
    def formats(self) -> Tuple[str, ...]:
        return tuple(sorted({asset.format for asset in self.assets if asset.format}))

    def asset(self, fmt: str) -> Optional[_Asset]:
        normalized = _FORMAT_ALIASES.get(fmt.lower().lstrip("."), fmt.lower().lstrip("."))
        matches = [asset for asset in self.assets if asset.format == normalized]
        if not matches:
            return None
        coordinates = [asset for asset in matches if asset.metadata.get("role") == "coordinates"]
        return (coordinates or matches)[0]


class Structures:
    """Formats and verified content of the structure attached to one data item.

    Calling an instance lists all stored formats. Attribute access such as
    ``data.structures.mol2`` reads the primary structure in that format. Missing
    formats return ``None``. If a record has several structures, retrieval is
    deliberately refused unless exactly one of them has a primary/canonical role.
    """

    def __init__(self, data: "Data") -> None:
        self._data = data

    def _rows(self) -> Tuple[_Structure, ...]:
        connection = self._data.dataset._store._connect()
        try:
            raw_rows = connection.execute(
                """SELECT s.structure_id,s.entity_id,s.coordinate_hash,s.atom_count,
                s.formal_charge,s.metadata_json,re.role AS record_role,
                a.asset_id,a.uri,a.media_type,a.content_sha256,
                a.metadata_json AS asset_metadata_json,sa.role AS asset_role,
                i.ingestion_id,i.asset_root,
                CASE WHEN i.ingestion_id=dr.ingestion_id THEN 0 ELSE 1 END AS root_priority
                FROM record_entities re
                JOIN structures s ON s.entity_id=re.entity_id
                JOIN records rec ON rec.record_id=re.record_id
                JOIN dataset_releases dr ON dr.release_id=rec.release_id
                LEFT JOIN structure_assets sa ON sa.structure_id=s.structure_id
                LEFT JOIN assets a ON a.asset_id=sa.asset_id
                LEFT JOIN ingestion_assets ia ON ia.asset_id=a.asset_id
                LEFT JOIN ingestions i ON i.ingestion_id=ia.ingestion_id
                WHERE re.record_id=?
                ORDER BY s.structure_id,a.asset_id,root_priority,i.ingestion_id""",
                (self._data.record_id,),
            ).fetchall()
        finally:
            connection.close()

        grouped: Dict[str, Dict[str, object]] = {}
        for row in raw_rows:
            entry = grouped.setdefault(
                row["structure_id"],
                {
                    "entity_id": row["entity_id"],
                    "coordinate_hash": row["coordinate_hash"],
                    "atom_count": row["atom_count"],
                    "formal_charge": row["formal_charge"],
                    "metadata": json.loads(row["metadata_json"]),
                    "roles": set(),
                    "assets": {},
                },
            )
            entry["roles"].add(row["record_role"])
            if row["asset_id"] is None:
                continue
            assets = entry["assets"]
            asset = assets.setdefault(
                row["asset_id"],
                {
                    "uri": row["uri"],
                    "media_type": row["media_type"],
                    "content_sha256": row["content_sha256"],
                    "metadata": {
                        **json.loads(row["asset_metadata_json"]),
                        "role": row["asset_role"],
                    },
                    "roots": [],
                },
            )
            if row["asset_root"] is not None and row["asset_root"] not in asset["roots"]:
                asset["roots"].append(row["asset_root"])

        structures = []
        for structure_id, entry in grouped.items():
            assets = tuple(
                _Asset(
                    asset_id,
                    asset["uri"],
                    asset["media_type"],
                    asset["content_sha256"],
                    asset["metadata"],
                    tuple(asset["roots"]),
                )
                for asset_id, asset in entry["assets"].items()
            )
            structures.append(
                _Structure(
                    structure_id,
                    entry["entity_id"],
                    entry["coordinate_hash"],
                    entry["atom_count"],
                    entry["formal_charge"],
                    entry["metadata"],
                    tuple(sorted(entry["roles"])),
                    assets,
                )
            )
        return tuple(structures)

    @staticmethod
    def _choose(structures: Tuple[_Structure, ...], data_id: str) -> Optional[_Structure]:
        if not structures:
            return None
        if len(structures) == 1:
            return structures[0]
        primary = [
            structure
            for structure in structures
            if _PRIMARY_ROLES.intersection(structure.roles)
            or any(role.startswith("primary.") for role in structure.roles)
        ]
        if len(primary) == 1:
            return primary[0]
        ids = ", ".join(structure.structure_id for structure in structures)
        raise AmbiguousStructureError(
            f"data {data_id!r} has multiple structures without exactly one primary role: {ids}"
        )

    def _primary(self) -> Optional[_Structure]:
        return self._choose(self._rows(), self._data.id)

    def __call__(self) -> List[str]:
        return sorted({fmt for structure in self._rows() for fmt in structure.formats})

    def __iter__(self) -> Iterable[str]:
        return iter(self())

    def __len__(self) -> int:
        return len(self())

    def __repr__(self) -> str:
        return f"Structures({self()!r})"

    @property
    def info(self) -> List[Dict[str, object]]:
        return [
            {
                "structure_id": structure.structure_id,
                "entity_id": structure.entity_id,
                "roles": list(structure.roles),
                "formats": list(structure.formats),
                "coordinate_hash": structure.coordinate_hash,
                "atom_count": structure.atom_count,
                "formal_charge": structure.formal_charge,
                "metadata": structure.metadata,
                "assets": [
                    {
                        "asset_id": asset.asset_id,
                        "uri": asset.uri,
                        "media_type": asset.media_type,
                        "content_sha256": asset.content_sha256,
                        "format": asset.format,
                    }
                    for asset in structure.assets
                ],
            }
            for structure in self._rows()
        ]

    def get(self, fmt: str) -> Optional[str]:
        normalized = _FORMAT_ALIASES.get(fmt.lower().lstrip("."), fmt.lower().lstrip("."))
        structures = self._rows()
        if not any(structure.asset(normalized) is not None for structure in structures):
            return None
        structure = self._choose(structures, self._data.id)
        asset = structure.asset(normalized)
        return None if asset is None else asset.text()

    def load_molecule(self):
        structure = self._primary()
        if structure is None:
            raise StructureUnavailableError(f"data {self._data.id!r} has no structure")
        for fmt in ("mol2", "sdf", "mol", "smi", "cif", "pdb", "xyz"):
            asset = structure.asset(fmt)
            if asset is None:
                continue
            from hotpot.cheminfo.core import Molecule

            molecule = Molecule.read_one(asset.text(), fmt=fmt)
            if hasattr(molecule, "data"):
                molecule.ds_data = self._data
            else:
                molecule.data = self._data
            return molecule
        raise StructureUnavailableError(
            f"data {self._data.id!r} has no Hotpot-readable molecular format"
        )

    def load_crystal(self):
        structure = self._primary()
        if structure is None:
            raise CrystalStructureUnavailableError(
                f"data {self._data.id!r} has no crystal structure"
            )
        for fmt in ("cif", "mol2", "pdb", "sdf", "mol", "xyz"):
            asset = structure.asset(fmt)
            if asset is None:
                continue
            from hotpot.cheminfo.core import Molecule

            molecule = Molecule.read_one(asset.text(), fmt=fmt)
            crystal = molecule.crystal
            if crystal is None:
                continue
            if hasattr(molecule, "data"):
                molecule.ds_data = self._data
            else:
                molecule.data = self._data
            if hasattr(crystal, "data"):
                crystal.ds_data = self._data
            else:
                crystal.data = self._data
            crystal.mol = molecule
            return crystal
        raise CrystalStructureUnavailableError(
            f"data {self._data.id!r} has no unit-cell information"
        )

    @property
    def smi(self) -> Optional[str]:
        return self.get("smi")

    @property
    def smiles(self) -> Optional[str]:
        return self.get("smi")

    @property
    def mol2(self) -> Optional[str]:
        return self.get("mol2")

    @property
    def xyz(self) -> Optional[str]:
        return self.get("xyz")

    @property
    def cif(self) -> Optional[str]:
        return self.get("cif")

    def __getattr__(self, name: str) -> Optional[str]:
        if name.startswith("_"):
            raise AttributeError(name)
        return self.get(name)


__all__ = [
    "AmbiguousStructureError",
    "AssetIntegrityError",
    "CrystalStructureUnavailableError",
    "StructureAccessError",
    "StructureUnavailableError",
    "Structures",
]
