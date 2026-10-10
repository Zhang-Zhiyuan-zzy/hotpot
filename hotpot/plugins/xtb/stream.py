"""Strict SDF records carrying reserved Hotpot xTB metadata."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from hotpot.cheminfo._io import MolReader
from hotpot.cheminfo.calculator.electronic_state import (
    ChargeInferenceSource,
    SpinInferenceSource,
)
from hotpot.cheminfo.core import Molecule

from .contracts import XTBMethod, XTBRunReport


__all__ = [
    "XTBStreamError",
    "XTBStreamMetadata",
    "XTBStreamProvenance",
    "XTBStreamRecord",
    "metadata_from_report",
    "read_sdf_records",
    "write_sdf_records",
]


class XTBStreamError(ValueError):
    """Raised when reserved xTB stream metadata is incomplete or inconsistent."""


@dataclass(frozen=True)
class XTBStreamProvenance:
    """Portable backend and state provenance retained in one SDF record."""

    backend_version: str
    backend_revision: Optional[str]
    executable_sha256: str
    charge_source: ChargeInferenceSource
    spin_source: Optional[SpinInferenceSource]


@dataclass(frozen=True)
class XTBStreamMetadata:
    """Validated xTB result metadata carried by one molecular record."""

    total_charge: int
    unpaired_electrons: Optional[int]
    method: XTBMethod
    energy_hartree: float
    provenance: XTBStreamProvenance


@dataclass(frozen=True)
class XTBStreamRecord:
    """A Hotpot molecule and optional complete xTB stream metadata."""

    mol: Molecule
    metadata: Optional[XTBStreamMetadata] = None


# Reserved SDF metadata schema.


_SCHEMA_VERSION = "1"
_TAG_PREFIX = "HOTPOT_XTB_"
_TAG_SCHEMA = f"{_TAG_PREFIX}SCHEMA"
_TAG_RECORD_SHA256 = f"{_TAG_PREFIX}RECORD_SHA256"
_TAG_TOTAL_CHARGE = f"{_TAG_PREFIX}TOTAL_CHARGE"
_TAG_UNPAIRED_ELECTRONS = f"{_TAG_PREFIX}UNPAIRED_ELECTRONS"
_TAG_METHOD = f"{_TAG_PREFIX}METHOD"
_TAG_ENERGY_HARTREE = f"{_TAG_PREFIX}ENERGY_HARTREE"
_TAG_PROVENANCE = f"{_TAG_PREFIX}PROVENANCE"
_REQUIRED_TAGS = frozenset(
    {
        _TAG_SCHEMA,
        _TAG_RECORD_SHA256,
        _TAG_TOTAL_CHARGE,
        _TAG_METHOD,
        _TAG_ENERGY_HARTREE,
        _TAG_PROVENANCE,
    }
)
_KNOWN_TAGS = _REQUIRED_TAGS | {_TAG_UNPAIRED_ELECTRONS}
_PROPERTY_HEADER = re.compile(r"^>\s*<([^>]+)>\s*$")
_INTEGER = re.compile(r"^[+-]?\d+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


# SDF record and metadata parsing helpers.


def _normalized_sdf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _split_sdf_records(text: str) -> Tuple[str, ...]:
    normalized = _normalized_sdf(text)
    if "$$$$" not in normalized:
        raise XTBStreamError("SDF input lacks a '$$$$' record terminator")

    parts = normalized.split("$$$$")
    if parts[-1].strip():
        raise XTBStreamError("SDF input contains content after its final terminator")

    records: List[str] = []
    for index, part in enumerate(parts[:-1]):
        record = part
        if index > 0 and record.startswith("\n"):
            record = record[1:]
        if not record.strip():
            raise XTBStreamError("SDF input contains an empty molecular record")
        records.append(record.rstrip("\n") + "\n")
    return tuple(records)


def _structure_block(record: str) -> Tuple[str, Tuple[str, ...]]:
    lines = record.splitlines()
    end_indices = tuple(
        index for index, line in enumerate(lines) if line.strip() == "M  END"
    )
    if len(end_indices) != 1:
        raise XTBStreamError("SDF record must contain exactly one 'M  END' line")
    end_index = end_indices[0]
    structure = "\n".join(lines[: end_index + 1]) + "\n"
    return structure, tuple(lines[end_index + 1 :])


def _record_sha256(structure: str) -> str:
    return hashlib.sha256(structure.encode("utf-8")).hexdigest()


def _reserved_properties(lines: Tuple[str, ...]) -> Dict[str, str]:
    properties: Dict[str, str] = {}
    index = 0
    while index < len(lines):
        line = lines[index]
        header = _PROPERTY_HEADER.match(line)
        if header is None:
            if _TAG_PREFIX in line:
                raise XTBStreamError("Malformed reserved xTB SDF property header")
            index += 1
            continue

        name = header.group(1)
        index += 1
        values: List[str] = []
        while index < len(lines) and lines[index] != "":
            if _PROPERTY_HEADER.match(lines[index]):
                break
            values.append(lines[index])
            index += 1

        if not name.startswith(_TAG_PREFIX):
            continue
        if name not in _KNOWN_TAGS:
            raise XTBStreamError(f"Unknown reserved xTB SDF property {name!r}")
        if name in properties:
            raise XTBStreamError(f"Duplicate reserved xTB SDF property {name!r}")
        if len(values) != 1 or not values[0].strip():
            raise XTBStreamError(
                f"Reserved xTB SDF property {name!r} must contain one value line"
            )
        properties[name] = values[0].strip()
    return properties


def _integer_value(value: str, name: str) -> int:
    if _INTEGER.fullmatch(value) is None:
        raise XTBStreamError(f"Reserved xTB property {name!r} is not an integer")
    return int(value)


def _finite_value(value: str, name: str) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise XTBStreamError(
            f"Reserved xTB property {name!r} is not numeric"
        ) from error
    if not math.isfinite(number):
        raise XTBStreamError(f"Reserved xTB property {name!r} is not finite")
    return number


def _unique_json_object(
    pairs: List[Tuple[str, object]],
) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise XTBStreamError(f"Duplicate provenance field {key!r}")
        result[key] = value
    return result


def _required_string(payload: Dict[str, object], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise XTBStreamError(f"xTB provenance field {field!r} must be a string")
    return value


def _optional_string(payload: Dict[str, object], field: str) -> Optional[str]:
    value = payload.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise XTBStreamError(
            f"xTB provenance field {field!r} must be null or a string"
        )
    return value


def _parse_provenance(value: str) -> XTBStreamProvenance:
    try:
        payload = json.loads(value, object_pairs_hook=_unique_json_object)
    except (json.JSONDecodeError, XTBStreamError) as error:
        raise XTBStreamError("Reserved xTB provenance is not valid JSON") from error
    if not isinstance(payload, dict):
        raise XTBStreamError("Reserved xTB provenance must be one JSON object")
    expected = {
        "backend_version",
        "backend_revision",
        "executable_sha256",
        "charge_source",
        "spin_source",
    }
    if set(payload) != expected:
        raise XTBStreamError("Reserved xTB provenance fields are incomplete")

    executable_sha256 = _required_string(payload, "executable_sha256")
    if _SHA256.fullmatch(executable_sha256) is None:
        raise XTBStreamError("xTB provenance executable SHA-256 is malformed")
    charge_source_text = _required_string(payload, "charge_source")
    spin_source_text = _optional_string(payload, "spin_source")
    try:
        charge_source = ChargeInferenceSource(charge_source_text)
        spin_source = (
            None
            if spin_source_text is None
            else SpinInferenceSource(spin_source_text)
        )
    except ValueError as error:
        raise XTBStreamError("xTB provenance contains an unknown state source") from error

    return XTBStreamProvenance(
        backend_version=_required_string(payload, "backend_version"),
        backend_revision=_optional_string(payload, "backend_revision"),
        executable_sha256=executable_sha256,
        charge_source=charge_source,
        spin_source=spin_source,
    )


def _validate_metadata(mol: Molecule, metadata: XTBStreamMetadata) -> None:
    if not math.isfinite(metadata.energy_hartree):
        raise XTBStreamError("xTB stream energy must be finite")
    if not metadata.provenance.backend_version:
        raise XTBStreamError("xTB provenance backend version is empty")
    if metadata.provenance.backend_revision == "":
        raise XTBStreamError("xTB provenance backend revision is empty")
    if _SHA256.fullmatch(metadata.provenance.executable_sha256) is None:
        raise XTBStreamError("xTB provenance executable SHA-256 is malformed")

    unpaired = metadata.unpaired_electrons
    if metadata.method is not XTBMethod.GFNFF and unpaired is None:
        raise XTBStreamError("GFN-xTB stream metadata requires unpaired electrons")
    if (unpaired is None) != (metadata.provenance.spin_source is None):
        raise XTBStreamError(
            "xTB spin provenance and unpaired-electron metadata are inconsistent"
        )
    electron_count = sum(atom.atomic_number for atom in mol.atoms) - metadata.total_charge
    if electron_count < 0:
        raise XTBStreamError("xTB total-charge metadata implies a negative electron count")
    if unpaired is None:
        return
    if unpaired < 0 or unpaired > electron_count:
        raise XTBStreamError("xTB unpaired-electron metadata is outside its domain")
    if unpaired % 2 != electron_count % 2:
        raise XTBStreamError(
            "xTB unpaired-electron metadata conflicts with molecular electron parity"
        )


def _metadata_from_properties(
    mol: Molecule,
    structure: str,
    properties: Dict[str, str],
) -> Optional[XTBStreamMetadata]:
    if not properties:
        return None
    missing = _REQUIRED_TAGS - properties.keys()
    if missing:
        raise XTBStreamError(
            f"Reserved xTB metadata lacks required properties {tuple(sorted(missing))!r}"
        )
    if properties[_TAG_SCHEMA] != _SCHEMA_VERSION:
        raise XTBStreamError("Unsupported reserved xTB stream schema")
    digest = properties[_TAG_RECORD_SHA256]
    if _SHA256.fullmatch(digest) is None or not hmac.compare_digest(
        digest,
        _record_sha256(structure),
    ):
        raise XTBStreamError("Reserved xTB metadata does not match its SDF record")

    total_charge = _integer_value(properties[_TAG_TOTAL_CHARGE], _TAG_TOTAL_CHARGE)
    unpaired = (
        None
        if _TAG_UNPAIRED_ELECTRONS not in properties
        else _integer_value(
            properties[_TAG_UNPAIRED_ELECTRONS],
            _TAG_UNPAIRED_ELECTRONS,
        )
    )
    try:
        method = XTBMethod(properties[_TAG_METHOD])
    except ValueError as error:
        raise XTBStreamError("Reserved xTB method is unknown") from error
    metadata = XTBStreamMetadata(
        total_charge=total_charge,
        unpaired_electrons=unpaired,
        method=method,
        energy_hartree=_finite_value(
            properties[_TAG_ENERGY_HARTREE],
            _TAG_ENERGY_HARTREE,
        ),
        provenance=_parse_provenance(properties[_TAG_PROVENANCE]),
    )
    _validate_metadata(mol, metadata)
    return metadata


def _provenance_value(provenance: XTBStreamProvenance) -> str:
    return json.dumps(
        {
            "backend_version": provenance.backend_version,
            "backend_revision": provenance.backend_revision,
            "executable_sha256": provenance.executable_sha256,
            "charge_source": provenance.charge_source.value,
            "spin_source": (
                None if provenance.spin_source is None else provenance.spin_source.value
            ),
        },
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _property(name: str, value: str) -> str:
    return f">  <{name}>\n{value}\n\n"


# Public record operations.


def metadata_from_report(
    report: XTBRunReport,
    previous: Optional[XTBStreamMetadata] = None,
) -> XTBStreamMetadata:
    """Create portable metadata, preserving prior spin across a GFN-FF node."""
    if report.energy_hartree is None or not math.isfinite(report.energy_hartree):
        raise XTBStreamError("A streamable xTB report requires finite energy")
    if report.charge_source is None:
        raise XTBStreamError("A streamable xTB report requires charge provenance")
    if (
        report.unpaired_electrons is None
        and previous is not None
        and previous.total_charge != report.charge
    ):
        raise XTBStreamError("GFN-FF cannot preserve spin across a charge change")

    unpaired = report.unpaired_electrons
    spin_source = report.spin_source
    if unpaired is None and previous is not None:
        unpaired = previous.unpaired_electrons
        spin_source = previous.provenance.spin_source
    return XTBStreamMetadata(
        total_charge=report.charge,
        unpaired_electrons=unpaired,
        method=report.effective_method,
        energy_hartree=report.energy_hartree,
        provenance=XTBStreamProvenance(
            backend_version=report.backend_info.version,
            backend_revision=report.backend_info.revision,
            executable_sha256=report.backend_info.executable_sha256,
            charge_source=report.charge_source,
            spin_source=spin_source,
        ),
    )


def read_sdf_records(text: str) -> Tuple[XTBStreamRecord, ...]:
    """Read SDF records and accept only complete, consistent reserved metadata."""
    records: List[XTBStreamRecord] = []
    for raw_record in _split_sdf_records(text):
        structure, property_lines = _structure_block(raw_record)
        try:
            mol = next(MolReader(raw_record + "$$$$\n", fmt="sdf"))
        except (OSError, RuntimeError, StopIteration, ValueError) as error:
            raise XTBStreamError("Cannot parse an SDF molecular record") from error
        metadata = _metadata_from_properties(
            mol,
            structure,
            _reserved_properties(property_lines),
        )
        records.append(XTBStreamRecord(mol=mol, metadata=metadata))
    return tuple(records)


def write_sdf_records(records: Iterable[XTBStreamRecord]) -> str:
    """Serialize molecular records with a narrow, versioned xTB property set."""
    serialized: List[str] = []
    for record in records:
        base = record.mol.write(fmt="sdf", write_single=True)
        structure, property_lines = _structure_block(
            _split_sdf_records(base)[0]
        )
        if _reserved_properties(property_lines):
            raise XTBStreamError("Molecule writer emitted reserved xTB metadata")
        if record.metadata is None:
            serialized.append(base.rstrip("\n") + "\n")
            continue

        _validate_metadata(record.mol, record.metadata)
        values = [
            (_TAG_SCHEMA, _SCHEMA_VERSION),
            (_TAG_RECORD_SHA256, _record_sha256(structure)),
            (_TAG_TOTAL_CHARGE, str(record.metadata.total_charge)),
        ]
        if record.metadata.unpaired_electrons is not None:
            values.append(
                (
                    _TAG_UNPAIRED_ELECTRONS,
                    str(record.metadata.unpaired_electrons),
                )
            )
        values.extend(
            (
                (_TAG_METHOD, record.metadata.method.value),
                (_TAG_ENERGY_HARTREE, f"{record.metadata.energy_hartree:.17g}"),
                (_TAG_PROVENANCE, _provenance_value(record.metadata.provenance)),
            )
        )
        properties = "".join(_property(name, value) for name, value in values)
        serialized.append(f"{structure}{properties}$$$$\n")
    return "".join(serialized)
