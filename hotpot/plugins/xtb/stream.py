"""Strict SDF records carrying reserved Hotpot xTB metadata."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from hotpot.cheminfo.operations.io import MolReader
from hotpot.cheminfo.structure._builder import MoleculeRecord
from hotpot.cheminfo.structure._cif_facts import _facts_from_data, _facts_to_data
from hotpot.cheminfo.structure.snapshots import MolecularFactsIdentity
from hotpot.cheminfo.rules.chemistry.scientific_state.composition import ScientificStateIdentity
from hotpot.cheminfo.operations.calculation.scientific_state.facts import extract_molecular_facts
from hotpot.cheminfo.structure.molecule import Molecule
from hotpot.cheminfo.rules.chemistry.bonding.hydrogen import _non_graph_hydrogens

from .contracts import XTBMethod, XTBRunReport
from .state_projection import _effective_nuclear_charge, _GFN_CORE_BOUNDARIES


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
    scientific_state_identity: Optional[ScientificStateIdentity] = None
    facts_identity: Optional[MolecularFactsIdentity] = None
    projection_convention: Optional[str] = None

    def __post_init__(self) -> None:
        binding = (
            self.scientific_state_identity,
            self.facts_identity,
            self.projection_convention,
        )
        if any(value is None for value in binding) and any(
            value is not None for value in binding
        ):
            raise ValueError(
                "xTB stream state provenance must be wholly present or absent"
            )


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


_SCHEMA_VERSION = "2"
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
_FACTS_TAG_PREFIX = "HOTPOT_FACTS_"
_TAG_FACTS = f"{_FACTS_TAG_PREFIX}CURRENT"
_FACTS_SCHEMA = "hotpot.sdf.facts/1"
_KNOWN_TAGS = _REQUIRED_TAGS | {_TAG_UNPAIRED_ELECTRONS, _TAG_FACTS}
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
            if _TAG_PREFIX in line or _FACTS_TAG_PREFIX in line:
                raise XTBStreamError("Malformed reserved Hotpot SDF property header")
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

        if not name.startswith((_TAG_PREFIX, _FACTS_TAG_PREFIX)):
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


def _factual_value(record: MoleculeRecord, structure: str) -> str:
    labels = tuple(f"A{row + 1}" for row in range(record.atom_count))
    return json.dumps(
        {
            "schema": _FACTS_SCHEMA,
            "structure_sha256": _record_sha256(structure),
            "facts": _facts_to_data(record, labels, bool(record.atom_count)),
        },
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _restore_factual_molecule(
    structure: str, properties: Dict[str, str],
) -> Optional[Molecule]:
    if _TAG_FACTS not in properties:
        return None
    try:
        payload = json.loads(properties[_TAG_FACTS], object_pairs_hook=_unique_json_object)
    except (ValueError, TypeError) as error:
        raise XTBStreamError("Reserved Hotpot facts are not valid JSON") from error
    if not isinstance(payload, dict) or set(payload) != {
        "schema", "structure_sha256", "facts",
    }:
        raise XTBStreamError("Reserved Hotpot facts fields are incomplete")
    if payload["schema"] != _FACTS_SCHEMA:
        raise XTBStreamError("Unsupported reserved Hotpot facts schema")
    digest = payload["structure_sha256"]
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None or not hmac.compare_digest(
        digest, _record_sha256(structure),
    ):
        raise XTBStreamError("Reserved Hotpot facts binding does not match its SDF record")
    try:
        record, _, _ = _facts_from_data(payload["facts"])
        return Molecule._build_molecule(record)
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, OverflowError) as error:
        raise XTBStreamError("Reserved Hotpot facts schema or record is malformed") from error


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
        "scientific_state_identity",
        "facts_identity",
        "projection_convention",
    }
    if set(payload) != expected:
        raise XTBStreamError("Reserved xTB provenance fields are incomplete")

    executable_sha256 = _required_string(payload, "executable_sha256")
    if _SHA256.fullmatch(executable_sha256) is None:
        raise XTBStreamError("xTB provenance executable SHA-256 is malformed")
    scientific_identity = _optional_string(
        payload,
        "scientific_state_identity",
    )
    facts_identity = _optional_string(payload, "facts_identity")
    projection_convention = _optional_string(payload, "projection_convention")

    try:
        return XTBStreamProvenance(
            backend_version=_required_string(payload, "backend_version"),
            backend_revision=_optional_string(payload, "backend_revision"),
            executable_sha256=executable_sha256,
            scientific_state_identity=(
                None
                if scientific_identity is None
                else ScientificStateIdentity(scientific_identity)
            ),
            facts_identity=(
                None
                if facts_identity is None
                else MolecularFactsIdentity(facts_identity)
            ),
            projection_convention=projection_convention,
        )
    except ValueError as error:
        raise XTBStreamError(str(error)) from error


def _validate_metadata(mol: Molecule, metadata: XTBStreamMetadata) -> None:
    if not math.isfinite(metadata.energy_hartree):
        raise XTBStreamError("xTB stream energy must be finite")
    if not metadata.provenance.backend_version:
        raise XTBStreamError("xTB provenance backend version is empty")
    if metadata.provenance.backend_revision == "":
        raise XTBStreamError("xTB provenance backend revision is empty")
    if _SHA256.fullmatch(metadata.provenance.executable_sha256) is None:
        raise XTBStreamError("xTB provenance executable SHA-256 is malformed")

    facts = extract_molecular_facts(mol)
    provenance = metadata.provenance
    if (
        provenance.facts_identity is not None
        and provenance.facts_identity != facts.identity
    ):
        raise XTBStreamError(
            "xTB state provenance does not match the SDF molecular facts"
        )

    unpaired = metadata.unpaired_electrons
    if metadata.method is not XTBMethod.GFNFF and unpaired is None:
        raise XTBStreamError("GFN-xTB stream metadata requires unpaired electrons")
    non_graph_hydrogens = (
        _non_graph_hydrogens(number, explicit_count, implicit_count)
        for number, explicit_count, implicit_count in zip(
            facts.atomic_numbers,
            facts.hydrogen_inventory.explicit_counts,
            facts.hydrogen_inventory.implicit_counts,
        )
    )
    spin_method = metadata.method
    if (
        spin_method is XTBMethod.GFNFF
        and provenance.projection_convention == "pure_spin_highest_ms"
        and unpaired is not None
    ):
        # Ordinary GFN0/1/2 share this convention; GFN-FF only carries its UHF.
        spin_method = XTBMethod.GFN2_XTB
    if spin_method is not XTBMethod.GFNFF:
        unsupported = tuple(
            number for number in facts.atomic_numbers
            if not 1 <= number <= _GFN_CORE_BOUNDARIES[-1][0]
        )
        if unsupported:
            raise XTBStreamError(
                f"xTB active-electron convention has unsupported atomic numbers {unsupported!r}"
            )
    electron_count = (
        sum(_effective_nuclear_charge(number, spin_method) for number in facts.atomic_numbers)
        + sum(count for count in non_graph_hydrogens if count is not None)
        - metadata.total_charge
    )
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
            "scientific_state_identity": (
                None
                if provenance.scientific_state_identity is None
                else provenance.scientific_state_identity.digest
            ),
            "facts_identity": (
                None
                if provenance.facts_identity is None
                else provenance.facts_identity.chemical_identity_digest
            ),
            "projection_convention": provenance.projection_convention,
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

    unpaired = report.unpaired_electrons
    scientific_identity = report.scientific_state_identity
    facts_identity = report.facts_identity
    projection_convention = report.projection_convention
    if (
        unpaired is None
        and previous is not None
        and previous.unpaired_electrons is not None
    ):
        previous_provenance = previous.provenance
        compatible = (
            scientific_identity is not None
            and facts_identity is not None
            and projection_convention == "charge_only"
            and previous.total_charge == report.charge
            and previous_provenance.scientific_state_identity
            == scientific_identity
            and previous_provenance.facts_identity == facts_identity
            and previous_provenance.projection_convention
            == "pure_spin_highest_ms"
        )
        if not compatible:
            raise XTBStreamError(
                "GFN-FF cannot preserve spin across incompatible state provenance"
            )
        unpaired = previous.unpaired_electrons
        projection_convention = previous_provenance.projection_convention
    return XTBStreamMetadata(
        total_charge=report.charge,
        unpaired_electrons=unpaired,
        method=report.effective_method,
        energy_hartree=report.energy_hartree,
        provenance=XTBStreamProvenance(
            backend_version=report.backend_info.version,
            backend_revision=report.backend_info.revision,
            executable_sha256=report.backend_info.executable_sha256,
            scientific_state_identity=scientific_identity,
            facts_identity=facts_identity,
            projection_convention=projection_convention,
        ),
    )


def read_sdf_records(text: str) -> Tuple[XTBStreamRecord, ...]:
    """Read SDF records and accept only complete, consistent reserved metadata."""
    records: List[XTBStreamRecord] = []
    for raw_record in _split_sdf_records(text):
        structure, property_lines = _structure_block(raw_record)
        properties = _reserved_properties(property_lines)
        mol = _restore_factual_molecule(structure, properties)
        if mol is None:
            try:
                mol = next(MolReader(raw_record + "$$$$\n", fmt="sdf"))
            except (OSError, RuntimeError, StopIteration, ValueError) as error:
                raise XTBStreamError("Cannot parse an SDF molecular record") from error
        metadata = _metadata_from_properties(
            mol,
            structure,
            {name: value for name, value in properties.items() if name.startswith(_TAG_PREFIX)},
        )
        records.append(XTBStreamRecord(mol=mol, metadata=metadata))
    return tuple(records)


def write_sdf_records(records: Iterable[XTBStreamRecord]) -> str:
    """Serialize molecular records with a narrow, versioned xTB property set."""
    serialized: List[str] = []
    for record in records:
        if record.metadata is not None:
            _validate_metadata(record.mol, record.metadata)
        factual_record, _ = record.mol._capture_factual_record(include_frames=False)
        base = Molecule._build_molecule(factual_record).write(fmt="sdf", write_single=True)
        structure, property_lines = _structure_block(
            _split_sdf_records(base)[0]
        )
        if _reserved_properties(property_lines):
            raise XTBStreamError("Molecule writer emitted reserved xTB metadata")
        facts_property = _property(_TAG_FACTS, _factual_value(factual_record, structure))
        if record.metadata is None:
            serialized.append(f"{structure}{facts_property}$$$$\n")
            continue

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
        serialized.append(f"{structure}{facts_property}{properties}$$$$\n")
    return "".join(serialized)
