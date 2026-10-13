"""Build the curated oxidation or spin dataset from canonical source tables.

This adapter reads the current repository tables directly.  It deliberately
does not accept any historical benchmark bundle format: Git identifies the
implementation, while the source tables and their digests identify the
scientific release.
"""

from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple, Union

from .._paths import DATA_ROOT
from ..levels import EvidenceLevel, QualityLevel
from ..models import (
    STANDARD_VERSION,
    Asset,
    ComputedDescriptor,
    Dataset,
    DatasetBundle,
    DatasetRecord,
    DatasetRelease,
    DatasetSubset,
    Entity,
    EvidenceItem,
    ExternalIdentifier,
    Observation,
    ObservationEvidence,
    ObservationMembership,
    RecordEntity,
    Source,
    Structure,
    StructureAsset,
    SubsetMembership,
)
from ..properties import builtin_property_types


_TABLES = {
    "oxidation": (
        ("SSE17-oxidation", "gold/sse17/labels.csv", "record_id"),
        ("SCO95-oxidation", "gold/sco95/labels.csv", "record_id"),
        ("TM-oxidation", "silver/tm_oxidation/labels.csv", "refcode"),
    ),
    "spin": (
        ("SSE17", "gold/sse17/labels.csv", "record_id"),
        ("SCO-95", "gold/sco95/labels.csv", "pair_id"),
        ("TM-GSspin", "silver/tm_gsspin/labels.csv", "refcode"),
    ),
}


def _rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: object) -> str:
    text = str(value).replace(" ", "_").replace("+", "plus")
    return "".join(character for character in text if character.isalnum() or character in "._-")


def _key(prefix: str, value: str) -> str:
    return f"{prefix}:{hashlib.sha256(value.encode()).hexdigest()[:24]}"


def _integer(value: object) -> int:
    return int(str(value))


def _real(value: object) -> float:
    return float(str(value))


def _source_doi(labels: Mapping[str, object]) -> Optional[str]:
    value = labels.get("curation_doi") or labels.get("data_doi")
    return str(value) if value else None


def _evidence_level(grade: str) -> EvidenceLevel:
    if grade.startswith("gold_A"):
        return EvidenceLevel.E3
    if grade.startswith("gold_B"):
        return EvidenceLevel.E2
    return EvidenceLevel.E1


def _property_types(dataset_name: str):
    selected = (
        {"oxidation_state"}
        if dataset_name == "oxidation"
        else {
            "spin_multiplicity",
            "spin_energy_gap",
            "spin_transition_temperature",
        }
    )
    return tuple(
        definition
        for definition in builtin_property_types()
        if definition.property_type_id in selected
    )


def _observation_specs(dataset_name: str, labels: Mapping[str, object]):
    if dataset_name == "oxidation":
        yield "oxidation_state", _integer(labels["metal_oxidation_state"]), None, None
        return
    if labels.get("ground_spin_multiplicity") not in (None, ""):
        yield (
            "spin_multiplicity",
            _integer(labels["ground_spin_multiplicity"]),
            None,
            None,
        )
    if labels.get("ground_multiplicity") not in (None, ""):
        yield "spin_multiplicity", _integer(labels["ground_multiplicity"]), None, None
    if labels.get("low_temperature_multiplicity") not in (None, ""):
        yield (
            "spin_multiplicity",
            _integer(labels["low_temperature_multiplicity"]),
            None,
            "low_temperature",
        )
    if labels.get("high_temperature_multiplicity") not in (None, ""):
        yield (
            "spin_multiplicity",
            _integer(labels["high_temperature_multiplicity"]),
            None,
            "high_temperature",
        )
    if labels.get("reference_delta_e_kcal_mol") not in (None, ""):
        yield (
            "spin_energy_gap",
            _real(labels["reference_delta_e_kcal_mol"]),
            "kcal/mol",
            None,
        )
    if labels.get("experimental_transition_temperature_k") not in (None, ""):
        yield (
            "spin_transition_temperature",
            _real(labels["experimental_transition_temperature_k"]),
            "K",
            None,
        )


def _subset_quality(dataset_name: str, subset: str) -> Dict[str, object]:
    return {
        "provenance_complete": True,
        "subject_mapping_rule_audited": True,
        "provenance_record": {
            "kind": "canonical_source_table",
            "dataset": dataset_name,
            "subset": subset,
        },
        "subject_mapping_report": {
            "kind": "content_addressed_structure_links",
            "link_table": "structures/record_structure_links.csv",
        },
        "task_definition": {
            "target": f"curated {dataset_name} observations",
            "grouping": "shared structure entities must remain in one evaluation split",
        },
    }


def _classification_rows(data_root: Path, dataset_name: str) -> Dict[str, Dict[str, str]]:
    path = data_root / "sources" / dataset_name / "validation" / "structure_composition.csv"
    return {row["structure_id"]: row for row in _rows(path)}


def _input_tables(data_root: Path, dataset_name: str):
    source_root = data_root / "sources" / dataset_name
    tables = []
    for subset, relative, id_column in _TABLES[dataset_name]:
        path = source_root / relative
        tables.append((subset, id_column, path, _rows(path)))
    return tables


def build_curated_dataset(
    dataset_name: str,
    *,
    data_root: Union[str, Path] = DATA_ROOT,
    version: str = "001",
    license_name: str = "mixed-upstream",
    access_policy: str = "mixed",
) -> DatasetBundle:
    """Build one current oxidation or spin bundle from canonical local data."""

    if dataset_name not in _TABLES:
        raise ValueError(f"dataset_name must be one of {sorted(_TABLES)}")

    root = Path(data_root).resolve()
    input_tables = _input_tables(root, dataset_name)
    structure_rows = {
        row["structure_id"]: row
        for row in _rows(root / "structures" / "structures.csv")
    }
    links = [
        row
        for row in _rows(root / "structures" / "record_structure_links.csv")
        if row["dataset"] == dataset_name
    ]
    links_by_record: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for link in links:
        links_by_record[(link["subset"], link["record_id"])].append(link)
    classifications = _classification_rows(root, dataset_name)

    release_id = f"{dataset_name}.v{_safe(version)}"
    subset_ids = {
        subset: f"{release_id}:subset:{_safe(subset)}"
        for subset, *_ in input_tables
    }
    source_digests = {
        path.relative_to(root).as_posix(): _sha256(path)
        for _, _, path, _ in input_tables
    }
    for relative in (
        "structures/structures.csv",
        "structures/record_structure_links.csv",
        f"sources/{dataset_name}/validation/structure_composition.csv",
    ):
        source_digests[relative] = _sha256(root / relative)

    datasets = (
        Dataset(
            dataset_name,
            f"Hotpot curated {dataset_name} dataset",
            f"Evidence-linked {dataset_name} observations with shared structure entities.",
        ),
    )
    releases = (
        DatasetRelease(
            release_id,
            dataset_name,
            _safe(version),
            QualityLevel.B0.value,
            license_name,
            access_policy,
            {
                "adapter": "curated-repository",
                "source_digests": source_digests,
                "asset_layout": "structures/objects/<sha-prefix>/<sha>.<format>",
            },
            {},
        ),
    )
    subsets = tuple(
        DatasetSubset(
            subset_ids[subset],
            release_id,
            None,
            subset,
            QualityLevel.B1.value,
            {"source_table": path.relative_to(root).as_posix()},
            _subset_quality(dataset_name, subset),
        )
        for subset, _, path, _ in input_tables
    )

    selected_structure_ids = {
        link["structure_id"]
        for link in links
    }
    structure_entities = {
        structure_id: f"entity:structure:{structure_id}"
        for structure_id in selected_structure_ids
    }
    entities: List[Entity] = [
        Entity(entity_id, "molecular_structure", None, {})
        for entity_id in structure_entities.values()
    ]
    assets: List[Asset] = []
    structures: List[Structure] = []
    structure_assets: List[StructureAsset] = []
    descriptors: List[ComputedDescriptor] = []
    for structure_id in sorted(selected_structure_ids):
        row = structure_rows[structure_id]
        uri = Path(row["store_path"]).as_posix()
        asset_id = f"asset:{row['canonical_file_sha256']}:{hashlib.sha256(uri.encode()).hexdigest()[:12]}"
        entity_id = structure_entities[structure_id]
        assets.append(
            Asset(
                asset_id,
                uri,
                {
                    "xyz": "chemical/x-xyz",
                    "mol2": "chemical/x-mol2",
                    "cif": "chemical/x-cif",
                }.get(row["format"], "application/octet-stream"),
                row["canonical_file_sha256"],
                "restricted-upstream",
                {"format": row["format"]},
            )
        )
        structures.append(
            Structure(
                structure_id,
                entity_id,
                row["coordinate_sha256"],
                int(row["atom_count"]),
                None,
                {},
            )
        )
        structure_assets.append(StructureAsset(structure_id, asset_id, "coordinates"))
        classification = classifications.get(structure_id)
        if classification:
            values = {
                "target_metal": classification["target_metal"],
                "target_metal_count": int(classification["target_metal_count"]),
                "all_metal_atom_count": int(classification["all_metal_atom_count"]),
                "metal_symbols": classification["metal_symbols"].split(";"),
                "inferred_donor_count": int(classification["inferred_donor_count"]),
                "ligand_count": int(classification["ligand_count"]),
                "organic_ligand_count": int(classification["organic_ligand_count"]),
            }
            for descriptor_type, value in values.items():
                descriptors.append(
                    ComputedDescriptor(
                        f"descriptor:{structure_id}:{descriptor_type}:composition-v1",
                        entity_id,
                        descriptor_type,
                        value,
                        None,
                        {"method": "hotpot-structure-composition", "version": "current"},
                        {},
                    )
                )

    records: List[DatasetRecord] = []
    memberships: List[SubsetMembership] = []
    record_entities: List[RecordEntity] = []
    external_identifiers = set()
    observations: List[Observation] = []
    observation_memberships: List[ObservationMembership] = []
    observation_evidence: List[ObservationEvidence] = []
    sources_by_id: Dict[str, Source] = {}
    evidence: List[EvidenceItem] = []

    for subset, id_column, _, label_rows in input_tables:
        for labels in label_rows:
            source_record_id = labels[id_column]
            record_id = f"{release_id}:record:{_safe(subset)}:{_safe(source_record_id)}"
            record_entity_id = f"entity:record:{record_id}"
            entities.append(Entity(record_entity_id, "chemical_record", None, {}))
            records.append(
                DatasetRecord(
                    record_id,
                    release_id,
                    f"{subset}/{source_record_id}",
                    {
                        "source_subset": subset,
                        "source_record_id": source_record_id,
                        "source_evidence_grade": labels["evidence_grade"],
                        "source_fields": dict(labels),
                    },
                )
            )
            memberships.append(SubsetMembership(subset_ids[subset], record_id))
            record_entities.append(RecordEntity(record_id, record_entity_id, "record_subject"))
            linked = links_by_record.get((subset, source_record_id), [])
            role_to_entity: Dict[str, str] = {}
            for link in linked:
                entity_id = structure_entities[link["structure_id"]]
                role_to_entity[link["structure_role"]] = entity_id
                record_entities.append(
                    RecordEntity(record_id, entity_id, link["structure_role"])
                )
                identifier_keys = (
                    f"{link['structure_role']}_refcode",
                    "source_refcode",
                    "structure_refcode",
                    "refcode",
                    "experimental_structure_id",
                )
                refcode = next(
                    (labels[key] for key in identifier_keys if labels.get(key)),
                    None,
                )
                if refcode:
                    external_identifiers.add(
                        (entity_id, "CSD_identifier", str(refcode), None)
                    )

            doi = _source_doi(labels)
            source_key = doi or f"{dataset_name}/{subset}"
            source_id = _key("source", source_key.lower())
            sources_by_id.setdefault(
                source_id,
                Source(
                    source_id,
                    "publication" if doi else "curated_dataset",
                    doi or f"Curated subset {dataset_name}/{subset}",
                    doi,
                    "citation-metadata-only",
                    "metadata-only",
                    {} if doi else {"dataset": dataset_name, "subset": subset},
                ),
            )
            evidence_id = _key("evidence", record_id)
            evidence.append(
                EvidenceItem(
                    evidence_id,
                    source_id,
                    "curated_annotation",
                    str(labels.get("evidence_locator") or f"record {source_record_id}"),
                    None,
                    str(
                        labels.get("oxidation_evidence")
                        or labels.get("spin_evidence")
                        or ""
                    )
                    or None,
                    {"source_evidence_grade": labels["evidence_grade"]},
                )
            )
            for index, (property_id, value, unit, role) in enumerate(
                _observation_specs(dataset_name, labels)
            ):
                subject = role_to_entity.get(role) if role else None
                if subject is None:
                    subject = next(iter(role_to_entity.values()), record_entity_id)
                observation_id = f"observation:{record_id}:{property_id}:{index}"
                uncertainty = {}
                conditions = {"structure_role": role} if role else {}
                if property_id == "spin_energy_gap":
                    if labels.get("uncertainty_kcal_mol") not in (None, ""):
                        uncertainty = {
                            "value": _real(labels["uncertainty_kcal_mol"]),
                            "unit": "kcal/mol",
                        }
                    conditions = {
                        "lower_spin_multiplicity": _integer(labels["lower_multiplicity"]),
                        "upper_spin_multiplicity": _integer(labels["upper_multiplicity"]),
                        "energy_definition": labels["energy_type"],
                    }
                elif property_id == "spin_transition_temperature":
                    conditions = {
                        "stimulus": "temperature",
                        "measurement_protocol": "reported_curated_dataset",
                        "branch": "not_reported",
                    }
                method = {"source_evidence_grade": labels["evidence_grade"]}
                if property_id == "spin_energy_gap":
                    method["sign_convention"] = (
                        "E_upper_multiplicity_minus_E_lower_multiplicity"
                    )
                observations.append(
                    Observation(
                        observation_id,
                        record_id,
                        subject,
                        property_id,
                        value,
                        unit,
                        uncertainty,
                        conditions,
                        "active",
                        _evidence_level(labels["evidence_grade"]).value,
                        method,
                        {},
                    )
                )
                observation_memberships.append(
                    ObservationMembership(subset_ids[subset], observation_id)
                )
                observation_evidence.append(
                    ObservationEvidence(observation_id, evidence_id, "supports")
                )

    return DatasetBundle(
        standard_version=STANDARD_VERSION,
        datasets=datasets,
        releases=releases,
        subsets=subsets,
        records=tuple(records),
        memberships=tuple(memberships),
        observation_memberships=tuple(observation_memberships),
        claim_memberships=(),
        entities=tuple(entities),
        external_identifiers=tuple(
            ExternalIdentifier(*row) for row in sorted(external_identifiers)
        ),
        record_entities=tuple(record_entities),
        entity_relations=(),
        assets=tuple(assets),
        structures=tuple(structures),
        structure_assets=tuple(structure_assets),
        descriptors=tuple(descriptors),
        property_types=_property_types(dataset_name),
        observations=tuple(observations),
        observation_participants=(),
        sources=tuple(sources_by_id.values()),
        evidence=tuple(evidence),
        observation_evidence=tuple(observation_evidence),
        claims=(),
        claim_evidence=(),
        claim_resolutions=(),
    )


__all__ = ["build_curated_dataset"]
