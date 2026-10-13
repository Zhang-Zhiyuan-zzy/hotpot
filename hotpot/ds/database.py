"""Generic relational dataset store for structures, properties, and evidence."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Union

from .models import STANDARD_VERSION, DatasetBundle, canonical_json
from .quality import QualityCertificate, verify_admission


SCHEMA_VERSION = 1

SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS schema_metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS quality_certificates(
 certificate_id TEXT PRIMARY KEY,certificate_sha256 TEXT UNIQUE NOT NULL,
 profile TEXT NOT NULL,subject_digest TEXT NOT NULL,issued_at TEXT NOT NULL,
 certificate_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS ingestions(
 ingestion_id INTEGER PRIMARY KEY AUTOINCREMENT,certificate_id TEXT UNIQUE NOT NULL
 REFERENCES quality_certificates(certificate_id),asset_root TEXT NOT NULL,
 stored_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);

CREATE TABLE IF NOT EXISTS datasets(
 dataset_id TEXT PRIMARY KEY,title TEXT NOT NULL,description TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS dataset_policies(
 dataset_id TEXT PRIMARY KEY REFERENCES datasets(dataset_id),revision INTEGER NOT NULL,
 config_json TEXT NOT NULL,config_sha256 TEXT NOT NULL,
 updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,CHECK(revision>0));
CREATE TABLE IF NOT EXISTS dataset_releases(
 release_id TEXT PRIMARY KEY,dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id),
 version TEXT NOT NULL,release_serial INTEGER NOT NULL,quality_level TEXT NOT NULL,
 core_labels_json TEXT NOT NULL,license TEXT NOT NULL,
 access_policy TEXT NOT NULL,provenance_json TEXT NOT NULL,quality_json TEXT NOT NULL,
 ingestion_id INTEGER NOT NULL REFERENCES ingestions(ingestion_id),
 UNIQUE(dataset_id,version),UNIQUE(dataset_id,release_serial),
 CHECK(release_serial BETWEEN 1 AND 999));
CREATE TABLE IF NOT EXISTS subsets(
 subset_id TEXT PRIMARY KEY,release_id TEXT NOT NULL REFERENCES dataset_releases(release_id),
 parent_subset_id TEXT REFERENCES subsets(subset_id),name TEXT NOT NULL,
 quality_level TEXT NOT NULL,selection_json TEXT NOT NULL,quality_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS records(
 record_id TEXT PRIMARY KEY,release_id TEXT NOT NULL REFERENCES dataset_releases(release_id),
 public_index INTEGER NOT NULL,local_id TEXT NOT NULL,metadata_json TEXT NOT NULL,
 UNIQUE(release_id,local_id),UNIQUE(release_id,public_index),CHECK(public_index>=0));
CREATE TABLE IF NOT EXISTS subset_memberships(
 subset_id TEXT NOT NULL REFERENCES subsets(subset_id),
 record_id TEXT NOT NULL REFERENCES records(record_id),PRIMARY KEY(subset_id,record_id));

CREATE TABLE IF NOT EXISTS entities(
 entity_id TEXT PRIMARY KEY,entity_kind TEXT NOT NULL,
 parent_entity_id TEXT REFERENCES entities(entity_id),metadata_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS external_identifiers(
 entity_id TEXT NOT NULL REFERENCES entities(entity_id),namespace TEXT NOT NULL,
 value TEXT NOT NULL,version TEXT,PRIMARY KEY(entity_id,namespace,value));
CREATE TABLE IF NOT EXISTS record_entities(
 record_id TEXT NOT NULL REFERENCES records(record_id),entity_id TEXT NOT NULL
 REFERENCES entities(entity_id),role TEXT NOT NULL,PRIMARY KEY(record_id,entity_id,role));
CREATE TABLE IF NOT EXISTS entity_relations(
 subject_entity_id TEXT NOT NULL REFERENCES entities(entity_id),predicate TEXT NOT NULL,
 object_entity_id TEXT NOT NULL REFERENCES entities(entity_id),metadata_json TEXT NOT NULL,
 PRIMARY KEY(subject_entity_id,predicate,object_entity_id));

CREATE TABLE IF NOT EXISTS assets(
 asset_id TEXT PRIMARY KEY,uri TEXT NOT NULL,media_type TEXT NOT NULL,
 content_sha256 TEXT,access_policy TEXT NOT NULL,metadata_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS structures(
 structure_id TEXT PRIMARY KEY,entity_id TEXT NOT NULL REFERENCES entities(entity_id),
 coordinate_hash TEXT,atom_count INTEGER,formal_charge INTEGER,metadata_json TEXT NOT NULL,
 CHECK(atom_count IS NULL OR atom_count>0));
CREATE TABLE IF NOT EXISTS structure_assets(
 structure_id TEXT NOT NULL REFERENCES structures(structure_id),
 asset_id TEXT NOT NULL REFERENCES assets(asset_id),role TEXT NOT NULL,
 PRIMARY KEY(structure_id,asset_id,role));
CREATE TABLE IF NOT EXISTS ingestion_assets(
 ingestion_id INTEGER NOT NULL REFERENCES ingestions(ingestion_id),
 asset_id TEXT NOT NULL REFERENCES assets(asset_id),
 PRIMARY KEY(ingestion_id,asset_id));
CREATE TABLE IF NOT EXISTS computed_descriptors(
 descriptor_id TEXT PRIMARY KEY,entity_id TEXT NOT NULL REFERENCES entities(entity_id),
 descriptor_type TEXT NOT NULL,value_json TEXT NOT NULL,unit TEXT,
 method_json TEXT NOT NULL,metadata_json TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS property_types(
 property_type_id TEXT PRIMARY KEY,name TEXT NOT NULL,value_kind TEXT NOT NULL,
 canonical_unit TEXT,schema_version TEXT NOT NULL,metadata_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS observations(
 observation_id TEXT PRIMARY KEY,record_id TEXT NOT NULL REFERENCES records(record_id),
 subject_entity_id TEXT NOT NULL REFERENCES entities(entity_id),property_type_id TEXT NOT NULL
 REFERENCES property_types(property_type_id),value_json TEXT NOT NULL,unit TEXT,
 uncertainty_json TEXT NOT NULL,conditions_json TEXT NOT NULL,status TEXT NOT NULL,
 evidence_level TEXT NOT NULL,method_json TEXT NOT NULL,metadata_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS observation_subset_memberships(
 subset_id TEXT NOT NULL REFERENCES subsets(subset_id),
 observation_id TEXT NOT NULL REFERENCES observations(observation_id),
 PRIMARY KEY(subset_id,observation_id));
CREATE TABLE IF NOT EXISTS observation_participants(
 observation_id TEXT NOT NULL REFERENCES observations(observation_id),
 entity_id TEXT NOT NULL REFERENCES entities(entity_id),role TEXT NOT NULL,
 stoichiometry REAL,metadata_json TEXT NOT NULL,
 PRIMARY KEY(observation_id,entity_id,role));

CREATE TABLE IF NOT EXISTS sources(
 source_id TEXT PRIMARY KEY,source_type TEXT NOT NULL,citation TEXT NOT NULL,doi TEXT,
 license TEXT NOT NULL,access_policy TEXT NOT NULL,metadata_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS evidence_items(
 evidence_id TEXT PRIMARY KEY,source_id TEXT NOT NULL REFERENCES sources(source_id),
 evidence_type TEXT NOT NULL,locator TEXT NOT NULL,quote TEXT,matched_text TEXT,
 metadata_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS observation_evidence(
 observation_id TEXT NOT NULL REFERENCES observations(observation_id),
 evidence_id TEXT NOT NULL REFERENCES evidence_items(evidence_id),role TEXT NOT NULL,
 PRIMARY KEY(observation_id,evidence_id,role));
CREATE TABLE IF NOT EXISTS claims(
 claim_id TEXT PRIMARY KEY,record_id TEXT NOT NULL REFERENCES records(record_id),
 subject_entity_id TEXT REFERENCES entities(entity_id),
 property_type_id TEXT REFERENCES property_types(property_type_id),
 text TEXT NOT NULL,status TEXT NOT NULL,evidence_level TEXT NOT NULL,
 metadata_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS claim_subset_memberships(
 subset_id TEXT NOT NULL REFERENCES subsets(subset_id),
 claim_id TEXT NOT NULL REFERENCES claims(claim_id),
 PRIMARY KEY(subset_id,claim_id));
CREATE TABLE IF NOT EXISTS claim_evidence(
 claim_id TEXT NOT NULL REFERENCES claims(claim_id),
 evidence_id TEXT NOT NULL REFERENCES evidence_items(evidence_id),role TEXT NOT NULL,
 PRIMARY KEY(claim_id,evidence_id,role));
CREATE TABLE IF NOT EXISTS claim_resolutions(
 claim_id TEXT NOT NULL REFERENCES claims(claim_id),
 observation_id TEXT NOT NULL REFERENCES observations(observation_id),
 resolution_type TEXT NOT NULL,metadata_json TEXT NOT NULL,
 PRIMARY KEY(claim_id,observation_id,resolution_type));

CREATE INDEX IF NOT EXISTS ix_release_dataset ON dataset_releases(dataset_id);
CREATE UNIQUE INDEX IF NOT EXISTS ux_release_public_serial
ON dataset_releases(dataset_id,release_serial);
CREATE INDEX IF NOT EXISTS ix_subset_release ON subsets(release_id);
CREATE INDEX IF NOT EXISTS ix_record_release ON records(release_id);
CREATE UNIQUE INDEX IF NOT EXISTS ux_record_public_index
ON records(release_id,public_index);
CREATE INDEX IF NOT EXISTS ix_entity_kind ON entities(entity_kind);
CREATE UNIQUE INDEX IF NOT EXISTS ux_structure_coordinate_hash
ON structures(coordinate_hash) WHERE coordinate_hash IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_observation_property ON observations(property_type_id);
CREATE INDEX IF NOT EXISTS ix_observation_subject ON observations(subject_entity_id);
CREATE INDEX IF NOT EXISTS ix_claim_property ON claims(property_type_id);
CREATE INDEX IF NOT EXISTS ix_descriptor_type ON computed_descriptors(descriptor_type);
CREATE INDEX IF NOT EXISTS ix_external_identifier ON external_identifiers(namespace,value);

DROP VIEW IF EXISTS observation_catalog;
CREATE VIEW observation_catalog AS
SELECT d.dataset_id,d.title,r.release_id,r.version,r.quality_level AS release_quality_level,
 rec.record_id,rec.local_id,o.observation_id,o.subject_entity_id,e.entity_kind,
 o.property_type_id,p.name AS property_name,p.value_kind,o.value_json,o.unit,
 o.uncertainty_json,o.conditions_json,o.status,o.evidence_level,o.method_json,o.metadata_json
FROM observations o JOIN records rec USING(record_id)
JOIN dataset_releases r USING(release_id) JOIN datasets d USING(dataset_id)
JOIN entities e ON e.entity_id=o.subject_entity_id
JOIN property_types p USING(property_type_id);

DROP VIEW IF EXISTS structure_catalog;
CREATE VIEW structure_catalog AS
SELECT d.dataset_id,r.release_id,rec.record_id,
 GROUP_CONCAT(DISTINCT re.role) AS record_roles,
 s.structure_id,s.entity_id,s.coordinate_hash,s.atom_count,s.formal_charge,
 s.metadata_json AS structure_metadata_json
FROM structures s JOIN record_entities re ON re.entity_id=s.entity_id
JOIN records rec USING(record_id) JOIN dataset_releases r USING(release_id)
JOIN datasets d USING(dataset_id)
GROUP BY d.dataset_id,r.release_id,rec.record_id,s.structure_id,s.entity_id,
 s.coordinate_hash,s.atom_count,s.formal_charge,s.metadata_json;

DROP VIEW IF EXISTS descriptor_catalog;
CREATE VIEW descriptor_catalog AS
SELECT d.dataset_id,r.release_id,rec.record_id,
 GROUP_CONCAT(DISTINCT re.role) AS record_roles,
 cd.descriptor_id,cd.entity_id,cd.descriptor_type,cd.value_json,cd.unit,
 cd.method_json,cd.metadata_json
FROM computed_descriptors cd JOIN record_entities re USING(entity_id)
JOIN records rec USING(record_id) JOIN dataset_releases r USING(release_id)
JOIN datasets d USING(dataset_id)
GROUP BY d.dataset_id,r.release_id,rec.record_id,cd.descriptor_id,cd.entity_id,
 cd.descriptor_type,cd.value_json,cd.unit,cd.method_json,cd.metadata_json;

DROP VIEW IF EXISTS claim_catalog;
CREATE VIEW claim_catalog AS
SELECT d.dataset_id,r.release_id,rec.record_id,rec.local_id,c.claim_id,
 c.subject_entity_id,c.property_type_id,c.text,c.status,c.evidence_level,c.metadata_json
FROM claims c JOIN records rec USING(record_id)
JOIN dataset_releases r USING(release_id) JOIN datasets d USING(dataset_id);
"""


@dataclass(frozen=True)
class IngestResult:
    ingestion_id: int
    certificate_id: str
    releases: int
    records: int
    structures: int
    observations: int
    claims: int
    descriptors: int


class DatasetSchemaError(ValueError):
    """The on-disk schema does not match the current implementation."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parent_first(rows: Iterable[Any], identity: str, parent: str) -> List[Any]:
    by_id = {getattr(row, identity): row for row in rows}
    ordered: List[Any] = []
    complete = set()
    visiting = set()

    def visit(row: Any) -> None:
        row_id = getattr(row, identity)
        if row_id in complete:
            return
        if row_id in visiting:
            raise ValueError(f"parent cycle contains {row_id}")
        visiting.add(row_id)
        parent_id = getattr(row, parent)
        if parent_id in by_id:
            visit(by_id[parent_id])
        visiting.remove(row_id)
        complete.add(row_id)
        ordered.append(row)

    for candidate in by_id.values():
        visit(candidate)
    return ordered


def _insert_immutable_rows(
    connection: sqlite3.Connection,
    table: str,
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
) -> None:
    """Insert globally shared rows, rejecting identifier-content collisions."""
    if not rows:
        return
    existing: Dict[Any, tuple] = {}
    identifiers = [row[0] for row in rows]
    for offset in range(0, len(identifiers), 500):
        chunk = identifiers[offset : offset + 500]
        placeholders = ",".join("?" for _ in chunk)
        for row in connection.execute(
            f"SELECT {','.join(columns)} FROM {table} "
            f"WHERE {columns[0]} IN ({placeholders})",
            chunk,
        ):
            existing[row[0]] = tuple(row)
    new_rows = []
    for row in rows:
        normalized = tuple(row)
        stored = existing.get(normalized[0])
        if stored is not None and stored != normalized:
            raise ValueError(
                f"immutable {table} identifier collision: {normalized[0]}"
            )
        if stored is None:
            new_rows.append(normalized)
    if new_rows:
        connection.executemany(
            f"INSERT INTO {table}({','.join(columns)}) VALUES({','.join('?' for _ in columns)})",
            new_rows,
        )


def _insert_dataset_policy(
    connection: sqlite3.Connection,
    dataset_id: str,
    config: Mapping[str, Any],
) -> None:
    payload = canonical_json(dict(config))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    existing = connection.execute(
        "SELECT config_sha256 FROM dataset_policies WHERE dataset_id=?",
        (dataset_id,),
    ).fetchone()
    if existing is not None:
        if existing["config_sha256"] != digest:
            raise ValueError(
                f"dataset policy for {dataset_id!r} differs from the established policy"
            )
        return
    connection.execute(
        """INSERT INTO dataset_policies(
        dataset_id,revision,config_json,config_sha256) VALUES(?,?,?,?)""",
        (dataset_id, 1, payload, digest),
    )


class DatasetStore:
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path))
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            existing_tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name NOT LIKE 'sqlite_%'"
                )
            }
            if "schema_metadata" in existing_tables:
                metadata = dict(
                    connection.execute("SELECT key,value FROM schema_metadata")
                )
                expected = {
                    "schema_version": str(SCHEMA_VERSION),
                    "record_standard": STANDARD_VERSION,
                }
                if any(metadata.get(key) != value for key, value in expected.items()):
                    raise DatasetSchemaError(
                        "dataset store schema is incompatible; rebuild it with the current "
                        "Hotpot revision"
                    )
            elif existing_tables:
                raise DatasetSchemaError(
                    "database is not an initialized Hotpot dataset store"
                )
            connection.executescript(SCHEMA)
            connection.execute(
                "INSERT OR IGNORE INTO schema_metadata VALUES('schema_version',?)",
                (str(SCHEMA_VERSION),),
            )
            connection.execute(
                "INSERT OR IGNORE INTO schema_metadata VALUES('record_standard',?)",
                (STANDARD_VERSION,),
            )

    def get_dataset_policy(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Return the persisted certificate configuration for a dataset."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM dataset_policies WHERE dataset_id=?", (dataset_id,)
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["config"] = json.loads(result.pop("config_json"))
        return result

    def set_dataset_policy(
        self,
        dataset_id: str,
        config: Dict[str, Any],
        expected_revision: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Persist a policy, requiring an explicit revision to replace one.

        The certificate layer decides whether a proposed revision is at least as
        strict as the prior policy.  This storage method prevents accidental or
        concurrent replacement by requiring the caller to name the current
        revision whenever the canonical policy content changes.
        """
        payload = canonical_json(config)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT revision,config_sha256 FROM dataset_policies WHERE dataset_id=?",
                (dataset_id,),
            ).fetchone()
            if existing is None:
                if expected_revision not in (None, 0):
                    raise ValueError("dataset policy revision does not exist")
                connection.execute(
                    """INSERT INTO dataset_policies(
                    dataset_id,revision,config_json,config_sha256) VALUES(?,?,?,?)""",
                    (dataset_id, 1, payload, digest),
                )
            elif existing["config_sha256"] != digest:
                if expected_revision != existing["revision"]:
                    raise ValueError(
                        "changing a dataset policy requires its current revision"
                    )
                connection.execute(
                    """UPDATE dataset_policies SET revision=?,config_json=?,
                    config_sha256=?,updated_at=CURRENT_TIMESTAMP WHERE dataset_id=?""",
                    (existing["revision"] + 1, payload, digest, dataset_id),
                )
        policy = self.get_dataset_policy(dataset_id)
        if policy is None:  # pragma: no cover - protected by the transaction above
            raise RuntimeError("dataset policy was not stored")
        return policy

    def ingest(
        self,
        bundle: DatasetBundle,
        certificate: QualityCertificate,
        asset_root: Union[str, Path],
        *,
        dataset_policies: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> IngestResult:
        verify_admission(bundle, certificate, asset_root)
        self.initialize()
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT ingestion_id FROM ingestions WHERE certificate_id=?",
                (certificate.certificate_id,),
            ).fetchone()
            if existing is not None:
                return IngestResult(
                    int(existing[0]), certificate.certificate_id, len(bundle.releases),
                    len(bundle.records), len(bundle.structures), len(bundle.observations),
                    len(bundle.claims),len(bundle.descriptors),
                )
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "INSERT INTO quality_certificates VALUES(?,?,?,?,?,?)",
                (
                    certificate.certificate_id,certificate.certificate_sha256,
                    certificate.profile,certificate.subject_digest,certificate.issued_at,
                    canonical_json(certificate.to_dict()),
                ),
            )
            cursor = connection.execute(
                "INSERT INTO ingestions(certificate_id,asset_root) VALUES(?,?)",
                (certificate.certificate_id,str(Path(asset_root).resolve())),
            )
            ingestion_id = int(cursor.lastrowid)

            _insert_immutable_rows(
                connection,"datasets",("dataset_id","title","description"),
                [(r.dataset_id,r.title,r.description) for r in bundle.datasets],
            )
            for dataset_id, config in (dataset_policies or {}).items():
                _insert_dataset_policy(connection, dataset_id, config)
            next_release_serial = {
                dataset_id: int(maximum or 0)
                for dataset_id, maximum in connection.execute(
                    "SELECT dataset_id,MAX(release_serial) FROM dataset_releases "
                    "GROUP BY dataset_id"
                )
            }
            release_serials: Dict[str, int] = {}
            release_core_labels: Dict[str, List[str]] = {}
            for release in bundle.releases:
                serial = next_release_serial.get(release.dataset_id, 0) + 1
                if serial > 999:
                    raise ValueError(
                        f"dataset {release.dataset_id!r} exceeds the three-digit "
                        "public release serial limit"
                    )
                next_release_serial[release.dataset_id] = serial
                release_serials[release.release_id] = serial
                declared = release.quality.get("core_labels", [])
                release_core_labels[release.release_id] = list(declared)
            connection.executemany(
                """INSERT INTO dataset_releases(
                release_id,dataset_id,version,release_serial,quality_level,
                core_labels_json,license,access_policy,provenance_json,quality_json,
                ingestion_id) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                [(
                    r.release_id,r.dataset_id,r.version,release_serials[r.release_id],
                    r.quality_level,canonical_json(release_core_labels[r.release_id]),
                    r.license,r.access_policy,canonical_json(r.provenance),
                    canonical_json(r.quality),ingestion_id,
                ) for r in bundle.releases],
            )
            ordered_subsets = _parent_first(bundle.subsets,"subset_id","parent_subset_id")
            connection.executemany(
                "INSERT INTO subsets VALUES(?,?,?,?,?,?,?)",
                [(
                    r.subset_id,r.release_id,r.parent_subset_id,r.name,r.quality_level,
                    canonical_json(r.selection),canonical_json(r.quality),
                ) for r in ordered_subsets],
            )
            record_indices: Dict[str, int] = {}
            release_record_counts: Dict[str, int] = {}
            for record in bundle.records:
                public_index = release_record_counts.get(record.release_id, 0)
                release_record_counts[record.release_id] = public_index + 1
                record_indices[record.record_id] = public_index
            connection.executemany(
                """INSERT INTO records(record_id,release_id,public_index,local_id,metadata_json)
                VALUES(?,?,?,?,?)""",
                [(
                    r.record_id,r.release_id,record_indices[r.record_id],r.local_id,
                    canonical_json(r.metadata),
                ) for r in bundle.records],
            )
            connection.executemany(
                "INSERT INTO subset_memberships VALUES(?,?)",
                [(r.subset_id,r.record_id) for r in bundle.memberships],
            )
            ordered_entities = _parent_first(bundle.entities,"entity_id","parent_entity_id")
            _insert_immutable_rows(
                connection,"entities",("entity_id","entity_kind","parent_entity_id","metadata_json"),
                [(r.entity_id,r.entity_kind,r.parent_entity_id,canonical_json(r.metadata)) for r in ordered_entities],
            )
            connection.executemany(
                "INSERT OR IGNORE INTO external_identifiers VALUES(?,?,?,?)",
                [(r.entity_id,r.namespace,r.value,r.version) for r in bundle.external_identifiers],
            )
            connection.executemany(
                "INSERT INTO record_entities VALUES(?,?,?)",
                [(r.record_id,r.entity_id,r.role) for r in bundle.record_entities],
            )
            connection.executemany(
                "INSERT OR IGNORE INTO entity_relations VALUES(?,?,?,?)",
                [(r.subject_entity_id,r.predicate,r.object_entity_id,canonical_json(r.metadata)) for r in bundle.entity_relations],
            )
            _insert_immutable_rows(
                connection,"assets",("asset_id","uri","media_type","content_sha256","access_policy","metadata_json"),
                [(r.asset_id,r.uri,r.media_type,r.content_sha256,r.access_policy,canonical_json(r.metadata)) for r in bundle.assets],
            )
            connection.executemany(
                "INSERT INTO ingestion_assets VALUES(?,?)",
                [(ingestion_id,r.asset_id) for r in bundle.assets],
            )
            _insert_immutable_rows(
                connection,"structures",("structure_id","entity_id","coordinate_hash","atom_count","formal_charge","metadata_json"),
                [(r.structure_id,r.entity_id,r.coordinate_hash,r.atom_count,r.formal_charge,canonical_json(r.metadata)) for r in bundle.structures],
            )
            connection.executemany(
                "INSERT OR IGNORE INTO structure_assets VALUES(?,?,?)",
                [(r.structure_id,r.asset_id,r.role) for r in bundle.structure_assets],
            )
            _insert_immutable_rows(
                connection,"computed_descriptors",("descriptor_id","entity_id","descriptor_type","value_json","unit","method_json","metadata_json"),
                [(
                    r.descriptor_id,r.entity_id,r.descriptor_type,canonical_json(r.value),r.unit,
                    canonical_json(r.method),canonical_json(r.metadata),
                ) for r in bundle.descriptors],
            )
            _insert_immutable_rows(
                connection,"property_types",("property_type_id","name","value_kind","canonical_unit","schema_version","metadata_json"),
                [(r.property_type_id,r.name,r.value_kind,r.canonical_unit,r.schema_version,canonical_json(r.metadata)) for r in bundle.property_types],
            )
            connection.executemany(
                "INSERT INTO observations VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                [(
                    r.observation_id,r.record_id,r.subject_entity_id,r.property_type_id,
                    canonical_json(r.value),r.unit,canonical_json(r.uncertainty),
                    canonical_json(r.conditions),r.status,r.evidence_level,
                    canonical_json(r.method),canonical_json(r.metadata),
                ) for r in bundle.observations],
            )
            connection.executemany(
                "INSERT INTO observation_subset_memberships VALUES(?,?)",
                [(r.subset_id,r.observation_id) for r in bundle.observation_memberships],
            )
            connection.executemany(
                "INSERT INTO observation_participants VALUES(?,?,?,?,?)",
                [(
                    r.observation_id,r.entity_id,r.role,r.stoichiometry,
                    canonical_json(r.metadata),
                ) for r in bundle.observation_participants],
            )
            _insert_immutable_rows(
                connection,"sources",("source_id","source_type","citation","doi","license","access_policy","metadata_json"),
                [(r.source_id,r.source_type,r.citation,r.doi,r.license,r.access_policy,canonical_json(r.metadata)) for r in bundle.sources],
            )
            connection.executemany(
                "INSERT INTO evidence_items VALUES(?,?,?,?,?,?,?)",
                [(r.evidence_id,r.source_id,r.evidence_type,r.locator,r.quote,r.matched_text,canonical_json(r.metadata)) for r in bundle.evidence],
            )
            connection.executemany(
                "INSERT INTO observation_evidence VALUES(?,?,?)",
                [(r.observation_id,r.evidence_id,r.role) for r in bundle.observation_evidence],
            )
            connection.executemany(
                "INSERT INTO claims(claim_id,record_id,subject_entity_id,property_type_id,"
                "text,status,evidence_level,metadata_json) VALUES(?,?,?,?,?,?,?,?)",
                [(
                    r.claim_id,r.record_id,r.subject_entity_id,r.property_type_id,
                    r.text,r.status,r.evidence_level,canonical_json(r.metadata),
                ) for r in bundle.claims],
            )
            connection.executemany(
                "INSERT INTO claim_subset_memberships VALUES(?,?)",
                [(r.subset_id,r.claim_id) for r in bundle.claim_memberships],
            )
            connection.executemany(
                "INSERT INTO claim_evidence VALUES(?,?,?)",
                [(r.claim_id,r.evidence_id,r.role) for r in bundle.claim_evidence],
            )
            connection.executemany(
                "INSERT INTO claim_resolutions VALUES(?,?,?,?)",
                [(
                    r.claim_id,r.observation_id,r.resolution_type,canonical_json(r.metadata),
                ) for r in bundle.claim_resolutions],
            )

        return IngestResult(
            ingestion_id,certificate.certificate_id,len(bundle.releases),len(bundle.records),
            len(bundle.structures),len(bundle.observations),len(bundle.claims),len(bundle.descriptors),
        )

    def list_datasets(self) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            return [dict(row) for row in connection.execute(
                """SELECT d.*,r.release_id,r.version,r.quality_level,r.license,r.access_policy
                FROM datasets d LEFT JOIN dataset_releases r USING(dataset_id)
                ORDER BY d.dataset_id,r.version"""
            )]

    def get_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as connection:
            dataset = connection.execute("SELECT * FROM datasets WHERE dataset_id=?",(dataset_id,)).fetchone()
            if dataset is None:
                return None
            result = dict(dataset)
            result["releases"] = [dict(row) for row in connection.execute(
                "SELECT * FROM dataset_releases WHERE dataset_id=? ORDER BY version",(dataset_id,)
            )]
            for row in result["releases"]:
                row["provenance"] = json.loads(row.pop("provenance_json"))
                row["quality"] = json.loads(row.pop("quality_json"))
                row["core_labels"] = json.loads(row.pop("core_labels_json"))
                row["subsets"] = [dict(item) for item in connection.execute(
                    "SELECT * FROM subsets WHERE release_id=? ORDER BY subset_id",(row["release_id"],)
                )]
                for subset in row["subsets"]:
                    subset["selection"] = json.loads(subset.pop("selection_json"))
                    subset["quality"] = json.loads(subset.pop("quality_json"))
            return result

    def iter_records(
        self,
        dataset_id: Optional[str] = None,
        release_id: Optional[str] = None,
        subset_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Iterator[Dict[str, Any]]:
        clauses = []
        parameters: List[Any] = []
        for column, value in (("d.dataset_id", dataset_id), ("r.release_id", release_id)):
            if value is not None:
                clauses.append(f"{column}=?")
                parameters.append(value)
        if subset_id is not None:
            clauses.append(
                "EXISTS (SELECT 1 FROM subset_memberships sm "
                "WHERE sm.record_id=rec.record_id AND sm.subset_id=?)"
            )
            parameters.append(subset_id)
        query = """SELECT d.dataset_id,r.release_id,r.version,r.release_serial,
        rec.record_id,rec.public_index,rec.local_id,rec.metadata_json FROM records rec
        JOIN dataset_releases r USING(release_id) JOIN datasets d USING(dataset_id)"""
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY d.dataset_id,r.release_serial,rec.public_index"
        if limit is not None:
            query += " LIMIT ?"
            parameters.append(limit)
        with self._connect() as connection:
            for raw in connection.execute(query, parameters):
                row = dict(raw)
                row["metadata"] = json.loads(row.pop("metadata_json"))
                yield row

    def iter_entities(
        self,
        entity_kind: Optional[str] = None,
        identifier_namespace: Optional[str] = None,
        identifier_value: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Iterator[Dict[str, Any]]:
        clauses = []
        parameters: List[Any] = []
        if entity_kind is not None:
            clauses.append("e.entity_kind=?")
            parameters.append(entity_kind)
        if identifier_namespace is not None:
            clauses.append("x.namespace=?")
            parameters.append(identifier_namespace)
        if identifier_value is not None:
            clauses.append("x.value=?")
            parameters.append(identifier_value)
        query = """SELECT DISTINCT e.entity_id,e.entity_kind,e.parent_entity_id,
        e.metadata_json FROM entities e
        LEFT JOIN external_identifiers x USING(entity_id)"""
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY e.entity_id"
        if limit is not None:
            query += " LIMIT ?"
            parameters.append(limit)
        with self._connect() as connection:
            for raw in connection.execute(query, parameters):
                row = dict(raw)
                row["metadata"] = json.loads(row.pop("metadata_json"))
                yield row

    def iter_observations(
        self,dataset_id: Optional[str]=None,release_id: Optional[str]=None,
        subset_id: Optional[str]=None,property_type_id: Optional[str]=None,
        evidence_level: Optional[str]=None,status: Optional[str]=None,
        subject_entity_id: Optional[str]=None,limit: Optional[int]=None,
    ) -> Iterator[Dict[str, Any]]:
        clauses=[]; parameters: List[Any]=[]
        for column,value in (
            ("dataset_id",dataset_id),("release_id",release_id),
            ("property_type_id",property_type_id),("evidence_level",evidence_level),
            ("status",status),("subject_entity_id",subject_entity_id),
        ):
            if value is not None: clauses.append(f"{column}=?"); parameters.append(value)
        if subset_id is not None:
            clauses.append(
                "EXISTS (SELECT 1 FROM observation_subset_memberships sm "
                "WHERE sm.observation_id=observation_catalog.observation_id AND sm.subset_id=?)"
            )
            parameters.append(subset_id)
        query="SELECT * FROM observation_catalog"+(" WHERE "+" AND ".join(clauses) if clauses else "")+" ORDER BY dataset_id,record_id,observation_id"
        if limit is not None: query+=" LIMIT ?"; parameters.append(limit)
        with self._connect() as connection:
            for raw in connection.execute(query,parameters):
                row=dict(raw)
                for field in ("value_json","uncertainty_json","conditions_json","method_json","metadata_json"):
                    row[field.removesuffix("_json")]=json.loads(row.pop(field))
                yield row

    def get_observation(self, observation_id: str) -> Optional[Dict[str, Any]]:
        """Return one observation with participants, evidence, and subset links."""
        with self._connect() as connection:
            raw = connection.execute(
                "SELECT * FROM observation_catalog WHERE observation_id=?",
                (observation_id,),
            ).fetchone()
            if raw is None:
                return None
            result = dict(raw)
            for field in (
                "value_json", "uncertainty_json", "conditions_json",
                "method_json", "metadata_json",
            ):
                result[field.removesuffix("_json")] = json.loads(result.pop(field))
            result["participants"] = [
                dict(row)
                for row in connection.execute(
                    """SELECT op.entity_id,op.role,op.stoichiometry,op.metadata_json,
                    e.entity_kind FROM observation_participants op
                    JOIN entities e USING(entity_id) WHERE op.observation_id=?
                    ORDER BY op.role,op.entity_id""",
                    (observation_id,),
                )
            ]
            for participant in result["participants"]:
                participant["metadata"] = json.loads(participant.pop("metadata_json"))
            result["evidence"] = self._linked_evidence(
                connection, "observation_evidence", "observation_id", observation_id
            )
            result["subsets"] = [
                dict(row)
                for row in connection.execute(
                    """SELECT s.subset_id,s.name,s.quality_level
                    FROM observation_subset_memberships m JOIN subsets s USING(subset_id)
                    WHERE m.observation_id=? ORDER BY s.subset_id""",
                    (observation_id,),
                )
            ]
            result["claim_resolutions"] = list(
                self._iter_claim_resolutions(connection, observation_id=observation_id)
            )
            return result

    def iter_structures(self,dataset_id: Optional[str]=None,limit: Optional[int]=None) -> Iterator[Dict[str,Any]]:
        query="SELECT * FROM structure_catalog"; parameters: List[Any]=[]
        if dataset_id is not None: query+=" WHERE dataset_id=?";parameters.append(dataset_id)
        query+=" ORDER BY dataset_id,record_id,structure_id"
        if limit is not None:query+=" LIMIT ?";parameters.append(limit)
        with self._connect() as connection:
            for raw in connection.execute(query,parameters):
                row=dict(raw)
                row["metadata"]=json.loads(row.pop("structure_metadata_json"))
                yield row

    def iter_descriptors(
        self,dataset_id: Optional[str]=None,descriptor_type: Optional[str]=None,
        entity_id: Optional[str]=None,limit: Optional[int]=None,
    ) -> Iterator[Dict[str,Any]]:
        clauses=[];parameters: List[Any]=[]
        for column,value in (
            ("dataset_id",dataset_id),("descriptor_type",descriptor_type),("entity_id",entity_id),
        ):
            if value is not None:clauses.append(f"{column}=?");parameters.append(value)
        query="SELECT * FROM descriptor_catalog"+(" WHERE "+" AND ".join(clauses) if clauses else "")+" ORDER BY dataset_id,record_id,descriptor_id"
        if limit is not None:query+=" LIMIT ?";parameters.append(limit)
        with self._connect() as connection:
            for raw in connection.execute(query,parameters):
                row=dict(raw)
                for field in ("value_json","method_json","metadata_json"):
                    row[field.removesuffix("_json")]=json.loads(row.pop(field))
                yield row

    def get_structure(self, structure_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as connection:
            raw = connection.execute(
                "SELECT * FROM structures WHERE structure_id=?", (structure_id,)
            ).fetchone()
            if raw is None:
                return None
            result = dict(raw)
            result["metadata"] = json.loads(result.pop("metadata_json"))
            result["assets"] = [dict(row) for row in connection.execute(
                """SELECT a.*,sa.role FROM structure_assets sa
                JOIN assets a USING(asset_id) WHERE sa.structure_id=? ORDER BY a.asset_id""",
                (structure_id,),
            )]
            for asset in result["assets"]:
                asset["metadata"] = json.loads(asset.pop("metadata_json"))
            result["records"] = [dict(row) for row in connection.execute(
                """SELECT re.record_id,re.role,r.release_id FROM record_entities re
                JOIN records r USING(record_id) WHERE re.entity_id=? ORDER BY re.record_id""",
                (result["entity_id"],),
            )]
            result["descriptors"] = [dict(row) for row in connection.execute(
                """SELECT descriptor_id,descriptor_type,value_json,unit,method_json,metadata_json
                FROM computed_descriptors WHERE entity_id=? ORDER BY descriptor_type""",
                (result["entity_id"],),
            )]
            for descriptor in result["descriptors"]:
                for field in ("value_json","method_json","metadata_json"):
                    descriptor[field.removesuffix("_json")]=json.loads(descriptor.pop(field))
            return result

    @staticmethod
    def _linked_evidence(
        connection: sqlite3.Connection,
        relation_table: str,
        identity_column: str,
        identity: str,
    ) -> List[Dict[str, Any]]:
        rows = [
            dict(row)
            for row in connection.execute(
                f"""SELECT rel.role AS link_role,ev.*,s.source_type,s.citation,s.doi,
                s.license AS source_license,s.access_policy AS source_access_policy,
                s.metadata_json AS source_metadata_json
                FROM {relation_table} rel
                JOIN evidence_items ev USING(evidence_id)
                JOIN sources s USING(source_id)
                WHERE rel.{identity_column}=? ORDER BY ev.evidence_id""",
                (identity,),
            )
        ]
        for row in rows:
            row["metadata"] = json.loads(row.pop("metadata_json"))
            row["source_metadata"] = json.loads(row.pop("source_metadata_json"))
        return rows

    def iter_claims(
        self,dataset_id: Optional[str]=None,release_id: Optional[str]=None,
        subset_id: Optional[str]=None,
        property_type_id: Optional[str]=None,
        evidence_level: Optional[str]=None,status: Optional[str]=None,
        limit: Optional[int]=None,
    ) -> Iterator[Dict[str, Any]]:
        clauses=[]; parameters: List[Any]=[]
        for column,value in (
            ("dataset_id",dataset_id),("release_id",release_id),
            ("property_type_id",property_type_id),
            ("evidence_level",evidence_level),("status",status),
        ):
            if value is not None: clauses.append(f"{column}=?");parameters.append(value)
        if subset_id is not None:
            clauses.append(
                "EXISTS (SELECT 1 FROM claim_subset_memberships sm "
                "WHERE sm.claim_id=claim_catalog.claim_id AND sm.subset_id=?)"
            )
            parameters.append(subset_id)
        query="SELECT * FROM claim_catalog"+(" WHERE "+" AND ".join(clauses) if clauses else "")+" ORDER BY dataset_id,record_id,claim_id"
        if limit is not None:query+=" LIMIT ?";parameters.append(limit)
        with self._connect() as connection:
            for raw in connection.execute(query,parameters):
                row=dict(raw);row["metadata"]=json.loads(row.pop("metadata_json"));yield row

    def get_claim(self, claim_id: str) -> Optional[Dict[str, Any]]:
        """Return one unresolved/resolved claim with its full audit chain."""
        with self._connect() as connection:
            raw = connection.execute(
                "SELECT * FROM claim_catalog WHERE claim_id=?", (claim_id,)
            ).fetchone()
            if raw is None:
                return None
            result = dict(raw)
            result["metadata"] = json.loads(result.pop("metadata_json"))
            result["evidence"] = self._linked_evidence(
                connection, "claim_evidence", "claim_id", claim_id
            )
            result["subsets"] = [
                dict(row)
                for row in connection.execute(
                    """SELECT s.subset_id,s.name,s.quality_level
                    FROM claim_subset_memberships m JOIN subsets s USING(subset_id)
                    WHERE m.claim_id=? ORDER BY s.subset_id""",
                    (claim_id,),
                )
            ]
            result["resolutions"] = list(
                self._iter_claim_resolutions(connection, claim_id=claim_id)
            )
            return result

    @staticmethod
    def _iter_claim_resolutions(
        connection: sqlite3.Connection,
        claim_id: Optional[str] = None,
        observation_id: Optional[str] = None,
    ) -> Iterator[Dict[str, Any]]:
        clauses = []
        parameters: List[Any] = []
        if claim_id is not None:
            clauses.append("cr.claim_id=?")
            parameters.append(claim_id)
        if observation_id is not None:
            clauses.append("cr.observation_id=?")
            parameters.append(observation_id)
        query = """SELECT d.dataset_id,r.release_id,cr.claim_id,cr.observation_id,
        cr.resolution_type,cr.metadata_json FROM claim_resolutions cr
        JOIN claims c USING(claim_id) JOIN records rec ON rec.record_id=c.record_id
        JOIN dataset_releases r USING(release_id) JOIN datasets d USING(dataset_id)"""
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY d.dataset_id,r.release_id,cr.claim_id,cr.observation_id"
        for raw in connection.execute(query, parameters):
            row = dict(raw)
            row["metadata"] = json.loads(row.pop("metadata_json"))
            yield row

    def iter_claim_resolutions(
        self,
        claim_id: Optional[str] = None,
        observation_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Iterator[Dict[str, Any]]:
        with self._connect() as connection:
            rows = self._iter_claim_resolutions(
                connection, claim_id=claim_id, observation_id=observation_id
            )
            for index, row in enumerate(rows):
                if limit is not None and index >= limit:
                    break
                yield row

    def counts(self) -> Dict[str,int]:
        tables=("datasets","dataset_policies","dataset_releases","subsets","records","entities","assets","ingestion_assets","structures","computed_descriptors","property_types","observations","observation_subset_memberships","observation_participants","sources","evidence_items","claims","claim_subset_memberships","claim_resolutions","quality_certificates","ingestions")
        with self._connect() as connection:
            return {table:int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]) for table in tables}

    def integrity_issues(self) -> List[str]:
        with self._connect() as connection:
            issues=[]
            check=connection.execute("PRAGMA quick_check").fetchone()[0]
            if check!="ok":issues.append(f"SQLite quick_check: {check}")
            issues.extend(f"foreign key violation: {tuple(row)}" for row in connection.execute("PRAGMA foreign_key_check"))
            for row in connection.execute(
                """SELECT release_id FROM dataset_releases
                WHERE release_serial IS NULL OR release_serial NOT BETWEEN 1 AND 999"""
            ):
                issues.append(f"invalid public release serial: {row['release_id']}")
            for row in connection.execute(
                """SELECT record_id FROM records
                WHERE public_index IS NULL OR public_index<0"""
            ):
                issues.append(f"invalid public record index: {row['record_id']}")
            for row in connection.execute(
                "SELECT release_id,core_labels_json FROM dataset_releases"
            ):
                try:
                    core_labels = json.loads(row["core_labels_json"])
                    if not isinstance(core_labels, list) or not all(
                        isinstance(label, str) for label in core_labels
                    ):
                        raise ValueError("core labels must be a string array")
                    record_count = connection.execute(
                        "SELECT COUNT(*) FROM records WHERE release_id=?",
                        (row["release_id"],),
                    ).fetchone()[0]
                    for label in core_labels:
                        covered = connection.execute(
                            """SELECT COUNT(DISTINCT o.record_id)
                            FROM observations o JOIN records r USING(record_id)
                            WHERE r.release_id=? AND o.property_type_id=?
                            AND o.status='active'""",
                            (row["release_id"], label),
                        ).fetchone()[0]
                        if covered != record_count:
                            issues.append(
                                f"incomplete core label {label!r} in {row['release_id']}: "
                                f"{covered}/{record_count}"
                            )
                except (TypeError, ValueError, json.JSONDecodeError) as error:
                    issues.append(
                        f"invalid core labels for {row['release_id']}: {error}"
                    )
            for row in connection.execute(
                "SELECT dataset_id,config_json,config_sha256 FROM dataset_policies"
            ):
                try:
                    config = json.loads(row["config_json"])
                    actual = hashlib.sha256(
                        canonical_json(config).encode("utf-8")
                    ).hexdigest()
                    if actual != row["config_sha256"]:
                        issues.append(
                            f"dataset policy digest mismatch: {row['dataset_id']}"
                        )
                except (TypeError, ValueError, json.JSONDecodeError) as error:
                    issues.append(
                        f"invalid dataset policy {row['dataset_id']}: {error}"
                    )
            for row in connection.execute(
                "SELECT certificate_id,certificate_sha256,certificate_json FROM quality_certificates"
            ):
                try:
                    certificate=QualityCertificate.from_dict(json.loads(row["certificate_json"]))
                    if certificate.certificate_id != row["certificate_id"]:
                        issues.append(f"certificate identifier mismatch: {row['certificate_id']}")
                    if certificate.certificate_sha256 != row["certificate_sha256"]:
                        issues.append(f"certificate digest column mismatch: {row['certificate_id']}")
                    if certificate.profile != "admission" or not certificate.passed:
                        issues.append(f"non-admission certificate stored: {row['certificate_id']}")
                except (ValueError,TypeError,QualityGateError) as error:
                    issues.append(f"invalid certificate {row['certificate_id']}: {error}")
            for row in connection.execute(
                """SELECT ia.ingestion_id,i.asset_root,a.asset_id,a.uri,a.content_sha256
                FROM ingestion_assets ia JOIN ingestions i USING(ingestion_id)
                JOIN assets a USING(asset_id) ORDER BY ia.ingestion_id,a.asset_id"""
            ):
                if "://" in row["uri"]:
                    continue
                root = Path(row["asset_root"]).resolve()
                path = (root / row["uri"]).resolve()
                try:
                    path.relative_to(root)
                except ValueError:
                    issues.append(
                        f"asset path escapes root for ingestion {row['ingestion_id']}: "
                        f"{row['asset_id']}"
                    )
                    continue
                if not path.is_file():
                    issues.append(
                        f"asset missing for ingestion {row['ingestion_id']}: {row['uri']}"
                    )
                elif row["content_sha256"] is None:
                    issues.append(
                        f"asset lacks digest for ingestion {row['ingestion_id']}: "
                        f"{row['asset_id']}"
                    )
                elif _sha256_file(path) != row["content_sha256"]:
                    issues.append(
                        f"asset digest mismatch for ingestion {row['ingestion_id']}: "
                        f"{row['uri']}"
                    )
            return issues
