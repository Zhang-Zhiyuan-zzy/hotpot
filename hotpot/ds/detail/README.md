# Normalized backend detail

This reference is for normalized storage, audits, importers, and automated agents. Normal users should
start with the [compact API](../README.md). Objects in `hotpot.ds.detail` are intentionally
absent from the top-level public export list.

The backend implements the one schema declared by the current Git revision. Dataset
payloads belong under the ignored `hotpot/dataset/data/` root; backend code, schema
definitions, and documentation remain tracked.

## Backend groups

| Group | Tables / model records |
| --- | --- |
| Governance | `quality_certificates`, `ingestions`, `ingestion_assets`, `dataset_policies` |
| Releases | `datasets`, `dataset_releases`, `subsets`, `records`, membership junctions |
| Entities | `entities`, `external_identifiers`, `record_entities`, `entity_relations` |
| Structures | `structures`, `assets`, `structure_assets`, `computed_descriptors` |
| Properties | `property_types`, `observations`, participants and subset memberships |
| Evidence | `sources`, `evidence_items`, observation/claim links, resolutions |

The backend exposes strict immutable record classes, normalized bundle serialization, quality
assessment, property definitions, and low-level SQLite queries. It does not replace the
`Dataset`/`Data` facade.

## Backend imports

```python
from hotpot.ds import detail

bundle = detail.load_bundle("my-submission")
certificate = detail.assess_bundle(bundle, ".")
backend = detail.DatasetStore("work/datasets.sqlite3")
```

Advanced names are not resolvable from the `hotpot.ds` first-level namespace.

## Write boundary

The low-level backend accepts a fully normalized bundle and a backend quality certificate. The
public `ds.store.ingest` path adds submission fences, configuration binding,
existing-dataset policy checks, and policy persistence. Application code should therefore use
the public bundle/certificate/store flow unless it is implementing a trusted importer.

## Read-only catalog views

- `observation_catalog`: flattened dataset/release/record observation context;
- `structure_catalog`: structures expanded through record/entity links;
- `descriptor_catalog`: computed descriptors expanded through record/entity links;
- `claim_catalog`: unresolved claims with dataset/release context.

These views do not replace participant, subset, or evidence junctions.

## Identity and sharing rules

- `release_serial` and `public_index` are persisted, never query-time row numbers.
- Releases are immutable; a new version receives a new serial.
- Global objects use same-ID/same-content reuse.
- A coordinate hash is globally unique when present.
- Claims and observations are distinct records.
- Observation, claim, and record subset memberships are separate relations.
- Structure roles belong to `record_entities`; asset format roles belong to
  `structure_assets`.

## Advanced CLI

```bash
hotpot ds detail status --db work/datasets.sqlite3
hotpot ds detail audit --db work/datasets.sqlite3
hotpot ds detail query claims --db work/datasets.sqlite3
```
