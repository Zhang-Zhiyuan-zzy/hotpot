# `hotpot.ds` public API

`hotpot.ds` has one small public workflow. Its first-level API contains exactly these
names, in this order:

```python
[
    "datasets_list", "load",
    "Dataset", "Data",
    "BaseSearcher", "DataSearcher", "DatasetSearcher", "SmartsSearcher",
    "write_certificate", "load_bundle", "Bundle",
    "store",
]
```

Normalized tables, certificate implementation classes, and forensic
queries are intentionally absent from this namespace. Automation that needs those details
may use `hotpot.ds.detail`.

## 1. `datasets_list` and `load`

### `datasets_list`

`ds.datasets_list` is a printable, iterable sequence of names in the default database.
Call it with `db=` to inspect another database.

```python
import hotpot.ds as ds

print(ds.datasets_list)  # ['csd-electronic-state', 'oxidation', 'spin']
names = ds.datasets_list(db="work/datasets.sqlite3")
```

| Form | Result |
| --- | --- |
| `list(ds.datasets_list)` | Names in the default database |
| `ds.datasets_list(db=path)` | Names in the selected database |

The default repository store is resolved independently of the current working directory at
`hotpot/dataset/data/store/datasets.sqlite3`. The complete `hotpot/dataset/data/` payload
root is ignored by Git, so a clean clone or an installation without a local store must pass
`db=`. Licensed structures and generated databases are never copied into wheels.

### `load`

```python
dataset = ds.load(ds_name="oxidation")
older = ds.load(ds_name="oxidation", version="001")
local = ds.load(ds_name="my-dataset", db="work/datasets.sqlite3")
```

| Parameter | Meaning |
| --- | --- |
| `ds_name` | Dataset name from `datasets_list` |
| `version` | Optional source version, release identifier, integer serial, or three-digit serial |
| `db` | Optional Hotpot dataset SQLite path |

`load` returns a `Dataset`. `dataset.info` is the JSON-compatible description of the
selected immutable release.

Dataset release identifiers describe scientific data identity. They do not select an older
Hotpot implementation, schema, or compatibility mode; Git owns implementation history.

## 2. `Dataset` and `Data`

### `Dataset`

A `Dataset` is a sequence-like view of one release. `Dataset` and `Data` are returned by
the query API rather than constructed as submission inputs. Indexing returns `Data`; slicing
and filtering return another `Dataset` while preserving the original identities.

| Member | Result |
| --- | --- |
| `dataset[index]` | One `Data`; a slice returns `Dataset` |
| `dataset.info` | Dataset properties, version, size, license, labels, and provenance |
| `dataset.handle` | Sorted DOI or citation handles represented in the view |
| `dataset.labels` | `{label_name: data_count}` for active labels |
| `dataset.core_labels` | Labels required on every item; may be `[]` |
| `dataset.subset(**filters)` | AND-filtered `Dataset` |
| `dataset.<source>_identifier[value]` | Dataset-specific external-identifier lookup |

Collection-valued filters use IN semantics. Lists, sets, tuples, and frozen sets are
accepted. Nested paths use `__`.

```python
dataset = ds.load(ds_name="oxidation")

first = dataset[0]
selected = dataset.subset(
    labels__oxidation_state=[2, 3],
    quality_level={"B2_curated", "B3_validated"},
)

# Identifier handles are dataset-specific.
csd_data = dataset.CSD_identifier["ACAHUJ"]
```

If an identifier is unique, lookup returns `Data`; if it is repeated, lookup returns a
`Dataset` containing all matches.

### `Data`

| Member | Result |
| --- | --- |
| `data.id` | Stable Hotpot identity |
| `data.info` | Complete JSON-compatible public record |
| `data.label` | Active label dictionary |
| `data.quality_level` | B0-B4 quality string |
| `data.evidence_level` | Conservative E0-E4 evidence string |
| `data.structures()` | Sorted stored formats, for example `['smi', 'mol2']` |
| `data.structures.smi` | Verified SMILES text or `None` |
| `data.structures.mol2` | Verified MOL2 text or `None` |
| `data.molecule` | Fresh `hotpot.Molecule` from the primary structure |
| `data.crystal` | Crystal object when lattice data are available; otherwise raises |

The identity format is:

```text
hp-<dataset_name>[<three-digit-release-serial>]-<zero-based-index>
```

For example, the first item of the first `oxidation` release is
`hp-oxidation[001]-0`. Subsets never renumber it.

Structure format access returns `None` when the selected primary structure lacks that
format. With multiple scientific variants, a primary role must be declared; Hotpot never
selects a different variant merely because it has the requested representation. Local
assets remain inside their registered root and are rechecked against SHA-256 before use.

The originating `Data` is attached to a loaded molecule or crystal as `.data`. If that
name is already owned by the object, Hotpot uses `.ds_data`.

Quality and evidence are independent string vocabularies:

| Quality | Interpretation |
| --- | --- |
| `B0_collection` | Governed collection |
| `B1_candidate` | Provenance-complete candidate with audited mapping |
| `B2_curated` | Item-level review and conflict adjudication |
| `B3_validated` | Primary-source support, dual review, and frozen evaluation rules |
| `B4_reference` | Independently corroborated reference release |

| Evidence | Interpretation |
| --- | --- |
| `E0_insufficient` | Insufficient support |
| `E1_extracted` | Extracted but unconfirmed |
| `E2_explicit_reported` | Explicit source statement |
| `E3_primary_source_supported` | Auditable primary-source support |
| `E4_independently_corroborated` | Independent corroboration |

## 3. `BaseSearcher`, `DataSearcher`, `DatasetSearcher`, and `SmartsSearcher`

`BaseSearcher` is the abstract common interface. Every concrete searcher supports
`match(item)`, `search(target)`, and direct calling as a predicate.

```python
data_query = (
    ds.DataSearcher(labels__oxidation_state=[2, 3])
    & ~ds.DataSearcher(quality_level="B0_collection")
)
matching_data = data_query.search(dataset)

dataset_query = ds.DatasetSearcher(core_labels="oxidation_state")
matching_datasets = dataset_query.search(ds.datasets_list)

substructure_query = ds.SmartsSearcher("[N]-[C]")
matching_structures = substructure_query.search(dataset)
```

| Searcher | Target |
| --- | --- |
| `DataSearcher(**filters)` | Fields in `data.info` |
| `DatasetSearcher(**filters)` | Fields in `dataset.info` |
| `SmartsSearcher(smarts, semantics=None)` | Readable molecular structures |

| Common operation | Result |
| --- | --- |
| `searcher.match(item)` | `True` when one item matches |
| `searcher(item)` | Predicate shorthand for `match` |
| `searcher.search(target)` | Matching `Dataset` view or tuple |
| `searcher.not_()` | Negated search expression |

Expressions compose as follows:

- `a & b`: both match;
- `a | b`: either matches;
- `a ^ b`: exactly one matches;
- `~a` or `a.not_()`: logical NOT.

Python does not provide an overloadable unary `!` operator. Using Python `and`, `or`, or
`not` on search expressions raises an explanatory error.

For a non-default database, construct the dataset targets explicitly:

```python
db = "work/datasets.sqlite3"
targets = [ds.load(name, db=db) for name in ds.datasets_list(db=db)]
matching_datasets = dataset_query.search(targets)
```

## 4. `write_certificate`, `load_bundle`, and `Bundle`

The only supported write preparation flow is:

```python
bundle = ds.load_bundle("my-submission")
certificate = ds.write_certificate(
    bundle,
    "my-submission.certificate.json",
    core_labels=["oxidation_state"],
)
```

### `load_bundle` and `Bundle`

```python
bundle = ds.load_bundle(source, asset_root=None)
```

`Bundle(source, *, asset_root=None)` is the equivalent direct constructor;
`load_bundle` is preferred for readability. `asset_root` is keyword-only and overrides the
root used to resolve relative structure paths.

`source` may be a submission folder, compact SQLite exchange database, or manifest
mapping. The returned `Bundle` is deliberately small:

| Member | Result |
| --- | --- |
| `bundle.name` | Submitted dataset name |
| `bundle.version` | Version when exactly one release is present |
| `bundle.versions` | Submitted version list |
| `bundle.info` | JSON-compatible name, title, description, versions, and size |
| `len(bundle)` | Number of submitted data records |

The normalized relational rows remain private.

### Assessment input fences

#### Folder input

```text
my-submission/
├── dataset.json
└── structures/
    └── AAAKSI.mol2
```

`dataset.json` connects record identities, structures, labels, and evidence:

```json
{
  "schema_version": "hotpot-ds-submission/current",
  "dataset": {
    "name": "csd-example",
    "version": "2026.1",
    "title": "CSD example",
    "description": "Chemist-labelled coordination complexes",
    "license": "CCDC licensed",
    "access_policy": "restricted",
    "provenance": {"source": "CSD"}
  },
  "records": [
    {
      "id": "AAAKSI",
      "identifiers": {"CSD_identifier": "AAAKSI"},
      "metadata": {"compound_name": "example"},
      "structures": [
        {
          "role": "reported_crystal_structure",
          "primary": true,
          "representations": {
            "mol2": {"path": "structures/AAAKSI.mol2"}
          }
        }
      ],
      "labels": {
        "oxidation_state": {
          "value": 2,
          "evidence_level": "E2_explicit_reported",
          "evidence": [
            {
              "source": "paper-1",
              "evidence_type": "reported_value",
              "locator": "page 4",
              "quote": "The metal is reported as Eu(II).",
              "role": "supports"
            }
          ]
        }
      }
    }
  ],
  "sources": {
    "paper-1": {
      "source_type": "publication",
      "citation": "Complete citation",
      "doi": null,
      "license": "CC-BY-4.0",
      "access_policy": "open",
      "metadata": {}
    }
  }
}
```

Every record must explicitly contain `structures` and `labels`; either may be empty.
Paths must be safe relative paths under the submission root. Multiple structures are linked
by their record, role, and optional `primary` flag. A reusable skeleton is available at
[`templates/dataset.json`](templates/dataset.json).

#### Compact SQLite input

The exchange database must contain these four tables:

| Table | Required columns |
| --- | --- |
| `dataset` | `name`, `version`, `title`, `description`, `license`, `access_policy`, `provenance_json` |
| `data` | `id`, `metadata_json`, `identifiers_json` |
| `structures` | `data_id`, `format`, `path` |
| `labels` | `data_id`, `name`, `value_json` |

`data.id` is the relationship key used by `structures.data_id` and `labels.data_id`.
Structure paths are resolved relative to the directory containing the SQLite file. The
`dataset` table must contain exactly one row. A minimal schema is:

```sql
CREATE TABLE dataset (
  name TEXT NOT NULL,
  version TEXT NOT NULL,
  title TEXT NOT NULL,
  description TEXT NOT NULL,
  license TEXT NOT NULL,
  access_policy TEXT NOT NULL,
  provenance_json TEXT NOT NULL
);
CREATE TABLE data (
  id TEXT PRIMARY KEY,
  metadata_json TEXT NOT NULL,
  identifiers_json TEXT NOT NULL
);
CREATE TABLE structures (
  data_id TEXT NOT NULL REFERENCES data(id),
  format TEXT NOT NULL,
  path TEXT NOT NULL
);
CREATE TABLE labels (
  data_id TEXT NOT NULL REFERENCES data(id),
  name TEXT NOT NULL,
  value_json TEXT NOT NULL
);
```

The relationship can be inspected without Hotpot-specific backend tables:

```sql
SELECT d.id, s.format, s.path, l.name, l.value_json
FROM data AS d
LEFT JOIN structures AS s ON s.data_id = d.id
LEFT JOIN labels AS l ON l.data_id = d.id
ORDER BY d.id, s.format, l.name;
```

Recommended optional structure columns are `structure_id`, `role`, and `primary`. Optional
label columns include unit, uncertainty, conditions, method, evidence level, status, and
evidence JSON. Optional `sources` and `property_types` tables hold citations and custom
property definitions. Equivalent SQL views may be used instead of copying source tables.

### `write_certificate`

For a new dataset, explicitly supply at least one of `quality_level`, `core_labels`, or
`core_info`. An empty list is an explicit declaration that no core field exists; omission
is not equivalent.

```python
certificate = ds.write_certificate(
    bundle,
    "certificate.json",
    quality_level="B1_candidate",
    core_labels=["oxidation_state"],
    core_info=["metadata.compound_name", "structures"],
    configure="policy.json",
)
```

| Parameter | Meaning |
| --- | --- |
| `bundle` | `Bundle` returned by `load_bundle` |
| `path` | Output certificate JSON path |
| `quality_level` | Optional B0-B4 declaration for a new dataset |
| `core_labels` | Labels required on every record; `[]` explicitly means none |
| `core_info` | Information paths required on every record; `[]` explicitly means none |
| `configure` | Policy JSON path or standard mapping for a new dataset |
| `dataset` | Existing `Dataset` or existing dataset name; omit for a new dataset |
| `db` | Database path when `dataset` is an existing name |

The return value is the certificate `Path`.

For an established dataset, provide the loaded `Dataset`. Its persisted rules replace all
caller configuration, and successful ingestion creates a new immutable release.

```python
established = ds.load("csd-example", db="work/datasets.sqlite3")
certificate = ds.write_certificate(
    bundle,
    "next-release.certificate.json",
    dataset=established,
)
```

Alternatively, pass `dataset="csd-example"` together with `db=`. Existing-dataset mode
rejects `quality_level`, `core_labels`, `core_info`, and `configure` overrides.

`configure` accepts a policy JSON path or an equivalent mapping. Dataset-specific rules
override general rules; identity, relationship, safe-path, asset-hash, schema, and
certificate-integrity gates cannot be disabled.

```json
{
  "schema_version": "hotpot-ds-certificate-config/current",
  "generic": {
    "min_records": 1,
    "require_structure_or_label": true
  },
  "datasets": {
    "csd-example": {
      "core_labels": ["oxidation_state"],
      "core_info": ["metadata.compound_name"],
      "minimum_evidence": {
        "oxidation_state": "E2_explicit_reported"
      }
    }
  }
}
```

The template is available at
[`templates/certificate-config.json`](templates/certificate-config.json).
`write_certificate` writes the integrity-bound assessment and returns its `Path`. The JSON
field `certificate.passed` records the result. A certificate proves content integrity and
policy evaluation; it is not an external chemical endorsement.

## 5. `store.ingest`

`store` is the single public storage object. There is no public storage-class constructor.

```python
result = ds.store.ingest(
    bundle,
    certificate=certificate,
    db="work/datasets.sqlite3",
)
```

| Parameter | Meaning |
| --- | --- |
| `bundle` | The exact assessed `Bundle` |
| `certificate` | Path returned by `write_certificate`; mandatory |
| `db` | Target SQLite dataset store; required to prevent accidental writes to a catalog |

The method returns a JSON-compatible count dictionary. Before writing, Hotpot verifies the
certificate envelope, normalized bundle digest, configured rules, asset paths and hashes,
and established-dataset policy. A failed, mismatched, or modified certificate is
rejected. Existing data are integrated as a new immutable release; certified releases are
never modified in place.

For commands, see the [`hotpot ds` CLI reference](cli_doc.md). For normalized backend
automation, see [`hotpot.ds.detail`](detail/README.md). The tracked resource layout and
ignored payload boundary are documented in [`hotpot/dataset`](../dataset/README.md).
