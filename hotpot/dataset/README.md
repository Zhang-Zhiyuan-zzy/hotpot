# Hotpot dataset resources

`hotpot.dataset` is the single filesystem home for datasets used by Hotpot.
The management implementation and public API live in `hotpot.ds`; data files do not.
A benchmark is a quality designation or a filtered subset, not a parallel storage system.

## Repository layout

```text
hotpot/
├── ds/                         # API, CLI, assessment, ingestion, and tools
│   └── providers/              # internal readers for upstream distributions
└── dataset/
    ├── README.md               # this organization contract
    ├── docs/                   # scientific scope, provenance, and attribution
    ├── schemas/                # current machine-readable record definitions
    ├── manifests/              # tracked layout and inventory metadata
    └── data/                   # local payloads; entirely excluded from Git
        ├── store/              # datasets.sqlite3 and generated certificates
        ├── sources/            # source-specific labels and structures
        ├── structures/         # shared, content-addressed structure objects
        └── audits/             # supplier submissions and generated audits
```

`hotpot/dataset/data/**` is ignored deliberately. A clone contains the rules and
documentation, but not licensed, large, generated, or review-sensitive payloads. Local data
must be restored from an authorized source or rebuilt with the tools in `hotpot/ds/tools`.

The default query store is:

```text
hotpot/dataset/data/store/datasets.sqlite3
```

All package code resolves this location through `hotpot.ds._paths`; scripts use no other
dataset payload root.

## Responsibility boundaries

| Area | Responsibility |
| --- | --- |
| `hotpot.ds` | Load, search, assess, certify, ingest, export, and validate data |
| `hotpot.ds.providers` | Internal source readers; not part of the public API facade |
| `hotpot.dataset/docs` | Human-readable scope, evidence limits, licenses, and provenance |
| `hotpot.dataset/schemas` | The only supported record and source schemas in this checkout |
| `hotpot.dataset/manifests` | Tracked inventory without scientific payloads |
| `hotpot.dataset/data` | The current local store and source payloads |

Git is the implementation version history. The checkout exposes one current schema and one
current storage contract.

## Scientific separation

Oxidation state, spin state, molecular formal charge, stability constants, distribution
ratios, separation factors, spectra, and structure-only collections are independent property
types. They may share an entity or coordinate object without sharing labels or evidence.

Quality and evidence are also independent:

- `quality_level` describes curation and benchmark readiness of a dataset or subset;
- `evidence_level` describes support for an individual claim or observation.

See the [Hotpot DS API](../ds/README.md), [CLI reference](../ds/cli_doc.md), and the
[local storage contract](docs/storage.md).
