# Hotpot dataset resources

`hotpot.dataset` is the single filesystem home for Hotpot dataset resources.
The public API, CLI, assessment engine, and ingestion implementation live in
`hotpot.ds`; scientific payloads are local, ignored data rather than tracked or
package-distributed resources.

A benchmark is one governed archive class or subset within a dataset. It is not
a parallel `benchmark/` database or package.

## Repository organization

```text
hotpot/
|-- ds/                          # API, CLI, Gates, assessment, storage, tools
|   |-- public/                  # concise public-object implementation
|   |-- gates/                   # Evidence, Quality, and Benchmark policy
|   |-- schemas/                 # Bundle and CLI transport schemas
|   `-- tools/                   # owner/admin rebuild and audit tools
`-- dataset/
    |-- README.md                # this organization contract
    |-- docs/                    # scientific scope, attribution, storage rules
    |-- schemas/                 # tracked source/record definitions
    |-- manifests/               # tracked inventories without payload bytes
    `-- data/                    # all local scientific data; ignored by Git
        |-- store/               # immutable generations and active pointer
        |-- bundles/             # canonical Bundle roots
        |-- sources/             # source-native labels, evidence, structures
        `-- audits/              # supplier submissions and assessment reports
```

The entire `hotpot/dataset/data/**` tree is excluded from Git. This includes
licensed structures, generated databases, Bundle facts, certificates,
assessment generations, supplier records, and audit outputs. A clone contains
schemas, rules, and documentation, but not scientific payloads. Restore data
from an authorized artifact source or rebuild it with current owner/admin
tools.

The default query store is:

```text
hotpot/dataset/data/store/active/datasets.sqlite3
```

Its target-owned content-addressed assets are adjacent:

```text
hotpot/dataset/data/store/active/datasets.sqlite3.assets/
`-- objects/
    `-- sha256/
        `-- <first-two-hex>/
            `-- <full-sha256>
```

Package code resolves these locations through `hotpot.ds._paths`. Scripts must
not introduce another dataset payload root. `store/active` is a directory
symlink to one complete directory below `store/generations`; owner cutover or
recovery operations replace that pointer atomically only after the database and
adjacent asset store pass parity checks. These are administrative operations,
not public query-time compatibility paths.

## Responsibility boundaries

| Area | Responsibility |
| --- | --- |
| `hotpot.ds` | Query, search, build Bundles, assess, certify, ingest, replicate formal stores, and export |
| `hotpot.ds.gates` | Persistent dataset-specific evidence, quality, and benchmark policy |
| `hotpot.ds.providers` | Private readers for upstream distributions |
| `hotpot.dataset/docs` | Scientific scope, evidence limits, licenses, provenance, and storage contracts |
| `hotpot.dataset/schemas` | Tracked source-facing definitions in this checkout |
| `hotpot.dataset/manifests` | Tracked inventory metadata without scientific records |
| `hotpot.dataset/data` | Untracked local sources, Bundles, target stores, assets, and reports |

Git is the implementation and schema history. Any checkout exposes exactly one
current Bundle format, target schema, Gate/checker protocol, certificate
envelope, and CLI communication map. Superseded stores are retained only as
explicit audit or rollback artifacts, never as compatibility backends.

## Data model

One Bundle represents exactly one dataset release and owns every Case in that
release. Canonical Case facts are classification-independent. Synchronization
derives one immutable assessment generation and separate indexes for:

- `incomplete`: missing/invalid required material or significant unresolved
  scientific conflict;
- `ordinary`: complete and archivable, but not admitted by BenchmarkGate;
- `benchmark`: complete, archivable, and admitted by BenchmarkGate.

Normal persisted queries include ordinary and benchmark data. Incomplete data
is isolated and requires an explicit read or forced-ingestion opt-in. Equal
asset bytes may be deduplicated without merging Case, scientific-sample,
chemical-entity, or Structure identity.

Quality and evidence are independent Case assessments. B0-B4 describes
curation and validation strength; E0-E4 describes the linked support for a
claim or observation. Neither is assigned directly by a source row. Benchmark
membership is a third, policy-controlled decision.

Evidence source type is an explicit fact, not something inferred from a URL or
DOI. The current closed values are `primary_literature`,
`secondary_literature`, `database_record`, `expert_statement`, and `other`.
AI verification is usable only when it records the verifier, tool and tool
version, procedure version, a timezone-aware timestamp, and a locator or quote
on the linked evidence. Missing or hash-mismatched file-backed evidence is
retained as a negative asset fact and cannot contribute support until repaired.

Property domains remain independent. Oxidation state, spin state, formal
charge, stability constants, separation factor, distribution ratio, spectra,
and structure-only collections may refer to the same entity or asset without
sharing labels, evidence, or assessment.

## Release workflow

```text
current-format intake folder
    -> one-time Bundle build
    -> audited fact edits, if needed
    -> explicit full synchronization
    -> root certificate
    -> certificate-required target ingestion
    -> immutable sealed release
    -> optional verified removal of reproducible Bundle intermediates
    -> optional exact replication into another Hotpot package
```

New datasets may propose Gates and comparison policy. First ingestion persists
that policy. Later releases of an established dataset must use the stored
policy; Bundle cannot replace it. Successful ingestion copies every positive
scientific asset into target-owned content-addressed storage and records an
authoritative receipt. The formal store keeps compact original facts and final
governance labels, not Bundle fact/generation/migration copies. After receipt
verification, `hotpot ds bundle rm --all` removes only allowlisted reproducible
Bundle products and preserves the intake, original evidence, policy,
certificate, receipt, and audit history.

`hotpot ds copy TARGET_HOTPOT_PACKAGE` can publish an exact verified replica of
the complete formal store into another current Hotpot package. This operation
preserves the store identity and does not rerun Bundle build, certification, or
ingestion; it never performs an implicit merge or overwrite.

See the [public API](../ds/README.md), [CLI reference](../ds/cli_doc.md), and
[storage and cutover contract](docs/storage.md).
