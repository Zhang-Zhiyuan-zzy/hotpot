# `hotpot ds` command-line reference

The installed command is `hotpot ds`; no `python -m` prefix is required:

```bash
hotpot ds --help
```

The public workflow is arranged by purpose:

```text
Query:       list, info, get, labels, subset, search, export
Prepare:     bundle build, bundle status/get/set/gate, bundle synchronize
Publish:     certify, ingest
```

Every successful command writes one validated JSON envelope to stdout. Human
presentation and diagnostics use stderr. The current Git revision defines the
only supported CLI and wire protocol; there is no runtime compatibility mode.

## 1. Query commands

Normal persisted-dataset queries include `ordinary` and `benchmark` data and
exclude the isolated `incomplete` archive class. Query and export commands
accept repeatable `--archive-class {ordinary,benchmark,incomplete}`. Supplying
the flag replaces that default and is the one explicit CLI opt-in to incomplete
data. A subset can narrow its loaded view. A pipeline carries selected IDs and
therefore cannot add records that were absent upstream.

### List datasets

```text
hotpot ds list [--db PATH]
```

```bash
hotpot ds list | jq -r '.payload.datasets[]'
```

### Inspect a dataset

```text
hotpot ds info DATASET [--archive-class CLASS ...]
                       [--version VERSION] [--db PATH]
hotpot ds labels DATASET [--archive-class CLASS ...]
                         [--version VERSION] [--db PATH]
```

`VERSION` may be a source version, immutable release ID, integer release
serial, or three-digit serial.

### Get data

```text
hotpot ds get DATASET SELECTOR
              [--index | --id | --identifier [NAMESPACE]]
              [--smi | --xyz]
              [--structure-index N | --structure-role ROLE | --structure-id ID]
              [--archive-class CLASS ...]
              [--version VERSION] [--db PATH]
```

`SELECTOR` is a zero-based sequence index by default. `--id` selects a stable
Hotpot ID. `--identifier [NAMESPACE]` selects an external identifier; its
namespace may be omitted only if the dataset exposes exactly one namespace.

```bash
hotpot ds get oxidation 0
hotpot ds get oxidation 'hp-oxidation[001]-0' --id
hotpot ds get oxidation AACANI11 --identifier CSD
```

`--smi` and `--xyz` are mutually exclusive and place the representation in the
JSON payload. If the record has multiple attached structures, select one with
an unambiguous attached-structure selector:

```bash
hotpot ds get spin 0 --smi --structure-role low_temperature
hotpot ds get spin 0 --xyz --structure-index 1
hotpot ds get spin 0 --xyz --structure-id STRUCTURE_ID
```

`--structure-index` follows the zero-based order in `data.info`. A role must
match exactly one attached structure. An ID must match exactly. Without a
selector, Hotpot requires one attached structure or one unambiguous primary
structure. Missing representations are converted with `hotpot.Molecule`; a
missing structure or failed conversion is an error. The same selector is
applied to every item in a protocol-driven batch and must resolve for all of
them.

When stderr is an interactive terminal, Hotpot may mirror requested SMI or XYZ
text there. The complete machine representation remains in stdout:

```bash
hotpot ds get spin 0 --xyz --structure-index 0 \
  | jq -rj '.payload.representations[0].content'
```

### Filter and search

```text
hotpot ds subset DATASET [--where FIELD=JSON ...]
                         [-q LEVEL] [-e LEVEL]
                         [--limit N] [--archive-class CLASS ...]
                         [--version VERSION] [--db PATH]

hotpot ds search [DATASET] [--scope data|dataset|smarts]
                            [--where FIELD=JSON ...]
                            [--smarts PATTERN] [--limit N]
                            [--archive-class CLASS ...]
                            [--version VERSION] [--db PATH]
```

Repeated `--where` predicates are ANDed. A JSON scalar means equality and a
JSON array means membership. Nested paths use dots or `__`.

`-q/--quality` and `-e/--evidence` accept integer thresholds `0` through `4`.
Zero, the default, disables that threshold. In particular, `-e 0` retains
records whose evidence level is not applicable; a positive evidence threshold
excludes them.

```bash
hotpot ds subset spin -q 1 -e 2 \
  --where 'labels.spin_multiplicity=[1,5]'

hotpot ds search oxidation --scope smarts --smarts '[N]-[C]'
hotpot ds search --scope dataset --where 'core_labels=["oxidation_state"]'
```

### Export

```text
hotpot ds export DATASET OUTPUT [ITEM]
                [--index | --id | --identifier [NAMESPACE]]
                [--range START STOP | --where FIELD=JSON ...]
                [--format FORMAT]
                [--archive-class CLASS ...]
                [--version VERSION] [--db PATH]
```

Omitting `ITEM`, `--range`, and `--where` exports the whole visible Dataset.
`--range START STOP` is a zero-based half-open range. Selection flags follow
the same rules as `get`.

```bash
hotpot ds export oxidation export/all
hotpot ds export oxidation export/one 0
hotpot ds export oxidation export/range --range 100 200
hotpot ds export oxidation export/oxidation-iii \
  --where 'labels.oxidation_state=3'
hotpot ds export oxidation export/mol2 --format mol2
```

Without `--format`, each governed structure is copied in its stored format.
With `--format`, every selected molecular structure is converted through
`hotpot.Molecule`; any failed conversion aborts the atomic export.

Exactly one record with exactly one structure uses:

```text
OUTPUT/
|-- struct.<format>
`-- info.json
```

Every multi-record or multi-structure export uses aligned file names and a
summary:

```text
OUTPUT/
|-- struct-000000-00.<format>
|-- info-000000-00.json
|-- struct-000000-01.<format>
|-- info-000000-01.json
`-- summary.json
```

The operation never overwrites an existing output path and never leaves a
partially populated result.

### Current-store diagnostics

These commands inspect the current `TargetStore`; they do not expose a second
Python storage API:

```bash
hotpot ds detail status --db work/datasets.sqlite3
hotpot ds detail audit --db work/datasets.sqlite3
hotpot ds detail query claims --db work/datasets.sqlite3 \
  --dataset spin --evidence-level E3 --limit 100
```

`status` reports table counts, `audit` runs SQLite integrity and foreign-key
checks, and `query claims` returns current Case claim facts with their Case
assessment level. These are diagnostics over the same store used by `load`.

## 2. Bundle commands

A Bundle root contains exactly one dataset release. Build, synchronize,
certify, and ingest are distinct state transitions and never call one another
implicitly.

### Build and locate

```text
hotpot ds bundle build BUNDLE_ROOT [--dataset NAME] [--db DB]

hotpot ds bundle [--root BUNDLE_ROOT] [--dataset NAME] [--db DB] status
hotpot ds bundle [--root BUNDLE_ROOT] [--dataset NAME] [--db DB] synchronize
hotpot ds bundle [--root BUNDLE_ROOT] [--dataset NAME] [--db DB] rm --all
```

`build` is the one-time canonical build operation and requires an unbuilt
current-format root. Every other Bundle command opens an already built Bundle
without importing or repairing it. `--root` defaults to the current directory
except for `build`, where `BUNDLE_ROOT` is required and `--root` is invalid.

The required `bundle.json.ds_name` is authoritative. `--dataset NAME` is an
optional assertion and must match it; it never changes the dataset name.
Build also binds one canonical target store. Later Bundle commands inherit
that target when `--db` is omitted and treat an explicit `--db` as a matching
assertion. The publish operation `ingest` still requires the target explicitly.
`bundle status` remains available while unsynchronized and reports the next
valid action. Its machine payload includes `runtime_policy_current`, which is
true only when the Bundle's persisted assessment, checker, parser, and runtime
binding matches this installed Hotpot source tree.

### Inspect

```text
hotpot ds bundle [LOCATOR OPTIONS] get info
hotpot ds bundle [LOCATOR OPTIONS] get summary
hotpot ds bundle [LOCATOR OPTIONS] get cases [CASE FILTERS]
hotpot ds bundle [LOCATOR OPTIONS] get case.info CASE_ID
hotpot ds bundle [LOCATOR OPTIONS] get case.assessment CASE_ID
```

`get cases` supports the closed filter vocabulary:

```text
--archive-class VALUE
--quality VALUE
--min-quality VALUE
--evidence VALUE
--min-evidence VALUE
--benchmark-status VALUE
--reason VALUE
--verification VALUE
--attestation VALUE
```

Repeat one flag for IN semantics; different flags are ANDed. Derived queries
require a synchronized Bundle. A zero evidence threshold means no evidence
filter and retains the not-applicable population.

```bash
hotpot ds bundle get cases --quality B2 --min-evidence E2
hotpot ds bundle get case.assessment CASE_ID
```

### Edit Case facts

```text
hotpot ds bundle [LOCATOR OPTIONS] set case.METHOD INPUT.json
```

`INPUT.json` contains `case_id` and that method's named arguments. The closed
CLI mutation allowlist is:

```text
case.set_core_info
case.replace_structure
case.add_observation
case.add_evidence
case.verify_evidence
case.attest_benchmark
```

The JSON file is an artifact argument, not raw stdin. One invocation performs
one audited fact transaction and leaves the Bundle unsynchronized. The smaller
CLI allowlist covers routine review work; the complete audited mutation API is
available on Python `Case` objects. Public CLI attestations may use `User` or
`Other`, never unsigned `Hotpot` authority.

```bash
hotpot ds bundle set case.add_evidence evidence.json
hotpot ds bundle set case.verify_evidence verification.json
```

For example, `evidence.json` is a closed object whose keys are the Case method
arguments:

```json
{
  "case_id": "case-0001",
  "target": "observation-0001",
  "source": "doi:10.1000/example",
  "relation": "supports",
  "locator": "Table 2",
  "details": {
    "source_kind": "primary_literature",
    "source_lineage_id": "experiment:example"
  }
}
```

The optional `details` object is deliberately closed. For
`case.add_evidence`, it accepts `source_kind`, `doi`, `url`,
`scientific_work_id`, `source_lineage_id`, and `derived_from` (an array of
lineage IDs). `source_kind` is one of `primary_literature`,
`secondary_literature`, `database_record`, `expert_statement`, or `other`.
DOI and URL facts are inferred from `doi:...` and `url:...` source references,
but they never imply a source kind. For `case.verify_evidence`, `details`
accepts `tool`, `tool_version`, `procedure_version`, and a timezone-aware
`verified_at` timestamp:

```json
{
  "case_id": "case-0001",
  "evidence_id": "case-0001:evidence:...",
  "method": "AI",
  "verifier": "review-agent@example.org",
  "details": {
    "tool": "review-tool",
    "tool_version": "2.1.0",
    "procedure_version": "primary-source-review/current",
    "verified_at": "2026-10-14T01:30:00Z"
  }
}
```

When verification details are omitted, Hotpot records its current procedure
identifier and the current UTC time. AI verification still requires explicit
`tool` and `tool_version`; `verifier` identifies the attributable agent.
File-backed evidence whose bytes are missing or whose declared hash mismatches
is retained as a negative fact and cannot contribute support. That condition
alone does not make the Case incomplete; with no other usable support the
affected target assesses at E0. These facts participate in synchronization and
certification and do not directly assign an evidence level.

### Inspect or edit Gates

```text
hotpot ds bundle [LOCATOR OPTIONS] gate evidence --show
hotpot ds bundle [LOCATOR OPTIONS] gate quality --show
hotpot ds bundle [LOCATOR OPTIONS] gate benchmark --show
```

Inspection works in both modes. Editing works only for a new dataset:

```text
# All Gates
--add-check NAME[=true|false]        --rm-check NAME
--checker NAME=FILE:ENTRYPOINT

# QualityGate and EvidenceGate only
--set-level LEVEL=NAME[,NAME...]     --rm-level LEVEL

# QualityGate only
--add-structure-check NAME[=true|false]
--rm-structure-check NAME

# BenchmarkGate only
--add-direct NAME[=true|false]       --rm-direct NAME
--add-veto NAME                      --rm-veto NAME
--set-tolerance PROPERTY=ABS[,REL]   --rm-tolerance PROPERTY
```

`--checker` defines the callable for a same-named item added by the same
invocation. It must follow the fixed two-argument checker protocol. Multiple
edits in one invocation are atomic. `--show` cannot be mixed with edits.
Existing datasets use their persisted policy and reject every Gate edit.

### Synchronize

```bash
hotpot ds bundle synchronize
```

Synchronization evaluates all Cases and atomically publishes a new immutable
assessment generation. It does not write a certificate. Users can make several
fact or Gate edits, then synchronize once.

### Remove sealed Bundle intermediates

```bash
hotpot ds bundle rm --all
```

`rm --all` is an irreversible post-ingestion cleanup operation. It is accepted
only for a sealed Bundle whose certificate, local ingestion receipt,
authoritative target receipt, target SQLite database, and target-owned assets
all pass complete integrity verification.  The command uses the target bound
by the Bundle when `--db` is omitted; an explicit `--db` is a matching
assertion.  Therefore the short form above works from the Bundle root even
when the Bundle already records its target.

The command takes an exclusive Bundle lock and deletes only a closed allowlist
of reproducible working material: the canonical facts SQLite files, assessment
generations and their leases/caches/temporary files, current-generation links,
and the two explicitly recognized derived controls
`evidence/migration/identity-migration.jsonl` and
`evidence/migration/legacy-identity-relationships.jsonl`. A free-form
`category: "migration"` declaration is not deletion authority. In particular,
`legacy-source-inventory.json` and `migration-provenance.json` are permanent
source/provenance records and are never removed. The command also keeps
`bundle.json`, the original intake, scientific structures and evidence,
persisted policy and checker source, the certificate, the ingestion receipt,
and audit history. A derived control that is also an intake or scientific
source is rejected as an ambiguous declaration rather than removed.

Before deletion the command atomically writes `bundle-cleanup.json` with the
authoritative receipt binding and the path, SHA-256 digest, and byte count of
every planned deletion. It marks that manifest complete only after deletion.
Repeating the same command first revalidates the target schema and integrity,
release content, authoritative and local receipts, and certificate. It then
resumes a planned cleanup or returns the completed result idempotently without
reopening the now intentionally non-loadable Bundle working state. The
ingested dataset remains available through the Python `ds.load` API and
`hotpot ds` query commands because the target store owns the release.

## 3. Certify and ingest

### Certify

```text
hotpot ds certify BUNDLE_ROOT [--dataset NAME] [--db DB]
```

Certification requires a synchronized Bundle and writes exactly
`BUNDLE_ROOT/certificate.json`. It verifies and serializes the current facts,
policy, scientific-asset inventory, governance snapshot, and assessment
generation. It neither evaluates Gates nor ingests data.

### Ingest

```text
hotpot ds ingest BUNDLE_ROOT --db DB [--dataset NAME]
                 [--include-incomplete --reason TEXT]
```

`--db` is mandatory. Ingestion requires the current root certificate and
rechecks the Bundle, generation, target, and policy inside the target
transaction. It copies materialized scientific assets into the target-owned
content-addressed asset area, records one authoritative receipt, and seals the
release.

Incomplete Cases are excluded unless both `--include-incomplete` and a
non-empty `--reason` are supplied:

```bash
hotpot ds ingest submission --db work/datasets.sqlite3 \
  --include-incomplete --reason 'Preserve supplier records pending repair'
```

An exact retry returns the existing receipt. Reusing the same release identity
with different content, policy, certificate, or selected Case sets is an error.

## 4. JSON stdin/stdout protocol

### Envelope

Every machine result has exactly four top-level fields:

```json
{
  "key": "hotpot.ds.subset",
  "channel": "default",
  "protocol_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "payload": {}
}
```

`key` identifies the producer, `channel` identifies its semantic result,
`protocol_sha256` binds the one current communication map and every referenced
payload schema, and `payload` contains the validated result. Unknown fields,
keys, channels, schema digests, or map digests are rejected.

Stdout contains exactly one complete envelope followed by LF, or nothing when
the command fails before producing a complete result. Diagnostics, warnings,
prompts, and optional human structure presentation use stderr. Exit status `0`
means a completed nonnegative result; status `2` covers usage, protocol,
execution, or governance failure. A completed domain-negative result may emit
its valid envelope and return `2`.

### Explicit protocol input

Only a documented positional `-` activates stdin:

```text
hotpot ds get DATASET - [GET OPTIONS]
hotpot ds export DATASET OUTPUT - [EXPORT OPTIONS]
```

The input must be one current Hotpot envelope, never raw JSON, an identifier
list, or JSON Lines. A pipe alone does not make a command read stdin.

```bash
hotpot ds subset spin -q 1 -e 2 | hotpot ds get spin -

hotpot ds search spin --scope data \
  --where 'labels.spin_multiplicity=5' \
  | hotpot ds get spin - --xyz --structure-index 0

hotpot ds subset spin -q 3 \
  | hotpot ds export spin export/reference - --format mol2

hotpot ds subset spin --archive-class incomplete \
  | hotpot ds get spin - --archive-class incomplete
```

The direct `DATASET` must match upstream context. Omitted `--version` and
`--db` inherit the immutable release and store; supplied values must match.
Selector-mode flags are invalid with `-`. Structure selection, display format,
export format, and output destination remain direct downstream controls.
Archive visibility is also a direct control: if upstream explicitly selected
`incomplete`, repeat the needed `--archive-class` values downstream so those
IDs are loadable. The consumer still emits only upstream-selected IDs.

`get` accepts zero or more upstream records, preserves order and duplicates,
and emits one envelope. `export` requires one or more. Every record and
conversion is resolved before stdout or output artifacts are published.

Bundle mutation and lifecycle commands do not accept `-` in the current
profile. Their JSON file parameters remain normal artifact inputs.

### Current producer inventory

Every producer/channel has a named closed payload contract; terminal outputs do
not use a `null` schema escape hatch. `CaseInfo`, `CaseAssessment`, and the
known fields nested in each data item have closed schemas. Dataset-defined
metadata and scientific fact extensions remain explicit JSON maps whose values
are recursively JSON-validated; they are not unvalidated object escapes.

| Producer key | Channels and contracts |
| --- | --- |
| `hotpot.ds.list` | `default -> dataset_list_result` |
| `hotpot.ds.info` | `default -> dataset_info_result` |
| `hotpot.ds.labels` | `default -> dataset_labels_result` |
| `hotpot.ds.get` | `default, structure -> data_collection` |
| `hotpot.ds.subset` | `default -> data_collection` |
| `hotpot.ds.search` | `data, smarts -> data_collection`; `dataset -> dataset_collection` |
| `hotpot.ds.export` | `default -> export_result` |
| `hotpot.ds.detail` | `default -> detail_result` |
| `hotpot.ds.detail.audit` | `default -> audit_result` |
| `hotpot.ds.detail.query.claims` | `default -> claims_result` |
| `hotpot.ds.detail.status` | `default -> detail_status_result` |
| `hotpot.ds.bundle.build` | `default -> bundle_build_result` |
| `hotpot.ds.bundle.get` | `info`, `summary`, `cases`, `case_info`, `case_assessment` |
| `hotpot.ds.bundle.set` | `default -> bundle_mutation_result` |
| `hotpot.ds.bundle.gate` | `default -> bundle_gate_result` |
| `hotpot.ds.bundle.status` | `default -> bundle_status_result` |
| `hotpot.ds.bundle.synchronize` | `default -> bundle_synchronize_result` |
| `hotpot.ds.bundle.rm` | `default -> bundle_cleanup_result` |
| `hotpot.ds.certify` | `default -> certificate_result` |
| `hotpot.ds.ingest` | `default -> ingestion_result` |

The machine-readable authority is `hotpot/ds/cli_communication.json`; the
repository protocol specification is
[`skills/hotpot-cli-stdio-protocol/references/protocol.md`](../../skills/hotpot-cli-stdio-protocol/references/protocol.md).

## 5. Bundle intake layout

The concise current-format layout is:

```text
BUNDLE_ROOT/
|-- bundle.json
|-- cases.jsonl              # or declared intake.sqlite3, not both
|-- structures/
|   `-- ...
`-- evidence/
    `-- ...
```

`bundle.json` declares the one dataset/release identity, release serial,
required core fields and structures, identity policy, comparator policy, and
active Case source. Each Case has explicit IDs connecting structures,
observations, claims, evidence, verifications, attestations, entities, and
identifiers. Local assets use safe root-relative paths. Missing assets use
explicit negative declarations.

After `bundle build`, the root also contains canonical facts and policy
projections. `bundle synchronize` adds immutable generation artifacts and
derived class indexes. `certify` adds the root `certificate.json`. Users do not
edit generated control or assessment files directly.

See the [Python API reference](README.md) and
[storage contract](../dataset/docs/storage.md) for the complete ownership and
cutover rules.
