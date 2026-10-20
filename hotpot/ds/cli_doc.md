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
Replicate:   copy
```

Every successful operational command writes one validated JSON envelope to
stdout. `--help` and `--doc` are presentation-only exceptions and render help
or Markdown directly. Human presentation and diagnostics use stderr during
normal operations. The current Git revision defines the only supported CLI and
wire protocol; there is no runtime compatibility mode.

Use `--help` for the option synopsis and `--doc` for examples, data-layout
requirements, and failure semantics. Documentation is available at the root
and at every command level:

```bash
hotpot ds --doc
hotpot ds get --doc
hotpot ds bundle --doc
hotpot ds bundle build --doc
hotpot ds bundle gate evidence --doc
```

## Fastest complete publication example

Assume `submission/` is an unbuilt Bundle intake containing `bundle.json`, one
declared Case source, and any structure or evidence files referenced by its
Cases. The shortest explicit lifecycle is:

```bash
hotpot ds bundle build submission
hotpot ds bundle --root submission status
hotpot ds bundle --root submission synchronize
hotpot ds certify submission
hotpot ds ingest submission
hotpot ds list
hotpot ds info spin
hotpot ds get spin 0
```

Run the first three commands from inside `submission/` to omit its path:

```bash
cd submission
hotpot ds bundle build
hotpot ds bundle synchronize
hotpot ds certify .
hotpot ds ingest .
```

No transition is implicit: build does not synchronize, synchronize does not
certify, and certify does not ingest. By default the Bundle and ingestion bind
the current store inside the imported `hotpot` package. To publish to a
separate store, pass the same explicit `--db PATH` when building, certifying,
and ingesting.

Common pitfalls:

- Do not point `bundle build` at a directory containing only structures or a
  bare SQLite file. It must be a Bundle root with `bundle.json` and its
  declared intake.
- Do not edit `bundle.json`, the Case intake, or generated controls after
  building. Use audited Bundle mutations for review changes; a deliberately
  replaced referenced asset also requires synchronization and recertification.
- A regular pip installation may not be writable. Use `--db` to select a
  writable publication store if the imported package directory is read-only.
- stdout is machine JSON. Use `jq` to render selected fields; diagnostics and
  optional human structure output are written to stderr.

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

Use an explicit store when inspecting a publication target other than the
store inside the imported Hotpot package:

```bash
hotpot ds list --db /srv/hotpot/datasets.sqlite3
```

Common pitfalls:

- `--db` names the SQLite database file, not its `.assets` directory and not a
  Bundle root.
- An empty list usually means that the running command imported a different
  Hotpot installation than the one into which data was ingested. Compare the
  database path reported by `hotpot ds detail status`.
- stdout is an envelope; dataset names are below `.payload.datasets` rather
  than printed as unstructured lines.

### Inspect a dataset

```text
hotpot ds info DATASET [--archive-class CLASS ...]
                       [--version VERSION] [--db PATH]
hotpot ds labels DATASET [--archive-class CLASS ...]
                         [--version VERSION] [--db PATH]
```

`VERSION` may be a source version, immutable release ID, integer release
serial, or three-digit serial.

```bash
hotpot ds info spin | jq '.payload.dataset'
hotpot ds labels spin | jq '.payload.labels'
hotpot ds info spin --version 001
hotpot ds info spin --archive-class incomplete
```

`info` describes one immutable release and its visible archive population.
`labels` reports the available label names and counts for that same visible
population.

Common pitfalls:

- Dataset names and release selectors are different. Pass the dataset name as
  `DATASET`; use `--version` only to choose among its releases.
- Ordinary queries exclude `incomplete`. Repeat `--archive-class` to combine
  classes explicitly, for example `--archive-class ordinary --archive-class
  benchmark`.
- If multiple releases make a selector ambiguous, use the immutable release
  ID or the three-digit serial shown by `info`.

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

Common pitfalls:

- Sequence indexes are zero-based and belong to the selected immutable
  release and archive view. They are not external identifiers.
- `--identifier` without a namespace works only when the Dataset exposes one
  identifier namespace. Otherwise provide it, for example `--identifier CSD`.
- `--structure-index`, `--structure-role`, and `--structure-id` are mutually
  exclusive. A multi-structure Case requires one selector unless exactly one
  primary structure is unambiguous.
- `--smi` and `--xyz` request one representation. They do not select a
  structure and cannot be combined.
- Conversion failure is an error rather than a silent `null`; inspect
  `data.info` without a representation flag to see the stored structures.

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

Common pitfalls:

- Quote every `FIELD=JSON` expression so the shell does not alter brackets,
  quotes, SMARTS characters, or whitespace.
- The right-hand side is JSON: strings need JSON quotes, while numbers,
  booleans, `null`, and arrays use their normal JSON spelling.
- `subset` and data/SMARTS search require `DATASET`. Dataset-scope search does
  not.
- `-q 0` and `-e 0` disable their thresholds; they do not mean “only B0” or
  “only E0”.
- A pipeline can only narrow upstream IDs. It never widens archive visibility
  or silently switches releases.

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

Common pitfalls:

- `OUTPUT` must not already exist. Choose a new directory or remove the old
  export deliberately before retrying.
- `--range START STOP` is half-open: `--range 0 10` exports ten records.
- `ITEM`, `--range`, `--where`, and protocol selector `-` are alternative
  scopes; do not combine them.
- `--format` converts every selected molecular structure. One unsupported or
  invalid conversion aborts the entire atomic export.
- Multi-structure output always uses aligned `struct-*` and `info-*` names;
  do not assume one file per Case.

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

Common pitfalls:

- `detail` inspects a formal target store, not a Bundle working directory.
- `detail query` currently requires the `claims` target; it is not a free-form
  SQL interface.
- `--evidence-level` uses the stored Case level spelling such as `E3`, not the
  numeric threshold syntax used by `subset -e 3`.
- Passing a different `--db` changes the inspected store. It does not copy or
  attach that store to the default installation.

## 2. Bundle commands

A Bundle root contains exactly one dataset release. Build, synchronize,
certify, and ingest are distinct state transitions and never call one another
implicitly.

### Build and locate

```text
hotpot ds bundle build [BUNDLE_ROOT] [--dataset NAME] [--db DB]

hotpot ds bundle [--root BUNDLE_ROOT] [--dataset NAME] [--db DB] status
hotpot ds bundle [--root BUNDLE_ROOT] [--dataset NAME] [--db DB] synchronize
hotpot ds bundle [--root BUNDLE_ROOT] [--dataset NAME] [--db DB] rm --all
```

`build` is the one-time canonical build operation and requires an unbuilt
current-format root. `BUNDLE_ROOT` defaults to the command's current working
directory. Every other Bundle command opens an already built Bundle without
importing or repairing it. Their `--root` option has the same current-directory
default; `--root` is invalid after `bundle build` because its positional root
already names the intake.

The required `bundle.json.ds_name` is authoritative. `--dataset NAME` is an
optional assertion and must match it; it never changes the dataset name.
Build also binds one canonical target store. Later Bundle commands inherit
that target when `--db` is omitted and treat an explicit `--db` as a matching
assertion. When `ingest --db` is omitted, the target is the default store inside
the Hotpot package imported by the running command.
`bundle status` remains available while unsynchronized and reports the next
valid action. Its machine payload includes `runtime_policy_current`, which is
true only when the Bundle's persisted assessment, checker, parser, and runtime
binding matches this installed Hotpot source tree.

A directly buildable JSONL intake can be as small as:

```text
submission/
|-- bundle.json
|-- cases.jsonl
|-- structures/
|   `-- sample-0001.smi
`-- evidence/
    `-- source-0001.pdf
```

The manifest names the Case source and every dataset-level completeness and
identity requirement. Empty requirements must still be explicit:

```json
{
  "schema": "hotpot-ds-bundle/current",
  "ds_name": "spin",
  "release_id": "spin.v001",
  "release_serial": 1,
  "version": "1.0.0",
  "title": "Curated molecular spin states",
  "description": "One reviewed spin-state release",
  "license": "project-specific",
  "access_policy": "local",
  "provenance": {"provider": "laboratory ELN"},
  "core_labels": ["spin_multiplicity"],
  "core_info": ["sample_id"],
  "intake": {"format": "jsonl", "path": "cases.jsonl"},
  "identity_schema": {
    "components": [
      {
        "key": "sample",
        "path": "core_info.sample_id",
        "required": true,
        "value_kind": "text",
        "normalizer": "text",
        "comparator": "exact"
      }
    ]
  },
  "required_structure_roles": ["primary"]
}
```

Each nonblank `cases.jsonl` line is one complete JSON object. The IDs create
the relationships: an Observation can name an Entity as its `subject`, and an
Evidence record names an Observation or Claim as its `target`. Structure and
Evidence paths are relative to the Bundle root:

```json
{"id":"case-0001","local_id":"sample-0001","public_index":0,"identifiers":{"ELN":"sample-0001"},"metadata":{},"core_info":{"sample_id":"sample-0001"},"entities":[{"entity_id":"entity-0001"}],"structures":[{"structure_id":"structure-0001","role":"primary","primary":true,"representations":{"smi":{"path":"structures/sample-0001.smi"}}}],"observations":[{"observation_id":"observation-0001","property":"spin_multiplicity","value":5,"subject":"entity-0001"}],"claims":[],"evidence":[{"evidence_id":"evidence-0001","target":"observation-0001","source":"doi:10.1000/example","source_kind":"primary_literature","relation":"supports","locator":"Table 1","path":"evidence/source-0001.pdf"}],"verifications":[],"attestations":[]}
```

For large submissions, declare
`{"format":"sqlite","path":"intake.sqlite3"}` instead. The SQLite file has
one table with the exact current shape:

```sql
CREATE TABLE cases (
    case_id TEXT PRIMARY KEY,
    public_index INTEGER NOT NULL UNIQUE,
    payload_json TEXT NOT NULL
);
```

`payload_json` is the same Case object used by JSONL;
`cases.case_id == payload_json.id` and the row `public_index` must equal any
index inside the payload. Use exactly one declared source format.

```bash
hotpot ds bundle build submission
hotpot ds bundle --root submission status | jq '.payload'
```

Common pitfalls:

- Build is one-time. It rejects a root that already contains canonical Bundle
  controls instead of rebuilding or silently upgrading it.
- JSONL blank lines, duplicate Case/local IDs, duplicate or negative public
  indexes, malformed UTF-8, and operative input fields such as
  `quality_level`, `evidence_level`, `archive_class`, or `benchmark` are
  rejected.
- Paths must use safe POSIX root-relative spelling. Absolute paths, `..`, and
  backslashes are rejected. Keep referenced files inside the Bundle root.
- A missing referenced scientific file is recorded as a negative fact; it is
  not downloaded or fabricated. Missing required structures can make a Case
  incomplete.
- `--dataset` is only an assertion against `bundle.json.ds_name`; it cannot
  rename the dataset. Likewise, `--db` binds the target store and is not the
  location of Bundle working files.
- After build, use `bundle status` to discover the next legal transition. Do
  not infer synchronization from the presence of generated directories.

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

Use `info` for immutable Bundle identity and policy-facing metadata, `summary`
for synchronized counts, `cases` for a filtered collection, and `case.info` or
`case.assessment` for one Case:

```bash
hotpot ds bundle --root submission get info | jq '.payload'
hotpot ds bundle --root submission get summary | jq '.payload'
hotpot ds bundle --root submission get cases --archive-class incomplete
hotpot ds bundle --root submission get case.info case-0001
```

Common pitfalls:

- `summary`, derived Case filters, and assessments require a synchronized
  Bundle. Use `bundle status` while facts are dirty.
- Exact `--quality B2` differs from `--min-quality B2`; the former selects one
  level and the latter selects B2 and higher.
- Repeat one filter for IN semantics. Filters of different kinds are ANDed.
- `case.info` and `case.assessment` require `CASE_ID`; Bundle-level views do
  not accept one.
- These views inspect a draft or sealed Bundle. To query the formal ingested
  release, use `hotpot ds info`, `get`, `subset`, or `search`.

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

Every mutation file includes the Case ID plus exactly the named arguments of
the selected method. For example, a core-info correction is:

```json
{
  "case_id": "case-0001",
  "name": "phase",
  "value": "solid",
  "reason": "Confirmed against the primary report"
}
```

```bash
hotpot ds bundle --root submission \
  set case.set_core_info set-phase.json
hotpot ds bundle --root submission status
hotpot ds bundle --root submission synchronize
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

Common pitfalls:

- `JSON_FILE` is a normal file path, not raw JSON and not protocol stdin.
- One invocation performs one Case transaction. It does not synchronize or
  regenerate a certificate.
- Do not edit `cases/facts.sqlite3` or nested objects returned by the Python
  API. Such edits bypass the audit contract and will fail integrity checks.
- Quality and evidence levels are derived and have no setters. Add or verify
  evidence, correct facts, or change a new-dataset Gate, then synchronize.
- `Hotpot` authority cannot be self-declared through the public CLI. Use
  `User` for an attributable owner approval or `Other` for a preserved external
  expert statement.

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

Inspect before editing:

```bash
hotpot ds bundle --root submission gate quality --show
hotpot ds bundle --root submission gate evidence --show
hotpot ds bundle --root submission gate benchmark --show
```

For a new dataset, one atomic edit can add a check and bind its implementation:

```bash
hotpot ds bundle --root submission gate evidence \
  --add-check supplier_verified \
  --checker supplier_verified=checkers.py:check_supplier \
  --set-level E2=extracted_material,explicit_report,usable_locator,supplier_verified
```

The checker is a deterministic top-level function with the fixed signature:

```python
def check_supplier(case, bundle):
    return (case["metadata"].get("supplier_verified") is True,
            "supplier verification is required")
```

Hotpot persists the exact checker source and digest. The callable must accept
exactly `(case, bundle)`, return `(passed: bool, reason: str)`, avoid hidden
state and side effects, use only allowed imports, and run successfully in the
isolated checker worker.

Common pitfalls:

- Gate edits are allowed only before publication for a new dataset. An
  existing dataset always reuses its persisted policy.
- `--show` cannot be combined with edits. Show the current Gate first, then
  issue a separate atomic edit.
- `--checker NAME=FILE:ENTRYPOINT` supplies the implementation for a
  same-named item added by that invocation; it does not add a check by itself.
- Gate changes make assessments and certificates stale. Run `synchronize`,
  inspect the results, and write a new certificate.
- A custom checker is governance code, not a convenient data loader. Network,
  filesystem side effects, closures, and unsupported imports are rejected.

### Synchronize

```bash
hotpot ds bundle synchronize
```

Synchronization evaluates all Cases and atomically publishes a new immutable
assessment generation. It does not write a certificate. Users can make several
fact or Gate edits, then synchronize once.

```bash
hotpot ds bundle --root submission status
hotpot ds bundle --root submission synchronize
hotpot ds bundle --root submission get summary
```

Common pitfalls:

- Synchronization is explicit and full-dataset. Neither reads nor certificate
  creation silently trigger it.
- Changing a referenced structure/evidence file, a Case fact, or a Gate makes
  the Bundle unsynchronized and any prior certificate stale.
- Synchronization can classify Cases as incomplete; it is not proof that every
  Case became ordinary or benchmark. Inspect `summary`, incomplete Cases, and
  their reason codes before certifying.
- Runtime-policy drift is not repaired implicitly. Follow the `next_action`
  reported by `bundle status`.

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

```bash
cd submission
hotpot ds bundle status
hotpot ds bundle rm --all
```

Common pitfalls:

- This command is for a sealed and successfully ingested Bundle only. It is
  not a way to discard a failed draft.
- `--all` is mandatory and deletion is irreversible. Original intake,
  scientific evidence, certificate, receipt, policy, and audit history remain.
- If the Bundle target differs from the default imported-package store, pass
  the exact bound `--db` as an assertion.
- After successful cleanup the Bundle is intentionally not an editable live
  working Bundle. Query its release from the formal dataset store.

## 3. Certify and ingest

### Certify

```text
hotpot ds certify BUNDLE_ROOT [--dataset NAME] [--db DB]
```

Certification requires a synchronized Bundle and writes exactly
`BUNDLE_ROOT/certificate.json`. It verifies and serializes the current facts,
policy, scientific-asset inventory, governance snapshot, and assessment
generation. It neither evaluates Gates nor ingests data.

```bash
hotpot ds bundle --root submission status
hotpot ds certify submission
jq '.certificate.dataset, .certificate.release.release_id, .envelope_sha256' \
  submission/certificate.json
```

Common pitfalls:

- Certification rejects an unsynchronized Bundle and never synchronizes it
  implicitly. Run `bundle synchronize` explicitly first.
- Any later Case, Gate, intake, or referenced-asset change makes the
  certificate stale. Synchronize and certify again before ingestion.
- The certificate belongs at the Bundle root and remains bound to that exact
  release, policy, assessment generation, and source content. Do not copy one
  from another Bundle.
- `--dataset` and `--db` are matching assertions for the identity and target
  already bound by the Bundle; they do not rewrite those bindings.

### Ingest

```text
hotpot ds ingest BUNDLE_ROOT [--db DB] [--dataset NAME]
                 [--include-incomplete --reason TEXT]
```

Ingestion requires the current root certificate and rechecks the Bundle,
generation, target, and policy inside the target transaction. It copies
materialized scientific assets into the target-owned content-addressed asset
area, records one authoritative receipt, and seals the release. `--db` selects
an explicit target. If omitted, the target is
`<imported-hotpot-package>/dataset/data/store/active/datasets.sqlite3`. An
editable installation therefore uses its source checkout, while a regular pip
installation uses its installed package directory. A non-writable package
directory is an error; Hotpot never silently redirects publication elsewhere.

Incomplete Cases are excluded unless both `--include-incomplete` and a
non-empty `--reason` are supplied:

```bash
hotpot ds ingest submission --db work/datasets.sqlite3 \
  --include-incomplete --reason 'Preserve supplier records pending repair'
```

An exact retry returns the existing receipt. Reusing the same release identity
with different content, policy, certificate, or selected Case sets is an error.

The most direct publication command, after build, synchronization, and
certification have succeeded, is:

```bash
hotpot ds ingest submission
hotpot ds info spin
```

To publish outside the imported package, bind the same target from build
through ingestion:

```bash
hotpot ds bundle build submission --db /srv/hotpot/datasets.sqlite3
hotpot ds bundle --root submission synchronize
hotpot ds certify submission --db /srv/hotpot/datasets.sqlite3
hotpot ds ingest submission --db /srv/hotpot/datasets.sqlite3
```

Common pitfalls:

- Ingest never builds, synchronizes, or certifies. A current root certificate
  is mandatory.
- The omitted `--db` target follows the Hotpot package actually imported by
  the running executable, not necessarily the source tree from which the
  command was launched.
- Do not pass the Bundle directory to `--db`; pass the target
  `datasets.sqlite3` file.
- Incomplete Cases require both `--include-incomplete` and an explicit
  non-empty `--reason`. They remain isolated and are still excluded by normal
  queries.
- A release is immutable once sealed. Correct published facts by preparing a
  successor release rather than modifying the target database.

### Copy an ingested store

```text
hotpot ds copy TARGET_HOTPOT_PACKAGE [--db SOURCE_DB]
```

`copy` replicates the complete already-ingested store used by the running
Hotpot package into another concrete `hotpot/` package directory. The source
defaults to the imported package store; `--db` selects a different source
store. The target argument is the package directory itself, not a project root
or a database filename. Its `hotpot.ds` runtime must exactly match the running
command; copying into an old, newer, partial, or otherwise different package is
rejected instead of invoking a compatibility path.

The command takes a consistent SQLite snapshot, copies exactly the referenced
content-addressed assets while verifying their sizes and SHA-256 digests,
recomputes both receipt-bound persisted roots, and audits the staged store
before atomically activating one target generation. Those roots cover Case
facts, assessments, identifiers, dataset policy and metadata, release metadata,
and the exact structure, scientific, and release-control asset-reference
relationships. It
does not rebuild, recertify, re-ingest, merge, or overwrite. An existing target
`store/active` is rejected. Historical receipt paths remain the signed facts
from original ingestion; ordinary reads resolve assets from the replica's
current database location. Both replicas retain the same store UUID and must
not be developed independently as competing publication authorities.

For example, copy the store from the Hotpot package imported by the current
command into another checkout containing the exact same Hotpot runtime:

```bash
hotpot ds copy /home/me/src/hotpot/hotpot
```

Or select an explicit source store:

```bash
hotpot ds copy /home/me/src/hotpot/hotpot \
  --db /opt/venv/lib/python3.12/site-packages/hotpot/dataset/data/store/active/datasets.sqlite3
```

The argument is the target package directory containing `__init__.py`, `ds/`,
and `dataset/`; it is not `/home/me/src/hotpot`, the target `dataset/`
directory, or a destination database path.

Common pitfalls:

- The target runtime must match exactly and its `store/active` location must
  not already exist. `copy` deliberately does not merge or overwrite stores.
- Copying into the same imported package is not a backup operation. Choose a
  distinct target installation.
- A successful copy creates a directly queryable replica; do not run build,
  certify, or ingest again for those releases.
- The replicas share one store UUID. Choose one publication authority instead
  of allowing both copies to diverge independently.
- Historical canonical paths in signed receipts remain unchanged. They are
  provenance facts, while normal reads resolve assets from the replica.

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
| `hotpot.ds.copy` | `default -> store_copy_result` |

The machine-readable authority is `hotpot/ds/cli_communication.json`; the
repository protocol specification is
[`skills/hotpot-cli-stdio-protocol/references/protocol.md`](../../skills/hotpot-cli-stdio-protocol/references/protocol.md).

## 5. Bundle intake layout

### Choose exactly one Case source

A Bundle intake is one directory for one dataset release. The recommended
layout is:

```text
BUNDLE_ROOT/
|-- bundle.json
|-- cases.jsonl                 # or intake.sqlite3, never both as active input
|-- structures/
|   `-- AAAKSI.mol2
`-- evidence/
    `-- AAAKSI-paper.pdf
```

Files may be arranged in deeper subdirectories. The manifest's `intake.path`
selects exactly one Case source; undeclared JSONL or SQLite files are not
combined automatically. A bare database or a directory of structures is not a
Bundle.

### `bundle.json`

The current packaged template is
[`templates/bundle.json`](templates/bundle.json). The required fields are:

| Field | Purpose |
| --- | --- |
| `schema` | Must be exactly `hotpot-ds-bundle/current`. |
| `ds_name` | Stable public dataset name. |
| `release_id` | Immutable identity for this release. |
| `version` | Human/source version string. |
| `title`, `description` | Non-empty release description. |
| `license`, `access_policy` | Usage and access declarations. |
| `provenance` | JSON source/provenance facts; may be `{}`. |
| `core_labels` | Observation/claim properties every complete Case is expected to provide; may be `[]`. |
| `core_info` | Case information keys required for completeness; may be `[]`. |
| `intake` | Exactly `format` (`jsonl` or `sqlite`) and safe relative `path`. |
| `identity_schema` | Explicit components used to decide whether Cases describe the same scientific sample. |
| `required_structure_roles` | Required valid structure roles; may be `[]`, or include `"*"` to require any valid structure. |

`dataset_id` may be supplied but must equal `ds_name`. `release_serial` is an
optional integer from 1 through 999. `property_definitions` and
`property_comparators` make property type, unit, context, and conflict rules
explicit. `control_assets` binds non-scientific release controls by safe path,
SHA-256, and byte count.

The packaged template shows the current representation:

```json
{
  "schema": "hotpot-ds-bundle/current",
  "ds_name": "metal_oxidation",
  "dataset_id": "metal_oxidation",
  "release_id": "metal_oxidation-r1",
  "release_serial": 1,
  "version": "1.0",
  "title": "Metal oxidation states",
  "description": "One immutable reviewed release.",
  "license": "CC-BY-4.0",
  "access_policy": "open",
  "provenance": {"created_by": "laboratory curator"},
  "core_labels": ["oxidation_state"],
  "core_info": ["sample_id"],
  "intake": {"format": "jsonl", "path": "cases.jsonl"},
  "identity_schema": {
    "components": [
      {
        "key": "sample_id",
        "path": "core_info.sample_id",
        "required": true,
        "normalizer": "text"
      }
    ]
  },
  "property_definitions": {
    "oxidation_state": {
      "type": "integer",
      "schema_version": "current",
      "unit": null,
      "context_fields": ["subject", "site", "phase"]
    }
  },
  "property_comparators": {
    "oxidation_state": {"kind": "exact"}
  },
  "required_structure_roles": ["reported_structure"]
}
```

Identity fields describe the sample and must not include the target result,
B/E level, archive class, or benchmark outcome. Otherwise contradictory
measurements could be incorrectly split into different samples and evade
conflict assessment.

### Case objects and relationships

The packaged current Case template is
[`templates/case.json`](templates/case.json). A Case requires a stable `id`
and `local_id`. A JSONL Case may omit `public_index`, in which case its
zero-based line position is used; declaring it explicitly is recommended.
`identifiers`, `metadata`, `core_info`, `entities`, `structures`,
`observations`, `claims`, `evidence`, `verifications`, and `attestations`
default to empty objects or arrays.

The following current-format Case uses `structures[].representations` rather
than placing a legacy format/path pair directly on the Structure:

```json
{
  "id": "case-AAAKSI",
  "local_id": "AAAKSI",
  "public_index": 0,
  "identifiers": {"CSD": "AAAKSI"},
  "metadata": {"supplier": "example laboratory"},
  "core_info": {"sample_id": "AAAKSI"},
  "entities": [
    {"entity_id": "entity-AAAKSI-metal", "entity_kind": "metal_site"}
  ],
  "structures": [
    {
      "structure_id": "structure-AAAKSI",
      "role": "reported_structure",
      "primary": true,
      "representations": {
        "mol2": {"path": "structures/AAAKSI.mol2"}
      }
    }
  ],
  "observations": [
    {
      "observation_id": "observation-AAAKSI-oxidation",
      "property": "oxidation_state",
      "schema_version": "current",
      "value": 2,
      "unit": null,
      "subject": "entity-AAAKSI-metal",
      "site": "metal-1",
      "phase": "solid"
    }
  ],
  "claims": [],
  "evidence": [
    {
      "evidence_id": "evidence-AAAKSI-paper",
      "target": "observation-AAAKSI-oxidation",
      "source": "https://doi.org/10.0000/replace-me",
      "source_kind": "primary_literature",
      "relation": "supports",
      "locator": "Table 1",
      "status": "active",
      "path": "evidence/AAAKSI-paper.pdf"
    }
  ],
  "verifications": [],
  "attestations": []
}
```

IDs, rather than array positions, express relationships:

- `observation.subject` can identify an entry in `entities`;
- an Evidence `target` identifies an Observation or Claim;
- a Verification identifies the Evidence and its attributable review method;
- an Attestation records an attributable benchmark opinion but never supplies
  operative B/E or archive fields.

Input must not assign `quality_level`, `evidence_level`, `archive_class`, or
`benchmark`. Synchronization derives these values from the facts and persisted
Gates. Supplier-provided classifications may be preserved only in the
explicit `claimed_*` fields and are not operative assessments.

### Structure and Evidence assets

Each Structure has a stable `structure_id`, a scientific `role`, and a
`representations` object such as `smi`, `mol2`, or `xyz`. Each representation
with local bytes uses `{"path":"structures/file.ext"}`. Evidence may likewise
use a safe Bundle-relative `path`, or declared nested assets where applicable.
Paths are data relationships: file names need not equal Case IDs, and one
content-identical asset may be referenced by more than one Case.

All paths must be POSIX-style and relative to `BUNDLE_ROOT`; absolute paths,
backslashes, and parent traversal are invalid. Present paths must identify
regular files. A declared but absent scientific asset is retained as a
negative fact and included in synchronization/certification instead of being
silently fetched or fabricated. A missing or invalid required structure can
make the Case incomplete. Missing or hash-invalid optional Evidence cannot
support its target and normally lowers evidence support rather than by itself
making the Case incomplete.

### JSONL and SQLite forms

For JSONL, write exactly one UTF-8 JSON object per physical line with no blank
lines. Case IDs, local IDs, and public indexes must be unique. This form is
convenient for review and versioned preparation.

For large intakes, set:

```text
"intake": {"format": "sqlite", "path": "intake.sqlite3"}
```

The SQLite file must contain exactly the current intake table columns:

```sql
CREATE TABLE cases (
    case_id TEXT PRIMARY KEY,
    public_index INTEGER NOT NULL UNIQUE,
    payload_json TEXT NOT NULL
);
```

`payload_json` contains the same Case object as one JSONL line. `case_id` must
equal `payload_json.id`; the row index must be non-negative and match a
declared payload index. The SQLite intake is not the formal target store and
must not use the formal `datasets.sqlite3` schema.

### Files created by the lifecycle

After `bundle build`, the root also contains canonical facts and policy
projections. `bundle synchronize` adds immutable generation artifacts and
derived class indexes. `certify` adds the root `certificate.json`. Users do not
edit generated control or assessment files directly.

Common pitfalls:

- Do not place multiple releases in one root or switch `intake.path` after
  build.
- Do not use an observation value, benchmark decision, or assessment level as
  sample identity.
- Do not confuse the intake SQLite file with the ingestion target selected by
  `--db`.
- Do not put structure `format`/`path` directly on new Structure records; use
  the current `representations` mapping shown above.
- Do not treat equal coordinate files as proof of equal scientific samples.
  Asset deduplication and sample identity are separate decisions.

See the [Python API reference](README.md) and
[storage contract](../dataset/docs/storage.md) for the complete ownership and
cutover rules.

## 6. Troubleshooting

### “The command found no datasets”

The executable may be importing a different Hotpot installation than the one
that owns the store. Inspect the current default and compare it with the
ingestion receipt:

```bash
hotpot ds detail status
python -c 'import hotpot; print(hotpot.__file__)'
```

Use the same environment, pass the intended `--db`, or use `hotpot ds copy`
to install a verified replica into the target package.

### “The Bundle is unsynchronized” or “certificate is stale”

Fact, asset, or Gate changes invalidate derived assessments. Inspect status,
synchronize explicitly, review the summary, then replace the certificate:

```bash
hotpot ds bundle --root submission status
hotpot ds bundle --root submission synchronize
hotpot ds bundle --root submission get summary
hotpot ds certify submission
```

Do not work around this by editing generated assessment or certificate files.

### “bundle build says the root is already built”

Build is a one-time intake transition. Continue with `bundle status` and its
reported next action. To publish corrected scientific content after a sealed
release, create a successor release with a new root and release identity; do
not rebuild or mutate the sealed release.

### “A Case became incomplete”

Query the assessment reasons rather than assigning a level manually:

```bash
hotpot ds bundle --root submission get cases --archive-class incomplete
hotpot ds bundle --root submission get case.assessment CASE_ID
hotpot ds bundle --root submission get case.info CASE_ID
```

Typical causes are missing `core_labels` or `core_info`, absent/invalid
required structures, malformed Evidence facts, identity indeterminacy, or a
significant unresolved scientific conflict. Correct the source fact through an
audited mutation or explicitly archive incomplete Cases during ingestion.

### “SMILES/XYZ display or export conversion failed”

First inspect the stored Structure inventory without requesting conversion:

```bash
hotpot ds get DATASET INDEX
```

For multi-structure Cases, add exactly one of `--structure-index`,
`--structure-role`, or `--structure-id`. Conversion requires a parseable
molecular representation; crystal-only, missing, or chemically invalid input
is reported as an error rather than silently substituted.

### “The target package is not writable”

Regular package installations may be read-only for the current user. Ingest
into an explicitly writable store and pass that same target throughout the
Bundle lifecycle:

```bash
hotpot ds bundle build submission --db /srv/hotpot/datasets.sqlite3
hotpot ds bundle --root submission synchronize
hotpot ds certify submission --db /srv/hotpot/datasets.sqlite3
hotpot ds ingest submission --db /srv/hotpot/datasets.sqlite3
```

Hotpot does not silently redirect ingestion because doing so would obscure the
publication authority.

### “A pipeline was rejected”

Only a literal positional `-` enables protocol stdin, and input must be one
current four-field envelope from a declared producer/channel. Dataset,
release, store, and archive visibility must agree downstream. Do not pipe raw
JSONL, a `jq`-modified envelope, or human `--doc` output into a DS command.
