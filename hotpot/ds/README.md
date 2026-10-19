# `hotpot.ds` public API

`hotpot.ds` is Hotpot's concise interface for governed scientific datasets. It
separates immutable dataset queries from an explicit Bundle workflow for
assessment, certification, and ingestion. Benchmark membership is a derived
state within a dataset, not a separate database.

Hotpot reports what the available evidence supports; it does not overwrite a
user's scientific statement or require benchmark-grade evidence merely to
preserve a record. Unsupported statements remain attributable facts, while
Gate-derived quality, evidence, and benchmark results remain separate.

The first-level namespace is intentionally closed and ordered as follows:

```python
hotpot.ds.__all__ == [
    "datasets_list", "load",
    "Dataset", "Data",
    "BaseSearcher", "DataSearcher", "DatasetSearcher", "SmartsSearcher",
    "load_bundle", "Bundle", "write_certificate",
]
```

Backend rows, certificates, assessment generations, stores, Cases, and Gate
classes are not additional first-level APIs. Gate configuration is available
only from `hotpot.ds.gates`.

## 1. Query datasets

### `datasets_list`

```python
import hotpot.ds as ds

print(ds.datasets_list)
names = list(ds.datasets_list)
other_names = ds.datasets_list(db="work/datasets.sqlite3")
```

`datasets_list` is an iterable view of the default store and is callable for an
explicit store:

```text
list(ds.datasets_list) -> list[str]
ds.datasets_list(*, db: str | Path | None = None) -> list[str]
```

### `load`

```python
dataset = ds.load("oxidation")
release = ds.load("oxidation", version="001", db="work/datasets.sqlite3")
incomplete = ds.load("oxidation", archive_class="incomplete")
mixed = ds.load(
    "oxidation",
    archive_class={"incomplete", "ordinary", "benchmark"},
)
```

```text
ds.load(
    ds_name: str,
    version: str | int | None = None,
    *,
    db: str | Path | None = None,
    archive_class: str | Iterable[str] | None = None,
) -> Dataset
```

`version` accepts a source version, immutable release ID, integer release
serial, or three-digit serial. By default, `load` returns the latest release
and includes `ordinary` and `benchmark` records. Incomplete records require an
explicit `archive_class` opt-in. A filtered Dataset cannot reintroduce archive
classes excluded by its original `load`.

The default local store is
`<imported-hotpot-package>/dataset/data/store/active/datasets.sqlite3`. It is
derived from the package actually imported by the process, not from the path
of the console-script wrapper. An editable installation therefore selects its
source checkout, while an ordinary pip installation selects its installed
package directory. Dataset payloads are not bundled with source distributions,
so a clean installation commonly needs `db=` until a release is ingested or an
existing formal store is replicated there with `hotpot ds copy`.

## 2. `Dataset` and `Data`

`Dataset` is a sequence-like view over one immutable release. Users obtain it
through `load`, slicing, searching, or `subset`; its constructor is not an
intake interface.

```python
first = dataset[0]                 # Data
window = dataset[10:20]           # Dataset
selected = dataset.subset(
    labels__oxidation_state={2, 3},
    quality_level={"B2", "B3", "B4"},
)
by_refcode = dataset.CSD_identifier["AAAKSI"]
```

Scalar criteria use equality. Lists, sets, tuples, and frozen sets use IN
semantics; all criteria are ANDed. A dataset-specific identifier lookup returns
one `Data` when unique and a `Dataset` when repeated.

### Dataset members

```text
len(dataset) -> int
iter(dataset) -> Iterator[Data]
dataset[index: int] -> Data
dataset[index: slice] -> Dataset
dataset.subset(**criteria) -> Dataset
dataset.<namespace>_identifier[value] -> Data | Dataset

dataset.name -> str
dataset.dataset_id -> str
dataset.release_id -> str
dataset.version -> str
dataset.version_serial -> int
dataset.info -> dict
dataset.handle -> list[str]
dataset.labels -> dict[str, int]
dataset.core_labels -> list[str]
dataset.quality_levels -> dict[str, int]
dataset.evidence_levels -> dict[str, int]
dataset.identifier_namespaces -> tuple[str, ...]
```

`dataset.handle` collects the release's DOI, URL, citation, and repository
handles. `dataset.labels` reports how many visible records contain each label.
`dataset.core_labels` may be empty. Quality and evidence on a Dataset are
distributions; a heterogeneous collection has no single B or E level. Cases
without an evidence-bearing required target have `evidence_level is None` and
are counted in the `not_applicable` distribution bucket.

### Data members

```text
data.dataset -> Dataset
data.id -> str
data.record_id -> str
data.local_id -> str
data.metadata -> dict
data.info -> dict
data.label -> dict
data.identifiers -> dict
data.entities -> list[dict]
data.quality_level -> str
data.evidence_level -> str | None
data.archive_class -> str
data.structures() -> list[str]
data.structures.<format> -> str | None
data.structures.content(format) -> str | None
data.structures.path(format) -> Path | None
data.structures.molecule() -> hotpot.Molecule
data.molecule -> hotpot.Molecule
data.crystal -> Crystal
```

Hotpot IDs use
`hp-<dataset_name>[<three-digit-release-serial>]-<persistent-index>`, for
example `hp-oxidation[001]-0`. Slicing and filtering never renumber them.

`data.structures()` lists available representations. Attribute access such as
`data.structures.smi` and `data.structures.mol2` returns verified content or
`None`; it does not raise merely because that format is absent.
If a Data item contains several Structure IDs, the accessor uses the unique
`primary` Structure; no unique primary raises an ambiguity error instead of
silently choosing a variant.

`data.info["evidence_assets"]` lists materialized evidence files by Evidence
ID, role, status, SHA-256, size, and target-owned CAS path. The original
Evidence declaration remains unchanged under `data.info["facts"]["evidence"]`;
the CAS reference is the durable byte location after ingestion.

`data.molecule` returns a fresh `hotpot.Molecule` and attaches the originating
Data as `.data`, or `.ds_data` if the object already owns `.data`.
`data.crystal` requires crystallographic information and otherwise raises.

## 3. Search

```python
query = (
    ds.DataSearcher(labels__oxidation_state={2, 3})
    & ~ds.DataSearcher(archive_class="incomplete")
)
matches = query.search(dataset)

dataset_query = ds.DatasetSearcher(core_labels="oxidation_state")
matched_datasets = dataset_query.search(ds.datasets_list)

substructure_query = ds.SmartsSearcher("[N]-[C]")
matched_structures = substructure_query.search(dataset)
```

```text
BaseSearcher                         # abstract
  .match(item) -> bool
  .search(target) -> Dataset | tuple
  .not_() -> BaseSearcher
  __call__(item) -> bool
  &, |, ^, ~

DataSearcher(**criteria)
  .criteria -> Mapping[str, Any]
DatasetSearcher(**criteria)
  .criteria -> Mapping[str, Any]
SmartsSearcher(smarts: str, *, semantics=None)
```

`&`, `|`, `^`, and `~` mean AND, OR, XOR, and NOT. Python has no overloadable
unary `!`; use `~query` or `query.not_()`. Do not use Python `and`, `or`, or
`not` with search expressions.

## 4. Build and manage a Bundle

A Bundle owns exactly one dataset release in exactly one directory. The normal
workflow is explicit:

```python
import hotpot.ds as ds

# One-time canonical build from an unbuilt current-format intake folder.
bundle = ds.Bundle(
    "submission",
    "my-dataset",
    db="work/datasets.sqlite3",
)

# Batch edits are allowed before one explicit reassessment.
bundle.synchronize()
certificate = bundle.write_certificate()
receipt = bundle.ingest()
```

Later processes reopen, rather than rebuild, the canonical Bundle:

```python
bundle = ds.load_bundle(
    "submission",
    "my-dataset",
    db="work/datasets.sqlite3",
)

restored = ds.Bundle.from_certificate(
    "submission/certificate.json",
    db="work/datasets.sqlite3",
)
```

```text
ds.Bundle(root, ds_name, *, db=None, gates=None)
ds.load_bundle(root, ds_name, *, db=None) -> Bundle
ds.Bundle.from_certificate(path, *, db=None) -> Bundle
ds.write_certificate(bundle) -> Path
```

`root` is the one Bundle directory; `ds_name` must equal
`bundle.json.ds_name`; `db` is the target current store used to decide whether
the dataset is new or established. `gates` accepts at most one of each Gate
type and is valid only for a new dataset. `from_certificate` reopens the Bundle
named by the root certificate and verifies its controls and materialized
assets. A certified missing/invalid asset remains a bound negative fact; it is
not treated as an unexpectedly missing control file.

`Bundle(...)` is the only public one-time build entry. `load_bundle(...)` is
validation-only and never imports, repairs, or rewrites facts. The thin module
function `write_certificate(bundle)` calls `bundle.write_certificate()` and
contains no assessment policy.

### Bundle members

```text
len(bundle) -> int
bundle.root -> Path
bundle.name -> str
bundle.mode -> "new" | "existing"
bundle.info -> dict
bundle.status -> dict
bundle.is_synchronized -> bool
bundle.summary -> dict
bundle.gates -> Mapping[str, Gate]
bundle.certificate_path -> Path
bundle.certificate_stale -> bool
bundle.cases -> CaseCollection
bundle.incomplete -> CaseView
bundle.ordinary -> CaseView
bundle.benchmark -> CaseView
bundle.use_gate(gate) -> None
bundle.synchronize() -> dict
bundle.write_certificate() -> Path
bundle.ingest(*, include_incomplete=False, reason=None) -> dict
```

`bundle.status` and raw facts remain readable while the Bundle is stale.
Derived levels, classifications, summaries, and derived views require a
current synchronized generation. `bundle.summary` has only the top-level
`cases`, `integrity`, `completeness`, `evidence`, `quality`, `benchmark`,
`attestation`, `conflict`, and `reasons` entries. Each named section contains
its deterministic status, level, archive-class, or reason-code distribution;
there are no duplicate top-level distribution aliases.
The lifecycle has no implicit transitions:

```text
fact or new-dataset Gate edit
    -> unsynchronized, certificate stale
bundle.synchronize()
    -> synchronized, certificate stale
bundle.write_certificate()
    -> synchronized, certificate current
bundle.ingest()
    -> immutable sealed release
```

Certification never synchronizes, and ingestion never synchronizes or
certifies. Successful ingestion seals the whole release. Later scientific
corrections require a successor release.

For a new dataset, omitted Gates use Hotpot defaults and supplied Gates become
the proposed persistent dataset policy. An existing dataset loads its stored
policy: passing `gates` or calling `use_gate` is an error.

### Case access and editing

`bundle.cases` contains every Case, including incomplete Cases, and remains the
single collection entry point:

```python
case = bundle.cases[0]
candidate = bundle.cases.select(
    archive_class={"ordinary", "benchmark"},
    min_quality="B2",
    min_evidence="E2",
)
```

```text
len(bundle.cases) -> int
iter(bundle.cases) -> Iterator[Case]
bundle.cases[index: int] -> Case
bundle.cases[index: slice] -> CaseView
bundle.cases.select(**criteria) -> CaseView
case.id -> str
```

Selection accepts only `archive_class`, `quality_level`, `min_quality`,
`evidence_level`, `min_evidence`, `benchmark_status`, `reason`, `verification`,
and `attestation`. Views return the original Bundle-owned Case objects; Cases
cannot be appended, removed, reparented, or moved between Bundles.

Raw Case properties include `info`, `core_info`, `structures`, `labels`,
`observations`, `claims`, `evidence`, and `attestations`. Derived read-only
properties include `evidence_levels`, `evidence_level`, `quality_level`,
`archive_class`, `benchmark`, `benchmark_status`, `reasons`, and `assessment`.
All mapping and sequence results are defensive snapshots.

Facts change only through named audited methods:

```text
case.set_core_info(name, value, *, reason=None) -> None
case.remove_core_info(name, *, reason) -> None

case.add_structure(data) -> str
case.replace_structure(structure_id, data, *, reason) -> str
case.withdraw_structure(structure_id, *, reason) -> None

case.add_observation(data) -> str
case.replace_observation(observation_id, data, *, reason) -> str
case.withdraw_observation(observation_id, *, reason) -> None

case.add_claim(data) -> str
case.replace_claim(claim_id, data, *, reason) -> str
case.withdraw_claim(claim_id, *, reason) -> None

case.add_evidence(
    target, source, *, relation="supports", locator=None, note=None, details=None
) -> str
case.replace_evidence(
    evidence_id, source, *, reason, locator=None, note=None, details=None
) -> str
case.withdraw_evidence(evidence_id, *, reason) -> None
case.verify_evidence(
    evidence_id, *, method, verifier, note=None, details=None
) -> str
case.withdraw_verification(verification_id, *, reason) -> None

case.attest_benchmark(
    approved, *, authority, actor, reason=None
) -> str
case.withdraw_attestation(attestation_id, *, reason) -> None
```

`data` arguments above are JSON-compatible mappings. `method` is `"AI"` or
`"Human"`; public `authority` is `"User"` or `"Other"`. Add and replacement
methods return the new fact ID. Replacement and withdrawal preserve the old
fact and append an audit event.

```python
evidence_id = case.add_evidence(
    target="observation-1",
    source="doi:10.1000/example",
    locator="Table 2",
    details={
        "source_kind": "primary_literature",
        "source_lineage_id": "experiment:example",
    },
)
case.verify_evidence(
    evidence_id,
    method="Human",
    verifier="reviewer@example.org",
)
bundle.synchronize()
```

`source` accepts `doi:...`, `url:...`, `file:...`, and
`source:<existing-source-id>` references. DOI and URL identities are extracted
from the first two forms. `details` is a small closed provenance object so the
Gate can use, rather than merely store, the facts required for E3/E4:

```text
Evidence details:     doi, url, scientific_work_id, source_lineage_id,
                      derived_from, source_kind
Verification details: tool, tool_version, procedure_version, verified_at
```

`source_kind` uses the closed current vocabulary `primary_literature`,
`secondary_literature`, `database_record`, `expert_statement`, or `other`.
Only an explicit `primary_literature` value can satisfy primary-source checks;
an `http://` or `https://` URL never implies primary literature by itself.
Existing evidence without this fact remains usable at lower evidence levels
but is not silently upgraded during migration.

`derived_from` is a JSON array of result-lineage IDs. Two E3 evidence chains
can reach E4 only when both their canonical scientific works and independent
lineage roots differ. Verification automatically records the current Hotpot
procedure and a UTC timestamp when those fields are omitted; callers may
supply the actual procedure and timezone-aware timestamp for an earlier review.
For AI review, `verifier` identifies the verifying actor or service, while
`tool` names the model/tool and `tool_version` records its explicit version.
Both AI fields are mandatory. Human review remains attributable through its
non-empty `verifier`. Unknown detail keys are rejected, and all accepted fields
remain source facts covered by the Bundle source digest, synchronization, and
certificate.

When Evidence declares a Bundle-local file, its generation AssetAssessment is
part of EvidenceGate input. Missing bytes or a declared SHA-256 mismatch make
that Evidence unusable for E1-E4 and are reported explicitly; the failed
optional evidence asset does not by itself make the Case incomplete.

Evidence and quality levels have no setters. A benchmark opinion uses an
attributable attestation, not `case.benchmark = True`. Public callers may attest
as `User` or `Other`; effective `Hotpot` authority comes only from Hotpot's
private signed governance path. Ed25519 verification uses Hotpot's required
`cryptography` runtime dependency; governance never selects an optional or
fallback signature backend.

## 5. Gates and levels

Gate classes live in their own submodule:

```python
from hotpot.ds.gates import BenchmarkGate, EvidenceGate, QualityGate
```

```text
EvidenceGate(checks=None, *, level_checks=None, require_locator=True)
QualityGate(checks=None, *, level_checks=None, structure_checks=None)
BenchmarkGate(
    checklist=None,
    direct_entry=None,
    *,
    vetoes=None,
    numeric_tolerances=None,
)
```

Each Gate exposes immutable JSON-compatible `.info` and a content `.digest`.
`checks`, `structure_checks`, `checklist`, and `direct_entry` map stable names
to a Boolean built-in switch or a checker function. `level_checks` maps each
`B0`-`B4` or `E0`-`E4` level to the required check names. `vetoes` names
BenchmarkGate checks that no admission route may bypass, and each numeric
tolerance supplies nonnegative `absolute` and/or `relative` values.
Custom checks use one fixed signature:

```python
def checker(case: Mapping, bundle: Mapping) -> tuple[bool, str]:
    """Return (passed, reason)."""
```

Hotpot persists the exact source and digest for a new dataset. Checkers must be
deterministic top-level functions, use only allowed imports, avoid network and
side effects, and execute through the isolated checker worker. Unsupported
isolation fails closed; there is no unsafe in-process fallback.

Production custom-checker execution requires Linux user, mount, PID, network,
IPC, and UTS namespaces provided by `bubblewrap` (`bwrap`). The worker sees an
isolated minimal root containing the read-only Python runtime and required
loader-library mounts, plus a private temporary filesystem; Bundle/store
roots, `/proc`, and `/dev` are absent. A fixed environment, resource limits,
private process tree, bounded JSON protocol, and restricted imports make the
execution contract deterministic and small. The startup probe verifies the
filesystem and no-route network boundary before any persisted checker runs;
any unavailable boundary or protocol failure aborts synchronization.

Every dataset policy contains one complete, self-digesting `runtime_policy`.
It binds the assessment source and store protocol, checker protocol, parser
selection and parser source/protocol/distribution identity, Python and SQLite,
and the relevant installed distributions. Parser-backed asset assessments
reference the selected entry in that frozen parser registry. Assessment
generations, certificates, and ingestion receipts bind the full document.

Frozen validation checks the document's closed shape and nested digests without
pretending that its runtime is the current process. This permits authenticating
a sealed release against the runtime it actually recorded; it does not load or
emulate old code. Any operation that executes assessment semantics--including
synchronization, certificate issuance, or ingestion--requires exact equality
with the one current `runtime_policy`. Drift makes an editable Bundle stale and
requires an explicit owner migration and reassessment. A sealed release remains
immutable, and scientific corrections use a successor release.

```python
from hotpot.ds.gates import BenchmarkGate

gate = BenchmarkGate(
    checklist={"verified_primary_source": True, "conflict_free": True},
    direct_entry={"hotpot_or_user_attestation": True},
    numeric_tolerances={"oxidation_state": {"absolute": 0.0}},
)
bundle = ds.Bundle("submission", "my-dataset", gates=[gate])
```

Gate arguments are accepted only while building a new dataset. A later
new-dataset policy edit constructs another immutable Gate and passes it to
`bundle.use_gate(gate)`. Existing datasets always use their persisted policy.

Quality and evidence are independent Case assessments:

| Quality | Meaning |
| --- | --- |
| `B0` | Governed collection only |
| `B1` | Provenance-complete candidate with audited mapping |
| `B2` | Item review and conflict adjudication |
| `B3` | Primary support and reproducible validation controls |
| `B4` | Independently corroborated reference material |

| Evidence | Meaning |
| --- | --- |
| `E0` | No usable linked support |
| `E1` | Extracted material, not yet explicit and verified |
| `E2` | Explicit source statement with a usable locator |
| `E3` | Verified primary-source support |
| `E4` | Independent primary result-lineage corroboration |

The default BenchmarkGate admits only complete, safely archivable Cases. It
requires its ordinary checklist or one active Hotpot/User approval, and always
applies enabled vetoes. An active Hotpot/User rejection blocks every default
route. `Other` preserves an expert statement but does not grant direct entry.
Scientific value conflict, identity indeterminacy, and attestation disagreement
remain separate reason categories.

The three archive classes are:

| Class | Meaning | Default query visibility |
| --- | --- | --- |
| `incomplete` | Missing required facts/assets, invalid required structure, or significant unresolved scientific conflict | Excluded |
| `ordinary` | Complete and archivable, but not admitted by BenchmarkGate | Included |
| `benchmark` | Complete, archivable, and admitted by BenchmarkGate | Included |

Incomplete Cases can be preserved deliberately with
`bundle.ingest(include_incomplete=True, reason="...")`; they remain classified
and isolated as incomplete.

### Scientific sample identity

The dataset's persisted `identity_schema` is the authority for cross-Case
sample identity. Its current format is a closed object containing a
`components` array. Each
component has a declared value type and normalizer and is compared exactly;
the allowed pairs are exactly `text/text`, `text/casefold`, `number/number`,
`boolean/boolean`, `set/set`, and `json/json` (written as
`value_kind/normalizer`);
paths address `core_info` or `identifiers`. A structure-based component must
also name one structure role and one semantic field such as molecular graph,
stereochemistry, isotope/protonation state, formal charge, or crystal context.
Raw coordinate bytes, representation paths, and asset digests are not sample
identity, and byte-identical coordinates prove only that an asset may be
reused.

Comparison is three-valued: matching normalized components establish `same`,
a decisive mismatch establishes `different`, and missing, invalid, ambiguous,
or non-decisive evidence yields `indeterminate`. The owner chooses the
components when creating the dataset; an established dataset reuses its
persisted identity policy.

## 6. Intake organization

A Bundle root contains one authoritative `bundle.json` and one Case source,
plus root-relative scientific assets. A concise JSON Lines intake is:

```text
submission/
|-- bundle.json
|-- cases.jsonl
|-- structures/
|   `-- record-0001.mol2
`-- evidence/
    `-- paper.pdf
```

A minimal manifest names every required contract explicitly, including empty
requirements:

```json
{
  "schema": "hotpot-ds-bundle/current",
  "ds_name": "metal-oxidation",
  "release_id": "metal-oxidation-r1",
  "release_serial": 1,
  "version": "1.0.0",
  "title": "Metal oxidation states",
  "description": "Reviewed oxidation-state records",
  "license": "project-specific",
  "access_policy": "local",
  "provenance": {"doi": "10.1000/example"},
  "core_labels": ["oxidation_state"],
  "core_info": ["CSD_identifier"],
  "identity_schema": {
    "components": [
      {
        "key": "csd_record",
        "path": "core_info.CSD_identifier",
        "required": true,
        "value_kind": "text",
        "normalizer": "casefold",
        "comparator": "exact"
      }
    ]
  },
  "required_structure_roles": ["complex"],
  "property_comparators": {},
  "intake": {"format": "jsonl", "path": "cases.jsonl"}
}
```

`required_structure_roles` may be empty, may name roles such as `"complex"`,
or may contain the reserved value `"*"`. The wildcard requires at least one
active Structure with a present, hash-valid, parse-valid representation without
requiring one exact scientific role. If wildcard and named roles are combined,
every named role is still required.

Each nonblank JSONL line is one Case fact object. Operative B/E levels and
archive classes are deliberately absent because synchronization derives them:

```json
{"id":"case-0001","local_id":"AAAKSI","public_index":0,"identifiers":{"CSD":"AAAKSI"},"core_info":{"CSD_identifier":"AAAKSI"},"entities":[{"entity_id":"entity-0001"}],"structures":[{"structure_id":"structure-0001","role":"complex","format":"mol2","path":"structures/case-0001.mol2"}],"observations":[{"observation_id":"observation-0001","property":"oxidation_state","value":2}],"claims":[],"evidence":[],"verifications":[],"attestations":[]}
```

The equivalent structured intake is one `intake.sqlite3` conforming to the
current intake schema. A bare SQLite path is not a Bundle; place it inside a
Bundle root and declare it in `bundle.json`. Do not provide both active JSONL
and SQLite Case sources.

`bundle.json` declares one dataset/release identity, release serial, required
core labels/info, required structure roles, identity schema, comparison policy,
and the active Case source. Every local path must be safe, POSIX-style, and
relative to the Bundle root. Case records connect structures, observations,
claims, evidence, verifications, attestations, identifiers, and entities by
explicit IDs. Missing scientific assets are declared as negative facts rather
than replaced with invented content.

`Bundle(...)` publishes canonical facts and policy projections under the same
root. Synchronization creates an immutable assessment generation with separate
derived indexes for incomplete, ordinary, and benchmark Cases. The root
certificate is always `certificate.json`.

After the release has an authoritative target receipt, run
`hotpot ds bundle rm --all` from the Bundle root to remove reproducible working
facts, assessment generations, caches, and declared migration products. The
command proves the compact formal-store facts and assets before deletion,
retains original intake/evidence and governance records, and writes
`bundle-cleanup.json`. A cleaned Bundle is deliberately not a live editable
Bundle; the installed `Dataset`/`Data` release remains fully readable from the
formal store.

Generation leases and garbage collection are private lifecycle details, not
additional public API. Reads, certification, and ingestion hold shared OS
locks. Maintenance retains the current, certificate-bound, receipt-bound, and
actively read generations and deletes only obsolete generations for which a
non-blocking exclusive lock succeeds. A crashed process releases its lock at
the operating-system boundary; no lease timeout can invalidate a live reader.

For concrete schemas and examples, inspect
[`schemas/bundle`](schemas/bundle/) and the
[dataset storage contract](../dataset/docs/storage.md). The CLI workflow and
its JSON transport are documented in [cli_doc.md](cli_doc.md).
