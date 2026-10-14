# Bundle-centered assessment, evidence, and benchmark governance

## 1. Executive decision

Hotpot must make it inexpensive to preserve an expert's judgment without
making an unsupported judgment indistinguishable from independently supported
scientific evidence.

The implementation therefore separates three decisions that are currently
collapsed into one certificate result:

| Decision | Meaning | Blocking behavior |
| --- | --- | --- |
| Archive admission | The records, relationships, paths, and content hashes are safe enough to preserve | Blocks ordinary storage when it fails; incomplete material has a separate forced-archive path |
| Evidence and quality assessment | The available material supports particular E0-E4 and B0-B4 levels | Produces levels and reasons; levels are never assigned directly |
| Benchmark promotion | The case satisfies an owner-controlled `BenchmarkGate` | Controls only benchmark membership, not preservation of an otherwise archivable case |

The central public object becomes a folder-backed, interactive `Bundle`.
`Bundle` owns its `Case` objects, synchronizes the configured policy, exposes
summaries, records evidence and benchmark attestations, and writes the
certificate for the resulting generation. The module-level
`write_certificate(bundle)` remains only as a thin convenience wrapper.

Every Bundle is also bound to one required dataset name (`ds_name`) in one
target dataset store. For a new dataset identity, Bundle accepts the proposed
assessment policy and persists it on first ingestion. For an existing dataset
identity, Bundle loads that persisted policy and rejects every caller-supplied
replacement.

The design intentionally makes the following breaking choices:

- a public `Bundle` always points to a directory;
- the default certificate is always `<bundle-root>/certificate.json`;
- `quality_level` and `evidence_level` are computed, read-only values;
- benchmark membership can be influenced only through gate results or an
  auditable benchmark attestation, never through a bare Boolean assignment;
- incomplete, ordinary, and benchmark are synchronized derived classifications,
  not competing sources of Case facts;
- the old declaration-mutation and single-`passed` workflow is removed rather
  than retained as a compatibility layer.

The governance layer assists rather than owns the user's scientific judgment.
It may report missing support, ambiguity, or a lower automatically supported
level, but it must preserve the submitted fact or expert statement with its
provenance. A user is not required to adopt Hotpot's benchmark policy in order
to preserve ordinary or explicitly forced-incomplete material. The normal
workflow stays short: build a folder once (or load an existing Bundle), inspect
status, synchronize, write a certificate, and ingest. Identity grouping,
conflict comparison, generation management, locking, and policy persistence
remain backend responsibilities.

The user-effort boundary is explicit:

| Required from the user | Optional |
| --- | --- |
| dataset name and a current-format folder/manifest | custom Gates or checker code for a new dataset |
| core labels/info declaration, including an explicit empty list when none are required | benchmark attestations |
| honest structure/evidence references, including declared missing material | primary literature enrichment beyond the available material |
| one explicit synchronization after a batch of edits | accepting Hotpot's benchmark subset as the user's own classification |
| a current certificate before ingestion | forced archival of incomplete Cases |

Hotpot validates and explains the consequences of these inputs. It does not ask
the user to assign B/E levels, construct internal database tables, manage
assessment generations, or collect evidence merely to make preservable data
pass as benchmark data.

## 2. Current implementation audit

### 2.1 Behavior to preserve

| Current behavior | Decision |
| --- | --- |
| B0/E0 data can be represented without complete supporting evidence | Preserve; this is the foundation of low-friction archival |
| Quality and evidence use independent B0-B4 and E0-E4 vocabularies | Preserve |
| Dataset-specific policy can strengthen general policy | Preserve through explicit assessment-policy configuration |
| Identity, relation, path, asset, and certificate integrity cannot be disabled | Preserve as immutable archive checks |
| Established datasets carry stored policy | Preserve; new releases must use the established assessment policy unless the dataset owner changes it explicitly |
| Certificates bind normalized content and configuration by digest | Preserve and extend to asset-state inventory, policy digests, and one immutable assessment generation |
| A certificate does not by itself prove scientific truth | Preserve in the schema and documentation |

### 2.2 Behavior to replace

| Current behavior | Problem | Replacement |
| --- | --- | --- |
| One `passed` value combines data integrity, configured requirements, declared quality support, and benchmark readiness | Safe but weakly documented data can be rejected as a whole | Separate archive, assessment, and benchmark results |
| `write_certificate(..., quality_level=...)` rewrites release declarations before assessing them | The assessor changes the object it claims to certify | Remove the parameter and certify an immutable Bundle snapshot |
| A high unsupported declaration fails admission | The high claim is lost unless the user manually downgrades it | Store the claim or expert attestation, calculate a lower supported level, and report the gap |
| `Bundle.info` exposes only basic dataset metadata | It cannot help the user decide what evidence is missing | Add completeness, evidence, quality, conflict, attestation, and benchmark summaries |
| Claims require evidence, but there is no first-class expert-attestation record | Expert knowledge is forced into unstructured metadata | Add typed, attributable, revocable attestations |
| A single configured minimum-evidence rule is an admission failure | It prevents candidate intake into a strict dataset | Move scientific thresholds to `EvidenceGate` and `BenchmarkGate`; keep structural admission independent |
| Public Bundle accepts folders, bare SQLite files, mappings, and internal bundles | Roots and certificate references are ambiguous | Require a folder root; importers must first create the standard folder layout |

The current code affected most directly is:

- `hotpot/ds/public/bundle.py`;
- `hotpot/ds/public/certificate.py`;
- `hotpot/ds/public/inputs.py`;
- `hotpot/ds/public/store.py`;
- `hotpot/ds/quality.py` and `hotpot/ds/levels.py`;
- `hotpot/ds/models.py` and `hotpot/ds/database.py`;
- `hotpot/ds/cli.py`, `hotpot/ds/README.md`, and `hotpot/ds/cli_doc.md`;
- `hotpot/dataset/schemas/record-standard.json` and the data rebuild tools.

## 3. Terminology and invariants

### 3.1 Terms

`Case`
: One independently classifiable dataset item. It groups or references its
  structures, labels/observations, claims, sources, evidence, verifications,
  and benchmark attestations.

`Bundle release`
: The one scientific dataset release governed by one Bundle root. A Bundle
  contains exactly one `dataset_id`/`release_id` pair; combining releases in a
  Bundle or splitting one Bundle across release identities is invalid.

`Case identity`
: The immutable identity of one dataset record. Case identity does not imply
  that another Case describes the same scientific sample.

`Identity schema`
: The persisted dataset-policy definition used to compare scientific sample
  identity. It declares only identity-bearing fields, their namespaces and
  normalization, selected structure roles, and comparator implementations.
  `core_info` remains a completeness contract and is not implicitly an identity
  key.

`Sample match`
: One of `same`, `different`, or `indeterminate`, produced under the dataset's
  identity schema. Asset, coordinate, or structure equality alone does not
  establish sample identity unless that schema explicitly makes it a component.

`Observation comparability`
: A property-policy decision that two observations address the same proposition
  under compatible subject/site, conditions, phase, participants,
  stoichiometry, method/branch, units, and uncertainty semantics.

`Evidence verification`
: An append-only record that a human or AI process inspected an evidence source
  and checked the relevant statement. Merely having a DOI or URL is not a
  verification.

`Benchmark attestation`
: An attributable approval or rejection of benchmark admission. It is not a
  replacement for evidence and does not assign B or E levels.

`Archive class`
: Exactly one of `incomplete`, `ordinary`, or `benchmark`.

`Incomplete`
: A case with a missing required core label/info field, a missing or invalid
  required structure, or a significant unresolved scientific value conflict.
  An incomplete case cannot become benchmark data under the default policy.

`Ordinary`
: A complete and archivable case that does not pass `BenchmarkGate`.

`Benchmark`
: A complete and archivable case that passes `BenchmarkGate`.

### 3.2 Non-negotiable invariants

1. `Case.quality_level` and `Case.evidence_level` have no public setters.
2. B/E levels change only by synchronizing changed source facts or assessment
   policy. They never change through direct assignment.
3. `case.benchmark = True` is not supported. A manual decision is an attestation
   and must record its authority type.
4. Gate results always include stable reason codes and human-readable reasons.
5. Direct benchmark entry bypasses ordinary checklist requirements only. It
   never bypasses unsafe paths, broken identity/relations, incomplete-case
   classification, or an enabled veto.
6. An `Other` attestation never grants direct entry under the default policy.
7. Evidence and attestations are append-only. Withdrawal and replacement create
   audit events; they do not erase the prior record.
8. A certificate binds an exact synchronized Bundle and assessment generation.
   Certification never assesses or mutates source facts.
9. Changing any source fact, scientific asset state, or new-dataset assessment
   policy makes the Bundle unsynchronized and the existing certificate stale.
10. No compatibility parser or deprecated alias is retained. Existing bundled
    project data is rebuilt into the new current format.
11. `ds_name` is mandatory. Existing/new status is determined against the
    target store before any assessment is performed.
12. An existing dataset's assessment policy is immutable from Bundle. Passing
    or replacing any Gate, identity schema, or property comparator in existing
    mode is an error.
13. `Bundle.ingest()` never creates a certificate implicitly. A current,
    matching root certificate must already exist.
14. Bundle is the sole owner of its Case objects. Ownership cannot be changed,
    Cases cannot be reparented, and every filtered view returns the same Case
    objects within that loaded Bundle.
15. Case identity, sample-equivalence identity, chemical-entity identity,
    structure identity, and file-asset identity are separate namespaces.
16. `core_info` declares required information. A field affects scientific
    identity only when the persisted identity schema names it explicitly.
17. Target values, claims, evidence/quality levels, benchmark decisions, and
    assessment results are never inputs to sample identity.
18. A scientific value conflict can exist only between observations whose
    Cases match as the same sample and whose contexts are comparable.
19. B0-B4 and E0-E4 are derived assessment results. Source observations and
    claims cannot carry an operative assigned grade.
20. A missing scientific asset may be preserved as a negative declaration, and
    hash/parse failure as a frozen assessment over identified bytes. A missing
    Bundle control record or Case fact record prevents reconstruction.
21. Case or new-dataset policy mutation commits source/policy state and audit
    events only. It makes the Bundle unsynchronized and the certificate stale;
    it never performs a hidden assessment.
22. `synchronize()`, `write_certificate()`, and `ingest()` are distinct steps.
    Neither of the latter two may invoke an earlier step implicitly.
23. One Bundle represents exactly one dataset release. Multi-release intake is
    split into separate Bundle roots before canonical build.
24. Successful ingestion copies every positively materialized scientific asset
    into the target store's content-addressed asset area. Published data never
    relies on the continued existence of a caller-owned Bundle path.

## 4. Target module structure

```text
hotpot/ds/
|- gates/
|  |- __init__.py
|  |- _base.py
|  |- evidence.py        # EvidenceGate
|  |- quality.py         # QualityGate
|  `- benchmark.py       # BenchmarkGate
|- public/
|  |- bundle.py          # Bundle, load_bundle, thin write_certificate wrapper
|  |- case.py            # Case and returned read-only collections/views
|  |- _bundle_state.py   # revisions, digests, and synchronization state
|  |- _assessment.py     # immutable assessment generations
|  |- certificate.py     # private envelope/serialization implementation
|  `- inputs.py          # strict folder loader
|- storage.py            # private class-aware ingestion backend
|- governance.py         # private Hotpot trust registry and signer boundary
|- governance_root.json  # tracked public genesis trust root
|- models.py             # normalized source and governance records
|- database.py           # persisted current schema
`- quality.py            # low-level integrity checks shared by gates
```

The gate classes are imported from `hotpot.ds.gates`:

```python
from hotpot.ds.gates import BenchmarkGate, EvidenceGate, QualityGate
```

They do not add more first-level names to `hotpot.ds`. This keeps the compact
facade while making policy configuration discoverable in one submodule.

### 4.1 Complete first-level public API signature tree

The target first-level namespace is deliberately closed. Its exact exported
names and order are:

```python
hotpot.ds.__all__ = [
    "datasets_list", "load",
    "Dataset", "Data",
    "BaseSearcher", "DataSearcher", "DatasetSearcher", "SmartsSearcher",
    "load_bundle", "Bundle", "write_certificate",
]
```

`Certificate`, `Configure`, `Store`, `Case`, `CaseCollection`, `CaseView`,
normalized rows, assessment reports, and database classes are not first-level
public names. Case and collection/view objects are returned protocols. Gate
classes live only under `hotpot.ds.gates`.

The complete signature/reachability tree is below. Properties are shown without
parentheses; direct constructors marked "returned object" are not the normal
user entry point. `Gate` below is a documentation abbreviation for the union of
the three classes in `hotpot.ds.gates`; `Checker` denotes the fixed callable
contract in Section 5.2. Neither abbreviation is another exported class.

```text
hotpot.ds
|
|- datasets_list
|  |- list(datasets_list) -> list[str]
|  `- datasets_list(*, db: Optional[Union[str, Path]] = None) -> list[str]
|
|- load(
|      ds_name: str,
|      version: Optional[Union[str, int]] = None,
|      *,
|      db: Optional[Union[str, Path]] = None,
|      archive_class: Optional[Union[str, Iterable[str]]] = None,
|  ) -> Dataset
|
|- Dataset                                  # returned object; no public constructor
|  |- len(dataset) -> int
|  |- iter(dataset) -> Iterator[Data]
|  |- dataset[index: int] -> Data
|  |- dataset[index: slice] -> Dataset
|  |- dataset.subset(**criteria: Any) -> Dataset
|  |- dataset.<namespace>_identifier[value] -> Union[Data, Dataset]
|  |- dataset.name -> str
|  |- dataset.dataset_id -> str
|  |- dataset.release_id -> str
|  |- dataset.version -> str
|  |- dataset.version_serial -> int
|  |- dataset.info -> dict
|  |- dataset.handle -> list[str]
|  |- dataset.labels -> dict[str, int]
|  |- dataset.core_labels -> list[str]
|  |- dataset.quality_levels -> dict[str, int]
|  |- dataset.evidence_levels -> dict[str, int]
|  `- dataset.identifier_namespaces -> tuple[str, ...]
|
|- Data                                     # returned object; no public constructor
|  |- data.id -> str
|  |- data.record_id -> str
|  |- data.local_id -> str
|  |- data.metadata -> dict
|  |- data.info -> dict
|  |- data.label -> dict
|  |- data.identifiers -> dict
|  |- data.entities -> list[dict]
|  |- data.quality_level -> str
|  |- data.evidence_level -> Optional[str]
|  |- data.archive_class -> str
|  |- data.structures() -> list[str]
|  |- data.structures.<format> -> Optional[str]
|  |- data.molecule -> hotpot.Molecule
|  `- data.crystal -> Crystal
|
|- BaseSearcher                             # abstract
|  |- searcher.match(item) -> bool
|  |- searcher(item) -> bool
|  |- searcher.search(target) -> Union[Dataset, tuple]
|  |- searcher.not_() -> BaseSearcher
|  |- searcher & other -> BaseSearcher
|  |- searcher | other -> BaseSearcher
|  |- searcher ^ other -> BaseSearcher
|  `- ~searcher -> BaseSearcher
|
|- DataSearcher(**criteria: Any) -> DataSearcher
|- DatasetSearcher(**criteria: Any) -> DatasetSearcher
|- SmartsSearcher(smarts: str, *, semantics=None) -> SmartsSearcher
|
|- load_bundle(
|      root: Union[str, Path],
|      ds_name: str,
|      *,
|      db: Optional[Union[str, Path]] = None,
|  ) -> Bundle
|
|- Bundle(
|      root: Union[str, Path],
|      ds_name: str,
|      *,
|      db: Optional[Union[str, Path]] = None,
|      gates: Optional[Iterable[Gate]] = None,
|  )
|  |- Bundle.from_certificate(
|  |      path: Union[str, Path],
|  |      *,
|  |      db: Optional[Union[str, Path]] = None,
|  |  ) -> Bundle
|  |- len(bundle) -> int                    # all archive classes
|  |- bundle.root -> Path
|  |- bundle.name -> str
|  |- bundle.mode -> str                   # "new" or "existing"
|  |- bundle.info -> dict
|  |- bundle.status -> dict                # always readable
|  |- bundle.is_synchronized -> bool       # derived, never assigned
|  |- bundle.summary -> dict
|  |- bundle.gates -> Mapping[str, Gate]
|  |- bundle.certificate_path -> Path
|  |- bundle.certificate_stale -> bool
|  |- bundle.cases -> CaseCollection       # every archive class
|  |- bundle.incomplete -> CaseView
|  |- bundle.ordinary -> CaseView
|  |- bundle.benchmark -> CaseView
|  |- bundle.use_gate(gate: Gate) -> None   # new mode only
|  |- bundle.synchronize() -> dict
|  |- bundle.write_certificate() -> Path
|  `- bundle.ingest(
|         *,
|         include_incomplete: bool = False,
|         reason: Optional[str] = None,
|     ) -> dict
|
|- Case                                     # returned via Bundle; not a module export
|  |- case.id -> str
|  |- case.info -> dict
|  |- case.core_info -> dict
|  |- case.structures -> tuple[dict, ...]
|  |- case.labels -> dict
|  |- case.observations -> tuple[dict, ...]
|  |- case.claims -> tuple[dict, ...]
|  |- case.evidence -> tuple[dict, ...]
|  |- case.attestations -> tuple[dict, ...]
|  |- case.evidence_levels -> dict[str, str]
|  |- case.evidence_level -> Optional[str]
|  |- case.quality_level -> str
|  |- case.archive_class -> str
|  |- case.benchmark -> bool
|  |- case.benchmark_status -> str
|  |- case.reasons -> list[dict]
|  |- case.assessment -> dict
|  |- case.set_core_info(
|  |      name: str,
|  |      value: Any,
|  |      *,
|  |      reason: Optional[str] = None,
|  |  ) -> None
|  |- case.remove_core_info(name: str, *, reason: str) -> None
|  |- case.add_structure(data: Mapping[str, Any]) -> str
|  |- case.replace_structure(
|  |      structure_id: str,
|  |      data: Mapping[str, Any],
|  |      *,
|  |      reason: str,
|  |  ) -> str
|  |- case.withdraw_structure(structure_id: str, *, reason: str) -> None
|  |- case.add_observation(data: Mapping[str, Any]) -> str
|  |- case.replace_observation(
|  |      observation_id: str,
|  |      data: Mapping[str, Any],
|  |      *,
|  |      reason: str,
|  |  ) -> str
|  |- case.withdraw_observation(observation_id: str, *, reason: str) -> None
|  |- case.add_claim(data: Mapping[str, Any]) -> str
|  |- case.replace_claim(
|  |      claim_id: str,
|  |      data: Mapping[str, Any],
|  |      *,
|  |      reason: str,
|  |  ) -> str
|  |- case.withdraw_claim(claim_id: str, *, reason: str) -> None
|  |- case.add_evidence(
|  |      target: str,
|  |      source: str,
|  |      *,
|  |      relation: str = "supports",
|  |      locator: Optional[str] = None,
|  |      note: Optional[str] = None,
|  |      details: Optional[Mapping[str, Any]] = None,
|  |  ) -> str
|  |- case.verify_evidence(
|  |      evidence_id: str,
|  |      *,
|  |      method: str,
|  |      verifier: str,
|  |      note: Optional[str] = None,
|  |      details: Optional[Mapping[str, Any]] = None,
|  |  ) -> str
|  |- case.withdraw_verification(
|  |      verification_id: str,
|  |      *,
|  |      reason: str,
|  |  ) -> None
|  |- case.replace_evidence(
|  |      evidence_id: str,
|  |      source: str,
|  |      *,
|  |      reason: str,
|  |      locator: Optional[str] = None,
|  |      note: Optional[str] = None,
|  |      details: Optional[Mapping[str, Any]] = None,
|  |  ) -> str
|  |- case.withdraw_evidence(
|  |      evidence_id: str,
|  |      *,
|  |      reason: str,
|  |  ) -> None
|  |- case.attest_benchmark(
|  |      approved: bool,
|  |      *,
|  |      authority: Literal["User", "Other"],
|  |      actor: str,
|  |      reason: Optional[str] = None,
|  |  ) -> str
|  `- case.withdraw_attestation(
|         attestation_id: str,
|         *,
|         reason: str,
|     ) -> None
|
`- write_certificate(bundle: Bundle) -> Path

hotpot.ds.gates
|
|- EvidenceGate(
|      checks: Optional[Mapping[str, Union[bool, Checker]]] = None,
|      *,
|      level_checks: Optional[Mapping[str, Iterable[str]]] = None,
|      require_locator: bool = True,
|  )
|  |- gate.info -> dict[str, Any]
|  `- gate.digest -> str
|
|- QualityGate(
|      checks: Optional[Mapping[str, Union[bool, Checker]]] = None,
|      *,
|      level_checks: Optional[Mapping[str, Iterable[str]]] = None,
|      structure_checks: Optional[Mapping[str, Union[bool, Checker]]] = None,
|  )
|  |- gate.info -> dict[str, Any]
|  `- gate.digest -> str
|
`- BenchmarkGate(
       checklist: Optional[Mapping[str, Union[bool, Checker]]] = None,
       direct_entry: Optional[Mapping[str, Union[bool, Checker]]] = None,
       *,
       vetoes: Optional[Iterable[str]] = None,
       numeric_tolerances: Optional[Mapping[str, Mapping[str, float]]] = None,
   )
   |- gate.info -> dict[str, Any]
   `- gate.digest -> str
```

`gates` on the one-time `Bundle(...)` build constructor is either `None` or an
iterable containing at most one instance of each Gate type. Validation-only
`load_bundle(...)` has no `gates` parameter. Type identity determines the Gate
slot, so callers do not depend on tuple order. Duplicate Gate types are errors.
In existing mode, `gates` must be `None`; persisted Gates are loaded
automatically. Gate instances and `bundle.gates` are read-only. A new-mode
change constructs a new Gate and passes it to `bundle.use_gate`; an
existing-mode replacement is an error.

`CaseCollection` is the single collection entry point and contains every Case:

```python
len(bundle.cases) -> int
iter(bundle.cases) -> Iterator[Case]
bundle.cases[index: int] -> Case
bundle.cases[index: slice] -> CaseView
bundle.cases.select(**criteria: Any) -> CaseView
```

`CaseCollection` is a read-only facade, not another owner. `CaseView` supports
the same sequence operations but has no selection or mutation API. Both return
the Bundle's original Case objects. Mutations live on `Case`; synchronization,
certification, and persistence live on `Bundle`. Neither returned type is
exported at the first level.

`bundle.cases.select(**criteria)` accepts only documented fields:

```text
archive_class, quality_level, min_quality, evidence_level, min_evidence,
benchmark_status, reason, verification, attestation
```

Scalar values use equality; list/set/tuple values use IN semantics. All supplied
criteria are ANDed. Unfiltered `bundle.cases` contains every archive class.
Derived criteria and `bundle.incomplete`/`ordinary`/`benchmark` require a
synchronized Bundle.

`load(..., archive_class=None)` defaults to `ordinary` plus `benchmark` and
therefore excludes incomplete stored records. A scalar selects one class; a
list/set/tuple uses IN semantics:

```python
only_incomplete = ds.load("spin", archive_class="incomplete")
incomplete_and_ordinary = ds.load(
    "spin",
    archive_class={"incomplete", "ordinary"},
)
```

This single keyword is the only database-read opt-in for incomplete data.
`Dataset.subset(archive_class=...)` may narrow an already loaded view but cannot
introduce classes that the original `load` excluded.

### 4.2 Synchronization state machine

Source editing and assessment are deliberately separate so a user can make
several inexpensive corrections and synchronize once:

```text
Case, asset, or new-dataset Gate mutation
    -> source/audit transaction committed
    -> unsynchronized + certificate stale

Bundle.synchronize()
    -> synchronized + certificate stale

Bundle.write_certificate()
    -> synchronized + certificate current

Bundle.ingest()
    -> immutable sealed release
```

The persisted state contains at least:

```text
source_revision and source_digest
policy_revision and policy_digest
governance_registry_revision and governance_registry_digest
synchronized_source_revision and synchronized_source_digest
synchronized_policy_revision and synchronized_policy_digest
synchronized_governance_registry_revision and synchronized_governance_registry_digest
current assessment generation ID and assessment digest
```

The policy digest covers the Gates, identity schema, property comparison rules,
custom checker artifacts, and their implementation/protocol digests. The
assessment-policy input additionally binds the trusted governance-registry
revision/digest and pinned genesis key ID without treating that registry as a
user-editable dataset Gate. The source
digest covers dataset/release declarations, Case facts, AssetDeclarations, and
actual local scientific-asset existence/content bytes.
`is_synchronized` is derived, never assigned:

```text
source_revision == synchronized_source_revision
AND source_digest == synchronized_source_digest
AND policy_revision == synchronized_policy_revision
AND policy_digest == synchronized_policy_digest
AND governance_registry_revision == synchronized_governance_registry_revision
AND governance_registry_digest == synchronized_governance_registry_digest
AND the referenced assessment generation is complete
```

For a draft, the left-hand governance values are the latest trusted registry.
For a sealed release, they are the immutable certificate/receipt-bound snapshot;
any newer registry is reported separately and cannot make an immutable release
look unsynchronized.

Revision counters make normal checks inexpensive; canonical digests are the
authority at process load, synchronization, certification, and ingestion
boundaries. An external file edit therefore invalidates synchronization even if
it bypasses the Python object.

For an editable Bundle, a newer trusted governance registry also invalidates
the assessment input. The next explicit `synchronize()` snapshots and audits
that trusted registry revision before reassessment; a caller cannot supply or
override it. A sealed release continues to verify against its certificate-bound
historical registry snapshot and reports a newer registry only as current
governance context, never by silently rewriting the sealed assessment.

The first implementation of `synchronize()` always reassesses every Case,
sample-equivalence group, conflict group, and Bundle summary in this order:

```text
freeze source, dataset policy, and trusted governance-registry snapshot under a Bundle lock
    -> folder and immutable-integrity checks
    -> sample matching and observation-context grouping
    -> EvidenceGate
    -> scientific conflict comparison and QualityGate
    -> incomplete classification
    -> BenchmarkGate vetoes, direct entry, and ordinary checklist
    -> immutable assessment generation and class indexes
    -> revision/digest compare-and-swap
    -> atomic publication of the new generation
```

A structure or identity edit can affect both its old and new groups, including
the bounded group context supplied to custom checkers. Incremental reassessment
is prohibited until a persisted dependency graph and checker dependency
declarations are proven digest-equivalent to full reassessment.

When unsynchronized, raw Case facts, `bundle.info`, and `bundle.status` remain
readable. Derived B/E levels, archive class, benchmark state, conflict and
quality summaries, and derived views raise `UnsynchronizedBundleError` rather
than presenting an old result as current. A previous generation may remain for
private recovery, but no normal public property returns it. `status` may report
its identifier and staleness.

`certificate_current` and target readiness are independent. A certificate is
current only when its envelope and bound source, policy, and generation digests
match; target-store creation, unrelated store activity, or a concurrent dataset
policy change does not rewrite scientific history or by itself make that
certificate stale. `bundle.status` separately reports
`target_precondition_satisfied` and `ingest_ready`:

```text
certificate_stale = NOT certificate_current

ingest_ready = is_synchronized
               AND certificate_current
               AND target_precondition_satisfied
               AND (release_is_draft OR exact_receipt_allows_idempotent_retry)
```

An absent target that becomes exact-policy existing may satisfy the documented
transition while the certificate remains current. A different persisted policy
sets `target_precondition_satisfied=false` and `ingest_ready=false`, but does not
reclassify the certificate as stale. `certificate_stale`, synchronization,
target readiness, and sealed state are therefore four separately derived facts.

## 5. Gate contracts

### 5.1 Common result contract

Gate results are immutable internal records containing:

```text
gate name and configuration digest
case ID and input digest
passed / failed / not-applicable status
computed level or benchmark decision, when relevant
stable reason codes
human-readable reasons
checks performed and checker fingerprints
evaluation timestamp and implementation version
```

The public API returns JSON-compatible summaries rather than exposing backend
row classes. Operational timestamps belong to generation metadata and are
excluded from the canonical ordered-assessment digest; identical source and
policy inputs must produce the same logical assessment digest.

### 5.2 Custom checker contract

Both the ordinary checklist and the direct-entry list may contain custom
checker functions. Users may implement a checker, but they may not customize
its call signature or return protocol:

```python
def checker(case: Mapping, bundle: Mapping) -> tuple[bool, str]:
    """Return (passed, reason)."""
```

The two arguments are canonical, JSON-compatible, read-only snapshots rather
than live `Case` and `Bundle` objects. The second mapping is a bounded checker
context: dataset metadata, policy facts, required fields, and the current
Case's precomputed identity/conflict-group context. It never contains a live
database handle or an eagerly materialized copy of every Case. Its exact field
set is frozen by the checker protocol. This keeps execution bounded and prevents
ordinary object API changes from changing checker behavior.
The Boolean result is a named policy fact. It cannot assign quality or evidence
levels directly; QualityGate and EvidenceGate map configured facts to their
normal automatic level criteria.

Rules for custom checkers:

- the function must be deterministic and free of side effects;
- it must not perform network access during synchronization;
- it must be a top-level function in a source file; lambdas, closures, notebooks,
  callable instances, native extensions, and dynamically generated functions
  are rejected;
- the first tuple item is exactly `bool`, and the second is a non-empty reason;
- an exception is recorded as `checker_error` and fails closed;
- only allowlisted standard-library imports are accepted; project-local and
  third-party runtime dependencies are rejected;
- the certificate stores the checker key, entry point, exact UTF-8 source
  digest, and checker protocol digest;
- `Bundle.from_certificate` never imports or executes arbitrary checker code
  from a certificate.

This resolves the apparent customization tension: checker implementations are
extensible, while the signature, semantics, and certificate representation are
fixed by Hotpot.

#### 5.2.1 Persistent checker artifacts

Custom checker source is part of the dataset's immutable assessment policy, not an
ambient Python plugin:

```text
bundle-root/policy/
|- policy.json
`- checkers/
   `- <source-sha256>.py
```

For a new dataset, Hotpot copies the exact checker source into this
content-addressed location and records its entry point and SHA-256 in
`policy.json`. On first successful ingestion, the store persists the complete
assessment policy and exact source text. The policy includes Gates, identity,
and property comparison rules; subsequent Bundle loads for that dataset use
only the stored policy and stored source. A local mirror is accepted only when
every digest equals the stored policy.

Source is persisted instead of pickle, `marshal`, code objects, or `.pyc`
files. Those formats are either unsafe to deserialize or tied to a Python
implementation/version. Source inspection and an import allowlist are admission
checks, not a Python sandbox. Custom code runs only in a separately spawned,
low-privilege OS-isolated worker with canonical JSON on stdin, an empty
environment and credential set, a dedicated isolated network namespace with no
usable interfaces, routes, or sockets (or an equivalent OS-enforced deny-all
network boundary), no Bundle/store access, no inherited file descriptors, and
enforced wall-time, CPU, and memory limits.
The worker exposes only the fixed checker entry point and a restricted built-in
and import set; its temporary filesystem is empty and disposable. If the host
cannot provide these isolation properties, Hotpot rejects custom checkers with
`CheckerIsolationUnavailableError`; it never falls back to in-process or
partially isolated execution. Built-in checkers do not require this user-code
path.

The persisted policy binds:

- Gate type, options, checklist/direct/veto membership, and tolerances;
- checker key, source text, source SHA-256, and entry point;
- the canonical checker-input field set and checker protocol digest;
- the relevant Hotpot built-in policy implementation digest;
- the owner approval event that admitted the custom code.

There is one current policy/checker protocol, not a collection of runtime
compatibility adapters. If a future Hotpot revision changes that protocol or a
built-in policy implementation, an existing mismatched policy fails closed
with `PolicyProtocolError`. Updating it is an explicit owner/admin dataset-policy
migration followed by reassessment and recertification. Bundle cannot perform
that migration or substitute a new checker. Git retains implementation history;
the live database is migrated atomically to the one current policy format.

### 5.3 `EvidenceGate`

`EvidenceGate` creates an immutable `EvidenceAssessment` for every active
observation or claim target and a conservative aggregate in `CaseAssessment`.
Observation and Claim remain source-fact records and never store an operative
E0-E4 value. A supplier-provided grade may be retained only as a non-operative
claimed value with provenance.

Default interpretation:

| Level | Minimum meaning |
| --- | --- |
| E0 | No usable linked support, missing source, withdrawn evidence, or unresolved evidence integrity failure |
| E1 | Extracted material exists but the relevant statement has not been verified |
| E2 | The linked source explicitly reports the value/claim and includes a usable locator or matched text |
| E3 | The relevant statement was checked against an identified primary source by AI or a human |
| E4 | At least two independent sources corroborate the result and satisfy the E3 conditions |

Additional rules:

- Human verification is recorded as a strong-assurance E3 route.
- AI verification is a valid E3 route only when model/tool identity, version,
  procedure, date, and checked source locator are recorded.
- AI plus human verification is retained as a stronger assurance annotation,
  but does not by itself create E4; E4 requires independent scientific source
  corroboration.
- E4 requires two E3 evidence chains with distinct canonical scientific-work
  identifiers and distinct primary result-lineage roots. Each Evidence chain
  records a normalized `source_lineage_id` and any `derived_from` relation; the
  two roots must differ, neither may derive from the other, and they may not
  share one underlying reported experiment, calculation, or determination.
  Different papers, URLs, mirrors, database records, or secondary reports are
  not independent when they reuse that same result lineage. DOI/work-ID
  normalization and duplicate-lineage detection are automatic. If affirmative
  independence cannot be established, the target remains E3 with an explicit
  reason until the lineage is clarified.
- An expert attestation may support an explicit claim, but cannot by itself
  become E3 or E4 unless it also points to qualifying primary evidence.
- `Case.evidence_levels` is a read-only projection of the target assessments;
  it is not another source of grades.
- `Case.evidence_level` is the minimum level among required/core targets. It is
  `E0_insufficient` when an expected evidence-bearing target is absent or
  unsupported. If the dataset declares no evidence-bearing required/core
  target, the aggregate is not a grade: Python returns `None`, JSON and SQLite
  store `null`, and aggregate distributions count the bucket key
  `not_applicable`. The literal string `"not_applicable"` is never persisted as
  the Case E value.

### 5.4 `QualityGate`

`QualityGate` consumes source integrity, structure checks, core-field checks,
conflict analysis, and `EvidenceGate` results. It computes Case-level B0-B4
quality and the separate incomplete/complete state.

The existing B0-B4 meanings remain:

| Level | Required character |
| --- | --- |
| B0 | Governed collection material; may be incomplete |
| B1 | Traceable candidate with provenance and subject mapping |
| B2 | Complete, reviewed, and conflict-adjudicated case |
| B3 | Primary-source-supported case with the configured validation/review controls |
| B4 | Independently corroborated reference case under a reproducible policy |

QualityGate must not infer high quality from a bare dataset declaration. It
derives quality only from evidence, review records, structure checks, conflict
results, and auditable artifacts.

Structure checking has two layers:

1. built-in availability, path, hash, parseability, and declared-role checks;
2. optional chemistry-specific structure checkers configured on QualityGate.

If a chemistry-specific checker is not configured, the report states
`structure_chemistry_not_checked`; it must not claim that chemical validity was
verified.

### 5.5 Dataset identity and scientific conflict

The identity schema is persisted as dataset policy, but is not another
first-level public class. Repository-managed datasets ship an owner-reviewed
built-in schema; most users therefore accept an existing schema and never
implement identity logic themselves. There is no universal chemical-identity
default. The safe generic fallback is Case-local identity: a Case matches
itself, while a cross-Case match is `indeterminate` unless persisted stable
identity components prove equivalence.

The assessment policy is the only schema authority. For a new dataset,
`bundle.json` contains only a built-in schema key or digest/reference; the
Bundle build step materializes the resolved component list into the canonical
policy. An existing dataset loads its persisted schema and only verifies a
local reference. A genuinely custom comparator uses the same fixed JSON input,
source hashing, isolation, and persistence rules as a custom Gate checker; it
does not introduce another public plugin interface.

Each identity component records only:

```text
component key and source namespace/path
required Boolean and value kind
normalizer and comparator
optional structure role
mismatch-is-decisive Boolean
implementation/protocol digest
```

`required` controls missing-value behavior only.
`mismatch-is-decisive` separately declares whether an unequal present value
proves `different`; a non-decisive mismatch produces `indeterminate` rather
than being ignored.

Cross-Case grouping must be deterministic and order-independent. It uses a
canonical identity key or component comparators proven reflexive, symmetric,
and transitive under the persisted protocol. Fuzzy geometry, numeric tolerance,
or similarity can provide corroborating evidence, but cannot define an
equivalence class. A schema that cannot meet these requirements fails policy
validation instead of creating order-dependent groups.

For two Cases, Hotpot normalizes and compares the declared components:

```text
sample_match(A, B):
    if A.case_id == B.case_id: same
    else if any present component with mismatch-is-decisive is unequal:
        different
    else if a required component is missing, ambiguous, invalid,
            or not comparable: indeterminate
    else if any present non-decisive component is unequal: indeterminate
    else if the persisted canonical keys/equivalence components match: same
    else: indeterminate
```

`indeterminate` is never silently treated as `same` or as a scientific
conflict. It may prevent benchmark admission when identity is required, but it
does not prevent ordinary archival. Structure is only one optional component.
Coordinate or file equality is evidence of asset reuse, not universal chemical
identity; molecular graph, charge, isotope, stereochemistry, protomer/tautomer
policy, crystallographic context, and structure role are used only when the
dataset schema explicitly requires them.

Observation comparison is a second, independent step. It is attempted only
when `sample_match(...) == same` and requires:

- the same property type and schema version, or an explicit equivalence
  transform;
- the same resolved subject, site, and relevant chemical state;
- compatible values for every comparison-relevant condition declared by the
  property type;
- compatible phase and participant-role/entity/stoichiometry multisets;
- compatible method, branch, and sign convention where required; and
- identical or convertible units and sufficient uncertainty semantics.

The context result is `comparable`, `noncomparable`, or `indeterminate`, with
stable reason codes. Missing required comparison context is indeterminate, not
a value conflict. A significant scientific conflict exists only when:

```text
sample_match(case(x), case(y)) == same
AND observation_context_comparable(x, y) == comparable
AND the property-specific comparator reports significant disagreement
```

Numeric comparators combine property-specific absolute/relative tolerances with
declared uncertainty when configured. Categorical comparators require
ontology-normalized mutually exclusive values. Free-text claims and
contradictory evidence links request adjudication; they do not automatically
become numeric conflicts.

Conflict candidates are the unique unordered pairs of distinct active
observations within the same canonical sample group and the same
property-equivalence family; self-pairs, withdrawn observations, and pairs from
different sample/property groups are excluded. Candidate IDs are sorted before
comparison, so input order cannot change the generation digest. Every
`conflict`, `noncomparable`, or `indeterminate` result is stored as an internal
`ConflictAssessment` row in the immutable generation; individual `consistent`
rows are compacted. Each row records the observation and Case IDs, canonical
sample key and identity-schema digest, property-family key, normalized
comparison context, comparator/policy digest, outcome, and reason codes. A
`ConflictGroupAssessment` persists total candidate count, consistent count,
exception-row count, and the digest of every canonical ordered pair/result, so
compaction cannot hide omissions or change the logical generation digest. An
unresolved `scientific_value_conflict` marks every active involved Case
incomplete. The
conflict remains until the relevant observation is corrected, superseded, or
withdrawn; the first implementation does not provide a Boolean override that
can suppress a scientific conflict without changing its facts.

The reason namespaces are distinct:

```text
scientific_value_conflict
observation_context_indeterminate
identity_indeterminate
benchmark_attestation_disagreement
```

Only `scientific_value_conflict` triggers the default scientific-conflict veto
and incomplete classification. Governance disagreement never contributes to a
scientific conflict count.

### 5.6 `BenchmarkGate`

`BenchmarkGate` has three policy components:

1. an ordinary checklist, for which every enabled item must pass;
2. a direct-entry list, for which any enabled item may grant direct entry;
3. veto items, which block both ordinary and direct entry.

Its public constructor remains deliberately small:

```python
BenchmarkGate(
    checklist=None,
    direct_entry=None,
    *,
    vetoes=None,
    numeric_tolerances=None,
)
```

Gate instances are policy configuration values. They do not expose a public
`evaluate(bundle)` method because that would bypass Bundle synchronization and
generation publication. Bundle invokes private Gate evaluators only inside
`synchronize()`. Reports are read through Bundle/Case summaries instead of
becoming additional public result classes.

Recommended public construction:

```python
from hotpot.ds.gates import BenchmarkGate

gate = BenchmarkGate(
    checklist={
        "verified_primary_source": True,
        "conflict_free": True,
        "project_rule": project_rule_checker,
    },
    direct_entry={
        "hotpot_or_user_attestation": True,
        "trusted_repository": trusted_repository_checker,
    },
    vetoes={"conflict_free", "attestation_not_rejected"},
    numeric_tolerances={
        "stability_constant": {"absolute": 0.05, "relative": 0.01},
    },
)
```

Mapping values have deliberately compact semantics:

- `True`: enable the named built-in checker;
- `False`: disable the named built-in checker;
- a callable: enable a custom checker under that key using the fixed checker
  signature.

Unknown names with Boolean values are rejected. A custom key must map to a
callable, so a typo cannot silently become a passing option.

#### Default policy

```python
BenchmarkGate(
    checklist={
        "verified_primary_source": True,
        "conflict_free": True,
    },
    direct_entry={
        "hotpot_or_user_attestation": True,
    },
    vetoes={"conflict_free", "attestation_not_rejected"},
    numeric_tolerances={},
)
```

The default decision is:

```text
benchmark = archive_integrity_passed
            AND case_is_complete
            AND all_enabled_vetoes_passed
            AND (
                any_enabled_direct_entry_passed
                OR all_enabled_checklist_items_passed
            )
```

An empty direct-entry list contributes `False`. An empty checklist does not
pass vacuously; without a passing direct route the result is `not_configured`,
not benchmark admission.

#### Default `verified_primary_source` checker

This checker requires all of the following for each core benchmark target:

1. an active linked scientific source;
2. a DOI or an HTTP(S) paper URL;
3. a usable locator, quote, or matched source text;
4. at least one active `AI` or `Human` verification record covering that
   source and target.

A DOI or URL without a verification record does not pass. AI and human
verification may coexist. Human verification is reported as very strong
evidence, but both methods satisfy this particular checklist item.

#### Default `conflict_free` checker and numeric tolerances

`conflict_free` is both a checklist item and a default veto. It evaluates active
observations through the identity and context comparators in Section 5.5.
Contradictory evidence is reported for adjudication but does not become a
scientific value conflict without comparable observations.

For two numeric values `a` and `b`, they are non-conflicting only when:

```text
abs(a - b) <= max(absolute_tolerance,
                  relative_tolerance * max(abs(a), abs(b)))
```

This formula is only the default numeric comparator when no uncertainty-aware
property comparator is configured. Tolerances are property- and context-specific.
If comparable core observations contain unequal values and no applicable
tolerance or uncertainty rule exists, the result is indeterminate and fails the
benchmark veto without being misreported as a value conflict. Values outside an
applicable rule are a `scientific_value_conflict`, make the Case incomplete, and
prevent benchmark entry.

#### Direct-entry attestations

Benchmark attestation authority is exactly one of:

| Authority | Meaning | Default direct-entry effect |
| --- | --- | --- |
| `Hotpot` | Approval by the Hotpot dataset owner/governance authority | Direct entry on its own |
| `User` | Approval by the submitting/owning user | Direct entry on its own |
| `Other` | An external expert statement whose authority is retained but not assumed | No direct entry |

`hotpot_or_user_attestation` passes when the Case has at least one active
Hotpot approval or at least one active User approval. Both may coexist, but the
default direct-entry rule requires only one of them. An `Other` approval remains
available as evidence and provenance but does not satisfy this rule.

`Hotpot` is a cryptographically verified governance authority, not a
caller-supplied string. The public method records only `User` and `Other`. The
private owner/admin path creates a Hotpot attestation by signing a canonical
JSON payload containing the fixed domain separator `hotpot.ds.attestation`,
signature-schema identifier, dataset identity, release ID, Case ID, decision,
reason, actor, event ID, creation time, supersession target, and key ID. These
bindings prevent cross-dataset/release replay. The one current signature
protocol is Ed25519; the attestation stores the signature and key ID.

The initial trust root is the `pinned_genesis_key_id` and public key shipped in
the tracked current resource `hotpot/ds/governance_root.json`. It contains no
private material. Store registry records are trusted
only when their signed chain verifies to that genesis key. Every assessment
snapshot records `governance_registry_revision`,
`governance_registry_digest`, and `pinned_genesis_key_id`; those values enter
the assessment-policy input digest, generation manifest, and certificate.
Loader, synchronization, certificate restoration, and ingestion verify the
chain and signature before an attestation has any Hotpot effect. A public key
found only in Bundle/certificate data is not a trust root. A loader never
upgrades self-declared metadata into Hotpot authority. User/Other actors remain
attributable declarations; they are not represented as Hotpot-verified
identities.

Key rotation is an explicit owner/admin governance migration: a newly signed
key record names its predecessor and activation boundary, and historical public
keys remain data required to verify already sealed releases. Revocation is
prospective from its signed effective boundary; signatures made before that
boundary remain verifiable under the certificate-bound registry snapshot, while
signatures at or after it have no Hotpot effect. An emergency retrospective
scientific/governance correction requires an explicit owner migration and a new
release rather than silently changing a sealed benchmark result. This is
trust-key history under one current verification protocol, not a code or
wire-format compatibility path. A revoked or unverifiable Hotpot signature is
retained as provenance and produces a stable verification reason.

There is exactly one internal signing operation, implemented in the private
`hotpot.ds.governance` module:

```python
_sign_hotpot_attestation(
    case,
    approved,
    *,
    actor,
    reason=None,
    signer,
) -> str
```

`case` is the internal Bundle-owned Case object, never a caller-created mapping;
it supplies dataset/release/Case identity, and the operation allocates the event
metadata before canonicalization. The signer protocol is exactly:

```python
signer.key_id -> str
signer.sign(payload: bytes) -> bytes
```

The private operation verifies `signer.key_id` against the active key in the
frozen trusted registry, inserts that ID into the canonical payload, and passes
the final bytes to `signer.sign`. The returned signature is verified immediately
against those same bytes and key before the fact transaction commits. `signer`
can be constructed only by the governance key service or authenticated
owner/admin execution context; it is not serializable and cannot be obtained
from Bundle files, CLI JSON, environment variables, or an ordinary Python
argument factory. Private key bytes never
enter the Bundle or dataset store. The operation appends the signed attestation
through the same canonical fact/audit transaction as other attestations. It is
not exported from `hotpot.ds` or exposed by public CLI.

For the same `(Case, authority, actor)`, a new decision supersedes the previous
one. An active approval and rejection from different Hotpot/User actors produce
`benchmark_attestation_disagreement`. The default
`attestation_not_rejected` veto blocks every benchmark path while any active
Hotpot/User rejection or disagreement remains; withdrawal or supersession
resolves it. This does not create `scientific_value_conflict`, make the Case
incomplete, or prevent ordinary archival. `Other` disagreement is informational
unless an explicit dataset policy promotes it.

The Hotpot-or-User attestation bypasses the ordinary primary-source checklist. It does
not bypass the conflict veto, incomplete classification, or immutable archive
checks. This is the deliberate balance between low-friction owner judgment and
protection against admitting known-broken data.

## 6. `Case` interaction model

### 6.1 Read-only assessment overview

Cases are created only by the Bundle loader/importer. A Case owner cannot be
replaced, no public append/remove/reparent operation exists, and hard deletion
is not a public workflow. A cross-Bundle copy is a new import with a new Case ID
and an explicit provenance relation, not migration of the original object.
Every view returns the same object within one loaded Bundle.

Raw source-fact properties remain readable while unsynchronized:

```python
case.id
case.info
case.core_info
case.structures
case.labels
case.observations
case.claims
case.evidence
case.attestations
```

Every mapping or sequence is a deep read-only snapshot or defensive copy. It
must be impossible to mutate nested dictionaries and bypass the audit log.
`case.info` contains source facts and synchronization state; it does not insert
an old assessment as though it were current.

The following derived properties require synchronization:

```python
case.evidence_levels       # per observation/claim target
case.evidence_level        # conservative core-target level
case.quality_level         # computed B0-B4
case.archive_class         # incomplete, ordinary, benchmark
case.benchmark             # computed bool
case.benchmark_status      # admitted, rejected, pending, vetoed
case.reasons               # structured classification reasons
case.assessment            # full JSON-compatible overview
```

They are projections of the current `CaseAssessment` and target
`EvidenceAssessment` records. They raise `UnsynchronizedBundleError` when no
current generation matches the source and policy digests.

Each reason contains:

```json
{
  "code": "missing_core_label",
  "category": "completeness",
  "severity": "error",
  "target": "oxidation_state",
  "gate": "QualityGate",
  "message": "Required core label oxidation_state is missing."
}
```

Stable reason codes are required for programmatic filtering. Messages are for
people and may improve without changing the code.

### 6.2 Audited fact editing

Fact editing follows one predictable add/replace-or-set/withdraw pattern. The
mapping payloads use the same current record schema as folder JSON and CLI
files; there is no alternative mutation schema.

```python
case.set_core_info(name, value, reason=None)
case.remove_core_info(name, reason="...")

case.add_structure(structure)
case.replace_structure(structure_id, structure, reason="...")
case.withdraw_structure(structure_id, reason="...")

case.add_observation(observation)
case.replace_observation(observation_id, observation, reason="...")
case.withdraw_observation(observation_id, reason="...")

case.add_claim(claim)
case.replace_claim(claim_id, claim, reason="...")
case.withdraw_claim(claim_id, reason="...")
```

Replacement creates a new fact and a `supersedes` relation. Withdrawal keeps
history and removes the fact only from the next assessment. Each successful
operation commits one source transaction and audit event, increments the source
revision, and makes the entire Bundle unsynchronized. It does not run a Gate.

| Fact | Public operations | Audit category | Invalidates |
| --- | --- | --- | --- |
| core info | set, remove | `core_info_changed` | whole assessment generation |
| structure | add, replace, withdraw | `structure_changed` | whole assessment generation |
| observation | add, replace, withdraw | `observation_changed` | whole assessment generation |
| claim | add, replace, withdraw | `claim_changed` | whole assessment generation |
| evidence | add, replace, withdraw | `evidence_changed` | whole assessment generation |
| verification | verify, withdraw | `verification_changed` | whole assessment generation |
| attestation | attest, withdraw | `attestation_changed` | whole assessment generation |
| new-dataset Gate | `use_gate` | `gate_policy_changed` | whole assessment generation |

This full invalidation rule is intentionally simple and correct. A future
optimized implementation may reuse work internally only after proving the
published generation identical to a full synchronization.

### 6.3 Recommended evidence API

Evidence addition should use a source-reference convention instead of a long
list of mutually exclusive DOI, URL, file, and source-ID parameters:

```python
evidence_id = case.add_evidence(
    target: str,
    source: str,
    *,
    relation: str = "supports",
    locator: str | None = None,
    note: str | None = None,
    details: Mapping[str, Any] | None = None,
) -> str
```

`target` accepts an observation ID, claim ID, or an unambiguous label/property
name. `source` uses one of these explicit forms:

```text
doi:10.1021/example
url:https://publisher.example/article
file:evidence/paper.pdf
source:existing-source-id
```

`relation` is one of `supports`, `contradicts`, or `context`. Local files must
remain inside the Bundle root and are hashed automatically. Adding evidence
does not mark it verified. `details` is a closed, optional provenance object;
it accepts `doi`, `url`, `scientific_work_id`, `source_lineage_id`, and
`derived_from`. Keeping these related facts in one object avoids a long public
signature while making the primary-work and independent-lineage facts required
for E3/E4 available to the Gate.

Verification is a separate, auditable action:

```python
verification_id = case.verify_evidence(
    evidence_id: str,
    *,
    method: str,       # "AI" or "Human"
    verifier: str,
    note: str | None = None,
    details: Mapping[str, Any] | None = None,
) -> str
```

AI verification metadata must include the model/tool and version in
`verifier`; the event also records the current procedure version and time.
Human verification records the reviewer's declared identity or stable local
identifier. The closed verification `details` object accepts only
`procedure_version` and timezone-aware `verified_at`; Hotpot supplies current
values when they are omitted.

An incorrect verification is withdrawn rather than edited:

```python
case.withdraw_verification(verification_id, reason="...")
```

Evidence is never overwritten silently:

```python
new_id = case.replace_evidence(
    evidence_id: str,
    source: str,
    *,
    reason: str,
    locator: str | None = None,
    note: str | None = None,
    details: Mapping[str, Any] | None = None,
) -> str

case.withdraw_evidence(evidence_id: str, *, reason: str) -> None
```

Replacement inherits the old target and relation, creates a new evidence
record, and records a `supersedes` relation. Withdrawal excludes the evidence
from new gate calculations but preserves it in the audit history.

After any evidence operation, the source change is durable but the Bundle is
unsynchronized until the user calls `bundle.synchronize()`. Direct assignment
such as `case.evidence_level = "E4..."` or
`case.quality_level = "B4..."` raises `AttributeError` because those properties
have no setters.

### 6.4 Benchmark attestation API

Manual benchmark adjustment is represented as an approval/rejection event:

```python
attestation_id = case.attest_benchmark(
    approved: bool,
    *,
    authority: Literal["User", "Other"],
    actor: str,
    reason: str | None = None,
) -> str

case.withdraw_attestation(
    attestation_id: str,
    *,
    reason: str,
) -> None
```

`reason` is optional as requested, but `authority` and `actor` are mandatory so
the statement remains attributable. The public method does not accept
`Hotpot`; that authority is created only by the private owner/admin governance
path after authenticating its principal. Approval/rejection disagreement is
reported as `benchmark_attestation_disagreement`, never as a scientific
conflict. Attestations use the same append-only withdrawal/supersession model as
evidence.

This method marks the Bundle unsynchronized. The next explicit synchronization
may change benchmark status; the statement never changes quality or evidence
levels directly.

## 7. `Bundle` interaction model

### 7.1 Construction and default gates

In this plan, the requested "database name" is the public dataset identity
`ds_name`; `db` remains the path of the physical Hotpot dataset store. The name
is mandatory even when the folder contains a matching declaration, so callers
cannot accidentally assess a folder against the wrong established policy.

`Bundle(...)` and `load_bundle(...)` have deliberately different verbs:

```python
import hotpot.ds as ds
from hotpot.ds.gates import BenchmarkGate, EvidenceGate, QualityGate

# One-time, explicit construction from a current-format intake folder.
bundle = ds.Bundle(
    "submission-folder",
    "csd-electronic-state",
    db="datasets.sqlite3",
    gates=[EvidenceGate(), QualityGate(), BenchmarkGate()],
)

# Validation-only reopening of an already built canonical Bundle.
bundle = ds.load_bundle(
    "submission-folder",
    "csd-electronic-state",
    db="datasets.sqlite3",
)
```

`Bundle(...)` is the only public build entry. It requires a current intake
manifest and no existing `cases/facts.sqlite3`, validates the complete input,
materializes canonical facts and policy in a temporary build, and publishes
`facts.sqlite3` last as the atomic build-complete marker. A failed build leaves
the intake folder unbuilt and safely retryable. If the marker already exists,
the constructor raises `BundleAlreadyBuiltError` and directs the caller to
`load_bundle(...)`. `load_bundle(...)` requires the marker, is validation-only,
and never creates, imports, repairs, synchronizes, or rewrites anything.

`db` may use the package default, but `ds_name` has no default. The dataset name
declared in `bundle.json` must equal `ds_name`. The build constructor determines
its policy mode by querying `db` before materialization:

| Mode | Condition | Gate rule |
| --- | --- | --- |
| `new` | The store does not yet contain `ds_name` | Gates in the single `gates=` collection are accepted; an omitted Gate type uses the current default |
| `existing` | The store already contains `ds_name` | Every Gate is loaded from the persisted dataset policy; non-`None` `gates=` is an error |

A missing database file or a store that does not contain `ds_name` selects new
mode without creating the store during construction. The resolved mode and
policy are persisted in the canonical Bundle; reopening recomputes the current
target context and verifies it rather than accepting caller overrides.

The persisted dataset assessment policy comprises the three Gates, identity
schema, property/context comparators, tolerances, and custom checker artifacts.
For an existing dataset none of these may be replaced through Bundle. A new
dataset takes its identity/property defaults from the current manifest/template
and its Gates from `gates=` or defaults.

Gate replacement is available only in new mode and marks the Bundle
unsynchronized:

```python
bundle.use_gate(BenchmarkGate(...))
```

In existing mode, a non-`None` `gates=` argument and `bundle.use_gate(...)` raise
`EstablishedGateError`. Bundle cannot relax, strengthen, replace, or migrate an
existing dataset's assessment policy. A policy change is a separate owner/admin
database operation outside Bundle and creates a fully audited current-policy
replacement followed by recertification.

The selected mode, dataset name, target-store context, and assessment-policy
digest are available through `bundle.info`. `new`/`existing` is a runtime policy
resolution state, not part of scientific content identity. If another process
creates the same dataset after a new-mode Bundle was built, ingestion succeeds
only through the exact idempotent-receipt or exact persisted-policy cases in
Section 10.3; it never silently adopts a different policy.

Gate objects receive the Bundle to assess; they do not own or mutate its source
facts.

### 7.2 Bundle summaries

The always-readable front door is intentionally small:

```python
bundle.info
bundle.status
bundle.is_synchronized
bundle.cases
```

`status` reports source/policy revisions and digests, the published assessment
generation, certificate state, sealed/draft state, and the exact next valid
action. It does not require users to understand the internal generation store.

When synchronized, Bundle also exposes:

```python
bundle.summary
```

`summary` is one JSON object with named `integrity`, `completeness`, `evidence`,
`quality`, `benchmark`, `attestation`, `conflict`, and `reasons` sections. It
contains counts and reason-code distributions and does not hide failed or
not-performed checks. Access while unsynchronized raises
`UnsynchronizedBundleError`; old assessment contents are not exposed as a
current-looking fallback.

### 7.3 Case collections and classification entry points

`bundle.cases` is the complete, read-only collection of incomplete, ordinary,
and benchmark Cases. This avoids a hidden population difference in the primary
Bundle object. Persisted `Dataset` reads continue to exclude incomplete records
by default.

Stable convenience views are:

```python
bundle.incomplete
bundle.ordinary
bundle.benchmark
```

`bundle.certificate_stale` is a Bundle-level Boolean, not a Case view.

All other classifications use one entry point instead of a property for every
level, reason, verifier, or attestor:

```python
missing_or_unverified = bundle.cases.select(
    archive_class={"incomplete", "ordinary"},
    min_quality="B1_candidate",
    reason={"missing_structure", "unverified_primary_source"},
)

conflicted = bundle.cases.select(reason="scientific_value_conflict")
human_checked = bundle.cases.select(verification="Human")
hotpot_attested = bundle.cases.select(attestation="Hotpot")
promotion_pending = bundle.cases.select(benchmark_status="pending")
```

Explicit incomplete-data access examples are:

```python
only_incomplete = bundle.cases.select(archive_class={"incomplete"})
incomplete_and_ordinary = bundle.cases.select(
    archive_class={"incomplete", "ordinary"}
)
```

All returned collections are sequence-like, preserve stable Case IDs, and
support iteration and indexing. They are views over the same Case objects, not
copies.

### 7.4 Mutation and persistence

Every Case or new-dataset Gate mutation is a source transaction: it updates the
canonical fact store, appends an audit event, advances a revision, and makes the
Bundle unsynchronized and its certificate stale. It never rewrites derived
class indexes or runs an assessment. This allows several edits before one
explicit full synchronization.

`synchronize()` publishes a complete immutable assessment generation through a
lock and compare-and-swap. `write_certificate()` requires that generation to be
current and only serializes/verifies it. `ingest()` rechecks the source, policy,
assessment, certificate, and target-store identities after its database
transaction begins. A concurrent edit fails closed instead of ingesting a
mixed snapshot.

## 8. Required Bundle folder layout

The public Bundle source is always a directory with this canonical layout:

```text
bundle-root/
|- bundle.json
|- cases/
|  `- facts.sqlite3
|- structures/
|  `- ...
|- evidence/
|  `- ...
|- policy/
|  |- policy.json                     # generated readable projection
|  `- checkers/
|     `- <source-sha256>.py
|- audit/
|  `- events.jsonl                    # generated readable projection
|- assessments/
|  |- leases/                         # runtime lock files; not certified
|  `- generations/
|     `- <generation-id>/
|        |- generation.json
|        |- governance-registry.json
|        |- assessment.sqlite3
|        |- incomplete.ids
|        |- ordinary.ids
|        |- benchmark.ids
|        `- summary.json
|- ingestion-receipt.json          # optional local projection; not certified
`- certificate.json
```

Rules:

- `bundle.json` contains the required `ds_name`, release identity, core
  labels/info, identity-schema key/digest reference, property definitions,
  relative asset roots, canonical inventory rules, and the one current intake
  declaration. It does not contain a competing identity schema.
- A not-yet-built intake root uses the same safe relative-path rules and points
  from `bundle.json` to exactly one current-format record index or exchange
  SQLite artifact retained under `evidence/intake/`; referenced scientific
  files remain under `structures/` or `evidence/`. The intake schema, required
  fields, and SQLite queries are normative dataset schemas, not heuristics.
  `Bundle(...)` is the explicit importer; it does not accept a bare SQLite path,
  mapping, or backend object.
- Users do not create or query `facts.sqlite3` tables manually. The build writes
  canonical facts and policy to temporary artifacts under a Bundle lock,
  verifies their digests, fsyncs them, and atomically renames
  `cases/facts.sqlite3` last. That file is the build-complete marker. A crash
  before the rename leaves a retryable intake root; a crash after it leaves a
  complete canonical Bundle. `load_bundle()` is validation-only and never
  creates or rewrites canonical facts.
- `facts.sqlite3` is the single transactional authority for Case facts,
  GovernanceEvents, source/policy revisions and canonical policy JSON, and the
  verified full ordered GovernanceRegistrySnapshots and current
  assessment-generation pointer. One SQLite transaction commits a fact or
  policy change, its audit event, and its revision. It contains no authoritative
  archive-class result.
- `audit/events.jsonl` and `policy/policy.json` are reproducible, human-readable
  projections created by the build and refreshed after successful mutations.
  They are not competing authorities: a missing/stale projection is reported
  but never changes facts, policy, or synchronization state, and validation-only
  loading does not repair it. Content-addressed asset or checker files are
  written before the SQLite transaction that references them; an unreferenced
  file after a crash is safe to garbage-collect.
- Each complete generation contains one immutable assessment store and three
  separate derived class indexes. Incomplete identifiers never share an index
  with ordinary or benchmark identifiers. Forced ingestion writes incomplete
  rows to a separate backend partition.
- Case facts never move when their derived class changes. Synchronization writes
  every generation artifact to a temporary directory, including the exact
  ordered public-key/rotation/revocation snapshot at
  `governance-registry.json`. It checkpoints and closes the assessment SQLite,
  removes WAL/SHM files, then computes the path, size, and SHA-256 of every
  generation artifact. It writes and fsyncs `generation.json` last; that
  manifest binds every other file, the canonical ordered-row digest, all
  source/policy/governance revisions and digests, and the pinned genesis key
  ID. Only after verifying the complete snapshot does synchronization publish
  the generation pointer in one `facts.sqlite3` compare-and-swap transaction.
  There is no second current pointer.
- Structure and evidence paths are safe relative POSIX paths inside the root.
- `certificate.json` is generated, not a source-of-truth replacement for case
  data.
- The certificate binds canonical facts, canonical policy/checker artifacts,
  the complete current assessment generation, and scientific asset inventory.
  It never binds itself, the regenerated `audit/events.jsonl` or
  `policy/policy.json` projections, or `ingestion-receipt.json`; updating those
  projections cannot create a self-invalidating certificate loop.
- In new mode, `policy/policy.json` and content-addressed checker sources are the
  proposed policy that will be persisted on first ingestion.
- In existing mode, the build materializes `policy/` as a read-only mirror of
  the database. A canonical root with a missing checker artifact or unequal
  policy/checker digest is invalid; `load_bundle()` never fills the gap and
  local files never override the database.
- A bare SQLite path, manifest mapping, or internal `DatasetBundle` is no longer
  accepted by the public constructor. Non-public preparation tools may place a
  supported current-format intake artifact into the directory, but
  `Bundle(...)` must perform the one authoritative build before
  `load_bundle(...)` can open it.
- If an exchange SQLite database is retained as an imported artifact, it lives
  under `evidence/` or another declared asset location; it is not the live
  Bundle backing store.

This generation layout is required for the current data scale. A single Case
edit must not rewrite hundreds of thousands of Case rows or the previous
assessment. A full synchronization may read all Cases, but it writes a new
transactional generation with bounded memory and leaves the last valid
generation intact until publication.

Generation retention is internal and reference-safe:

- the generation named by the canonical current pointer is always retained;
- the generation referenced by the root `certificate.json` is retained even
  when that certificate is stale;
- an open reader holds a shared OS lock on
  `assessments/leases/<generation-id>.lock`, so its generation cannot disappear
  mid-query;
- a newly published certificate atomically replaces the old root reference;
  only then may an older generation with no pointer, certificate, receipt, or
  reader reference be removed; and
- crash leftovers in temporary directories or never-published generations may
  be removed only under the Bundle lock after their absence from every
  reference table is verified.

Garbage collection must obtain a non-blocking exclusive OS lock on that same
generation lock file after checking every durable reference. Process exit or a
crash releases the shared lock automatically; there is no timeout that can
expire a live reader. If the platform cannot establish or verify the exclusive
lock, GC conservatively retains the generation. Lock files are runtime state,
excluded from certificates, and never used as generation content. Garbage
collection runs opportunistically after publication or through private
maintenance; it is not another first-level API or CLI workflow. These rules
prevent unbounded generation growth without risking an active reader or
stale-certificate round trip.

Scientific assets separate submitted facts from derived checks:

`AssetDeclaration` (source fact)
: safe path or URI, role, optional non-operative `claimed_required`, optional
  expected SHA-256, provenance, and whether a remote object was intentionally
  not materialized. It never claims a parser result, current remote
  availability, or effective requiredness.

`AssetAssessment` (generation result)
: observed local materialization (`present` or `missing`), observed SHA-256 when
  present, expected-hash result (`match`, `mismatch`, or `unchecked`), parse
  result (`valid`, `invalid`, or `not_run`), effective requiredness resolved
  from dataset policy, checker/parser identity and digest, and reason codes.

Dataset policy is the sole operative authority for which asset roles are
required. A declaration can preserve a supplier's `claimed_required` opinion,
but it cannot weaken or strengthen `policy_required(role)`. For a new dataset,
the requirement enters the proposed assessment policy through its manifest or
Gate configuration; for an existing dataset, only the persisted policy is used.

The current source-input digest covers declarations and actual local
existence/content bytes, so a newly appearing, disappearing, or changing file
makes the Bundle unsynchronized. Hash mismatch and parse invalidity remain
derived assessment results; changing a parser changes the policy/assessment
digest rather than rewriting source facts. A remote declaration uses
`remote_not_materialized`; restoration verifies that declaration and the frozen
assessment, not current network availability.

Negative asset assessments preserve incomplete material without inventing
content. Bundle control files, Case facts, policy artifacts, and assessment
generation files cannot be declared missing. Only a missing/invalid asset whose
role is required by dataset policy makes a Case incomplete. An optional
structure/evidence asset instead lowers applicable B/E support or adds a reason.

The loader distinguishes an unreadable root from an incomplete Case:

- unsafe path traversal, invalid root JSON, duplicate Case IDs, or relations
  that cannot be associated with a Case prevent Bundle construction;
- a missing or invalid required scientific asset, missing core field, or
  scientific value conflict is retained and can classify the Case as incomplete
  after synchronization. Optional-asset failure remains visible without being
  promoted to incomplete automatically.

## 9. Certificate model and round trip

### 9.1 Public calls

Certificate production moves onto Bundle:

```python
path = bundle.write_certificate()
assert path == bundle.root / "certificate.json"
```

The module-level function is a thin wrapper with no independent policy logic:

```python
path = ds.write_certificate(bundle)
```

It performs only:

```python
def write_certificate(bundle):
    return bundle.write_certificate()
```

Certificate writing requires `bundle.is_synchronized is True`. It does not
flush pending edits, call `synchronize()`, execute Gates, or reuse an assessment
whose input digest no longer matches. Failure reports the exact prerequisite and
suggests `bundle.synchronize()`. It snapshots under the Bundle lock, rechecks
source/policy/generation digests before atomic publication, and never leaves a
certificate that was current for only part of a concurrent edit.

For a sealed release, Section 10.4 takes precedence: the method verifies and
returns the already receipt-bound certificate path without rewriting it.

There is no public `quality_level` argument, no external certificate path, and
no mutation of core declarations. New versus existing dataset policy is
resolved when Bundle loads with its mandatory `ds_name` and target `db`.
Existing mode never accepts a Gate argument.

### 9.2 Certificate contents

The root certificate includes:

```text
schema and assessor versions
dataset name and release identity
target-store precondition and target-context digest
source revision and canonical source digest
policy revision and Gate/identity/property-policy digest
assessment generation ID, relative generation-manifest path, and manifest SHA-256
canonical ordered-assessment digest
root-relative positive and negative scientific-asset inventory digest
Gate configurations and configuration digests
custom-checker entry points, source hashes, and checker protocol digest
Hotpot-attestation signature/key IDs, pinned genesis key ID, and governance
registry revision/digest
immutable integrity results
Case count, archive-class counts, aggregate level/reason summaries
certificate issuer, time, and envelope digest
```

The target-store precondition is one of these exact records:

```text
existing target store:
    canonical_path, store_state="existing", store_uuid, schema_id,
    dataset_state="present" or "absent",
    dataset policy revision/digest when present

absent target store:
    canonical_path, store_state="absent", store_uuid=null,
    expected current schema_id, dataset_state="absent"
```

The whole SQLite file hash is never a store identity because ingestion changes
the file. Detailed target and Case assessments stay in the referenced immutable
`assessment.sqlite3`; the certificate does not duplicate an array proportional
to the number of Cases. `generation.json` inventories every other file in the
generation by safe relative path, size, and SHA-256. The certificate binds the
manifest hash and canonical ordered-assessment digest, thereby binding the
assessment database, all three class indexes, and `summary.json`. Assessment
SQLite is checkpointed, closed, stripped of WAL/SHM files, and reopened
read-only before these hashes are calculated. A Merkle tree is not required
until a concrete partial-proof use case exists.

The certificate may record that policy resolution was `new` or `existing` at
certification time for audit display, but that string is not a scientific
identity and is not required to remain equal after first ingestion. The bound
target precondition, policy digest, and target receipt govern that transition.

Claimed expert status, assessed levels, and effective benchmark status remain
separate fields. For example, a Case can retain an `Other` gold-standard claim,
be assessed as B1/E2, and remain ordinary.

### 9.3 Lossless reconstruction

```python
restored = ds.Bundle.from_certificate(
    "submission-folder/certificate.json",
    db="datasets.sqlite3",
)
```

If `db` is omitted, restoration uses the certified target locator. If supplied,
it is an assertion and must resolve to that same canonical target; it cannot
retarget a certificate.

`from_certificate`:

1. verifies the certificate envelope;
2. treats the certificate parent as the Bundle root;
3. requires every certificate-bound control object, canonical Case fact store,
   checker artifact, generation manifest, manifest-listed generation file, and
   positively inventoried local scientific asset, then verifies every bound
   path, size, and digest;
4. verifies each certified `AssetDeclaration` together with its frozen
   `AssetAssessment` without inventing content: the declaration's
   `remote_not_materialized` state is checked without network access; a local
   `missing` assessment still observes absence; and a hash-mismatch or
   parse-failure assessment retains the same observed bytes and expected-hash
   digests;
5. verifies source, policy, generation-manifest, every generation-file, and
   canonical ordered-assessment digest;
6. resolves the certified target context against `db`; an established dataset
   must have the same store UUID, schema, and persisted policy, while a target
   certified absent may now exist only when it contains the exact idempotent
   ingestion receipt or satisfies the exact-schema/exact-policy, unused-release
   transition defined in Section 10.3;
7. reconstructs Cases, Gate descriptors, the current assessment generation and
   class indexes, and canonical audit state;
8. derives synchronization, certificate, and sealed state instead of trusting
   stored Booleans or a local receipt projection; and
9. verifies that the reconstructed canonical Bundle and assessment digests
   equal the certified values.

Restoration does not rerun a structure parser, chemistry checker, or custom
checker. A frozen `parse_invalid` result is valid only while its observed input
digest and the certified parser/policy digest remain unchanged; the next
explicit `synchronize()` reruns it. Missing or changed bound data raises a
certificate-integrity error; it never silently returns a partial Bundle.
Certified absence is itself data: a formerly missing local file that appears,
or invalid/mismatched input bytes that change, makes the certificate stale and
requires synchronization and recertification. No network access is required to
restore a `remote_not_materialized` declaration. A certificate copied without
its Bundle directory is intentionally insufficient.

The round-trip acceptance invariant is:

```python
restored.status["source_digest"] == bundle.status["source_digest"]
restored.status["generation_digest"] == bundle.status["generation_digest"]
restored.summary == bundle.summary
```

For custom checkers, restoration reuses the certified frozen results and
verifies the persisted source hash. Certificate loading never executes code
named by certificate JSON. A later reassessment runs only the already trusted
source from the new dataset's content-addressed policy directory or the existing
dataset's persisted policy, and only through the isolated checker worker.
Hotpot signatures are reverified against the exact
`governance-registry.json` frozen in the certified generation. Its full ordered
key/rotation/revocation chain must reproduce the manifest digest and verify to
the installed pinned genesis key. The current external/store registry is used
only to determine draft reassessment readiness or report newer governance
context; it cannot replace the frozen snapshot during restoration. A public key
supplied only by the certificate or Bundle is never accepted as its own trust
root.

## 10. Three archive classes and storage behavior

### 10.1 Classification

| Class | Definition | Normal query behavior |
| --- | --- | --- |
| `incomplete` | Missing required core data/structure, invalid required structure, or significant unresolved scientific value conflict | Excluded |
| `ordinary` | Complete and safely archivable, but BenchmarkGate did not admit it | Included |
| `benchmark` | Complete, safely archivable, and BenchmarkGate admitted it | Included |

Benchmark is a state/subset of governed data, not a separate top-level dataset.

### 10.2 Forced archival of incomplete cases

The certificate always reports incomplete cases. Storage excludes them unless
the caller explicitly authorizes forced archival:

```python
bundle.ingest(
    include_incomplete=True,
    reason="Preserve supplier records pending structure recovery.",
)
```

`include_incomplete=True` requires a non-empty reason. The store records the
actor/context available to the CLI or Python caller and keeps incomplete cases
in a dedicated storage partition/table. This option does not change their
quality, evidence, or benchmark results.

The certificate always covers all three classes. The ingestion receipt records
the count and canonical Case-ID-set digest for both the included and excluded
populations. A successful ingestion seals the whole release even when
`include_incomplete=False`; excluded incomplete Cases cannot later be appended
to that release. Repair or additional archival uses a successor release.

Normal `Dataset`/`Data` reads and exports exclude the incomplete partition.
Explicit incomplete-only and incomplete-plus-normal views must be requested.
Sequence IDs remain stable and are not renumbered by filtering.

### 10.3 `Bundle.ingest`

Ingestion is a Bundle operation because Bundle already owns the dataset name,
target store, policy mode, assessment state, and certificate location:

```python
result = bundle.ingest()
```

`Bundle.ingest()` never certifies automatically. Before opening a write
transaction it requires `bundle.is_synchronized is True` and
`<bundle-root>/certificate.json`. It verifies that:

- the certificate envelope, every required control file, and every positively
  inventoried file are intact, while certified negative assets remain in their
  recorded states;
- the certificate matches the current Bundle and is not stale;
- `ds_name`, target-store context, dataset-state precondition or its explicitly
  permitted transition, and assessment-policy digest match the loaded Bundle;
- an existing dataset still has the same persisted assessment policy;
- every custom checker source/protocol digest and governance-registry
  revision/digest matches the certified assessment input.

After opening the target database transaction, ingestion verifies the source,
policy, complete assessment-generation manifest, certificate, store UUID/schema,
dataset policy, release identity, and selected included/excluded Case-set
digests again. A concurrent edit or store-policy race aborts the transaction.

If the certificate is absent, the operation raises `MissingCertificateError`
with the actionable message: `No current Bundle certificate exists; run
bundle.write_certificate() or hotpot ds certify first.` A stale or mismatched
certificate produces a different integrity error. Neither condition triggers
implicit recertification.

Target resolution uses these rules inside that transaction:

1. If the target store already existed at certification, its UUID and schema
   must still match. If it was absent, ingestion may create the current schema
   and a new UUID, or adopt a concurrently created store at the same canonical
   path only after verifying its schema and all dataset preconditions.
2. If the dataset was already established, its persisted policy revision and
   digest must match exactly and Bundle cannot modify it.
3. If the dataset was absent, ingestion inserts the proposed policy and exact
   custom checker source atomically with the first release and persists the
   verified governance-registry snapshot. If another process
   established the dataset first, this Bundle may continue only when the
   persisted policy is byte-for-byte/digest-identical and the release ID is
   still unused; this is the sole permitted `new -> existing` policy transition.
4. If the store already contains an ingestion receipt for the same dataset,
   release, Bundle digest, certificate digest, and included/excluded set
   digests, ingestion is idempotent and returns that receipt. The same release
   identity with any different digest is a conflict and is never overwritten.

The authoritative ingestion receipt is inserted in the target transaction and
has a uniqueness key of `(store_uuid, dataset_id, release_id)`. The row stores
the Bundle, certificate, and selection digests; equality makes a retry
idempotent and inequality makes the occupied release a conflict. It also records
final store schema, policy digest, assessment
generation digest, included/excluded set digests, actor/reason, and commit ID.
Only after target commit does Hotpot write the optional local
`ingestion-receipt.json` projection. If that local write fails, the ingestion is
still committed; the next `load_bundle()` or `from_certificate()` query finds
the exact target receipt and derives sealed state without writing the root.
An explicit identical ingest retry may regenerate the optional projection.
Thus a crash cannot produce a second write or leave the scientific certificate
stale.

Before the target transaction, every positively materialized local scientific
asset is streamed into a staging file under the target-owned content-addressed
asset area and verified against its certified SHA-256. The current layout is:

```text
<database-filename>.assets/
`- objects/
   `- sha256/
      `- <first-two-hex>/
         `- <full-sha256>
```

The staged file is fsynced and atomically renamed to its immutable object name;
an existing object is accepted only after digest and size verification. The
database stores one canonical asset object per digest and separate Case/
Structure references to that object. Equal bytes therefore deduplicate storage
without merging Case, sample, Entity, or Structure identity. The receipt binds
the ordered asset-object inventory digest. If the later database transaction
fails, an unreferenced object is harmless and is removed only by reference-safe
maintenance; a committed receipt never points to caller-owned files. Remote or
certified-negative declarations remain declarations and do not fabricate an
object. This ordering is the filesystem/database atomicity boundary and is
covered by crash-recovery tests.

The target database retains canonical Case facts once. Ordinary, benchmark,
and incomplete physical separation is implemented as derived transactional
indexes/partitions referencing those facts, never as three competing fact
copies.

`Bundle.ingest()` is the only public Python ingestion entry in the target
design. The former public `ds.store.ingest` route is removed rather than kept as
a second workflow; its database writer remains private implementation used by
Bundle.

### 10.4 Release lifecycle

```text
draft/editable -> unsynchronized or synchronized -> certified
    -> ingested/sealed
```

Before successful ingestion, a user may repair an incomplete Case through the
audited Case methods and then synchronize and recertify. Successful ingestion
seals the entire Bundle release, including any certified but excluded incomplete
population, and every mutating Case/Gate method fails. A later scientific
correction creates a new release-bound Case in a new Bundle, preserves stable
source/external identifiers, records parent/supersedes lineage, and uses the
existing dataset's persisted Gates. No published database row is repaired in
place. This is scientific release lineage, not a compatibility implementation.

Sealing also freezes the published generation and certificate:

- `synchronize()` on a sealed Bundle raises `SealedReleaseError`; it never
  publishes a replacement generation for that release;
- `write_certificate()` on a sealed Bundle first verifies that the local
  certificate, generation, and authoritative receipt digests agree, then
  returns the existing path without rewriting bytes, issuer, or time;
- a missing or mismatched bound object raises `SealedIntegrityError` rather
  than generating a replacement; and
- `ingest()` with all receipt-bound digests equal remains the sole idempotent
  retry and returns the existing receipt.

Reassessment, recertification, or scientific repair after sealing therefore
requires a successor release, except for an explicit owner/admin store
migration that preserves its own audited migration record.

## 11. Data and schema changes

### 11.1 New current-schema records

`BundleControl`
: the internal `facts.sqlite3` control rows containing source/policy revisions
  and digests, trusted governance-registry revision/digest, synchronized input
  revisions/digests, and the sole current assessment generation pointer.
  Certificate-current state is derived from the certificate file and bound
  digests; sealed state is derived from the target-store receipt. Neither the
  certificate nor an ingestion receipt is written into this certificate-bound
  control record.

`IdentitySchemaPolicy`
: dataset identity components, namespaces/paths, required flags, normalizers,
  comparators, `mismatch_is_decisive` flags, selected structure roles, and
  implementation/protocol digests. Every field participates in the policy
  digest. It is persisted with the dataset policy and is normally supplied by a
  built-in dataset template.

Add current-schema records for:

`EvidenceVerification`
: evidence ID, target/scope, `AI` or `Human` method, verifier, procedure
  version, timestamp, note, status, and supersession link.

`BenchmarkAttestation`
: Case ID, approved Boolean, `Hotpot`/`User`/`Other` authority, actor, optional
  reason, timestamp, status, and supersession link. Hotpot records additionally
  carry the canonical signed-payload digest, Ed25519 signature, governance key
  ID, and verification status; unsigned Hotpot authority is never effective.

`GovernanceKey`
: Hotpot governance key ID, Ed25519 public key, activation/revocation boundary,
  predecessor/successor relation, and owner-signed rotation event. These are
  trust records under the one current signature protocol, not executable code.

`GovernanceRegistrySnapshot`
: pinned genesis key ID, ordered GovernanceKey records, registry revision,
  canonical digest, and owner-signed update event. Its revision/digest is an
  assessment input and is bound by every generation and certificate that gives
  a Hotpot attestation governance effect. The verified full snapshot is stored
  in canonical facts and copied into each immutable generation as
  `governance-registry.json`; a digest without those records is insufficient.

`GovernanceEvent`
: append-only add/verify/replace/withdraw/attest events with actor, time, target,
  and before/after references.

`EvidenceAssessment`
: an immutable derived record keyed by assessment generation and
  observation/claim target, containing E0-E4, input/evidence digests, performed
  checks, and reason codes.

`CaseAssessment`
: the only authoritative derived Case state for one synchronized generation. It
  contains input/policy digests, Case B level, conservative aggregate E level,
  references to target EvidenceAssessments, completeness and identity status,
  scientific-conflict result, archive class, benchmark status, and reason
  codes. Source facts remain separate.

`AssetDeclaration`
: the source-fact declaration of a safe local path or URI, asset role,
  optional non-operative supplier `claimed_required`, optional expected hash,
  provenance, and intentional `remote_not_materialized` state. It never stores
  a current parser verdict or effective requiredness.

`AssetAssessment`
: the immutable generation result for one declaration: local presence,
  observed hash, expected-hash outcome, parse/check outcome, parser/checker
  identity and digest, policy-resolved effective requiredness, and reason codes.
  Optional-asset failure remains visible without automatically making a Case
  incomplete.

`ConflictAssessment`
: one immutable comparison result containing involved Case/observation IDs,
  canonical sample key and identity-policy digest, normalized observation
  context and comparator digest, outcome, and reason codes. An unresolved
  scientific-value conflict propagates incomplete status to every involved
  active Case.

`ConflictGroupAssessment`
: one immutable sample/property-family summary containing deterministic
  candidate, consistent, and exception counts plus the ordered digest of all
  candidate pair/results. It proves completeness while avoiding one stored row
  for every consistent pair.

`AssessmentGeneration`
: an immutable full-Bundle assessment store with generation ID, input and
  policy digests, ordered-content digest, summaries, separate
  incomplete/ordinary/benchmark indexes, and a manifest binding every
  generation artifact. Only a complete atomically published generation can be
  current.

`DatasetAssessmentPolicy`
: dataset name, Gate configurations, identity schema, property/context
  comparators, tolerances, policy/checker protocol digests, built-in
  implementation digests, bound governance-registry input fields, owner
  approval, creation time, and active status. Dataset-owned Gate/comparison
  policy is inserted for a new dataset and read-only to Bundle thereafter;
  trusted registry snapshots advance only through owner/admin governance, never
  through Bundle input.

`GateCheckerArtifact`
: policy ID, checker key/list role, entry point, exact UTF-8 source text,
  source SHA-256, allowed-import inventory, and checker protocol digest. No
  pickle, Python bytecode, or executable database object is stored.

`IngestionReceipt`
: an authoritative target-store row unique on
  `(store_uuid, dataset_id, release_id)`. Bundle, certificate, and
  included/excluded Case-set digests are content fields used to distinguish an
  idempotent retry from a conflicting reuse; they are not part of a key that
  permits another row for the same release. The record also stores
  schema/policy identity, assessment generation, actor/reason, commit ID, and
  the ordered target-owned asset-object inventory digest. The root JSON receipt
  is only a recoverable projection of this row.

### 11.2 Existing field changes

- User input no longer sets `DatasetRelease.quality_level` as an authoritative
  level. Persisted release summaries are derived from Case distributions.
- User input no longer sets `Observation.evidence_level` as an authoritative
  level. `Observation.evidence_level` and `Claim.evidence_level` are removed as
  operative fields; Gate results live only in EvidenceAssessment and
  CaseAssessment.
- Supplier or user opinions become claims/attestations rather than operative
  levels.
- Evidence provenance includes first-class normalized `source_lineage_id` and
  `derived_from` fields/relations, both included in the canonical source digest,
  so E4 independence is decided from primary result lineage rather than
  publication/URL count.
- If claimed levels are retained for provenance, they use explicit
  `claimed_quality_level` or `claimed_evidence_level` fields and never replace
  assessed values.
- Database indexes are added for archive class, quality level, evidence level,
  benchmark status, reason code, attestation authority, and verification method.
- assessment-policy and checker-artifact tables are keyed by dataset identity and
  content digest. Policy rows and their referenced source artifacts are written
  in the same transaction as first ingestion.

### 11.3 Current-version replacement

Update the current schema and rebuild repository-managed stores. Do not add
legacy schema detection, compatibility aliases, or dual writers. Git preserves
the old implementation. The rebuild must regenerate:

- the local dataset database;
- existing certificates;
- dataset inventory/summary records;
- any checked documentation examples and CLI snapshots.

### 11.4 Impact on currently ingested datasets

The current oxidation, spin, CSD electronic-state, and other managed datasets
require a one-time organizational migration. This is a storage/governance
rebuild, not an automatic scientific-data rewrite.

#### 11.4.1 Organizational changes

| Current representation | Target representation | Scientific-content effect |
| --- | --- | --- |
| Source folder, exchange SQLite, or manifest used directly for assessment | Canonical folder-backed Bundle | None; source rows and assets are normalized into Cases and references |
| `gold`/`silver` directory placement or supplier classification | Preserved supplier/provenance claim plus newly computed archive class | Old classification is retained as history but does not automatically grant benchmark status |
| User/supplier-declared operative quality or evidence fields | Explicit claimed fields plus Gate-derived B/E results | The claim is preserved; the operational level may change after reassessment |
| One mixed collection of normal records | One canonical fact store plus separate derived incomplete, ordinary, and benchmark indexes | Default visibility may change; source facts and record identity do not |
| Certificates stored separately from the submitted data | `certificate.json` at each Bundle root | All old certificates become obsolete and are regenerated |
| Certificate configuration or partial dataset policy | Persisted EvidenceGate, QualityGate, BenchmarkGate, and checker artifacts | Existing policy is bootstrapped once and becomes read-only to Bundle |
| Current normalized dataset database | Rebuilt current schema with Case assessments, assessment policy, checker artifacts, attestations, verifications, and archive-class indexes | Logical relations are preserved while derived governance rows are replaced |

The migration may change which records appear in normal queries. Records with
missing core fields, missing/invalid required structures, or significant
unresolved conflicts move to the incomplete partition and are excluded by
default. Complete records become ordinary unless BenchmarkGate admits them.

#### 11.4.2 Scientific-content preservation boundary

Migration must preserve the following source facts unless a separate,
explicitly audited scientific correction is performed later:

- molecular/crystal coordinates and structure-file bytes;
- stored SMILES and other structure representations;
- observed/calculated label values, units, uncertainty, and conditions;
- source text, citations, DOI/URL values, and evidence locators;
- record/source/evidence relationships and independently validated scientific
  entity/structure relationships;
- original supplier quality, evidence, gold, or silver assertions as
  provenance claims.

Legacy relationships whose identifiers conflate an Asset digest with a
Structure or scientific entity are not silently preserved as operative
scientific relationships. Their exact source rows remain provenance, while the
rebuild emits corrected current relations plus an explicit migration-map entry.
Every changed operative relation must therefore be traceable without treating a
known modeling error as scientific truth.

Before conversion, the exact canonical serialization of every legacy
identity/relationship row is frozen at
`evidence/migration/legacy-identity-relationships.jsonl`. It is declared in the
scientific asset inventory, content-hashed, and therefore certificate-bound.
This inventory is the lossless source-row record; the operative migration map
references its stable row ID and row digest rather than paraphrasing it.

Migration must not silently repair, reinterpret, or replace an invalid
structure or conflicting value. Such a Case is classified as incomplete with a
reason code. Any later repair is a new governance event that preserves the old
object and records the replacement/supersession relationship.

Unit-normalized values or parsed structure descriptors used by Gates are
derived data. They may be newly materialized, but the original value/file
remains intact and its digest remains queryable.

Physical asset placement may change. A structure can be copied, hard-linked,
or materialized from a content-addressed object store into a Bundle-relative
location, but its certified content SHA-256 must remain unchanged. Shared Asset
content identity and valid byte-deduplication relationships must be retained.
They do not prove shared Structure or sample identity. The Bundle layout must
not create a new scientific Structure merely because it materializes another
file reference, nor merge Structures merely because their Asset bytes match.

#### 11.4.3 Identity preservation

The concise legacy importer derives both `entity_id` and `structure_id` from
the SHA-256 of the first structure representation. That proves representation
content reuse only; it is not accepted as scientific sample identity in the new
schema.

The rebuild preserves exactly:

- dataset name and scientific release identity;
- release serial/version used in Hotpot IDs;
- valid Case/record ID and local/source ID;
- the persistent zero-based record index used by `hp-<dataset>[NNN]-<idx>`;
- external identifiers, source facts, and valid relationship targets; and
- asset bytes, content hashes, and content-deduplication relations.

Legacy SHA-derived entity/structure identifiers are retained as provenance
aliases when needed, not used as sample-equivalence keys. Subject/entity
mappings are rebuilt only from audited identifiers and the dataset identity
schema. When identity cannot be established, migration creates distinct
per-Case subjects and records `identity_indeterminate` instead of merging them.
Multiple Case/Structure instances may reference one shared Asset, so byte
deduplication never forces scientific identity.

Reclassification or default exclusion of incomplete Cases must never renumber
ordinary or benchmark data. The persistent record index must be stored rather
than recomputed from a filtered query order.

Migration emits certificate-bound
`evidence/migration/identity-migration.jsonl` with legacy inventory row ID/hash,
old ID, new ID(s), affected operative relation(s), relation outcome (`retained`,
`split`, `replaced`, or `unresolved`), comparison basis, and reason.
Entity/Structure counts and every changed entity/structure relation may differ
only when each difference is explained by this map. The map is provenance, not
a runtime compatibility API.

#### 11.4.4 Initial assessment-policy bootstrap

Currently ingested datasets predate the complete assessment policy and custom
checker artifact model. Before the new Bundle API treats them as existing
datasets, an owner/admin migration must:

1. translate their current core-field and quality requirements into the new
   EvidenceGate, QualityGate, and BenchmarkGate configuration;
2. define their identity schema and property/context comparison policy without
   treating legacy file hashes as sample identity;
3. make an explicit owner decision for any rule that cannot be translated;
4. persist the resulting policy, built-in implementation digest, protocol
   digest, and any exact custom checker source;
5. reassess all Cases and generate a new root certificate.

This bootstrap is not a Bundle Gate override. It runs once as an owner/admin
database migration before the new public workflow is enabled. After bootstrap,
all oxidation, spin, CSD electronic-state, and other existing datasets enter
`existing` mode and Bundle rejects supplied/replacement Gates.

#### 11.4.5 Migration and cutover procedure

The required cutover is:

1. snapshot the current store and record database/file digests;
2. export each current dataset/release to a canonical Bundle root;
3. verify source asset hashes and produce the identity/relationship inventory;
4. bootstrap and persist the owner-approved assessment policy;
5. synchronize Cases into incomplete, ordinary, and benchmark derived indexes;
6. regenerate the root certificate;
7. ingest into a new temporary current-schema database;
8. run parity checks for identities, structures, hashes, labels, sources,
   relationships, and record counts by classification;
9. atomically switch the configured store only after the parity report passes.

The previous database remains an immutable rollback artifact until acceptance
is complete; it is not queried as a compatibility backend. The implementation
must not migrate the only live copy in place.

Any count difference must be explained by the migration report. Classification
changes are expected; unexplained loss or modification of source facts is a
release blocker.

## 12. CLI alignment

The Python model remains primary, but the CLI expresses the same explicit state
transitions. The folder's required `bundle.json.ds_name` is authoritative for
CLI commands. Optional `--dataset NAME` is an assertion and must match; it never
overrides the manifest. Python keeps the required `ds_name` argument as an
explicit assertion at construction.

### 12.1 Status, synchronization, certification, and ingestion

```text
hotpot ds bundle build BUNDLE_ROOT [--dataset NAME] [--db DB]
hotpot ds bundle status
hotpot ds bundle synchronize
hotpot ds bundle --root BUNDLE_ROOT [--dataset NAME] [--db DB] status
hotpot ds bundle --root BUNDLE_ROOT [--dataset NAME] [--db DB] synchronize

hotpot ds certify BUNDLE_ROOT [--dataset NAME] [--db DB]
hotpot ds ingest BUNDLE_ROOT --db DB [--dataset NAME]
```

`build` is the CLI counterpart of `Bundle(...)`: it accepts only an unbuilt
current-format intake root, publishes canonical facts once, and never
synchronizes. All other Bundle commands use validation-only `load_bundle(...)`.
`--root` defaults to the current directory for the concise Bundle forms. A
successful source/Gate mutation reports `is_synchronized: false`; users may
perform several edits and synchronize once. `synchronize` never writes a
certificate. `certify` rejects an unsynchronized Bundle, writes only the root
certificate, and never ingests. `ingest` requires an explicit target `--db`, a
current synchronized generation, and a current certificate; it never performs
either preceding operation. `certify --db` may use the package default, but the
resolved store context is recorded in the certificate.

Forced incomplete archival remains explicit:

```text
hotpot ds ingest BUNDLE_ROOT --db DATASETS.sqlite3 \
  --include-incomplete --reason "Preserve pending repair"
```

### 12.2 Query, fact editing, and Gate editing

The query vocabulary is a fixed allowlist, not dynamic `getattr`:

```text
hotpot ds bundle get info
hotpot ds bundle get summary
hotpot ds bundle get cases --quality B2 --min-evidence E2
hotpot ds bundle get case.info CASE_ID
hotpot ds bundle get case.assessment CASE_ID
```

`get cases` exposes exactly the `CaseCollection.select` vocabulary as
`--archive-class`, `--quality`, `--min-quality`, `--evidence`,
`--min-evidence`, `--benchmark-status`, `--reason`, `--verification`, and
`--attestation`. Repeating a flag supplies IN values and different flags are
ANDed. An evidence threshold of zero means no evidence filter and retains the
`not_applicable` population; a positive threshold excludes
`not_applicable`. The same zero-threshold rule applies to the persisted-dataset
`subset -e/--evidence` command.

Derived queries fail with an actionable synchronization message when stale.
Case modifications use the exact Case API names and a current-schema JSON file:

```text
hotpot ds bundle set case.set_core_info change.json
hotpot ds bundle set case.replace_structure replacement.json
hotpot ds bundle set case.add_observation observation.json
hotpot ds bundle set case.add_evidence evidence.json
hotpot ds bundle set case.verify_evidence verification.json
hotpot ds bundle set case.attest_benchmark attestation.json
```

The file contains `case_id` plus the method arguments. It is plain JSON because
it is a file parameter, not stdin protocol data. The CLI validates it against
the same method schema, commits one audit event, and leaves the Bundle
unsynchronized. There is no second mutation path.

Gate inspection remains available in both modes:

```text
hotpot ds bundle gate benchmark --show
hotpot ds bundle gate quality --show
hotpot ds bundle gate evidence --show
```

The complete current editing flags are deliberately uniform and closed:

```text
# all Gates
--add-check NAME[=true|false]        --rm-check NAME
--set-level LEVEL=NAME[,NAME...]     --rm-level LEVEL
--checker NAME=FILE:ENTRYPOINT

# QualityGate only
--add-structure-check NAME[=true|false]
--rm-structure-check NAME

# BenchmarkGate only (ordinary checks use --add-check/--rm-check)
--add-direct NAME[=true|false]       --rm-direct NAME
--add-veto NAME                      --rm-veto NAME
--set-tolerance PROPERTY=ABS[,REL]   --rm-tolerance PROPERTY
```

`--checker` defines the callable for a same-named item added by
`--add-check`, `--add-structure-check`, or `--add-direct`; an unreferenced
checker is rejected. Multiple add/remove/set operations in one invocation form
one atomic policy edit and one audit event. `--show` cannot be combined with an
edit. Quality/Evidence `LEVEL` is one of their closed `B0`-`B4`/`E0`-`E4`
vocabularies and refers only to named Boolean checks, never an assigned Case
grade. Custom items therefore use the same small checker signature as built-in
items without dynamic CLI attributes or arbitrary JSON patches. These editing
flags are available only in new mode. Existing mode permits only `--show` and
rejects every modification.

Hotpot validates each custom checker contract and copies the exact source into
the content-addressed policy directory. Any Gate edit is atomic, marks the
Bundle unsynchronized, and does not run Gates.

The public CLI cannot create effective `Hotpot` authority from an input string.
Owner/admin Hotpot attestations use the private trusted governance path; normal
CLI attestations are `User` or `Other`.

`--root`, `--dataset`, and `--db` are options of the `bundle` parent command.
`--root` defaults to the current directory for every command except `build`;
`build` requires its one `BUNDLE_ROOT` positional and rejects simultaneous
`--root`. This single layout avoids command-specific locator variants.

### 12.3 Complete communication contract

Every machine result retains the existing exact four-field envelope:

```json
{
  "key": "hotpot.ds.bundle.status",
  "channel": "default",
  "protocol_sha256": "<current-map-sha256>",
  "payload": {}
}
```

`hotpot/ds/cli_communication.json` is the only executable map and the repository
stdio protocol skill is its only prose wire specification. This stage replaces
the map and runtime together; it does not add a second map or accept an old map
digest. Every producer, including a terminal producer, maps to a named semantic
contract. `null` contracts are forbidden because they would skip payload
validation. Each contract contains a closed JSON Schema reference under
`hotpot/ds/schemas/cli/`, the lowercase SHA-256 of that exact schema file, plus
extraction bindings only when a downstream consumer exists. The communication
map digest covers both each reference and each declared schema digest. Runtime
recomputes every referenced schema digest before accepting or emitting an
envelope, so changing an external schema cannot leave the protocol fingerprint
unchanged. Every referenced object schema uses explicit `required` fields and
`additionalProperties: false`.

The complete current producer inventory after this stage is:

| Producer key | Channels -> named contract |
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
| `hotpot.ds.bundle.get` | `info -> bundle_info_result`; `summary -> bundle_summary_result`; `cases -> bundle_cases_result`; `case_info -> case_info_result`; `case_assessment -> case_assessment_result` |
| `hotpot.ds.bundle.set` | `default -> bundle_mutation_result` |
| `hotpot.ds.bundle.gate` | `default -> bundle_gate_result` |
| `hotpot.ds.bundle.status` | `default -> bundle_status_result` |
| `hotpot.ds.bundle.synchronize` | `default -> bundle_synchronize_result` |
| `hotpot.ds.certify` | `default -> certificate_result` |
| `hotpot.ds.ingest` | `default -> ingestion_result` |

No other producer key/channel is valid. Removing or renaming a command removes
or renames its key in the same change. The existing dataset-query schemas remain
the normative closed schemas for the first eleven rows; Stage 13 does not alter
their data-selection semantics. It replaces every former terminal `null` with
the corresponding named result contract.

The Stage 13 contracts use these exact shared types:

```text
BundleLocator = {
    root: string, dataset: string, release_id: string,
    database: string
}
CaseSetDigest = {count: integer >= 0, sha256: 64-char lowercase hex string}
StoreIdentity = {
    canonical_path: string, store_uuid: string, schema_id: string
}
```

`BundleLocator.database` is always the resolved canonical absolute target path,
including when the caller relied on the package default; `null` is not a stable
store context and is rejected.

Their required payload fields are:

| Contract | Required payload fields |
| --- | --- |
| `bundle_build_result` | `bundle: BundleLocator`, `mode: string`, `source_revision: integer`, `policy_revision: integer`, `policy_digest: string`, `is_synchronized: boolean` |
| `bundle_info_result` | `bundle: BundleLocator`, `info: BundleInfo` |
| `bundle_summary_result` | `bundle: BundleLocator`, `summary: BundleSummary` |
| `bundle_cases_result` | `bundle: BundleLocator`, `criteria: CaseSelectionCriteria`, `items: [{case_id: string}]` |
| `case_info_result` | `bundle: BundleLocator`, `case_id: string`, `info: CaseInfo` |
| `case_assessment_result` | `bundle: BundleLocator`, `case_id: string`, `assessment: CaseAssessment` |
| `bundle_mutation_result` | `bundle: BundleLocator`, `action: string`, `case_id: string`, `event_ids: [string]`, `source_revision: integer`, `is_synchronized: false` |
| `bundle_gate_result` | `bundle: BundleLocator`, `gate: string`, `action: string`, `policy_revision: integer`, `policy_digest: string`, `is_synchronized: boolean` |
| `bundle_status_result` | `bundle: BundleLocator`, `source_revision: integer`, `source_digest: string`, `policy_revision: integer`, `policy_digest: string`, `generation_id: string | null`, `generation_digest: string | null`, `is_synchronized: boolean`, `certificate_state: string`, `target_precondition_satisfied: boolean`, `ingest_ready: boolean`, `release_state: string`, `next_action: string` |
| `bundle_synchronize_result` | `bundle: BundleLocator`, `generation_id: string`, `generation_digest: string`, `archive_counts: {incomplete: integer, ordinary: integer, benchmark: integer}`, `summary: BundleSummary`, `is_synchronized: true` |
| `certificate_result` | `bundle: BundleLocator`, `certificate_path: string`, `certificate_digest: string`, `source_digest: string`, `policy_digest: string`, `generation_digest: string` |
| `ingestion_result` | `bundle: BundleLocator`, `store: StoreIdentity`, `receipt_digest: string`, `included: CaseSetDigest`, `excluded: CaseSetDigest`, `idempotent: boolean` |

`BundleInfo`, `BundleSummary`, `CaseSelectionCriteria`, `CaseInfo`, and
`CaseAssessment` are references to the same closed current JSON Schemas used by
the Python/folder model; the CLI must not invent looser copies. Literal `false`
and `true` above are JSON Schema `const` values. String enum fields are closed
in those schemas. This table and the referenced schemas are generated into the
map before runtime implementation, so fields, types, requiredness, and routing
are machine-checked rather than inferred from prose.

Only the existing `data_collection` contract is initially consumable: `get`
accepts zero or more selectors and `export` one or more, with the already
declared ordering/duplicate rules. All Stage 13 result contracts are terminal in
the first implementation. Ordinary mutation JSON is accepted only as a named
file argument. A positional `-` always means one validated protocol envelope;
because no Bundle mutation consumer is declared, Bundle build/set/gate/status/
synchronize/certify/ingest commands do not accept `-`. Raw JSON stdin, unknown
keys/channels, undeclared producer/consumer pairs, and map-digest mismatches fail
closed.

Stdout contains exactly one complete validated envelope or nothing on failure.
Human presentation, prompts, warnings, and diagnostics use stderr. Existing
`get --smi/--xyz` requests keep structure data in the machine payload and may
only mirror a human view to interactive stderr.

Exit status follows the single stdio specification: `0` means a completed
nonnegative result; `2` covers usage, protocol, execution, or governance
failure. A completed domain-negative assessment may emit its valid envelope and
return `2`; a failure before a complete result emits no stdout. No additional
Stage-specific exit-code vocabulary is introduced.

## 13. Implementation phases

### Phase 0: freeze the standard

Tasks:

1. Finalize terminology, reason-code registry, default BenchmarkGate rules, and
   the owner-confirmed rule that either Hotpot or User approval is sufficient
   for direct entry when no active rejection remains.
2. Finalize the intake, canonical folder, certificate, generation-manifest,
   and CLI payload JSON Schemas.
3. Freeze the Bundle revision/digest state machine, assessment-generation
   schema, and certified negative-asset states.
4. Define B/E derivation tables, the dataset identity-schema protocol,
   three-valued sample matching, property comparison contexts, and
   scientific-versus-governance reason namespaces.
5. Freeze the canonical custom-checker JSON input, source restrictions, OS
   isolation requirements, and checker protocol digest procedure.
6. Freeze the Hotpot signature payload, pinned genesis key, governance registry
   schema/digest, and prospective revocation semantics.
7. Freeze the exact `hotpot.ds.__all__`, first-level signatures, and
   `hotpot.ds.gates` constructor signatures from Section 4.1.
8. Freeze every CLI producer/channel/named payload contract and the single
   stdio exit-code behavior, including every referenced schema digest.
9. Update `record-standard.json` before implementation so tests can target one
   declared contract.

Exit criteria:

- schemas validate representative ordinary, incomplete, and benchmark roots;
- direct-entry/veto precedence has unambiguous examples;
- no field permits direct B/E assignment;
- identity and observation comparison return explicit indeterminate states; and
- no command or schema relies on an undeclared compatibility version.

### Phase 1: folder-backed Bundle and Case model

Tasks:

1. Add `Case`, returned `CaseCollection`/`CaseView`, immutable ownership, and
   stable reason records without exporting another first-level class.
2. Restrict public Bundle input to directories; implement explicit one-time
   `Bundle(...)` build and validation-only `load_bundle(...)`.
3. Require `ds_name`, resolve new/existing mode against `db`, and reject
   caller-supplied Gates in existing mode.
4. Implement the canonical folder loader and safe relative-path inventory.
5. Implement the single transactional Case/audit/control fact store,
   defensive-copy reads, audited fact mutation methods, and source/policy
   revision/digest state.
6. Separate AssetDeclaration source facts from generation-local
   AssetAssessments; preserve declared absence and frozen failure results while
   continuing to reject missing control records.
7. Add immutable assessment-generation storage and atomic current-generation
   publication.
8. Add reference-safe internal generation leases and garbage collection.

Exit criteria:

- `bundle.cases` contains every archive class and all views return original
  Bundle-owned objects;
- source facts are independent of derived class indexes;
- nested returned mappings cannot bypass audit/state invalidation;
- path traversal and unassociable relation errors still fail loading.

### Phase 2: EvidenceGate and QualityGate

Tasks:

1. Add evidence verification, EvidenceAssessment, ConflictAssessment,
   CaseAssessment, and governance event models.
2. Implement automatic per-target E grades and one conservative Case aggregate,
   including explicit `not_applicable` when no evidence-bearing target exists.
3. Implement the persisted identity schema and three-valued sample matching.
4. Implement observation-context comparison before property-specific value
   conflict comparison.
5. Implement completeness, structure, core-field, conflict, and B-grade rules.
6. Remove authoritative input assignment of quality/evidence levels.
7. Implement full-Bundle `synchronize()` with lock, compare-and-swap, and
   atomic generation publication.
8. Add reason-code indexes and Bundle summaries.

Exit criteria:

- fact edits do not assess implicitly and full synchronization is deterministic;
- absent chemistry checks are reported as not performed;
- only comparable same-sample observations can create a scientific conflict;
- significant scientific conflicts and required-structure/core-field failures
  classify a Case as incomplete.

### Phase 3: BenchmarkGate and attestations

Tasks:

1. Implement Boolean/callable checklist and direct-entry maps.
2. Implement fixed JSON-snapshot checker validation, source persistence,
   isolated execution, and protocol fingerprinting.
3. Implement source-verification and conflict built-ins.
4. Implement numeric tolerance normalization and veto behavior.
5. Implement Hotpot/User/Other attestations and Hotpot-or-User direct entry.
6. Implement canonical Ed25519 Hotpot attestations, the private signer boundary,
   pinned-genesis registry persistence/rotation/revocation, and signature
   verification at every trust boundary; keep attestation disagreement separate
   from scientific conflict.
7. Persist proposed policy/source artifacts for new datasets and load only
   database-owned policy/source artifacts for existing datasets.

Exit criteria:

- any enabled direct checker can bypass ordinary checklist failures;
- no direct checker bypasses incomplete state or vetoes;
- Hotpot-only and User-only approvals each directly admit when no veto applies;
- Other-only approval does not directly admit;
- an ordinary caller cannot self-declare effective Hotpot authority;
- active Hotpot/User rejection or cross-actor disagreement blocks every default
  benchmark route but does not make scientific data incomplete;
- custom checker source and checker protocol hashes are reproducible and fully
  auditable.

### Phase 4: Bundle-centered certificates

Tasks:

1. Keep assessment orchestration in `Bundle.synchronize()` and certificate
   serialization in `Bundle.write_certificate()` as distinct operations.
2. Remove assessment-time declaration mutation.
3. Make module `write_certificate(bundle)` a thin wrapper.
4. Implement root-default atomic certificate writing.
5. Implement `Bundle.from_certificate` and exact digest round trips without
   executing parsers or checkers during restoration.
6. Bind dataset/release identity, target precondition, source/policy digests,
   the complete generation manifest, custom checker digests, summaries, and
   asset-state inventory without copying every Case assessment into the
   certificate.

Exit criteria:

- certification rejects an unsynchronized Bundle and never changes its source
  or assessment generation;
- missing required facts or positive assets prevent restoration while certified
  negative scientific-asset facts round-trip losslessly;
- custom code is never executed from certificate JSON;
- Bundle-to-certificate-to-Bundle is lossless under the documented invariant.

### Phase 5: class-aware store, query, and CLI

Tasks:

1. Add separate incomplete storage plus ordinary/benchmark indexes without
   making any class index the source of Case facts.
2. Add certificate-required `Bundle.ingest()` and remove the duplicate public
   `ds.store.ingest` workflow.
3. Persist a new dataset's assessment policy and checker sources atomically
   with first ingestion; enforce read-only policy for existing datasets.
4. Add authoritative transactional ingestion receipts, idempotent retry,
   target-store recovery, and included/excluded Case-set digests.
5. Require explicit reasoned opt-in for incomplete ingestion and seal the whole
   release after either inclusion choice.
6. Exclude incomplete data from normal Dataset, export, subset, and search
   results unless requested.
7. Keep `build`, `synchronize`, `certify`, and `ingest` separate and add the
   `hotpot ds bundle` build/status/get/set/gate/synchronize command family.
8. Require explicit `ingest --db`; keep CLI dataset identity manifest-driven
   with optional `--dataset` assertion.
9. Preserve JSON stdout/human stderr separation and replace the communication
   map atomically with every exact key/channel/named-payload/consumer
   declaration.

Exit criteria:

- ordinary workflows cannot accidentally return incomplete cases;
- forced archive does not relabel incomplete data;
- ingestion without a current root certificate fails with an actionable error;
- certification and ingestion both reject unsynchronized state;
- existing dataset Gate modification fails through both Python and CLI;
- CLI and Python produce identical assessments for the same root and gates.

### Phase 6: rebuild project data and documentation

Tasks:

1. Freeze a digest/inventory snapshot and recoverable copy of the current
   database and assets.
2. Convert current oxidation, spin, CSD electronic-state, and other managed data
   into canonical Bundle directories without changing source facts.
3. Preserve persistent record indices/Hotpot IDs and asset hashes, detect every
   SHA-derived legacy entity/structure identity, and emit the complete audited
   identity-migration map.
4. Bootstrap and persist owner-approved Gates for every existing dataset.
5. Reassess, regenerate class indexes, certificates, and store summaries.
6. Build a fresh database and produce a source-fact/parity report before atomic
   cutover; do not rewrite the only live database in place.
7. Replace current API and CLI documentation rather than adding compatibility
   notes for removed calls.
8. Update examples, templates, rebuild tools, and dataset-specific policies.
9. Audit tracked files to ensure real dataset content remains outside Git while
   schemas, templates, and non-sensitive examples remain tracked.

Exit criteria:

- no old certificate/declaration workflow remains in implementation or docs;
- repository-managed datasets rebuild reproducibly;
- every source structure/asset hash, scientific value, relationship, and stable
  identity either matches the snapshot or has an explicit audited explanation;
- coordinate/object reuse remains deduplicated without being treated as sample
  identity;
- classification changes are summarized separately from source-content parity;
- the public API inventory and formal documentation describe only the new
  current behavior.

## 14. Required test matrix

### Public API surface

- `hotpot.ds.__all__` exactly matches the ordered list in Section 4.1.
- `dir(hotpot.ds)` exposes no `Certificate`, `Configure`, `Store`, `Case`,
  `CaseCollection`, `CaseView`, generation, normalized-row, or
  assessment-report class.
- `Case`, `CaseCollection`, and `CaseView` are reachable only from Bundle and
  are not duplicated as first-level constructors.
- EvidenceGate, QualityGate, and BenchmarkGate are importable from
  `hotpot.ds.gates` and are not duplicated at first level.
- `inspect.signature` matches every function, supported public constructor, and
  public method in the signature tree.
- only the one-time `Bundle(...)` builder exposes one `gates=` collection;
  validation-only `load_bundle(...)` exposes no Gate override.
- `write_certificate(bundle)` delegates to `bundle.write_certificate()` and
  contains no policy logic.
- `Bundle.ingest()` is the canonical ingest implementation; no second public
  Store workflow contains independent behavior.
- `CaseCollection` provides sequence operations plus `select`; `CaseView`
  provides only sequence operations.
- `Dataset` exposes B/E distributions rather than an ambiguous scalar grade.
- public B/E properties remain read-only.

### Synchronization, ownership, and release lifecycle

- `bundle.cases` contains incomplete, ordinary, and benchmark Cases.
- Case owner cannot be replaced; no collection/view can append, remove, or
  reparent a Case, and every view returns the original object.
- every returned nested mapping/sequence is read-only or a defensive copy.
- every fact/Gate mutation persists an audit event, increments a revision,
  makes the Bundle unsynchronized, and does not execute a Gate.
- facts, GovernanceEvents, revisions, and the current-generation pointer commit
  in one `facts.sqlite3` transaction; regenerated JSON projections are never a
  second authority.
- an external asset/fact edit is detected by digest in a new process even when
  revision counters were bypassed.
- raw facts and `bundle.status` remain readable while derived properties and
  views raise `UnsynchronizedBundleError`.
- full synchronization publishes a new generation atomically and updates
  Cases in both old and new identity/conflict groups.
- a revision/digest change during synchronization fails compare-and-swap and
  leaves the old generation current.
- synchronization leaves the certificate stale; certification makes it current.
- certification and ingestion both reject unsynchronized state and never
  synchronize implicitly.
- target-policy/store changes affect `target_precondition_satisfied` and
  `ingest_ready`, not certificate freshness; an exact-policy new-to-existing
  transition leaves the scientific certificate current.
- successful ingestion seals the release; a later repair creates a new release
  with lineage rather than modifying published rows.
- sealed `synchronize()` is rejected, sealed `write_certificate()` returns only
  the verified existing file without rewriting it, and a digest mismatch raises
  `SealedIntegrityError`.

### Gate semantics

- DOI present but unverified: primary-source checklist fails.
- Paper URL plus AI verification: primary-source checklist passes.
- Paper URL plus human verification: passes and records strong assurance.
- AI plus human verification: both methods are retained but do not create E4
  without a second independent source.
- Two verified evidence chains with affirmatively independent primary result
  lineages: E4 is possible.
- two URLs, mirrors, or database records for one scientific work do not create
  E4, and ambiguous source lineage remains E3.
- a dataset with no evidence-bearing core target returns Python `None`, stores
  JSON/SQLite `null`, and counts the distribution bucket `not_applicable`, not
  E0.
- one Case can expose different target E levels while retaining one
  conservative Case aggregate, all from the same assessment generation.
- identical coordinate bytes with different protonation/bond-order identity
  facts do not establish the same sample.
- alternative structure representations can match when the persisted dataset
  identity schema says they describe the same sample.
- missing required identity input returns `indeterminate`, never a conflict.
- sample grouping is reflexive, symmetric, transitive, and input-order
  independent; fuzzy geometry cannot create an equivalence class.
- the same numeric property at a different site, phase, temperature, protocol,
  or branch is noncomparable or indeterminate rather than conflicting.
- participant roles/stoichiometry, unit conversion, and uncertainty are applied
  before the value comparator.
- Numeric duplicates inside absolute tolerance: no significant conflict.
- Numeric duplicates inside relative tolerance: no significant conflict.
- Values outside tolerance: conflict veto and incomplete classification.
- a cross-Case `ConflictAssessment` records both sides and makes every involved
  active Case incomplete until an observation is corrected, superseded, or
  withdrawn.
- conflict candidate pairs are unique, unordered, same-sample/same-property,
  self-pair-free, and input-order independent; group counts/digests detect any
  omitted compacted consistent result.
- Multiple unequal numeric values without a tolerance: veto fails closed.
- Hotpot-only approval: direct entry despite ordinary source-check failure,
  provided completeness and vetoes pass.
- User-only approval: the same default direct-entry behavior as Hotpot.
- Other-only approval: no default direct entry.
- ordinary callers cannot create an effective Hotpot attestation by supplying a
  string.
- the private governance signer cannot be constructed from Bundle/CLI input and
  no private key material enters Bundle files or the dataset store.
- effective Hotpot attestations require a valid canonical Ed25519 signature
  under a persisted governance key; invalid/revoked signatures have no Gate
  effect.
- cross-dataset/release signature replay fails; genesis-chain, registry
  revision/digest, activation, and prospective revocation boundaries reproduce
  the same result after certificate restoration.
- a trusted registry advance makes an editable Bundle require reassessment but
  does not rewrite a sealed release or make its bound certificate digest stale.
- active approval/rejection disagreement produces
  `benchmark_attestation_disagreement`, blocks every default benchmark route,
  and never appears in scientific-conflict counts or incomplete classification.
- A direct-entry checker cannot override the conflict veto.
- A custom checklist checker and a custom direct checker use the fixed
  JSON-snapshot signature and appear in the persisted source/policy inventory.
- A throwing or malformed custom checker fails closed with `checker_error`.
- Pickle, bytecode, closures, non-allowlisted imports, and protocol-mismatched
  custom checkers are rejected.
- custom checkers execute only under the required OS isolation; a host that
  cannot supply it rejects custom code rather than running it in process.

### Case behavior

- B/E properties cannot be assigned.
- core info, structure, observation, claim, evidence, verification, and
  attestation methods all use the current record schema and audit path.
- adding evidence or verification makes the Bundle unsynchronized; the next
  full synchronization changes only derivable assessment values.
- Replacement preserves and deactivates the old evidence through audit events.
- withdrawal does not erase history and never recalculates implicitly.
- Attestation reasons may be omitted, but authority and actor are required.
- Every classification reason has a stable code.

### Bundle views

- `ds_name` is required and must match the folder identity.
- `Bundle(...)` builds an unbuilt intake root once; `load_bundle(...)` rejects
  unbuilt roots and never creates or rewrites canonical facts.
- a failed build before the atomic marker is retryable, while an already built
  root directs callers to `load_bundle(...)`.
- a new dataset accepts supplied/default Gates and stages them for persistence.
- an existing dataset loads only persisted Gates; constructor overrides and
  `bundle.use_gate` fail.
- `bundle.cases` returns every Case, including incomplete.
- `bundle.incomplete` returns only incomplete cases.
- combined class selection explicitly returns incomplete plus normal cases.
- `bundle.cases.select` correctly filters missing/invalid structures, missing core
  fields, conflicts, verification methods, attestation authorities, and
  pending-promotion status without additional magic properties.
- no B0-B4/E0-E4 shortcut-property family is exposed.

### Folder and certificate integrity

- unsafe paths fail construction.
- a missing required asset survives as a negative declaration and an invalid
  required asset as an assessment over identified bytes; either produces an
  incomplete Case after synchronization, while the same condition for an
  optional asset remains visible without forcing incomplete classification.
- canonical facts are independent of classification; synchronized generations
  contain separate incomplete/ordinary/benchmark indexes.
- certificate defaults to the Bundle root.
- data modification, deletion, gate change, or attestation change makes the
  Bundle unsynchronized and the certificate stale.
- `Bundle.from_certificate` rejects missing required/positive files while all
  certified negative asset states round-trip exactly.
- certificate restoration never reruns parsers or checkers; unchanged observed
  bytes plus the frozen parser/policy digest preserve a parse assessment until
  the next explicit synchronization.
- a formerly missing asset that appears makes the old certificate stale.
- the certificate binds, but does not duplicate, the complete immutable
  assessment generation.
- changing an index, summary, assessment SQLite file, or generation manifest is
  detected; WAL/SHM files are absent before hashing.
- regenerating audit/policy projections or writing a local ingestion receipt
  does not invalidate the certificate.
- a single Case edit does not rewrite all Case facts or any published
  assessment generation.
- generation GC retains the current pointer, stale/current certificate target,
  receipt target, and active shared-lock readers; only verified unreferenced or
  unpublished generations for which an exclusive lock succeeds are removed.
- a 252,665-Case synchronization uses bounded-memory iteration and atomic
  generation publication; certificate size is not linear in Case count.
- canonical and assessment digests survive a complete round trip.
- certificate JSON cannot cause arbitrary checker imports or execution.
- persisted checker source hashes and checker protocol digests survive the round
  trip.
- the complete frozen governance registry file is manifest-bound, verifies to
  the pinned genesis key, and cannot be omitted or replaced by the current
  external registry during restoration.

### Store and query behavior

- `Bundle.ingest()` without a certificate fails and recommends certification.
- `Bundle.ingest()` rejects a stale or wrong-policy certificate and never
  recertifies implicitly.
- ingestion rechecks source, policy, assessment, certificate, and store
  identities inside its write transaction.
- absent and existing target stores use the documented path/UUID/schema and
  dataset-state preconditions; no whole-database file hash is treated as store
  identity.
- first ingestion atomically persists the new dataset assessment policy and checker
  source; subsequent ingestion cannot alter either through Bundle.
- incomplete ingestion requires both explicit opt-in and a reason.
- successful ingestion records included/excluded Case-set digests and seals the
  whole release even when incomplete Cases were excluded.
- an identical retry returns the authoritative receipt; the same release with a
  different Bundle, certificate, or selection digest fails.
- a committed target receipt restores sealed state when the optional local
  receipt projection was never written.
- forced incomplete ingestion remains physically/logically isolated.
- normal Dataset, subset, search, get, and export ignore incomplete cases.
- `load(..., archive_class="incomplete")` returns incomplete-only data, and a
  class collection supports explicit incomplete-plus-normal reads.
- explicit incomplete-only and mixed-class queries work without ID changes.
- established dataset gate policy is enforced on subsequent releases.
- a sealed release cannot be edited; a corrected successor retains auditable
  lineage without changing the old release.

### CLI behavior

- `build`, `synchronize`, `certify`, and `ingest` never invoke one another
  implicitly.
- `bundle status` remains usable while unsynchronized and identifies the next
  valid command.
- `hotpot ds bundle get` matches Bundle/Case attributes and emits protocol JSON.
- each `hotpot ds bundle set case.<method>` command validates and invokes the
  corresponding Case API.
- built-in and custom checklist/direct/veto edits work in new mode.
- quality/evidence Gate inspection and supported modifications work in new
  mode.
- every Gate modification command fails in existing mode while `--show`
  remains available.
- a custom `--checker NAME=FILE:ENTRYPOINT` source is copied, hashed, audited,
  and represented in the eventual certificate.
- `ingest` without explicit `--db` is rejected by argument parsing;
  `--dataset` is optional and a mismatching assertion fails.
- every new command emits the exact declared key/channel/payload; no Bundle
  mutation accepts raw stdin or an undeclared `-` envelope.
- every terminal producer uses a named, closed payload contract; no `null`
  contract bypasses output validation, and `bundle get status` does not exist as
  a duplicate of `bundle status`.
- stdout is one complete envelope and failures produce no partial stdout.
- exit codes remain the single protocol's `0`/`2` behavior rather than a
  Stage-specific vocabulary.

### Existing-data migration and parity

- oxidation, spin, and CSD electronic-state source structures retain identical
  content hashes after Bundle materialization.
- labels, values, units, conditions, sources, identifiers, and original
  relationship rows match the pre-migration source inventory; operative
  relationship corrections are separately mapped and justified.
- legacy SHA-derived entity/structure IDs are detected and never reused as
  scientific sample-equivalence keys without independent identity support.
- shared asset hashes remain deduplicated without merging distinct Cases or
  subjects.
- every entity/structure identity difference is explained by
  certificate-bound `evidence/migration/identity-migration.jsonl`, whose rows
  reference the frozen legacy identity/relationship inventory.
- every changed operative entity/Structure relation is explained, while shared
  Asset bytes remain deduplicated without being promoted to Structure identity.
- supplier gold/silver classifications survive as provenance but do not bypass
  the configured BenchmarkGate unless represented by a qualifying attestation.
- invalid structures and unresolved significant scientific value conflicts are isolated as
  incomplete without silent repair.
- class-index changes do not renumber persistent record indices or Hotpot IDs.
- old certificates fail under the new envelope and every migrated Bundle has a
  newly generated root certificate.
- each previously ingested dataset has one persisted bootstrapped three-Gate
  policy before it can load in existing mode.
- an unexplained source-fact count/hash difference prevents cutover.
- the rebuilt temporary database passes parity before the configured store is
  switched atomically.
- the frozen pre-migration database remains recoverable but is never queried as
  a compatibility backend.

## 15. Acceptance criteria

The stage is complete only when all of the following are true:

1. A user can archive a low-evidence expert-labelled Case without fabricating
   B4/E4 support.
2. Hotpot preserves who made a gold/benchmark assertion, under which authority,
   and why when a reason was supplied.
3. Quality and evidence cannot be directly assigned anywhere in the public
   Bundle/Case input path.
4. Default benchmark admission requires no active Hotpot/User rejection and
   either the verified-primary-source/conflict-free checklist or one active,
   undisputed Hotpot/User approval.
5. Significant scientific conflict remains a veto even for direct entry;
   identity/context indeterminacy and governance disagreement retain their own
   reason categories.
6. Incomplete data can be deliberately preserved but never appears in ordinary
   reads or shares a derived class index/backend partition with normal data.
7. Synchronization is the only operation that assesses data; certificate
   generation is pure verification/serialization of the current generation.
8. The module-level writer contains no policy logic.
9. Certificate restoration checks every bound object and complete generation
   manifest, verifies certified negative scientific-asset states without
   rerunning parsers/checkers, and reconstructs the same Bundle assessment.
10. Custom checker functions are extensible but conform to one fixed, simple,
    auditable JSON-snapshot protocol, their exact source is persisted with the
    dataset policy, and unavailable OS isolation causes rejection rather than
    unsafe fallback execution.
11. A Bundle always has a dataset name and explicit path/UUID/schema/dataset
    target context; an established dataset's assessment policy cannot be
    changed through Bundle or Bundle CLI.
12. `Bundle.write_certificate()` and `Bundle.ingest()` reject unsynchronized
    state; ingest also refuses to run without a current matching certificate.
13. `build`, `synchronize`, `certify`, and `ingest` remain distinct CLI
    commands, while `hotpot ds bundle` covers API-aligned status, inspection,
    and mutation.
14. Current managed datasets and documentation use only the new format and
    workflow; no compatibility implementation remains.
15. Migrated scientific source facts, structure hashes, relationships, release
    identities, persistent record indices, and Hotpot IDs are preserved or
    covered by an explicit owner-reviewed migration exception.
16. Existing datasets receive their initial persisted assessment policy,
    including the three Gates and identity/comparison rules, through the
    owner/admin bootstrap migration, never through a Bundle override.
17. Migration reports source-content parity separately from expected
    incomplete/ordinary/benchmark classification changes.
18. Cutover uses a verified rebuilt database and an atomic store switch; the
    only live copy is never migrated destructively in place.
19. The implemented `hotpot.ds` namespace, signatures, returned-object methods,
    and `hotpot.ds.gates` namespace exactly match Section 4.1 without duplicate
    compatibility entry points.
20. `bundle.cases` contains every Case, while normal persisted Dataset reads
    continue to exclude incomplete records unless explicitly requested.
21. Sample identity is dataset-policy-driven and distinct from core completeness,
    structure/coordinate equality, and file-asset identity.
22. The certificate binds the manifest and every file of an immutable
    assessment generation without copying a per-Case assessment array or
    requiring a first-version Merkle tree.
23. Successful ingestion seals a release; every later correction creates a new
    release with auditable lineage.
24. The normal user workflow remains build once (or load), inspect, synchronize,
    certify, and ingest; locks, revisions, generations, and CAS are not exposed
    as new first-level objects.
25. Canonical Bundle creation is an explicit one-time build; every later load is
    validation-only and cannot silently import or repair facts.
26. An authoritative target-store ingestion receipt makes exact retries
    idempotent, recovers sealed state after a local projection failure, and
    rejects digest-different reuse of a release identity.
27. Every effective Hotpot attestation is verified under the one current
    Ed25519 governance protocol; public User/Other APIs cannot forge it.
28. The single CLI communication map assigns a closed named payload contract to
    every producer/channel, including terminal outputs, and contains no
    compatibility digest or unvalidated `null` contract.
29. Governance trust is reproducible from a pinned genesis key and a
    certificate-bound registry revision/digest; signature replay and
    caller-created Hotpot authority are impossible through public inputs.
30. Assessment-generation GC never removes a current, certificate-referenced,
    receipt-referenced, or actively leased generation, and introduces no public
    lifecycle API.
31. One Bundle maps to one release, and every materialized ingested asset is
    owned by the target content-addressed asset area rather than a caller-owned
    path.
32. The communication-map fingerprint transitively binds every external CLI
    payload schema by its declared and runtime-verified SHA-256.

## 16. Owner-confirmed decisions

The owner has confirmed the following decisions. Items not separately
commented on are accepted as written:

1. Either one active Hotpot approval or one active User approval is sufficient
   for the default direct-entry rule when no active Hotpot/User rejection is in
   dispute. Both approvals are not required.
2. "Direct entry ignores other options" means it bypasses the ordinary
   checklist, but not immutable integrity, incomplete classification, or an
   explicitly marked veto such as significant scientific value conflict.
3. User-defined checkers are allowed, but their signature and return protocol
   are not customizable.
4. Human verification is very strong E3 evidence; it does not become E4 without
   affirmatively independent primary result-lineage corroboration.
5. The three archive classes are separated into three derived generation
   indexes and separate backend partitions where applicable. Canonical Case
   facts are classification-independent and are never moved on edit.
6. Bundle is always bound to a required `ds_name`. New datasets may propose an
   assessment policy; existing datasets load their persisted policy and reject
   all Bundle overrides.
7. Dataset assessment policies, including exact custom checker source, are
   persisted. No pickle/bytecode compatibility path is permitted; protocol
   mismatch fails until an explicit owner/admin current-policy migration is
   completed.
8. Ingestion is exposed as certificate-required `Bundle.ingest()`.
9. CLI `build`, `synchronize`, `certify`, and `ingest` remain separate. `hotpot
   ds bundle` provides API-aligned status, Bundle/Case query, mutation, and
   new-dataset Gate editing.
10. Bundle remains the only Case owner. No first-level `Case`, `Cases`,
    generation, synchronization-result, or certificate implementation class is
    added.
11. CLI reads authoritative `ds_name` from `bundle.json`; `--dataset` is only an
    optional assertion. `ingest --db` is mandatory.
12. Hotpot authority cannot be created by an ordinary caller-supplied string.
13. Users may preserve ordinary or forced-incomplete data without accepting
    Hotpot benchmark membership as their own classification.
14. `Bundle(...)` is the explicit one-time canonical build entry;
    `load_bundle(...)` is validation-only and cannot accept Gates.
15. Ingestion seals the complete release even when incomplete Cases are
    excluded, and records both included and excluded Case-set digests.
