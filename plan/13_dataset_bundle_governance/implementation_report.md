# Stage 13 implementation report

## Status and conclusion

Stage 13 has replaced the earlier certificate-centered intake path with one
Bundle-centered governance workflow in the working branch:

```text
build once (or load) -> inspect -> synchronize -> certify -> ingest
```

The implementation now keeps scientific facts, derived assessment, benchmark
admission, certificate proof, and target storage distinct while presenting them
through the existing concise `hotpot.ds` facade. Ordinary and explicitly
forced-incomplete material can be preserved without claiming benchmark quality.
Quality and evidence grades remain derived; an attributable owner statement may
admit an otherwise complete Case to the benchmark subset but cannot erase a
scientific conflict, broken relationship, or invalid required structure.

Stage 13 is operationally complete on the implementation branch. The managed
store was rebuilt from the verified legacy source, parity-checked, certified,
ingested, and atomically activated. Activated-store API/CLI smoke tests and the
final current dataset suite passed; exact evidence is recorded in Step 46 of
the [implementation log](implementation_log.md).

## Delivered behavior

### Concise user model

- `Bundle` permanently owns all of its Cases; filtered views return those same
  objects.
- A batch of Case or new-dataset Gate edits changes durable source/policy facts
  and makes assessment stale. It never triggers hidden assessment.
- `Bundle.synchronize()` is the sole assessment operation.
- Issuing a new certificate through `Bundle.write_certificate()` requires
  synchronized, current-runtime state and only serializes/verifies the
  published generation; a sealed historical Bundle can only verify and return
  its existing frozen certificate.
- `Bundle.ingest()` requires a current root certificate and never certifies or
  synchronizes implicitly.
- Existing datasets load their persisted policy. Bundle cannot weaken, replace,
  or migrate it.
- Successful ingestion seals one release. Later repair creates a successor
  release instead of mutating published facts.

The first-level namespace remains the eleven names already approved in the
[signature tree](bundle_gate_certificate_implementation.md#41-complete-first-level-public-api-signature-tree).
Case collections, Gate internals, assessment generations, certificates,
receipts, runtime policies, leases, and database writers remain returned or
private implementation objects rather than new first-level concepts.

### Assessment and scientific integrity

The implementation derives one Case-level B0-B4 quality result, one
conservative Case-level E0-E4 result (or `None` when there is no target), and
one archive class. Per-target evidence assessments are supporting detail, not a
second operative grading system.

Sample identity is controlled by the persisted dataset identity schema.
Coordinate/file equality is reusable-asset evidence, not universal sample
identity. Scientific conflicts require both same-sample identity and comparable
observation context. Identity indeterminacy blocks benchmark admission but does
not make an otherwise complete Case unarchivable.

The Bundle manifest carries one closed `identity_schema.components` array.
Canonical build restricts scalar roots to `core_info`/`identifiers`, permits
only role-selected semantic Structure components, materializes typed
normalizers plus the exact comparator and current protocol digests, and
persists the normalized value in dataset policy. `sample_match` is explicitly
three-valued. When cross-Case identity cannot be established, conflict
discovery uses a Case-local key, so distinct observations in that same Case are
still compared. Conflict rows bind the normalized identity-policy and effective
property-comparator digests.

Evidence now has executable relationship semantics:

- only explicit `supports` links raise E levels;
- source type is one explicit closed `source_kind`; DOI and URL identifiers do
  not imply `primary_literature`;
- valid AI verification records the attributable verifier, tool and version,
  procedure version, timezone-aware time, and checked locator, while Human
  verification requires an attributable reviewer;
- missing or hash-mismatched file-backed Evidence is certificate-bound but
  unusable for support and may leave a target at E0 without, by itself, making
  the Case incomplete;
- a target resolves by exact observation/claim ID or one unambiguous
  property/label alias;
- active verifications, supersession links, result-lineage ancestry, entity
  references, participants, and subjects must resolve consistently;
- malformed operative relationships remain preserved as facts but classify the
  Case as incomplete;
- normalized scientific-work IDs and independent result-lineage roots prevent
  obvious false E4 assignments; and
- missing declared core targets produce E0, while an empty core-label policy
  assesses all active targets.

Hotpot can verify the internal consistency and traceability of submitted work
and lineage declarations. It cannot prove that a supplier's assertion of
scientific independence is true without external evidence. The resulting E4
decision is therefore auditable and policy-bound, not a claim of omniscient
truth.

### Benchmark authority

Either one active verified Hotpot approval or one active User approval may
provide the default direct-entry route. An Other approval remains provenance
unless dataset policy explicitly promotes it. Hotpot authority is available
only through the private Ed25519 signing boundary backed by the mandatory
`cryptography` dependency; an ordinary caller cannot create it by passing a
string.

A new public User/Other decision requires an exact Boolean and non-empty actor
and supersedes every active decision from the same authority/actor. Active
cross-principal approval/rejection disagreement is governance disagreement, not
a scientific value conflict. Rejection, incompleteness, integrity failure, and
scientific conflict retain veto priority over direct entry.

### Deterministic runtime and custom code

One self-digesting runtime-policy document binds assessment source and selected
semantic symbols, assessment-store protocol, checker worker/protocol, parser
registry and selection, parser source/protocol/distributions, Python ABI,
SQLite, and relevant installed distributions. Dataset policy, assessment
generation, certificate, and ingestion receipt contain the same frozen
document. Every parsed asset additionally records its selected parser binding.

The implementation deliberately separates two checks:

1. structural and digest validation of a frozen policy proves which runtime
   produced a historical result; and
2. exact comparison with the ambient current policy is required before new
   synchronization, certificate issuance, or ingestion.

This permits verification of a sealed historical release without loading old
code. A mismatch directs an editable dataset to an explicit owner migration and
a sealed dataset to a successor release. No compatibility parser or runtime is
executed.

Custom checkers run only inside the required Bubblewrap/Linux namespace
boundary. The worker receives bounded canonical JSON containing the checker
source and digest; no Bundle/store/checker file is mounted. It has a fixed
minimal environment, read-only interpreter/library mounts, disposable `/tmp`,
no network, no inherited file descriptors, restricted imports/built-ins, and
CPU, memory, process, file, output, and wall-time limits. Failure to establish
isolation blocks synchronization. An individual checker timeout, exception, or
protocol violation fails closed as an auditable `checker_error`.

### Certificate, generations, and ingestion

Assessment publication uses immutable complete generations and one atomic
current pointer. Readers, certificate operations, and ingestion hold shared
generation leases; conservative garbage collection cannot remove current,
certificate-bound, receipt-bound, unknown-reference, or actively read
generations.

The certificate binds the Bundle facts, policy/runtime, governance snapshot,
generation manifest, checker artifacts, positive and negative asset inventory,
and summary without copying every Case assessment into JSON. Restoration checks
the frozen generation and asset states and never executes parser or checker code.

Ingestion streams Cases through a temporary SQLite spool and materialized files
into a target-owned content-addressed object area. The authoritative receipt
binds:

- target context and release identity;
- complete policy and frozen runtime policy;
- certificate and assessment-generation digests;
- canonical Case-content digest covering identity/order, facts, assessment, and
  identifiers;
- included and excluded Case-set digests; and
- target asset-inventory digest.

After the target write transaction is acquired, a Bundle compare-and-swap guard
recomputes these inputs before any release row or receipt is published. Exact
retry is idempotent; the same release identity with different content is a
conflict. Existing/new mode is fixed by the Bundle's persisted build-time target
context, so target disappearance cannot grant policy-authoring authority.

## Audit-required changes to the reviewed plan

The implementation did not preserve several initially convenient behaviors
because later audit showed that they could not prove the business invariant.
These changes are mandatory:

| Change | Why it was required |
| --- | --- |
| Introduce one complete `CaseCollection` and immutable Bundle ownership | Split ownership/views could not define lifecycle, synchronization, or stable identity consistently. |
| Make synchronization explicit and full-Bundle | A structure, identity, relation, or custom checker change can affect other Cases; local recomputation was not provably correct. |
| Return no stale derived properties while a draft is unsynchronized | A read-only old grade could look current and mislead users. |
| Keep certification and ingestion from synchronizing implicitly | Hidden reassessment would make certificates and user intent non-reproducible. |
| Separate sample identity, observation comparability, and value disagreement | Structure equality alone cannot prove scientific identity, and same sample alone does not make two measurements comparable. |
| Close and materialize the manifest identity schema | An open mapping or ambient/custom comparator could make grouping non-reproducible; typed exact components and policy digests make equivalence auditable. |
| Keep identity indeterminacy ordinary but make broken operative relations incomplete | Scientific uncertainty should not destroy archivable data, while an invalid relation graph cannot safely support benchmark admission. |
| Require explicit Evidence relation and lineage semantics | Contradictory/context evidence and duplicate work/lineage aliases previously could inflate E grades. |
| Supersede all same-principal attestations atomically | Multiple active decisions from one actor made direct-entry state ambiguous. |
| Use mandatory Ed25519 and OS-isolated custom checkers | Optional cryptography or source inspection alone did not enforce the trust/security boundary. |
| Replace partial implementation hashes with the complete runtime policy | A flat hash did not prove parser selection, protocols, ABI, or relevant distributions and could not distinguish frozen verification from creating a new result. |
| Add generation leases | Atomic publication did not stop deletion between pointer lookup and completion of a read. |
| Bind Case content and recheck it inside the target transaction | Case-set/asset hashes could miss a same-ID scientific-content substitution, and checks before a long spool left a TOCTOU window. |
| Persist build-time dataset mode and restore stored checker artifacts | Live target disappearance must not convert an established-dataset Bundle into a new-policy Bundle, and users should not have to resupply persisted executable policy. |

These corrections serve scientific correctness and reproducibility. They remain
behind the simple Bundle/Case workflow and do not require users to understand
revision tables, digest algorithms, leases, or target transactions.

## Current-version and compatibility result

The live design has one schema/protocol/API for this checkout. Obsolete public
certificate configuration, standalone store ingestion, legacy record cards,
alternate parser/runtime execution, and old database layouts are not retained
as compatibility paths. A non-current store is rebuilt through the audited
owner migration and atomically activated. Git retains implementation history;
runtime code does not emulate it.

## Verification and release result

The detailed incremental evidence, commit IDs, unexpected findings, and push
checkpoints are in the [implementation log](implementation_log.md). Focused
checks cover governance signatures, checker isolation and escape regressions,
attestation replacement, Evidence relations/lineage, scientific identity and
conflicts, assessment generations, certificate round trip, ingestion receipts
and CAS, generation leases, public API signatures, CLI payload schemas, and
formal documentation. The complete pre-cutover suite passed 277 tests.

The first managed-store rehearsal was intentionally discarded when the final
scientific audit found incomplete source-kind, AI-verification, file-Evidence,
owner-provenance, and protocol bindings. The valid 3-dataset/252,665-record
legacy source remained unchanged at 1,136,664,576 bytes and SHA-256
`076096dead161b0689764d275ddd141cdf59e1fe3167881b883877dc8e22b72b`.
This restart was mandatory because changed assessment semantics change the
certificate input; partial output under the earlier semantics could not be
promoted.

The clean current run completed on 2026-10-15:

- wall time: 2:22:39; maximum resident memory: 299,816 KiB; no swap;
- active current-schema database: 5,093,584,896 bytes, SHA-256
  `1a2f02acf77068924d4d9d761220d8c6cb6a2d027930281a8adb1e7c40ab2ee5`;
- 252,665 Cases: 3,568 incomplete, 249,097 ordinary, and zero benchmark;
- 592,436 identifiers, 2,317 target-owned content-addressed assets, and three
  policies, releases, certificates, and authoritative receipts;
- SQLite integrity/foreign-key checks, source/target Case parity, asset parity,
  rollback verification, certificate bindings, and atomic activation passed;
  and
- activation parity SHA-256
  `720a27ade4ab1f6c78a85d767fe2ec4d29fb252b1f4a03929aad78bcd676be76`.

Release classification is scientifically intentional: oxidation has 2,175
ordinary Cases; spin has 2,080 ordinary and 95 incomplete Cases; CSD electronic
state has 244,842 ordinary and 3,473 incomplete Cases. Every Case currently has
E2 text-attribution evidence. Complete Cases are B1 and incomplete Cases B0.
Supplier gold/`Other` assertions remain provenance, not forged User/Hotpot
benchmark approvals, so no Case is promoted to benchmark without the required
verified support or owner decision.

Activated-store API smoke passed discovery, load, archive visibility, indexed
identifier access, slices/subsets, structures/Molecule, field/SMARTS search,
and Bundle/certificate state. CLI smoke passed list/info/labels, get,
quality/evidence subset, the JSON pipeline, multi-structure selection,
`--xyz`/`--smi`, export/conversion, and Bundle status. A late smoke-test finding
showed that the first identifier implementation still hydrated the full CSD
view. The final index/search/CLI corrections (`73a4c6c`, `1738375`, and
`3b18684`) were therefore required for the public query contract: the exact
244,842-visible-Case CSD CLI lookup now completes in 4.64 seconds and unique
matches hydrate only one Case.

After the successful cutover, the one-time schema-1 executable, bootstrap
manifest, and migration-only tests were deleted. Git plus the generated
content-addressed, certificate, receipt, parity, cutover, and rollback artifacts
preserve the audit evidence; the installed current source contains no legacy
converter or compatibility runtime. Both retired imports fail, and freshly
built wheel and source-distribution archives contain neither retired module nor
the bootstrap manifest. The current inventory and query closure were pushed
through `3b18684`. The final current `tests/test_ds` suite passes `259 passed`
in `65.60 s`, with no failures or skips. The Stage 13 release gate is closed.
