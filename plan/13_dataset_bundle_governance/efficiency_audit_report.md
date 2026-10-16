# Stage 13 long-term workflow efficiency audit

Status: audit revalidated against commit `80ce620` after the completed
2026-10-15 managed-store cutover. The confirmed findings were remediated on
branch `perf/dataset-bundle-efficiency`; the current-only runtime was rebuilt,
proved equivalent, and activated on 2026-10-16. Section 14 records the final
performance and reliability evidence.

Before implementation, the complete dataset payload, current inventory, and
retention manifest were archived as
`.backups/dataset-pre-efficiency-20261015.tar.xz`. The verified archive contains
17,693 members and has SHA-256
`3b2b9ac08003a712aed7f6e685772de31f4dd8e5cc9c76c8bb8fb1c76f52a0d3`.
The archive is intentionally excluded from Git.

## 1. Scope

This report consolidates the runtime observations, code-path review, integrity
review, implementation log, and implementation report findings collected while
running Stage 13. Its primary scope is the long-term user workflow that starts
from a standards-compliant raw Bundle intake and ends with a committed release:

```text
1. build -> 2. inspect/edit -> 3. synchronize -> 4. certify -> 5. ingest
```

The one-time schema-1 managed-store migration is not the target architecture
for these optimizations. Its export, parity, rollback, and activation phases
are considered only where they supplied measurements or reveal reusable
runtime problems. Their code-lifecycle treatment belongs to a separate
post-Stage-13 audit and is not an acceptance condition for the efficiency work
described here.

The audit itself authorized no compatibility path or data deletion. The later
implementation used independent, reviewable commits and a separately verified
rebuild/activation procedure.

## 2. Executive conclusion

Stage 13 has the correct high-level integrity model, but the implementation
repeatedly proves the same fact inside a single trust boundary. Large Bundles
therefore pay for repeated JSON parsing, SQLite scans, generation verification,
content hashing, and serialization even when the input snapshot has not
changed.

The central optimization rule is:

> Reuse one proof inside one locked immutable snapshot, but retain independent
> verification at every separate trust and publication boundary.

The following boundaries must remain independent:

1. synchronization proves source facts, scientific assets, policy, governance,
   runtime, parser/checker behavior, and Case assessment;
2. certification independently proves that the published generation and all
   certificate-bound objects still match the Bundle;
3. ingestion proves the certificate before staging and revalidates the same
   bound snapshot inside the target write transaction before publication; and
4. administrative migration parity independently proves source, exported
   Bundle, target store, rollback, and activation state.

Eliminating duplicate work inside one boundary does not weaken certification.
Replacing a later independent boundary with a cached result from an earlier
boundary does weaken certification and is prohibited.

## 3. Evidence baseline

The interrupted 2026-10-15 managed-store migration supplied the original
diagnostic timings. It was not release evidence. Its later clean restart
rebuilt, parity-checked, certified, ingested, and atomically activated all
252,665 Cases under the then-current schema. The historical implementation log
records 2:22:39 end to end. This is context for the audited base, not a timing
for the 2026-10-16 efficiency rebuild or a controlled before/after result.

The CSD electronic-state Bundle contained:

| Item | Observed value |
| --- | ---: |
| Cases | 248,315 |
| `cases.jsonl` | 3,142,624,271 bytes |
| `cases/facts.sqlite3` | 3,357,147,136 bytes |
| `assessment.sqlite3` | 1,661,300,736 bytes |
| Bundle working tree | approximately 9.7 GiB |
| Root certificate | 9,677 bytes |

Approximate artifact-timestamp intervals were:

| Phase | CSD diagnostic interval |
| --- | ---: |
| legacy export | approximately 8 minutes |
| source/export parity | approximately 11 minutes |
| canonical build | approximately 4 minutes |
| synchronization | approximately 12--14 minutes |
| certification | approximately 22 minutes |
| ingestion | interrupted diagnostic run did not publish a receipt; the clean cutover later did |

For comparison, the 2,175-Case oxidation and spin releases each required about
30 seconds to certify and 71--73 seconds to ingest. These values are not a
controlled benchmark. They identify scale-sensitive paths and establish the
need for instrumented measurements before and after remediation.

The current inventory contains the CSD certificate and authoritative receipt.
The certificate size rules out certificate serialization itself as the scale
driver. Code-path counters and multi-gigabyte inputs make repeated verification
the leading explanation for certificate cost, but the timestamp intervals are
not a controlled profiler result and do not establish exact attribution.

Read-only counter instrumentation on current code confirmed the amplification:

| Operation | Small-fixture observation |
| --- | ---: |
| 3-Case `CaseCollection.select` | 9 complete source-digest computations; 6 complete generation validations |
| synchronize | 5 complete source-digest computations; 2 complete Case passes |
| certify | 15 complete source-digest computations; 8 complete generation validations |
| certify facts hashing | 3 complete `facts.sqlite3` hashes |
| certify assessment hashing | 9 complete `assessment.sqlite3` hashes |

These counters are deterministic path diagnostics, not wall-time benchmarks.

## 4. Integrity constraints derived from the implementation record

Optimization must preserve the following implemented findings:

- canonical facts and external scientific-asset state remain source-digest
  inputs;
- every fact mutation is durable, audited, revisioned, and immediately makes
  derived assessment unavailable;
- synchronization remains a full-Bundle operation until a correct dependency
  graph exists;
- parser input is an immutable byte snapshot, not a caller-owned live file;
- generation publication remains atomic and compare-and-swap protected;
- generation readers retain a lease for the complete read operation;
- certification never performs hidden synchronization;
- ingestion never performs hidden certification or synchronization;
- target publication revalidates Case content, selection, assets,
  certificate, generation, policy/runtime, and target identity inside the
  write transaction;
- each materialized release asset is target-owned and content addressed; and
- migration source/export parity, target parity, rollback verification, and
  activation verification are independent computations.

The controlling implementation evidence is recorded in:

- [canonical Bundle authority](implementation_log.md#step-7--build-the-canonical-bundle-fact-authority);
- [immutable assessment generations](implementation_log.md#step-9--publish-immutable-assessment-generations);
- [stable asset snapshots](implementation_log.md#step-15--freeze-bytes-across-hashing-and-chemical-parsing);
- [complete Bundle lifecycle](implementation_log.md#step-16--complete-the-bundle-lifecycle);
- [atomic Case mutations](implementation_log.md#step-19--make-every-case-edit-one-source-transaction);
- [generation leases](implementation_log.md#step-40--protect-assessment-generations-with-leases-and-safe-collection); and
- [ingestion content and target race closure](implementation_log.md#step-44--seal-ingestion-against-content-and-target-races).

## 5. Phase 1: canonical build

### Current work

Canonical construction currently:

1. parses and validates every intake Case;
2. serializes and inserts every canonical Case;
3. reparses every stored payload to compute the source digest;
4. hashes referenced assets;
5. atomically publishes `facts.sqlite3`; and
6. immediately reopens the new store, runs SQLite integrity validation, and
   validates every Case again.

The implementation is bounded-memory, but a canonical payload is still parsed
and normalized multiple times. Inserts are issued one Case at a time inside one
transaction.

### Bottlenecks

- repeated full JSON parsing and canonical serialization;
- repeated Case schema validation immediately after successful construction;
- full source-digest reconstruction after the same payloads were just
  canonicalized;
- one-statement-per-Case dispatch rather than batched writes; and
- repeated control-asset hashing across manifest validation and later proof
  layers. Case-declared source assets are already deduplicated by relative path
  inside one source-digest computation.

### Safe remediation

1. Validate, canonicalize, derive the Case content digest, and batch-insert in
   one streaming pass.
2. Build an ordered source digest from domain-separated canonical Case and
   asset digests. The algorithm must be explicitly defined and runtime-bound.
3. Hash each unique source or control asset once per build snapshot.
4. Use bounded `executemany` batches while preserving one atomic transaction.
5. Retain one end-of-build SQLite integrity check and atomic fsync/rename.
6. Retain full validation for every later independent `load_bundle` operation.
7. Avoid a second complete Case-schema pass immediately after construction
   when the build already produced and checked the same canonical bytes.

### Prohibited shortcut

Do not remove the final database-integrity check or turn later loads into
implicit repair/import operations. A build-created in-memory object may reuse
the proof produced by that build; a separately loaded object must validate the
persisted authority.

### Acceptance measures

- each intake Case is fully parsed no more than once during normal build;
- each canonical payload is serialized no more than once before storage;
- every unique source asset is hashed once per stable build snapshot;
- peak memory remains bounded independently of Case count; and
- crash/retry and duplicate-identity tests retain their current behavior.

## 6. Phase 2: inspect and edit

### Current work

The current `CaseCollection.select` path constructs Cases individually and
accesses derived assessment through per-Case calls. A derived read may reopen
and fully verify the generation, while a bulk selection repeats that work for
many Cases. This is an N+1 query and verification pattern.

Current instrumentation shows that the amplification is greater than a normal
N+1 query: a synchronized three-Case selection performed nine complete source
digest computations and six complete generation validations. At large scale,
the current path can therefore approach quadratic work in Bundle size.

Each Case mutation also recomputes the complete Bundle source digest and
rewrites the complete audit JSONL projection. Repeated interactive edits can
therefore approach `O(number_of_edits * Bundle_size)` work even though each
edit changes one Case.

### Bottlenecks

- one SQLite/generation access per selected Case;
- repeated source-state and generation validation during one bulk read;
- full source-digest reconstruction after every small edit; and
- complete audit-projection rewrite after every new event.

### Safe remediation

1. Resolve one synchronized generation once, hold one lease, and execute Case
   selection directly against indexed assessment rows.
2. Return the original Bundle-owned Case objects from the selected ID list, so
   public ownership and view semantics do not change.
3. Keep raw facts readable while continuing to reject every derived read from
   an unsynchronized Bundle.
4. For mutation, either:
   - transactionally update a cryptographic per-Case/per-asset digest index and
     its deterministic aggregate; or
   - transactionally set an explicit fail-closed `source_digest_state=dirty`
     and rebuild the complete digest at synchronization.
5. Append and fsync audit projections, or regenerate them lazily from the
   authoritative audit table. A projection is never a second authority.
6. Reuse validation results only for the duration of one locked operation.

The current assessment schema has no archive/quality/evidence/benchmark
selection indexes, and verification/attestation criteria live in canonical
facts. The remediation must add explicit private indexes or projections rather
than merely moving the existing Python loop.

### Conditional requirement for deferred digest work

Merely leaving an old source digest in a field that appears current is not
acceptable. A deferred digest must have an explicit durable dirty state in the
same fact transaction. Unsynchronized derived reads, certification, and
ingestion must fail closed after a crash or process restart.

### Prohibited shortcut

Do not use a process-persistent path/mtime cache as authoritative evidence that
external facts or assets have not changed. A new process must still detect an
out-of-band edit that bypassed revision counters.

### Acceptance measures

- one `select` operation uses one generation lease and a bounded number of SQL
  statements independent of result count;
- one Case mutation performs work proportional to the changed Case and digest
  index, not the multi-gigabyte Bundle;
- every mutation remains one source transaction, revision, and audit event;
- derived reads fail immediately while dirty; and
- out-of-band fact/asset mutation remains detectable before synchronization,
  certification, or ingestion can succeed.

## 7. Phase 3: synchronize

### Current work

Synchronization currently refreshes source and governance state, reopens or
revalidates source state during generation construction, streams source facts
into work and assessment databases, traverses Cases again for final assessment,
queries supporting rows per Case on shared connections, computes ordered
digests and summaries,
VACUUMs the temporary assessment database, writes three indexes, hashes the
generation inventory, reopens and fully validates the new generation, and
recomputes source state for publication CAS.

Conflict candidates are grouped by sample/property and compared pairwise.
Current observed groups are small, but the worst-case group cost is quadratic.
The current CSD generation's largest observed candidate-pair count is only
three, so conflict comparison is not a present priority without new evidence.

Present local assets are copied to a private snapshot and repeatedly hashed so
parsing and hashing refer to stable bytes. This protection was added after the
implementation audit identified an `A -> B -> A` live-file race.

### Bottlenecks

- repeated complete source-state scans within one synchronization;
- multiple source Case passes and per-Case supporting-row queries;
- full `VACUUM` rewrite of a temporary multi-gigabyte database;
- separate passes for summary, ordered digest, and archive indexes;
- duplicate build-time digest/summary/index traversals followed by one complete
  post-publication generation validation; and
- potentially quadratic conflict groups.

### Safe remediation

1. Establish one synchronization input snapshot and expected revisions/digests
   under the Bundle lock.
2. Use ordered streaming cursors and merge facts, identity, assets, conflicts,
   and assessment rows without per-Case supporting-row query dispatch.
3. Compute summary, ordered assessment digest, and archive indexes during one
   deterministic assessment traversal.
4. Remove `VACUUM` from the temporary generation path unless a measured target
   requirement justifies its full rewrite.
5. Retain exactly one complete post-publication generation validation, then
   reuse that validated object within the same synchronization operation.
6. Cache assessment of one unique asset only by the complete key comprising
   content digest, format, parser binding, parser implementation, parser
   distributions, and runtime policy.
7. Preserve one immutable parser input snapshot and verify source/snapshot
   stability before publication.
8. Partition conflict candidates only with a comparability key proved
   equivalent to the persisted property policy. Built-in comparator
   optimizations must be differential-tested against the pairwise reference.
9. Preserve pairwise evaluation for custom or otherwise unproven comparators.
10. Keep full-Bundle synchronization and final revision/digest CAS.

### Prohibited shortcuts

- no unproven local/incremental reassessment;
- no group-size threshold that silently skips conflict candidates;
- no ordinary path cache in place of stable source/snapshot verification;
- no parser-result reuse across a different runtime/parser binding; and
- no removal of the final generation integrity proof.

### Acceptance measures

- no per-Case supporting-row query dispatch or generation open;
- a bounded number of complete source passes documented by instrumentation;
- one final generation integrity verification per synchronization;
- no temporary full-database `VACUUM` unless explicitly measured and approved;
- identical Case assessments, group digests, summaries, and archive indexes
  compared with the pre-optimization implementation; and
- pairwise-oracle equivalence tests for every optimized comparator.

## 8. Phase 4: certify

### Current work

Certificate creation currently reaches synchronization status and generation
validation through several helper layers. It inventories control files,
scientific assets, facts, generation files, checker artifacts, and summary,
then builds a certificate and verifies the newly built result against the
Bundle. The same multi-gigabyte facts/generation content may be scanned and
hashed repeatedly during one certificate command.

This is the clearest measured long-term bottleneck: the 248,315-Case CSD
certificate was only 9,677 bytes but took approximately 22 minutes to write.
The interval is confirmed by artifact timestamps, but it is diagnostic rather
than a controlled benchmark.

On a small current fixture, one certificate command performed 15 complete
source-digest computations, eight complete generation validations, three full
facts-store hashes, and nine full assessment-store hashes. This establishes
duplicate work independently of the wall-time attribution.

### Safe remediation

Introduce one private `CertificationSnapshot` with the following semantics:

1. acquire the Bundle lock and generation lease;
2. require synchronized current-runtime state;
3. open and safely resolve every bound file;
4. compute the source state once;
5. fully validate the generation once;
6. hash facts, generation files, checker files, governance, and each unique
   scientific asset once with before/after stability checks;
7. construct and schema-validate the certificate from those exact results;
8. compare every certificate field with the snapshot in memory;
9. recheck the authoritative state/CAS immediately before atomic rename; and
10. fsync the certificate and parent directory.

The independent public/restoration verification path remains a full disk
verification. Only duplicate validation inside one locked certificate command
is removed.

### Prohibited shortcuts

- no certificate derived solely from stored digest strings;
- no omission of negative scientific-asset state;
- no reuse of a validation result across a lock/lease boundary;
- no hidden synchronization; and
- no weakening of frozen/current runtime-policy rules.

### Acceptance measures

- one complete generation verification per new certificate command;
- one facts-file content hash per command;
- one hash per unique scientific/checker/control asset per command;
- byte-identical logical bindings and successful certificate round trip;
- all current tamper, path, negative-asset, governance, runtime, and generation
  tests continue to pass; and
- target diagnostic goal: reduce the CSD certificate phase by at least three
  times before considering more invasive work.

## 9. Phase 5: ingest

### Current work

Ingestion currently verifies the certificate before snapshot construction,
again immediately afterward, and again in the target-transaction precommit
boundary. Snapshot construction repeatedly traverses source Cases,
opens an assessment connection per Case, hashes assets, and computes content
and selection digests. Target ingestion serializes complete normalized Cases to
a temporary SQLite spool, rereads the spool to verify digests, acquires the
target transaction, executes another full Bundle/certificate/snapshot check,
then reparses each spooled JSON document and inserts target rows one by one.

Asset storage hashes a source before copy and reads/hashes it again during copy.
An existing target object can also be rehashed for every reference rather than
once per unique content digest.

The current precommit recomputation is not accidental overhead. It was added
after the audit demonstrated that Case-set and asset digests alone did not
detect same-ID Case-content substitution, and that a long spool creates a
TOCTOU interval.

Static path counting shows approximately five per-Case assessment lookups for
one ingest. At CSD scale this is about 1.24 million generation connections and
point queries. The full normalized Case JSON is serialized once and parsed
twice. Target writes are one atomic transaction, not one transaction per row,
but still use per-Case and per-relation SQL dispatch.

### Safe remediation

1. Keep two independent proof points:
   - complete validation before staging; and
   - complete bound-snapshot revalidation inside the target write transaction.
2. At each proof point, use one ordered facts cursor and one ordered assessment
   cursor rather than per-Case `get_case` calls.
3. Represent the expected ingestion proof in one private immutable
   `IngestionSnapshot`, including target context, policy/runtime, certificate,
   generation, Case content, included/excluded sets, and asset inventory.
4. Compute content/set/asset digests during a single ordered stream.
5. Replace full JSON spool round trips with either:
   - read-only attached certified facts/assessment databases and set-oriented
     insertion; or
   - a typed normalized spool that supports `INSERT ... SELECT`.
6. Keep an independent digest verification of whatever staging representation
   is committed.
7. Batch target Case, assessment, identifier, structure, and asset-reference
   writes.
8. Copy each unique absent asset once while hashing the copied stream; compare
   it with the certified expected digest and fsync it before reference
   publication.
9. Verify each existing target CAS object once per unique digest per ingest
   command, not once per reference.
10. Recompute the full required logical proof inside precommit using the bulk
    readers, and compare it with the expected `IngestionSnapshot`.
11. Retain target transaction, target preconditions, receipt atomicity,
    idempotent retry, and sealed-release behavior.

### Prohibited shortcuts

- precommit must not be reduced to revision, mtime, or an in-memory Boolean;
- a pre-transaction check cannot replace in-transaction revalidation;
- the receipt cannot prove itself;
- staging serialization cannot silently change canonical Case content; and
- target-owned assets cannot be replaced with caller-owned paths.

### Acceptance measures

- no per-Case generation connection;
- exactly two independently measured proof boundaries, without extra duplicate
  scans inside either boundary;
- Case input is streamed a bounded number of times and never materialized as a
  full Python collection;
- every unique asset is copied/verified once per command;
- receipt semantics and bound target projections remain identical for the same
  scientific input; receipt identities may differ when runtime/target context
  differs;
- same-ID content substitution and every existing TOCTOU regression still
  fail; and
- controlled lifecycle measurements record wall/CPU time, process peak RSS,
  semantic output, integrity, and unique-asset count; the real oxidation, spin,
  and CSD rebuild separately proves complete ingest and parity.

## 10. Cross-phase observability requirements

The remediation must begin with private instrumentation rather than speculative
rewrites. Instrumentation must not add first-level public API names or alter
the machine stdout protocol.

The following is the preferred diagnostic matrix when investigating a specific
phase. It is not a requirement to add permanent instrumentation for every item
when deterministic operation counters, controlled lifecycle measurements, and
full-data parity already test the optimization being claimed:

- monotonic wall time and process CPU time;
- peak resident memory;
- Cases, observations, evidence rows, structures, and unique assets processed;
- SQLite connections and statements by operation class;
- sequential bytes read and written per major file;
- full source-digest computations;
- full generation validations;
- facts/generation/certificate/asset hash passes;
- parser and custom-checker invocations;
- spool and target database sizes;
- conflict group count, maximum group size, and pair-comparison count; and
- final output digests and result counts.

Human progress and performance diagnostics belong on stderr or in an explicit
audit artifact. Closed CLI stdout envelopes must remain unchanged.

## 11. Remediation sequence

With forward implementation and cutover evidence complete, remediation uses
small reversible commits in this order:

1. add private performance instrumentation and fixed regression datasets;
2. remove N+1 assessment/source access in inspect and ingest paths;
3. implement the single-snapshot certificate proof;
4. implement bulk/two-boundary ingestion without weakening precommit proof;
5. optimize transactional dirty/digest handling for edits;
6. optimize synchronization passes, temporary database work, and safe
   comparator grouping;
7. optimize canonical build;
8. run full correctness, tamper, concurrency, API, and CLI suites;
9. benchmark a fixed synthetic lifecycle before and after the change, and use
   the actual oxidation, spin, and CSD rebuild for semantic parity and
   operational validation; a separate controlled full-CSD before/after run is
   required before claiming a CSD-scale speedup;
10. update the implementation log/report with measured before/after results;
11. only then execute the code-lifecycle separation audit and cleanup described
    in the companion report.

Every change to assessment, parser, checker, digest, generation, certificate,
or receipt semantics changes current runtime bindings. Partial migration
artifacts produced under an earlier bound implementation must not be promoted.

## 12. Release gate for the optimization work

Performance remediation was accepted only after all of the following held:

1. the complete current dataset test suite passes;
2. tamper, negative-asset, A/B/A asset-race, generation-lease, precommit race,
   idempotence, parity, and sealed-release tests pass;
3. public API and CLI signatures remain unchanged unless separately approved;
4. no compatibility implementation or historical parser/runtime is added;
5. assessment and receipt outputs are scientifically and logically equivalent,
   or an intentional current-version change is explicitly reviewed and forces
   a clean rebuild;
6. each claimed speedup has recorded counters explaining the removed work;
7. no independent trust boundary has been replaced by cached evidence from an
   earlier boundary; and
8. any runtime-policy-bound change is validated through a clean successor or
   rebuild; no frozen certificate, receipt, or partial artifact is reused as a
   current result.

## 13. Revalidation verdict

The build, inspect/edit, synchronize, certify, and ingest amplification claims
were real at the audited base commit `80ce620`. Three qualifications controlled
the remediation:

1. case-declared asset paths are already deduplicated within one source-digest
   pass; the remaining problem is repeated passes and repeated control-asset
   hashing;
2. generation construction performs one complete post-publication validation,
   while the avoidable duplication is chiefly the work repeated before and
   around that validation; and
3. certificate and ingest timings are diagnostic observations. Deterministic
   scan/hash/connection counters, plus controlled before/after benchmarks, are
   required to claim a quantified speedup.

The precommit ingest proof, public certificate verification, final generation
integrity proof, source/publication CAS, and stable parser-input snapshot are
not redundant trust boundaries and must remain independent.

## 14. Final implementation evidence

### 14.1 Controlled performance result

The reproducible benchmark used 1,000 Cases, eight edits, three trials per
variant, and `PYTHONHASHSEED=0`. It compared base commit `80ce620` with optimized
commit `f3f7207`. Medians are:

| Phase | Baseline (s) | Optimized (s) | Speedup | Change |
| --- | ---: | ---: | ---: | ---: |
| build | 1.013 | 1.386 | 0.731x | +36.76% |
| mutation | 0.556 | 0.531 | 1.049x | -4.64% |
| synchronize | 12.818 | 2.318 | 5.530x | -81.92% |
| certify | 7.106 | 1.050 | 6.770x | -85.23% |
| ingest | 16.393 | 3.576 | 4.584x | -78.19% |
| phase sum | 37.886 | 8.860 | 4.276x | -76.61% |

Median process peak RSS fell from 230,528 to 224,616 KiB (-2.56%). The build
regression is real and is not hidden by the aggregate improvement; further
build optimization requires its own evidence and is not a release blocker for
the large synchronize/certify/ingest reductions. All six runs produced the
same scoped semantic digest
`45baabb87c420460cc485cccd10046b767dd61f671ad64f0593430e2bb5abdb4`,
passed SQLite integrity checking, and had zero foreign-key violations.

These are synthetic lifecycle measurements, not an extrapolation to CSD. No
controlled historical/current full-CSD timing comparison was run. The real
data rebuild below proves correctness and operability, not a quantified
CSD-scale speedup. Raw and summarized results are in
`efficiency_benchmark_results.jsonl` and `efficiency_benchmark_summary.json`.

The original audit proposed an exhaustive wall/CPU/RSS/bytes/SQL/parser/
conflict matrix for every real-data phase. The final release gate narrows that
proposal to evidence needed for the claims actually made: deterministic
connection/query/hash/pass counters, controlled lifecycle wall/CPU/RSS
measurements, and complete real-data parity. This adjustment is necessary to
avoid permanent invasive instrumentation and measurement overhead that do not
strengthen either the reported synthetic speedup or the full-data reliability
proof. The omitted matrix remains a targeted profiling option, not satisfied
evidence and not a basis for any claim in this report.

### 14.2 Full-data reliability result

The one-time current-only rebuild processed all 252,665 Cases and compared the
source and candidate stores using complete scientific projections, control
identities, receipt semantics, stable generation projections, and stable CAS
content. The semantic digest is
`ffb944dbabbd3e5ca0488a967ae6864350c9408e95d9b07853c4757b25d954e0`.
The parity artifact SHA-256 is
`a3858d9922892b24494fecad9e3b5cc039af6ca33b584f935f04a962ecac98cb`.

The candidate became `store/active` at
`2026-10-16T05:07:00.068761Z`. The previous `stage13-current` generation remains
the explicit rollback target. Post-activation verification established:

- 3 datasets, 3 releases, and 252,665 Cases/assessments;
- SQLite `integrity_check=ok` and zero foreign-key violations;
- facts schema version 4 and assessment schema version 3 for all three
  rebuilt Bundles;
- 2,317/2,317 CAS objects rehashed successfully, totaling 7,645,485,715
  bytes, plus 2,317 validated lock files;
- the stable 2,296-object source/target CAS projection exactly matched the
  source: 2,292,279,163 bytes and SHA-256
  `4bfea9cca55e00db4bef29b6b2fbbe147b586028315714a28070f6c6b8a1914f`;
- the final dataset suite passed 354 tests in 110.41 seconds; and
- live API and CLI checks passed dataset discovery/load, Bundle status,
  `subset -> get` JSON piping, multi-structure index/role/ID selection, XYZ
  rendering, XYZ-to-SMILES conversion, and protocol/human-output separation.

The activated store, current inventory, parity proof, certificate and receipt
bindings therefore satisfy the release gate. Historical migration Bundles,
the previous generation, and resumable rebuild state remain retained under
`hotpot/dataset/retention-cleanup.txt`; no retained rollback material was
deleted during this work.
