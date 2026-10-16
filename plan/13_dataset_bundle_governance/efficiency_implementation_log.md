# Dataset workflow efficiency implementation log

## Safety baseline

- Branch: `perf/dataset-bundle-efficiency`
- Base commit: `80ce620`
- Dataset snapshot:
  `.backups/dataset-pre-efficiency-20261015.tar.xz`
- Snapshot SHA-256:
  `3b2b9ac08003a712aed7f6e685772de31f4dd8e5cc9c76c8bb8fb1c76f52a0d3`
- Snapshot members: 17,693
- Source payload logical size: 25,797,313,655 bytes
- Compression: `xz -T64 -6`
- Compression wall time: 1:48.87
- Compression peak RSS: 8,164,764 KiB
- Validation: `xz -t` passed; full archive listing passed; dataset root,
  active-store symlink, current inventory, and retention manifest are present.

The archive is local, ignored by Git, and must remain available until the
optimized release has passed all correctness, parity, certificate, receipt,
and activated-store checks.

## Implementation rules

1. Each optimization is an independent, reversible commit.
2. Public API and CLI contracts remain unchanged.
3. Proof reuse is limited to one locked immutable operation snapshot.
4. Certification, public verification, ingest boundary A, and transaction
   precommit boundary B remain independent.
5. Deterministic call/scan/hash/SQL counters are CI assertions; wall time and
   RSS are diagnostic measurements.
6. Runtime-policy-bound changes require a clean successor or rebuild. Frozen
   certificates, receipts, and partial artifacts are never promoted.
7. Unrelated working-tree files are preserved and excluded from commits.

## Planned checkpoints

| Step | Scope | Status | Required evidence |
| --- | --- | --- | --- |
| 0 | Correct audit and freeze backup | Complete | verified archive and corrected current-state facts |
| 1 | Add deterministic efficiency regression harness | Complete | selection, certification, synchronization, ingestion, build, and lifecycle evidence recorded |
| 2 | Bulk leased assessment reads and Case selection | Complete | stable IDs/order/object identity; one source proof, generation validation, lease, connection, and assessment scan |
| 3 | Single certification snapshot | Complete | one generation proof and one stable hash per unique bound file per command |
| 4 | Two-boundary bulk ingestion | Complete | two independent bulk proofs; relational spool and set-based publication |
| 5 | Synchronization/generation pass reduction | Complete | one locked input snapshot, streamed supports, fused assessment products, final full validation |
| 6 | Canonical source-proof and mutation refactor | Complete; build residual retained | source proof v4, one-pass stable proof, durable dirty state, append-only audit projection |
| 7 | Full reliability and performance verification | Complete | 354 tests, controlled benchmark, integrity/CAS/API/CLI verification |
| 8 | Current-runtime data rebuild and atomic activation | Complete | 252,665-Case parity, new certificates/receipts, retained rollback, updated inventory |

## Unexpected findings and adjustments

- The pre-cutover report understated `CaseCollection.select`: a three-Case
  selection performs nine complete source-digest computations and six complete
  generation validations.
- Case-declared assets are already deduplicated within one source-digest pass;
  repeated control-asset and cross-proof hashing is the actual issue.
- Synchronization uses shared SQLite connections for supporting rows. The
  target is bounded bulk queries, not merely eliminating per-Case connections.
- The current CSD release has an authoritative receipt; the report's missing
  receipt statement described only the interrupted diagnostic run.
- Read/certify/ingest optimizations can be isolated from runtime-bound
  assessment changes. Synchronization, source-digest, mutation, and canonical
  build changes require a later clean data rebuild.
- Manifest validation initially hid repeated hashing of the CSD Bundle's two
  largest declared control assets (approximately 2.24 GB combined). The
  certification snapshot now validates stored and live manifests directly,
  routes every declared control asset through the same stable per-command hash
  cache, and retains its file stamp through atomic publication.
- The first full rebuild parity pass rejected the CAS lock files as unexpected
  filesystem objects. They are intentional persistent per-digest locks, not
  scientific assets. The verifier was narrowed to accept only zero-byte,
  regular `.<sha256>.lock` files in the matching shard and only when the
  digest is registered. All other hidden, linked, malformed, misplaced, or
  orphan lock files still fail closed. This adjustment was necessary because
  rejecting valid locks made a correct rebuild impossible; broadly ignoring
  hidden files would have weakened the filesystem proof.
- The controlled benchmark found a 36.76% build-time regression alongside the
  large lifecycle gain. The regression is recorded rather than hidden. It did
  not alter scientific semantics or make the optimized lifecycle slower
  overall, so further build work was separated from this release.
- A controlled before/after full-CSD benchmark was not run. Reconstructing and
  executing the historical implementation against restricted multi-gigabyte
  data would add operational risk and was not required to prove activation
  equivalence. Synthetic timings support quantified speed claims; the real
  rebuild supports only correctness, operability, and parity claims.

## Executed checkpoints

### Checkpoint A — Storage staging and publication

- `b3963b3` makes verification and target-owned copying proportional to unique
  asset digests rather than references.
- `0fc0339` replaces reparsed Case JSON and row-at-a-time target publication
  with a bounded relational spool and set-based target inserts.
- A 10,000-Case no-asset synthetic ingest improved from 1.066 s to 0.684 s
  (1.56 times). Deterministic tests also prove one target insert statement per
  bulk Case, assessment, and identifier relation.
- Independent detached-worktree validation after this checkpoint passed all
  266 dataset tests.

### Checkpoint B — Bulk Case selection

- `ae616ca` resolves and verifies one generation, retains one lease, opens one
  read-only assessment connection, and performs one ordered assessment scan.
- Size-parametrized counter tests at 3 and 24 Cases prove constant generation
  work, no per-Case `get_case()` call, stable order, and original Bundle-owned
  Case object identity.
- The focused selection regression passed 38 tests.

### Checkpoint C — Immutable certification snapshot

- `86bf957` captures canonical facts, policy, governance, generation, summary,
  checkers, and scientific assets under one Bundle lock and generation lease.
- `813739b` removes hidden repeated control-asset hashing while preserving
  manifest validation, symlink rejection, stable-file checks, and the
  pre-rename compare-and-swap guard.
- Counter tests prove one complete generation validation and one stable hash
  for each unique facts, generation-manifest, checker, scientific, and declared
  control path. Tamper tests cover scientific and control assets during
  publication.
- The post-checkpoint complete dataset suite passed 270 tests in 71.69 s.

Checkpoint A was pushed through `0fc0339`; Checkpoints B and C were pushed
through `813739b` to `origin/perf/dataset-bundle-efficiency`.

### Checkpoint D — Baseline parity and trust-boundary corrections

- `d031697` captured the active 252,665-Case runtime projection before
  runtime-bound changes. It records 37/37 logical projections, all 2,317 CAS
  objects, and the pre-change certificates and receipts.
- `7f45aca..e35a799` closed issues found by adversarial review: declared Python
  selection runtime, deduplicated-source validation, sealed derived-read error
  semantics, and complete certification generation-tree binding.
- Each correction remained fail-closed and was covered by a focused regression
  before the next optimization began.

Plan conformance: yes. These were required trust-boundary corrections, not
scope expansion. Proceeding without them would have optimized an incomplete
proof.

### Checkpoint E — Bounded ingest proofs and canonical source authority

- `ed64aa2` streams both ingestion proof boundaries through ordered facts and
  assessment cursors without per-Case generation opens.
- `68aa2a9..e9320ff` introduced the normalized facts/source-proof model,
  complete fact semantics, aligned certificate binding, durable revision-dirty
  mutation, and one-pass stable proof publication.
- `4fcd95e`, `4ee9e50`, `b8f5bb8`, and `5467894` prove transactional
  idempotence, crash fences, aggregate snapshot stability, and early rejection
  of conflicting retries.
- `0dbad41` kept CLI ingest on the already validated Bundle snapshot instead of
  constructing an extra proof.

Plan conformance: yes. Facts format changes are current-only; no reader for an
older proof schema was retained.

### Checkpoint F — Synchronization and immutable generation efficiency

- `387084c` streams grouped Case support rows.
- `7ac39e2` binds append-only audit projection content while leaving the facts
  database authoritative.
- `1bf3fde` avoids duplicate generation payload materialization.
- `28d446d`, `05c5ed5`, and `88a4171` hold one immutable synchronization input
  snapshot and close both facts and source-state A/B/A publication gaps.
- `d021167` keeps stale policy state visible without exposing stale derived
  assessment.
- `39da78a` and `1f2b0fa` bind only implemented product helpers into runtime
  identity.
- `6e0c9d8` computes assessment rows, summary, ordered digests, and archive
  indexes during the same streamed traversal, followed by the retained full
  post-publication validation.

Plan conformance: yes. Synchronization remains full-Bundle; no unproved local
incremental assessment or cross-boundary proof cache was introduced.

### Checkpoint G — Complete indexed reads and shared asset work

- `9519ec9` routes certification through the single public immutable snapshot.
- `475ed04` and `f5c12ff` add assessment facet indexes and complete indexed
  Case selection while preserving Python value equality, stable public order,
  original Case ownership, and explicit incomplete opt-in.
- `12a184b` streams each absent CAS object once while hashing and publishing it.
- `ab5e5cd` reuses one scientific asset assessment only under the complete
  content/parser/runtime key.
- `35f1dae` consolidates duplicate work inside each ingest boundary while
  preserving boundary A and transaction-precommit boundary B as independent
  recomputations.
- `75a4cdb` and `f3f7207` reject symlinked Bundle assets, linked CAS objects,
  non-regular nodes, and path substitution.

Plan conformance: yes. Public first-level API and CLI contracts did not grow;
indexes and snapshots are private implementation details.

### Checkpoint H — Reproducible performance measurement

- `4cef61e` added the fixed lifecycle benchmark.
- `782f0ba` recorded three trials for base commit `80ce620` and optimized commit
  `f3f7207`, using 1,000 Cases, eight edits, and `PYTHONHASHSEED=0`.
- All six trials produced semantic SHA-256
  `45baabb87c420460cc485cccd10046b767dd61f671ad64f0593430e2bb5abdb4`,
  `integrity_check=ok`, and zero foreign-key violations.

Median wall results:

| Phase | Baseline (s) | Optimized (s) | Speedup |
| --- | ---: | ---: | ---: |
| build | 1.013 | 1.386 | 0.731x |
| mutation | 0.556 | 0.531 | 1.049x |
| synchronize | 12.818 | 2.318 | 5.530x |
| certify | 7.106 | 1.050 | 6.770x |
| ingest | 16.393 | 3.576 | 4.584x |
| phase sum | 37.886 | 8.860 | 4.276x |

Median peak RSS changed from 230,528 to 224,616 KiB (-2.56%). Build regressed
36.76%; this is a recorded residual, not a claimed improvement. The benchmark
is synthetic and cannot be extrapolated to CSD scale.

Plan conformance: adjusted. The audit originally listed full oxidation, spin,
and CSD before/after timing. The final evidence deliberately separates a
controlled reproducible synthetic benchmark from a real-data parity rebuild.
This prevents an uncontrolled operational run from being reported as a
performance benchmark.

### Checkpoint I — Verified current-only rebuild

- `4bdbef7` added a resumable `plan/run/status/activate/rollback` operator. It
  writes only a new Bundle namespace and a new managed-store generation.
- It validates the original migration proof, copies the exact referenced
  source closure, builds facts schema 4 and assessment schema 3, certifies,
  ingests, and compares source/target scientific projections.
- Parity includes Case and assessment semantics, complete control identities,
  stable generation projections, independently recomputed receipt Case/content
  sets, policy identity, binding uniqueness, and all stable CAS content.
- The first real parity attempt exposed valid persistent CAS lock files.
  `7cb74f1` admitted only structurally and cryptographically bound lock files;
  focused rebuild tests remained 21 passed.
- The resumed run completed all copy/process/parity/ready phases for oxidation,
  spin, and CSD electronic state.

Plan conformance: yes, with the necessary lock-file correction described
above. No partial artifact was promoted, and no legacy proof/runtime reader was
added to the installed package. The explicit one-time operator under `plan/`
continues to validate the schema-v2 source proof until rollback retention ends.

### Checkpoint J — Activation and post-activation validation

- The full parity artifact has semantic digest
  `ffb944dbabbd3e5ca0488a967ae6864350c9408e95d9b07853c4757b25d954e0`
  and file SHA-256
  `a3858d9922892b24494fecad9e3b5cc039af6ca33b584f935f04a962ecac98cb`.
- The stable source/target CAS projection is identical: 2,296 objects,
  2,292,279,163 bytes, SHA-256
  `4bfea9cca55e00db4bef29b6b2fbbe147b586028315714a28070f6c6b8a1914f`.
- Activation `activation-5d037954aa104dfbb6e6d3028833e82a` atomically changed
  `store/active` to `stage13-efficiency-current` at
  `2026-10-16T05:07:00.068761Z`. `stage13-current` remains the recorded rollback
  target.
- The active database contains 252,665 Cases and assessments, passes SQLite
  integrity and foreign-key checks, and has three authoritative receipts.
- A post-activation rehash verified 2,317 CAS objects totaling 7,645,485,715
  bytes and all 2,317 lock files.
- Python API smoke passed dataset discovery/load, IDs, labels, structures,
  subsets, and molecule access.
- CLI smoke passed protocol JSON `list/info/get`, quality/evidence filtering,
  `subset -> get`, multi-structure index/role/ID selection, `--xyz`, converted
  `--smi`, and terminal stderr mirroring without corrupting stdout.
- Bundle status reports synchronized, current certificate, sealed release, and
  ingest-ready state.
- `9a89308` updated the tracked inventory and retention cleanup manifest. The
  prior generation, source migration Bundles, rebuild state, and safety archive
  are retained until the owner ends their retention periods.

Plan conformance: yes. Activation required an exact run ID, reran parity and
CAS checks immediately before swap, and preserved an explicit rollback target.

## Final verification

- `python -m pytest -q tests/test_ds`: 354 passed in 110.41 seconds.
- CLI communication/help focus: 31 passed in 1.17 seconds.
- Rebuild operator focus: 21 passed.
- Three facts databases: schema version 4 and `integrity_check=ok`.
- Three assessment databases: schema version 3 and `integrity_check=ok`.
- Active managed store: `integrity_check=ok`, zero foreign-key violations.
- `hotpot ds detail audit`: `passed=true`, no issues.
- Current inventory database, release, certificate, generation, receipt, and
  activation bindings were independently rechecked.

Reproducible live smoke commands included:

```bash
hotpot ds subset spin -q 1 -e 1 --limit 2 | hotpot ds get spin -
hotpot ds subset spin -q 1 -e 1 --limit 1 | hotpot ds get spin - --xyz --structure-index 0
hotpot ds get spin 0 --smi --structure-index 1
hotpot ds bundle --root hotpot/dataset/data/bundles/stage13-efficiency-current/spin/spin.v001 --dataset spin status
hotpot ds detail audit
```

## Push record

Work was committed in small reversible steps. Pushes were made after groups of
three to five commits where practical; the final activation inventory batch
(`782f0ba`, `7cb74f1`, `9a89308`) was pushed together. Unrelated Plan 12,
Plan 13 README, root plan index, and Plan 14 working-tree changes were never
staged by this work.

## Final adjustment decision

The implementation keeps independent synchronization, certification, ingest
preflight, ingest precommit, rebuild parity, and activation proofs. Efficiency
comes from bounded streaming, indexed selection, deduplication, and proof reuse
inside one immutable boundary only. This is the smallest adjustment that
removes the measured amplification without weakening the data-quality model or
forcing users into a more complex public API.

## Immediate cleanup

- Removed ignored Python bytecode caches produced by the final test/rebuild
  checks.
- Removed the unreferenced zero-byte
  `store/generations/stage13-current/dataset.sqlite3`; the valid rollback store
  is the plural `datasets.sqlite3` and remains intact.
- Removed `generate_efficiency_runtime_parity_baseline.py` after its immutable
  JSON evidence was recorded. It had no caller, hard-coded the superseded
  migration layout, was outside the installed package, and remains recoverable
  from Git history.
- Retained `rebuild_current_runtime.py` and its focused tests because they
  implement the verified rollback path. Both are now declared in the retention
  manifest for removal with the old generation after owner approval.
