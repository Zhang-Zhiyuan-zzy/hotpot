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
| 1 | Add deterministic efficiency regression harness | Pending | baseline counters and existing correctness suite |
| 2 | Bulk leased assessment reads and Case selection | Pending | stable IDs/order/object identity; bounded generation work |
| 3 | Single certification snapshot | Pending | one generation proof and unique-file hashing per command |
| 4 | Two-boundary bulk ingestion | Pending | exactly two independent proofs; no per-Case assessment connection |
| 5 | Synchronization/generation pass reduction | Pending | identical Case assessments, summaries, indexes, and classifications |
| 6 | Canonical build and mutation efficiency | Pending | equivalent canonical facts/digests and fail-closed dirty behavior |
| 7 | Full reliability and performance verification | Pending | complete suite, attack tests, before/after counters and timings |
| 8 | Current-runtime data rebuild and atomic activation | Pending | parity, certificate, receipt, inventory, rollback, API/CLI smoke |

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
