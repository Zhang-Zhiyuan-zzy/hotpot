# Dataset workflow CPU parallelization implementation plan

Status: proposed implementation plan. This document does not authorize a new
compatibility path or a weaker proof boundary. It defines one deterministic
execution graph that may use one or more workers.

## 1. Objective

Reduce the wall time of the current Bundle lifecycle and full managed-store
rebuild while preserving:

- byte-identical canonical facts for the same inputs;
- identical Case assessments, archive classes, summaries, and ordered digests;
- stable source-file observations before and after every read;
- independent synchronization, certification, ingest-preflight,
  ingest-precommit, parity, and activation proofs;
- one SQLite writer per database;
- atomic generation, certificate, CAS, receipt, and active-link publication;
- the current concise public API and closed CLI stdout protocol; and
- one current implementation, without serial/parallel compatibility forks.

The first target is eight workers. The machine has 64 physical cores, but the
current workload contains single-writer SQLite sections, uneven multi-gigabyte
files, and trust boundaries that must remain sequential. Eight workers capture
most available parallelism without excessive process memory or I/O contention.

## 2. Measured baseline

### 2.1 Current managed-store size

The formally ingested active generation is 12,744,683,520 bytes (11.87 GiB):

| Component | Bytes |
| --- | ---: |
| `datasets.sqlite3` | 5,093,584,896 |
| 2,317 CAS objects | 7,645,485,715 |
| CAS locks and activation metadata | approximately 5.5 MiB |

Bundle working directories are not part of this total.

The latest Bundle working tree and resumable rebuild state are stored as the
verified cold archive described by
`hotpot/dataset/manifests/stage13-efficiency-intermediate-archive.json`. A
future implementation benchmark must restore that archive into a new run root,
verify its recorded SHA-256, and must not overwrite the active generation.

### 2.2 Current lifecycle profile

The controlled 1,000-Case optimized benchmark produced these median values:

| Phase | Wall (s) | CPU (s) | CPU/wall | Eight-core mathematical ceiling |
| --- | ---: | ---: | ---: | ---: |
| build | 1.386 | 1.144 | 82.5% | 3.60x |
| mutation | 0.531 | 0.036 | 6.7% | 1.06x |
| synchronize | 2.318 | 1.703 | 73.5% | 2.80x |
| certify | 1.050 | 1.023 | 97.5% | 6.81x |
| ingest | 3.576 | 3.064 | 85.7% | 3.99x |

The ceiling assumes that all measured CPU time parallelizes and all non-CPU
time remains fixed. It is deliberately optimistic.

### 2.3 Current full-data rebuild

Persisted rebuild state records:

| Event | Timestamp or duration |
| --- | ---: |
| run created | 2026-10-16 03:52:52Z |
| source copy complete | 1 min 20 s |
| oxidation processed | approximately 1 min 5 s |
| spin processed | approximately 1 min 6 s |
| CSD processed | approximately 30 min 49 s |
| final parity | approximately 22 min 57 s |
| ready | 2026-10-16 04:50:09Z |
| run-to-ready | 57 min 17 s |

The later activation timestamp includes operator delay and is not a clean
activation-phase benchmark.

### 2.4 Hashing and storage measurements

On the current host:

- a 1 GiB direct read completed in 0.44 s, approximately 2.4 GB/s;
- SHA-256 of the 5.09 GB active database took 17.61 s at 99% CPU,
  approximately 0.29 GB/s;
- eight representative files totaling 19.44 GiB required an estimated 72.2 s
  sequentially and 17.79 s with eight hash processes; and
- the observed representative hashing speedup was 4.06x, with 407% average CPU
  utilization because the largest file determined the final tail.

Large-file certification and parity are therefore primarily single-core hash
limited on this machine, not storage-throughput limited.

## 3. Execution model

### 3.1 One task graph

Implement one private execution engine with these properties:

```text
ordered task source
    -> bounded worker pool
    -> sequence-tagged immutable results
    -> bounded reorder buffer
    -> one deterministic reducer / SQLite writer
```

`workers=1` executes the same tasks and reducer directly. It is a resource
setting, not a legacy implementation. Tests must compare one-worker and
eight-worker results byte for byte or by the existing canonical logical
digests where timestamps are intentionally variable.

### 3.2 Shared worker budget

Use one operation-scoped worker budget rather than creating eight workers in
every nested helper. A weighted semaphore prevents oversubscription across
datasets, hashing, parsing, and Case assessment.

Initial policy:

- automatic default: `min(8, available physical cores)`;
- explicit `workers=1`: deterministic low-resource execution;
- maximum accepted value: available logical CPUs;
- subprocess work: spawn-safe process pool with bounded input/output queues;
- hashing/copy work: thread pool because OpenSSL hashing and file I/O release
  the GIL; and
- no process-persistent content cache across trust boundaries.

### 3.3 Minimal configuration surface

Do not add a first-level public class. Add the same optional `workers` keyword
only to heavy operations and the corresponding CLI commands:

```python
bundle.synchronize(workers=None)
bundle.write_certificate(workers=None)
bundle.ingest(..., workers=None)
ds.load_bundle(..., workers=None)  # only when canonical build is required
```

```text
hotpot ds bundle ... synchronize --workers 8
hotpot ds certify ... --workers 8
hotpot ds ingest ... --workers 8
```

`None` selects the capped automatic policy. Read-only dataset queries do not
need a worker option. Worker diagnostics belong on stderr or in an audit file;
protocol stdout remains unchanged.

## 4. Implementation stages

### Stage 0 — Instrumentation and fixed baselines

1. Add operation-scoped counters for worker tasks, bytes hashed/copied,
   reorder-buffer high-water mark, SQLite write batches, and wall/CPU time.
2. Preserve the current six benchmark rows and full rebuild parity artifact as
   immutable baselines.
3. Add a reproducible large-file fixture with uneven file sizes so load-balance
   regressions are visible.

Acceptance:

- no public API additions other than the common optional worker count;
- no stdout protocol change; and
- instrumentation disabled or near-zero overhead outside benchmark mode.

### Stage 1 — Parallel stable-file hashing

Create a private parallel snapshot hasher used by certification, parity, CAS
verification, and activation reproof.

For each file task:

1. open without following symlinks;
2. require a regular file;
3. record `fstat` before reading;
4. hash the complete byte stream;
5. record `fstat` after reading and compare it with the live path;
6. return digest, size, and stable stamp; and
7. assemble results in canonical path order.

Only independent files run concurrently. A standard SHA-256 digest for one
file is not split into a new tree-hash format. Files below a measured threshold
remain inline to avoid scheduling overhead.

Expected benefit:

- real CSD certification: 3.0-4.0x;
- parity/activation file hashing: 3.0-4.1x; and
- small oxidation/spin certificates: neutral to 1.3x.

Risk: A/B/A file replacement or worker failure must invalidate the entire
snapshot. Existing tamper and symlink tests must run at both one and eight
workers.

### Stage 2 — Parallel unique-asset assessment

Parallelize structure parsing and chemical validation by unique complete cache
key:

```text
(content digest, format, parser binding, parser implementation,
 parser distributions, runtime policy)
```

Use processes, not threads, for OpenBabel/RDKit work. Workers receive immutable
snapshot paths and return plain canonical data. The parent process performs all
SQLite writes and deterministic reduction.

Expected benefit:

- structure-heavy synchronization: 2.0-4.0x;
- current CSD synchronization, which has no materialized structures: smaller,
  approximately 1.2-1.5x from this stage alone.

Risk: parser crashes, excessive worker memory, and non-deterministic library
state. Each worker must be replaceable; a failed task fails the complete
generation rather than silently downgrading a structure.

### Stage 3 — Parallel Case canonicalization and assessment

Use bounded process batches for CPU-heavy JSON decoding, schema validation,
canonical serialization, independent Gate preparation, and per-Case assessment
inputs.

The parent process must retain:

- identity grouping and conflict graph construction;
- custom checker authority and isolation policy;
- canonical public-index order;
- all ordered digest reducers; and
- the single SQLite writer.

Workers return sequence-tagged immutable records. The reorder buffer must be
bounded by worker count times batch size. Start with 256 Cases per task and tune
against CSD.

Expected benefit:

- canonical build: 1.5-2.3x;
- current CSD synchronization overall after Stages 2 and 3: 1.5-2.2x; and
- mutation: no meaningful benefit.

Mutation should instead gain a separate bulk-edit method that validates Cases
in parallel but commits one revision and one SQLite transaction. Individual
mutations remain serialized.

### Stage 4 — Parallel CAS source verification and copy

Schedule work by unique content digest. Each task retains the current protocol:

1. acquire the per-digest lock;
2. verify every distinct source path bound to that digest;
3. verify an existing target object once, or stream one absent object to a
   private temporary file while hashing;
4. fsync, compare digest/size/stamps, and atomically rename; and
5. return a publication record to the single database transaction.

Never publish database references from worker processes. Ingest preflight and
transaction-precommit remain independent proofs and must not execute
concurrently with each other.

Expected benefit:

- empty-target ingest: 1.5-2.3x overall;
- existing-CAS retry: 1.2-1.8x; and
- asset-heavy releases may improve more until the storage device saturates.

### Stage 5 — Dataset-level scheduling

Build/synchronize/certify oxidation and spin concurrently with CSD under the
same eight-worker budget. Serialize target-store commit transactions. This
hides the approximately two minutes currently spent processing the two small
datasets without creating concurrent SQLite writers.

The CSD critical path remains dominant; dataset-level concurrency is a final
tail optimization, not a substitute for Stages 1-4.

### Stage 6 — Reliability and release validation

Required tests:

- one-worker/eight-worker differential output tests;
- repeated-run deterministic digest tests;
- worker crash and cancellation cleanup;
- source mutation before, during, and after worker reads;
- symlink, non-regular file, malformed lock, and path-substitution attacks;
- parser crash and timeout behavior;
- SQLite busy/rollback and disk-full recovery;
- generation lease and garbage-collection races;
- certificate publication compare-and-swap;
- ingest boundary-A/boundary-B substitution attacks;
- idempotent retry and sealed-release behavior;
- complete `tests/test_ds`; and
- a clean 252,665-Case rebuild into a new generation followed by the existing
  parity and activation proof.

## 5. Efficiency projection for the current database

### 5.1 Per-phase expectation with eight workers

| Work | Current | Expected | Basis |
| --- | ---: | ---: | --- |
| source closure copy | 1.3 min | 1.0-1.3 min | primarily I/O; limited parallel gain |
| oxidation + spin processing | 2.2 min | hidden behind CSD | dataset-level scheduling |
| CSD build/sync/certify/ingest | 30.8 min | 13-18 min | weighted 1.7-2.3x lifecycle gain |
| final source/target parity | 23.0 min | 6.5-9.5 min | measured 4.06x hashing, serial row reducers retained |
| run-to-ready | 57.3 min | 22-30 min | includes scheduling and safety margin |
| activation reproof + atomic swap | not cleanly measured | 3-5 min | independent parallel reproof retained |
| complete run-to-active | approximately 57 min plus operator delay | 25-35 min | conservative release estimate |

The expected headline is therefore **about 25-35 minutes from a clean start to
an independently reverified active generation**, compared with 57 minutes to
ready in the current run. The target is approximately 2.0-2.3x end-to-end,
not eight times faster.

An optimistic lower bound is around 20 minutes, but it is not an acceptance
commitment. SQLite serialization, the largest individual file, ordered digest
reducers, and independent trust boundaries prevent linear eight-core scaling.

### 5.2 Release performance gates

Accept the implementation only if the fixed host and inputs demonstrate:

- certification at least 3.0x faster for the CSD Bundle;
- final parity at least 2.5x faster;
- complete run-to-ready no more than 30 minutes;
- complete run-to-active no more than 35 minutes excluding human delay;
- no small-dataset phase more than 10% slower;
- peak RSS below 8 GiB for the dataset workflow;
- no more than eight CPU-intensive workers;
- identical scientific parity digest and stable CAS projection; and
- all reliability tests pass.

If a gate fails, retain the single task graph, reduce worker count or task
granularity, and remeasure. Do not restore a separate serial implementation.

## 6. Commit and rollout sequence

1. instrumentation and executor contracts;
2. parallel snapshot hashing plus adversarial tests;
3. process-isolated unique-asset assessment;
4. bounded Case canonicalization/assessment workers;
5. parallel unique CAS materialization;
6. dataset-level scheduler and shared worker budget;
7. controlled benchmarks and full current-data rebuild;
8. parity, activation, inventory, and final report.

Each step is independently reversible and should be pushed after three to five
small commits. No step may activate a partially rebuilt generation.

