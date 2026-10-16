# Dataset Bundle efficiency implementation report

## Outcome

The Stage 13 efficiency remediation is complete on
`perf/dataset-bundle-efficiency`. The implementation removes repeated work
inside a single immutable operation while retaining independent proofs at the
synchronization, certification, ingest-preflight, ingest-precommit, rebuild,
and activation boundaries.

The optimized current-only runtime was rebuilt from the three active releases,
proved semantically equivalent, and atomically activated on 2026-10-16. The
previous generation remains available for rollback. No compatibility runtime,
old schema reader, or implicit upgrade path was added to the installed
`hotpot.ds` package. The explicitly retained one-time operator under `plan/`
can read the proved schema-v2 migration source until rollback retention ends.

## Implemented changes

- Bulk Case selection resolves and leases one assessment generation, then uses
  indexed ordered reads without changing Case ownership or order.
- Certification captures one locked immutable snapshot and hashes each unique
  bound file once within that command.
- Ingestion keeps two independent proofs but streams each through bounded
  ordered facts/assessment readers; target publication uses a normalized
  relational spool and set-oriented inserts.
- CAS staging verifies or copies each unique digest once and rejects symlinks,
  non-regular objects, unstable files, malformed locks, and unregistered
  filesystem content.
- Canonical facts use the current source-proof schema, durable dirty state,
  atomic revision/audit mutation, and stable publication fences.
- Synchronization holds one input snapshot, streams supporting facts, reuses
  asset assessment only under the complete parser/runtime key, and fuses
  assessment rows, summaries, ordered digests, and indexes before one retained
  full generation validation.
- Complete indexed facets support archive, quality, evidence, benchmark,
  reason, verification, and attestation filtering without expanding the public
  first-level API.

## Performance result

A reproducible three-trial benchmark compared base commit `80ce620` with
optimized commit `f3f7207` using 1,000 Cases, eight edits, and
`PYTHONHASHSEED=0`.

| Phase | Baseline median (s) | Optimized median (s) | Speedup |
| --- | ---: | ---: | ---: |
| build | 1.013 | 1.386 | 0.731x |
| mutation | 0.556 | 0.531 | 1.049x |
| synchronize | 12.818 | 2.318 | 5.530x |
| certify | 7.106 | 1.050 | 6.770x |
| ingest | 16.393 | 3.576 | 4.584x |
| phase sum | 37.886 | 8.860 | 4.276x |

The median lifecycle phase sum fell 76.61%, and median process peak RSS fell
2.56% (230,528 to 224,616 KiB). Build time regressed 36.76%. That residual is
explicitly retained in the result and should be investigated separately; it
does not negate the measured end-to-end gain.

All six trials produced the same scoped semantic digest, passed SQLite
integrity checking, and had zero foreign-key violations. The benchmark is
synthetic. It supports the numbers above but does not support a claimed
full-CSD speedup.

## Real-data reliability and activation

The rebuild processed 252,665 Cases across `oxidation`, `spin`, and
`csd-electronic-state`. It proved equality of scientific Case/assessment
semantics, control identities, stable generation projections, receipt
semantics, included/excluded Case sets, policy identity, and stable CAS
content.

- Semantic parity digest:
  `ffb944dbabbd3e5ca0488a967ae6864350c9408e95d9b07853c4757b25d954e0`
- Parity artifact SHA-256:
  `a3858d9922892b24494fecad9e3b5cc039af6ca33b584f935f04a962ecac98cb`
- Activated generation: `stage13-efficiency-current`
- Activation ID: `activation-5d037954aa104dfbb6e6d3028833e82a`
- Activation time: `2026-10-16T05:07:00.068761Z`
- Rollback generation: `stage13-current`

After activation, the 5,093,584,896-byte managed store passed SQLite
`integrity_check` and had zero foreign-key violations. All three Bundle facts
stores use schema version 4; all assessment stores use schema version 3 and
pass integrity checking. A complete CAS rehash verified all 2,317 objects and
7,645,485,715 bytes. The stable 2,296-object source/target projection matched
exactly.

## API, CLI, and test result

The final dataset suite passed:

```text
python -m pytest -q tests/test_ds
354 passed in 110.41s
```

Live API checks covered dataset discovery/load, metadata, IDs, labels,
structures, subsets, and molecule access. Live CLI checks covered list/info/get,
quality/evidence subset filtering, the JSON `subset -> get` pipeline,
multi-structure primary/index/role/ID selection, native XYZ output, XYZ to
SMILES conversion, and Bundle status. Protocol stdout remained a valid JSON
envelope; optional terminal-oriented structure text was sent to stderr.

The principal live smoke commands were:

```bash
python -c "from hotpot import ds; d=ds.load('spin'); x=d[0]; print(list(ds.datasets_list), len(d), x.id, x.quality_level, x.evidence_level, x.structures())"
hotpot ds subset spin -q 1 -e 1 --limit 2 | hotpot ds get spin -
hotpot ds subset spin -q 1 -e 1 --limit 1 | hotpot ds get spin - --xyz --structure-index 0
hotpot ds get spin 0 --smi --structure-index 1
hotpot ds bundle --root hotpot/dataset/data/bundles/stage13-efficiency-current/spin/spin.v001 --dataset spin status
hotpot ds detail audit
```

The last command returned `passed: true` with no issues. CLI protocol/help
tests independently passed 31 tests in 1.17 seconds.

## Necessary changes to the audit and implementation plan

Four findings required explicit changes to the written plan:

1. **Performance evidence was separated from parity evidence.** The original
   audit requested full oxidation, spin, and CSD timing. A historical/current
   CSD run would not be a controlled benchmark unless the historical runtime,
   environment, cache state, and restricted multi-gigabyte inputs were
   reproduced. The final plan therefore uses a controlled fixed synthetic
   benchmark for speed claims and the real rebuild only for semantic parity
   and operability. This change is mandatory to avoid presenting incomparable
   timings as a CSD speedup.
2. **Persistent CAS lock files became an explicit filesystem class.** The first
   real parity pass correctly exposed that the initial verifier treated valid
   lock files as unknown content. The final rule accepts only zero-byte regular
   `.<sha256>.lock` files in the correct shard and with a registered object.
   This change is mandatory for correct operation; ignoring arbitrary hidden
   files would weaken reliability, while rejecting every lock makes valid CAS
   stores unverifiable.
3. **The build regression remains visible.** Aggregate lifecycle performance
   improved substantially, but build did not. The plan/report now records the
   regression and leaves further build optimization as separately measured
   work. This is mandatory for an honest performance conclusion.
4. **Observability was scoped to the claims being tested.** The audit's initial
   exhaustive per-real-dataset matrix included OS byte counters, all SQL
   classes, parser/checker counts, and conflict metrics whether or not a change
   affected them. The final gate uses deterministic operation counters,
   controlled lifecycle wall/CPU/RSS measurements, and complete real-data
   parity. The larger matrix remains available for targeted profiling. This
   adjustment is mandatory to avoid instrumentation complexity and workload
   distortion while still requiring evidence for every performance and
   reliability claim made here.

No change relaxed the scientific conflict model, Gate semantics, certificate
binding, precommit verification, source/publication compare-and-swap, or
stable parser-input requirements.

## Retention and cleanup

Immediately disposable test/cache artifacts were removed during the work. The
prior active generation, historical migration Bundles, resumable rebuild state,
and pre-change tar.xz archive were not deleted. Their eventual owner-approved
removal is declared in `hotpot/dataset/retention-cleanup.txt`; the active
generation, current parity/activation evidence, current certificates, source
datasets, and active assets are excluded from that list.

The retained `rebuild_current_runtime.py` and its focused test are rollback
operators outside the installed package, not compatibility runtime. They are
eligible for removal with the retained rollback generation. Dataset-specific
recipe separation is governed by the separate lifecycle audit and was not
mixed into this efficiency release.

The cleanup also removed an unreferenced zero-byte `dataset.sqlite3` typo from
the old generation and retired the no-caller, old-layout parity-baseline
generator after preserving its JSON result. The valid plural
`datasets.sqlite3`, all rollback evidence, and the active data were untouched.

## Conclusion

The implementation meets the intended business and interface constraints: it
helps users assess and archive reliable data without imposing a new public
classification workflow, preserves the concise API/CLI model, contains no
old-version compatibility path, and makes backend efficiency serve rather than
distort the governance logic. The remaining measured issue is canonical build
performance, not data reliability or release correctness.
