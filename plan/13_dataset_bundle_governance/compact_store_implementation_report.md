# Compact formal-store implementation report

## Result

The current 252,665-Case corpus was rebuilt and activated in the single current
compact formal-store schema. The active formal release occupies 959,831,795 bytes
(0.894 GiB), compared with approximately 11.86 GiB for the prior active
database and control CAS. Scientific facts, original evidence and structures,
supplier/owner classification facts, Gate policy, final quality/evidence and
benchmark labels, concise decision reasons, receipts, and compact provenance
remain available through the public API.

The reduction comes from removing duplication and reproducible implementation
products, not from dropping scientific records. Per-Case legacy relational-row
mirrors, full Gate traces, archive-class duplicate indexes, Bundle facts,
assessment generations, and large generated identity maps are outside the
formal target.

## Reliability conclusion

The active release passed:

- exact Case counts for oxidation (2,175), spin (2,175), and
  csd-electronic-state (248,315);
- full retained-fact, identifier, structure, Evidence asset, and final scalar
  quality/evidence/archive/benchmark parity, plus output-to-target parity for
  current benchmark status and concise reasons;
- SQLite integrity and foreign-key checks;
- digest verification of every compressed fact and every CAS object;
- a hard formal-size limit of 1 GiB;
- the complete `tests/test_ds` suite (368 passed, 1 skipped);
- an end-to-end test that removes the Bundle and reads, queries, exports, and
  audits the release from the formal store alone.

The old generation is an immutable rollback artifact. Real Bundle pruning is
deliberately deferred until the rollback acceptance window ends, because
removing its generated facts and assessments would prevent another full
pre-activation parity run.

## Necessary departures from the initial plan

The implementation plan required correction in five places:

1. Parity now distinguishes complete comparison of retained facts from binding
   the unmodified original intake. Calling the former a full comparison of all
   original implementation metadata was not accurate.
2. Source inventory and owner/tool migration provenance are permanent. A
   category-wide `migration` deletion rule would have discarded original
   classification basis, so cleanup now authorizes only two exact generated
   map identities.
3. Cleanup resume repeats all authority and integrity checks. Persisting a
   Boolean or an old cleanup plan cannot prove current target state.
4. The one-time rebuild has a migration-local reader for the actual sealed
   schema-v2 source and schema-v1 assessment generation. Adding runtime
   compatibility would violate the project's one-version rule.
5. Legacy assessment authority is accepted only after recomputing its logical
   generation digest and artifact hashes. Merely comparing two copied digest
   strings would not prove integrity.

These changes were mandatory to preserve the business logic: Hotpot helps the
owner judge and archive evidence without forcing working-state internals into
the permanent database, while still proving that compaction did not alter the
scientific or governance result.

## Cleanup interface

From a sealed, successfully ingested Bundle root:

```bash
hotpot ds bundle rm --all
```

The fully asserted form is:

```bash
hotpot ds bundle --root BUNDLE_ROOT --dataset DATASET --db DB rm --all
```

The command preserves intake, original scientific assets, policies/checkers,
certificate, receipt, audits, source inventory, migration provenance, and
unknown user files. It removes only the closed, tested allowlist and writes a
digest-bound `bundle-cleanup.json` record.

