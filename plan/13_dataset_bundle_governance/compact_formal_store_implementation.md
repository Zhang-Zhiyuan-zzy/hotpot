# Compact formal dataset store implementation

## Objective

Build one current-only formal-store layout that keeps the scientific facts and
the final governance decisions needed by the public `hotpot.ds` read API while
discarding reproducible Bundle build products after ingestion.  The active
database plus its owned assets must be approximately 1 GiB for the current
252,665-Case corpus.

This is a new store generation, not an in-place migration.  No reader or writer
for the superseded layout will remain in the current code.

## Permanent information boundary

The formal store retains:

- original observations, claims, evidence, verifications, attestations,
  entities, `core_info`, labels, identifiers, and source classifications;
- original structure and evidence bytes referenced by a Case;
- final `quality_level`, `evidence_level`, `archive_class`, benchmark decision,
  benchmark status, and concise reason codes;
- the dataset Gate policy, including persisted custom-checker source;
- the certificate, ingestion receipt, logical content digests, and release
  metadata.

The formal store does not retain:

- migration relational-row projections or migration implementation metadata;
- Bundle `facts.sqlite3` copies;
- assessment-generation databases, class ID lists, summaries, or Gate traces;
- per-Case asset/conflict working assessments and other values reproducible
  from retained facts and the persisted policy;
- generated migration relationship reports.

## Storage model

1. Strip only the closed, implementation-owned metadata keys
   `migration_rows`, `migration_schema`, `shared_identifier_relations`, and
   `quarantined_entity_relations` from the target projection.  Dataset-owned
   metadata is otherwise preserved.
2. Store canonical Case facts as deterministic zlib-compressed JSON BLOBs.
   Bind and verify the SHA-256 digest of the uncompressed canonical JSON.
3. Store the final assessment fields relationally and retain only canonical
   reason-code JSON instead of the complete assessment trace.
4. Keep identifiers and scientific asset references normalized and indexed.
5. Store only irreplaceable scientific assets and the compact certificate in
   target CAS.  Bundle facts, generations, and migration reports are never
   target assets.
6. Bind ingestion to the compact logical Case projection.  Physical SQLite
   page layout and compressed bytes are not scientific identities.

## Bundle cleanup

`hotpot ds bundle --root BUNDLE_ROOT --dataset DATASET --db DB rm --all`
is a verified pruning command.  It runs only after a valid authoritative
ingestion receipt exists.  It acquires the Bundle lock, revalidates the sealed
release, removes an explicit allowlist of reproducible build products, and
writes `bundle-cleanup.json` with removed paths, sizes, digests, and receipt
binding.  It never recursively deletes the Bundle root or unknown user files.

The command retains the intake, original structures/evidence, `bundle.json`,
Gate policy/checkers, certificate, ingestion receipt, audit history, and the
cleanup manifest.  A pruned Bundle is an archived intake record; editing,
synchronization, and recertification require rebuilding its derived products.

## Verification gates

- focused storage, API, certificate, CLI, and cleanup tests pass;
- the complete dataset test suite passes;
- a new full store generation passes SQLite integrity, foreign-key, CAS,
  receipt, certificate, label, identifier, structure, and assessment parity;
- no retained original scientific fact or original evidence object is lost;
- active formal database plus owned CAS is at or near 1 GiB;
- activation uses the existing parity-gated atomic generation switch;
- the source branch is merged only after all gates pass.

## Measured feasibility baseline

The current 252,665 records were evaluated read-only before implementation:

| Projection | Logical bytes | zlib level 6 bytes |
| --- | ---: | ---: |
| Complete stored facts | 3,207,002,168 | 852,478,809 |
| Facts without migration implementation metadata | 1,171,057,125 | 392,432,311 |
| Complete assessments | 1,190,582,078 | not retained |
| Final assessment fields and reason codes | 61,828,101 | 44,424,446 |

The measurement includes every current Case.  It demonstrates that the target
is achievable before considering removal of duplicated release-control assets
or obsolete archive-class tables.
