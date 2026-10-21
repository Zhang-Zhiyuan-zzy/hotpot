# Stage 13 long-term and temporary code separation audit

Status: classification and cleanup plan only. Execution must begin after the
long-term workflow efficiency remediation in
[Stage 13 long-term workflow efficiency audit](efficiency_audit_report.md) is
implemented, verified, benchmarked, and reported.

## 1. Purpose

Stage 13 currently contains three different kinds of implementation:

1. permanent `hotpot.ds` runtime code used by every future dataset;
2. reproducible dataset-specific preparation and scientific-audit recipes that
   may remain useful but should not be installed as core runtime machinery; and
3. one-time schema-1 migration, bootstrap, parity, and cutover code and data.

Leaving all three under one installed `hotpot.ds.tools` namespace obscures
ownership, expands the apparent maintenance surface, and risks turning a
finished migration into an accidental compatibility backend. Deleting all
migration-related material indiscriminately would be equally incorrect because
some generic store activation behavior and historical audit artifacts remain
necessary.

This report defines the separation and the deletion gates. It does not perform
relocation, extraction, deletion, data cleanup, or documentation changes.

## 2. Required execution order

The ordering is mandatory:

```text
current forward work and cutover
  -> long-term five-phase efficiency remediation
  -> correctness and performance verification
  -> long-term/temporary code extraction
  -> one-time code and staging-data deletion
  -> final documentation and inventory audit
```

The cleanup must not run before performance remediation because the current
migration tool and its large-data fixtures expose the scale problems that the
long-term implementation still needs to reproduce and measure. Conversely,
the migration tool must not remain installed indefinitely merely because it
was useful as a benchmark source.

## 3. Classification rules

A component is **long-term runtime** only if it is required by the ordinary
five-phase workflow for new current-format data or by ordinary current-store
queries.

A component is a **long-term recipe** if it can reproducibly rebuild or audit a
scientific source dataset, has declared inputs and outputs, records provenance,
and is useful for a future refresh independent of the schema-1 migration.

A component is **one-time migration material** if it understands the removed
schema-1 layout, owner bootstrap used only for this cutover, legacy IDs,
legacy-to-current relation conversion, Stage 13 migration directories, or a
specific cutover attempt.

A component is a **retained historical artifact** if it is no longer executed
but is required to prove or reverse the completed cutover. Historical artifacts
must not be imported, queried, or treated as a compatibility backend.

Deletion requires positive proof that the item is neither a runtime dependency,
a reproducible recipe, a certificate/receipt-bound asset, nor an unexpired
rollback/audit artifact.

## 4. Long-term runtime: retain and optimize

The following responsibilities belong to the maintained current-version
runtime:

- Bundle/Case lifecycle and canonical fact authority;
- Evidence, Quality, and Benchmark Gates;
- full synchronization and immutable assessment generations;
- governance signatures, checker isolation, and runtime policy;
- certificate creation, restoration, and verification;
- target-owned CAS assets, transactional ingestion, and receipts;
- Dataset/Data query, search, subset, get, and export;
- the closed CLI stdin/stdout protocol;
- generation leases and safe generation garbage collection; and
- selection of the one active current store.

The principal maintained modules include:

- `hotpot/ds/public/bundle.py`;
- `hotpot/ds/public/case.py`;
- `hotpot/ds/public/_bundle_state.py`;
- `hotpot/ds/public/_generation.py`;
- `hotpot/ds/public/certificate.py`;
- `hotpot/ds/storage.py`;
- current query/search/public-facade modules;
- current Gates, governance, runtime-policy, and checker worker modules; and
- current CLI and communication-contract modules.

These modules must not import the schema-1 reader, migration bootstrap,
identity migration map builder, or legacy parity orchestration.

## 5. Long-term performance statistics

The reusable statistics introduced by efficiency remediation must be owned by
the long-term workflow rather than the migration tool.

### Required maintained measurements

For build, inspect/edit, synchronize, certify, and ingest, retain stable
internal measurements for:

- wall and CPU time;
- peak RSS;
- Case and relation counts;
- unique and referenced asset counts;
- bytes read, written, hashed, copied, and parsed;
- SQLite connection, transaction, and statement counts;
- source-digest and generation-validation counts;
- parser/checker invocation and failure counts;
- conflict group and pair-comparison counts;
- stage output sizes; and
- output logical/content digests.

### Maintenance requirements

- measurements must use a small internal schema with an explicit current
  identifier, not ad hoc print statements;
- diagnostics must not change API return values or closed CLI stdout envelopes;
- performance tests must assert upper bounds on pass/connection counts rather
  than fragile wall-clock values alone;
- small deterministic fixtures must run in the normal test suite;
- full oxidation/spin/CSD measurements belong in release audit artifacts, not
  unit tests; and
- statistics code must contain no schema-1, dataset-name, or cutover-specific
  logic.

## 6. Generic administrative store replacement: extract if retained

The storage contract permits future owner-directed current-schema replacement
or disaster recovery beside the active store. Therefore the abstract ability
to validate and atomically activate a complete current store generation may be
long-lived even though this schema-1 migration is one-time.

Before deleting the migration module, audit and, if necessary, extract only
the generic behavior needed for:

1. validating one current target store and adjacent CAS inventory;
2. proving exact receipt/release coverage supplied by an independent current
   administrative audit;
3. writing an activation record;
4. atomically replacing the `store/active` directory symlink; and
5. retaining/recovering the prior current store according to policy.

The extracted component must be private administrative code, not another
public ingestion API. It must accept only current target-store contracts and
must not understand schema-1 tables, legacy releases, migration bootstraps, or
legacy identity maps.

If no post-cutover current-schema replacement workflow is approved, retain only
the documented `store/active` layout and remove the executable activation code
with the migration tool. This decision must be made explicitly; accidental
retention is not acceptable.

## 7. Dataset-specific preparation code: convert to recipes or delete

The following installed modules are not generic `hotpot.ds` runtime behavior:

- `hotpot/ds/tools/analyze_structure_composition.py`;
- `hotpot/ds/tools/oxidation/validate.py`;
- `hotpot/ds/tools/spin/validate.py`;
- `hotpot/ds/tools/electronic_state/build_database.py`; and
- `hotpot/ds/tools/electronic_state/assess_quality.py`.

They are candidates for reproducible long-term scientific recipes because they
validate source structures, compute composition statistics, build the CSD
electronic-state source evidence database, or reproduce its quality audit.
They should not remain under the installed runtime tools namespace merely
because they were used during dataset preparation.

For each module, perform this decision test after efficiency remediation:

1. Is the code required to regenerate a maintained source dataset from declared
   upstream inputs?
2. Are its inputs, licenses, expected hashes, outputs, and scientific
   assumptions documented?
3. Does it emit deterministic outputs plus provenance and a summary?
4. Does at least one focused test cover its scientific invariants?
5. Is a named maintainer or owning dataset documented?

If all answers are yes, move it to a clearly non-runtime recipe area such as:

```text
hotpot/dataset/recipes/<dataset-or-audit>/
```

and provide one recipe manifest containing:

- recipe name and owner;
- declared inputs and digests;
- required software/runtime binding;
- command and parameters;
- generated files and digests;
- scientific assumptions and limitations;
- row/error/structure statistics; and
- regeneration verification command.

If the code cannot meet that standard or only created an obsolete intermediate
format, delete it with its obsolete tests and documentation. Do not preserve a
wrapper in `hotpot.ds.tools`.

The CSD `build_database.py` needs particular review: if its normalized SQLite
output remains the documented reproducible scientific source for future CSD
refreshes, it is a recipe. If that database existed only to feed the completed
schema-1 migration, both builder and schema belong to the deletion set.

## 8. One-time schema-1 migration code: delete after its gates pass

The following tracked items are explicitly one-time candidates:

- `hotpot/ds/tools/migrate_legacy_store.py`;
- `hotpot/dataset/manifests/stage13-migration-bootstrap.json`;
- `tests/test_ds/test_stage13_migration.py`; and
- `tests/test_ds/test_stage13_migration_bootstrap.py`.

The migration module contains several different concerns. Before deleting the
file, classify its symbols as follows.

### Delete with the one-time migration

- schema-1 table/column declarations and `LegacyNormalizedStore`;
- `DatasetBootstrap` and the Stage 13 owner bootstrap loader;
- schema-1 release association spools;
- legacy row inventory and identity migration builders;
- schema-1-to-Bundle normalization and export;
- legacy source/export parity orchestration;
- migration-specific rollback manifests and parity-report schemas;
- the end-to-end `migrate_legacy_store` command and parser; and
- any Stage 13 cutover pointer format not consumed by the current runtime.

### Consider extraction before deletion

- current-store generation validation;
- target CAS inventory verification;
- atomic `store/active` directory activation; and
- generic rollback/activation record handling.

Only behavior that is genuinely independent of schema-1 and useful for future
current-schema administration may be extracted. Everything else is removed,
not deprecated, aliased, or retained behind a compatibility flag.

### Test transition

Migration API tests must be removed with the deleted API. Scientifically useful
invariants must be reassigned rather than lost:

- generic target asset ownership stays in storage tests;
- receipt content and precommit CAS stay in ingestion-integrity tests;
- generic active-store atomicity stays only if generic activation is retained;
- duplicate external identifier preservation stays in current query/storage
  tests;
- scientific source counts/hashes stay in source-data regression sentinels or
  the final cutover report; and
- historical schema-1 export details do not remain in the normal test suite.

## 9. Working and stored data classification

### Ephemeral interrupted-attempt data

The current interrupted attempt includes paths such as:

- `hotpot/dataset/data/bundles/stage13-migration/`;
- `hotpot/dataset/data/store/generations/stage13-current/`; and
- temporary `.hotpot-target-ingest-*` spools below that candidate generation.

These are not active releases and must not be used as compatibility data. They
may be deleted only by the later cleanup implementation after explicit path,
symlink, active-pointer, receipt, and source-preservation checks. A clean rerun
must not reuse its partial CSD generation, certificate, spool, or target.

### Original source and rollback material

The original verified source database is not temporary merely because the
migration is one-time. It remains the authority for a clean rerun until
cutover. After successful activation, the replaced store or verified rollback
snapshot remains an immutable audit/rollback artifact for the owner-approved
retention period and is never queried as a fallback backend.

It may be deleted only after all of the following are recorded:

1. successful source/export and target parity for every release;
2. successful atomic activation and activated-store API/CLI smoke tests;
3. verified backup/restore procedure or expiration of the rollback retention
   policy;
4. owner approval of the exact deletion target; and
5. proof that no certificate, receipt, activation record, or audit manifest
   references the bytes being removed.

### Audit artifacts that remain historical evidence

Final owner bootstrap decisions, source inventory, identity migration map,
source/export parity report, target parity report, activation record, timing,
peak-resource results, and final digests should be retained in the designated
audit/archive location. They must not remain executable configuration or be
loaded by normal `hotpot.ds` code.

Certificate- or receipt-bound control assets cannot be deleted from the active
target CAS merely because their generating migration script is removed.

### Staging Bundle cleanup after activation

Staging Bundle roots may be deleted after target ownership and receipt
restoration are proven, provided the retained audit reports do not refer to
their paths as the only surviving evidence. The target CAS must contain every
positive control/scientific asset required by the sealed release. Negative
asset declarations remain database/certificate facts and require no missing
file placeholder.

## 10. Workflow and convention cleanup

### Retain as long-term conventions

- the five-phase current Bundle workflow;
- one Bundle per release;
- current-only runtime policy;
- target-owned SHA-256 asset objects;
- immutable release receipts;
- `store/active` selecting one complete current store generation, if generic
  administrative replacement remains supported;
- successor releases rather than mutation of sealed releases; and
- historical rollback stores never serving as query fallbacks.

### Remove after the one-time migration

- schema-1 as an accepted input to installed runtime code;
- the Stage 13 owner-bootstrap schema and manifest;
- `stage13-migration` as a workflow or directory convention;
- legacy entity/Structure ID conversion rules in runtime code;
- migration-only CLI/parser entry points;
- temporary cutover-pointer formats not consumed by runtime;
- migration API documentation presented as an available user workflow; and
- tests whose only purpose is preserving the removed migration implementation.

### Preserve only as historical prose

The Plan 13 implementation log/report may continue to describe the completed
migration and why its controls existed. Active API, CLI, storage, and user
documentation must describe only the current runtime. Historical prose must
not look like an executable compatibility instruction.

## 11. Deletion safety gate

The cleanup implementation must produce a reviewed inventory with one row per
candidate path or symbol containing:

- classification;
- current imports/references;
- runtime/package inclusion status;
- certificate/receipt/audit reference status;
- replacement or extraction destination;
- tests removed, retained, or reassigned;
- exact deletion prerequisite; and
- deletion commit.

Before deletion:

1. run `rg` for imports, CLI/docs references, package-data entries, and test
   dependencies;
2. inspect the active store target and reject any cleanup path containing or
   resolving through `store/active`;
3. verify exact explicit paths and reject symlinks for recursive cleanup;
4. verify the legacy source and required rollback/audit artifacts;
5. extract and test approved generic current-store functionality first;
6. remove code, tests, package data, and active documentation in the same small
   logical commits;
7. run the complete dataset test suite and installed-package smoke tests; and
8. append the exact retained/deleted inventory to the implementation log and
   report.

No compatibility wrapper, deprecated import, alternate schema loader, or
fallback data path may remain after cleanup.

## 12. Completion criteria

The separation work is complete only when:

1. ordinary `hotpot.ds` runtime imports no legacy migration code;
2. installed package contents contain no schema-1 migration command or
   bootstrap manifest;
3. every retained dataset-specific program is a documented, tested,
   provenance-producing recipe outside the core runtime namespace;
4. generic store activation, if retained, accepts only the current store
   contract and has focused atomicity/integrity tests;
5. interrupted-attempt working data is removed without touching the active
   store, original source, or retained audit/rollback artifacts;
6. final cutover evidence remains available but is not executable runtime
   configuration;
7. current API/CLI/storage documentation contains no obsolete workflow;
8. the complete dataset suite and package-resource checks pass; and
9. the implementation log/report records what was extracted, retained,
   archived, and deleted, with reasons and commit IDs.
