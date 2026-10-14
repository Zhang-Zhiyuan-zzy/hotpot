# Stage 13 implementation log

## Recording policy

This log is updated during implementation, not reconstructed afterward. Each
step records:

- the planned result;
- the work actually performed;
- verification evidence;
- unexpected findings;
- whether the implementation plan required an adjustment and why; and
- the commit and push checkpoint that published the work.

Only Stage 13 files are staged. Pre-existing changes under `plan/12_xtb_workflow`
and `plan/README.md` are outside this implementation and remain untouched.

## Baseline

| Item | Result |
| --- | --- |
| Date | 2026-10-14 |
| Working branch | `refactor/unified-dataset-management` |
| Starting commit | `8142cf1` |
| Remote | `origin/refactor/unified-dataset-management` |
| Baseline ds test command | `python -m pytest -q tests/test_ds` |
| Baseline result | `104 passed in 84.27s` |

The existing branch already isolates the unified dataset-management work and is
tracking the corresponding GitHub branch. Creating another branch would split
one logical refactor without improving rollback or review, so implementation
continues on this branch.

An initial plain `pytest` invocation resolved the user-level executable under a
different Python installation and failed before collection because that
interpreter lacked RDKit. The active project interpreter is selected reliably
with `python -m pytest`; this is an execution-environment correction, not a
product or test-plan relaxation.

## Step record

### Step 0 — Freeze the reviewed specification

Status: complete

Planned:

- freeze the Bundle/Gate/certificate design before implementation;
- keep the public facade small; and
- obtain independent reviews of state, scientific identity, CLI, scale, and
  security semantics.

Performed:

- completed `bundle_gate_certificate_implementation.md` and the Stage 13
  overview;
- resolved all blocking findings from three independent read-only reviews; and
- verified English-only formal documentation, balanced Markdown fences,
  non-duplicated headings, and valid local links.

Adjustment:

- the reviewed design added explicit build/load separation, deterministic
  conflict candidates, a certificate-bound governance registry, idempotent
  ingestion receipts, sealed-release behavior, and generation GC. These changes
  were mandatory because the earlier design could not guarantee reproducible
  trust decisions, crash-safe publication, or unambiguous release ownership.

### Step 1 — Establish the executable baseline

Status: complete

Planned:

- run the complete current `tests/test_ds` suite before changing behavior.

Performed:

- ran `python -m pytest -q tests/test_ds`;
- confirmed all 104 existing dataset tests pass.

Unexpected finding:

- the shell-level `pytest` executable used the wrong interpreter. The command
  was corrected to `python -m pytest`; no dependency or test was skipped.

Adjustment: none to product design.

### Step 2 — Map implementation boundaries

Status: complete

Planned:

- map the current API, storage, CLI, protocol, and tests to the Stage 13 phases;
- sequence changes so every commit has focused verification.

Execution batches:

1. current schemas, Gate contracts, canonical Bundle facts, and returned Case
   objects;
2. deterministic synchronization, evidence/quality/benchmark assessment, and
   immutable generations;
3. certificate round trip, ingestion receipt, sealed release, and database
   projections;
4. public query API integration and current-version schema rebuild;
5. CLI build/status/get/set/gate/synchronize/certify/ingest and the closed stdio
   contract;
6. migration tools, formal documentation, API/CLI integration tests, and final
   report.

No compatibility implementation will be added. Tests that describe superseded
behavior will be replaced in the same commit as the corresponding current API.

Performed:

- audited the public API/model, CLI/protocol, and target storage independently;
- mapped every superseded entry point to its current replacement;
- measured the managed store before designing migration: 252,665 records,
  594,631 entities, 201,501 observations, 69,596 claims, and 277,072 evidence
  items in an approximately 1.136 GB SQLite database;
- confirmed that 2,185 of 2,280 stored structure assets are shared between the
  oxidation and spin datasets; and
- identified the test slices and the 3–5-step push boundaries used below.

Unexpected findings:

1. The existing schema gives `coordinate_hash` a uniqueness constraint and the
   concise importer derives structure and entity identity from the same asset
   digest. That proves byte reuse, not scientific identity.
2. Existing ingestion leaves scientific files owned by the caller's source
   path. A sealed release could therefore lose its structures after a Bundle
   folder is moved or deleted.
3. Referencing an external CLI payload schema by path alone does not bind its
   content to `protocol_sha256`.
4. The reviewed CLI prose named Gate concepts but did not yet freeze all
   add/remove flags, Case-selection flags, zero-threshold evidence behavior, or
   the parent-option layout.
5. Existing import and synchronization paths fully materialize large tuples or
   lists. This is unacceptable for the measured 252,665-record store.

Necessary plan adjustments and final decisions:

- one Bundle now represents exactly one release; multi-release intake is split
  before build. This removes ambiguity in sealing, receipts, and stable IDs;
- ingestion streams local assets to a target-owned SHA-256 object area and
  stores only derived class indexes over one fact copy. This is required for
  durable releases without confusing byte equality with scientific identity;
- the CLI map records and verifies each payload schema SHA-256, so the protocol
  fingerprint covers its complete validation contract;
- Gate mutation flags, `bundle get cases` flags, `-e 0` semantics, and Bundle
  locator option placement are now closed in the plan; and
- synchronization, migration, and ingestion must stream/iterate rows rather
  than materializing the managed dataset.

These adjustments refine implementation boundaries without changing the
owner-confirmed scientific or governance model. They are mandatory correctness
constraints, not optional compatibility behavior.

### Step 3 — Add Bundle-owned Case behavior

Status: complete

Planned:

- make Bundle the permanent owner of every Case;
- expose defensive fact snapshots and synchronized assessment snapshots; and
- cover every documented fact mutation without permitting direct B/E or
  benchmark assignment.

Performed:

- added the returned `Case`, `CaseCollection`, and `CaseView` protocols;
- made ownership and Case identity immutable and ensured filtered/sliced views
  return the owner's original Case objects;
- added the complete core-info, structure, observation, claim, evidence,
  verification, and benchmark-attestation mutation surface; and
- made replacements/withdrawals append-only and delegated every commit to the
  owning Bundle so one fact transaction can update audit and revision state.

Verification:

- `python -m pytest -q tests/test_ds/test_stage13_case.py`
- result: `5 passed`.

Unexpected finding: none.

Adjustment: none; this follows Phase 1 and the frozen signature tree.

### Step 4 — Implement deterministic assessment primitives

Status: complete

Planned:

- keep evidence, identity, context comparison, conflict, quality, and benchmark
  rules deterministic and separate from generation publication; and
- ensure B/E values exist only in derived assessment output.

Performed:

- added automatic per-target E0–E4 assessment and a conservative Case
  aggregate with a true `null`/not-applicable state;
- required independently verified work and result-lineage roots for E4;
- added policy-driven identity keys, observation-context comparison, numeric
  tolerance/uncertainty comparison, and deterministic conflict groups;
- implemented B0–B4 derivation, the three archive classes, and separate
  scientific-conflict versus attestation-disagreement reasons; and
- enforced incomplete and veto precedence over both checklist and direct User
  approval.

Verification:

- `python -m pytest -q tests/test_ds/test_stage13_assessment.py`
- result: `5 passed`.

Unexpected finding: none.

Adjustment: none; immutable generation publication is deliberately implemented
as the next layer over these pure rules.

### Step 5 — Add immutable Gate policy configuration

Status: complete

Planned:

- provide the three compact Gate constructors from the frozen signature tree;
- persist deterministic configurations and exact custom-checker source; and
- fail closed when the required custom-code isolation boundary is unavailable.

Performed:

- added immutable `EvidenceGate`, `QualityGate`, and `BenchmarkGate` values with
  defensive JSON summaries and content digests;
- encoded cumulative B/E checks and the default benchmark checklist,
  Hotpot-or-User direct route, vetoes, and numeric tolerances;
- added strict top-level checker-source/signature/import inspection,
  content-addressed source materialization, and round-trip serialization; and
- prohibited in-process custom execution and required an explicitly verified
  OS-isolated runner.

Verification:

- `python -m pytest -q tests/test_ds/test_stage13_gates.py`
- result: `14 passed`.

Unexpected finding:

- an implementation-task note called the default item
  `primary_source_verified`, while the reviewed plan consistently names it
  `verified_primary_source`.

Adjustment:

- retained the reviewed `verified_primary_source` identifier. This is not a
  design change; it prevents an accidental second policy vocabulary.

### Step 6 — Add the current target store and asset ownership boundary

Status: complete

Planned:

- replace release ingestion with a current-schema transactional target;
- keep one canonical Case fact copy and separate derived class indexes; and
- make the target store own all positively materialized assets.

Performed:

- added store UUID/schema identity, immutable dataset policies, stable release
  serials and public Case indexes, assessments, identifiers, and separate
  ordinary/benchmark/incomplete indexes;
- added target-owned SHA-256 asset objects with streamed verification, atomic
  publication, byte deduplication, and separate Structure references;
- added bounded-memory Case spooling before the target transaction;
- added authoritative release receipts, exact retry idempotence, digest-
  different conflict detection, and in-transaction policy/release rechecks;
  and
- excluded incomplete Cases by default while requiring an explicit reason to
  include them.

Verification:

- `python -m pytest -q tests/test_ds/test_stage13_storage.py`
- result: `7 passed`;
- full dataset suite at this boundary: `148 passed`.

Unexpected finding: none beyond the asset-ownership gap recorded in Step 2.

Adjustment: none; this implements the mandatory Step 2 storage decision.

## Push checkpoints

| Checkpoint | Included steps | Commit range | Verification | Push result |
| --- | --- | --- | --- | --- |
| 1 | Steps 0–4: specification, baseline, mapping, Case model, assessment primitives | `b8322f7..e387af9` | baseline `104 passed`; new focused `10 passed` | pushed to `origin/refactor/unified-dataset-management` |
