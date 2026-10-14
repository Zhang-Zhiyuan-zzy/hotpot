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

### Step 7 — Build the canonical Bundle fact authority

Status: complete

Planned:

- build one release from a strict JSONL or SQLite intake exactly once;
- reopen only by validation; and
- commit facts, audit events, revisions, digests, and the current-generation
  pointer through one SQLite authority.

Performed:

- added the strict one-release manifest and Case schemas, safe relative-path
  checks, streaming JSONL/SQLite intake, atomic `facts.sqlite3` publication,
  and validation-only reopen;
- added defensive Case iteration, transactional fact/policy mutation, audit
  and policy projections, source/policy revisions and digests, and
  compare-and-swap generation publication;
- made scientific asset bytes/existence part of the current source digest and
  added an audited refresh that advances the source revision after an
  out-of-band asset change; and
- made the built manifest immutable and aligned child identities with the
  frozen `structure_id`, `observation_id`, `claim_id`, `evidence_id`,
  `verification_id`, and `attestation_id` API.

Verification:

- `python -m pytest -q tests/test_ds/test_stage13_bundle_state.py tests/test_ds/test_stage13_case.py`
- result: `19 passed`;
- all shipped Bundle JSON schemas parse as valid JSON.

Unexpected findings:

1. The first fact-store draft used one generic child `id`, which contradicted
   the already frozen and implemented explicit Case method identities.
2. Detecting a changed external asset made the Bundle correctly stale, but the
   draft had no operation to record the new source digest/revision before CAS
   generation publication.

Necessary adjustment and final decision:

- use only the explicit per-fact identity keys and audit external asset-state
  changes through `refresh_source_state()` at synchronization start. This is
  required to make the public Case contract and the persisted revision model
  agree; it does not add a user-facing workflow.

### Step 8 — Establish the Hotpot attestation trust boundary

Status: complete

Planned:

- authenticate Hotpot benchmark attestations outside caller-controlled Case
  facts; and
- freeze the governance registry used by each assessment generation.

Performed:

- added an installed, pinned Ed25519 genesis public key and a canonical signed
  governance registry;
- implemented key activation, rotation, prospective revocation, and exact
  dataset/release/attestation context binding;
- made Hotpot signing require a private capability object while verification
  accepts only the pinned registry chain; and
- kept caller-provided `signature_verified` out of canonical input facts.

Verification:

- `python -m pytest -q tests/test_ds/test_stage13_governance.py`
- result: `8 passed, 1 skipped`; the skip covers the explicitly tested
  fail-closed path when an optional cryptographic backend is unavailable.

Unexpected finding:

- no matching private key is stored in the repository, by design. Production
  Hotpot attestations therefore require an owner-controlled signing service or
  secret that corresponds to the installed public root.

Adjustment: none to the trust model. The operational key provision will be
documented without weakening the public-key boundary.

### Step 9 — Publish immutable assessment generations

Status: complete

Planned:

- reassess the complete Bundle into an atomic, immutable generation; and
- bind summaries, three archive indexes, governance, and assessment rows by
  digest.

Performed:

- added full-snapshot generation into a temporary directory followed by atomic
  rename and Bundle-state compare-and-swap publication;
- streamed canonical facts into SQLite work/assessment stores, parsed declared
  assets, grouped comparable observations, and generated all three Case class
  indexes;
- wrote a deterministic manifest that inventories every generation artifact
  and rejects changed, omitted, or unexpected files; and
- exposed both the logical generation digest and the byte digest/path of
  `generation.json` required by certificate binding.

Verification:

- `python -m pytest -q tests/test_ds/test_stage13_generation.py`
- result: `9 passed`.

Unexpected finding:

- the first generation object exposed only its logical digest, whereas the
  certificate specification separately binds the physical manifest file.

Necessary adjustment and final decision:

- expose read-only `manifest_path` and `manifest_digest` generation
  properties. This is a private integration correction required to avoid
  conflating logical content identity with file-integrity binding.

### Step 10 — Close release and supplied-trust intake gaps

Status: complete

Planned:

- keep source facts free of operative derived trust; and
- preserve the stable three-digit public release identity.

Performed:

- allowed an optional, validated `release_serial` from 1 through 999 in the
  one-release manifest; and
- rejected caller-supplied `signature_verified` values at schema and loader
  boundaries.

Verification:

- `python -m pytest -q tests/test_ds/test_stage13_bundle_state.py`
- result: `15 passed`.

Unexpected finding: none beyond the two integration gaps already identified
during governance/generation review.

Adjustment: none to workflow; these constraints make the frozen identity and
trust rules executable.

### Step 11 — Execute persisted Gate policy at managed scale

Status: complete

Planned:

- make persisted Gate configuration, rather than hard-coded defaults, the
  complete assessment authority; and
- preserve deterministic conflict results without unbounded group memory or
  repeated full-table scans.

Performed:

- wired configured evidence/quality level maps, enabled/disabled built-ins,
  structure checks, Benchmark direct routes, vetoes, and numeric tolerances
  through assessment generations;
- made custom checks require the verified isolated-runner boundary and fail
  closed on unavailable isolation, exceptions, or malformed results;
- replaced conflict-group materialization with deterministic SQLite range
  streaming and added a Case-to-conflict link index; and
- streamed archive indexes during generation write and verification.

Verification:

- `python -m pytest -q tests/test_ds/test_stage13_assessment.py
  tests/test_ds/test_stage13_gate_integration.py
  tests/test_ds/test_stage13_generation.py tests/test_ds/test_stage13_gates.py`
- result: `36 passed`.

Unexpected finding:

- the initial generation implementation preserved deterministic output but
  materialized every row in an identity/property group and scanned the full
  conflict table for each Case. This was not safe for the measured managed
  dataset.

Necessary adjustment and final decision:

- use indexed, output-sensitive streaming. Exact pairwise conflict semantics
  necessarily remain proportional to the number of emitted candidate pairs,
  but avoidable unbounded memory and repeated scans are removed.

### Step 12 — Bind immutable manifest and governance state

Status: complete

Planned:

- make synchronization a proof over all persisted control state.

Performed:

- compared live `bundle.json` semantically with the canonical built manifest
  on every state access and mutation;
- persisted current and synchronized governance registry revision/digest; and
- added audited governance refresh plus governance-aware generation
  compare-and-swap publication.

Verification:

- BundleState, generation, and Bundle flow slice: `31 passed`.

Unexpected finding:

- a Bundle object opened before an out-of-band manifest edit could previously
  synchronize against the changed file and then become impossible to reopen;
  registry rotation also did not invalidate a draft assessment.

Necessary adjustment and final decision:

- manifest identity is immutable after build and the trusted registry binding
  is a first-class synchronization input. Both are required for reproducible
  certificates and cannot be treated as display metadata.

### Step 13 — Replace certification with the current closed envelope

Status: complete

Planned:

- serialize only a current synchronized generation; and
- permit lossless root-certificate restoration without rerunning assessment.

Performed:

- removed the old configurable assessment/certificate facade;
- added the root-only closed envelope, atomic publication, strict JSON/path
  checks, certified positive and negative scientific-asset states, facts,
  checker, policy, target, summary, and generation-manifest bindings; and
- made verification associate the certificate with exactly one Bundle root.

Verification:

- certificate focused suite: `9 passed`;
- certificate, Bundle, and target-store slice: `20 passed`.

Unexpected finding:

- the obsolete public `store` export imported removed Certificate/Configure
  types and prevented collection as soon as the no-compatibility certificate
  replacement landed.

Necessary adjustment and final decision:

- remove `store` from both first-level export lists immediately. This follows
  the already frozen closed API and prevents an obsolete facade from dictating
  the new certificate model.

### Step 14 — Enforce certified target ownership and preconditions

Status: complete

Planned:

- make the target own all bytes required by an immutable release; and
- recheck certified target identity and policy inside the write transaction.

Performed:

- added generic scientific-asset references for evidence, withdrawn history,
  and observed hash-mismatch bytes in addition to active structure views;
- added release control-asset references for exact checker sources and the
  frozen governance registry;
- copied and verified every materialized object into target-owned SHA-256
  storage; and
- bound and rechecked canonical target path, schema, store UUID, dataset-state
  precondition, and exact policy during ingestion.

Verification:

- `python -m pytest -q tests/test_ds/test_stage13_storage.py`
- result: `9 passed`.

Unexpected finding:

- the first target implementation owned active structure bytes only. Evidence
  and control artifacts could still disappear with the Bundle folder, and a
  replacement SQLite file at the same path was not rejected transactionally.

Necessary adjustment and final decision:

- target ownership covers all positively materialized scientific and policy
  control bytes. Asset reuse remains content-addressed while scientific
  identity remains Case/declaration based. This is required for sealed-release
  durability and does not expose another public storage API.

### Step 15 — Freeze bytes across hashing and chemical parsing

Status: complete

Planned:

- ensure an assessment generation describes one stable scientific-asset
  snapshot even when files are changed outside Hotpot.

Performed:

- streamed each present asset into a private temporary snapshot while hashing;
- made structure parsers read only that exact snapshot;
- compared source and snapshot byte digests plus device/inode/mode/size/mtime/
  ctime signatures before and after parsing; and
- guaranteed bounded-memory copying and cleanup on success or failure.

Verification:

- generation suite: `12 passed`;
- assessment/Gate/generation slice: `23 passed`.

Unexpected finding:

- checking only the aggregate source digest before and after synchronization
  permits a transient `A -> B -> A` change while a parser observes `B`.

Necessary adjustment and final decision:

- hashing and parsing must consume the same immutable temporary bytes and any
  source or private-snapshot mutation aborts synchronization. A Bundle lock
  alone cannot govern external filesystem writers, so this byte-level check is
  mandatory.

### Step 16 — Complete the Bundle lifecycle

Status: complete

Planned:

- connect one-time build, validation-only load, synchronization,
  certification, ingestion, restoration, and sealing behind the concise
  Bundle API.

Performed:

- implemented mandatory dataset/target resolution, new versus existing policy
  behavior, Gate replacement in new mode, and all Case views;
- added explicit full synchronization with source/governance refresh and CAS;
- made certification and ingestion distinct, locked state transitions;
- added target-owned scientific/control asset snapshots and transactional
  ingestion; and
- derived sealed state from the authoritative exact-bound receipt, rejected
  receipt/certificate/policy/generation mismatch, supported exact retry, and
  prohibited sealed mutation.

Verification:

- Bundle/certificate/storage/Case lifecycle slice: `27 passed`;
- end-to-end Bundle flow: `4 passed`.

Unexpected finding:

- a receipt row with the same release name was initially treated as sufficient
  proof of sealing, and certificate verification and asset copying were not in
  one Bundle lock interval.

Necessary adjustment and final decision:

- sealing requires an exact source, policy, generation, certificate, Case-set,
  asset, and original target-context binding. Ingestion verifies the
  certificate both before and after snapshot construction and lets target
  asset copying reject later byte changes. This closes the mixed-snapshot
  window without introducing an implicit synchronize or certify operation.

### Step 17 — Preserve entity and property-policy source facts

Status: complete

Planned:

- keep the current intake schema sufficient for scientific identity and
  property-context comparison.

Performed:

- admitted explicit entity facts with stable `entity_id` values; and
- admitted property definitions and property comparator policy in the one
  immutable release manifest and existing-dataset policy check.

Verification:

- BundleState suite: `19 passed`.

Unexpected finding:

- strict intake rejected entities promised by `Data.entities`, and Bundle
  policy attempted to read property comparators that its own manifest schema
  did not permit.

Necessary adjustment and final decision:

- make both concepts current canonical input. This removes an internal
  contradiction; it does not add a second identity model or a compatibility
  format.

### Step 18 — Finalize the concise persisted-query API

Status: complete

Planned:

- implement only the documented Dataset/Data/search surface over current
  TargetStore releases.

Performed:

- added ordinary-plus-benchmark default reads and explicit incomplete/mixed
  views without allowing a subset to widen its source population;
- fixed `<namespace>_identifier` lookup, nested subset/search fields, label
  coverage counts, and JSON-safe Data/Dataset summaries;
- made multi-structure access select one explicit primary before format lookup
  and reject ambiguity; and
- verified target-owned CAS path, digest, and size before structure access or
  Molecule/Crystal construction.

Verification:

- query suite: `5 passed`;
- query, storage, and Bundle integration slice: `18 passed`.

Unexpected finding:

- the requested `dataset.CSD_identifier[...]` spelling was being interpreted
  as a literal `CSD_identifier` namespace rather than the stored `CSD`
  namespace; label counts also summed numeric values instead of counting Cases.

Necessary adjustment and final decision:

- `_identifier` is the public accessor suffix, not part of the persisted
  namespace. Label coverage counts presence once per Case. Both corrections
  implement the already documented user semantics.

### Step 19 — Make every Case edit one source transaction

Status: complete

Planned:

- commit each user-visible Case mutation with one revision and one audit event.

Performed:

- moved read-latest, state-dependent validation, mutation, schema validation,
  digest/revision update, and audit insertion under the Bundle file lock and
  one SQLite `BEGIN IMMEDIATE` transaction; and
- made evidence verification test active evidence and append the verification
  in that same transaction.

Verification:

- Case/Bundle/BundleState/governance slice: `39 passed, 1 skipped`.

Unexpected finding:

- the first Case implementation read a defensive snapshot before acquiring the
  mutation lock, so two valid callers could overwrite one another or verify
  evidence that another caller had just withdrawn.

Necessary adjustment and final decision:

- transaction callbacks operate on the latest canonical Case payload. They
  remain private implementation machinery and do not enlarge the public Case
  API.

### Step 20 — Close identifier multiplicity and first-level facade contracts

Status: complete

Planned:

- enforce the frozen eleven-name public facade and stable external identifier
  lookup.

Performed:

- added exact first-level/signature-tree tests; and
- removed the target-store uniqueness rule that incorrectly required one Case
  per `(namespace, value)` while retaining per-Case identifier uniqueness.

Verification:

- public API, query, and storage slice: `18 passed`.

Unexpected finding:

- the managed source contains one real `CSD` identifier associated with two
  spin Cases. The query API already returns a Dataset for multi-matches, but the
  first target schema would have rejected that valid source relationship.

Necessary adjustment and final decision:

- external identifiers are lookup keys, not globally unique scientific
  identities. Multiple matches remain explicit in the returned Dataset and no
  Case merging is performed.

### Step 21 — Freeze the CLI communication protocol

Status: complete

Planned:

- make every machine-facing `hotpot ds` pipeline use one closed, versionless
  JSON contract while keeping screen-oriented rendering off stdout.

Performed:

- froze the complete 19-producer inventory and its allowed consumer edges;
- required the four fields `protocol`, `schema_sha256`, `key`, and `payload`
  on every envelope;
- bound both emission and ingestion to the shipped closed payload schema; and
- rejected undeclared keys, channels, consumers, fields, and schema bytes.

Verification:

- focused protocol, CLI, query, and structure-rendering slice: `41 passed`.

Unexpected finding:

- the previous protocol described producers in one JSON file but did not bind
  runtime messages to the exact payload-schema bytes, leaving documentation and
  implementation able to drift independently.

Necessary adjustment and final decision:

- the schema digest is now part of every message and is checked at both ends.
  This is a current protocol contract, not a compatibility-version mechanism.

### Step 22 — Replace the CLI with the current Bundle workflow

Status: complete

Planned:

- expose the concise query, export, Bundle, certify, and ingest lifecycle over
  only current storage and object models.

Performed:

- implemented positional dataset/item grammar, multi-structure selectors,
  format conversion, stable export layouts, quality/evidence/archive filters,
  fixed Bundle mutations, Gate editing, explicit synchronization, and
  certificate-required ingestion;
- kept machine output on stdout and human structure rendering on stderr; and
- reimplemented diagnostic `detail` producers over `TargetStore` without the
  deleted Python detail API.

Verification:

- CLI/protocol/help/export/structure and Bundle-focused handoff: `45 passed`;
- independent CLI/protocol/query/structure slice: `41 passed`.

Unexpected finding:

- the old export test built data through the removed submission/load/store API,
  so retaining it would have required restoring precisely the compatibility
  surface the plan prohibits.

Necessary adjustment and final decision:

- replace that test with current TargetStore and Bundle end-to-end coverage.
  No adapter or fallback route was added.

### Step 23 — Publish one current English documentation set

Status: complete

Planned:

- document the exact public API, Bundle/storage organization, quality workflow,
  CLI grammar, and communication protocol in English.

Performed:

- rewrote the dataset and `hotpot.ds` entry documentation around the eleven
  first-level API names and current lifecycle;
- documented incomplete-data visibility, multiple structures, migration and
  cutover, certificate-gated ingestion, and the fixed CLI mutation allowlist;
  and
- replaced the protocol skill reference with the schema-bound current contract.

Verification:

- documentation-related Stage 13 slice: `32 passed`;
- skill validation and `git diff --check`: passed;
- no Chinese formal dataset documentation or legacy public API names found.

Unexpected finding:

- partial drafts used legacy suffixed B/E strings, a non-existent assessment
  directory layout, and more CLI mutations than the intentionally small CLI
  allowlist.

Necessary adjustment and final decision:

- documentation follows executable current facts: `B0`–`B4`, `E0`–`E4`,
  `cases/facts.sqlite3`, generation-local indexes, and six fixed CLI Case
  mutations. The broader Python Case API remains documented separately.

### Step 24 — Ship every runtime schema as package data

Status: complete

Planned:

- make a wheel installation contain the same schemas used from a source tree.

Performed:

- added explicit package-data rules for Bundle JSON schemas, the SQLite intake
  schema, and the CLI protocol schema; and
- removed the package-data reference to the obsolete detail documentation.

Verification:

- package-data declarations and every named schema source were programmatically
  verified.

Unexpected finding:

- the source tests passed while the prior wheel manifest omitted all nested
  schema files. A wheel build in this environment is additionally blocked by a
  missing build-only `pybind11` dependency; `python -m build` is shadowed by the
  repository's `build` package.

Necessary adjustment and final decision:

- package the resources explicitly and record the environment-specific build
  limitation rather than claiming an unperformed artifact test. Final source
  and installed-CLI tests remain separate gates.

## Push checkpoints

| Checkpoint | Included steps | Commit range | Verification | Push result |
| --- | --- | --- | --- | --- |
| 1 | Steps 0–4: specification, baseline, mapping, Case model, assessment primitives | `b8322f7..e387af9` | baseline `104 passed`; new focused `10 passed` | pushed to `origin/refactor/unified-dataset-management` |
| 2 | Steps 5–7: Gate policy, target store, canonical Bundle facts | `f3c093f..dcff2b6` | focused Stage 13 suites passed | pushed to `origin/refactor/unified-dataset-management` |
| 3 | Steps 8–10: governance, assessment generations, release/trust intake closure | `e3bb3cb..e64dbc5` | governance `8 passed, 1 skipped`; generation `9 passed`; Bundle state `15 passed` | pushed to `origin/refactor/unified-dataset-management` |
| 4 | Steps 11–14: configured assessment, governance CAS, certificate envelope, target ownership | `ffbc244..87599f3` | configured Gate slice `36 passed`; Bundle-state flow `31 passed`; certificate/storage slice `20 passed` | pushed to `origin/refactor/unified-dataset-management` |
| 5 | Steps 15–17: stable asset snapshots, complete Bundle lifecycle, identity-policy intake | `db5b4de..4a0d59f` | generation `12 passed`; Bundle lifecycle `27 passed`; BundleState `19 passed` | pushed to `origin/refactor/unified-dataset-management` |
| 6 | Steps 18–20: query API, atomic Case edits, facade/identifier closure | `5242694..5bbfcc6` | query integration `18 passed`; mutation slice `39 passed, 1 skipped`; facade/storage/query `18 passed` | pushed to `origin/refactor/unified-dataset-management` |
| 7 | Steps 21–24: protocol, current CLI, formal documentation, package resources | `c83d661..d0052bb` plus log commit | independent CLI slice `41 passed`; handoff `45 passed`; docs `32 passed`; package declarations verified | pending push |
