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
- required the four fields `key`, `channel`, `protocol_sha256`, and `payload`
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

### Step 25 — Verify sealed releases against frozen governance

Status: complete

Planned:

- keep a sealed release immutable and verifiable after the trusted governance
  registry advances, while preserving exact-receipt retry.

Performed:

- separated draft synchronization against live governance from sealed
  verification against the receipt-bound generation;
- removed the receipt-to-synchronization recursion;
- reported newer governance as context without rewriting historical results;
  and
- made a valid exact receipt eligible for idempotent `ingest()` retry.

Verification:

- independent Bundle/BundleState/certificate/storage/Case slice: `51 passed`.

Unexpected finding:

- receipt verification called `_require_generation()`, which consulted live
  governance and then called receipt verification again. A registry advance
  could therefore make an intact sealed release appear corrupt.

Necessary adjustment and final decision:

- sealed integrity uses frozen source, policy, governance, generation,
  certificate, and receipt bindings directly. Live governance is reported by
  `governance_update_available` and affects only a successor release.

### Step 26 — Make documentation assertions current-only

Status: complete

Planned:

- test the documented eleven-name facade and every exposed CLI family.

Performed:

- removed assertions for the deleted `store` and Python `detail` APIs;
- aligned heading/order checks with the new concise reference; and
- documented current-store `detail` diagnostics as CLI-only TargetStore views.

Verification:

- documentation and CLI-help slice: `30 passed`.

Unexpected finding:

- the documentation test itself still required the obsolete public API and a
  `python -m` fallback even after the formal documentation had been replaced.

Necessary adjustment and final decision:

- tests now enforce the current installed `hotpot ds` interface and distinguish
  its diagnostic `detail` command from the removed Python package.

### Step 27 — Make property policy executable

Status: complete

Planned:

- persist the dataset's scientific property/context comparison contract and
  use it for observation validation and conflict decisions.

Performed:

- added closed property definitions for type, unit, schema, vocabulary,
  normalization, and comparison context;
- added exact, categorical, and numeric comparators with tolerances,
  uncertainty handling, and auditable linear unit conversions;
- rejected invalid observations during intake and Case mutation;
- bound comparator context and digest into conflict results; and
- required an established dataset's later releases to use the exact persisted
  definitions and comparators.

Verification:

- independent lifecycle/property/query/public-API slice: `72 passed`.

Unexpected finding:

- `property_definitions` had previously been accepted but ignored, and
  `property_comparators` were persisted but conflict comparison used only a
  BenchmarkGate tolerance. The apparent scientific policy was inert.

Necessary adjustment and final decision:

- property policy is now a real dataset policy. Different or missing required
  context becomes noncomparable or indeterminate, never a fabricated
  scientific conflict. Benchmark tolerances are only a documented fallback
  when no property comparator exists.

### Step 28 — Separate source audit facts from derived grades

Status: complete

Planned:

- retain the CSD source audit and owner decisions without allowing them to
  assign operative B/E or benchmark results.

Performed:

- replaced suffixed legacy grade assignments with record-kind and
  support-status recommendations;
- marked owner adjudications as preserved claims, evidence, or descriptors;
  and
- added Bundle and target-asset directories to the canonical local layout.

Verification:

- source-tool/layout/documentation slice: `15 passed`;
- full 248,315-entry source audit completed without legacy suffixed grades.

Unexpected finding:

- an otherwise current source-quality tool still emitted release-level legacy
  grades, which would compete with Case-level Gate assessment.

Necessary adjustment and final decision:

- source tools report facts and recommended record kinds only. B/E and
  benchmark state are derived for one exact Bundle generation.

### Step 29 — Replace intake templates with the current Bundle contract

Status: complete

Performed:

- removed the obsolete certificate-config and flat dataset templates; and
- added one current Bundle manifest and one current Case template.

Verification:

- a temporary Bundle was constructed from the templates and loaded by the
  current implementation.

Unexpected finding and necessary adjustment:

- the old templates described a deleted ingestion model. They could not remain
  as examples because that would preserve a second, incompatible user path.

### Step 30 — Remove the obsolete dataset implementation island

Status: complete

Performed:

- removed the superseded database, model, quality, adapter, detail, and old
  public-ingestion modules and their compatibility tests; and
- retained the raw scientific-source regression sentinel in a current test.

Verification:

- current dataset suite excluding the not-yet-committed migration tests:
  `160 passed, 1 skipped`.

Unexpected finding and necessary adjustment:

- several old tests validated the removed API rather than scientific facts.
  Only the source-fact sentinel was retained; no compatibility wrapper was
  introduced.

### Step 31 — Implement an auditable legacy-store migration

Status: complete

Performed:

- added a strict read-only schema-1 reader, explicit owner bootstrap, stable
  Case/identifier mapping, rollback snapshot, Bundle export, certification,
  ingestion, parity reports, and atomic generation activation;
- preserved the intentional duplicate `spin/CSD/DOQRAC` identifier on both
  Cases instead of merging them; and
- distinguished an explicitly missing source asset from a copied asset that
  subsequently disappeared. Declared hash mismatch and copied-asset damage now
  fail source/export parity.

Verification:

- migration plus Bundle/storage/certificate slice: `34 passed`;
- a 2,175-Case oxidation smoke export and parity pass completed successfully.

Unexpected finding and necessary adjustment:

- independent review found that global row inventories and target counts were
  insufficient proof for a 252,665-Case cutover. The real migration is paused
  until per-Case operative-fact parity, exact included/excluded ID-set binding,
  release-owned migration controls, frozen-source use, identity-relation
  coverage, and pre-activation rollback/target revalidation are executable;
- all listed controls were then implemented: migration reads the verified
  rollback snapshot, receipt-bound release controls preserve the Bundle facts,
  certificate and generation even when all Cases are excluded, and activation
  rehashes both rollback and target before replacing the directory symlink.
  These changes are mandatory because a count-preserving corruption must never
  pass migration merely for operational convenience.

### Step 32 — Route the runtime through one atomic store generation

Status: complete

Performed:

- changed the default store to
  `hotpot/dataset/data/store/active/datasets.sqlite3`;
- defined the immutable `store/generations` root and the `store/active`
  directory-symlink contract; and
- aligned the tracked layout manifest, storage documentation, and path tests.

Verification:

- path, documentation, and query slice: `21 passed`.

Unexpected finding and necessary adjustment:

- the prior audited cutover JSON was provenance only and was never consumed by
  runtime queries. Runtime selection therefore had to use the same atomic
  directory pointer that switches the database and adjacent CAS assets.

### Step 33 — Make E3/E4 evidence facts reachable through the concise API

Status: complete

Performed:

- added one optional, closed `details` mapping to evidence add/replace/verify
  operations and the same CLI mutation documents;
- persisted scientific-work identity, DOI/URL, lineage/derivation, procedure,
  and verification time; and
- required attributable Human verification while retaining automatic current
  procedure/time defaults.

Verification:

- evidence/API/CLI focused slice: `29 passed`;
- current dataset suite excluding migration: `166 passed, 1 skipped`.

Unexpected finding and necessary adjustment:

- the planned simple signatures could create evidence but could not provide
  the facts the Gate itself requires for E3/E4. A single closed mapping is the
  minimum correction: it keeps the first-level API small while preventing an
  unreachable assessment level. The signature tree and evidence section were
  amended as an audit-required plan correction.

### Step 34 — Replace the legacy record card with one current information fence

Status: complete

Performed:

- rewrote the tracked record standard around the current Bundle manifest,
  Case source facts, property policy, three Gates, Case-level B/E assessment,
  negative asset declarations, certificate, and sealed target release; and
- added executable documentation checks against the authoritative Bundle JSON
  schemas and prohibited legacy grade/API vocabulary.

Verification:

- JSON validation plus documentation/path tests: `16 passed`.

Unexpected finding and necessary adjustment:

- the previous card described release/subset scalar grades and the removed
  `store.ingest` workflow. Keeping it would have made the formal information
  fence contradict the only current implementation, so it was replaced rather
  than retained as a compatibility document.

### Step 35 — Express an any-valid-Structure dataset requirement

Status: complete

Performed:

- defined reserved `required_structure_roles=["*"]` as at least one active
  Structure with a present, hash-valid, parse-valid representation;
- retained the existing all-named-roles semantics; and
- documented and tested missing, invalid, and valid wildcard cases.

Verification:

- assessment, generation, Gate, documentation, and Bundle slice: `39 passed`.

Unexpected finding and necessary adjustment:

- oxidation and spin require a valid structure, but their scientific roles are
  intentionally heterogeneous (`canonical_structure`, temperature variants,
  and multiplicity variants). Requiring one literal role would either reject
  valid Cases or erase scientific role meaning. The reserved wildcard adds the
  required invariant without changing the public API or rewriting roles.

### Step 36 — Freeze the owner-approved migration bootstrap

Status: complete

Performed:

- added explicit current Gate, identity, property/context comparator, core
  field, structure, incomplete-archive, owner, and decision policy for CSD
  electronic state, oxidation, and spin; and
- chose distinct per-Case identity keys until independent sample equivalence is
  available, preserving duplicate external identifiers without merging.

Verification:

- strict bootstrap deserialization and policy assertions: `2 passed`.

Unexpected finding and necessary adjustment:

- source property metadata was descriptive but not executable comparator
  policy. The bootstrap closes each observation vocabulary and makes numeric
  context/tolerance decisions explicit before first current-schema ingestion;
  it does not infer grades from old gold/silver membership.

### Step 37 — Harden migration parity and atomic activation

Status: complete

Performed:

- froze migration input on a verified rollback database and URI-preserving
  asset snapshot;
- compared every source/export Case's stable identity and complete operative
  projection, normalized identifier namespaces and relation endpoints, and
  quarantined unresolved cross-Case relations with full mapping coverage;
- bound migration inventories, Bundle manifest/facts, certificate, generation,
  governance, and checker files as release controls in the ingestion receipt;
- recomputed included/excluded Case sets, stored fact digests, certificate and
  policy/generation bindings, and every target CAS object; and
- reverified rollback, target database, target assets, and exact receipt set
  immediately before atomically replacing an existing active symlink.

Verification:

- migration, storage, Bundle, certificate, documentation, and owner-bootstrap
  slice: `51 passed`.

Unexpected finding and necessary adjustment:

- the first migration implementation could pass global row/count parity after
  a same-count operative-fact substitution, and migration evidence attached to
  one excluded Case could disappear from the target. The revised design uses
  disk-backed per-Case parity and deterministic release controls. This is an
  audit-required correction to the plan, not optional defensive code, because
  the former proof did not establish the scientific-content invariant it
  claimed.

### Step 38 — Make Ed25519 governance a mandatory verified dependency

Status: complete; committed and pushed

Planned:

- keep Hotpot authority behind a private signing boundary;
- verify every effective Hotpot attestation against the certificate-bound
  governance registry; and
- fail closed when the required signature implementation is unavailable.

Performed:

- declared `cryptography>=44,<47` as a direct project dependency;
- reduced the governance implementation to one current Ed25519 backend instead
  of an optional PyNaCl/cryptography compatibility path;
- removed the test skip that made rotation, revocation, private signing, and
  signed attestation verification conditional on the host environment; and
- added a signature-bit tampering assertion in addition to the existing
  cross-dataset/release context-replay checks.

Verification:

- with `cryptography 46.0.7` installed in the isolated
  `/tmp/hotpot-crypto-deps` dependency path, the focused governance suite
  completed with `9 passed`;
- the implementation commit is `541b69d`.

Unexpected finding and necessary adjustment:

- final gap review found that a skipped signature test and optional crypto
  backend made a core trust boundary environment-dependent. That contradicted
  the plan's fail-closed Hotpot authority model. A required, pinned Ed25519
  implementation and an always-executed real-signature test are therefore
  correctness requirements, not deployment conveniences.

### Step 39 — Isolate custom checkers and start runtime binding

Status: checker containment complete and pushed; the complete runtime/parser
binding is recorded in Step 45

Planned:

- execute user-defined checker code only across a real operating-system
  isolation boundary;
- use the fixed JSON snapshot/result protocol and fail closed on unavailable
  isolation, timeout, malformed output, or resource exhaustion; and
- bind certificates, assessment generations, and ingestion receipts to the
  exact built-in assessment/parser/checker implementation used to derive them.

Performed:

- added a Bubblewrap worker with separate user, PID, network, IPC, and UTS
  namespaces, a fixed minimal environment, read-only interpreter/library
  mounts, unprivileged UID/GID, resource limits, a wall timeout, and JSON-only
  input/output;
- sent exact checker source and its digest inside bounded JSON stdin rather
  than mounting the checker or exposing the Bundle/store;
- verified that Bundle/store paths are not covered by runtime mounts and probed
  filesystem, process-device, and network isolation before executing custom
  code;
- rejected generators, coroutines, interpreter-frame traversal, dunder escape
  paths, and non-allowlisted imports in both admission and worker validation;
- bound the fixed worker source, request/result fields, environment, limits,
  imports, and isolation contract into the checker protocol digest; and
- made a single timeout, exception, limit violation, or malformed response fail
  closed as auditable `checker_error`, while inability to establish the OS
  boundary blocks synchronization before publication.

Verification:

- focused isolation, protocol, malformed-output, timeout, and real
  generator-frame escape regressions passed;
- the implementation commit is `3cb191b` and is pushed.

Unexpected finding and necessary adjustment:

- source inspection and an import allow-list cannot contain the risks of
  executing Python code, while source hashes alone cannot reproduce results if
  built-in parser or assessment behavior changes. The audited design already
  required both controls, but the runtime had not implemented them. OS
  isolation and a deterministic implementation/protocol binding are mandatory
  before the real migration can produce trustworthy current certificates. The
  isolation boundary was completed here; the later audit found that the first
  aggregate implementation digest was still too weak, so Step 45 replaces it
  with one structurally validated, self-digesting runtime/parser policy.

### Step 40 — Protect assessment generations with leases and safe collection

Status: complete; committed and pushed

Planned:

- prevent a reader, certificate operation, or ingestion transaction from
  observing a generation that is concurrently removed;
- retain all current, certificate-referenced, receipt-referenced, or actively
  leased generations; and
- keep generation lifecycle machinery private so the concise public API does
  not expand.

Performed:

- added shared operating-system leases for complete generation reads,
  certificate work, and ingestion;
- made garbage collection acquire a non-blocking exclusive lease and retain
  current-pointer, certificate, receipt, active-reader, and conservatively
  unknown references; and
- relied on process-owned locks for crash release without unsafe timeouts,
  while leaving lock files as non-certified runtime state.

Verification:

- the implementing agent reported a focused generation/certificate/Bundle
  slice of `30 passed` using the isolated cryptography dependency path;
- Python compilation and diff checks also completed cleanly;
- the final combined dataset-suite result is recorded after the runtime-binding
  and scientific-integrity changes settle, rather than treating this focused
  slice as the release gate;
- the implementation is part of commit `9839b61`, which is pushed.

Unexpected finding and necessary adjustment:

- immutable directories and an atomic current pointer do not protect an old
  generation between pointer lookup and completion of a read. Deleting such a
  generation is a race even when publication itself is atomic. Reference-safe
  leases and conservative collection are therefore required for the lifecycle
  invariant; exposing them as user-facing API would add complexity without
  improving user control, so they remain private implementation details.

### Step 41 — Make benchmark principal state unambiguous

Status: complete; committed and pushed

Planned:

- retain the low-friction User/Other attestation API without permitting several
  active decisions from the same principal to create accidental disagreement.

Performed:

- required an exact Boolean `approved` value and a non-empty actor;
- made a new decision supersede every active attestation with the same
  `(authority, actor)` rather than only one prior row; and
- stored the complete supersession list so relationship validation and audit
  reconstruction see the same principal history.

Verification:

- replacement, active-decision, invalid-type, and empty-actor regressions
  passed;
- the implementation commit is `73eb2fb` and is pushed.

Unexpected finding and necessary adjustment:

- a principal could accumulate multiple active decisions and truthy non-Boolean
  input could cross the public boundary. Because direct benchmark entry depends
  on exact approval/rejection state, normalizing these after assessment would
  hide a governance ambiguity. Atomic supersession at fact mutation is the
  necessary correction.

### Step 42 — Bind local Evidence source bytes

Status: complete; committed and pushed

Planned:

- ensure every `file:` evidence source participates in the same source,
  certificate, and target-asset integrity chain as other local scientific
  assets.

Performed:

- materialized the safe Bundle-relative `path` from `file:<path>` into the
  Evidence fact; and
- thereby made source-state hashing, certificate asset inventory, target CAS
  copying, and sealed receipt verification discover the referenced bytes
  without a second user parameter.

Verification:

- focused Evidence mutation and local-file binding tests passed;
- the implementation commit is `2bdfb99` and is pushed.

Unexpected finding and necessary adjustment:

- retaining only the human-readable `source="file:..."` string left the common
  asset walker unable to bind those bytes. Adding a parallel storage API would
  violate the concise interface; deriving the canonical `path` inside the
  existing operation closes the integrity gap without exposing another route.

### Step 43 — Close evidence, identity, and relationship semantics

Status: complete; committed and pushed

Planned:

- make the implemented Evidence/Quality/Benchmark ordering match the reviewed
  scientific-integrity rules; and
- prevent broken Case relations or indeterminate identity from being hidden by
  a direct-entry attestation.

Performed:

- counted only explicit `supports` relations toward E levels and resolved an
  Evidence target by exact ID or one unambiguous property/label alias;
- normalized DOI/work identities, interpreted `derived_from` as result-lineage
  IDs, checked ancestry resolution/cycles, and required distinct work and root
  lineages for E4;
- made an empty core-label policy aggregate all active targets and made a
  missing declared core target E0;
- validated active Evidence, verification, reciprocal supersession, lineage,
  entity, participant, and subject links before benchmark assessment;
- classified broken operative relationships as incomplete, while keeping
  `identity_indeterminate` ordinary but benchmark-vetoed;
- required the current explicit `relation` field in migrated Evidence instead
  of preserving an implicit legacy default;
- closed `bundle.json.identity_schema` to one components array, normalized its
  scalar `core_info`/`identifiers` and role-aware semantic Structure paths,
  materialized typed normalizer/exact-comparator protocol digests, and persisted
  the normalized policy;
- implemented exact three-valued `sample_match`, Case-reflexive local conflict
  grouping when cross-Case identity is unavailable, and identity/property-policy
  digests on conflict assessments; and
- closed the tracked manifest JSON schema and current migration output to that
  same identity/evidence contract.

Verification:

- focused assessment and Gate-integration regressions cover contradictory
  Evidence, alias ambiguity, false E4, missing/cyclic lineage, empty/missing
  core targets, identity indeterminacy, broken relations, and list-valued
  attestation supersession;
- the main scientific-integrity implementation commit is `4079fc9`; the
  current migration/schema normalization follow-up is `a70ed5e`; both are
  pushed.

Unexpected finding and necessary adjustment:

- earlier code could treat contradictory/context evidence as support, could
  miss property-name targets, and could infer E4 from differently written
  identifiers or unrelated Evidence IDs. It also allowed relationship defects
  to survive direct entry. These were scientific-semantic defects, so the plan
  now states the implemented resolution graph explicitly. Hotpot validates the
  submitted graph; true independence of supplier-declared scientific lineages
  still requires traceable source facts and owner review.

### Step 44 — Seal ingestion against content and target races

Status: complete; committed and pushed

Planned:

- prove that the certified Bundle snapshot is still the snapshot committed to
  the target store; and
- preserve established-dataset policy authority when target availability
  changes.

Performed:

- derived `Bundle.mode` from persisted build-time target context rather than
  the current target's presence;
- materialized persisted custom-checker artifacts automatically when building
  a successor Bundle for an established dataset;
- added a canonical `case_content_digest` over ordered Case identity, facts,
  assessment, and identifiers, and persisted it in the authoritative receipt;
- revalidated target context, policy/runtime, certificate, generation,
  Case-content, selection, and asset bindings after acquiring the target write
  transaction and immediately before publication; and
- made a target store without the new receipt column fail as non-current rather
  than accepting a compatibility layout.

Verification:

- the dedicated ingest-integrity suite plus the related Bundle/storage/
  certificate slice completed with `40 passed` before the final runtime-policy
  integration;
- the implementation is part of commit `9839b61`, which is pushed.

Unexpected finding and necessary adjustment:

- Case-set and asset digests alone do not detect a same-ID fact/assessment
  substitution, and a pre-transaction check leaves a race while a large ingest
  is spooled. The receipt therefore needs an operative content digest and the
  Bundle needs an in-transaction compare-and-swap callback. Likewise, live
  target disappearance must never convert an established-dataset Bundle into a
  policy-authoring Bundle.

### Step 45 — Replace aggregate implementation hashes with one runtime policy

Status: complete; committed and pushed

Planned:

- bind every assessment result to the exact current scientific implementation
  while preserving verification of sealed historical results without loading
  old code.

Performed:

- replaced partial `runtime_bindings`/implementation-digest concepts with one
  closed, self-digesting `runtime_policy` document;
- bound full-file and selected semantic-symbol source digests, the assessment
  store and checker protocols, parser registry/selection/protocol/source, Python
  ABI, SQLite, and normalized relevant distribution identities;
- separated structural validation of a frozen policy from equality with the
  ambient current policy;
- stored the exact document in dataset policy, generation, certificate, and
  ingestion receipt, and included it in the generation logical digest;
- recorded parser binding/protocol/source/distribution fields in every parsed
  AssetAssessment and checked them against the frozen generation registry on
  open; and
- required the current policy only for build/mutation, synchronization, new
  certificate issuance, and ingestion, while allowing historical generation,
  certificate, and receipt verification against their internally consistent
  frozen policy.

Verification:

- compilation and direct import/public-facade smoke tests passed during
  integration;
- focused implementation-binding, generation, certificate, storage, Bundle,
  and historical-frozen/current-runtime tests are part of the `61 passed`
  combined focused result recorded in Step 46;
- the implementation is part of commit `9839b61`, which is pushed.

Unexpected finding and necessary adjustment:

- one top-level implementation digest did not prove parser selection,
  distribution identity, or the complete nested protocol, while comparing an
  old certificate only to the ambient runtime would make historical
  verification impossible. The corrected split is: validate frozen content for
  historical proof; require exact ambient equality only before creating a new
  scientific result. No compatibility parser or runtime is retained.

### Step 46 — Final integration, managed-store cutover, and report

Status: complete; managed-store cutover, activated-store smoke tests, current
inventory, migration-path retirement, and final regression verification passed

Verification recorded so far:

- the combined focused Stage 13 audit slice completed with `61 passed`;
- the first complete `tests/test_ds` run completed 260 tests with `259 passed`
  and `1 failed`;
- the current-interface correction in `8718923` and the subsequent runtime,
  identity, scientific-integrity, migration-provenance, public-signature, CLI
  contract, and formal-documentation corrections were integrated and pushed
  through `93f50c7`;
- the complete current `tests/test_ds` suite then passed `277 passed` in
  `93.38 s`, with no failures or skips; and
- the final public-documentation, CLI, signature-tree, and scientific-integrity
  slice passed `29 passed` after the Plan/API prose correction.

#### Interrupted migration rehearsal and mandatory clean restart

The first real managed-store migration was intentionally interrupted before
activation. During that run, the final scientific audit found that the then
current implementation could infer primary literature from identifiers, did
not completely bind AI tool/version metadata or file-backed Evidence
assessment, and had not yet closed the CLI payload/summary contracts or owner
migration provenance. Continuing would have produced internally consistent
bytes under scientifically superseded rules. Reusing its generated oxidation,
spin, or CSD Bundles would therefore have violated the current-only rule.

The interruption did not change `store/active` and did not modify the legacy
source store. Before cleanup, the source was rechecked as a valid SQLite
database with 3 datasets and 252,665 records. Its retained file is
1,136,664,576 bytes with SHA-256
`076096dead161b0689764d275ddd141cdf59e1fe3167881b883877dc8e22b72b`.
The generated rehearsal Bundle root, target generation, and rollback-copy
directory were then removed after explicit path and non-symlink checks. A
subsequent preflight reconfirmed that all three paths and `store/active` were
absent while the source digest remained unchanged.

The necessary adjustment was to close those scientific and protocol defects,
bind normalized owner bootstrap/provenance into the certificate path, rerun the
complete suite, and restart from the untouched legacy database. This was not an
optional cleanup: assessment semantics are certificate inputs, so an old
partial migration cannot be promoted after they change.

#### Completed clean migration and activation

The clean restart completed on 2026-10-15 from the unchanged legacy source.
The one process rebuilt, assessed, certified, ingested, parity-checked, and
atomically activated all three releases. It exited zero after 2:22:39 wall
time (6,632.78 s user CPU, 1,923.90 s system CPU), used 299,816 KiB maximum
resident memory, and used no swap.

The activated database is
`hotpot/dataset/data/store/active/datasets.sqlite3`, with the active directory
symlink resolving to `store/generations/stage13-current`. The database is
5,093,584,896 bytes, has SHA-256
`1a2f02acf77068924d4d9d761220d8c6cb6a2d027930281a8adb1e7c40ab2ee5`,
uses schema `hotpot-ds-store/current`, and has store UUID
`62db5feb-0f62-413c-aae7-08542f93ff35`. SQLite integrity checking passed and
foreign-key checking returned no rows. The cutover parity digest is
`720a27ade4ab1f6c78a85d767fe2ec4d29fb252b1f4a03929aad78bcd676be76`;
the cutover report digest is
`b11abf02a9f1f3520543d84565813b0962aa521e9d99d3bacdf9a559aa09a054`.

Exact release results were:

| Release | Cases | Incomplete | Ordinary | Benchmark | Quality | Evidence |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `oxidation.v001` | 2,175 | 0 | 2,175 | 0 | B1: 2,175 | E2: 2,175 |
| `spin.v001` | 2,175 | 95 | 2,080 | 0 | B0: 95; B1: 2,080 | E2: 2,175 |
| `csd-electronic-state.vreported-text-v1` | 248,315 | 3,473 | 244,842 | 0 | B0: 3,473; B1: 244,842 | E2: 248,315 |

The zero benchmark count is intentional: supplier `Other`/gold assertions are
preserved as provenance, but they do not forge a User/Hotpot direct-entry
decision, and E2 text attribution does not satisfy verified-primary-source
admission. The target contains 252,665 Cases and Case assessments, 592,436
identifiers, 2,317 content-addressed target assets, 4,465 structure references,
and three policies, releases, and authoritative ingestion receipts. Rollback
verification covered all 2,280 source structure assets with no declared hash
mismatch, undeclared loss, or missing negative fact.

`hotpot/dataset/manifests/current-inventory.json` now binds the active store,
release/classification counts, content-addressed asset inventory, activation,
migration audit, and source/audit digests. The generated dataset bytes,
certificates, receipts, migration report, cutover report, and rollback material
remain below the Git-ignored dataset payload root. After their successful use,
the schema-1 migration executable, its bootstrap manifest, and migration-only
tests were removed from the current checkout. Git and the generated signed or
hashed artifacts retain the audit history; no legacy converter or compatibility
runtime remains callable. Ignored orphan bytecode for the retired migration and
rebuild modules was deleted, both imports now raise `ModuleNotFoundError`, and
fresh wheel (664 members) and source-distribution (752 members) builds contain
neither retired module nor the retired bootstrap manifest.

#### Activated-store API, CLI, and performance verification

Read-only API smoke covered dataset discovery and load for all three releases,
archive filtering and incomplete opt-in, index/slice/subset access, stable
Hotpot IDs, labels and structures, Molecule loading, field and SMARTS search,
and synchronized/current Bundle and certificate status. CLI smoke covered
help/list/info/labels, index and identifier `get`, quality/evidence `subset`,
the JSON subset-to-get pipeline, multi-structure role/index selection,
`--xyz`/`--smi`, single/default/converted export, and Bundle status. Protocol
envelopes retained their declared producer keys and requested representations
remained in JSON stdout under `payload.representations`; only the optional human
mirror and diagnostics used stderr.

The activated 244,842-visible-Case CSD view exposed one late performance defect:
identifier selection in API/CLI and dataset-name search still caused per-record
hydration. This was a release issue rather than optional cleanup because a
single-record CLI request exceeded 60 seconds. Commits `73a4c6c`, `1738375`,
and `3b18684` moved identifier access to the target-store index, made metadata
search prefer direct Dataset properties, and routed CLI identifiers through the
same index. The exact command
`hotpot ds get csd-electronic-state AACANI10 --identifier CSD` then completed
in 4.64 s with one result (including process startup and 244,842-header view
construction); direct repeated Python index lookups take about 0.02 s after the
initial load. Regression tests prove that CLI identifier selection does not
call `Dataset.subset`, hydrates one Case for a unique match, and hydrates none
for a multiple-match error.

The final current `tests/test_ds` suite, after deliberate retirement of the
migration-only tests, passed `259 passed` in `65.60 s`, with no failures or
skips. The active inventory was committed in `1328dce`; one-time migration
retirement in `14407dc`; indexed API/search/CLI closure in
`73a4c6c..3b18684`. Those five commits were pushed to
`origin/refactor/unified-dataset-management` on 2026-10-15.

## Push checkpoints

| Checkpoint | Included steps | Commit range | Verification | Push result |
| --- | --- | --- | --- | --- |
| 1 | Steps 0–4: specification, baseline, mapping, Case model, assessment primitives | `b8322f7..e387af9` | baseline `104 passed`; new focused `10 passed` | pushed to `origin/refactor/unified-dataset-management` |
| 2 | Steps 5–7: Gate policy, target store, canonical Bundle facts | `f3c093f..dcff2b6` | focused Stage 13 suites passed | pushed to `origin/refactor/unified-dataset-management` |
| 3 | Steps 8–10: governance, assessment generations, release/trust intake closure | `e3bb3cb..e64dbc5` | governance `8 passed, 1 skipped`; generation `9 passed`; Bundle state `15 passed` | pushed to `origin/refactor/unified-dataset-management` |
| 4 | Steps 11–14: configured assessment, governance CAS, certificate envelope, target ownership | `ffbc244..87599f3` | configured Gate slice `36 passed`; Bundle-state flow `31 passed`; certificate/storage slice `20 passed` | pushed to `origin/refactor/unified-dataset-management` |
| 5 | Steps 15–17: stable asset snapshots, complete Bundle lifecycle, identity-policy intake | `db5b4de..4a0d59f` | generation `12 passed`; Bundle lifecycle `27 passed`; BundleState `19 passed` | pushed to `origin/refactor/unified-dataset-management` |
| 6 | Steps 18–20: query API, atomic Case edits, facade/identifier closure | `5242694..5bbfcc6` | query integration `18 passed`; mutation slice `39 passed, 1 skipped`; facade/storage/query `18 passed` | pushed to `origin/refactor/unified-dataset-management` |
| 7 | Steps 21–24: protocol, current CLI, formal documentation, package resources | `c83d661..798b840` | independent CLI slice `41 passed`; handoff `45 passed`; docs `32 passed`; package declarations verified | pushed to `origin/refactor/unified-dataset-management` |
| 8 | Steps 25–28: sealed governance, documentation assertions, property policy, source-audit boundary | `b5ba161..315a120` | lifecycle/property `72 passed`; sealed slice `51 passed`; docs/help `30 passed`; source metadata `15 passed` | pushed to `origin/refactor/unified-dataset-management` |
| 9 | Steps 29–33: current templates, legacy-code removal, migration foundation, active generations, evidence enrichment | `6a59fe1..cbc97fa` | current suite `160 passed, 1 skipped`; migration slice `34 passed`; paths `21 passed`; evidence slice `22 passed` | pushed to `origin/refactor/unified-dataset-management` |
| 10 | Steps 34–37: current information fence, any-Structure policy, owner migration policy, exact migration parity/activation | `6b94003..6e5b8e7` | migration, storage, Bundle, certificate, documentation, and bootstrap slice `51 passed` | pushed to `origin/refactor/unified-dataset-management` |
| 11 | Steps 38, 39, and 41: mandatory Ed25519, checker containment, attestation principal state | `541b69d..73eb2fb` | governance, isolation/escape, checker protocol, and attestation regressions passed | pushed to `origin/refactor/unified-dataset-management` |
| 12 | Steps 40 and 42–45: Evidence assets, scientific identity/integrity, generation leases, runtime/parser binding, ingestion CAS, migration normalization | `2bdfb99..a70ed5e` | combined focused slice `61 passed`; first full run `259 passed, 1 failed` | pushed through `a70ed5e` to `origin/refactor/unified-dataset-management` |
| 13 | Step 46 integration closure before cutover: current facade, runtime/identity contracts, scientific evidence integrity, owner migration provenance, returned signatures, closed CLI payloads, formal dataset docs | `8718923..93f50c7` | focused scientific slice `60 passed`; migration slice `20 passed`; CLI slice `38 passed`; complete pre-cutover suite `277 passed`; documentation/API/CLI/scientific slice `29 passed` | pushed through `93f50c7` to `origin/refactor/unified-dataset-management` |
| 14 | Step 46 operational closure: clean rebuild/cutover, active inventory, migration-path retirement, indexed activated-store queries | `1328dce..3b18684` | exact parity/integrity checks; activated API/CLI smoke; final current suite `259 passed`; indexed CSD CLI get `4.64 s` | pushed through `3b18684` to `origin/refactor/unified-dataset-management` |
