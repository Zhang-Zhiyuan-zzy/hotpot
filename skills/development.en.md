# Hotpot Development Standard

[中文版](development.md) · [Rule staging area](development.tmp.md)

This document is Hotpot's current development contract for human developers and
automated coding agents. It is a curated specification of the current rules,
not a chronological discussion log.

The terms **MUST**, **SHOULD**, and **MAY** are normative. Module contracts MAY
add constraints, but they MUST NOT implicitly weaken this document.

## 1. Goals and rule governance

Hotpot's core goal is to provide unified, inspectable chemical objects and
computational infrastructure suitable for coordination chemistry. Every change
MUST prioritize:

1. explicit chemical and numerical semantics, without silent guesses that
   manufacture apparently successful results;
2. visible data ownership, backend boundaries, units, applicability domains,
   and failure states;
3. reuse of established abstractions without unnecessary parallel
   implementations;
4. consistent behavior in the source tree, installed packages, and explicitly
   supported runtimes;
5. logic that is testable, reviewable, and reversible, without presenting an
   experimental conclusion as a validated contract.

### 1.1 Staging and integrating rules `[DEV-GOV-001]`

- A rule discovered through conversation, code review, or failure analysis is
  first recorded in `development.tmp.md` with its scope, rationale, and evidence
  commits.
- Integrating rules MUST rebuild the Chinese and English normative documents
  together, consolidate overlap, and remove wording that is obsolete or no
  longer applicable.
- Normative documents are not extended indefinitely. Every newly integrated
  rule MUST record its integration date and source commit hashes in the
  provenance table.
- `development.tmp.md` retains only the active batch and the most recent
  integration receipt; it is not a permanent append-only history.

## 2. Establish the contract before changing code

Before implementation begins, developers MUST:

1. search Core, I/O, conversion, graph, geometry, search, calculator, and model
   directories for identical or related implementations;
2. identify public entry points, data ownership, index conventions, units,
   exceptions, and downstream consumers;
3. distinguish structural cleanup, defect repair, compatibility work,
   scientific-semantic changes, and performance work;
4. establish a minimal regression test before fixing a defect and a behavioral
   equivalence baseline before a pure refactor;
5. define the contract, applicability domain, migration impact, and acceptance
   evidence before changing public behavior.

MUST NOT copy a converter, reader, parser, site detector, or model invocation path
to bypass an existing implementation. Extract a shared helper only when input,
output, invariants, ownership, lifecycle, and failure semantics are genuinely
shared. MUST NOT reduce superficial line count by introducing many `mode` flags,
Booleans, broad callbacks, or unions.

## 3. Architecture and responsibility boundaries

### 3.1 Hotpot objects are the internal source of chemical truth

- Normalize external input at the system boundary through
  `hotpot.cheminfo.convert.to_hotpot_mol()`.
- Internal chemical semantics, atom indices, site detection, and attached
  results use Hotpot `Molecule`, `Atom`, and `Bond` objects.
- Conversion from RDKit, Open Babel, Pybel, or third-party graphs MUST exist only
  in shared boundary code and MUST NOT be reimplemented in individual features.
- Open Babel is responsible for reading the molecular formats it supports and
  supplying the perception information it can provide; its results remain
  subject to Hotpot boundary contracts.
- Conversion MUST retain atom order, known bond semantics, and available
  metadata. Hotpot indices are 0-based; only human-facing output MAY display
  1-based indices.
- An existing `Molecule` input SHOULD NOT be copied, reparsed, or round-tripped
  through SMILES without cause, because that can lose object identity,
  coordinates, bond metadata, or coordination information.
- A third-party backend supplies perception or numerical evidence; it MUST NOT
  automatically become the normative source for Core chemistry semantics.

When a mutable-object workflow requires transaction and commit semantics, its
owning module contract MUST define them. This repository-wide standard does not
impose one transaction model on every Hotpot function.

### 3.2 Separate mathematical facts, scientific semantics, and workflow responsibilities `[DEV-ARCH-001]`

- Mathematical fact layers and scientific semantic layers MUST be strictly
  distinguished and SHOULD be separated in module and code structure wherever
  practical.
- Mathematical modules such as `geometry` provide only mathematical objects,
  measurements, relations, degeneracy, and uncertainty. They MUST NOT judge
  whether a structure is chemically or physically reasonable, realistic,
  high-quality, or applicable, and MUST NOT choose a repair policy.
- Tolerances used for floating-point stability, degeneracy detection, or
  mathematical relation classification belong to the mathematical layer.
  Thresholds, scores, and acceptance rules that express a chemical or physical
  standard MUST belong to the `chemistry`, `forcefields`, or other scientific
  module that owns that meaning.
- A scientific semantic layer MAY consume mathematical facts and map them to
  chemistry, physics, model-domain, or force-field quality conclusions. These
  scientific judgments MUST NOT be placed back into a mathematical module.
- A controller alone decides retries, perturbations, rollback, selection, and
  exit; these decisions remain explicit in the main workflow.
- A recorder MAY store, query, score, rank, and persist facts, but it MUST NOT
  choose workflow transitions.
- CLI, plotting, and movie belong to presentation; serialization and persistence
  belong to delivery. Both MUST NOT alter scientific execution.
- Core convenience methods and CLIs SHOULD be thin entry points and MUST NOT
  duplicate domain workflows.

## 4. Public contracts, compatibility, and source structure

### 4.1 Evidence-based compatibility `[DEV-COMP-001]`

- Preserve compatibility only for an explicitly supported released public API,
  artifact schema, Python version, or backend version.
- Public return types, object relationships, read-only properties, and exception
  semantics MUST NOT change merely for implementation convenience.
- Internal fields without a historical contract use the strict current
  contract. Speculative fallback field names, aliases, and wrappers are
  prohibited.
- Version or backend selection MUST be concentrated at one explicit composition
  boundary. Specific behavior belongs in isolated façades or adapters, while
  shared domain logic remains a single source of truth.
- Parallel façades MUST have equivalent public names, signatures, return types,
  units, and exception contracts, verified by automated tests.
- A reserved but unimplemented public parameter MUST state its current behavior
  in the docstring and MUST NOT pretend to have an effect.

### 4.2 Types and chemical-object naming

- `Any` is prohibited except at the smallest dynamic third-party boundary that
  genuinely cannot be represented statically.
- An unavoidable `Any` MUST have its reason documented in an adjacent comment or
  document and MUST NOT spread along the call chain into internal domain logic.
- Known Hotpot objects use concrete types such as `Molecule`, `Atom`, `Bond`,
  and `Crystal`. Resolve circular imports with `TYPE_CHECKING`, deferred
  annotations, or forward references.
- Structural polymorphism uses a minimal `Protocol`, type alias, generic, or
  union. Serialized payloads prefer `TypedDict`, dataclasses, or recursive value
  types.
- Class names express complete domain concepts; local variables and parameters
  prefer concise conventional names such as `mol`, `atom`, `bond`, and `cbond`.
- When source, ownership, or lifecycle differs, retain the object and role in
  the name, such as `source_mol`, `clone_mol`, `working_mol`, or `target_atom`.
- When a meaning is the sole default at an API layer, use the base name. Add a
  qualifier only when a genuinely competing meaning exists at that layer.

### 4.3 Public module surface and layout `[DEV-MOD-001]`

For new or substantially refactored modules:

- declare the public surface near the top with `__all__`;
- place exceptions, enums, and data contracts before implementation helpers;
- group private helpers by responsibility;
- place public operations after their supporting implementation, ordered from
  lower-level operations to higher-level entry points;
- public or persisted finite-state vocabularies SHOULD use `Enum`; local static
  type restrictions MAY use `Literal`;
- do not retain unused imports in production modules. Star imports are allowed
  only at a package composition point governed by `__all__`.

### 4.4 Cleanup after a replacement takes ownership `[DEV-CLEAN-001]`

- Once a new source of truth, lifecycle, or entry point is connected, MUST
  search the repository for all consumers.
- Superseded private helpers, fields, wrappers, tests, and unsupported
  compatibility branches MUST be removed. Two active implementations of one
  responsibility MUST NOT remain.
- Zero in-repository consumers are evidence of dead code, but are not sufficient
  grounds to remove a documented public abstraction.
- A legacy public API that MUST remain MUST use an explicit deprecation contract,
  not a hidden wrapper maintained indefinitely.

## 5. Chemical and scientific semantics

- Different chemical interpretations use named profiles or enums, not ambiguous
  Booleans or hidden branches.
- Compute semantic views on copies or read-only views. MUST NOT mutate the source
  graph by temporarily deleting and restoring bonds.
- `BondKind` is the source of truth for bond semantics. MUST NOT infer `UNKNOWN`,
  `ZERO`, or `DATIVE` solely from numeric bond order.
- The ring family is part of an algorithm contract. `cycle basis`, `relevant
  cycles`, and other ring sets MUST NOT be conflated, and downstream behavior
  SHOULD NOT depend on an arbitrary basis or input order.
- Units MUST be explicit in APIs, fields, headers, and plot labels. Normalize
  backend quantities once at the boundary into the internal unit.
- A semantic profile MUST NOT silently recompute implicit hydrogen,
  aromaticity, or bond types. Backend information loss MUST be represented
  through `UNKNOWN`, an exception, a fixture, or documentation.

Search retains the `Query*`, `Substructure`, `Searcher`, and `Hit/Hits` public
abstractions. Production SMARTS matching uses NetworkX-backed Hotpot search.
RDKit MAY support features, conformers, format bridges, and drawing, but MUST
NOT silently replace that source of truth.

## 6. Failures, diagnostics, and control flow `[DEV-ERR-001]`

- Unknown program errors MUST propagate unchanged; they MUST NOT become a
  success, empty result, zero value, or CPU result.
- Execution failure, an unusable result, and a usable result that fails a
  scientific quality criterion are distinct states and MUST be explicit in the
  API.
- MUST NOT use broad `try/except`, layered `if/else`, or default values as an
  unconditional fallback.
- A fallback MAY exist only when the product contract permits it and MUST have a
  name, status marker, documentation, and dedicated tests.
- Preserving a terminal value, last frame, report, or diagnostic trajectory MUST
  NOT declare success; the failure remains explicit.
- When an API promises failure evidence, both warning-return and exception paths
  MUST retain it.
- An uncomputed read-only scientific property SHOULD raise `AttributeError` with
  invocation guidance instead of returning a fabricated value.

## 7. Optional scientific history and persistence `[DEV-OBS-001]`

This section applies only to workflows that offer history, trajectory, or
provenance capture. It does not require every calculation to record every state
by default.

- Full-history capture MUST be explicitly enabled by the user or API. A default
  MAY retain only a bounded summary or selected terminal state required by the
  module contract, and MUST document its resource cost.
- Before implementation, developers MUST assess frame count, state size,
  evidence computation, memory, compression, and I/O. Prefer deduplication,
  pooling, or bounded retention.
- Each observation MUST contain the minimum complete state needed to interpret it.
  When topology changes, a coordinate-only conformer is not authoritative.
- Recording, explicit selection, materialization into a domain object, and disk
  persistence are separate responsibilities.
- A persistent format MUST declare its schema or format version, units, and
  missing-value semantics. JSON MUST NOT contain `NaN` or `Infinity`.
- Paths SHOULD be portable; readers SHOULD NOT depend on pickling arbitrary
  objects; archives MUST pass round-trip tests.
- Overwriting an archive MUST deterministically remove stale artifacts. A lossy
  view MUST declare which information was omitted.

## 8. Models, APIs, and CLIs

- Keep model input conversion, feature or conformer construction, runtime, raw
  output, scientific filtering, Core attachment, and CLI in separate layers.
- A raw prediction is not a chemistry site validated by an applicability domain.
  Reject or explicitly mark out-of-domain input.
- Inference-only releases exclude private training loops, checkpoints, and
  training data. Model artifacts require a manifest, hashes, license,
  applicability-domain documentation, and quantitative parity evidence.
- An explicitly requested compute backend MUST fail when unavailable. Only an
  explicit `auto` mode MAY select another backend automatically.
- Compression, quantization, pruning, or a new model MUST receive quantitative
  parity validation before release.
- A CLI is a thin wrapper over the Python API. Stdout contains only stable,
  redirectable data; logs and warnings go to stderr.
- A CLI MUST NOT reimplement readers, models, site detection, or drawing logic,
  and MUST NOT hide underlying exceptions.

## 9. Verification and delivery

- All tests and test-only fixtures belong under `tests/`. Benchmarks MAY live in
  a separate directory or external artifact, but MUST NOT enter the production
  package.
- Test depth is proportional to risk: pure-function or contract tests, module
  integration, real reader/backend/CLI closure, and an outside-source-tree wheel
  smoke test when installed contents change.
- Cover the success and failure contracts relevant to the change. A test double
  satisfies the current data contract; production code is not weakened to
  accommodate a stale mock.
- When changing a shared API, installed content, or a cross-version path, use the
  repository compatibility runner or CI. Detailed corpora and matrices belong
  to the relevant module test documentation.
- A focused green result proves only its covered scope and MUST NOT be reported
  as repository-wide compatibility.
- Golden expectations require human diff review and MUST NOT be overwritten from
  the current implementation to conceal a regression.

## 10. Performance, dependencies, and runtime boundaries

- Performance work MUST NOT alter chemical semantics, failure contracts, or
  result ordering. Performance claims require reproducible benchmarks.
- Full-history capture MUST be opt-in and default history MUST be bounded.
  Expensive search SHOULD be opt-in or explicitly bounded.
- A cache key or signature covers every atom, bond, connectivity, `BondKind`,
  aromaticity, and semantic state consumed by the calculation.
- A cache MUST NOT be produced through a mutate-source, calculate, then restore
  workflow.
- Potentially combinatorial searches SHOULD offer an existence fast path,
  streaming iteration, and an explicit limit; truncation is never silent.
- Tests, fuzzing, and benchmarks SHOULD use fixed seeds. Production randomized
  algorithms SHOULD expose an optional seed but need not fix it by default;
  sources of nondeterminism MUST be recorded.
- When runtime dependencies, entry points, or package data change, check
  `pyproject.toml`, `setup.py`, and `MANIFEST.in` together.
- Import optional heavy dependencies lazily. Type annotations SHOULD NOT import
  training frameworks or unrelated heavy dependencies.

## 11. Git and workspace discipline

- Develop on a dedicated feature branch.
- Each logical milestone is one independently understandable and reversible
  commit using a Conventional Commit message.
- Test fences, implementation, documentation, and build changes MAY be separate
  commits, but the final series MUST explain why behavior changed.
- MUST NOT commit unrelated user or collaborator changes, untracked directories,
  caches, temporary images, or local environment files.
- MUST NOT use reset, checkout, or whole-file replacement to destroy another
  contributor's uncommitted work. Reconcile intent before editing overlap.

## 12. Definition of Done

- [ ] Contract, ownership, units, applicability domain, and failure semantics are explicit.
- [ ] Existing abstractions were reused; new abstractions have clear invariants and responsibilities.
- [ ] No silent fallback, swallowed exception, unconditional recovery, or fabricated success remains.
- [ ] Risk-proportionate success/failure, integration, real-path, and packaging tests pass, and reporting stays within their scope.
- [ ] If scientific history or persistence is provided, resource cost and schema, round-trip, overwrite, and lossy-output semantics are verified.
- [ ] Superseded paths are removed; public surface, lint, formatting, and `git diff --check` pass.
- [ ] Documentation describes user-visible behavior and limits; commits are atomic and contain no unrelated files.

## 13. Active rule provenance

| Rule | Integrated | Staging commit | Implementation evidence commits |
|---|---|---|---|
| `DEV-COMP-001` | 2026-09-23 | `339c53e` | `0e90c63`, `c604ef5`, `c5e53c9`, `f5a66a7`, `15cb23c` |
| `DEV-ARCH-001` | 2026-09-23 | `339c53e`, `882d9dd` | `aa72c67`, `b8dc637`, `5e344ad`, `bf67119`, `ebb4a53`, `dc3fc2b` |
| `DEV-ERR-001` | 2026-09-23 | `339c53e` | `43b83d9`, `186d7b4`, `1ea39cc`, `81fd33b` |
| `DEV-OBS-001` | 2026-09-23 | `339c53e` | `81e02de`, `3631c7c`, `5795d8f`, `ebb4a53`, `dc3fc2b` |
| `DEV-MOD-001` | 2026-09-23 | `339c53e` | `e789e7a`, `d08792a`, `f26ca0f`, `6b8755e`, `ba67c91` |
| `DEV-CLEAN-001` | 2026-09-23 | `339c53e` | `08639d7`, `87277f0`, `c1ace9f`, `f5a66a7`, `15cb23c` |
| `DEV-GOV-001` | 2026-09-23 | `339c53e` | `df78c7c`, `6e2ebac`, `a34375b`, `58a43df` |

## 14. Module contracts and references

- `tests/README.md`
- `tests/smarts_conformance/SMARTS_CONFORMANCE.md`
- `tests/smarts_conformance/README.md`
- `hotpot/cheminfo/geometry/README.md`
- `hotpot/cheminfo/graph/README.md`
- `hotpot/cheminfo/kekulize/Kekulize.md`
- `hotpot/cheminfo/AImodels/INFERENCE_COMPATIBILITY.md`
- `hotpot/cheminfo/AImodels/mca/MODEL_CARD.md`
- `hotpot/cheminfo/AImodels/mca/README.md`
- `hotpot/cheminfo/AImodels/cbond/README.md`

If this document appears to conflict with a more specific module contract,
first determine the applicable scope and the reason for the conflict, then
update the standard or module design. MUST NOT choose the less restrictive
interpretation unilaterally.
