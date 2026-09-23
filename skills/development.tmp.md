# Development rule staging ledger

> Status: staged on 2026-09-23.
>
> This file is a non-normative inbox for development rules discovered during
> implementation and review. A rule becomes normative only after it is merged
> into both `development.md` and `development.en.md` during a holistic rewrite.
> This file is replaced after each integration cycle; it is not an append-only
> history.

## Integration procedure

1. Record each proposed rule here with a stable ID, scope, rationale, and the
   Git commits that supplied implementation evidence.
2. Review whether the rule is repository-wide or belongs in a module contract.
3. Rebuild the Chinese and English normative documents together. Consolidate
   overlaps and remove obsolete wording instead of appending another section.
4. Record the integration date and evidence commits in both normative files.
5. After integration, replace this batch with an empty inbox template.

## Staged rules

### DEV-COMP-001 — Evidence-based compatibility

- **Scope:** repository-wide.
- **Rule:** Compatibility exists only for an explicitly supported public API,
  artifact schema, Python version, or backend version. Version selection belongs
  at one package composition root; version-specific behavior belongs in isolated
  adapters; shared domain logic remains a single source of truth. Parallel
  façades must expose equivalent public contracts. Speculative aliases, fallback
  field names, and wrappers without a historical contract are prohibited.
- **Rationale:** compatibility branches had spread into shared force-field code,
  while newly introduced trajectory fields were incorrectly treated as legacy
  formats.
- **Evidence commits:** `0e90c63`, `c604ef5`, `c5e53c9`, `f5a66a7`, `15cb23c`.

### DEV-ARCH-001 — Separate facts, evaluation, control, recording, and presentation

- **Scope:** repository-wide.
- **Rule:** A fact-producing layer reports measurements, relationships, and
  uncertainty. A policy layer interprets those facts. A controller alone decides
  retries, perturbations, rollback, selection, and exit. A recorder stores and
  queries observations without controlling the workflow. Presentation and
  persistence options must not alter scientific execution. Similar-looking code
  is abstracted only when its invariants, lifecycle, failure semantics, and
  ownership are genuinely shared.
- **Rationale:** geometry/force-field ownership and trajectory/movie lifecycle
  required repeated separation before each layer had one responsibility.
- **Evidence commits:** `aa72c67`, `b8dc637`, `5e344ad`, `bf67119`, `ebb4a53`,
  `dc3fc2b`.

### DEV-ERR-001 — Preserve explicit failure semantics and evidence

- **Scope:** repository-wide.
- **Rule:** Execution errors, unusable results, and usable results that fail a
  scientific quality criterion are distinct outcomes. Preserving a terminal
  value, report, or diagnostic trace never converts failure into success.
  Unknown errors propagate; allowed recovery is named, observable, documented,
  and tested. When the API promises diagnostic evidence, it must remain
  accessible on both warning-return and exception paths.
- **Rationale:** several force-field paths originally discarded the most useful
  failed structure or terminal frame.
- **Evidence commits:** `43b83d9`, `186d7b4`, `1ea39cc`, `81fd33b`.

### DEV-OBS-001 — Optional scientific history and portable persistence

- **Scope:** repository-wide when a workflow offers history, trajectory, or
  provenance capture; it does not require every workflow to record all states.
- **Rule:** Recording must be opt-in or bounded according to the API contract,
  after considering memory, compute, and I/O cost. Record only the state required
  to interpret each observation; when topology changes, coordinates alone are
  insufficient. The recorder cannot choose workflow transitions. Persisted
  artifacts require an explicit schema/version, units, strict serialization,
  portable paths, round-trip tests, and deterministic overwrite cleanup. Lossy
  views must declare omitted information.
- **Rationale:** complete force-field histories are valuable for diagnosis, but
  unconditional capture can be expensive and coordinate-only movies cannot
  represent topology-changing workflows.
- **Evidence commits:** `81e02de`, `3631c7c`, `5795d8f`, `ebb4a53`, `dc3fc2b`.

### DEV-MOD-001 — Explicit module surface and readable source layout

- **Scope:** repository-wide for new or substantially refactored modules.
- **Rule:** Declare the public surface near the top with `__all__`. Place
  exceptions, enums, and data contracts before implementation helpers. Group
  private helpers by responsibility and place public operations after their
  supporting implementation, from lower-level to higher-level entry points.
  Public or persisted finite-state vocabularies should use `Enum`; local-only
  type restrictions may use `Literal`. Production modules must not retain unused
  imports.
- **Rationale:** package splits and façade work became easier to review only
  after the public surface, contracts, and helper boundaries were explicit.
- **Evidence commits:** `e789e7a`, `d08792a`, `f26ca0f`, `6b8755e`, `ba67c91`.

### DEV-CLEAN-001 — Remove superseded internal paths

- **Scope:** repository-wide.
- **Rule:** Once a new source of truth or lifecycle is connected, search all
  consumers and remove superseded private helpers, fields, wrappers, tests, and
  speculative compatibility branches. Do not retain two active implementations
  of the same responsibility. Zero in-repository consumers are not by themselves
  proof that a documented public abstraction is dead; public removal requires an
  explicit compatibility or deprecation decision.
- **Rationale:** duplicate ring, façade, compatibility, and trajectory paths kept
  responsibilities ambiguous until their old consumers and fields were removed.
- **Evidence commits:** `08639d7`, `87277f0`, `c1ace9f`, `f5a66a7`, `15cb23c`.

### DEV-GOV-001 — Stage discoveries and periodically rebuild the normative skill

- **Scope:** maintenance of this development skill.
- **Rule:** Conversation-derived rules first enter this staging file. Integration
  updates both normative language versions together, consolidates existing text,
  and may delete obsolete or duplicated rules. Every newly integrated rule must
  record its integration date and evidence commit hashes. Normative documents are
  current specifications, not chronological append-only logs.
- **Rationale:** the original Chinese document, later English mirror, and
  incremental additions caused structural growth and temporary language drift.
- **Evidence commits:** `df78c7c`, `6e2ebac`, `a34375b`, `58a43df`.

## Deliberately scoped out

- Force-field working-copy transactions remain a force-field workflow contract,
  not a repository-wide rule.
- Full trajectory capture is not a default requirement; `DEV-OBS-001` requires an
  explicit cost and retention decision.
- Detailed test matrices, force-field stage names, retry counts, ring-size
  thresholds, and archive file layouts remain in module-specific documentation.
