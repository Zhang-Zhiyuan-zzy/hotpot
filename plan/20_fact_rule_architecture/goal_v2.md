Status: review draft; not effective. This proposed replacement leaves [goal.md](goal.md) effective until the owner's explicit approval. [Review notes](goal_v2_review.md) record incorporated directions and accepted supporting practices.

Complete all remaining approved [Plan20](fact_rule_architecture_implementation_plan.md) work and [Plan12](../12_xtb_workflow/README.md) functional requirements through one implementation mainline and its fact-carrier/rule architecture, lifecycle, validation, reporting, and acceptance framework. Pursue the global objective beyond individual checkpoints through applicable implementation, migration, scientific and engineering validation, reports, and validated Git delivery. Stage0–12 main-scope acceptance and validated `B_exp` must precede the separately gated Stage13A/B experiment through `D_exp` and `I_exp`; the experiment cannot substitute for or alter that acceptance.

## Main-agent ownership and alignment

This goal is directional rather than a frozen micro-plan. The main agent owns the active implementation route and is accountable for technical and scientific decisions at the code, integration, and experimental-validation boundaries. Direct inspection or experimental evidence may justify revisions to internal design, stage sequencing, algorithms within approved scope, repair/fallback routes, diagnostics, warning/break policy, tests, and evidence strategy.

Revisions must preserve the global objective, settled owner direction, approved public-interface boundaries, scientific constraints, data integrity, and final acceptance criteria. Never conceal failures, delete samples, reduce declared work/computation or denominators, suppress findings, weaken scientific or integrity rules, or bypass approved gates to manufacture successes or performance gains; introduce no unspecified public API or schema. Preserve approved scientific algorithms and trained-feature meanings; scientific changes require explicit scope and attributable evidence.

Align stage/sub-stage entry and ongoing work with `fact_rule_architecture_implementation_plan.md`, current amendments and normative companions, accepted owner decisions, the repository, completed evidence, and predecessor gates—especially after material plan revisions or architectural/scientific route changes. On contradiction, stop at the smallest coherent checkpoint, review affected records, correct obsolete wording within approved direction, and record the decision, scope, and verification consequences before continuing. Stage completion is a checkpoint within the overall goal.

## Scientific rigor and high-throughput stability

Hotpot is a high-throughput chemistry harness providing flexible methods for chemists in concrete environments, pursuing “Not Error” within stated domains and useful, low-cost, scientifically bounded structures and properties for screening, downstream calculations, model inputs, and refinement. It need not establish experimental truth or a unique chemical interpretation. Avoid unsupported claims, preserve established scientific behavior, and expose concrete problems without letting an isolated sample destroy a batch's statistical objective.

**Hotpot provides cheminformatics methods and cliff-edge guardrails; it does not replace chemists or the main agent as the final scientific judge.** Researchers retain interpretation and modeling choices. Structures and properties depend on method, representation, and environment; the main agent must explain assumptions and limits.

**Every Hotpot-produced value—discrete, floating-point, structured, or otherwise—must be interpreted through the Hotpot version, invoked method (rule, calculation, AI model, or other implementation), effective parameters including defaults, and relevant execution conditions. These known inputs must not be repeated in the ordinary result.** Reproducibility must account for declared numerical/stochastic variation. Ordinary success returns only the requested value, including a candidate set, distribution, or trajectory; an existing command may return `None` by contract. Add no generic success status, result envelope, evidence grade, admissibility label, or confidence score. Keep necessary internal diagnostics, provenance, source bindings, and lifecycle metadata at their calculation, control, attribution, or persistence owners, outside ordinary output.

**A method's implementation details, representation conventions, assumptions, and applicability limits belong in its versioned companion documentation. Its available scientific support belongs in benchmark-validation reports, not in a trustworthiness assertion emitted by each result.** Reports may cite external literature as available method evidence. If required calculations, corpora, software, or hardware are unavailable, issue an explicit placeholder report: mark evidence as not yet obtained, state what remains to run, and leave results blank; never fabricate confidence or benchmark outcomes.

**Scientific confidence is not a universal discrete label and need not be reducible to a single scalar.** It depends jointly on implementation, representation conventions, applicability domain, validation corpus, observed benchmarks, and intended use. Expose calibrated uncertainty only when the method defines it with explicit calibration procedures and validation evidence; never invent it to replace scientific analysis.

> **Example 1 — Model-derived partial charge**
>
> An RDKit-produced partial charge may remain an ordinary floating-point value. Its scientific meaning comes from the documented charge method, implementation details, domain, and benchmark evidence—not from attaching `HIGH`, `LOW`, `ADMISSIBLE`, or a fabricated probability to that atom value.

> **Example 2 — Evidence location and an unavailable benchmark**
>
> The existing 182/187 corpus is benchmark evidence for the `CBond -> FF` route; CBond-specific scientific support may be supplied by its referenced literature. If a required benchmark cannot currently be executed, its validation document may remain an explicit, versioned placeholder with the missing protocol and evidence clearly marked, rather than assigning confidence in the returned values.

Findings report only interpretation/control-relevant information discovered during evaluation and otherwise unavailable to the caller. Use stable, domain-scoped codes naming the exact rule or phenomenon, with only the affected subject and necessary measured value, reference, threshold, unit, or reason. Successful evaluation without a relevant problem remains silent.

**A named hard-rule gate must reject the affected input, candidate, request, or result within the rule's declared domain. A hard rejection is local to the affected operation or sample unless continuing would make later results untrustworthy.** It must report the exact violated rule and preserve the available diagnostic observation; it must not be converted into a vague rating. Atom, electron and charge accounting, multiplicity, geometric impossibilities, and finite-energy requirements must be defined for the relevant method, backend, representation, and conditions; heuristic geometry and energy thresholds do not establish universal chemical impossibility.

> **Example 3 — Hard-rule rejection without batch loss**
>
> Unintended charge loss in a conversion violates a charge-preservation contract. A non-finite energy cannot satisfy a method’s finite-energy requirement. An electron-count/spin request outside a backend’s legal state domain cannot be projected as a valid request. Each case produces a named local failure, while independent samples continue.

**Every non-hard gate reports a possible problem rather than issuing a scientific verdict or an arbitrary quality grade.** A defined, finite, attributable value remains an observation and is returned by default with a precise finding. The finding must not silently erase, replace, or "repair" that observation. A result with missing or inconsistent required units, mapping or attribution, or with broken schema or transaction integrity, is unavailable and requires a named local integrity failure even when its numeric fields are finite.

> **Example 4 — Possible-problem findings**
>
> - Geometry gates may report atoms that are unusually close, a bond–ring piercing, or apparent ring/bond interpenetration in the current geometry; they do not prove that the molecule cannot exist.
> - Chemistry gates may report an aromaticity inconsistency under a named representation rule or a named valence/bond-order limit; they do not by themselves establish a universal chemical impossibility.
> - Force-field gates may report an implausible finite energy. A high-energy finding is meaningful only with a comparable method, composition/state, reference, unit, and threshold, and the finite value remains reportable.

Findings describe what happened independently of warning/break policy; their content must not change with policy. Non-hard scientific or plausibility findings default to reporting and continuation. Users may promote a specific code to an operation, sample, or batch break—expected control, not unexpected program failure. Unexpected termination requires execution that objectively cannot proceed or damage to identity, result attribution, atom mapping, schema, shared state, transaction integrity, or infrastructure that makes later results untrustworthy.

User-supplied facts, Hotpot rule evaluations, and toolkit projections remain distinct. Hotpot rules, fallbacks, sanitization, aromaticity perception, Kekulé assignment, and external adapters must not silently rewrite authoritative user input. A conflict exists only when statements address the same subject, authoritative fact slot, scope, lifecycle, and representation convention and cannot be reconciled by a valid equivalent projection.

> **Example 5 — Equivalent aromatic projections**
>
> An aromatic representation and a valid coupled single/double Kekulé assignment are normally alternate projections, not a conflict.

> **Example 6 — Distinct bond semantics and adapter limits**
>
> Covalent-bond count, operational valence, coordination number, and covalent contribution are different quantities and must not be equated automatically. Registered adapters translate quietly under documented, versioned maps, minimizing information loss without extra runtime labels. Hotpot-initiated roundtrips preserve original Hotpot facts' authority; independently supplied native adapter objects are authoritative inputs under documented import maps. Calculated observations remain distinct from conversion. Detached carriers inspect only their contents and gates; conversion cannot recreate absent facts or lost upstream relations. Unexpected or unmapped material loss and integrity failures require precise adapter-scoped diagnostics.

Record and test each actual rule, gate, adapter boundary, and workflow stage in its owning records:

`finding code | declared domain and trigger | whether a defined value remains available | default warning/local-failure behavior | permitted per-code break scope | commit/fallback consequence`

This goal does not preselect entries. The main agent may revise balance points as evidence develops, recording and testing decisions within this direction. Distinguish returned observations, unavailable local results, explicit user breaks, and genuinely fatal loss of execution integrity.

Isolate and attribute independent sample failures in stable denominators. Retry, repair, or fallback must retain original failures, actual attempts, available last finite observations, and changes material to scientific interpretation; retained observations do not make a failed operation successful. Preserve approved hard scientific rules, integrity contracts, and benchmark protections. Explain heuristic changes by named methods, rules, inputs, and versions; historical equality and fixed scientific success counts are not universal truth criteria.

## Architecture and implementation discipline

Prioritize chemical business facts, core representation, semantics, and intended end-to-end scientific workflows over peripheral integrations. The appendix gives business priorities, including inherited dependency priority; detailed implementation mapping stays in the plan.

Use one authoritative owner per durable fact, read-only rules, small named operations, and validated atomic commits with rollback. Rule outputs must be immutable with isolated nested numeric buffers. Keep dependency invalidation finite and attributable; never present stale derived results as current or rewrite returned immutable scientific observations. Preserve ordinary chemical APIs, efficient scalar/batch access, and the sole topology foundation. Migrate actual writers, consumers, adapters, and rollback together; remove superseded implementations, duplicate truth, and unapproved compatibility routes. Plan12 uses these same owners. Plan21 retains MetalCluster runtime work; Plan20 retains approved design registration.

Use **chemical semantics > acceptable performance > all-version adaptation**. Deliver and validate core Python 3.11/CP311 first, then adapt the other planned Python 3.10–3.14 versions. Report untested/deferred versions honestly and claim support only with evidence.

Design efficiently from the outset for batch operations, local reuse, conversion costs, and resource bounds; comprehensive optimization is not required at every stage. The main agent may accept explained, usable degradation within authorization, recording workload, purpose, completion, timing, and resources. The owner's approximately fivefold slowdown example illustrates contextual judgment, not a numeric gate. Neither universal parity nor a fixed allowance for rare small regressions applies. OOM, unbounded resource growth, and failure to complete representative batches remain reliability problems on affected paths.

Measure comparable semantic work under declared methods, inputs, and resources. Preserve raw measurements, failed attempts, actual work, and uncertainty. The same failure, denominator, finding, and rule protections apply to performance claims; backend changes require attribution.

## Verification, records, and completion

Before evaluating a bounded change, state expected facts, allowed differences, rejection conditions, and checks. Judge chemical reasonableness through independently assessed positive, negative, and boundary fixtures, adding useful equivalence, round-trip, or semantics-preserving checks; historical outputs or sides sharing one implementation cannot be the sole oracle. History informs comparison without universal historical identity, byte equality, or fixed success counts. Hashes attribute artifacts and transport integrity, not chemical truth; exact checks remain valid for deterministic contracts and unchanged or lossless transport. Judge reactions, protonation, electron transfer, and other declared transformations by their allowed changes, not universal conservation.

For numerical approximations and stochastic methods, declare appropriate method, state, units, context, tolerances, and invariants. Use justified convergence, geometric equivalence, repetition, distribution, or stability criteria and record actual seeds when used. Retain actual values, attempts, failures, and denominators; never relax thresholds after observing results to absorb errors.

Evaluate, preserve, and compare actual business execution, including representative actual backend/model workflows mixing successes and local failures, with attribution, currentness, rollback, and finite-recovery checks. A checker exit code establishes neither business success nor chemical meaning; distinguish broken validation from broken business behavior. Favor simple, robust validators standardizing meaningful checks and supporting execution. Peripheral checker failures may defer with recorded limits; required business still needs trustworthy direct or alternative evidence. Bypassing a tool cannot fabricate a pass; work without evidence remains unverified.

Select checks by affected behavior and risk using the maintained [execution-unit/business mapping](functional_707_business_mapping.md) and verification records. Inventory size and historical pass counts are not coverage targets. Between major milestones, skip unrelated checks, rerun affected checks, and reuse valid unaffected evidence whose source, inputs, methods, and scope still apply. New changes, failures, or unresolved concerns justify further verification. Broad execution, reflection, fresh archives, and profiling are not a universal prerequisite chain for each slice.

At delivery/acceptance milestones, complete applicable validation for approved scope and support claims: focused/default tests, source/installed-package behavior, native/ABI boundaries, clean wheel and sdist delivery, documentation examples, required backend/model workflows, external/GPU tiers, and comparable performance reports. Staged execution does not waive this work. A private native candidate supports only its tested source/installation boundary. Installable capability requires actual delivery evidence, including a representative installed CP311 workflow outside the checkout attributing the actual native, backend, and model dependencies used. No unexecuted tier may be described as passed.

Keep [implementation_log.md](implementation_log.md) append-only: record alignment, decisions, API/owner effects, changed responsibilities, exact commands, source/artifact identities, results and pass/fail/skip counts, scientific observations, performance/resources, failures, missing evidence, remaining gates, and commits. Keep the final report consistent with final code, the API tree, deletion/consolidation evidence, scientific/performance records, delivery evidence, and known limitations.

Use small, validated commits for coherent business progress and normal pushes, enabling prompt isolation and rollback of erroneous business changes. Include related documentation, tests, necessary evidence, checks, and limitations; preserve caller-owned/unrelated work and honest failure evidence. Do not make consecutive documentation-only commits without business progress or commits devoted only to hashes, evidence packets, bookkeeping, or status.

Distinguish bounded usable delivery, stage completion, main-scope acceptance, and overall completion. Deferrals retain explicit status, impact, responsible owner, and verification conditions; they are neither passes nor silent scope deletion. Approved scope exceptions follow their own rules. CP311 evidence qualifies only its tested boundary. Completion requires the full approved Plan20 mainline, integrated Plan12 requirements, applicable final scientific/engineering acceptance, consistent logs/reports, and validated Git milestones. Required unfinished work stays open; staged compatibility and checker deferral waive no final obligations.

## Appendix — Current Plan20 business priorities

These priorities guide execution within the complete approved scope. The implementation plan retains detailed owners, migrations, stages, dependencies, and acceptance mapping.

| Priority | Requirement | Business scope |
|---|---|---|
| 1 | Mandatory | Molecule, Atom, and Bond representation and transformations, including fact ownership, membership/endpoint mapping, copy, components, merge, and related lifecycle behavior. |
| 2 | Mandatory | Graph, aromatic/Kekulé, stereochemistry, valence, and H semantics under explicit representation conventions. |
| 3 | Mandatory | CBond → FF → GFN-FF → xTB, including geometry gates, real stage handoffs, legal states, result attribution, and local failure handling. |
| 4 | Mandatory | MCA, including actual model inference, atom/site mapping, result attachment, and currentness. |
| 5 | Best effort | I/O, including supported structural semantics, record/frame transport, and explicit lossy boundaries. |
| 6 | Best effort | RDKit/Open Babel business interfaces and non-native/Python communication, including actual object conversion and declared support limits. |
| 7 | Defer and report | Other independent third-party communication or functionality issues. |

An I/O, toolkit, model, or other dependency required by priorities 1–4 remains part of that mandatory business path. Lower-priority remaining work stays visible in the project scope and delivery records.
