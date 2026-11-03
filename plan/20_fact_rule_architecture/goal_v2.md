Status: review draft; not effective. This is a proposed replacement for [goal.md](goal.md), which remains effective pending the owner's explicit approval. Review notes and optional suggestions are in [goal_v2_review.md](goal_v2_review.md).

Complete all remaining approved work under the current [Plan20 implementation plan](fact_rule_architecture_implementation_plan.md) as the single implementation mainline. Within Plan20's fact-carrier/rule architecture, lifecycle, validation, reporting, and acceptance framework, complete the functional requirements established by [Plan12](../12_xtb_workflow/README.md). Achieve Plan20's global objective rather than optimizing for any current checkpoint. Complete the applicable implementation, migration, scientific and engineering validation, reports, and validated Git delivery. Establish Stage0–12 main-scope acceptance and its validated `B_exp` before the separately gated Stage13A/B experiment proceeds through `D_exp` and `I_exp`; the experiment must not substitute for or alter the preceding acceptance result.

## Main-agent ownership and alignment

This goal is directional rather than a frozen micro-plan. The main agent is the accountable technical and scientific decision-maker at the code, integration, and experimental-validation boundary. It owns the active implementation route and may revise internal design, stage-local sequencing, algorithms within approved scope, repair/fallback routes, diagnostic placement, warning/break policy, tests, and evidence strategy when direct code inspection or experimental results support a better route.

Such revisions must preserve the global objective, settled owner direction, approved public-interface boundaries, scientific constraints, data integrity, and applicable final acceptance criteria. They must not conceal failures, reduce declared work or denominators, weaken a scientific or integrity rule or bypass an approved acceptance gate merely to obtain more successes, or introduce an unspecified public API or schema change. Preserve approved scientific algorithms and trained-feature meanings through migration; a scientific change requires explicit scope and attributable evidence.

Before entering a stage or separately gated sub-stage, align the route with `fact_rule_architecture_implementation_plan.md`, its current amendments and normative companions, accepted owner decisions, the actual repository, completed evidence, and open predecessor gates. Maintain that alignment as work develops, especially after material plan revisions or architectural or scientific route changes. When evidence exposes a contradiction, stop at the smallest coherent checkpoint, review the affected current records, correct obsolete wording within the approved direction, and record the decision, affected scope, and verification consequences before continuing. Stage completion remains a checkpoint within the overall goal.

## Scientific rigor and high-throughput stability

Hotpot is a high-throughput chemistry harness intended to produce useful, low-cost, scientifically bounded structures and properties for screening, downstream calculations, model inputs, and further refinement. It provides flexible cheminformatics methods for chemists working in concrete environments.

**Principle — Pursue “Not Error” within stated domains.** Hotpot supplies methods and safeguards against explicit errors under declared representation conventions, method assumptions, backend capabilities, and conditions. Researchers retain their interpretation and modeling choices; Hotpot does not establish a uniquely correct empirical or theoretical description. The main agent remains accountable for the technical and scientific implementation and must explain its assumptions and limitations.

> **Example 1 — Environment-dependent chemical interpretation**
>
> Partial charges, metal coordination counts, bond lengths, conformers, and MCA predictions depend on methods, representations, and environments. A tool can expose an inconsistency or an observed limitation under named conditions; it cannot proclaim a universally unique value or structure for every research setting.

**Principle — Attribute values to their methods and conditions.** Interpret each Hotpot-produced value through the Hotpot version, invoked rule, calculation or model, effective parameters including defaults, and relevant execution conditions. Reproducibility must account for declared numerical and stochastic variation. Ordinary success returns the requested value, which may itself be a candidate set, distribution, or trajectory; an existing command may return `None` under its contract. Do not repeat known inputs or add a generic success status, result envelope, quality grade, admissibility label, or fabricated confidence score. Necessary internal diagnostics, provenance, source bindings, and lifecycle metadata remain at their owning calculation, control, attribution, or persistence boundaries without expanding ordinary output.

**Principle — Keep method meaning and scientific support in their owning documents.** Versioned companion documentation describes implementation details, representation conventions, assumptions, and applicability limits. Benchmark-validation reports contain available scientific support, including literature where appropriate. Missing calculations, corpora, software, or hardware remain explicitly missing, with the required protocol and work identified and results left blank. Scientific confidence depends on the method, domain, implementation, evidence, and intended use together. A method-defined uncertainty may follow its reviewed contract when calibration and validation are explicit; no universal confidence label or scalar is imposed.

**Principle — Scope hard safeguards to the contract they protect.** Atom, electron and charge accounting, multiplicity, geometric impossibilities, and finite-energy requirements must be defined for the relevant method, backend, representation, and conditions. A named hard rule rejects the affected input, candidate, request, or result within that domain, reports the exact violated rule, and preserves available observations. Rejection remains local unless continuing would make later results untrustworthy. Heuristic geometry and energy thresholds must not become universal claims of chemical impossibility.

> **Example 2 — Named local rejection**
>
> Unintended charge loss during a conversion violates a charge-preservation contract. An electron-count/multiplicity mismatch outside a backend's legal state domain invalidates that request. A non-finite energy cannot satisfy a method's requirement for a finite energy. Each case retains its diagnosis while independent samples continue.

**Principle — Findings describe observations; control policy determines the response.** Only a problem discovered during evaluation, otherwise unavailable to the caller and relevant to interpretation or control, needs a finding. Use a stable, domain-scoped code, name the exact rule or phenomenon, and include the affected subject and only the measurements, reference, threshold, unit, or reason needed to explain it. Successful evaluations with no relevant problem remain silent. Non-hard findings normally report and continue, retaining a defined, finite, attributable observation. A finding must not silently erase, replace, or repair that observation. Missing required units, mapping, attribution, schema integrity, or transaction integrity makes a result unavailable even if its numeric fields are finite and requires a named local integrity failure.

> **Example 3 — Geometry and finite-energy observations**
>
> Unusual atomic proximity, a bond crossing a ring, or apparent ring/bond interpenetration may produce a named geometry finding under stated conditions. Such a finding alone does not establish universal impossibility. An unreasonable but finite energy needs a comparable method, composition/state, reference, unit, and threshold; the measured value remains an observation.

The diagnostic content stays the same when warning/break policy changes. A user may promote a specific finding code to an operation, sample, or batch break; that is expected control. Unexpected termination is reserved for objective inability to proceed or damage to identity, attribution, mapping, schema, shared state, transactions, or infrastructure that makes subsequent results untrustworthy. For each actual rule, gate, adapter boundary, and workflow stage, determine, record, and test the behavior in its owning records:

`finding code | declared domain and trigger | whether a defined value remains available | default warning/local-failure behavior | permitted per-code break scope | commit/fallback consequence`

User-supplied facts, Hotpot rule results, and toolkit projections remain distinct. Rules, fallbacks, sanitization, aromaticity perception, Kekulé assignment, and adapters must not silently rewrite authoritative input. A conflict requires incompatible statements about the same subject, authoritative fact slot, scope, lifecycle, and representation convention.

> **Example 4 — Equivalent projections and distinct bond semantics**
>
> An aromatic representation and a valid coupled single/double Kekulé assignment are normally equivalent projections. Covalent-bond count, operational valence, coordination number, and covalent contribution are distinct quantities. An adapter's unsupported or lossy coordination projection must be reported at that boundary, without redefining Hotpot facts or declaring the chemistry impossible.

Independent sample failures must be isolated, attributed, and retained in stable denominators. Retry, repair, and fallback retain original failures, actual attempts, and available last finite observations, together with changes material to interpretation. Retained observations do not turn a failed operation into a success. Approved hard rules and integrity protections must not silently regress; changes in heuristic observations remain visible and explained by their methods, inputs, and versions.

## Architecture and implementation discipline

Chemical business facts and semantics come first. Core representation, chemical semantics, and intended end-to-end scientific workflows take priority over peripheral integrations. Dependencies necessary for those capabilities inherit their priority. The appendix states the current Plan20 business order; detailed implementation mapping belongs in the implementation plan.

Complete the architecture with one authoritative owner for each durable fact, read-only rules, small named operations, and validated atomic commits with rollback. Rule outputs must be immutable and isolate nested numeric buffers. Keep dependency invalidation finite and attributable; never present a stale derived result as current or rewrite an already returned immutable scientific observation. Preserve ordinary chemical APIs, efficient scalar and batch access, and the sole topology foundation. Trace and migrate actual writers, consumers, adapters, and rollback paths together, removing superseded implementations, duplicate truth, and unapproved compatibility routes. Implement Plan12 through these same owners. Plan21 retains MetalCluster runtime work; Plan20 retains its approved design registration.

Use the execution order **chemical semantics > acceptable performance > all-version adaptation**. Deliver and validate core Python 3.11/CP311 behavior first, then adapt the other planned Python 3.10–3.14 versions. State untested and deferred versions honestly, and claim support only where evidence exists.

Choose efficient methods from the outset, accounting for batch operations, local reuse, conversion costs, and resource bounds, without requiring comprehensive optimization at every stage. The main agent may accept an explained, usable degradation within its authorization and record the workload, purpose, completion, timing, and resource basis. The owner's approximately fivefold slowdown example illustrates contextual judgment and sets no numeric gate. Acceptance does not impose universal parity or a fixed allowance for rare small regressions. OOM, unbounded resource growth, and failure to complete representative batches remain reliability problems on the affected paths.

Measure comparable semantic work under declared methods, inputs, and resource conditions. Preserve raw measurements, failed attempts, actual work, and uncertainty. Do not manufacture success or performance gains by deleting samples, changing denominators, reducing computation, suppressing findings, weakening rules, or changing backends without attribution. Use small coherent implementation checkpoints and preserve caller-owned and unrelated work.

## Verification, records, and completion

Judge chemical reasonableness and justified semantic expectations. Before evaluating a bounded business change, state the expected facts, allowed differences, rejection conditions, and necessary checks. Historical evidence informs comparison without requiring universal historical identity, byte equality, or a fixed success count. Hashes establish artifact attribution and transport integrity; they do not establish chemical truth. Exact checks remain appropriate for genuine deterministic contracts and unchanged or lossless transport. Intended reactions, protonation, electron transfer, or other declared transformations must be judged by their allowed changes rather than a false universal conservation rule.

For numerical approximations and stochastic methods, declare the method, state, units, context, tolerances, and invariants appropriate to the task. Use justified convergence, repetition, distribution, or stability evidence where needed. Retain actual values, attempts, failures, and denominators; do not relax thresholds after observing results to absorb errors.

Evaluate, preserve, and compare actual business execution outcomes. Distinguish a broken validator from broken business behavior: a checker exit code alone establishes neither business success nor its chemical meaning. Favor simple, robust validators that standardize meaningful checks and support business execution. A peripheral checker failure may be deferred with its limits recorded; required business still needs trustworthy direct or alternative evidence. Bypassing a tool cannot fabricate a pass, and required work lacking evidence remains unverified.

Use the maintained [execution-unit/business mapping](functional_707_business_mapping.md) and verification records to select checks by affected behavior and risk. Inventory size and historical pass counts are not coverage targets. Outside major milestones, skip unrelated checks, rerun affected checks, and reuse valid unaffected evidence whose source, inputs, methods, and scope still apply. New changes, failures, and unresolved concerns justify further verification. Broad execution, reflection, fresh archives, and profiling do not form a universal prerequisite chain for every business slice.

At the appropriate delivery and acceptance milestones, complete the applicable technical and scientific validation for the approved scope and support claims: focused/default tests, source and installed-package behavior, native/ABI boundaries, clean wheel and sdist delivery, documentation examples, required backend/model workflows, external/GPU tiers, and comparable performance reports. These obligations remain real work even when their execution is staged. A private native candidate supports only its tested source and installation boundary; ordinary installable capability needs actual delivery evidence. No unexecuted tier may be described as passed.

Maintain [implementation_log.md](implementation_log.md) as an append-only execution record. Record current alignment, decisions, API and owner effects, changed responsibilities, exact commands, source/artifact identities, results and pass/fail/skip counts, scientific observations, performance and resources, failures, missing evidence, remaining gates, and commits. Package each small coherent business change with its necessary checks and limitations in reviewable commits and normal pushes. Keep the final report consistent with final code, the API tree, deletion/consolidation evidence, scientific and performance records, delivery evidence, and known limitations.

Distinguish bounded usable delivery, stage completion, main-scope acceptance, and overall goal completion. Deferred work retains an explicit status, impact, responsible owner, and conditions for later verification; deferral is neither a pass nor silent scope deletion. Existing approved scope exceptions follow their own rules. CP311 evidence alone establishes only the tested boundary. Completion requires the full approved Plan20 mainline, integrated Plan12 requirements, applicable final scientific and engineering acceptance under this goal, consistent logs and reports, and validated Git milestones. Required unfinished work remains open; staged compatibility and checker deferral do not waive final obligations.

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
