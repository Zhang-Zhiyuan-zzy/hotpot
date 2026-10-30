Status: effective under the owner's 2026-10-30 authorization.

Complete all remaining approved work under the current `plan/20_fact_rule_architecture/fact_rule_architecture_implementation_plan.md` as the implementation mainline. Within Plan20's fact-carrier/rule architecture, lifecycle, validation, reporting, and acceptance framework, complete the functional requirements established by `plan/12_xtb_workflow/`. Achieve Plan20's global objective rather than optimizing for any current checkpoint. Complete every required implementation, migration, test, scientific execution, build/package tier, performance comparison, report, and validated Git delivery. Establish the Stage0–12 main-scope acceptance and `B_exp` before carrying out the separately gated Stage13A/B experiment; the experiment must not waive or revise the preceding acceptance result.

## Main-agent ownership and alignment

This goal is directional rather than a frozen micro-plan. The main agent is the accountable technical and scientific decision-maker at the code, integration, and experimental-validation boundary. It owns the active implementation route and may revise internal design, stage-local sequencing, algorithms within approved scope, repair/fallback routes, diagnostic placement, warning/break policy, tests, and evidence strategy when direct code inspection or experimental results support a better route.

Such revisions must preserve the global objective, settled owner direction, approved public-interface boundaries, scientific constraints, data integrity, and final acceptance criteria. They must not conceal failures, reduce declared work or denominators, weaken an approved rule or acceptance gate merely to obtain more successes, or introduce an unspecified public API or schema change. When current evidence conflicts with the plan, stop at the smallest coherent checkpoint, reread the complete current plan and normative companion records, revise contradictory or obsolete wording rather than blindly following it, and record the evidence, decision, affected scope, and verification consequences before continuing.

Before every stage begins, the main agent must read `/home/zhangzhiyuan/hotpot/plan/20_fact_rule_architecture/fact_rule_architecture_implementation_plan.md` in full, together with its current revisions and normative companion records. It must then assess the actual repository state, completed evidence, and open gates; revise the plan and this goal where direct evidence requires clarification or correction; and explicitly align the stage route with the global objective. Repeat this full alignment after every material plan revision and before every architectural or scientific route change. Stage completion is a checkpoint, not a substitute for full-plan completion.

## Scientific rigor and high-throughput stability

Hotpot is a high-throughput chemistry harness intended to produce useful, low-cost, scientifically bounded structures and properties for screening, downstream calculations, model inputs, and further refinement. It is not required to establish experimental truth or a universally unique chemical interpretation. It must nevertheless avoid unsupported claims, preserve established scientific behavior, and expose concrete problems without allowing an isolated sample to destroy the statistical objective of a batch.

**Hotpot provides cheminformatics methods and cliff-edge guardrails; it does not replace chemists or the main agent as the final scientific judge.**

**The identity of every Hotpot-produced value—discrete, floating-point, structured, or otherwise—is determined by the Hotpot version, the invoked method (rule, calculation, AI model, or other implementation), the supplied parameters including effective defaults, and all other information known before execution. These known inputs must not be repeated in the ordinary result.** The requested value may itself be a candidate set, distribution, or trajectory. Ordinary success returns only that requested value; it does not add a generic success status, result envelope, evidence grade, admissibility label, or confidence score. Necessary internal diagnostics, provenance, source bindings, and lifecycle metadata remain at their owning boundaries for calculation, control, attribution, or persistence; this does not add them to ordinary output.

**A method's implementation details, assumptions, and applicability limits belong in its companion documentation. Its available scientific support belongs in benchmark-validation reports, not in a trustworthiness assertion emitted by each result.** A validation report may cite external literature where that is the available method evidence. When the required calculation, corpus, software, or hardware is not yet available, create an explicit placeholder report that marks the evidence as not yet obtained, states what remains to be run, and leaves results blank; never fabricate a confidence verdict or benchmark outcome.

**Scientific confidence is not a universal discrete label and need not be reducible to a single scalar.** It arises from the concrete implementation, representation conventions, applicability domain, validation corpus, observed benchmark behavior, and intended use together. A calibrated uncertainty may be exposed only when the method defines it and its calibration procedure and validation evidence are explicit; it must not be invented as a replacement for scientific analysis.

> **Example 1 — Model-derived partial charge**
>
> An RDKit-produced partial charge may remain an ordinary floating-point value. Its scientific meaning comes from the documented charge method, implementation details, domain, and benchmark evidence—not from attaching `HIGH`, `LOW`, `ADMISSIBLE`, or a fabricated probability to that atom value.

> **Example 2 — Evidence location and an unavailable benchmark**
>
> The existing 182/187 corpus is benchmark evidence for the `CBond -> FF` route; CBond-specific scientific support may be supplied by its referenced literature. If a required benchmark cannot currently be executed, its validation document may remain an explicit, versioned placeholder with the missing protocol and evidence clearly marked, rather than assigning confidence in the returned values.

Only information discovered during evaluation, otherwise unavailable to the caller, and important to interpretation or control should produce a diagnostic finding. Each finding must use a stable, domain-scoped code, name the exact rule or observed phenomenon, and contain only the affected subject and the measured value, reference, threshold, unit, or reason needed to understand the problem. Successful evaluations that discover no relevant problem remain silent.

**A named hard-rule gate must reject the affected input, candidate, request, or result within the rule's declared domain. A hard rejection is local to the affected operation or sample unless continuing would make later results untrustworthy.** It must report the exact violated rule and preserve the available diagnostic observation; it must not be converted into a vague rating.

> **Example 3 — Hard-rule rejection without batch loss**
>
> A non-finite energy is not a valid energy value. A hard electron-count/spin legality violation cannot be projected as a valid backend request. Each case produces a named local failure, while independent samples continue.

**Every non-hard gate reports a possible problem rather than issuing a scientific verdict or an arbitrary quality grade.** A finite, attributable value remains an observation and is returned by default with a precise finding. The finding must not silently erase, replace, or "repair" that observation. A result with missing or inconsistent required units, mapping or attribution, or with broken schema or transaction integrity, is unavailable and requires a named local integrity failure even when its numeric fields are finite.

> **Example 4 — Possible-problem findings**
>
> - Geometry gates may report atoms that are unusually close or a bond–ring piercing in the current geometry; they do not prove that the molecule cannot exist.
> - Chemistry gates may report an aromaticity inconsistency under a named representation rule or a named valence/bond-order limit; they do not by themselves establish a universal chemical impossibility.
> - Force-field gates may report an implausible finite energy. A high-energy finding is meaningful only with a comparable method, composition/state, reference, unit, and threshold, and the finite value remains reportable.

A finding and its control action are separate concerns. The finding states what happened; warning or break states how the configured workflow responds. The diagnostic content must not change when policy changes. Non-hard scientific or plausibility findings default to reporting and continuation. The user may promote a specific finding code to a break at the operation, sample, or batch boundary. Such an explicit break is expected control, not an unexpected program failure. Unexpected termination is reserved for cases where execution objectively cannot proceed, or identity, result attribution, atom mapping, schema, shared state, transaction integrity, or infrastructure damage makes later results untrustworthy.

User-supplied facts, Hotpot rule evaluations, and toolkit projections remain distinct. Hotpot rules, fallbacks, sanitization, aromaticity perception, Kekulé assignment, and external adapters must not silently rewrite authoritative user input. A conflict exists only when statements address the same subject, authoritative fact slot, scope, lifecycle, and representation convention and cannot be reconciled by a valid equivalent projection.

> **Example 5 — Equivalent aromatic projections**
>
> An aromatic representation and a valid coupled single/double Kekulé assignment are normally alternate projections, not a conflict.

> **Example 6 — Distinct bond semantics and adapter limits**
>
> Covalent-bond count, operational valence, coordination number, and covalent contribution are different quantities and must not be equated automatically. Coordination and covalent bond semantics remain explicit Hotpot concepts; an RDKit or other adapter limitation must be reported as an adapter-scoped unsupported or lossy projection, not reclassified as a chemical impossibility or used to redefine Hotpot facts.

For every actual rule, gate, adapter boundary, and workflow stage, the main agent must determine and record:

`finding code | declared domain and trigger | whether a defined value remains available | default warning/local-failure behavior | permitted per-code break scope | commit/fallback consequence`

This goal does not preselect every entry in that table. The main agent may revise individual balance points as implementation and experimental evidence develop, provided the direction above remains satisfied and the decision is recorded and tested. The selected behavior must distinguish a returned observation, an unavailable local result, an explicit user break, and a genuinely fatal loss of execution integrity.

Independent sample failures must be isolated, attributed, and retained in stable denominators. Retry, repair, or fallback must not silently change the scientific meaning of an output or erase the original failure when that change is material to interpretation. Existing approved hard scientific rules, technical integrity contracts, and benchmark protections must not silently regress. Heuristic scientific observations may change with named methods, rules, inputs, and versions, but those changes must be visible and explained; neither fixed historical equality nor a fixed scientific success count is a universal truth criterion.

## Architecture and implementation discipline

Complete the Plan20 architecture with one authoritative owner for every durable fact, read-only rules, small named operations, validated atomic commits, finite dependency invalidation, ordinary user-facing chemical APIs, and no duplicate implementation, compatibility implementation, second source of truth, or unapproved public abstraction. Plan12 functional requirements must be implemented through this Plan20 architecture rather than as a parallel or bypass implementation. Preserve caller-owned and unrelated work in the dirty tree.

Use small coherent implementation checkpoints. Do not describe static inspection, historical evidence, partial tests, or a different source snapshot as validation of the current bytes. Do not manufacture success or performance gains by deleting samples, changing denominators, reducing work, suppressing warnings, weakening gates, or omitting failed attempts.

## Verification, records, and completion

Execute the complete affected validation required by the current plan: focused and default tests, supported Python versions, native/ABI checks, clean wheel and sdist validation, documentation examples, scientific reports, external/GPU tiers where specified, and frozen-work performance comparisons. Include the tests and scientific validations required to demonstrate the Plan12 functional requirements within the Plan20 mainline. Record exact commands, source identities, pass/fail/skip counts, unavailable evidence, per-case scientific observations, and unresolved findings. No unexecuted tier may be described as passed.

Maintain `plan/20_fact_rule_architecture/implementation_log.md` as an append-only execution record. Each stage records its full-plan reread, current repository assessment, implementation decisions, warning/break-node changes, files and responsibilities changed, tests and scientific runs, performance evidence, failures, plan revisions, remaining gates, and commits. Keep the main-scope final report consistent with the final code, test matrix, scientific reports, benchmark records, API tree, deletion/consolidation evidence, and known limitations.

Do not mark this goal complete because any individual stage or focused test selection passes. Completion requires the full Plan20 mainline scope, the integrated Plan12 functional requirements, all applicable final acceptance criteria, the required scientific and engineering validation, complete logs and reports, and the corresponding validated Git milestones.
