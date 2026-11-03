# Review of the Plan20 goal revision

**Status: review only; not activated.** [goal_v2.md](goal_v2.md) is a complete proposed replacement for the effective [goal.md](goal.md). This revision contains the English draft, these review notes, and two comparison artifacts. It changes no public API, schema, runtime behavior, stage status, or acceptance result. Approval must precede activation and synchronization of conflicting records.

The draft follows the effective goal's opening objective and four main headings. The appendix holds current business priorities; implementation mapping stays in the [main plan](fact_rule_architecture_implementation_plan.md). The [interactive tracked changes](goal_v2_tracked_changes.html) and [word-level diff](goal_v2_word_diff.diff) are regenerated comparison artifacts against the unchanged effective goal.

## Placement and requirement coverage

The owner directions below are incorporated requirements. The owner has also accepted S1–S4 as supporting practices; their acceptance does not activate the replacement goal.

| Basis | Requirement retained or revised | Location in the draft |
|---|---|---|
| Original scope | Complete Plan20 and integrated Plan12 through one mainline, including applicable implementation, validation, reports, and Git delivery; Stage0–12 acceptance and `B_exp` precede Stage13A/B and `D_exp`/`I_exp`. | Opening objective; completion section |
| Original accountability | The main agent owns technical and scientific decisions, aligns continuously with current plans, amendments, companions, repository state and evidence, and records bounded route corrections. | Main-agent ownership and alignment |
| Owner point 1 | Chemical business facts, core representation, semantics, and real scientific workflows take priority over peripheral tooling. | Architecture section; appendix |
| Owner point 2 | Judge chemical meaning and justified invariants without forced historical identity or fixed success counts; hashes attribute artifacts and transport; numerical/stochastic comparisons use declared context. | Verification section |
| Owner point 3 | Separate validator failure from business failure; use simple meaningful checks and trustworthy direct/alternative evidence; bypass does not create a pass. | Verification section |
| Owner point 4 | Map execution units to actual business, select affected checks, and reuse valid evidence without making inventory counts or a universal tool chain the goal. | Verification section |
| Owner point 5 | Priorities 1–4 are mandatory, 5–6 best effort, and other independent issues may defer with reporting; dependencies needed for mandatory paths inherit that priority. | Architecture section; appendix |
| Owner point 6 | Semantics > acceptable performance > version adaptation; Python 3.11/CP311 first, other planned versions later; explained usable degradation may be accepted by the main agent within authorization. The approximately fivefold example sets no fixed threshold or extra per-case approval requirement. | Architecture section; completion section |
| Owner point 7 | Flexible tools pursue “Not Error” within stated domains, preserve researchers' interpretation and context-dependent modeling, and scope hard safeguards. Retain the original prose, bold normative statements, numbered examples, and RDKit partial-charge example. | Scientific rigor and high-throughput stability |
| Existing contracts | Preserve one fact owner, read-only rules, atomic commit/rollback, finite invalidation, currentness, ordinary requested-value returns, versioned method support, findings/control separation, local failures, and stable denominators. | Scientific and architecture sections |
| Accepted delivery direction | Small validated commits serve coherent business progress and enable prompt rollback after business errors; related documentation, tests, and evidence stay with the checkpoint. Avoid consecutive documentation-only commits without business progress and commits solely for hashes, packets, bookkeeping, or status. Preserve unrelated work and honest evidence. Completion distinctions and deferred scope remain explicit. | Verification, records, and completion |

## Clarifications and accepted supporting practices

The attribution wording permits declared numerical and stochastic variation while preserving the known-input/ordinary-return boundary and internal provenance ownership. The scientific section follows the effective goal’s connected prose, bold normative statements, and blockquoted numbered examples, including its original RDKit partial-charge example. Context-dependent modeling, “Not Error,” and scoped safeguards qualify interpretation without approving a new API or scientific algorithm.

The owner has accepted S1–S4 as supporting practices. The goal states them briefly; the owning implementation plan and validation reports select cases, methods, criteria, and evidence. These refine existing business-validation obligations rather than establish separate businesses or gates.

| Practice | Accepted support and boundary |
|---|---|
| S1 | Independently judged positive, negative, and boundary chemical fixtures, with equivalence, round-trip, or semantics-preserving metamorphic checks where useful. Historical output or two sides sharing one implementation cannot be the sole oracle; no fixed corpus size or unique chemical interpretation is imposed. |
| S2 | Representative actual backend/model cases mix ordinary successes and local failures, with relevant attribution, currentness, rollback, and finite-recovery checks. |
| S3 | Method/state/unit tolerances, convergence, geometric equivalence, and actual seed or distribution comparisons appropriate to each business slice. No new numeric thresholds are set here. |
| S4 | A representative installed CP311 workflow outside the checkout with actual native/backend/model dependency attribution. Other Python versions do not become prerequisites for each increment. |

## Conflicting records to synchronize after approval

These records remain untouched in this authoring step. Synchronization must preserve the complete approved scope, scientific and public-interface boundaries, and Stage0–12 → `B_exp` → Stage13A/B ordering.

| Record | Consequential synchronization |
|---|---|
| Main plan §12.2, §14 item 10, and inherited direction in §16.1 | Replace parity and rare-small-regression hard gates with explained usability, comparable work, resource bounds, and main-agent accountability. Retain unfavorable historical measurements and actual performance obligations. |
| Main plan §11, §14 item 9, and related Stage6/7E/10 version matrices | Express CP311-first execution, later adaptation, and honest support status. Staged or deferred checks remain visible obligations; they do not become final acceptance waivers. |
| Main plan §10/§13 and affected stage/verification records | Align stage-entry assessment, affected-check selection, reusable evidence, and logging with the revised goal without a blanket reread or tool-chain prerequisite. |
| Main plan §13–14/§17.2 and final reports | Reconcile applicable acceptance wording without silently restoring superseded performance/version gates, deleting scope, or allowing an experiment to close an earlier gate. |
| chemical_type_interface_contract.md §6; data_object_scope_audit.md §9 item 5; plan17_plan20_execution_alignment.md §5 | Reconcile inherited parity/faster wording while retaining fact ownership, interface review, independent baselines, and approved algorithm boundaries. |

Document-integrity checks for this compression and regenerated comparisons are recorded below. They do not establish chemical correctness, activate the draft, or change any business evidence. No chemistry calculation, business validator, test suite, build, benchmark, dependency/environment change, Git mutation, or remote operation is part of this authoring step.

S1–S4 are accepted supporting practices. Owner review of the revised draft and explicit approval remain necessary before goal replacement or synchronization; this revision performs neither. Existing implementation failures, missing scientific evidence, untested versions, and incomplete delivery remain in their actual states.

## Compression of the review draft — 2026-11-03

The complete precompression draft and comparison artifacts are preserved under `/tmp/hotpot_plan20/long/goal-v2-compression-20261103/before/`. Counts use whitespace-delimited tokens in the complete Markdown source, including headings, examples, and the appendix: **2,946 → 2,555 words; 391 removed (13.27%)**. This is a modest editorial compression of the review draft, not a reduction of approved work or an activation of it.

The table identifies what was shortened and where every grouped requirement remains. “Original lines” refer to the preserved precompression `goal_v2.md`.

| Original paragraphs / requirement groups | How compressed | Final location |
|---|---|---|
| Status and objective (lines1–3) | Shorten approval wording and combine the single mainline, global objective, complete delivery, and experiment-order statements; retain Plan12 and `B_exp`/`D_exp`/`I_exp`. | Status/opening objective |
| Accountability, revision limits, continuous alignment (7–11) | Tighten repeated route/evidence wording; keep accountability, all approval/scientific boundaries, trained-feature meaning, predecessor review, and recorded correction before continuation. No blanket in-full reread is introduced. | [Ownership/alignment](goal_v2.md#main-agent-ownership-and-alignment), all three paragraphs |
| Context, ordinary results, support, confidence (15–23) | Shorten explanatory prose around unchanged bold scientific statements; retain “Not Error,” research choices, context/variation, `None`, internal provenance, unavailable-evidence placeholders, and calibrated-uncertainty conditions. | [Scientific rigor](goal_v2.md#scientific-rigor-and-high-throughput-stability), paragraphs1–5 |
| Findings, warning/break policy, behavior table, local batch failure (33,49,61–67) | Combine repeated explanation of finding/control separation, recorded decisions, failures/attempts/denominators, and four result/control states. Keep codes, required diagnostic content, break scopes, last finite observations, and no relabeling failure as success. Hard/non-hard gate paragraphs remain unchanged. | Scientific rigor: finding paragraph, control paragraph, behavior table and closing paragraphs |
| Core priorities and dependent integrations (71;101–113) | Replace duplicate priority/dependency explanation with a pointer to the unchanged complete appendix. | [Architecture](goal_v2.md#architecture-and-implementation-discipline), opening paragraph; [priority appendix](goal_v2.md#appendix--current-plan20-business-priorities) |
| Durable ownership, immutable outputs, migration, checkpoints, unrelated work (73;79;95) | Tighten the ownership/lifecycle list; place repeated checkpoint/work-preservation duties with Git delivery. Retain one owner, isolated nested buffers, atomic rollback, currentness, scalar/batch access, all connected consumers, Plan12, and the Plan20/21 boundary. | Architecture, ownership paragraph; [verification/delivery](goal_v2.md#verification-records-and-completion), commit paragraph |
| Performance/version ordering and measurement protections (75–79;9) | Shorten prose and reference the common failure/denominator/finding/rule protections once instead of repeating their list in performance text. Keep CP311 first, later planned versions, explained usable cost, every resource/reliability condition, and no numeric/parity/rare-small-regression gate. Backend attribution stays explicit. | Architecture, final three paragraphs; ownership, revision-limits paragraph |
| Chemical oracles and numerical/stochastic judgments (83–85) | Merge overlapping historical-output/hash caveats and shorten the method/criteria list. Retain independent positive/negative/boundary fixtures, useful equivalence checks, exact deterministic/lossless contracts, transformation-specific allowed changes, actual seeds, and no after-the-fact threshold relaxation. | Verification, first two paragraphs |
| Actual business, validator separation, affected checks, required delivery tiers (87–91) | Tighten overlapping execution/evidence wording while retaining actual mixed backend/model outcomes, simple validators, limited checker deferral, trustworthy alternatives, mapping-based affected checks, reusable evidence, staged final obligations, and installed CP311 dependency attribution. | Verification, business/validator, mapping, and milestone paragraphs |
| Append-only records, Git discipline, deferral and completion (93–97) | Consolidate repeated evidence/failure/ownership and unfinished-work statements. Keep every log field, small validated business commits/normal pushes, prompt rollback, no consecutive documentation-only commits, four completion levels, deferral status/impact/owner/conditions, scope exceptions, and full final acceptance. | Verification, final three paragraphs |
| Six numbered scientific examples; bold scientific statements; priority table | **Unchanged in full.** The RDKit floating-point example remains verbatim, including `HIGH`, `LOW`, `ADMISSIBLE`, and fabricated probability. No example is replaced by a shorter label. | Scientific rigor, Examples1–6; priority appendix |

S1–S4 remain accepted supporting practices, briefly expressed in the goal and detailed in the existing review table above. No obligation, permission exception, scientific boundary, required acceptance tier, or deferred scope has been intentionally removed.

## Model and effort references

A case-insensitive search of the current main plan and the three Hotpot governance/stage/validation skills found no agent model/effort selection policy or table. Scientific/backend/model-workflow references describe business capabilities rather than agent selection. The current user-supplied `AGENTS.md` instructions govern agent model and reasoning-effort choices; this review adds no second model policy.

## Document verification

The effective `goal.md` remains byte-for-byte unchanged and matches its recorded comparison baseline, Git `92aa921b0cf4bd06fc871d3bd7e9e929f28eea8a`. The [tracked HTML](goal_v2_tracked_changes.html) and [Git word diff](goal_v2_word_diff.diff) compare that complete effective document with the final compressed draft, not with a partial excerpt or the precompression draft. Independent HTML parsing reconstructed both complete UTF-8 documents exactly; the plain comparison matches Git's `--no-index --word-diff=plain` output for the final files. Checks passed for all 19 local links/fragments, six complete examples, bold statements, four main headings, the complete appendix, review-only status, and counts. Exact results are retained in `/tmp/hotpot_plan20/long/goal-v2-compression-20261103/document_checks.json`; no browser visual-inspection result is claimed.
