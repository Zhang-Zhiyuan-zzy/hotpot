# Cross-stage deferred issues

Current disposition index for Plan20 and its integrated Plan12 requirements.
Established and initially reviewed on 2026-11-03. The main agent maintains it
alongside coherent business changes; stage records retain detailed observations,
commands and historical decisions. An entry records a known compromise or a
specific observation awaiting assessment, not an invented failure or a pass.

## Use and scheduling

- Assign the next unused `P20-Dnnn` when accepting a deferral. IDs never change
  or get reused; retain closed rows. Reference legacy report/issue IDs rather
  than renumbering their evidence. Related observations may share one entry only
  when responsibility and closure conditions are shared; otherwise split them
  under new IDs with reciprocal references.
- Record severity, category, current state, business impact, responsible stage,
  latest treatment checkpoint, evidence and closure conditions. The main agent
  is accountable; a transient agent assignment does not replace stage ownership.
  Update the row and append the decision to `implementation_log.md` when scope,
  priority, state, responsibility or acceptance changes.
- Consolidated optimization follows the Stage0–12 implementation/migration
  passes—thirteen numbered main stages—within Stage12 final hardening, before
  main-scope acceptance and `B_exp`. Cross-stage fixes are expected. This does
  not postpone a dependency needed earlier or waive a delivery/final gate.
  Stage13A/B remains the separately gated experiment after `B_exp`, not a place
  to hide unfinished main-scope acceptance.
- Revisit affected entries during related business edits and milestone review;
  do not run the entire backlog at each stage. A newly demonstrated core
  semantic/integrity failure, unbounded growth, OOM or inability to complete a
  representative mandatory workflow brings the affected item forward.
- Routine unfinished core implementation remains in the main plan and
  verification map. In particular, selected partial charges, coordinate frames
  and Stage7E H/isotope/stereo canonical ownership are not optional optimization
  debt. Stage7E still gates Stage8. Existing architecture-debt IDs and ratchets
  remain in their specialized inventory; this file indexes cross-stage
  dispositions without duplicating that inventory.

## Severity and state

Severity describes impact; state and the treatment checkpoint determine when an
item is handled. A dependency used by mandatory business inherits that priority.
Missing evidence is not proof of a defective implementation.

| Severity | Meaning and scheduling consequence |
| --- | --- |
| S0 — blocking | Demonstrated loss of core chemical semantics, attribution, ownership, transaction integrity or representative-batch reliability. Resolve or isolate the affected path before dependent work or capability claims; do not defer it merely for speed. |
| S1 — required | An applicable mandatory workflow, support or delivery obligation remains unsatisfied/unverified. May follow CP311 core work when independent, but must meet its owning dependency/delivery/final gate. |
| S2 — bounded | Usable behavior with measured adverse cost, a bounded limitation or an unresolved observation. Consolidate later unless new evidence promotes its impact. |
| S3 — peripheral | Independent third-party, optional tooling or simplification work without a demonstrated core effect. Treat in the consolidated window; any final exception needs the existing scope authority. |

States are `OPEN` (not yet dispositioned), `DEFERRED` (accepted schedule),
`IN_PROGRESS`, `RESOLVED` (candidate treatment awaiting evidence review),
`CLOSED` (closure conditions met), and `ACCEPTED_LIMITATION` (explicit final
scope decision). The last state requires the decision and supporting evidence;
it is not a pass and cannot waive a mandatory gate. A scoped stage acceptance
is not automatically final closure. Reopen the same ID when its conditions cease
to hold. Initial rows below are all open obligations, not final exceptions.

## Issue index

`—` in the final column means **no closure evidence yet**. Links lead to the
smallest maintained evidence entry; raw measurements stay at the linked owning
archive. Historic results apply to their recorded source/workloads, not every
later implementation.

| ID | Severity / category | State | Issue and currently accepted boundary | Responsible stage; treatment checkpoint | Evidence and closure condition | Closure evidence |
| --- | --- | --- | --- | --- | --- | --- |
| P20-D001 | S2 / performance | DEFERRED | Stage3–5 storage, graph/view and lazy-fact costs include regressions and incomplete/non-comparable aggregates. Historical local completion supplies no global efficiency pass. | Stage12, coordinating Stage3–5 owners; consolidated hardening. | [Storage](stages/stage03/performance/report.md), [views](stages/stage04/performance/report.md), [lazy facts](stages/stage05/performance/report.md). Close with comparable final family/workflow timing, scaling and resource evidence, and explicit usable-cost dispositions for adverse or inconclusive results. | — |
| P20-D002 | S2 / performance | DEFERRED | Stage6 stereo capture/packing has locally accepted necessary overhead; some repeat intervals remain inconclusive. No global parity claim. | Stage12 with Stage6 owner; consolidated hardening. | [Stereo report](stages/stage06/performance/report.md) and [dispositions](stages/stage06/performance/evidence.json). Close with final representative integration/cost/resource evidence or an attributable bounded final disposition; retain original adverse/inconclusive samples. | — |
| P20-D003 | S2 / performance | DEFERRED | Complete SDF writes remain usable with historical GM ratio 2.527527, six adverse intervals and twelve higher Python peaks. Local mandatory parity repair was disposed. | Stage12 with Stage7A conversion owner; consolidated hardening. | [SDF decision](stages/stage07/external-adapters/validation/sdf_resources_20261103.json), `usable_cost`. Close with final comparable complete-work evidence and usable-cost disposition; optimize when useful without reducing frames, records or chemical work. | — |
| P20-D004 | S2 / resource attribution | DEFERRED | SDF trace retains a 3,264-byte batch10 dictresize step of unresolved ownership and unresolved native frames. Bounded sufficiency was accepted; neither a leak nor universal leak freedom was established. | Stage12 with Stage7A allocation owners; final resource review, or earlier on the recorded revalidation triggers. | [SDF decision](stages/stage07/external-adapters/validation/sdf_resources_20261103.json), `native_memory` / `revalidation_triggers`. Close through applicable representative resource evidence and an explicit bounded disposition; exact attribution of this small step is not automatically required. Growth/OOM/incomplete work requires renewed investigation. | — |
| P20-D005 | S2 / performance | DEFERRED | Constraint projections retain 6.276–19.854× selective-axis and 2.651–2.741× Bond-export costs in their historical fixtures. Topology work retains fourteen adverse timing medians, ten adverse peaks and sampled superlinear growth. Local costs were accepted. | Stage12 with Stage7B owners; consolidated scaling/workflow review. | [Constraints](stages/stage07/constraints/validation/report.md), [topology](stages/stage07/topology-selection/validation/report.md), [Stage7B acceptance](stages/stage07/completion_20261103.json). Close with final integrated/scaling/resource evidence and disposition of remaining usable costs; no universal speed-parity requirement. | — |
| P20-D006 | S1 / version adaptation | DEFERRED | CP311 leads. Historical Python3.10/3.12/3.13/3.14 results do not qualify every current owner; current adaptation/evidence remains incomplete. No defect in those versions is asserted. | Stage10 delivery and Stage12 final acceptance; before corresponding support claims/final gate. | [Stage7B remaining obligations](stages/stage07/completion_20261103.json), [current MCA limits](stages/stage07/selected-results/validation/report.md), [verification map](stage_verification_map.json). Close with applicable actual current-source/installed/native results for declared versions; unexecuted tiers cannot pass. | — |
| P20-D007 | S1 / native delivery | DEFERRED | Current CP311 MCA uses one freshly rebuilt fact-native plus five reused binaries and shared system-site dependencies. Clean all-native, ABI/SDK and applicable clean wheel/sdist/platform qualification remain open. | Stage10; applicable native/package delivery gate, then Stage12 final acceptance. | [MCA build/installed scope](stages/stage07/selected-results/validation/checkpoint.json), [Stage7B delivery limits](stages/stage07/completion_20261103.json). Close with actual supported clean artifacts, native/ABI/SDK checks and representative installed workflows outside the checkout; a private candidate cannot establish delivery. | — |
| P20-D008 | S1 / package contents | DEFERRED | Forty-one measured non-Python package omissions remain. Tested installed paths passed; that does not prove omitted assets irrelevant to other workflows. | Stage10 delivery, with Stage9/11 asset owners as applicable; before a dependent installed workflow/support claim or final acceptance. | [Current installed receipt index](stages/stage07/selected-results/validation/evidence_index.json), [checkpoint](stages/stage07/selected-results/validation/checkpoint.json), `installed`. Classify all omissions; package required contents and validate consumers, or cite an authorized scope disposition for optional contents. Preserve the original omission list. | — |
| P20-D009 | S3 / optional backend optimization | DEFERRED | The private Open Babel GTD optimization has bounded evidence but no repository-maintained ordinary installation path. It is not a shipped speedup. | Stage10 with Stage7A backend owner; final delivery/performance disposition. | [Business review](business_validation_review_20261103.md), §§1/5. Close by validating the actually maintained/shipped implementation, or explicitly leave the optional optimization unshipped and measure the real selected baseline. Mandatory workflow reliability still applies. | — |
| P20-D010 | S3 / simplification | DEFERRED | Nine statically identified redundant-check candidates await owning-business cleanup. They are not nine demonstrated failures or measured performance bottlenecks. | Stage12, coordinating affected owners; opportunistically during related edits or consolidated cleanup. | [Defensive-code audit](defensive_programming_audit_20261103.md), A1–A9. Dispose each candidate from its actual call contract; remove only unjustified duplication without changing model features, necessary scientific gates, ownership or rollback. No replacement guard framework. | — |
| P20-D011 | S2 / core-adjacent semantic assessment | DEFERRED | Whole-xTB JSON finite checks, null/sentinel report encoding, optional-frame selection and Gaussian missing-frame zero filling need business-level assessment. B1–B4 are not collectively proven defects. | Stage12 review, with Stage7D frame / Stage8 workflow owners; assess earlier when those connected consumers change or actual core failures appear. | [Defensive-code audit](defensive_programming_audit_20261103.md), B1–B4. Close with actual relevant backend/log/frame outcomes and per-item disposition of unavailable vs observed values and failure cause. Confirmed core semantic defects must be promoted and fixed before dependent acceptance, not left until final cleanup. | — |
| P20-D012 | S3 / independent integrations | DEFERRED | Gaussian parse-cause merging, thermo missing/error conflation, dataset search/parse fallbacks, undefined correlation replacement, fitness sentinels and visualization gaps need bounded assessment. Their actual harm is not established by static patterns. | Stage12 consolidated review, coordinating independent integration owners; earlier only if a mandatory path depends on them. | [Defensive-code audit](defensive_programming_audit_20261103.md), B5–B10. Record each existing behavior as retained documented domain logic, corrected actual defect or authorized optional limitation, using necessary business evidence. Do not invent guards for hypothetical failures. | — |
| P20-D013 | S2 / diagnosed symptom | OPEN | MCA cache fallback lost the original cause of a historical OSError/cache-path assertion. ENOSPC was not established; later successful inference does not diagnose that cause. | Stage7C MCA/cache owner; assess during remaining connected work, latest Stage12 review. | [Defensive-code audit](defensive_programming_audit_20261103.md), existing-failure row; historical log entries referenced there. Preserve/identify the original cause where available and validate the bounded cache/temporary-copy behavior without weakening model-content integrity; record any unrecoverable historical cause honestly. | — |
| P20-D014 | S3 / validation tooling | DEFERRED | The installed driver's immediate receipt postcheck raced final writes after successful pytest. Same-attempt reconciliation supplied trustworthy business evidence. Other historical comparator/tool-policy limitations remain separate from business correctness. | Stage12 validation owner; simplify on next necessary tool change or consolidated review. | [Current installed reconciliation](stages/stage07/selected-results/validation/checkpoint.json), `installed.resume_disposition` / `original_failure`; [performance policy](fact_rule_architecture_implementation_plan.md#122-comparable-work-usable-cost-and-resource-bounds). Close with a simpler reliable check or documented retirement/replacement and trustworthy actual-business evidence; do not rerun passed unchanged work merely to satisfy the broken tool. | — |
| P20-D015 | S3 / record-layout tooling | DEFERRED | The current read-only stage-layout audit exits1 with21 findings on16 already committed files: catalog/navigation drift plus false link parsing of inline chemistry and `:line` references. This is not a chemical execution failure or a passed layout audit. | Stage12 record/validation owner; next necessary navigation cleanup or consolidated hardening. | [Recorded audit outcome](implementation_log.md), 2026-11-03 layout-checker entry; `audit_stage_records.py`. Close by resolving actual navigation omissions and simplifying/replacing inaccurate parsing, with valid chemical examples and file-line links retained. New index links/JSON have separate passing checks. | — |

## Closure and evidence

Before treatment, state expected business facts, allowed differences, rejection
conditions and the smallest affected checks. Use method/state/unit/context-aware
numerical tolerances and stochastic criteria where needed. Hashes identify
artifacts and unchanged transport; matching historical hashes or exact numerical
outputs is not the chemical success criterion.

`RESOLVED` records the candidate change and evidence entry. `CLOSED` requires the
main agent to review the applicable closure condition and fill the final column
with an evidence-index/report link, relevant commands/outcomes and delivered
source/artifact identity. A coherent business commit may contain that closure;
record its resulting Git identity in the next normal log/update rather than a
hash-only commit. Existing useful evidence may be reused with its scope intact.
No fixed success count, blanket checker rewrite or new exhaustive evidence
pipeline is required by this index.

A final `ACCEPTED_LIMITATION` entry also cites the explicit decision, retained
impact, support/scope boundary and revalidation trigger. Cross-stage optimization
may close an explained, usable cost without claiming parity or a speedup.
Unverified behavior remains unverified. Relevant S0/S1 obligations and existing
mandatory gates cannot be cleared by a backlog status change; applicable final
exceptions retain their existing owner-approval rules.
