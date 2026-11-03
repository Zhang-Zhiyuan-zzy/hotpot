# Stage 20 — Fact-Carrier and Rule Architecture

Status: implementation authorized on 2026-10-28; Stage0–6 owning-stage gates complete. Stage7A's external-object, adapter and Crystal extraction and applicable CP311 gates are now [complete](stages/stage07/external-adapters/completion_20261103.json), with bounded SDF resource/cost acceptance and explicit later obligations. Stage7B is [complete within its applicable CP311 owning scope](stages/stage07/completion_20261103.json): sole-owner constraints and complete Bond topology with operation-local selections have source/installed and actual-workflow evidence, with adverse usable costs retained and accepted. Stage7C is in progress: its [CP311 MCA owner/currentness checkpoint](stages/stage07/selected-results/validation/report.md) is root accepted; the complete selected partial-charge cutover is next and whole-stage acceptance remains open. Known H/isotope/stereo work remains assigned to Stage7E; clean-native, version adaptation, scientific/global efficiency and full-plan acceptance remain open. The owner-approved [goal.md](goal.md) is effective; the previous version is retained as [goal_v1.md](goal_v1.md).

All public chemical types follow a common protocol for fact ownership, interface mapping, permissions, dependencies, buffer ownership and lifecycle. Atom's unified fact table is one concrete implementation. MetalCluster runtime remains Plan21 scope. Existing historical checkpoints below retain their own source and evidence bounds.

Implementation branch: `implement/plan20-fact-rule-architecture`

Stage3 checkpoints: [foundation evidence](stages/stage03/foundation/evidence.json),
[architecture coverage correction](stages/stage03/architecture/coverage_correction.json),
and [private storage review](stages/stage03/storage/review.md) with
[verification evidence](stages/stage03/storage/evidence.json). The production
[table cutover review](stages/stage03/table-cutover/review.md) now records the connected
Molecule-owned table, factual imports, controlled writes and complete rollback.
Remaining architecture debt is281. The [Stage3 report](stages/stage03/report.md)
records its completed owning-stage gates and explicit later global acceptance limits.

The current checkpoint adds [member/source identity and field provenance](stages/stage03/identity/review.md),
with [execution evidence](stages/stage03/identity/evidence.json). The approved
complete numeric property is `atom_matrix`, directly renamed from the existing
entry; its implementation and callers remain singular.
Closed Atom update candidates and table serialization are now verified in the
[preparation review](stages/stage03/atom-update/review.md) and its
[execution evidence](stages/stage03/atom-update/evidence.json). Source validation:
Python3.10–3.14 each273 passed; default4096 passed,123 skipped,3 xfailed.
These are historical preparation results; the production checkpoint below
supersedes their implementation status.
Atom's existing parent is now read-only; detached inspection and current-version
pickle/deepcopy are covered by [parent evidence](stages/stage03/parent/evidence.json).
The [complete import-record design](stages/stage03/builder/review.md) now has verified
immutable codecs and [execution evidence](stages/stage03/builder/evidence.json):
Python3.10–3.14 each394 passed; default4217 passed. This is record preparation;
the actual builder and table were connected in the next production checkpoint.
The frozen Python-engine checkpoint passes4349 default tests, with123 skipped,
3 xfailed and49 subtests. Its466 focused contract tests also pass. Source
hashes and commands are retained with the implementation log. The
[performance escalation record](stages/stage03/performance/escalation.json)
retains measured scalar/batch regressions after both Sol-max and bounded Astra
optimization. These are historical checkpoint results. Final build8 [validation](stages/stage03/validation/report.md)
passes4432 default tests, with123 skipped,3 xfailed and49 subtests;
each Python3.10–3.14 passes1023 source/75 debug-native/941 installed tests
(8 external-xTB skips in source/installed). [Storage/scaling evidence](stages/stage03/performance/report.md)
retains current regressions and memory increases. Stage3 is complete; final
scientific/global efficiency acceptance remains open at its owning later gates.
Original planning branch: `refactor/fact-rule-architecture`

Original planning baseline: `0a4b0ee`; reviewed documentation: `d9eb80e`.
Stage1 must record a fresh executable/performance baseline at the selected
Plan17/Plan20 boundary before production refactoring.

## Completed Stage4 owning gates

The [Stage4 report](stages/stage04/report.md) binds the extracted mathematics,
source-bound views, constraint ownership and preserved pair observations.
After bounded local optimization, default4585 tests pass; each Python3.10–3.14
passes2287 source/175 debug/2157 installed tests, with current package audits.
The [focused performance report](stages/stage04/performance/report.md) retains
original and follow-up samples: graph/view2.054×B20, numerical screening1.006×,
Crystal0.994×; the complete graph family remains1.437×. No efficiency exception
is approved and final performance acceptance remains blocked.

The [owning-stage gate record](stages/stage04/gates/review.md) closes Stage4's
actual technical and valid focused-comparison obligations after its bounded
work completed. This follows the main plan's existing gate scope; it changes no
final threshold or workload. Stage5 starts with a fresh full normative read and
project-state assessment. Stage12 global acceptance and the later separately
branched Stage13 experiment retain their original gates.

## Completed Stage5 owning gates

The [independent implementation checkpoint](stages/stage05/review.md) retains
477 passed/5 pending-public-H exclusions/3 strict xfails and seven bounded
B20 assignment observations as historical intermediate evidence. The final
[report](stages/stage05/report.md) supersedes its pending owning-gate status. The owner clarified that explicit H counts are
retained and unspecified counts must be inferred from valence rules. The bounded
[Sol-max scientific/default design](stages/stage05/hydrogen-defaults/decisions.md)
is complete, with a final304-pass focused handoff and accepted bounded static
review. Rebuilt native evidence is retained. The first final five-version source matrix
failed on the same74 nodes; [bounded corrections](stages/stage05/source-matrix-corrections/report.md)
now cover those original failures locally. Current Python3.11 default passes4945 tests. The final
[technical matrix and package report](stages/stage05/validation/report.md)
records each Python3.10–3.14 passing2969source/175debug/2837installed tests and
the548-member sdist. [Focused performance](stages/stage05/performance/report.md)
retains all11raw measurements and the corrected comparability disposition. Aggregate review corrects three clean-read output mismatches:8tasks remain
comparable(four regressions,three improvements,one parity),3are non-comparable,
and the complete lazy-family/overall aggregate is not accepted. The
[owning gate decision](stages/stage05/gates/review.md) closes Stage5technical
obligations;final science/performance acceptance remains open. The earlier optional
scalar proposal is superseded. The [read contract](stages/stage05/contracts/decisions.md)
retains historical placeholder display while strict consumers check internal
readiness; no new public return type or H-to-NaN export is adopted.

[Checkpoint bookkeeping](stages/stage05/bookkeeping/review.md) reconciles the
current256-entry architecture allowlist and actual776/353/8/2 class/dataclass/
protocol/TypedDict census. The source ratchet and document-layout audit pass.
[Validation preparation](stages/stage05/validation-preparation/review.md)
records the immutable pre-correction snapshot, failed helper notification and
three local static findings assigned for bounded reproduction/correction.
No current full-suite, matrix, remote or final performance acceptance is
inferred from this preparation.

## Completed Stage6 owning gates

Root's separate [final closure](../../.plan20-artifacts/20261030/stage6/root-final-closure/attempt1/evidence.json)
closes Stage6 after C62 `aeb72d0b97fb1c2e50df86a057c31585a2e34e0c` delivery and
remote verification. Validated runtime source remains C61. The
[Stage7A entry](stages/stage07/external-adapters/entry_review.json) now releases
bounded design; the separate [root design acceptance](stages/stage07/external-adapters/root_acceptance.json) and [implementation release](stages/stage07/external-adapters/implementation_release.json) now govern exact authoring scope. Candidate runtime remains held. Earlier integration reports
and check hashes retain their historical state and bind the later closure separately.

Root's [final evidence decision](../../.plan20-artifacts/20261030/stage6/root-gate-decision/attempt1/evidence.json)
accepts frozen C61 static/runtime/bounded-cost evidence and ends the runtime
freeze. The [exact Task38 result/provenance](stages/stage06/review-result/evidence.json)
preserve the historical conclusion, matching actual audit and separate root
acceptance. [Validation](stages/stage06/validation/report.md) and
[exact evidence](stages/stage06/validation/evidence.json) record focused65;
each Python3.10–3.14 source1,397/installed1,396/native-source6 passes plus2
missing SDK skips; default4,998passed/123skipped/3strictxfailed plus49passedsubtests,
accepted through strict artifact-only accounting. All15 supplementary commands
pass with source/wheel preservation/count/retention observations.

[Bounded performance](stages/stage06/performance/report.md) and
[dispositions](stages/stage06/performance/evidence.json) retain90 original
comparisons/all72 adverse metrics accepted for Stage6 only. Fixed60-pair
installedchain512 repeat classification follows the frozen noise rule; original
inconclusive observation/broad interval/order effects remain. No global parity
or waiver. The [owning gate record](stages/stage06/gates/review.md) and
[exact root/delivery evidence](stages/stage06/gates/evidence.json) preserve the
original integration snapshot and separately bind the completed closure. Stage7A
entry follows the post-Stage6 source/import/test/wheel rebind; its bounded design
is root accepted with separate exact implementation/test authoring releases. Mandatory Stage7E
removes transitional whole-payload stereo storage before Stage8/global ownership;
Stage10 clean-native/ABI/SDK/sdist and final science/performance/Plan12/experimental
prerequisites remain open. The [Stage6 index](stages/stage06/README.md) links all
accepted decisions/historical preparation/current evidence.

## Records

- [Current implementation constitution](goal.md)
- [Integrated Plan12 requirements and remaining evidence](integrated_requirements.json)
- [Architecture audit and implementation plan](fact_rule_architecture_implementation_plan.md)
- [Implementation log and stage-entry assessments](implementation_log.md)
- [Unified deferred issues: severity, responsible stages and closure evidence](deferred_issues.md)
- [Stage2 execution evidence](stages/stage02/evidence.json)
- [Stage3 entry assessment](stages/stage03/foundation/entry_review.json)
- [Stage1 API and consumer review](stages/stage01/api/review.md)
- [Stage1 concrete interface decisions](stages/stage01/contracts/decisions.md)
- [Stage1 common interface semantic review](stages/stage01/interfaces/review.md)
- [Stage1 assembly asset review](stages/stage01/assets/review.md)
- [Chemical-type public-interface protocol and fact storage](chemical_type_interface_contract.md)
- [Plan17/Plan20 execution alignment](plan17_plan20_execution_alignment.md)
- [Data-object classification and consolidation audit](data_object_scope_audit.md)
- [Plan21 MetalCluster and multi-attachment superatom draft](../21_metal_cluster/metal_cluster_implementation_plan.md)

The protocol covers carriers, independent values, collections, views,
scientific results and transport contracts. Roles determine mutability and
lifecycle; no common base class or physical storage layout is required.
MetalCluster is recorded as a registered aggregate related to Molecule,
Atom and Bond, with `Molecule.metal_clusters -> list[MetalCluster]` planned.
Plan20 registers that structure only; Plan21 defines concrete cluster kinds,
matching and superatom execution after the owner supplies their forms.

Stage0 selects Plan20 first: the minimum Plan17 MCA native closure is not yet
implemented, so B20/F20 use the current sole Python-RDKit MCA algorithm. The
17-MCA batch follows F20 with separate B17/F17 evidence. Scientific outcomes
are versioned monitoring reports; 187/182
does not impose a fixed success count. Compare equivalent semantic work with the pre-refactor baseline; accept explained,
usable cost under the effective goal and retain all adverse observations.

Every stage and independently gated sub-stage begins with a full reread of the
plan and normative companions plus a recorded project-state assessment.
Executed work has an append-only implementation log and a final report.
Planning records and proposed commands are not execution or benchmark evidence.

Stage1 measurement policies live in `performance_manifest.json`,
`stages/stage01/sampling/sampling_plan.json` and `python_matrix_manifest.json`. The supported
Python matrix uses `run_python_matrix_baseline.py` and
`compare_python_matrix_baselines.py`; its per-version comparisons remain
separate from the original performance-family weights. The implementation log
binds executed reports and records incomplete or failed attempts explicitly.

## Stage records and root assets

Use the [stage catalog](stages/README.md) to select a stage and topic, then read
its entry review, design/decisions and execution evidence. Each stage README
indexes every record. Markdown filenames have the finite meanings defined in
the catalog; repeated topics use separate directories rather than numbered
copies or overwritten reviews.

The main plan and its four normative companions keep their existing homes.
The Plan21 companion remains in its own project. Cross-stage policies and
ledgers also remain here: the verification map, performance/Python manifests,
architecture allowlist, reviewed interface table, data-object budget,
implementation log, baseline inventory and shared execution/audit tools.
Stage1's sampling record is filed with Stage1; it does not replace the global
performance manifest or change frozen measurement policy.

Historical commands, evidence paths, input names and digests retain the exact
text recorded at execution. The [path migration index](stages/path_migration.json)
maps former root filenames to current records and preserves their pre-move
identities. It does not claim that a relocated or link-adjusted document was
revalidated by its old command. Run the [layout audit](audit_stage_records.py)
for current filename, index, JSON and local-link checks.

Stage7A's [contract proposal](stages/stage07/external-adapters/contracts/review.md),
[decision register](stages/stage07/external-adapters/contracts/decisions.md) and
[exact proposal evidence](stages/stage07/external-adapters/contracts/evidence.json)
record accepted P01–P07 design boundaries; P07 future probe bytes still require
root release. Reviews5–8 bind qualified C61 cost/retention, injected-failure,
five-version reflection and cell-cold observations without candidate acceptance.
Candidate validation, comparable cost and native-memory sufficiency remain open.

The [connected-consumer supplement](stages/stage07/external-adapters/connected_consumer_supplement.json) adds the dataset parser path-hash owner migration and all36 test_ds files: final source/installed affected selection is211 on each Python3.10–3.14. The original175-file proposal/baseline remains preserved; no candidate pass or Stage13 experiment follows.
