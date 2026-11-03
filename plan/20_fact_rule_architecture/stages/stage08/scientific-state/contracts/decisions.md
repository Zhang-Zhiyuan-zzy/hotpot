# Stage8A — Scientific-state ownership decisions

Root accepts A1–A6 for bounded implementation at `989f27594ec08a4cd1b6f1266e565ba34cee62a1`. This is the first scientific-state migration checkpoint, not completion of Plan20 or approval of new public schemas. The effective goal, main-plan Sections6/9.2/Stage8A/16.3–4, reviewed interfaces, data-object audit and integrated Plan12 requirements govern it.

## A1 — One implementation and explicit owners

Move deterministic charge/Lewis, oxidation/counting, electronic-state, legality and concrete provider implementations under `rules/chemistry/scientific_state`. Move selection, defaults, profile eligibility, registry, controller, runtime manifest, rule-outcome validation, codecs and composition algorithms under `operations/calculation/scientific_state`. Extract the existing carrier-aware snapshot functions into `adapters/structure/scientific_state.py`. `BackendIdentity` remains its existing frozen digest dataclass; `BackendContext` remains its Protocol, both defined once in `adapters/calculation/contracts.py`.

[The exact migration](migration.json) binds leaf paths and mixed-symbol owners. Old scientific calculator submodules are removed with their connected production/test/benchmark consumers; no alias modules, alternate implementations or runtime switch are installed. `calculator` retains only its independently owned MCA entry until Stage8D. The new operation package exports the twelve scientific entry points/contracts previously accompanying MCA; no signature or field changes are authorized. Scientific leaf package initializers export contracts downward without eagerly importing inference functions; callers use the actual named function owners. Historical scientific-state pickle loading through deleted calculator paths is not a supported migration route; no such historical asset or compatibility promise was established. Current cold pickle must resolve the one new owner and retain values. Candidate/decision/trace JSON tags, fields, ordering and meaning stay unchanged.

## A2 — Lower immutable provider-input closure

Inspection found the planned mixed-contract split would force pure rules to import operations. Keep existing provider signatures/context fields and zero new types instead of adding replacement DTOs or hidden duck-typed interfaces. Define `MolecularFacts`, its identity, ComponentIdentity, HydrogenInventory, SemanticBond and SemanticBondKind once in `structure/snapshots.py`. Rules consume those values, never concrete carriers. They remain a semantic projection, not a second topology or a replacement for the existing source-identity domains.

Keep immutable quantity/evidence/provenance/legality contracts, ProviderValue, CandidateBudget and its default, and ScientificOperationRequirements in the scientific rule contracts. The latter two are evaluation inputs; operations still creates/normalizes/selects their values. Keep the existing composition data definitions in `rules/.../composition.py`: unchanged RuleContext.candidate_space and provider proposals require the factor/path/frontier contract closure. This file contains DTOs only; all composition/search/certification/resolution algorithms stay operational. Keeping the full existing contract family avoids public shape changes or a second schema.

Move StateRequest's existing frozen explicit scientific input definition to `rules/.../request.py`; its evidence imports remain downward. Keep pure RuleToggle, RuleContext, RuleOutcome and RuleTrace with provider contracts. Move StatePolicy, ProviderToggle, ProviderSet, ActivatedRule and RuleSet to operation contracts. Move the one existing `_molecular_subject_identity` formatting helper below both consumers. No rule may import operations, adapters, core or a facade that imports those; type-only imports are not a bypass. No adapter imports operations.

## A3 — Scientific behavior and facts

Preserve charge and electron accounting, scientific 2S versus backend UHF, hard-rule rejection, exact-context history, candidate alternatives/provenance, correlated constraints, budgets/frontiers/certification, and findings/control distinctions. No new guards, retries or fallback results. Snapshot extraction moves unchanged; existing implicit-H-zero/absence and public-ID/row mapping limitations remain explicitly known, with no lossless-extraction claim. Chemistry-altering corrections require their own diagnosed trigger and owning evidence.

## A4 — Scientific assets and optional activation

Move all twelve exact JSON assets to the scientific-state data owner. Retain canonical-JSON neutral-reference digests and raw-byte oxidation/rule/profile digests. Load through that package using importlib.resources; package/wheel evidence must exercise real providers and registries. `manifest.py` is executable runtime configuration and belongs to operations. The optional profile stays `planned`; no provider review, applicability corpus, held-out validation or owner activation prerequisite is waived.

## A5 — D01–D03 remain separate

Retain current five typed result records and existing finding/activation classes and fields. Existing confidence production is legacy behavior; this migration neither introduces a new ordinary-output rating nor approves it as scientific meaning. It has no identified decision reader in surveyed production consumers, while candidates, admissibility, findings, authority, evidence IDs and backend-history records have real control/persistence readers. D01's obsolete generic-envelope proposal is not implemented. D01–D03 exact replacements still require their explicit public/schema review; their predicted type reductions are not measured achievements. This checkpoint may close relocation/independence evidence only; unapproved consolidation is held at its dependent step, remains in Stage8A/Stage12 acceptance records and cannot be silently marked complete.

## A6 — Validation and rollback

Expected facts: unchanged candidate values, hard-rule outcomes, search/certification semantics and JSON schemas; one class/function definition per owner; no reverse-layer imports; unchanged resource digests and inactive profiles; current-source and installed CP311 workflows remain attributable. Allowed differences: approved ownership/import/module paths, package exports split by role, current pickle defining paths and equivalent numerical backend observations within existing method tests. Reject lost candidates/findings, fabricated projections, scientific/backend spin conflation, source mismatch acceptance, broken immutable buffers or actual workflow/sample containment.

Run the complete mapped Stage8A provider/legality/conservation/conflict/composition/search/serialization/resolution suite, changed and connected pipeline/xTB/benchmark consumers, new independence/cold-identity fixtures, and representative real CBond→FF→GFN-FF/GFN2 source/installed workflows. Preserve original failures and actual outputs, use current six unchanged native binaries with their accepted provenance, and build/install the changed Python/resource wheel outside the checkout. Other Python versions, clean all-native/ABI/SDK/sdist, optional activation, broad science/GPU/performance and D01–D03 remain explicit later gates. Run bounded snapshot/resolve/projection cost observations; acceptable explained costs do not block unrelated semantics. Root freezes runtime bytes after static integration review; no worker releases runtime or accepts gates. Rollback is the coherent Stage8A business commit, preserving predecessor evidence and unrelated user work.


## A7 — Non-mutating preparation after the diagnosed extractor failure

The current-source CP311 probe at `extractor-probe/attempt01` reproduced input
writes in all six fresh fixtures: missing/stale H was inferred or recomputed,
and even explicit-zero/parsed controls changed dependency state. The public
extractor promises non-mutation; the adapter's lazy H getter crossed into
chemistry operations. This is a diagnosed ownership failure, not a speculative
validation guard or a relocation-only claim.

Keep the public signature and existing H inference/chemistry algorithms. Define
`extract_molecular_facts` once in `operations/calculation/scientific_state/facts.py`:
use the existing lock-protected Molecule copy lifecycle, prepare H once on that
owned working copy through the existing named chemistry operation, then invoke
the private read-only adapter capture. Preparation must precede all captured
semantic bonds, H and identities so one snapshot cannot mix pre/post-preparation
facts. The input's factual bytes, presence/source/status and dirty seeds must
remain unchanged on success and the existing conflict failure. No durable copy,
cache, schema, new type, retry, fallback or alternate inference is added.

The adapter capture reads prepared stored H presence/status/counts without any
lazy chemistry getter or operation import. Preserve the existing conflict error
and zero/absence behavior (B-H-PRESENCE), public-ID/row limitation, hard rules and
codec schemas; those separate scientific corrections are not silently included.
Move only the existing Molecule import ceiling slot to the operation owner;
Bond/BondKind stay at the adapter. Update connected imports and identity checks.

The copy cost is measured with extraction/resolve/projection before disposition;
reuse existing lifecycle code rather than implement a second copy path. Earlier
relocation AST evidence and focused-attempt01 remain historical evidence for
their actual bytes. Re-freeze after this bounded correction and rerun affected
source/installed and actual workflows. D01–D03, optional activation and all
later/global gates remain open. Root owns integration, acceptance and Git.


## A8 — Preserve current implicit-H zero in the scientific snapshot

At delivered `98e6d2bc762ab142c04dbe86395ff235ccc063af`, root reviewed
P20-D020/B-H-PRESENCE, the actual six-fixture observations, the sole adapter
capture and its concrete chemical/provider/identity consumers. A present CURRENT
H count of zero is a known value. Returning None discards that fact and can
collapse known-zero and unresolved-H snapshot identities. Root authorizes the
small correction now; this supersedes only A3/A7's temporary zero-to-None
preservation. H inference and preparation algorithms remain unchanged.

Return the stored integer in the existing present/CURRENT adapter branch,
including explicit/imported/inferred zero. Non-current or absent H remains None;
CONFLICT retains its existing exact RuntimeError. A7 still copies the carrier,
prepares H once and captures one coherent prepared working state without
modifying input. Do not infer H explicitness from a numeric zero, add types,
guards, codecs, fallback, history migration or a compatibility switch. No public
field, signature, permission, JSON tag or optional rule activation changes.

The expected facts are known zero versus absence/unresolved, unchanged raw input
on all success/failure fixtures, independent chloride charge/electron/spin and
existing Lewis/donor/oxidation/conservation behavior. All surveyed arithmetic
readers currently group None/zero or add zero; correcting presence alone does
not change those readers or claim complete scientific interpretation of unknown
H. Existing public-ID/row extraction assumptions remain qualified.

Affected fact/candidate/path/history identities intentionally change when their
previous H0 was encoded as null. Existing identity checks must reject stale
provided evidence; historical None cannot be losslessly upgraded without the
carrier. Digest-based ordering and budget truncation may select different
alternatives while preserving the bounded-search contract. Do not force old
digests or selected-path equality. Reuse unchanged A1–A7 ownership/native/cost
evidence; re-freeze after this correction and run new presence/identity/stale
regressions, the affected CP311 scientific/controller/xTB suite and actual
source/installed pipeline and MCA workflows. Preserve actual backend differences
under existing method-specific tests. No new performance comparison is needed
for this one existing scalar return; full-copy cost remains P20-D019. Root owns
runtime release, acceptance, Git and any additional diagnosed change. D01–D03,
whole Stage8A and later/global gates remain open.

## A9 — Accepted public calculation reports and confidence-only removal

**Root accepts this concrete design on 2026-11-03 at baseline
`7952d1f14994e88892dd094cd775e92970c6c0e1`, before product authoring, under the
previously granted full-authority mandate and effective goal.** Root reviewed
actual owners/consumers, the exact constructors/exports, codecs and pickle
consequences in the bounded local review. This is an explicit public-contract
disposition, not an inference from docstrings or a new approval request. It
supersedes A5's D01–D03 design hold only within the scope below. Implementation,
source/installed evidence and root runtime release remain pending; whole
Stage8A and global gates remain open.

For paths below, `R` is `hotpot/cheminfo/rules/chemistry/scientific_state` and
`O` is `hotpot/cheminfo/operations/calculation/scientific_state`. These are
local documentation abbreviations, not Python aliases or proposed owners.

### D01 — Retain five typed calculation records; remove generic confidence

Retain each existing frozen class, defining module/qualname, candidate element
type and package export. Every constructor has the following **six required
fields in this exact order**, with no new defaults:

```python
NetChargeResult(identity: str, facts_identity: MolecularFactsIdentity, candidates: Tuple[NetChargeCandidateEvidence, ...], admissibility: ScientificAdmissibility, resolution: ResolutionStatus, findings: Tuple[ScientificFinding, ...])
LewisFormalChargeResult(identity: str, facts_identity: MolecularFactsIdentity, candidates: Tuple[LewisFormalChargeCandidate, ...], admissibility: ScientificAdmissibility, resolution: ResolutionStatus, findings: Tuple[ScientificFinding, ...])
CoordinationElectronCountingResult(identity: str, facts_identity: MolecularFactsIdentity, candidates: Tuple[CoordinationCountingCandidate, ...], admissibility: ScientificAdmissibility, resolution: ResolutionStatus, findings: Tuple[ScientificFinding, ...])
OxidationStateInferenceResult(identity: str, facts_identity: MolecularFactsIdentity, candidates: Tuple[OxidationStateCandidateEvidence, ...], admissibility: ScientificAdmissibility, resolution: ResolutionStatus, findings: Tuple[ScientificFinding, ...])
ElectronicStateInferenceResult(identity: str, facts_identity: MolecularFactsIdentity, candidates: Tuple[ElectronicStateCandidate, ...], admissibility: ScientificAdmissibility, resolution: ResolutionStatus, findings: Tuple[ScientificFinding, ...])
```

| Retained class | Defining owner | Exact candidate element type |
|---|---|---|
| `NetChargeResult` | `R/charge/contracts.py` | `NetChargeCandidateEvidence` |
| `LewisFormalChargeResult` | `R/charge/contracts.py` | `LewisFormalChargeCandidate` |
| `CoordinationElectronCountingResult` | `R/oxidation_state/contracts.py` | `CoordinationCountingCandidate` |
| `OxidationStateInferenceResult` | `R/oxidation_state/contracts.py` | `OxidationStateCandidateEvidence` |
| `ElectronicStateInferenceResult` | `R/electronic_state/contracts.py` | `ElectronicStateCandidate` |

Before A9 implementation each constructor has the same fields plus required
`confidence: ChemicalConfidence` immediately before `findings`. Delete only
that field, all confidence production/copy arguments and the three `_confidence`
helpers. Findings moves from positional argument seven to six; old seven-field
construction, `.confidence`, generated repr/equality/hash, field introspection,
`asdict` and positional pattern shape intentionally change. Keep candidate
values/order/correlation, result/candidate identity payloads, hard legality,
source checks, resolution and precise findings unchanged. Admissibility is the
existing hard-rule calculation/control assessment, not an ordinary-value rating.

Delete `R/contracts.py::ChemicalConfidence`, including
AUTHORITATIVE/HIGH/MEDIUM/LOW/UNRESOLVED, and its import/export in
`R/__init__.py`. Both modules' ordered exports become exactly:

```python
['ScientificQuantity', 'ScientificAdmissibility', 'ResolutionStatus',
 'CoverageStatus', 'StateValueAuthority', 'ScientificFinding',
 'CandidateProvenance', 'ProviderValue', 'CandidateBudget',
 'DEFAULT_CANDIDATE_BUDGET', 'ScientificOperationRequirements']
```

Do not create `StateInferenceResult`, replacement ratings, quantity tags,
subclasses, compatibility arguments/classes/aliases or a second implementation.
Candidate provenance, authority, evidence/assumption IDs, coverage and existing
backend history remain at their single current owners.

### Exact retained public report purpose and signatures

Root explicitly chooses these existing APIs as requested detailed calculation,
decision and backend-projection reports. They remain public:

- `resolve_scientific_state` returns the exact existing `ScientificStateBundle`:
  `facts, net_charge_results, lewis_results, coordination_counting_results,
  oxidation_state_results, electronic_state_results, candidate_space,
  scientific_resolution, execution_decision, rule_trace`, in current field order.
  The five ordered provider-output collections support explicit inspection;
  candidate space, resolution, decision, trace and facts retain calculation and
  source binding. Their retention does not claim each collection is read by the
  xTB projector.
- `project_state` returns the exact existing `BackendProjectionResult(status,
  backend_identity, request, projection, findings)`. Its domain-specific control
  distinguishes available projection from precise unavailable findings.
- `resolve_and_project_state` returns the exact existing
  `ScientificStateWorkflowResult(scientific_state, backend_projection)`.

The first report/function is defined in `O/state_resolver.py`; the latter two
reports/functions in `O/execution_selection.py`. Preserve all existing fields,
defaults, annotations and behavior, except the five nested constructor changes
above. These APIs add no wrapper, status, admissibility or confidence rating to
ordinary Molecule/Atom charge, spin or structure values. Keep all eight reviewed
callable signatures exactly:

```python
# R/charge/net_charge.py
resolve_net_charge(facts: MolecularFacts, request: StateRequest = StateRequest(), *, provided: Tuple[NetChargeResult, ...] = ()) -> Tuple[NetChargeResult, ...]
# R/charge/lewis.py
solve_main_group_lewis(facts: MolecularFacts, net_charge: NetChargeResult, *, candidate_budget: CandidateBudget = DEFAULT_CANDIDATE_BUDGET) -> Tuple[LewisFormalChargeResult, ...]
# R/oxidation_state/coordination_counting.py
infer_coordination_electron_counting(facts: MolecularFacts, *, lewis_results: Tuple[LewisFormalChargeResult, ...] = (), conventions: Tuple[CountingConvention, ...] = ()) -> Tuple[CoordinationElectronCountingResult, ...]
# R/oxidation_state/inference.py
infer_oxidation_states(facts: MolecularFacts, request: StateRequest = StateRequest(), *, counting_results: Tuple[CoordinationElectronCountingResult, ...] = (), provided: Tuple[OxidationStateInferenceResult, ...] = ()) -> Tuple[OxidationStateInferenceResult, ...]
# R/electronic_state/inference.py
infer_electronic_states(facts: MolecularFacts, request: StateRequest = StateRequest(), *, net_charge_results: Tuple[NetChargeResult, ...], oxidation_state_results: Tuple[OxidationStateInferenceResult, ...] = (), provided: Tuple[ElectronicStateInferenceResult, ...] = ()) -> Tuple[ElectronicStateInferenceResult, ...]
# O/state_resolver.py
resolve_scientific_state(facts: MolecularFacts, request: StateRequest = StateRequest(), *, policy: StatePolicy = StatePolicy.STRICT, requirements: Optional[ScientificOperationRequirements] = None, provider_set: ProviderSet = ProviderSet(), provided_results: ProvidedScientificResults = ProvidedScientificResults(), rule_set: RuleSet = RuleSet(), candidate_budget: CandidateBudget = DEFAULT_CANDIDATE_BUDGET) -> ScientificStateBundle
# O/execution_selection.py
project_state(state: ScientificStateBundle, backend: BackendContext, *, request: BackendStateRequest = BackendStateRequest(), execution_policies: ExecutionPolicySet = ExecutionPolicySet(), execution_history: Tuple[BackendExecutionEvidence, ...] = ()) -> BackendProjectionResult
resolve_and_project_state(facts: MolecularFacts, backend: BackendContext, request: StateRequest = StateRequest(), *, policy: StatePolicy = StatePolicy.STRICT, requirements: Optional[ScientificOperationRequirements] = None, provider_set: ProviderSet = ProviderSet(), provided_results: ProvidedScientificResults = ProvidedScientificResults(), rule_set: RuleSet = RuleSet(), candidate_budget: CandidateBudget = DEFAULT_CANDIDATE_BUDGET, backend_request: BackendStateRequest = BackendStateRequest(), execution_policies: ExecutionPolicySet = ExecutionPolicySet(), execution_history: Tuple[BackendExecutionEvidence, ...] = ()) -> ScientificStateWorkflowResult
```

The operation facade's twelve ordered exports stay `extract_molecular_facts,
resolve_scientific_state, project_state, resolve_and_project_state,
MolecularFacts, StateRequest, StatePolicy, ScientificOperationRequirements,
ScientificStateBundle, BackendStateRequest, BackendProjectionResult,
ScientificStateWorkflowResult`. Extraction remains defined once in
`O/facts.py`; MolecularFacts and its composed snapshots remain in structure.
Pure rule/provider inputs retain A2's downward closure; no rule imports
operations/adapters/carriers, and no adapter imports operations.

### D02 and D03 — NO_CONSOLIDATION

D02 retains all four existing definitions, exports, class identities and codecs:

```python
# R/contracts.py
ScientificFinding(code: str, message: str, related_identities: Tuple[str, ...] = ())
# O/execution_selection.py
BackendProjectionFinding(code: str, message: str, related_identities: Tuple[str, ...] = ())
# O/state_selection.py
DecisionReason(code: str, message: str, related_identities: Tuple[str, ...] = ())
StaticEligibilityFinding(code: str, message: str)
```

Static-eligibility findings keep exactly `{code, message}` in the decision
snapshot; decision reasons keep `{code, message, related_identities}`. Keep
existing domain code namespaces and carrier roles without new severity/role
labels or a common `StateFinding` class.

D03 retains `O/contracts.py::ProviderToggle(provider_id: str, enabled: bool,
configuration: Tuple[Tuple[str, ProviderValue], ...] = ())` and
`R/providers/contracts.py::RuleToggle(rule_id: str, enabled: bool,
configuration: Tuple[Tuple[str, RuleValue], ...] = ())`, their owners and
existing `RuleValue = ProviderValue` reuse. Current built-in providers admit
only PROVIDER-001/002/003, reject unknown/duplicate IDs and nonempty
configuration, and run only enabled explicit toggles. Rules resolve registered
specifications, versions/dependencies/phases and pass configuration to action;
disabled/abstained outcomes and STRICT optional-rule prohibition remain.

Preserve `O/profile_eligibility.py::_effective_profile_identity`'s exact payload:
`facts_identity` is the chemical digest; `policy` its enum value;
`profile_artifact_digest` the coordination-screening profile digest only under
COORDINATION_SCREENING_V1, otherwise null; `requirements` contains operation_id
and separately sorted required/allowed_unresolved quantity-value tuples;
`providers` is the sorted tuple of `(provider_id, enabled, configuration)`;
`rules` is the layer-ordered tuple of `(rule_id, rule_version, enabled,
configuration)`; `candidate_budget` is the existing three-limit tuple in field
order. Keep these separate keys, ordering and hashing, not a unified activation
namespace or `StateActivation` type. The optional profile remains `planned`.

The historical 5→1/4→1/2→1 estimates remain historical proposals only. Accepted
counts are 5→5/4→4/2→2: **actual and prospective dataclass delta zero**. Enum
retirement is a separate **prospective enum-class delta −1**, pending product
evidence; it supplies no dataclass reduction or count-led design requirement.
D02/D03 retention is resolved as NO_CONSOLIDATION, not unimplemented consolidation.

### Persistence, bounded cutover and remaining evidence

Root accepts a coherent hard cut for pre-change seven-field Python object
pickles, including graphs referencing the retired enum at its current rule
owner. No alias, migration loader or dual runtime is permitted; no preservation
promise is made for external/reflective callers or old Python assets. New
six-field values must cold-load at their retained current owners. A1's earlier
calculator-path retirement remains separately scoped.

Known explicit JSON schemas/digests are unchanged: candidate variants,
`scientific-state-candidate-space-v1`, `scientific-state-decision-snapshot-v2`,
`scientific-state-candidate-budget-v1`, `scientific-state-frontier-v1`,
`hotpot-scientific-state-profile-v1`, trace/source bindings and selected
pipeline/xTB/benchmark persistence fields. Confidence is not encoded by those
codecs. No wire-version bump, identity rehash or history upgrade is authorized.

Implement as one coherent cutover across eleven product files: R contracts
and initializer, the three quantity-contract files, charge/net_charge.py,
charge/lewis.py, oxidation_state/inference.py,
oxidation_state/coordination_counting.py, electronic_state/inference.py, and
O/state_resolver.py. Update the nine connected constructor/export/ownership
test consumers identified by the reviewed inventory. D02/D03 need no replacement
product code. Root reviews normative changes before product authoring and owns
integration, source freeze, runtime release, evidence and Git.

A9 runtime/tests/build/science are **NOT_EXECUTED** at this decision checkpoint.
Require the affected Stage8A source and installed CP311 selection, exports and
six-field constructors, provided-result revalidation, current cold pickle,
hard legality/conservation/conflict, bounded search/truncation, unchanged
codecs/digests, pipeline/xTB and actual pipeline/MCA delivery checks. Retain
independent methane charge0/electron10/2S0/multiplicity1, parity-negative and
known-H0 chloride charge−1/electron18 fixtures. Reject changed candidate values,
ordering, decisions or explicit JSON under this confidence-only migration;
existing numerical method/context tolerances still govern actual backends.
A7 non-mutation and A8 H0/None/stale-evidence semantics remain required.

Rollback restores the connected constructors, producers, exports, tests and
records together; no alternate runtime is retained. Historical checkpoint
passes qualify only their frozen bytes. Optional activation, unknown-H
interpretation, public-ID mapping corrections, Stage8B–E, other versions,
clean-native/ABI/SDK/sdist, broader science/GPU/performance and Plan20/Plan12
final acceptance remain under their existing open gates. No speedup is claimed.


### A9 runtime fulfilment — 2026-11-03

Root accepts [the confidence-only CP311 checkpoint](../validation/confidence_checkpoint.json)
after exact eleven-product/nine-test integration:443focused,990affected+5skipped
per source/installed boundary and3actual pipeline/3MCA checks each. Allfive current
types cold-load at both boundaries; five enum-bearing historical graphs fail as
approved. Five constructor fingerprints rename and one enum case retires; no
chemical fixture is removed. Dataclass delta0/enum delta−1 is measured. The
release-time NOT_EXECUTED statements above retain their historical scope.
WholeStage8A, unknown-H accounting disposition, optional activation and all
later/global obligations remain open; this accepts no Stage8B entry.
