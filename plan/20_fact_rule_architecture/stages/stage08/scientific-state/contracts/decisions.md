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
