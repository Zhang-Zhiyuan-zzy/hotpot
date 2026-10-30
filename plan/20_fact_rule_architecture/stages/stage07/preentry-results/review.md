ANALYSIS_RESULT
task_id=p20-state-result-consumer-gap-inventory-704283-37
revision=1
verification_code=704283
task_hash=447ea0ec46c752938309b37f11ad733758a25dd1c16a598083282bbbecaf021e
source_commit=979455ed8bada61f4d44038935e7f02c1b204c86
diff_or_tree_hash=b862f1106d8fed488038dba4112dd0118f712d29c998f398b06287b27ec0e665
capture_identity_kind=SHA256 canonical capture-manifest.json raw bytes; NOT Git tree
status=PREENTRY_GAP_INVENTORY_ONLY

## Delta coverage

This delta fills task30’s previously explicit gaps for:

- Complete backend projection/workflow contracts and bodies.
- Composition, certification, decision-snapshot serialization and validation.
- Pipeline state carriage, reduced state hashing, manifests, stage control and CLI propagation.
- Present recovery behavior.
- MCA result production, public/CLI reads and current Molecule attachment.
- CBond score/result/report propagation.
- Current partial-charge calculation, storage, identity-domain and native-packing paths.

Bodies remain index-only or absent for several dynamic/external consumers: force-field state carriage, MCA serialization internals, toolkit/import paths, dataset and benchmark consumers, installed-package users, dynamic aliases, third-party callers and some per-record xTB execution internals. The repository-wide `rg` indexes are discovery evidence, not proof of complete use or nonuse.

Task30’s audited findings remain authoritative for its original slice; they are not repeated here.

## Remaining dependency and lifetime matrix

| Area | Producer → reader/control | Current lifetime and concrete gap |
|---|---|---|
| Scientific projection | `project_state()` resolves the selected candidate path, scientific pure spin, explicit backend UHF, exact-context history and defaults, then constructs either a projection or a finding-only result (`execution_selection.py:386-790`, range SHAs `8c0c…`, `4bb6…`, `70bf…`, `248a…`; file SHA `96d76c…`). | `BackendProjectionResult` enforces that only `PROJECTED` carries a projection and that backend identities agree (`157-177`, SHA `796fe5…`). Scientific `2S` remains distinct from backend UHF. Failures do not fabricate projection values. |
| Combined workflow | `resolve_and_project_state()` resolves science once and immediately projects it (`793-833`, SHA `599c5e…`). `ScientificStateWorkflowResult` contains exactly the bundle and projection result (`180-185`, SHA `a85159…`). | This is an orchestration return, not evidence that its complete envelope is the ordinary requested public value. That public-role decision remains open. |
| Envelope consumption | Composition accepts only matching-facts, admissible results; candidates and findings drive factors and resource-truncation state (`compose.py:74-156,198-396`, SHAs `2dea…`, `d0b1…`, `2225…`, `6bf7…`; file SHA `8b05…`). Authoritative result domains create reconciliation constraints (`constraints.py:123-281`, SHAs `0ef2…`, `d110…`, `651b…`). | Identity, admissibility, candidates, findings and bounded-domain state are live inputs. Generic confidence is produced and preserved but has no supplied composition, selection, projection or pipeline-control reader. This is not a repository-wide unused finding. |
| Candidate search | Expansion rejects broken references, coupled hard-rule failures and constraint conflicts; retains rejection summaries, frontier and truncation state (`expand.py:979-1224`, SHA `39e707…`; file SHA `820ee1…`). Resolution derives unique/ambiguous/unresolved and exhaustiveness from the candidate space (`resolution.py:17-53`, SHA `fc5e88…`). | Candidate structures and frontier evidence are natural scientific/operational values, not generic quality grades. Their serialization and certification remain behaviorally required. |
| Decision snapshot | Candidate space, scientific resolution, execution decision and rule trace are serialized, reconstructed and cross-validated (`_decision_snapshot.py:57-380`, SHAs `7d7c…`, `2ea5…`, `30e4…`, `4220…`, `40c2…`, `2503…`; file SHA `979d48…`). | This is the supplied complete durable decision representation. No supplied body connects it to pipeline cold recovery. |
| Pipeline state carriage | Frozen `MolecularRecord` carries a Molecule and optional `ScientificStateBundle` (`pipeline/contracts.py:81-86`, SHA `57ebaa…`; file SHA `4741d2…`). xTB stages reuse carried state and return it with the molecule (`xtb/stage.py:254-299`, SHAs `18669e…`, `63ca3e…`). | State is preserved in memory across the demonstrated stage route. Cold SDF reconstruction of that full state is not demonstrated. |
| Pipeline identity | `payload_sha256()` includes topology/geometry plus a reduced state document: decision action/source/profile, facts identity, resolution, exhaustiveness and selected identity (`artifacts.py:38-105,230-240`, SHAs `f7f58a…`, `21901d…`, `7940ad…`; file SHA `1f53f1…`). | Candidates, findings, rule trace, backend projection and full decision snapshot are not included in this hash. Existing tests prove sensitivity to geometry and a different charged state, not every necessary state distinction (`test_artifacts.py:54-98`, SHA `e34a85…`). |
| Pipeline control | Runner passes payloads in memory, atomically persists successful stage outputs and stops on `StageExecutionError` (`runner.py:196-238,275-426`, SHAs `4a133a…`, `dee4b3…`; file SHA `a789c4…`). Intermediate `QUALITY_FAILED` continues; final `QUALITY_FAILED` fails the pipeline (`test_runner.py:197-309`, SHAs `047dc0…`, `6fa4c7…`). | This is pipeline-level control. It does not establish independent-record continuation for every stage. |
| Recovery | `decide_state_recovery()` validates policy IDs, then stops when disabled, when the default one-attempt budget is exhausted, or because no continuation exists (`state_recovery.py:1-124`, SHA `a69206…`). | It does not currently inspect state/backend inputs on any returning path and produces no retry. Exact search shows definition/tests but no supplied production caller. Recovery integration remains unproven. |
| xTB CLI containment | CLI aggregates per-index outcomes, writes successful molecular payloads, records local errors and returns nonzero if any outcome is not `ok` (`xtb/cli.py:511-610`, SHAs `a25f85…`, `bbf8a5…`; file SHA `7eab82…`). | `_process_work_items` was not supplied, so exact scheduling and containment internals remain unverified. Official pipeline evidence supplied here is success-only. |
| Hard findings versus control | Providers create hard findings and mark results inadmissible/unresolved; revalidation may retain candidates while appending findings (`net_charge.py:92-205`; electronic inference `186-334`; oxidation inference `130-301`). Composition filters those results; projection emits finding-only failures. | Finding content, availability and control are distinct in current bodies. Generic envelope confidence remains a separate produced field, not an evidenced control input. |
| Generic confidence | Net charge, oxidation, electronic and Lewis/counting producers assign `AUTHORITATIVE/HIGH/MEDIUM/UNRESOLVED` from authority or candidate presence (`net_charge.py:60-73`; electronic `170-183`; oxidation `41-54`; Lewis `251-378`). Revalidation preserves supplied confidence. | It is publicly exported and serialized by tests, but the supplied operational bodies do not consume it. Its compatibility with the current ordinary-result goal requires an owning-stage decision; this inventory authorizes neither retention nor deletion. |

## MCA production, attachment and reads

The immutable public payload is:

- `AtomPrediction(atom_index, element, mca_kj_mol)`.
- `SitePrediction(atom_index, element, site_type, mca_kj_mol)`.
- `MoleculePrediction(smiles, formal_charge, sites, model_variant, atom_predictions=())`.

These definitions and `to_dict()` behavior are complete at `result_types.py:1-43`, SHA `adc7b86…`.

`build_prediction()` copies model values into all-atom predictions and independently duplicates selected site values into site predictions (`pipeline.py:63-92`, SHA `beddda…`; file SHA `4419bd…`). CPU and GPU pipelines retain input order until every atom row is filled, then construct this payload (`cpu_backend.py:143-246`, SHAs `b70dc1…`, `fb0bd6…`; `gpu_backend.py:361-622,729-838`, SHAs `3eb3d5…`, `def12a…`, `a5bd2c…`). GPU result IDs, GPU index, batch ordinal, keys and prediction counts are checked before incorporation.

The public API requires explicit CPU/GPU selection and closes the predictor after use (`api.py:1-46`, SHA `6ad420…`). CLI readers render atom values and site membership, preserve streamed molecule order and optionally plot selected values (`mca/cli.py:132-318`, SHAs `dbc9d9…`, `e3552a…`, `0b5b50…`).

Current attachment makes three retained representations (`mca_inference.py:1-39`, SHA `cc4617…`):

1. The exact `MoleculePrediction` object is stored at `mol.properties["mca"]`.
2. Every atom receives a copied scalar `_mca`.
3. The Molecule receives `_mca_sites`, a dictionary from current Atom objects to copied site floats.

`Atom.mca` reads `_mca`; `Molecule.mca_sites` returns a dictionary copy (`core.py:1588-1596,2862-2870`, SHAs `37f76b…`, `42fb4c…`). Tests prove absent access raises, attached values match the returned prediction, returned site dictionaries are isolated, and properties retain the exact prediction reference (`test_mca_calculator.py:11-70`, SHAs `7e7786…`, `41bff4…`, `30fccd…`).

The supplied pickle path includes arbitrary Molecule state and Atom extras, so `_mca_sites`, `properties["mca"]` and Atom `_mca` are not among the excluded owned fields (`core.py:4764-4819`, SHAs `fa427f…`, `eabb86…`). No supplied test proves coherent attachment identity after pickle, copy, component extraction, merge, membership reorder or relevant source mutation. The factual-record capture itself does not explicitly carry an MCA record (`core.py:2202-2253`, SHA `4c824d…`).

Thus Stage7C must observe, without presupposing a new schema: payload identity, model/method binding, mutation currentness, reattachment replacement, copy/component/pickle behavior, all-Atom versus scalar projection cost and absence behavior. Current tests cover values and immediate access only.

## CBond value propagation

`cbond_prediction()` returns raw model output, candidate indices and existing-bond flags (`apply.py:126-138`, SHA `674302…`). `_predict_state()` maps candidate atom indices to float scores; `_step()` preserves both raw score and sigmoid probability (`237-300`, SHAs `626954…`, `944512…`).

Current public results preserve natural requested values:

- `CBondPathResult`: resulting molecule, ordered steps, path probability and donor indices.
- `CBondStructureResult`: terminal molecule, normalized probability, representative steps, donor indices, merged path count, raw path weight and log weight.

Definitions are at `apply.py:23-53`, SHAs `537f46…`, `7b49a6…`, `16983d…`. Greedy construction returns detailed results or legacy molecule/probability forms (`313-384`, SHA `14bd7e…`). Enumeration merges equal donor states in log space and returns either detailed distributions or legacy molecule/value lists (`426-569`, SHAs `9caad7…`, `f2f43b…`).

The pipeline stage copies selected score/probability/path fields into reports and carries produced molecules as records (`cbond/stage.py:52-110,165-198`, SHAs `cba2d0…`, `8d24a4…`, `a01a7c…`). No CBond result is shown attached persistently to the Molecule. These method-specific scores/distributions are not generic confidence.

Existing tests cover one public detailed call, multi-structure emission and ranks (`test_builtin_stages.py:49-128`, SHAs `ed61dc…`, `dd0307…`). Future observations still need result/report ordering and equality, mapping identity, threshold/no-result behavior, bounded enumeration failure and mixed-sample containment after ownership moves.

## Partial-charge ownership and status

Current behavior is raw Atom-table storage, not a selected typed result:

- `partial_charge` is one of the fixed 19 numeric fields, with current default `0.0` (`core.py:2530,2549-2560`, SHAs `5674f2…`, `ceb39f…`).
- Table presence and source metadata can distinguish present from absent even when numeric bytes remain (`test_atom_fact_table.py:414-432`, SHA `c5e5b9…`).
- Batch public updates reject partial charge as nonwritable (`test_interface_permissions.py:74-81`, SHA `124266…`).
- `get_partial_charge()` obtains a finite, complete tuple from an Open Babel model without mutating formal charge or Molecule charge (`core.py:1343-1413`, SHA `5a1d16…`).
- `assign_partial_charge()` calls that calculation, writes every Atom scalar and returns the tuple; the method is wrapped in `_FactTransaction` (`core.py:1415-1440,4928-4930`, SHAs `af1c6b…`, `52ff7f…`).
- Conformer partial-charge arrays are a distinct legacy frame result (`core.py:2167-2182,4436-4542`).

Field-selective native packing currently reads the raw table column directly when `PARTIAL_CHARGES` is requested and otherwise returns `None` (`_native/buffers.py:28-43,151-170,438-583`, SHAs `7fb6f9…`, `a231d7…`, `f32b89…`). It does not consult a selected-result identity or dedicated result status.

Source identity treats partial charge differently by domain:

- It is included in `bonding_inputs` and legacy force-field numeric input through `_BONDING_FIELDS`.
- It is excluded from `representation`; the supplied test changes partial charge without invalidating representation (`_identity.py:37,173-244`, SHAs `5ecbdd…`, `4ac033…`; `test_source_identity.py:99-111`, SHA `cc19ee…`).
- Presence/source metadata participates where the selected domain includes the field.

The finite `_FactStatus` states govern valence, implicit H, aromatic representation and formal-charge coverage, not partial-charge selected-result currentness (`_dependencies.py:1-197`, SHA `92fcc3…`). These are internal calculated-fact lifecycle states, not ordinary public scientific confidence.

The accepted future Stage1/Stage7C target—absent direct partial charge as `None`, fixed numeric projection as `NaN`, explicit zero preserved—does not describe current raw-field behavior. Future evidence must therefore cover the exact cutover: absent versus explicit zero, source/method identity, stale access, atomic whole-set assignment and rollback, copy/component/pickle/conformer behavior, field-selective native packing and unchanged model tensor encoding. No current field removal is authorized.

The xTB adapter can parse optional finite partial-charge arrays (`xtb/adapter.py:335-393`, SHA `76cca7…`), but the supplied bodies do not establish their eventual selected-result commit path.

## Concrete future observations

Before Stage7C/8A design ownership, rebind changed source, imports, source tests and installed-wheel selections after preceding stages. Then evidence should include:

- Source and installed-wheel public export/signature/class identity for scientific-state, MCA, projection and pipeline contracts.
- Full decision-snapshot round trip and cold pipeline persistence, including whether necessary findings, rule trace and candidate identities survive or affect payload lineage.
- Mutation/currentness tests for carried scientific state and MCA/partial-charge projections.
- MCA attachment replacement, copy/component/merge/pickle/remap behavior, scalar/all-Atom agreement and one-check batch reads without quadratic hashing.
- Partial-charge absent/present/explicit-zero/NaN projection, provenance, native packing and atomic failure rollback.
- Mixed valid/invalid records demonstrating local hard failure, finite-observation retention, configured quality continuation and stable sample denominators.
- CBond and MCA input/output cardinality, ordering, site/atom joins, early failure and unexpected-output handling through actual owning workflows.
- Scientific `2S` versus backend UHF, explicit UHF conflicts, exact-context history and no-fabricated-projection behavior.
- An owning-stage disposition for generic confidence based on complete consumers and the approved ordinary-result boundary—not limited-search absence.
- The already assigned Stage7C/8A H/aromatic diagnostic observations from the integrated requirements, without reopening Stage5 algorithms.

No tests, imports, builds, engines, models or scientific calculations ran for this inventory.

This is factual preentry gap inventory only. It is not a Stage6 dependency or gate, Stage7C/8A entry, design ownership release, public/schema/scientific approval, or implementation authorization. Root retains all decisions and maintained records. The task30 audit PASS establishes historical exact-turn transport only; this result still requires its own exact actual-turn audit and separate root key review.