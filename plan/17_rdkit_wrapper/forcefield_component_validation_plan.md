# Plan 17 force-field component validation plan

Status: owner-approved for implementation and validation execution; provider
cutover remains subject to the completed evidence and a separate owner decision

Document date: 2026-10-23

Repository validation baseline: `8b556dd` on
`perf/mca-resource-efficient-workflow`

Implementation baseline retained for provenance:
`9640d906a9d13e969152d0f3a4d92a3a16b9e0ce`

Normative implementation sources:

- `rdkit_wrapper_implementation_plan.md`;
- `implementation_log.md`; and
- `implementation_report.md`.

This document re-evaluates the remaining Plan-17 validation work after the
owner clarified the scientific acceptance contract. It does not revise the
production force-field workflow or itself change the current default provider.
The owner has authorized its implementation and benchmark execution. It
supersedes only the obsolete historical-artifact prerequisites and force-field
acceptance interpretation in the earlier plan/report.

## 1. Owner corrections and validation question

### 1.1 The historical output is not a reference answer

The structures produced at `9640d906` passed the geometry gate and manual
inspection. They are examples of acceptable outcomes, not coordinate, energy,
or conformer ground truth. A new run is not required to reproduce:

- historical coordinates or conformer identity;
- historical force-field energies;
- a byte-identical manifest, trajectory, or report;
- the same stochastic intermediate frames; or
- the same final structure as a separately repeated old-code run.

The correct question is:

> Does replacing only the initial ligand-coordinate provider with native
> ETKDG still allow the unchanged Hotpot force-field workflow to return a
> physically acceptable optimized structure for every eligible complex?

A fresh `O0` route remains useful as a same-revision control, but neither `O0`
nor `9640d906` is an oracle for the `F1` coordinates.

### 1.2 Correct numerical acceptance semantics

The hard numerical requirement applies to the structure selected and returned
by the workflow, not to every exploratory frame.

Definitions used by this plan:

- **exploratory frame**: a perturbation, topology-recovery state, rejected
  ligand candidate, preliminary route, or other frame that is not selected as
  the returned result;
- **selected optimal structure**: the selected trajectory frame committed to
  the output molecule and described by the final `ForceFieldRunReport`;
- **gradient unavailable**: the force field cannot provide a finite RMS or
  maximum gradient, so no valid optimization direction can be obtained; and
- **standard recovery**: a recovery action already established by the current
  Hotpot workflow, such as perturbing excessively close coordinates or moving
  a force-field-specific forbidden angle away from 180 degrees; and
- **recovered numerical excursion**: an exploratory energy, explosion, or
  unavailable-gradient event after which the standard workflow successfully
  re-establishes a finite gradient and continues optimization.

The acceptance policy is therefore:

1. the selected optimal structure must have finite coordinates;
2. its `best_energy` must be finite;
3. its selected-frame RMS and maximum gradients must be finite;
4. a non-finite gradient is a hard failure only when it prevents force-field
   optimization from continuing and the established standard workflow has no
   successful recovery for that state;
5. the selected structure must not itself be marked exploded;
6. the selected structure must pass the existing `standard` geometry gate;
   and
7. intermediate high, infinite, or exploded energies do not independently
   fail a case when the workflow rejects/rolls back that frame and ultimately
   returns a structure satisfying items 1-6.

There is deliberately no absolute energy ceiling. Energies from different
molecules are not directly comparable, and a large finite value is not by
itself proof of an unusable geometry. Intermediate `nonfinite_energy` or
`explosion_detected` observations remain stage-resolved diagnostics rather
than automatic case failures. A `nonfinite_gradients` observation is likewise
not judged from its label alone. The validator must determine whether an
existing recovery action was applied, whether a later finite gradient was
obtained, and whether force-field optimization actually resumed. A recovery
that was attempted but did not restore optimization still fails. The
benchmark must not invent a one-off rescue that is absent from the standard
workflow merely to convert a failed case into a pass.

The current `standard` acceptance level is retained. It treats convergence as
a warning rather than an error, which matches the owner contract: strict
convergence is not required when the selected structure is geometrically
acceptable and numerically usable.

The implementation already maps `ForceFieldRunReport.best_energy` into the
selected-frame acceptance evidence, so its finite-energy check is aligned with
this contract. It also checks the selected RMS/maximum gradients and selected
explosion flag. That report alone does not inspect discarded frames, and the
native failure classifier checks non-finite energy before non-finite gradient.
The new recovery audit in A06 is therefore required to detect an unavailable
gradient even when another terminal label would otherwise mask it, and then to
classify whether the standard workflow recovered and continued. A normal
function return is also insufficient evidence because some workflows retain a
quality-rejected frame for inspection; the runner must explicitly require the
quality report to pass.

### 1.3 Scope exclusion

xTB, DFT, and other QM input generation, execution, convergence, and geometry
refinement are outside this validation round. They provide no hard or soft
acceptance criterion here and are not prerequisites for a provider decision.

## 2. Cohorts and route semantics

### 2.1 The 187-ligand source

The immutable source identity is the 187 physical SMILES lines with SHA-256
`e67ae3167a1a96d3c6a4c6df3ce12b70abfaf479bdd89bb156bfaefa771795a7`.
The exact text is recoverable from Git commit `11c0e3d` at
`molecules/extractant/extractants.smi` and is also embedded case-by-case in the
completed R0 evidence. The currently absent managed-dataset file is therefore
not an unavailable scientific asset.

All 187 ligands have two roles:

1. standalone native ETKDG component coverage (`R0`); and
2. input to one current-code CBond reconstruction that determines complex
   eligibility.

The five no-CBond ligands remain valid R0 cases. They are not rdWrapper or
force-field failures and do not enter the complex route.

### 2.2 The 182-complex semantic cohort

The complex cohort will be reconstructed from the canonical 187 source with
the current code, the recorded Am suite, and the standard settings:

- first CBond threshold: `-0.5`;
- subsequent CBond threshold: `-0.125`;
- base seed: `20260921` plus the one-based case index; and
- the unchanged standard force-field budgets recorded by
  `BenchmarkSettings`.

The expected partition is 182 CBond-positive and five no-CBond cases. The
known no-CBond indices are 139, 182, 185, 186, and 187. A different partition
is reported as an upstream CBond/cohort discrepancy and investigated; it is
not silently redefined to keep a favorable denominator.

CBond is executed once per ligand for cohort construction. For each positive
case the validator records coordinate-free semantics:

- case index and source SMILES;
- metal and donor indices;
- atom identity, formal charge, and stable row order;
- bond endpoints, order, kind, aromaticity, and dative direction;
- represented stereo payload;
- seed and scientific settings; and
- a canonical semantic graph digest.

The same reconstructed graph semantics feed `O0`, `F1`, `F2`, `C1`, and `C2`.
No historical coordinates are supplied to any route. A normalized semantic
manifest and the native graph JSON needed by C1/C2 are outputs of this
reconstruction. They are current-run reproducibility evidence, not missing
historical prerequisites or accepted structures.

### 2.3 Route matrix

| Route | Population | Coordinate proposal | Downstream workflow | Validation role |
| --- | ---: | --- | --- | --- |
| `R0` | 187 ligands | Native ETKDG, counts 1/8/32 | None | Standalone component contract |
| `O0` | 182 complexes | Current Open Babel adapter | Complete current complex workflow | Same-revision engineering control |
| `F1` | 182 complexes | Native RDKit adapter, one conformer | Exactly the same complete workflow as `O0` | Force-field business acceptance |
| `F2` | 182 complexes | Precomputed/frozen ETKDG ligand coordinates supplied by a replay provider | The same O0 gate -> topology -> optimization flow | Separate ETKDG proposal quality from in-workflow adapter execution |
| `C1` | 182 complexes | Stock whole-complex ETKDG | Existing complete-complex gate/topology checks followed by force-field optimization | Required whole-complex ETKDG experiment |
| `C2` | 182 complexes | Coordination-bounded whole-complex ETKDG | The same post-generation force-field path as `C1` | Required coordination-aware ETKDG experiment |

`F1` and `F2` deliberately share the same downstream scientific policy but
exercise different integration boundaries. `F1` invokes
`RDKitConformerAdapter` inside the ligand-build worker. `F2` first generates
and freezes ETKDG coordinates outside that workflow, then a data-only replay
provider supplies those exact coordinates to the established O0
gate/topology/optimization flow. F2 is a controlled validation route, not a
second ETKDG algorithm or a duplicated production workflow. If F2 succeeds
where F1 fails, the difference localizes the problem to provider integration;
if both fail on the same case, ETKDG proposal/downstream compatibility is the
leading explanation.

`C1` and `C2` remain original Plan-17 whole-complex hypotheses, but their
coordinate-only smoke tests are no longer sufficient. Both full 182-case
experiments are mandatory in this validation round and must continue into
force-field optimization. Their ETKDG whole-complex coordinates must remain
the coordinates entering that force-field experiment; rerunning ligand
construction or metal placement in a way that discards them would make the
experiment vacuous. C1/C2 use the same final acceptance contract as O0/F1/F2,
although their earlier stage trace is necessarily different.

## 3. Final force-field acceptance contract

### 3.1 Per-case automated hard gates

Every one of the 182 eligible cases in each evaluated O0/F1/F2/C1/C2 route is
judged by the rows below. F1 is eligible to replace O0 as the complex default
only if all 182 F1 cases pass; there is no aggregate-rate substitution for
that requirement.

| ID | Hard gate | Evidence |
| --- | --- | --- |
| A01 | The route returns its required mapped coordinate proposal | Provider/generator status, shape, atom-row count, finite coordinates |
| A02 | The force-field setup succeeds | `setup_succeeded` and absence of setup exception |
| A03 | The selected output coordinates are finite | Selected trajectory frame and committed molecule |
| A04 | The selected optimal energy is finite | `optimization.best_energy` |
| A05 | The selected RMS and maximum gradients are finite | `optimization.rms_gradient` and `optimization.max_gradient` |
| A06 | Every unavailable-gradient event is successfully recovered by the established standard workflow, or the case fails | Frame evidence, named standard recovery action, later finite gradient, and proof that optimization resumed |
| A07 | The selected frame is not exploded | Selected-frame evidence and `optimization.exploded == false` |
| A08 | Existing `standard` acceptance passes | `quality_report.passed == true` and named checks |
| A09 | Source graph semantics are preserved | Atom order/identity, formal charge, topology, dative direction, and represented stereo comparison |
| A10 | Stage failures and recoveries are explicit | Provider, ligand, placement, restoration, and optimization stage records; no hidden provider fallback |

The following are recorded but are not independent hard failures:

- lack of strict convergence when the standard gate passes;
- finite but large energy or gradient magnitude;
- intermediate or rejected-frame energy explosion;
- a non-finite-gradient frame followed by successful established recovery and
  continued optimization;
- a recovered preliminary-route failure;
- coordinate or energy differences from `O0` or `9640d906`; and
- exact trajectory or selected-conformer hash differences across separate
  complete force-field runs.

R0 fixed-seed determinism remains an independent standalone component
contract. Complete force-field stochastic reproducibility is reported as
outcome consistency, not byte-identical coordinates.

### 3.2 Geometry and topology

The validator reuses the existing `standard` terminal report. It does not add
a new geometry gate after ETKDG and does not repeat an expensive ring scan
already represented by the workflow's final native checkpoint.

The terminal gate must retain its current checks for, as applicable:

- finite selected coordinates;
- overlaps and minimum atom-pair distance;
- explicit covalent and metal-ligand bond geometry;
- bond-ring piercing and undetermined ring relations;
- coordination-environment facts; and
- original atom and bond topology.

Topology equality is semantic rather than serialized-text equality. SDF/MOL2
visualization loss of dative representation must not be mistaken for a
production graph mutation; the in-memory/trajectory topology is authoritative.

### 3.3 Manual physical-plausibility review

The 182 final structures from every completed O0/F1/F2/C1/C2 route are rendered
in exhaustive, consistently styled contact-sheet/gallery sets. Manual review
is a veto check, not an attempt to identify a unique correct conformer.

The reviewer reports only structures that are visibly nonphysical, including
obvious interpenetration, disconnection, extreme bond distortion, collapsed
ligands, or implausible placement that escaped the automated gate. The review
artifact records:

- validation revision and route/run identity;
- gallery/render hashes;
- reviewer and completion date; and
- zero or more flagged case indices with reasons.

A completed review with no issue report means no visually nonphysical output
was found. Mere absence of a review is not evidence.

### 3.4 Aggregate decision

The default-provider decision is reviewable when:

- the cohort reconstruction reports the expected 187 = 182 + 5 partition;
- all 182 F1 cases satisfy A01-A10;
- the completed F1 manual review reports no visibly nonphysical case;
- O0/F1/F2 stage and resource comparisons are available;
- the required C1/C2 generation-plus-force-field experiments and their
  separate acceptance results are available; and
- focused and affected regression tests remain green.

If F1 achieves fewer than 182/182 passes, native `rdWrappers`, its implemented
interfaces, and the conformer-provider adapter/options are retained, but F1
does not replace O0 as the default complex route. Open Babel remains the
complex default unless a later owner decision approves a different route from
complete evidence. C1/C2 failure does not retroactively invalidate an F1
182/182 result; it means those whole-complex hypotheses are rejected under the
same acceptance standard.

Here, retained adapter/options means the existing standalone `rdWrappers` API
and private injectable provider contract. It does not add a public
backend/profile selector or a second public complex-workflow interface.

No minimum speedup is invented in this plan. Performance and resource use are
reported for the decision but do not override a correctness failure.

## 4. Reassessment of original implementation objectives

Status vocabulary:

- **Implemented**: the production/component code exists at the validation
  baseline;
- **Partially implemented**: the core exists but required benchmark or release
  support is missing;
- **Validated**: direct evidence covers the stated scope;
- **Partially validated**: focused evidence exists but not the complete stated
  population or claim; and
- **Not validated**: no result supports the stated claim yet.

| ID | Original objective | Implementation | Validation | Current reason/evidence | This round | Missing prerequisite or action |
| --- | --- | --- | --- | --- | --- | --- |
| G01 | Pin one Hotpot release to RDKit 2026.09.1 and update lazily | Implemented | Validated | Metadata, native identity, lock files, and wheel installs agree | No rerun unless code changes | None |
| G02 | Lock and build the exact offline RDKit dependency closure | Implemented | Validated | Six archives, clean native build, symbol/dependency audit | No rerun unless native/build files change | None |
| G03 | Expose standalone Python `rdWrappers` | Implemented | Validated | API, native fixture, R0, and wheel tests pass | Retain regression only | None |
| G04 | Expose independently usable C ABI and C++17 facade | Implemented | Validated | C11 plus ABI-0/ABI-1 C++17 consumers pass | Retain regression only | None |
| G05 | Avoid Python RDKit molecule translation in native sampling/force-field use | Implemented | Validated | Native graph path, dependency and symbol audits | Retain source-fence test | None |
| G06 | Generalize the Python/C++ molecule transport and allow field subsets | Implemented | Validated | Common-buffer/view and consumer-policy tests pass | Retain regression only | None |
| G07 | Preserve the public Open Babel component and behavior | Implemented | Validated | Open Babel API retained; parity and mixed-invalid tests pass | O0 provides new population evidence | None |
| G08 | Keep standalone sampling coordinate-only and nonmutating | Implemented | Validated | R0 and focused tests prove no Hotpot gate/FF call and no input mutation | Complete | None |
| G09 | Execute standalone R0 on all 187 ligands at 1/8/32 | Implemented | Validated | 559 success, two deterministic partial, zero failed/unsupported | No scientific rerun required | Existing R0 stereo payload is empty; see G17 |
| G10 | Add a private backend-neutral provider seam | Implemented | Validated | 177 focused passes; provider crosses spawned worker boundary | Retain regression only | None |
| G11 | Replace only the initial ligand provider and keep all downstream stages unchanged | Implemented | Partially validated | Focused call-trace/budget tests pass; no 182-case trace evidence | Yes | Dedicated paired O0/F1/F2 runner and trace recorder |
| G12 | Run F1 over the complete 182-complex cohort | Partially implemented | Not validated | Adapter/seam exist; formal runner and execution do not | Yes, primary objective; all 182 cases | Semantic cohort reconstruction, runner, installed runtime, compute budget |
| G13 | Make an owner-visible default-provider decision | Not implemented by design | Not validated | Open Babel correctly remains default pending F1 | Decision only after this round; F1 requires 182/182 | Accepted F1 evidence and owner instruction |
| G14 | Implement and optimize stock whole-complex C1 ETKDG | Partially implemented | Partially validated | Coordinate generator, unit coverage, and one-case Eu smoke exist; force-field continuation does not | Yes, required | Semantic graph artifacts, force-field entry bridge, obsolete hash removal |
| G15 | Implement and optimize coordination-bounded C2 ETKDG | Partially implemented | Partially validated | Coordinate generator/bounds tests and one-case Eu smoke exist; force-field continuation does not | Yes, required | Same as G14 |
| G16 | Implement Level-1 stereo/atom/bond transport and lifecycle | Implemented | Validated on dedicated fixtures | Copy, component, native, FF transaction, H/AXIS/group tests pass | Retain regression; compare F1 payloads | None for represented payload parity |
| G17 | Implement Level-2 Hotpot stereo semantics | Implemented | Validated on dedicated golden/fixture suites; not demonstrated by R0 corpus | All seven gates pass, but canonical 187 import yielded empty payloads | No new stereo science required for F1 | Source-SMILES stereo coverage would require a governed sidecar/importer and is separate |
| G18 | Preserve atom order, topology, charges, aromaticity, dative direction, and represented stereo through sampling | Implemented | Partially validated | R0/focused tests pass; complete O0/F1/F2/C1/C2 population evidence is absent | Yes | Per-case semantic snapshot and comparator |
| G19 | Retain both public `obWrappers` and `rdWrappers` | Implemented | Validated | Imports, API tests, and wheel coexistence pass | Retain regression only | None |
| G20 | Keep one public force-field interface; add no backend/profile switch; keep `candidate_count` reserved | Implemented | Validated | Public API and focused workflow tests | Retain regression only | None |
| G21 | Report stage-resolved performance and resource use | Partially implemented | Partially validated | R0 timing exists; complete O0/F1/F2/C1/C2 comparison is absent | Yes | Runner instrumentation and clean execution policy |
| G22 | Support and test CPython 3.10-3.14 wheels | Implemented | Validated locally | Five pinned workflow-equivalent wheel runs pass | Rerun only if validation changes packaged code | None for local claim |
| G23 | Obtain GitHub-hosted CI provenance | Not implemented locally | Not validated | Existing evidence is explicitly local | No, unless separately requested | GitHub workflow execution and retained artifacts |
| G24 | Satisfy strict standalone auditwheel manylinux policy | Not an original product requirement; explicit non-claim | Not validated | Open Babel remains a controlled cross-wheel dependency | No | Different Open Babel packaging design would be required |
| G25 | Recover byte-identical historical 182 manifest/graph sidecars | Obsolete prerequisite | Not required | Historical coordinates/reports are not scientific truth and volatile bytes cannot be reconstructed | Remove as blocker | None; create current semantic evidence instead |
| G26 | Build a normalized current semantic 187/182 cohort | Not implemented | Not validated | Existing cohort loader still expects historical/reference artifacts | Yes | New coordinate-free cohort builder and tests |
| G27 | Perform exhaustive manual review of O0/F1/F2/C1/C2 structures | Gallery infrastructure exists; new-route galleries absent | Not validated | Existing gallery covers old workflow only | Yes | Completed route runs, renderer, reviewer |
| G28 | Enforce the corrected selected-structure and optimization-continuation contract | Partially implemented | Not validated on full routes | Reports contain best energy/gradients and frame evidence; no evaluator proves recovery and continuation | Yes | Benchmark acceptance evaluator and focused recovery tests |
| G29 | Add F2 as precomputed ETKDG replay through the O0 downstream flow | Not implemented | Not validated | Current F1 generates inside the worker; no frozen-coordinate replay provider/runner exists | Yes, required | Data-only replay provider, artifact identity, paired runner |

## 5. Validation experiment inventory

| ID | Experiment | Existing status | Required result | Run in this round? | If not currently executable, what is missing? |
| --- | --- | --- | --- | --- | --- |
| V01 | Focused baseline and affected force-field regressions | Validated | No product assertion regression | Yes, after runner/evaluator work | Nothing |
| V02 | Exact RDKit/runtime/source identity audit | Validated | 2026.09.1 identities agree | Verify manifest at run start | Installed matching build |
| V03 | Python/C/C++ standalone API and SDK tests | Validated | Existing API/ABI tests pass | Only if packaged/native code changes | Nothing |
| V04 | Common-buffer and Open Babel parity tests | Validated | Existing results/errors remain stable | Retain focused regression | Nothing |
| V05 | Canonical R0 187 x 1/8/32 | Validated | Existing 559 success/two partial evidence retained | No mandatory rerun | Only rerun if rdWrapper/native code changes |
| V06 | Current semantic cohort reconstruction | Not implemented/not validated | Exact source hash; expected 182 positive and five negative cases; stable graph records | Yes | Builder/exporter and access to normal CBond model/runtime |
| V07 | Fresh O0 complete workflow | Not validated in this revision | All 182 named outcomes and stage/resource evidence | Yes | Paired runner and compute time |
| V08 | F1 complete workflow | Not validated | All 182 cases satisfy A01-A10 for default eligibility | Yes | Paired runner, RDKit native runtime, compute time |
| V09 | O0/F1/F2 post-provider trace parity | Unit-level partial validation | Same downstream stage sequence, budgets, checks, and recovery policy | Yes | Stage-trace schema/assertions in runner |
| V10 | Selected best-energy and selected-gradient audit | Not validated | Every returned O0/F1/F2/C1/C2 selected structure has finite best energy/RMS/max gradient | Yes | Acceptance evaluator |
| V11 | Unavailable-gradient recovery/continuation audit | Not validated | Every event either has an established successful recovery and later optimization progress or fails the case | Yes | Traverse all trajectories and record recovery transitions |
| V12 | Recovered intermediate numerical-excursion classification | Not validated | Energy/gradient excursions are stage-resolved; successful standard recovery continues, unrecovered interruption fails | Yes | Evaluator tests covering recovery and continuation |
| V13 | Standard terminal geometry acceptance | Old O0 reference passed; new routes absent | Every route is evaluated over all 182 cases | Yes | Complete route runs |
| V14 | Semantic topology/stereo parity | R0/unit partial validation | Zero O0/F1/F2/C1/C2 graph-semantic mutations | Yes | Coordinate-free snapshots and comparator |
| V15 | Complete O0/F1/F2/C1/C2 manual gallery review | Not validated | Each route review completed; zero visibly nonphysical accepted outputs | Yes | Route SDF/trajectory artifacts, renderer, reviewer |
| V16 | O0/F1/F2/C1/C2 timing and resource comparison | Not validated | Stage p50/p95/max/aggregate plus worker/thread/host identity reported | Yes | Recorded execution policy and monitoring fields |
| V17 | F1 repeated-run outcome consistency | Not validated | All 182 F1 cases are repeated; business pass/failure classification is stable, coordinates need not match | Yes | A second complete 182-case F1 run |
| V18 | Full 182-case C1 generation plus force-field matrix | One-case coordinate-only smoke | All 182 generated structures enter force-field optimization and are judged by A01-A10 | Yes, required | Semantic graph exporter, C1 force-field bridge, runner update |
| V19 | Full 182-case C2 generation plus force-field matrix | One-case coordinate-only smoke | Same as V18 plus coordination-bound observations | Yes, required | Same as V18 for C2 |
| V20 | F1 source-SMILES stereochemistry preservation | Not measurable with current 187 importer | No claim from empty payloads | No, separate scope | Governed stereo sidecar or importer implementation |
| V21 | CPython 3.10-3.14 wheel matrix | Validated locally | Rerun only if packaged production/native code changes | Conditional | Pinned manylinux environment and execution time |
| V22 | GitHub-hosted CI provenance | Not validated | Hosted run artifacts | No, separate release task | GitHub Actions authority/execution |
| V23 | xTB/DFT/QM handoff | Existing unrelated old benchmark only | No criterion in this plan | No, explicitly excluded | Future owner-approved scope |
| V24 | F2 precomputed-coordinate replay | Not implemented/not validated | All 182 frozen ETKDG proposals enter the O0 gate/topology/optimization flow and are judged by A01-A10 | Yes, required | Replay provider, coordinate artifacts, runner integration |

## 6. Required validation implementation

No production behavior should change merely to run this validation. The
minimum implementation is benchmark/test infrastructure.

### 6.1 Semantic cohort builder

Add a coordinate-free builder that:

1. verifies the 187-source SHA-256;
2. runs current CBond once per case with the recorded Am settings;
3. records the complete 182/5 partition without filtering unfavorable cases;
4. serializes normalized graph semantics without timestamps, absolute paths,
   runtime, coordinates, or old validation payloads; and
5. can produce C1/C2 native graph JSON as a derived artifact when requested.

The new identity is a digest of canonical scientific content. It intentionally
does not attempt to reproduce SHA-256
`4416bb8558ef420fad3176a82845d02bd1e83d9b271347c97d65174f799a0377`.

### 6.2 Paired O0/F1/F2 runner

Add one internal benchmark runner, tentatively
`run_o0_f1_f2_validation.py`, that:

- consumes the semantic cohort;
- rebuilds/copies the same graph for all three routes;
- injects `OpenBabelConformerAdapter`, `RDKitConformerAdapter`, or a data-only
  frozen-coordinate replay provider only through the existing private sampler
  seam;
- invokes the complete complex workflow with identical settings;
- enables full trajectory evidence;
- never falls back from RDKit to Open Babel inside F1;
- writes the F2 ETKDG proposal before force-field work, hashes it, and proves
  that the replay provider supplies the same rows unchanged;
- records sampling, ligand build, placement, restoration, complex
  optimization, acceptance, and resource/timing evidence separately; and
- writes per-case JSON plus a normalized aggregate summary.

The runner must use the complete complex workflow where the provider seam is
implemented. O0, F1, and F2 use the same gate/topology/optimization code. It
must not claim that the separate native-fast preliminary path was converted to
RDKit.

### 6.3 Acceptance evaluator

Add a pure evaluator over reports and trajectory archives. It must distinguish:

- selected versus exploratory frames;
- selected finite best energy from terminal/intermediate energy excursions;
- finite selected gradients from unavailable-gradient events elsewhere in
  force-field optimization evidence;
- an established recovery that restores finite gradients and continued
  optimization versus an unrecovered interruption;
- selected-frame explosion from recovered rejected-frame explosion; and
- geometry/topology failure from nonblocking convergence warnings.

Focused synthetic tests must cover at least:

1. finite selected structure with a recovered infinite-energy exploratory
   frame: pass if all gradients remain finite;
2. a non-finite-gradient frame followed by an established perturbation,
   finite gradients, and resumed optimization: pass with diagnostic;
3. a non-finite-gradient frame with no standard recovery: fail;
4. a recovery attempt that does not restore optimization: fail;
5. infinite selected `best_energy`: fail;
6. non-finite selected RMS or maximum gradient: fail;
7. selected-frame explosion: fail;
8. standard geometry failure: fail; and
9. budget exhaustion with a valid selected structure: pass with diagnostic.

### 6.4 C1/C2 force-field continuation

Add an experimental whole-complex force-field entry bridge shared by C1 and
C2. It must:

- consume the generated complete-complex graph and coordinates without
  rebuilding the ligand or replacing the metal position;
- apply the existing complete-complex entry geometry/topology checkpoint and
  established recovery policy;
- perform the subsequent Open Babel force-field optimization and terminal
  `standard` acceptance;
- preserve every C1/C2 generation artifact so the force-field result can be
  traced to its exact proposal; and
- use the same acceptance evaluator and manual-review contract as O0/F1/F2.

This bridge begins as benchmark infrastructure. Promoting a passing C1 or C2
route into production requires a separate owner decision because its stage
entry differs from the current ligand-provider substitution.

### 6.5 Gallery and report

Reuse the existing gallery infrastructure for the selected structures from all
five 182-case routes.
The final validation report contains:

- cohort identity and 182/5 partition;
- per-goal and per-experiment final status matrices;
- per-route and per-stage result counts;
- all named failures and recoveries;
- selected energy/gradient and geometry-gate summaries;
- manual-review result and flagged cases;
- O0/F1/F2/C1/C2 timing/resource tables; and
- the owner decision, without implying coordinate equivalence to old runs.

## 7. Execution sequence after approval

1. Freeze this reviewed validation contract.
2. Implement and unit-test the semantic cohort builder.
3. Implement and unit-test the corrected numerical acceptance evaluator.
4. Implement the paired O0/F1/F2 runner and prove fixture-level trace parity.
5. Materialize the exact 187 source and reconstruct the 182/5 Am cohort.
6. Implement and test the C1/C2 whole-complex force-field entry bridge.
7. Execute O0, F1, F2, C1, and C2 over all 182 cases in clean, recorded
   processes.
8. Repeat all 182 F1 cases for outcome-consistency evidence.
9. Audit all automated A01-A10 results and rerun only explicitly invalid or
   interrupted infrastructure cases; never discard scientific failures.
10. Render and complete the manual gallery review for every route.
11. Report performance/resources without inventing a cutover threshold.
12. Rerun affected regressions and any conditionally affected packaging tests.
13. Write the final validation report and request the owner decision on the
    default provider.

## 8. Explicit non-goals for this round

This validation does not:

- reproduce historical coordinates, energies, trajectories, or manifest
  bytes;
- treat old Open Babel output as a structural oracle;
- require every exploratory optimization energy to remain finite or small;
- add, remove, reorder, or weaken a force-field stage;
- activate multi-conformer selection or `candidate_count`;
- expose a public backend/profile switch;
- make a failed C1/C2 research hypothesis override an independently passing
  F1 182/182 default decision;
- repair source-SMILES stereo import for the 187 corpus;
- establish GitHub-hosted CI or strict standalone auditwheel provenance; or
- run or assess xTB, DFT, or other QM calculations.

## 9. Fixed decisions and remaining design review

The following owner decisions are fixed for execution:

1. force-field continuation after established recovery, not the mere presence
   of an intermediate non-finite gradient, is the operative criterion;
2. F1 requires 182/182 cases to replace O0 as the complex default;
3. a sub-182 F1 result retains the interfaces and adapter/options but remains
   non-default;
4. F2 is added and uses ETKDG coordinates with the O0
   gate/topology/optimization flow;
5. C1 and C2 both run on all 182 cases and continue into force-field
   optimization under the same final acceptance contract;
6. all 182 F1 cases are used in the primary run and the repeat run; and
7. xTB/DFT/QM remains outside this round.

Implementation review must pay particular attention to two boundaries before
coding:

- F2 must remain a precomputed-coordinate/replay control rather than duplicate
  F1's ETKDG algorithm or O0's downstream implementation; and
- C1/C2 must enter the force field without discarding their whole-complex
  coordinates, while reusing existing complete-complex gate/topology/recovery
  primitives instead of creating a second chemical policy.

## 10. Owner-authorized ancillary data work

The following two deliverables are required side work for this validation
round. They do not change the O0/F1/F2/C1/C2 acceptance contract or the F1
default-cutover decision. They must be completed before the final validation
report is closed.

### 10.1 Restore the tracked 187-extractant source

Restore `molecules/extractant/extractants.smi` as a Git-tracked source asset.
The restored file must be the exact canonical 187-line text defined in Section
2.1: Git commit `11c0e3d`, SHA-256
`e67ae3167a1a96d3c6a4c6df3ce12b70abfaf479bdd89bb156bfaefa771795a7`.

The restoration must:

- preserve all 187 physical lines and their original order;
- perform no SMILES canonicalization, deduplication, salt rewriting, or stereo
  rewriting;
- keep the file under normal Git tracking rather than under an ignored local
  data directory;
- add a regression check for the line count and exact content digest; and
- continue to derive the 182 CBond-positive and five no-CBond cases through
  the semantic cohort procedure in Section 2.2 rather than by maintaining a
  second hand-edited list.

This file is the immutable proposal source for the benchmark. It is not a set
of accepted 3D structures.

### 10.2 Publish the selected O0/F1/F2/C1/C2 structures

After the five full-route experiments, publish the selected best final
force-field structure for every O0/F1/F2/C1/C2 route-case result as the
owner-authorized Hotpot `CBond -> force field` benchmark dataset. The governed
dataset belongs under `hotpot/dataset`; it must use the existing `hotpot.ds`
Bundle, assessment, certificate, and ingestion workflow rather than introduce
a parallel benchmark database or a new payload root.

The complete result matrix contains 182 source cases by five routes, for 910
route-case result records and, when every route-case has an admissible selected
structure, 910 coordinate assets. Each record must retain at least:

- source line index, source SMILES, and the canonical 187-source digest;
- the reconstructed semantic graph digest and CBond donor/metal mapping;
- route (`O0`, `F1`, `F2`, `C1`, or `C2`), run ID, seed, Hotpot commit, pinned
  dependency versions, and scientific settings;
- the selected structure bytes and digest, atom-row mapping, selected energy,
  RMS gradient, maximum gradient, and final topology/geometry report;
- A01-A10 decisions, named recovery actions, terminal route status, and any
  manual-review flag; and
- explicit owner authorization for inclusion in Hotpot and provenance linking
  the structure to this validation round.

Only the route's actually selected final force-field structure may be stored
as its selected benchmark structure. A failed route-case must remain an
explicit failed result with its diagnostic evidence; it must not receive an
O0 structure, a rejected frame, or another route's structure merely to fill
the matrix. Consequently, any shortfall from 910 admitted coordinate assets
must be visible in both the dataset inventory and final validation report.

F1 repeat classification is control evidence for the provider decision, not a
filter on the primary 910-record publication matrix. Any primary/repeat pass or
gate mismatch must be preserved explicitly in the conversion manifest and
final report. It denies a stability claim and strengthens the decision to keep
F1 non-default, but it must not erase otherwise valid primary route results or
block the owner-authorized failed-result-aware dataset publication.

The coordinate payloads and generated Bundle/store artifacts follow the
existing `hotpot/dataset/data/**` local-data policy and therefore are not
Git-tracked. The dataset definition, attribution/authorization statement,
schema or Gate policy, and non-payload inventory manifest required to rebuild
and audit this release remain tracked under `hotpot/dataset`.

These structures are reproducible, governed reference outputs for regression,
route comparison, and future `CBond -> force field` validation. They are not a
single coordinate ground truth: a future result need not reproduce their
coordinates or energies if it independently satisfies the applicable
scientific acceptance contract.
