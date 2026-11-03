# Stage7B constraint-owner validation — 2026-11-03

**Root decision: constraint slice DELIVERY_READY; Stage7B IN_PROGRESS.** Remaining Atom-axis/Bond ownership is consolidated through the accepted [implementation release](../implementation_release.json). Hidden-topology consumer cutover and the Stage7B gate remain open. [Stage7A completion](../../external-adapters/completion_20261103.json) retains its bounded extraction/CP311 qualification. The effective goal remains unchanged; goal_v2 is review-only.

## Ownership and preserved contracts

The existing Molecule-owned `_ConstraintSet` now owns Atom axes as UID-keyed immutable28-byte value/presence/source cells and Bond constraints as UID-keyed raw8-byte cells. Angle/Torsion tuples retain their owner. The Atom table declares16 non-axis projection fields and Bond physically stores order/id only; public Atom19/Bond3 exports remain transient independent projections. All eight changed source files and their exact final hashes are in the [checkpoint](checkpoint_20261103.json).

| Acceptance criterion | Actual affected evidence |
|---|---|
| Existing setters, constructor aliases and permissions; one durable value owner | Scalar/batch/native tests check sole maps, no axis columns/Bond mirror, constant-work scalar access, isolated registration, and unchanged rejection of axes by public updata_atoms. The low-level unknown-field KeyError contract is preserved. |
| Raw transport and scientific meaning | Signed zero, NaN payloads, non-boolean raw floats, independent presence/source/producer metadata and19/3 field order survive factual install/copy/components/merge/current pickle. Public boolean codecs retain their existing meaning. |
| Source/lifecycle integrity | Constraint identity keeps its boolean semantics; forcefield_inputs retains raw values/metadata sensitivity. Detached Atom reads remain available with membership-rejected writes; nonmember/removed Bond permissions, UID gaps, endpoint-equal removal and map isolation are checked. |
| Complete rollback | Owner-only/mixed native value/source/presence/enrollment KeyboardInterrupt/MemoryError failures restore old cell/map identity. Mixed numeric/Bond/sequence/registration/topology rollback, source-stale rejection, nested poisoning and the six existing working-copy success/failure cases are included. |
| Existing topology behavior | Hidden values survive hide/restore and active-only clear_constraints. Permanent hidden clear filters owner entries while old handles remain inspectable. Hidden-topology retirement itself is pending. |

No public signature, return schema, type or exception was added. The dormant `_commit_constraints` atom_update/bond_entries branches were removed, including their unused slot2/id bug. Current force-field paths do not acquire a new physical-restraint algorithm from this storage migration; RDKit explicit embedding constraints remain distinct. Chemistry, model weights/features and finding/control semantics are unchanged.

## Evidence actually obtained

Source parent is `48780a669c9b1ca5b0e6c2cadf63365579ad3ccf` plus eight explicitly bound production overlays. The first source attempt remains839 passed/2 failed/0 errors/0 skipped. One failure exposed the unknown-field guard regression, corrected in the product; the other exposed a new test's positional metadata ordered by input instead of canonical candidate.fields, corrected without weakening its raw-value/metadata assertions. Original test inputs and failures remain preserved.

| Boundary | Actual result | Attribution and limits |
|---|---:|---|
| Final CP311 source | 841 passed;0 failed/errors/skips | Same ordered841cases as attempt1;149 actual Hotpot module origins in checkout; source/native/tests stable. |
| Current CP311 wheel outside checkout | 841 passed;0 failed/errors/skips | Identical ordered cases;149 actual Hotpot modules from the installation; assertions unchanged. |
| Installed actual CPU FP16 MCA batch | 1 passed;0 failed/errors/skips |153 actual Hotpot modules from the installation; CN succeeds, charged C[NH3+] input2 rejects locally, CO succeeds; CLI returns1 and input remains unchanged. |

The model-specific observed C/N values were308.75/489.50kJ/mol and C/O283.50/318.25kJ/mol. The acceptance criteria are finite complete rows, original input/output attribution, local domain rejection and independent continuation; these observations are neither fixed scientific truth nor model-wide accuracy evidence. Test totals describe the actual selections, not independent capability counts or a chemical correctness oracle.

Ordinary build_py/bdist_wheel assembly used all455 current production Python files, all8 overlays and6native artifacts. Root rebuilt only `_fact_native` (SHA256 `cea3ed1ab971ad7c108fa6b44db56a85df5f6e61eccd9caed51c44041cab4bad`); five natives were reused. The6,289,427-byte CP311 wheel has SHA256 `f670fafe2281269a1fd7ea2b64b1d840a8fbe39889d8247c517bdf3726d34f5c`. The fresh --system-site-packages venv shares existing dependencies. All455 Python files,8 overlays,6native artifacts and model inputs are bound before/after;490 input identities shared with source-attempt2 match. All41 non-Python wheel omissions remain explicitly listed; none of the eight overlays is missing. This is current-source/rebuilt-extension installation evidence, not clean all-native/ABI/SDK/sdist/manylinux or all-version acceptance.

Exact actual argv, cwd, environment, interpreter/model/wheel identities, JUnits and origins remain in the verifier-owned [archive](../../../../../../.plan20-artifacts/20261103/stage7b/constraint-consolidation/archive-manifest.json): [native build](../../../../../../.plan20-artifacts/20261103/stage7b/constraint-consolidation/native-build-attempt1/execution.json), [source retry](../../../../../../.plan20-artifacts/20261103/stage7b/constraint-consolidation/test-implementation/source-attempt2/execution.json), and [installed execution](../../../../../../.plan20-artifacts/20261103/stage7b/constraint-consolidation/installed-validation/execution.json). It preserves111 evidence files/5,793,197bytes and the original failures. Large wheel/runtime/model assets retain their recorded external paths. This report task inspected evidence and source; it ran no business tests, build, install or Git command.

## Bounded usable cost

Root accepts the [same-driver comparison](../../../../../../.plan20-artifacts/20261103/stage7b/constraint-consolidation/cost-current/comparison.json) and retains its [complete raw report](../../../../../../.plan20-artifacts/20261103/stage7b/constraint-consolidation/cost-current/report.md). Both sides complete45/45cells at128/1024/8192atoms, with3warmups and10samples per cell,133770timed iterations, identical driver/runtime/counts/order and matching deterministic transport observations. Current process RC0/reaped; production bytes are unchanged.

| Workload | Observed current/baseline | Current cost / meaning |
|---|---:|---|
| Requested three-axis projection |6.276–19.854x |6.772ms at8192atoms; bounded linear projection overhead is retained. |
| Full Bond3 export |2.651–2.741x |21.097ms at8192atoms; repeated per-Bond projection overhead is retained. |
| Full Atom19 export |1.045–1.063x |All declared fields remain included. |
| Copy / clear constraints |1.443–1.657x /1.334–1.367x |Complete remap/clear work remains included. |
| Fixed16-atom copy/edit/export/clear workflow |1.242–1.252x |About2.07ms versus1.65–1.67ms. |

Scalar reads/writes remain O(1) in molecule size by the touched-key implementation and observed scaling. Under the accepted semantics > usable performance > version-adaptation direction, these explained absolute costs are usable for this slice and do not block the next semantic migration. Reassess upon a representative workload failure, unbounded resources or a concrete batch bottleneck. All adverse measurements remain. Sequential baseline/current runs on an uncontrolled shared host provide no paired-randomized CI, parity, speedup or global causal conclusion. Tracemalloc omits some native allocations; RSS is process-level/current/monotonic peak, not retained-memory or leak proof.

## Remaining gates

Stage7B hidden-bond consumer chemistry/masks, complete ordered-Bond cutover, stale-view/native publication/transport and whole-sub-stage acceptance remain pending. Current active-only clear scope remains until that connected cutover. Stage7C/D/E and later stages remain open; the known H/isotope/stereo semantics and unapproved H/model/storage choices remain Stage7E obligations. Clean all-native/ABI/SDK/sdist, other versions, broader scientific/GPU/external evidence where required, Plan12, global performance and final Stage0–12/B_exp-before-Stage13 acceptance remain open. Root owns business Git delivery. This checkpoint activates no draft goal, new stage or scientific route.
