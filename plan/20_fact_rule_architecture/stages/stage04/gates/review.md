# Stage4 owning-stage gate closure

Stage4's implemented ownership, bounded local optimization, technical checks
and valid focused comparison are complete. Final performance acceptance is
open and blocked by measured regressions. This record changes no main-plan
requirement, public contract, scientific route, threshold, workload or policy.

## Controlling scope

The main plan's Stage4 gate requires deterministic graph/cycle/search order,
geometry and Crystal matrix parity, native geometry/graph checks, stale/member
identity, detached inspection, UID remap, constraints, default tests and
"focused performance comparison". Matrix parity is numerical correctness;
the separately named performance obligation is comparative evidence.

Section12.2 says a convincing family/end-to-end slowdown "remains unresolved
and blocks performance acceptance until fixed or specifically reviewed".
Stage12 requires the complete Sections11–12 validation/performance matrix;
Section14 retains final global parity. Neither passage makes final global
acceptance a new prerequisite for every intermediate implementation stage.
Stage3's existing owning-stage closure already preserves that distinction.

The prior Stage4 in-progress status was valid while its bounded investigation
and follow-up were pending. Those tasks are now complete. This is the explicit
state transition from that checkpoint, with executed evidence, before Stage5
entry; the old status is not silently bypassed. A reviewer identity does not
approve a gate or grant an exception.

## Actual closure evidence

| Obligation | Evidence and result |
|---|---|
| Mathematical extraction, deterministic ordering, numerical/Crystal convention | Full mapped source selection, unchanged-body checks and default tests in [implementation](../implementation/evidence.json) and [validation](../validation/evidence.json) |
| Source/member identity, detached inspection, UID remap, constraints, pair facts and rollback | Full affected source/runtime selections on all five versions; additional focused320 cases and retained original failure in [profiling](../performance/profiling/evidence.json) |
| Complete default |4585 passed,123 skipped,3 xfailed,49 subtests; frozen bytes stable |
| Python3.10–3.14 and current packages | Each2287 source/175 debug/2157 installed passed;42 source/installed external/scientific skips; wheel audit/smoke and542-member sdist passed |
| Native graph/geometry | Seven actual standalone builds/executables retained; unchanged source/test/recipe/binary closure reverified; no Stage10 clean rebuild claimed |
| Bounded optimization | [Local review](../performance/review.md): frozen implementation, preserved fields/bytes/locks/ordering and completed focused follow-up |
| Focused comparison validity | Six workers,78 attempts,60 measured, original3+10 sampling; equal actual work,stable source/native and zero comparator environment mismatches; [exact samples](../performance/evidence.json) |

The source commit is `0bc19f8` plus the exact frozen production/test digests in
the evidence's `optimization_followup`. All changed production/test bytes
match the independently reviewed delta; the peer ran no tests. The local
commands, environments, wheels, original failures, artifact identities and
coverage limits are recorded separately from static review.

## Open final acceptance

The latest graph/view workload is2.05414×B20,95%CI2.03997–2.06512;
the complete graph/geometry family is1.43729,CI1.43035–1.44402.
Both remain regressions. Numerical screening1.00568 and Crystal0.99424 are
parity under the frozen neutral band. Original4.053/1.103 measurements remain
visible. No exception is requested or approved, and lower process RSS provides
no waiver. Stage3 retained-memory/runtime issues also remain open.

Stage5 may begin its normal full-plan/companion reread and predecessor/source
assessment. It does not receive performance acceptance from this closure.
The remaining scientific, all71-workload, GPU/end-to-end and overall memory
obligations continue to block final main-scope acceptance until satisfied.
Stage13 stays separately branched and gated after completed, tested,
benchmarked and committed main-scope delivery.
