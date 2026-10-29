# Stage4 extraction checkpoint

Status: implementation and technical gates are verified; Stage4 remains in
progress because its focused measurements expose regressions under bounded
Sol/max investigation. This is a reviewable checkpoint, not final Stage4 or
main-plan acceptance.

## Implemented ownership

Existing graph/geometry and InternalCoordinates/cell calculations have their
mathematical owners; graph-spectrum policy and carrier geometry conversion
have their separate chemical descriptor and structure-adapter owners. Old
Python bodies are removed with actual consumers rewired. Existing native
module names, sources, build recipes and independent component boundaries
remain unchanged until Stage10.

Angle/Torsion and pair views have fixed source/member identity, on-demand
production, detached inspection and current-version serialization behavior.
One Molecule-owned private ConstraintSet preserves angle/torsion constraints
before their cache removal. Sparse supplied Wiberg observations, including
explicit zero, survive factual copy/subset/merge/current pickle independently
of regenerated mathematical pairs. Their final result/collection ownership
remains at Stage7B/7C. Ring writers/caches and eager chemistry remain Stage5.
[Exact contracts](contracts/review.md) and [schema](contracts/schema.json)
record public removals, retained independent read containers, finite private
fields and the actual rollback limits. No general experimental rule protocol
is implemented here.

The architecture ratchet retires13 actual entries and passes with268 remaining,
without new debt or a widened allowlist. Mathematical AST equivalence and the
canonical-tree recursion regression remain in
[implementation evidence](implementation/evidence.json). New private fact
ownership is actually included in the write scanner's negative tests.

## Executed evidence

[Technical report](validation/report.md): complete default4571 passed,
123 skipped,3 xfailed and49 subtests; each Python3.10–3.14 passes2273 source,
162 debug-allocator and2143 installed tests (42 source/installed skips),
plus wheel audit/smoke, actual source-distribution audit and7 standalone C++
source-API builds. Original failures and every subsequent repair are retained.

The default gate also exposed a pre-existing DS source-content cache defect
on timestamp collisions. A deterministic failing regression and100 actual
filesystem observations support its minimal release-time proof repair;
complete DS and supported-version evidence cover it. This adds no architecture,
public interface or scientific route.

[Performance report](performance/report.md): all six focused workers/78
attempts complete with equal work. Graph/search/view reads are4.053×B20,
Crystal conversion1.103×, numerical geometry screening0.997× (parity).
These regressions are not waived or hidden by lower process RSS. A bounded
Sol/max review investigates local creation/delegation costs before further
Stage4 closure; the current production bytes remain frozen until a concrete
change partition is accepted.

## Remaining boundaries

Stage4 is not declared complete at this checkpoint. The final71-workload,
scientific, end-to-end and overall memory gates remain open, including prior
Stage3 regressions. Stage5–12 and the separately branched Stage13 experiment
are not entered by this report. Stage13 still requires completed, tested,
benchmarked and committed main-plan delivery first.
