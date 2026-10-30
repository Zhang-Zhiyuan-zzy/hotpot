# Stage6 bounded stereochemistry extraction design — revision 2

Status: **root-key accepted design; final root integration after revision2 review**.
Production/test ownership remains **HOLD** pending exact release. This record
supplies no runtime, scientific, performance, native, wheel or Stage6 acceptance.
The assigned author is GPT-6.1 Sol/max. The separate historical reviewer result
and root decisions are recorded in [decisions.md](decisions.md); partitions,
contracts, preservation identities and static checks are in [evidence.json](evidence.json).

## Review basis and scope

Revision1 was blocked by task `p20-s6-contract-review-704283-34`, revision1,
verification704283. Its result SHA256 is
`40ae276830957e41f409ede01727d2478d4a7d063115588f073f7f6d728be354`.
The matching actual audit is a historical receipt/identity PASS, **not design
approval or inspection of the current implementation**. Root retained the raw
review/audit in the task34 archive; its maintained review pair is now under
[initial-review](initial-review/review.md). This revision addresses all six
findings; it does not reuse the blocked result as an acceptance.

Task34 revision2 resolved the six findings statically and retained root-key and
implementation-evidence blockers. Its exact result is6,721 bytes, SHA256
`5b44416bfa79280d617f5032c72751ca750a61082c71e7796a0cb9dbde05f063`,
turn `01a18cc5-cec4-7b93-aabb-d0e509405603`, historical source
`979455ed8bada61f4d44038935e7f02c1b204c86`, capture-manifest SHA256
`3e2476a82e925f9c33683bdb1b10361cefff111b0d6a2ad8703fa2eaf238da82`.
The matching actual-turn audit PASS establishes receipt/identity only. Root then
separately accepted the two exact RuntimeError contracts, N09 and mandatory
Stage7E, and directed the cost-disposition rule below. These are final root
integration edits, not revised reviewer acceptance. All five reviewed input files
were preserved byte-for-byte before editing, with their original hashes retained
in evidence and the ignored `/tmp` archive.

Extract the one current 4,747-line `hotpot/cheminfo/stereo.py` at SHA256
`4c8c66f411af3d1f627293a47558d8479ee45cbe62a54220fc44ea873dad34e0`.
The original116 definitions,23 assignments,18 public classes, five functions,
ordered23 exports and13 direct private-symbol test consumers remain the frozen
source-partition basis. Entry is [entry_review.json](../entry_review.json),
checkpoint `90ff36c2c0b415bb9cf27628265bb4528e425922`. Root's later C49/C50
checkpoints are record-only; no repeat source audit is inferred or requested.

One implementation remains after a coherent move/delete cutover. No scientific
algorithm, H convention, native matcher/schema, public schema, duplicate DTO,
Protocol, enum, facade-private compatibility alias or runtime selector is added.
Stage6 preserves installed whole-payload storage through an access/capture seam.
Main Section5.3 and the mandatory Stage7E scope now assign its complete canonical
replacement before Stage8 and Stage12/global ownership acceptance.

## Sole owners and imports

Owners below are relative to `hotpot/cheminfo/`:

| Owner | Responsibility |
|---|---|
| `structure/stereo.py` | The existing18 public enums/frozen records, including Level2 issues/reports/families; one contract owner, no carrier/rule/operation/adapter/native import. |
| `structure/_stereo.py` | Lock-protected `_get_stereo_payload` access primitive; Stage7E later installs assembly here. No remap rule body. |
| `structure/snapshots.py` | New exact three-field `_StereoSnapshot`; relocation of existing four-field `_Gate6Graph`; existing bonding contracts unchanged. |
| `adapters/structure/stereochemistry.py` | Raw capture and operation-local identity referents, deferred raw-capture check, requested prepared scalar extraction, exact native dictionary packing. No H preparation, interpretation, authoritative writes or rule import. |
| `operations/chemistry.py` | Existing H preparation, relocated `_set_stereo_payload`, and real lifecycle `_remap_stereo_transport` entry invoking algebra once. Current-H materialization uses the structure getter and local setter. |
| `operations/stereochemistry.py` | Public-call control, per-owner capture/preparation scopes, mapping/coordinate/native boundaries, conditional H/projection/native control and ordered report/proposal return. |
| `rules/chemistry/stereochemistry/algebra.py` | Descriptor/permutation/rotation/record/group transforms, `_bond_key`, `_axis_bond_keys`, and the sole pure `_remap_stereo_payload`. |
| `rules/chemistry/stereochemistry/graph.py` | Represented-graph preflight/projection, Gates2/5/6/7 topology/chelate/stereogenicity; deterministic task-local native callable supplied by the operation. |
| `rules/chemistry/stereochemistry/coordinates.py` | Existing TH/DB/AX and coordination geometry, finite-array checks and geometry policies. |
| `rules/chemistry/stereochemistry/validation.py` | Pure Gate2/5/7 report assembly and graph/coordinate issue assembly. |
| `rules/chemistry/stereochemistry/assignment.py` | Direct-priority subset and proposal/report assembly from captured values. |
| `rules/chemistry/stereochemistry/alignment.py` | Mapping diagnostics over copied entries/counts/payloads, issue prefixes, partial remap and final alignment/chelate/group comparison. |
| `stereo.py` | Explicit18 class reexports, existing ordered `__all__`, five annotated forwarding definitions. No rule bodies or private aliases. |

Algebra also owns `_record_atom_indices`, `_record_issues`, `_gate2_issue`,
`_validate_gate2_double_bond_carriers`, `_validate_gate5_axial_carriers`,
`_mapped_alignment_record`, `_alignment_record_identity`, `_transform_gate3_double_bond`,
`_transform_gate5_axial`, `_transform_supported_record`, `_AlignmentMatch`,
`_relative_group`, `_alignment_group_issues`, `_records_atom_union`, and
`_coordinate_group_consistency_issues`. The shared `_Gate7CoordinationSite`,
`_gate7_site_from_record` and `_gate7_mapped_chelate_pairs` are algebra contracts
and transforms. None needs native execution or a validation controller.
`_Gate7CoordinateCandidate` stays in coordinates; `_Gate6SemanticContext` and
`_Gate6Equivalence` stay in graph. These are relocations, not new types.

Graph's `_molecule_stereo_topology` and `_molecule_stereo_bond_kinds` take captured
inputs; names may become snapshot-specific with all private consumers migrated.
`_validate_gate2_stereochemistry` becomes a pure validation rule. Conditional-live
`_validate_graph_stereochemistry` and `_gate7_projection_result` belong to
operations. Six rule owners suffice; no extra shared-rule module is needed.

Contracts → algebra → graph/coordinates → validation/assignment/alignment →
operations → facade is acyclic. The structure adapter consumes contracts and
existing structure source helpers only. Chemistry operations may import algebra;
algebra has no operation edge. Core's convenience/lifecycle methods use local
operation imports; no rule imports Core, an adapter, operation, facade, toolkit
or extension. Empty rule package initialization is sufficient. The evidence
records the14-node stereo slice plus its direct runtime annotation/external and
existing-internal boundary dependencies; `TYPE_CHECKING` edges are separate.
This is a proposed import graph, not an implemented import-order pass.

## Exact capture and deferred preparation

N09 is root-key accepted, revised from unused `StereoUpdate` to one frozen
private dataclass with exactly these fields and order:

```python
payload: StereoPayload
atom_keys: tuple[int, ...]
bonds: tuple[tuple[int, int, float, str, str | None], ...]
```

`atom_keys` are ordered transient `id(atom)` integers. Bond rows contain endpoint
identity keys, raw float order, kind value and raw direction in exact active →
hidden-metal → hidden-covalent order. Resolve keys into rows in the rules;
object IDs never become native labels. Native real/virtual-H labels remain
consecutive row integers. Preserve missing-endpoint diagnostic lookup behavior.

Capture returns one snapshot per distinct owner. The operation also holds ordinary
strong tuples of atoms, each represented-bond partition, and captured endpoint
objects until computation finishes. Bonds alone would not protect a former
endpoint if another thread changed that Bond's endpoints. These referents and
an existing owner-only `_SourceIdentity` remain operation-local; they are not
snapshot fields, rule inputs, stored state or a new type. Capture reads no
coordinates, aromaticity or H readiness and does not normalize/reseal the payload.

Each rule invocation consumes immutable captured values. Prepared strict-H tuples
and requested Z/charge/aromatic triples are captured only when the original route
first needs them, after preflight and under that owner's lock. This is a
field-selective composition, not another DTO or a Molecule mirror. Reuse prepared
atomic numbers for direct-priority work. No whole-operation simultaneous snapshot
of two molecules is promised.

Deferred preparation first reacquires only its owner, calls existing owner-only
`_check_source`, and compares the raw capture against the current owner. The
adapter checks payload value/schema/order, ordered atom identity, each active/
hidden bond-partition identity and order, captured endpoints and raw bond fields.
The retained referents prevent transient-key reuse; partition checks also catch
an active↔hidden move whose flattened graph would otherwise appear identical.
Float source comparison preserves raw bytes/sign, rather than `True == 1` or
ordinary float equality. This check constructs no second `_StereoSnapshot`.

On mismatch the root-key accepted local failure is
`RuntimeError("Captured stereochemistry facts changed; retry the operation")`,
before H/native work for that owner. It returns no mixed-state report or proposal.
An owner-token mismatch keeps the existing `_check_source` error. This is an
accepted new read-operation contract, not evidence that the old exception behavior
was unchanged; implementation and runtime proof remain unreleased.

The raw check deliberately binds only facts already consumed. Z/charge/aromatic/
H-readiness values not yet consumed are selected at the original later
preparation point and frozen then. Do not bind the broad `representation` domain:
that would read unrelated aromatic/source/frame state before preflight and add
conformer-lock/hash work to ineligible paths. Existing H proposal source binding
and its effective input tuple stay unchanged. H preparation may legitimately
change its own current values after the raw check, as today; no post-H stale check
rejects those changes. Ordinary no-projection paths perform no H, prepared-field
read, native construction or deferred source check.

## Lock scopes and exact execution order

No two fresh stereo-owned molecular capture/preparation scopes overlap. For each
owner, acquire, capture or check/prepare, and release; pure rules, mapping input
callbacks, coordinate coercion and native mapping work then run outside those
scopes. Reference and target are processed in the original order. Same-owner
alignment reuses one capture and, when needed, one prepared projection and
native target/report; both issue prefixes remain in their original positions.

Root has separately key-accepted this **narrow transaction route**: if
distinct-owner alignment is entered with any tracked
outer `_FactTransaction` in the current `_TRANSACTIONS` context, every stereo-owned
molecular scope in that alignment uses `acquire(blocking=False)`, including both
initial captures and deferred preparation. A same-thread RLock reentry or an
uncontended other owner succeeds. Contention raises
`RuntimeError("Stereochemistry alignment cannot acquire a distinct owner while a factual transaction is active")`
at that attempted acquisition. Do not wait, spin, sort/reorder outer locks,
release a caller's lock, or reject all active transactions. Include a tracked
third-owner transaction when selecting this route; checking only the two inputs
would miss an inherited owner. Same-owner calls keep their reentrant route.

This prevents this operation from adding a blocking wait for another molecular
owner in that tracked route. Current H/commit helpers reacquire only the prepared
owner reentrantly. Owner-only `_capture_source(mol, ())` and `_check_source` with
empty dependencies do **not** acquire a conformer lock. Nonempty dependency
capture/check paths reached by existing H helpers retain their existing same-owner
conformer-lock hierarchy. No new cross-owner molecular acquisition is hidden in
the inspected route. This is a static bounded sufficiency argument, not a
runtime deadlock proof: inherited outer locks remain held during callbacks/native
work, existing conformer-lock behavior remains, and arbitrary private-held RLocks,
caller callbacks, leaked/copied transaction contexts or monkeypatches can add
other waits. Do not claim general deadlock freedom or callbacks outside caller-
owned locks. If implementation reveals another cross-owner wait in these helpers,
stop at that concrete boundary for a revised review; do not broaden rejection.

A failed initial capture occurs before mapping/H/native work. A late contention
or stale-capture failure occurs at its original deferred preparation point after
any preceding mapping callback/reference work. It does not undo a prior successful
lazy preparation on another owner or caller mutation. Stereo introduces no paired
transaction and no blanket rollback. An exception escaping a caller's existing
transaction retains that transaction's ordinary rollback behavior. Releasing a
stereo scope releases exactly its own acquisition and never the outer recursion.

Preserve this control sequence:

1. Capture the reference and target raw inputs independently, reference first;
   reuse one capture if identical. For align, copy/sort `atom_mapping.items()`
   with the existing `(repr(key), repr(value))` key outside stereo scopes. Preserve
   count, per-entry/range, many-to-one, then missing-record coverage diagnostics.
   Do not eagerly coerce coordinates or prepare target H here.
2. A parsed coordination **head**, even with an invalid coordination value,
   requests projection before Gate2 validation. Otherwise Gate2/5 report assembly
   comes before valid TH/DB/AX candidate selection. Preserve same-center, parse,
   anchor, carrier, range, duplicate-anchor, topology and Gate7 issue order.
3. `_gate6_projection(mol, *, snapshot=None, ...)` belongs to operations.
   Production supplies its one capture/context; direct private diagnostic callers
   enter the same capture/preparation path. Pure preflight diagnoses atom-fact
   range/duplicates, missing endpoints, self-loops, parallel edges, nonfinite
   order and unknown dative direction in the current order. Any preflight issue
   returns before strict H, aromatic reads or native construction.
4. On preflight success, reacquire/check the raw owner as above, call existing
   `_require_implicit_hydrogens(mol)` once at this original point, capture the
   requested prepared tuples and release. Keep selected-row/readiness/exception
   semantics. Pure projection preserves isotope0/1 protium normalization,
   negative-H-before-native diagnostics, cyclic aromatic normalization, noncyclic
   biaryl bridges, virtual H, H-slot signatures and sorted node/edge order. Raw
   supplied/inferred counts keep today's convention; no H import repair occurs.
5. Candidate selection retains invalid/unsupported sets, TH/DB/AX order and valid
   coordination descendant context. No candidate adds no projection/H/native
   call. Coordination-only projection may prepare H but creates no native target.
   A failed coordination projection is not reported twice.
6. A successful candidate graph constructs one existing
   `search.native._NativeTarget`. Supply its bound `iter_mappings(query_spec)`
   callable synchronously in frozen pair order, do not retain it, and execute it
   after release. Preserve current pins/queries, postfilters, tri-state accumulation
   and first-equivalent short circuit. Missing-extension/native errors retain
   their boundary; there is no fallback or public capability framework.
7. Optional `np.asarray(..., dtype=np.float64)` coordinate coercion remains
   **after complete graph validation** and is skipped if any graph issue has
   `record_index is None`. User `__array__` runs outside stereo scopes, never
   early to simplify locking. Preserve `(N,3)`, finite diagnostics and current
   view/copy behavior. Classifiers do not write caller arrays; the existing
   no-concurrent-array-mutation assumption remains.
8. Assignment validates once, proposes only, returns the captured exact payload
   when coordinates are absent, retains unchanged record sharing, and derives the
   same bounded TH/DB names and AX unsupported issue. Alignment runs mapping
   validation; reference then target coordination projections; reference then
   target complete graph validation **even for invalid mappings**; mapping issues
   then prefixed reference/target issues; partial remap/early return; then chelate,
   descriptor and enhanced-group comparison. For noncoordination target candidates,
   target H/native validation remains after reference native validation. Freeze
   consumed values rather than moving target preparation earlier.

## Public introspection and transitional storage

The actual source already uses `from __future__ import annotations`; `Molecule`
is imported only under `TYPE_CHECKING`. Task34's wording about evaluated public
annotations was incorrect. Preserve this actual contract, not that premise.
The facade keeps the future import, runtime `typing.Mapping`, `numpy as np` and
`numpy.typing.NDArray`, and its type-only Core import. It gains no runtime Core
import merely to improve default `get_type_hints` behavior. The full signatures
and exact annotation strings remain:

```python
get_stereo_payload(mol: Molecule) -> StereoPayload
set_stereo_payload(mol: Molecule, payload: StereoPayload) -> None
validate_stereochemistry(mol: Molecule, *, coordinates: NDArray[np.float64] | None = None) -> StereoValidationReport
assign_stereochemistry(mol: Molecule, *, coordinates: NDArray[np.float64] | None = None) -> StereoAssignmentReport
align_stereochemistry(reference: Molecule, target: Molecule, atom_mapping: Mapping[int, int]) -> StereoAlignmentReport
```

In a clean current module namespace, default `typing.get_type_hints` is statically
expected to raise `NameError` for missing `Molecule`; this has not been runtime-
tested in this design task. Explicit `localns={"Molecule": core.Molecule}` should
resolve the existing strings. Freeze and compare both outcomes, full
`inspect.signature` (including defaults/keyword-only/annotations), `__annotations__`
and explicit-namespace results in source and installed-wheel baselines/cutover.
A desired default-hints improvement is a separate public introspection change.

Define each public class once in `structure/stereo.py`; explicitly retain
`__module__ == "hotpot.cheminfo.stereo"` for the closed18 roster and preserve
`__qualname__`, fields/annotations/defaults/frozen status and enum values/order.
The facade reexports those same objects. Test cold unpickling in both import
orders and class/instance pickle round trips without injecting aliases. Verify
`_identity._canonical` enum/dataclass module+qualname tokens and source digest
inputs. Static acyclicity cannot prove cold imports/pickle resolution.

Get keeps the current exact installed object. Set retains lock → owner source →
existing `_commit_stereo(mol, source, payload)`, builder sealing, source checks,
UID-effective H-input comparison, journaling and rollback. Only changed effective
radical/current bracket-H/no-implicit values seed H; isotope, source-H and opaque
stereo alone do not. Preserve copy/component/merge/import/remap/pickle/native and
working-copy value/order/schema/sharing behavior throughout Stage6.

`get(mol) is payload`, distinct constructor payloads and exact reference restoration
are transitional Stage6 tests. Main Section5.3/Stage7E now require deletion of whole
`_stereo_payload` and mirrors, canonical Atom isotope/radical/current-H declarations,
separate Bond display/dative direction, source-only ledger, atomic assembly/
decomposition and complete connected migration. Public payload identity is not a
permanent guarantee. Opaque schema47/record-row81 on an empty Molecule and accepted
duplicate/out-of-range atom-fact rows must remain lossless; malformed projection
diagnostics must not become earlier setter rejection. Stage7E freezes canonical
slot/presence/source and unbound-source handling before code, with zero new types
and no silent site-total/omitted-H normalization. Its unmet gate blocks Stage8 and
Stage12/global acceptance; Stage6 cannot close it.

Preserve all Gate2–7 algorithms/constants: relative tolerance `1e-8`, Gate7 raw-
radian acceptance12° and margins2°/3°, rotations, representatives, proper-rotation/
Kabsch fitting, descriptor grammar, carrier counts and diagnostic text/order.
Keep hidden-bond participation and finite-default/readiness behavior. Open Babel
has no direct stereo import; preserve its factual-record payload handoff without
adding an engine or ledger. Stage7C/8A/8C/10/13 implementations stay outside scope.

## Exact native packing and bounded observations

Relocate `_Gate6Graph` with its existing four fields. Make `target_spec` deeply
read-only without changing its logical schema. The structure adapter packs plain
nested dictionaries once per constructed existing native target, never per
carrier pair or query. Private query dictionaries stay task-local; no native ABI
or matcher move is included.

Dictionary equality is insufficient. A recursive frozen-baseline-versus-packed
check must require `type(left) is type(right)`; identical insertion-ordered dict
keys; exact container kinds and sequence order/length; scalar values/None; and
exact float bytes, including signed zero. This covers bool versus int, tuple versus
list, all top-level/nested field names, normalized kind/order/aromatic values,
real/virtual labels, H slots, atoms/edges/bonds order and every `object=None`.
Compare the ordinary packed form to frozen pre-cutover source/installed-wheel
fixtures, not to the mapping-proxy wrapper's intentional private type. Capture
fixtures before authorized cutover using the bound existing source/wheel/native
identities; do not preserve a second old scientific implementation in production.

Native acceptance and actual mapping results must remain equal. Spies at the
adapter and real `_NativeTarget` constructor prove one pack per target and zero
pair/query repacks. Existing real `iter_mappings` observations (`<=2`, exactly1)
remain. The sole bound callable is synchronous, task-local and unretained.

Deep freezing plus packing adds an O(V+E) pass. Future bounded observations use
representative existing chemistry fixtures and fixed transport graphs with16,
128 and512 real rows, recording augmented V/E and H expansion. Use the main
policy's3 warmups and at least10 measured samples, fixed equivalent prepared-H
state, source/native builds, process/resources and fixtures. Report capture,
preflight/projection, freeze, pack and native time separately, Python peak/retained
allocations and RSS/native-allocation coverage limits. Include the deferred raw
check's additional traversal. These observations support only this bounded cost
assessment; they establish no global speedup/parity. The three invalid clean-read
comparisons remain non-comparable.

The following **root-directed adverse-disposition rule is frozen before any
bounded cost measurement** (S6-R2-09):

1. Freeze fixture membership, identical semantic work, raw input/native/artifact
   identities, prepared-H/readiness state, augmented V/E, resources, instrumentation
   and sampling/noise policy before baseline/candidate runs. Compare complete
   equivalent work and report capture/check/projection/freeze/pack/native components
   alongside totals. A new component without an old counterpart has separately
   reported cost; never fabricate its baseline. Use Section12.1 raw samples,
   median/MAD, meaningful p95 and paired-bootstrap95% intervals, and Section12.2's
   existing noise/uncertainty policy; no new final global threshold is introduced.
2. Retain all raw observations, attempts, failures, adverse results and missing,
   excluded or non-comparable cases with reasons. Unequal semantic work or
   prepared state cannot support a comparison. Missing or inconclusive required
   bounded evidence remains **OPEN** and cannot close the Stage6 gate.
3. Any correctness, target/pack/query-count or retention violation, unexplained
   scaling, repeated per-query work, or unexplained adverse time/allocation cost
   **BLOCKS Stage6 pending a fix and repeat**. Expected O(V+E) work alone cannot
   convert an adverse observation into a pass. Increased retained carrier/graph/
   callable state after operation completion is a retention violation, distinct
   from measured transient packing allocations.
4. Every attributable adverse time/allocation result must receive an explicit
   root disposition under the frozen sampling/noise policy before Stage6
   acceptance. Each disposition identifies fixture/work/size, raw sample and
   allocation artifacts, observed median/spread/uncertainty, cause, and either
   fix/repeat or reasoned **Stage6-local acceptance of measured necessary bounded
   O(V+E) overhead**. Established unexplained or unresolved adverse results stay
   blocked; missing/inconclusive comparisons stay open and cannot downgrade an
   established violation or adverse result. No result is dropped or auto-accepted.
5. A local overhead acceptance preserves its adverse observations and the
   unchanged Section12/Stage12 global time, memory and end-to-end performance
   debt. It establishes no speedup/parity and grants no global-performance waiver.
   Close this local gate only after all required bounded evidence is conclusive,
   compared work/state match, correctness/count/retention checks pass, and every
   adverse result is fixed/repeated or explicitly accepted on the bounded terms
   above. No bounded-cost pass or measured acceptance exists in this record.

## Connected cutover, checks and gates

Future ownership is requested, **not released**: the13 owner paths above, empty
rule `__init__.py`, `core.py`, `structure/_builder.py`, `_native/buffers.py`,
`rdconvert.py`, `conformers/adapters.py` and `forcefields/working_copy.py`.
`_commit.py`, `_identity.py`, `search/native.py`, `obconvert.py` and native sources/
headers remain read/validation consumers in Stage6. A necessary production change
to one requires an exact separately released scope. Source binding uses their
existing owner-only API; no identity-domain extension is proposed.

Rewire internal transport imports directly to structure contracts; reads to the
structure getter; setters and lifecycle remap to chemistry operations. The latter
invokes the sole algebra rule once. Core/builder/RD/native/conformer/working-copy
routes remain connected and stop importing upward through the public facade.
Public-only tests/benchmark users keep the facade. Stage7E later migrates the sole
surviving toolkit/native/workflow paths through its entry rename manifest.

| Existing private access/spy | Actual new import/patch owner |
|---|---|
| Descriptor/rotation/record/group transforms, including pure remap | `rules.chemistry.stereochemistry.algebra` |
| Geometry candidates, policies, fits/classifiers | `rules.chemistry.stereochemistry.coordinates` |
| Graph/chelate/query/postfilters | `rules.chemistry.stereochemistry.graph`; report assembly in `validation` |
| `_gate6_projection(mol)` direct callers | `operations.stereochemistry._gate6_projection`, using production capture/preparation |
| Gate7 projection spy | Patch the operation lookup and accept private captured context; keep distinct-input call observations `[reference]`, `[reference]`, `[reference,target]`; separately test same-owner reuse. |
| Assignment complete-validation spy | Patch operation `_validate_graph_stereochemistry`; retain count1. |
| Real mapping spies | Existing `search.native._NativeTarget.iter_mappings`; retain real <=2/exactly1 observations. |
| Direct `NativeTarget(graph.target_spec)` fixtures | Actual adapter `_pack_gate6_target(graph)`, with typed/order/native assertions. |

Keep the exact requested16 test paths in evidence:13 existing private consumers,
focused API/buffer additions and new `test_stereo_extraction.py`. Scientific golden
JSON/expectations remain unchanged. New meaningful contracts cover immutable
capture, raw-source drift, same-owner/reentrant/uncontended/distinct tracked-
transaction contention, reversed-ID pairs and third-owner outer transactions;
custom Mapping/`__array__`/native-spy reentry; no stereo-owned lock during callbacks;
late-failure side effects; error/issue order; exact packing/counts; and public
introspection/cold pickle. Use bounded thread barriers/timeouts, not a hanging
stress loop or a claim that arbitrary private-lock use is safe.

The sole class addition is `_StereoSnapshot`: +1 class/+1 dataclass, zero
Protocol/TypedDict/enum/ABI-mirror/owner-class/explicit-alias additions. N09's cap1
is unchanged; no `StereoUpdate` also exists. Moves earn no retirement credit.
N03 is already consumed by `_BondingSnapshot`. The recorded776 classes/353
dataclasses/8 Protocols/2 TypedDicts are inherited measured counts, not a new census.
Existing source/record/update/transaction families give Stage7E a zero-new-type
budget unless a separate concrete entry decision changes it.

Reviewable checkpoints are accepted contracts/budget/owners; one complete move/
delete cutover; exact `SOURCE_READY`; focused and full affected validation; then
owning reports/gates. No dormant second scientific implementation is committed.
Root owns log/map/index/README, allowlist retirement, Git and cadence; this r2
writer owns only this contract trio, assigned main sections and N09.

Static checks record JSON/links, exact source-symbol partitions and exports,
annotation premises, designed DAG, private-consumer coverage, and main/budget edit
whitelists. No runtime suite, build, source/wheel matrix or benchmark ran here.
The full mapped Stage6 selection (23 existing technical files and three native
files), connected builder/commit/source/UID/H/conformer tests and new extraction
contracts must join Python3.10–3.14 source **and installed-wheel** selection before
release. Root's validation-preparation payloads are future commands, not passes.
Keep exact source/wheel/native/interpreter/argv/results and ordinary-skip counts;
recipe/header cases remain source-only, external SDK ABI uses its frozen prefix.
Old binaries/content audits do not prove a fresh build or SDK/sdist acceptance.

Revision2 review and its matching actual audit are historical inputs; root key
acceptance is separately recorded. Remaining gates are exact production/test
ownership release, preserved pre-cutover runtime/transport/cost baselines, current
`SOURCE_READY`, affected runtime/native/wheel matrix, the bounded cost gate with
explicit adverse-result dispositions, and maintained Stage6 report/gate. Stage7E remains
a later mandatory blocking ownership gate. Global Full71/GPU/RSS/end-to-end,
full187/O0/seeds42–44 and integrated Plan12, eight comparable performance cases
with adverse findings retained, Stage10 clean-native/SDK/sdist, Stage12 global
architecture/types/API/docs/performance, and committed B_exp → separate D_exp →
I_exp ordering remain open. No missing or unfavorable gate is waived.
