# Stage6 extraction decisions — revision 2

Recorded date: 2026-10-30. Author: assigned GPT-6.1 Sol/max. This is the owning
Stage6 decision record. Root separately key-accepted the exact contracts below
after the historical revision2 review/audit. This final root integration is not
revised reviewer acceptance; production/test ownership remains HOLD pending exact
release. The [design](review.md) and [evidence](evidence.json) own the full
partition, sequencing and unexecuted validation obligations.

## Review identity and actual audit limit

Task `p20-s6-contract-review-704283-34`, revision1, verification704283:

- Task hash: `21ad8134d66f80c4f8df735a6f46e5c6f0f377a8bf3e370527ebf7b10df43698`.
- Source checkpoint: `90ff36c2c0b415bb9cf27628265bb4528e425922`.
- Canonical capture SHA256: `1bed097223b51033a780f5dc1ae452ffcdc01761bb0e4e52e23dc7ccad33b39f`;
  this is a capture-manifest digest, not a Git tree.
- Review result SHA256: `40ae276830957e41f409ede01727d2478d4a7d063115588f073f7f6d728be354`;
  status `BLOCKED_WITH_REQUIRED_DESIGN_DELTAS_AND_ROOT_DECISIONS`.
- The matching actual audit PASS attests the historical receipt/identity. It did
  not inspect current code or accept the design. Root retained the raw task34
  archive and [initial-review](initial-review/review.md) record. No repeated audit
  is requested by this revision.

## Revision2 reviewer provenance and separate root acceptance

The historical revision2 result remains unchanged:

- Task/hash: `p20-s6-contract-review-704283-34`, revision2,
  `2af2b832a8ea05daf98b30d47db18cdb48f9464f171f7bdad806dca87f44579a`.
- Result:6,721 bytes, SHA256
  `5b44416bfa79280d617f5032c72751ca750a61082c71e7796a0cb9dbde05f063`;
  status `BLOCKED_PENDING_ROOT_KEY_ACCEPTANCE_AND_IMPLEMENTATION_EVIDENCE`.
- Exact turn `01a18cc5-cec4-7b93-aabb-d0e509405603`; historical source
  `979455ed8bada61f4d44038935e7f02c1b204c86`; capture-manifest SHA256
  `3e2476a82e925f9c33683bdb1b10361cefff111b0d6a2ad8703fa2eaf238da82`,
  not a Git tree or a new checkout audit.
- Matching actual audit PASS: context1500, range1496–1538; read-only,
  network-restricted, zero tools/commands/delegation. This proves historical
  receipt/identity, not design, implementation or runtime acceptance.

The reviewer statically resolved six findings, retaining the two RuntimeError
root choices and a missing packing-cost disposition rule. Root then explicitly
accepted S6-R2-05 and07 with the exact triggers/messages/order/side effects,
N09's cap-one three fields and mandatory Stage7E. Root directed the now-frozen
S6-R2-09 adverse-disposition rule. No new review is requested or claimed.

Before these final edits, all five reviewed files were byte-verified against
root's frozen-document records and preserved in
`/tmp/p20-stage6-contracts-reviewed-r2-before-root-integration-20261030/`.
Its manifest SHA256 is
`daab249c51a592d940f5c4fe53cb30ee7def93a9a555e798810eb3c6ae175783`.
Evidence retains every original path/hash/byte count separately from current
integrated output hashes. This document does not overwrite the historical result.

## Root-directed scope and reservation changes

| ID | Direction/status | Concrete effect |
|---|---|---|
| S6-R2-01 | Root-key accepted reservation; implementation HOLD | N09 changes from unused `StereoUpdate` to exact `_StereoSnapshot(payload, atom_keys, bonds)`, cap1. No source binding/referents in the DTO and no duplicate update wrapper. Existing `_commit_stereo(mol, source, payload)` is reused. Measured counts and forecast cap arithmetic do not change. |
| S6-R2-02 | Root-key accepted normative owner/gate; implementation later | Section5.3 storage decomposition belongs to Stage7E, completed before Stage8 and Stage12/global acceptance. Stage6 `get(mol) is payload` is transitional, not a public storage guarantee or global waiver. |
| S6-R2-03 | Root-directed boundary correction | `_bond_key`, `_axis_bond_keys`, `_remap_stereo_payload` have one pure owner in algebra. Core/RD lifecycle calls enter the real chemistry operation and invoke that rule once. Structure owns access, not a second remap implementation. |
| S6-R2-04 | Root-directed sequencing constraint | Mapping callbacks occur before projection/H errors as today. Coordinate `__array__` remains after complete graph validation and global-issue suppression, outside stereo-owned scopes. Target noncoordination H/native preparation remains after reference native validation. |
| S6-R2-05 | Root-key accepted exact new RuntimeError contract; implementation HOLD | Narrow nonblocking tracked-outer-transaction route, with the exact trigger/error/side effects below. Preserve same-owner and uncontended/reentrant behavior; no blanket active-transaction rejection or general deadlock guarantee. |

## Accepted capture and concurrency contract

S6-R2-06 is the selected internal design accepted by root with S6-R2-05/07.
One immutable raw snapshot is captured per distinct owner. The operation retains
strong tuples of atoms, bonds in their three partitions and captured endpoint
objects. This closes transient `id()` reuse even when endpoints are changed later.
No rule receives a carrier, reference-holding DTO or stored mirror. Same-owner
alignment reuses raw/prepared inputs and deterministic projection/native/report
work, while emitting both original issue-prefix sequences.

Release every fresh stereo-owned molecular scope before acquiring another and
before native/user mapping/coordinate callbacks. Existing outer factual locks
remain caller-owned. A rule invocation's inputs are frozen; two molecules are not
presented as one atomic simultaneous read. Reference/target public diagnostic and
exception order remains the existing sequence documented in the review.

S6-R2-05's exact root-key accepted trigger is distinct-owner alignment entered with any
tracked outer `_FactTransaction` in the current `_TRANSACTIONS` context, including
an unrelated third owner. All its stereo-owned molecular acquisitions, initial
capture and deferred preparation alike, use nonblocking acquire. Same-thread
RLock reentry and uncontended acquisition succeed. On contention raise exactly:

```text
RuntimeError: Stereochemistry alignment cannot acquire a distinct owner while a factual transaction is active
```

The error is local to that attempted read operation. Do not spin, sleep/retry,
unlock the caller's transaction, reorder its locks, or blanket-reject active
transactions. The operation releases only acquisitions it made. No fresh stereo
capture/preparation scopes overlap; a successful nonblocking acquisition can
briefly coexist with a caller-owned outer lock. This is the explicit narrow
exception to a literal claim that a thread never holds two owner locks.

Static sufficiency is bounded to waits introduced for a distinct molecular owner
by these scopes. Inspected H/commit helpers reacquire their own molecular owner
reentrantly. Owner-only `_capture_source(mol, ())` and `_check_source` with empty
dependencies do not acquire a conformer lock. Nonempty dependency capture/check
paths used by existing H helpers retain their same-owner conformer-lock hierarchy.
This design does not alter that hierarchy or claim that
arbitrary callbacks, private RLocks, copied transaction contexts or monkeypatches
are generally deadlock-free. Native/callback work is outside stereo-owned scopes,
not necessarily outside caller-owned outer locks. Any newly discovered hidden
cross-owner wait requires a concrete follow-up design, not broader rejection.

## Accepted deferred raw-capture integrity contract

S6-R2-07 is a **root-key accepted new read-operation RuntimeError contract**;
implementation remains HOLD. Use existing owner-only `_SourceIdentity`/`_check_source`, plus an
adapter comparison of the exact raw payload/schema/order, atom identity/order,
bond partition identities/order, endpoint identities and raw bond values against
the retained capture. No fourth snapshot field or new identity domain/type is
needed. Recheck only immediately before required live preparation, after pure
preflight and before strict H. On raw drift raise exactly:

```text
RuntimeError: Captured stereochemistry facts changed; retry the operation
```

An owner-token mismatch keeps the existing owner-source error. This is an
intentional technical-integrity failure instead of combining incompatible raw
states. Root accepted this exact new read trigger separately from the historical
review. It must not be hidden inside a claim of exact exception preservation.

Unconsumed Z/charge/aromatic/readiness fields are selected at their original late
preparation point, then frozen; the raw guard does not bind them early. This is
simpler than a broad `representation` digest, which would add unrelated
aromatic/source/frame reads before preflight and potentially extra lock/hash
work. The guard checks partition identity as well as flattened rows, so an
active-to-hidden move cannot change H inputs unnoticed. Compare raw scalar
values type/byte exactly; do not depend on bool/int or signed-zero equality.
It constructs no second snapshot and makes no permanent source binding on a
read-only report. Existing Stage5 H proposal bindings remain unchanged.

A failure during initial capture precedes mapping/H/native work. A late error
comes at the original H preparation point: mapping callbacks, reference work or
a prior successful lazy preparation may already have occurred. No paired undo
transaction is introduced, and such completed preparation/caller changes are not
silently reverted. Existing outer-transaction exception/rollback semantics remain
in force. Once preparation succeeds its own legitimate changes do not trigger a
post-H raw rejection. No projection means no guard/prepared-field/H/native work.

## Actual introspection contract and corrected review premise

S6-R2-08 preserves actual current behavior. `stereo.py:3` already enables future
annotations; its only `Molecule` import is under `TYPE_CHECKING`. There is no
runtime Core binding in this module. The review's evaluated-annotation premise
is therefore corrected. Keep string `__annotations__`, full signatures/defaults/
keyword-only shape and runtime globals `Mapping`, `np`, `NDArray`; keep Core
type-only. Do not silently improve default `get_type_hints` by importing Core.

Default hints in a fresh current namespace are statically expected to raise
`NameError` for `Molecule`; explicit `localns={"Molecule": core.Molecule}` should
resolve. These are static expectations, not executed results in this task.
Capture/compare both source and installed-wheel outcomes before/after cutover.
A default-hints fix, if desired, needs a separate concrete public decision.

Define all18 public contracts once in `structure/stereo.py`; retain their public
`hotpot.cheminfo.stereo` module spelling by an explicit closed roster and preserve
qualnames/fields/defaults/annotations/enum order. Reexport the same class objects.
Cold import orders, class/instance pickle round trips and `_identity._canonical`
module/qualname tokens are runtime gates. No dynamic module aliases or private
facade exports are authorized.

## Type-sensitive native transport and bounded cost gate

S6-R2-09 selects deep read-only graph inputs and one plain-dict pack per existing
native target, zero per-pair/query repacks. This is a private representation
change with no native ABI/schema or matcher replacement. Required baseline
comparison is recursive type identity, ordered dict keys, container kind,
sequence order/length, scalar values/None and exact float bytes. Include real/
virtual labels, H slots, normalized fields and `object=None` at every nesting
level. Ordinary dictionary equality is inadequate.

Use frozen pre-cutover source/wheel fixtures bound to current native identities,
real native acceptance/mapping output, actual pack/constructor spies and existing
mapping-count expectations. Run the callable synchronously after release in
frozen pair order; do not retain it or add a capability framework. Preserve
tri-state filters and first-equivalent early return.

Record bounded capture/check/projection/freeze/pack/native time and Python peak/
retained allocation observations, plus RSS/native limits, on representative
existing chemistry and fixed16/128/512-real-row transport graphs. Record H
expansion/augmented V/E and equivalent prepared-H state. Main policy requires3
warmups and at least10 samples. Extra O(V+E) passes are real work; no speedup,
parity or global acceptance follows from this design or these bounded future
observations. The bad clean-read fixture remains non-comparable.

Root freezes this executable adverse-disposition rule **before measurements**:

1. Freeze identical semantic work/prepared state, fixtures and augmented V/E/H
   expansion, baseline/candidate artifact/native/interpreter identities, resources,
   instrumentation and sampling/noise policy. Report complete equivalent work
   plus components; new components without old counterparts have separately
   reported cost, not fabricated baselines. Apply the existing main Section12.1
   sampling/raw-data/median/MAD/p95/paired-bootstrap policy and Section12.2 noise/
   uncertainty policy without inventing final global thresholds.
2. Retain every raw observation and attempt, including adverse, failed, missing,
   inconclusive and excluded/non-comparable results and reasons. Missing or
   inconclusive required bounded evidence stays OPEN; unequal work or prepared
   state supplies no comparison or Stage6 pass.
3. Correctness/count/retention violations, unexplained scaling/repeated per-query
   work or unexplained adverse cost BLOCK Stage6 pending fix and repeat. Leaked
   carrier/graph/callable retention after completion is a violation; transient
   O(V+E) allocation is measured separately. Expected overhead never auto-passes.
4. Root must explicitly disposition every attributable adverse time/allocation
   result before Stage6 acceptance. Record its fixture/work/size, raw sample/
   allocation artifacts, median/spread/uncertainty and cause, followed by either
   fix/repeat or reasoned Stage6-local acceptance of measured necessary bounded
   O(V+E) overhead. Established unexplained/unresolved adverse findings remain
   blocked; missing/inconclusive comparisons remain open and cannot downgrade an
   established violation/adverse result. No adverse case disappears from the report.
5. Local overhead acceptance retains the result and all Section12/Stage12 global
   performance debt. It establishes no speedup/parity or Section12 waiver. No
   bounded measurement, disposition or Stage6 cost acceptance has occurred here.
   Close the local gate only with conclusive required evidence, matched compared
   work/state, passing correctness/count/retention checks and every adverse result
   fixed/repeated or explicitly accepted under the bounded terms above.

## Mandatory Stage7E canonical storage gate

S6-R2-02 now has concrete main-plan ownership, source budget, reuse and connected
gates. Stage7E owns the structure access/fact schema/table/mutation/dependency/
identity/builder/commit/native-fact engine, current carrier lifecycle, chemistry/
stereo operations, structure/native/toolkit/conformer adapters and force-field
working-copy publication/rollback. Entry freezes the sole surviving paths after
Stage7A/7B/7D through an exact rename manifest; later Stage8/9 implementation
stays outside that assignment. No old/new owner copies are allowed.

Delete persistent whole payload/mirrors. Reuse existing records/schema, source-
only ledger, canonical Atom isotope/radical/current-H values/presence/source and
separate Bond display versus dative direction. Preserve the legacy19-column
projection/native ABI prefix while migrating any private extension across every
writer/native configuration/undo path. Reuse N01/N02/N05, public transport and
current transaction families; zero new types, no second N09 wrapper.

Entry must resolve exact H-slot/scoping and unbound-source mapping before code.
Current transport `implicit_hydrogens` is output, not another H-rule input;
legacy site-total and toolkit omitted-H counts cannot silently become one
quantity. Any semantic normalization needs separate concrete approval. Preserve
effective radical/current-bracket/no-implicit H inputs and seeds. Opaque
schema47/row81 and duplicate/out-of-range atom-fact entries remain lossless and
keep current projection diagnostics, without invented target facts or a parallel
bound-current ledger.

Atomic set/get, `set(get(mol))`, copy/component/merge/remap/pickle/toolkit/native/
conformer/working-copy publication and failure rollback, schema/type/presence/
source/order and no-mirror tests form one gate. Replace transitional object-
identity storage assertions with canonical ownership and lossless equality tests.
Full affected source/installed-wheel Python3.10–3.14 and the current bound native
build are required. Until completion, Stage8 and Stage12/global ownership
acceptance are blocked; Stage6 extraction is not a storage waiver.

## Release and evidence limits

Only the assigned contract trio, main Section5.3/Section9 ownership rows/Stage6/
new Stage7E and N09 were authorized for this r2 writer. No production/test edits,
runtime tests/builds, new delegation or Git mutations are authorized here.
Root owns log/map/README/index/allowlist/Git maintenance and review submission.

The r1 and reviewed-r2 partition/audit evidence is historical. Final integration
checks are bounded static document, preserved-byte identity, decision/cost-rule
and exact edit-scope checks only; they do not rerun or expand the review. Runtime
import/pickle/concurrency/native/wheel/performance gates remain open. Root key
acceptance is recorded, while exact connected production/test release, preserved
pre-cutover runtime/transport/cost baselines and `SOURCE_READY` remain gates before
implementation validation. Full global science/performance/native/Stage12 and
B_exp → D_exp → I_exp gates remain unchanged, including adverse/unavailable data.
