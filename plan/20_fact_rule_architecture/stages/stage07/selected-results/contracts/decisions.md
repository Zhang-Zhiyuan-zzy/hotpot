# Stage7C selected results: ownership and first connected slice

Root decision: accepted implementation route, 2026-11-03. Stage7C is IN_PROGRESS;
no Stage7C or global gate is closed. This design follows the active goal and
Stage1 A2/A6/A8; it introduces no public function, result envelope or confidence.

## Objective and sequence

Complete Stage7C through (1) selected MCA attachment/currentness and (2) selected
partial-charge ownership with connected import/native/model projection, then
whole-stage validation. The first coherent code slice implements MCA plus the
shared finite attachment/transaction mechanism. Atom partial-charge storage is
transitional until the second slice; completing MCA does not accept Stage7C.
Coordinate frames remain Stage7D and canonical H/isotope/stereo storage Stage7E.

## Settled public behavior

- `calculator.mca(mol, *, backend, **predictor_options)` returns the same existing
  immutable `MoleculePrediction`; its three existing DTO types/fields stay fixed.
- `Atom.mca` and `Molecule.mca_sites` retain scalar/dictionary values and units.
  Absent access raises AttributeError naming the explicit MCA operation. Stale
  direct access raises RuntimeError naming the changed dependency and that
  operation. No getter starts a model or toolkit. Returned DTOs remain historical
  immutable values after mutations or replacement.
- Remove calculator writes to Atom `_mca`, Molecule `_mca_sites` and the reserved
  `properties["mca"]` storage together. User annotations remain annotations;
  the ordinary properties mapping has no scientific-currentness authority.
- Stage1 already accepts absent charge None, stale direct charge RuntimeError,
  fixed19-column unavailable charge NaN, explicit zero retained and complete
  assignment as the sole public write. Those take effect in the second slice,
  together with producer/import/native/model migrations; no early data deletion.

## One attachment and finite validity

Spend only reserved N15, a private frozen generic dataclass in
`structure/_results.py`:

`_ResultAttachment(payload, source, invalidated_by=())`

`payload` is the existing domain value, `source` the existing `_SourceIdentity`,
and `invalidated_by` an immutable tuple of subscribed domain names. Absence is
None at the named molecular slot. An empty invalidation tuple means CURRENT;
nonempty means NEEDS_RECALC. No extra mutable status, generation/revision counter,
content cache, per-Atom result, general attachment registry or new exception.
Use `_mca_result` as MCA's sole selected slot. Private helpers capture/check source
once around calculation and return/create replacements; getters only inspect
owner identity and the finite attachment marker, then index current rows.

MCA subscribes to private `mca_representation` and existing `geometry` identities.
The MCA representation reuses the existing representation content except its
final spin entry; existing `representation` encoding and consumers stay intact.
Geometry is a real input: the actual conformer route retains valid input3D,
otherwise ETKDG/MMFF/UFF generates it. Supplied partial charges and spin are not
read by MCA feature/site rules. Ordinary representation mutation events mark the
MCA domain; charge declaration marks it, spin-only declaration does not. No new
record, public domain/API, accuracy label or general dependency registry is added.

### R1 source-review amendment — exclude non-input spin

The initial source-only assignment used the coarse existing representation key.
Connected inspection showed that direct `mol.conformers.clear()` erases legacy
selected-frame spin without an owner callback, and exact whole-copy omits those
frames. MCA consumes neither declared nor frame spin. Root therefore narrows the
private applicability key rather than adding speculative frame backreferences or
pulling Stage7D into this slice. Spin-only declarations, selected-frame changes
with unchanged current coordinates, owned or direct frame clear, and whole-copy
must retain MCA. Coordinate or actual representation changes still invalidate.
Tests must also prove the existing representation key continues to detect spin.
The initial design/assignment hashes remain historical in the release record;
this amendment precedes any candidate build/runtime.

## Complete mutation map for MCA

| Successful upstream action | MCA consequence | Owning ingress |
|---|---|---|
| Atom/Bond add/remove/reorder/endpoints/full replacement, direct Atom registration | mark representation/geometry (topology is included) | topology commits, complete factual install, direct registration and existing topology marks |
| Element/formal-charge/H/aromatic value, presence or producer change | mark representation | common native scalar/fields/columns/prepared-update publication |
| Current x/y/z value, presence or producer change | mark geometry | same native publication; conformer load uses it |
| Bond order/kind/direction/aromatic/source/metadata | mark representation | Bond and coupled aromatic commits |
| Isotope or charge declaration or explicit stereo | mark MCA representation | sparse/charge-declaration/stereo commits |
| Constraints, cell, id/labels/annotations, supplied partial charges, frac_valence, spin | retain | no subscription to these inputs |
| Frame storage with unchanged selected coordinates/representation | retain | Stage7D owns later frame redesign |
| Explicit result selection | check captured source and atom mapping, replace one slot atomically | MCA calculation/selection operation |
| Failed or poisoned transaction | restore exact previous attachment with other facts | same native/Python journals |

Controlled writes mark finite subscribed groups without whole-molecule hashing.
Native table dispatch bypasses the Python wrappers, so its common preparation/
publication/undo path must cover scalar, fields, columns and prepared updates.
For native numerical writes, compare existing before/after value bits, presence
and source for actual changed fields; a true no-op need not invalidate. Named
coarse representation/topology commits may mark their subscribed domain without
hashing to prove mathematical equivalence. Private scratch table ingress without
a molecular owner does not select or invalidate a molecular result.

## Native/transaction integration

The existing native `_configure` receives two private bindings:
`result_attributes=("_mca_result",)` and
`prepare_result_updates=_prepare_atom_result_updates`.
The helper signature is `(mol, changed_fields, transaction)` and returns a tuple
of `(attribute_name, before_attachment, after_attachment)` triples. It makes no
fact writes; when an outer transaction exists it first enrolls the original
selected slot with `transaction.touch_results()`.

Native `prepare_undo` determines actual changed field names and, only when an
attachment exists and a dependency-bearing molecular write occurs, prepares
those triples. Existing Undo installs replacements and restores exact previous
slot references on failure, including when there is no Python transaction. Outer
journal rollback also restores its original selected slots after native cell
restoration; this covers native invalidation followed by a later selection in
the same transaction. Use existing Undo containers rather than new native record
classes or ABI structures. Add a precise `result_invalidation` failure boundary.
No external/user callback or chemistry calculation enters this hook. The native
source changes require a fresh CP311 `_fact_native` build and source/installed
execution; historical six-native evidence cannot prove this candidate.

Python non-table commits call `_invalidate_results(mol, domains, transaction)`
with the same finite replacement policy; they enroll `touch_results()` before
publication. Slot snapshots participate in topology, nested transaction, direct
registration, failure injection and later result-replacement rollback. Do not
wrap or replace native dispatch with a second Python mutation implementation.

## Calculation and row binding

Normalize the Hotpot molecule exactly once using the existing MCA normalization
under its current molecular lock. That normalization prepares the existing H/
aromatic inputs and creates the existing DTO; capture representation/geometry
after this preparation (using the MCA representation key). Release the lock, call `predict_mca` once with that DTO,
then enter one factual transaction, check the captured source and atom mapping,
and select the exact returned payload. No model executes while a fresh result-
selection transaction holds the lock. A concurrent/reentrant source change makes
selection fail while keeping the prior selected result/other facts coherent.

Validate complete ordered atom prediction indices/element bindings and site
membership once at this explicit ownership boundary, before selecting. These
are mapping/integrity contracts, not speculative NaN/Inf checks. Do not add a
finite MCA scan, retries, defaults or broad exception handling. Direct Atom
access uses existing O(1) current membership/index lookup; site reads build a new
dictionary over current handles from the same payload. Scalar/all-Atom reads
perform zero `_capture_source`/`_check_source` calls after selection.

## Copy, components, persistence and annotations

- Exact whole-molecule copy preserves the immutable payload by reference and
  rebinds its source once to the copied owner/current row mapping; stale markers
  remain stale. Only this named copy route rebinds an existing result.
- Component/selected-graph projection drops the selected MCA attachment because
  the prediction belongs to the whole original graph/context. Merge/removal and
  other source changes make a previous selection stale rather than guessing a
  fragment prediction. No recalculation occurs.
- Current-version pickle/deepcopy retain the payload, marker and owner/member
  relationships using the existing molecular extra-state transport. The actual
  raw restore passes the serialized owner token back into the restored Molecule;
  source-token sharing must be tested, not replaced by a speculative blanket
  rebind. Current stale state survives. No old scalar-store compatibility reader.
- Copy/user annotations retain existing isolation behavior; selected-result logic
  never interprets arbitrary annotations as results or copies a scalar store.

## Remaining partial-charge slice

Reserve N14 for one immutable selected `PartialChargeSet`, with complete row
identity, explicit presence, charge values/units, named method/arguments and
provenance. Confirm exact fields with current import/native consumers before
writing that record. Reuse N15 and the finite native hook; do not create an MCA
wrapper DTO, extra validity owner or charge cache. Remove physical Atom scalar
charge storage only with all private import/persistence/projection writers.

Native charge requests must distinguish selected absence from explicit zero and
must not feed public NaN absence into native paths requiring finite values.
Document task-side zero encoding only where it preserves existing model/tensor
or registered adapter conventions, without creating authoritative charge facts.
Importers preserve genuine supplied charge presence and method/source provenance.
Conformer charges remain frame data. No guessed import presence from numeric zero,
new science, native return write-back or trained-feature change is authorized.

## Validation and release boundary

Start with focused positive/negative/currentness/transaction/lifecycle tests,
then affected CP311 source and installed-native qualification and actual CPU MCA
mixed-input continuation. Include all native field write forms, native failure
with/without outer transaction, selection after invalidation then rollback,
source mutation during prediction, exact copy/components/merge/drop/pickle,
annotation independence, no model on access and zero per-Atom source hashing.
Use existing actual models/parameters and retain failures and denominator3; no
scientific golden-value/parity claim. Add bounded cost/call-count evidence for
cheap scalar writes and repeated scalar/site reads, then whole-Stage7C gates
including the partial-charge slice. Other Python/native delivery/global tiers
remain explicit later obligations. Root reviews/commits/pushes coherent business
slices; no documentation-only completion or later-stage release follows.
