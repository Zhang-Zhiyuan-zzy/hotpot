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


## PC1–PC9 — Connected selected partial-charge cutover

Root Sol/max accepts this internal design on2026-11-03 against delivered
`a54c660d8bff79462dd1cee48d0d3c9e65a53d87`, effective goal and Stage1 A2/A6/A8.
Two local Sol/medium read-only scouts verified actual transport/lifecycle and
adapter/native/model consumers. This extends the already approved Stage7C
objective; no new public export/function, scientific algorithm or model feature
is introduced. Implementation authoring is released below; runtime awaits the
connected source-ready freeze. WholeStage7C/global acceptance stays open.

### PC1 — One payload, one applicability owner

Spend reservedN14 only, defining private-to-the-package `PartialChargeSet` in
`structure/_results.py` with these ordered fields:

`atom_uids, values, method, method_version, parameters, units, provenance`

All fields are immutable: ordered integer subjectUIDs; parallel optional-float
values (`None` is explicit row absence); method string; known implementation
version orNone; tuple of `(name, optional-string)` effective parameters; unit
`"e"`; and parallel existing integer Atom producer codes. SubjectUIDs plus N15's
source bind the complete owner/order. This is an internal scientific payload,
not an ordinary return envelope or a second per-Atom fact. No confidence/status
field or new type family. Use sole slot `_partial_charge_result`, reusing N15,
`_RESULT_ATTRIBUTES` and existing native/Python publication/Undo.

Actual calculations name the requested OpenBabel model, known OB release version
and effective model_args (includingNone). Imported values with no known original
calculation method are honestly `supplied`, versionNone, with preserved per-row
source codes; never infer a method, presence or certainty from numeric zero.

### PC2 — Source and finite invalidation

Capture `charge_representation` and `geometry`, never selected charges themselves.
The private charge representation shares the existing representation content
without unused spin, as MCA already does; retain existing other domains. Map
ordinary representation mutations to both private result domains; charge
selection itself retains MCA. All existing representation/geometry native and
Python ingresses invalidate charge, including signed-zero/local metadata changes
when either result exists. Spin, labels, constraints and annotations retain it.
Do not add per-getter hashes, revision counters, frame backreferences or native
ABI/classes. Existing generic native configuration accepts the added slot and
read-only derived charge field; rebuild only if native source actually changes.

### PC3 — Direct, numeric and task projections

Direct `Atom.partial_charge` is read-only Optional[float]: absenceNone, current
value including0.0, staleRuntimeError naming dependency/recalculation. Check owner,
finite marker and O(1) current membership/UID; no model/hash work on read.
Detached/noncurrent Atom handles have no selected molecular charge and readNone.
The fixed19-column row/matrix projects unavailable (absent orstale) charge asNaN.
Remove physical column8 storage by classifying it derived; keep19-column public
order and16-column canonical table addressing, with no second scalar store.

Provide strict `_current_partial_charge_result(mol)` and optional
`_partial_charge_projection(mol)` (ordered optional values; absent/stale→None
rows, foreign owner remains an integrity error). Use the latter for fixed views,
optional toolkit export, MCA normalization and documented model/FF task encoding.
General `pack_molecule(...PARTIAL_CHARGES...)` uses strict current selection and
returnsNone for absent/incomplete sets; unrelated field requests never consult
charge. OpenBabel/FF task packing explicitly encodes missing/unavailable rows0.0
and preserves current supplied rows. The trained first15fields+metal task keeps
its existing layout and missing0 convention. These task zeros are never imported
as facts or written back. No public NaN reaches finite native/model inputs.

### PC4 — Explicit calculation and atomic selection

Keep ordinary get/assign signatures and tuple returns. `get_partial_charge` stays
an explicit nonselecting model operation, with its existing actual model/count/
finite-output contract. Assignment prepares aromatic inputs under owner lock,
captures charge source and orderedUIDs, releases the fresh lock, invokes the
existing get operation once, then source/membership-checks and publishes one
complete payload in one factual transaction. Remove the old outer whole-method
transaction wrapper for assignment. No per-Atom setter loop, model under fresh
selection lock, retry or broad exception fallback. Failed selection publication
restores its own writes and the prior exact attachment. Completed aromatic
preparation and independent mutations retain existing operation side effects;
a failed external calculation does not unwind them. A caller-owned outer
transaction restores all enclosed factual changes and the exact attachment.
This clarifies the existing preparation boundary rather than adding a rollback
of concurrent work.

Private APIs for connected writers are:
`_capture_partial_charge_source(mol)`,
`_select_partial_charge_result(mol, payload, source, atom_uids)`, and
`_import_partial_charges(mol, values, *, producer, method="supplied",
method_version=None, parameters=())` for already supplied complete ordered sets.
The import operation captures/checks the current owner without any calculation.

### PC5 — Explicit internal record schema, no hidden side channel

MoleculeRecord23/tagged-v2 cannot carry method/version/arguments or stale markers;
its bond metadata and nine frame slots have fixed unrelated semantics. Add one
keyword-only field `partial_charge_result=None`, containing eitherNone or
`(PartialChargeSet, invalidated_by_tuple)`. Bump its private tagged pickle codec to
schema3/24fields and reject old v2 explicitly, retaining existing older-format
rejections; no compatibility reader or public export is added.

Record payloadUIDs are canonical row0..N-1; install explicitly rebinds them to
actual targetUIDs/source once, retaining stale markers. At record construction,
legacy supplied input buffers' charge values/presence/source are moved once into
a `supplied` payload; clear charge presence/source in Atom buffers and replace its
unowned display bytes withNaN. A record with an explicit result cannot also own
raw supplied charge bits. Factory and dataclass.replace paths share this codec.
No values persist in both result and table/record scalar columns.

### PC6 — Copy, subsets, merge, working-copy and persistence

Exact copy preserves values/method/parameters/provenance/current-or-stale state,
remappingUIDs/rebinding applicability once. Current copy validates its original
source once; stale never becomes current. Factual capture transports the result
explicitly. Graph/row subsets project matching historicalUID values and sources,
fill genuinely unobserved subjectsNone, and mark representation stale whenever
the full original graph/order/context is not retained. No fragment recalculation
or claim that whole-molecule charges are current for a new context.

Nonempty merge does not invent a mixed-method full result. Retain the receiver's
historical selection asstale; the incoming owner keeps its own observations.
An empty receiver may rebind the complete incoming record. All count-changing
record replacements explicitly address the new field. Working-copy publication
transports its selected record/marker atomically; rollback restores original
slots after native-cell restoration. Current/stale pickle/deepcopy use existing
extra-state owner sharing plus the explicit record. Detached reads store no
separate charge scalar or attachment.

### PC7 — Toolkit and DTO maps

RDKit `_GasteigerCharge` property presence maps to row presence; export omits
unavailable rows. OpenBabel import checks molecule-wide perceived-charge state
before any `GetPartialCharge`, preventing implicit perception on absent inputs.
A complete supplied OB set exports as perceived. For partial Hotpot sets,
preserve supplied per-row values through a documented adapter-private OB data
presence map, so default missing zeros cannot be reimported as facts; do not
claim OB itself offers row-presence semantics. That map is transport metadata,
not a scientific confidence label, and must survive the exercised OB copy.
Do not calculate charges merely to convert. CCDC absent values remain absent.

MCA AtomDTO's existing charge field becomes optional; serialization/reconstruction
preserveNone versus explicit0 without changing tensor/model/site algorithms.
The DTO carries observations, not guessed source truth. tmQM builds topology
first, then imports the complete supplied `.q` set once; remove prohibited Atom
constructor charge kwargs. Conformer charge arrays remain Stage7D frame data.

### PC8 — Connected evidence

Before runtime, freeze all connected source/test/native bytes. Test direct and
19-column absence/zero/stale permissions; complete selection drift/failure/
rollback; native scalar/fields/columns/Undo invalidation; copy/subset/merge and
record3/pickle; imported per-row presence and provenance; RDKit/OB/DTO roundtrips;
field-selective packing and finite task-only zeros; unchanged model feature
shape/values; both tmQM paths. Use actual CP311 OpenBabel charge calculation and
existing native FF/pipeline/MCA cases, source and installed boundaries. Reuse
unchanged native/artifact evidence honestly; retain failures, units and actual
values. Getter/call-count evidence must exclude per-Atom dependency hashing.
Broader versions/clean-native/global work remain indexed, not passed here.

### PC9 — Source authoring partitions and delivery

A owns `_results.py`, `_fields.py`, `_builder.py`, `_identity.py`, `_commit.py`,
`core.py`, `operations/chemistry.py` and new selected-charge owner/lifecycle tests.
B owns `operations/conversion.py`, RDKit/OB conversion adapters, CCDC converter
and charge/adapter operation tests. C owns common/native/FF task packing, MCA
serialization, trained data extractor, tmQM producer and their focused tests.
All share this settled interface/design; no overlapping writers, nested agents,
production runtime before integrated release, or independent gate decisions.
Root owns plan/records/integration, source-ready/validation and Git delivery.
Future titles use `type(P20-S7C-cheminfo): ...` for this business slice, or the
actual stage/module/type for later work, as directed by the owner.
