# Stage7D coordinate-frame ownership

Root Sol/max route decision, 2026-11-03, against delivered
`eb68f98f136755ac84e57166e1f8090c5d7a8e57`. Stage7C's applicable CP311 owning gate
is accepted; Stage7D/E and global gates remain open. This implements the existing
Section5.3/5.5, Decision19 and Stage1 conformer contracts, without a new public
export, scientific algorithm or trained-feature meaning.

## F1 — Owners and unchanged public operations

Current Cartesian coordinates remain solely in the Atom table, in angstrom.
Molecule owns `_source_frames` (immutable tuple of N16), `_conformer_results`
(immutable attachment-order tuple of N15 containing N17), `_frame_result`
(optional N15 selected observation), and the existing selection cursor.
`Conformers` bound to a Molecule is a projection; it stores no assembled arrays
or result mirrors. Preserve its existing import/class/pickle spelling and all
existing read/add/clear signatures. A standalone `Conformers()` remains a
caller-owned source collection using the same pure frame preparation/projection;
it does not create molecular facts or publish generated results. This is the
existing standalone domain, not an old/new runtime or compatibility switch.
Bound direct `.add/.clear` delegate to the Molecule's named transactional
operations. Only a standalone instance owns its independent source tuple and
lock; a bound projection holds its owner and no duplicate frame store.

Preserve `conformer_add(coords=None, energy=None)`, `get`, `load`, `clear`, count,
coordinate iteration, and their established return/shape contracts. Source
insertion order precedes ensemble attachment order and original member order;
no sort/dedup/second persisted array. Integer/slice dictionaries always have the
nine existing keys. Absent scalar/per-Atom observations returnNone, explicit zero
survives. In batch numeric projections missing rows areNaN; each integer access
still distinguishesNone from zero. Empty iteration is empty. Prepare complete
coordinate/energy shapes before publication; append energy correctly and retain
mixed present/absent frames without invented zero. Invalid shape/count failure
is atomic. Adding a source never selects it or changes current coordinates. When source
rows are appended before generated ensembles, rebase the existing cursor to keep
the same selected member; preserve positive/negative cursor spelling when the
collection does not change. Whole copy must retain a still-current selected
source observation after such insertion.

Existing direct assignments to the seven frame-observation names (`energy`,
`partial_charge`, `gibbs`, `force`, `zero_point`, `thermo`, `capacity`) retain
assignment/read behavior, but route to selected typed observations at the current
geometry rather than shadowing in `__dict__`. Use a closed descriptor/assignment
route, not a generic result registry. Coordinates/spin keep their established
coordinate/declaration operations. Known producers should publish their complete
observation set once; source attribution for previous fields must survive a
single-field replacement. User annotations remain separate.

## F2 — Exact record budget

Use only N16/N17/N18 and existing N15/N01/N02. All nested numeric buffers must
be independent and immutable (including inability to re-enable writeability).
Reuse the existing closed `_freeze_value` codec for nested observations/metadata;
getters return independent decoded values. No enum, Protocol, generic metadata
framework or additional owner/DTO class.

- N16 `CoordinateFrame(atom_uids, coordinates, observations, observation_index,
  provenance)`: one source coordinate value `(N,3)`, ordered subjectUIDs, an
  optional N18 observation record with row selection, immutable source metadata.
  The current Atom block is never retained here as a live alias.
- N17 `ConformerEnsembleResult(atom_uids, batch, observations, input_source)`:
  compose the existing `ConformerBatch` with sealed `(K,N,3)` coordinates and
  original ordinals; optional N18 holds frame observations. `input_source` is
  historical N01 input provenance, separate from output applicability. Do not
  duplicate the batch in per-frame stored CoordinateFrame values.
- N18 `EnergyForceResult(atom_uids, values, frame_indices, method,
  method_version, parameters, units, provenance)`: immutable eight-slot values
  in the established non-coordinate key order, encoded by the existing closed
  codec. `frame_indices` explicitly maps each frame/field to its stored raw
  observation: integer row, `whole` for scalar/calculation-scope input, orNone
  for unavailable/unmapped. This permits one shared observation record per
  imported/generated batch, with no energy/force copies at selected accessors.
  Metadata documents effective producer/method, parameters/seed where known,
  field units and scopes; per-field replacement retains existing attribution in
  provenance rather than relabeling other observations. No inferred seed/unit.

Selected `_frame_result` payload is the immutable `(N18, observation_index)`
reference. It duplicates no numeric observation. Current scalar/array getters
inspect finite markers, not molecular hashes per Atom/read. A getter never
starts a calculation.

## F3 — Applicability, selection and transactions

Add private `frame_representation`: existing representation content with explicit
spin declaration, excluding selected-frame spin. Ensemble output applicability
subscribes to it, cell and constraints, excluding starting geometry and source
frames. Preserve full actual calculation input capture separately in N17.
Committed ensemble history must not invalidate itself when final coordinates
are selected. Representation/cell/constraint changes mark affected ensembles;
loading a stale generated member raises a named currentness error. Its immutable
historical payload survives. Source coordinate values remain attributable facts;
source observation/currentness must never be confused with recalculation.

`conformer_load` resolves/validates under the owner transaction, commits only
current coordinates and selection, then binds the chosen N18 row to final
geometry/representation. It never writes Atom charges or net/spin declarations.
Selected energy/force/charge/thermo observations become stale on actual geometry
or representation change, absent after clear, and current again only through
explicit frame selection/result publication. Explicit spin declaration wins;
selected frame spin remains a source state observation and does not become an
energy result or make representation hashing recursive. Internal identity reads
must inspect declared/selected historical spin without invoking a stale-energy
getter. All nine public projection values remain historical frame observations;
strict current molecular projections separately enforce final-frame validity.

Enroll source tuple, ensembles, selection and selected observation in existing
FactTransaction undo; native scalar/field/column/prepared-coordinate publication
must use the existing result-update callback. Extend its finite known slots and
closed ensemble tuple handling. F3-C1 below permits the necessary native callback
repair; no native ABI, schema, numeric encoding or scientific algorithm changes. Failed
selection/publication restores exact prior owners/attachments/cursor; completed
external work and independent mutations keep existing failure boundaries. No
retry, speculative guard or broad fallback.

**F3-C1 — Observed constraint callback gap, 2026-11-03.** The first focused
run returned323passed/3failed/1skipped. A real failure showed that Atom constraint
changes did not invalidate generated results. Inspection of native
`prepare_undo` establishes that its constraint branch never records changed
fields and its chemistry-mask early return precedes the existing result callback.
The root authorizes recording actual Boolean constraint changes and preparing
existing immutable result updates before that return. Presence/producer changes
with the same Boolean constraint remain applicability noops. Existing native undo
owns atomic install/rollback; add no parallel Python invalidation path. Rebuild
and attribute the exact CP311 fact extension before revalidation; reuse only the
other five unchanged natives. This corrects an implementation assumption within
F1–F7, with no public/ABI/schema/scientific change or relaxed gate. Original
failed evidence and pre-repair freeze remain retained.

`source_frames` identity includes explicit source values and provenance only;
generated attachment cannot change it. Legacy forcefield input capture must
inspect canonical source/selected observations, without assembling a persistent
all-frame matrix. Existing source mutation/concurrency checks stay strict.

## F4 — Lifecycle and internal transport

Whole copy preserves/remaps source P1 frames with isolated values, drops generated
P3 ensembles/runtime and their selected observations. Components subset source
coordinates/per-Atom fields through UID mapping. Whole-system observations do
not become fragment energies: retain original source evidence in provenance,
leave unassignable fragment fields unavailable. Current coordinates stay copied
facts. A copied selected source can rebind only when its coordinates/context
match; otherwise no false current observation.

Membership changes preserve attributable source rows under UID mapping; never
invent coordinates for newly added atoms. A source frame lacking complete current
membership cannot load as a complete current frame; report that exact local
mapping failure. Bond-only changes retain source coordinates. Generated outputs
mark representation stale or are dropped by the specifically approved
copy/component policy. Empty-receiver merge may rebind the incoming source
collection; nonempty merge retains both source collections in insertion order
with their explicit subjectUID subsets, rather than fabricating a Cartesian
product or coordinates for missing subjects. Their complete-frame projection/load
requires complete membership. No shared mutable containers.

Replace N02's `legacy_frame_values`/`legacy_frame_index` with explicit
`source_frames`, `conformer_results`, `frame_selection`: same class25fields,
private tagged-v4 codec, explicit old-v3 rejection. Canonical record subjectUIDs
are rows and install rebinds once. Source/generated roles, observation scopes,
stale markers and selection survive record/pickle. Capture frame owners once;
remove the separate legacy nine-array pickle mirror. UID restoration after pickle
must rebind canonical records to the restored realUIDs. Reuse complete existing
installer/working-copy undo, preserving original Atom/Bond handles and high-water
IDs. Do not overwrite preexisting source frames when publishing an accepted FF
working copy; remap surviving source facts explicitly across any addedH prefix.

Private raw producer import is `_import_frame_values(mol, values, index, *,
producer, method=None, method_version=None, parameters=(), units=(),
frame_indices=None)`, where values are the nine existing domains. It constructs
canonical source owners once; no retained legacy array/compatibility path.
Canonical owner capture/install helpers transport the three explicit N02 fields.
Known direct raw nine-slot test fixtures migrate to this producer seam; private
writable array mutation is not preserved as a second truth.

## F5 — Connected producers

FF trajectory materialization publishes N17 as generated output, appending to
prior ensembles and preserving source frames. `keep_all` keeps the same selected
or all original frames, order/energy units and ordinal meaning. Do not turn missing
trajectory energy into a scientific zero/NaN observation; retainNone internally
and useNaN only in numeric batch display. Keep the existing topology-aware
trajectory/archive authoritative for changing bonds; the conformer projection
remains coordinate-only. `_finalize_trajectory` supplies actual operation input
source and existing method/seed options, never invents them. Publish final
coordinates, ensemble and selected observations atomically on working and original
owners; source drift before original commit still rejects. Preserve failure
archive-before-publication behavior and all seeds/backend/scientific algorithms.

Gaussian source import retains all raw observations and their known scope.
One-dimensional Mulliken charge is observed only at the last frame: earlier
rows are unavailable, never padded with measured zeros. Equal-length histories
map by supplied order; scalar calculation-scope values retain their documented
broadcast. Unequal coordinate/observation histories remain retained raw N18
observations but have no fabricated frame association; report a precise
`frame_observation_unmapped` warning naming field/counts, and keep usable
coordinate input. Preserve known eV conversions and existing force units without
inventing new conversions. This addresses D011's concrete B4 case; other D011
items keep their own dispositions. Frame partial-charge arrays do not implicitly
select Atom charges.

Both tmQM producers attach their known energy as an explicit attributed current
observation after topology/geometry; full supplied charge import remains once
and preserves trained/model encoding. The legacy Gaussian updater's known
scientific assignments must use one complete observation publication, followed
by explicit complete Mulliken selection after geometry/declarations where supplied;
retain unsupported/orbital attrs separately and original failure causes. Actual
existing-log update/import evidence is needed before closing D016; parser-default
or independent Gaussian-engine issues remain separately indexed.

## F6 — Private seams for implementation partitions

Owner module `structure/conformer.py` defines records, projection and helpers:
`_frame_records_from_values(atom_uids,values,index,*,producer,method=None,
method_version=None,parameters=(),units=(),frame_indices=None)` is the pure
nine-domain preparation seam for pre-installation decoders: it returns canonical
`(source_frames, conformer_results, frame_selection)` without a live Molecule.
`_select_frame_observations(mol,values,*,producer,method=None,
method_version=None,parameters=(),units=(),provenance=())` selects one complete
eight-domain non-coordinate observation set for already committed current facts,
without adding source frames. Single-field assignment uses that same owner and
replaces only its field attribution; keep bounded per-field source metadata, not
an ever-growing chain of predecessor results.

Other private seams are
`_bound_conformers(mol)`, `_add_source_frames(mol,coords,energy=None)`,
`_clear_frames(mol)`, `_load_frame(mol,index)`, `_import_frame_values`,
`_current_frame_value(mol,name)`, `_assign_frame_value(mol,name,value)`,
`_capture_frame_records(mol,rows=None,include_generated=True)`,
`_install_frame_records(mol,source_frames,conformer_results,frame_selection)`,
`_copy_source_frames(mol,clone,rows=None)`, `_frame_transaction_state(mol)` /
`_restore_frame_transaction_state(mol,state)`, `_frame_spin(mol)`,
`_source_frame_inputs(mol)`, `_frame_numeric_inputs(mol)`,
`_publish_conformer_ensemble(mol,batch,values,selected_index,*,input_source,
method,method_version=None,parameters=(),units=(),provenance=())`.
Private implementation details may be simplified without changing these
responsibilities/public behavior; coordinate any interface correction with root
before dependent edits. Reuse N15 and delayed imports to avoid builder/core cycles.
Core retains only facade calls and the public Conformers spelling. Record helpers
canonicalize/remap source and generated payloadUIDs in one implementation.

## F7 — Validation and completion

Before runtime root integrates every partition, freezes affected source/test and
six CP311 natives, and releases focused-first execution. Required cases cover
source/generated ordering, standalone/bound add/clear, empty/mixed energy and
atomic malformed-input failure, nine-key integer/slice isolation, source add
without selection, exact load/clear/publication rollback, generated member
selection without self-invalidation, stale final energy on native-coordinate
writes, actual source mutation rejection, partial UID completeness and source
copy/component/merge, role-preserving record/pickle, retained source frames through
FF/H publication and unchanged seed/provider behavior. Test Gaussian absent/zero/
last-only/unmapped observations and real existing-log import/update without an
engine; preserve actual values, scope findings and denominators.

Then complete affected CP311 source and installed selections, actual FF,
CBond→FF→GFN-FF/xTB and MCA, with actual scientific observations and bounded
frame/ensemble allocation/read/publication costs. Reuse unchanged native/model
artifacts honestly; no rebuild claim. Unchanged successful evidence may be reused
only with explicit scope. Stage7E still blocks Stage8; broader versions, clean
native/package/science/GPU/global efficiency and Plan20/Plan12 acceptance remain
open. No stage closes merely on static code or a selected small passing test.
