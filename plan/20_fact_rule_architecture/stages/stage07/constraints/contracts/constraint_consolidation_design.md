# Stage7B constraint consolidation — concrete release candidate, 2026-11-03

Status: DESIGN PROPOSAL FOR ROOT REVIEW; production/test/build release is still held by `constraints/entry_review.json`.
Source inspected: `48780a669c9b1ca5b0e6c2cadf63365579ad3ccf`; this proposal changes no scientific rule or public API.
Effective `goal.md`, main plan and four normative companions remain governing; `goal_v2.md` remains review only.
This is the remaining Atom-axis/Bond ownership partition. Completed Angle/Torsion and pair-view behavior stays in scope only as a connected invariant.

## Selected sole ownership and physical layout

Extend the existing frozen `_ConstraintSet` with two private mutable builtin dictionaries, after the existing Angle/Torsion fields:
- `atom_axes: dict[int, bytes]`: one entry per current Atom UID. Each immutable 28-byte cell is 3 raw float64 values (24 bytes),
  one local presence byte (bits0/1/2), then 3 source-code bytes. Values use canonical little-endian IEEE754 bit encoding.
- `bond_values: dict[int, bytes]`: one raw float64 cell (8 bytes) per complete represented Bond UID, including both hidden partitions.
- Keep `angles` and `torsions` as the existing ordered UID tuples. `updated()` shares the same two maps when replacing sequence tuples;
  `subset()`/complete install builds new filtered maps once, with explicit current Atom and complete Bond UID sets.
No new class/dataclass/Protocol/enum or result/ABI mirror. Existing native structs may gain fields; no new native type definition is needed.
Dense current-member keys are deliberate: scalar changes replace an existing exact-int key, so rollback never reinserts a deleted key.
Shared immutable default bytes may serve many keys; they contain real default values/metadata, not a second store or placeholder columns.

Each Atom/Bond holds one private `_constraint_values` reference to its actual map; this is a binding, with no copied value/cache.
Registered handles reference the corresponding map in the Molecule's current `_ConstraintSet`.
A removed Atom or nonmember Bond references an isolated singleton map with its own cell. That map cannot retain the whole source owner.
There is never an attached table/Bond value plus an authoritative constraint-map copy. Undo/staging copies live only for the operation.

Keep public `ATOM_SCHEMA`, `ATOM_FIELDS`, `FIELD_INDEX`, aliases/codecs and legacy factual projection fields at 19.
Declare separate private table fields = historical indices0–15, table presence mask = low16 bits, table source width =16.
The table keeps the same two blocks: six scalar factual columns plus XYZ; seven derived columns remain legitimately unstored.
Its `_columns` tuple, private row/matrix/source methods, append/remap/take and current private pickle use width16, with no axis slots.
`Bond._attrs` physically becomes exactly `(bond_order, id)`. Public `Bond.attrs` remains `(bond_order, constraint, id)`.
The public 19-column Atom and three-slot Bond formats are transient owned exports, never persistent compatibility buffers.

## Constructors, registration and nonmember lifecycle

Actual source: `Atom(mol=None)` creates a new Molecule and registers the Atom there; `.mol` is not None.
Both Atom and Bond receive `_member_uid`; `_uid` is its property. The root probe's `__dict__.get('_uid') == None` does not mean missing UID.
Preserve UID allocation/validation timing, including gaps after rejected construction; do not roll back existing high-water behavior.
Atom constructor aliases still compare/coerce bool exactly as today. Prepare a local constraint cell together with the legacy row;
append the 16-column table row and register the handle, then enroll/publish the UID cell and bind its map within the same transaction.
No source-less Atom path is invented; private staged handles are bound before they can be published or read.

`Bond(atom1, atom2, constraint=..., id=...)` is NOT a registered Molecule Bond. Initialize its physical two slots and isolated singleton cell;
unknown-kwarg failure cannot leave a Molecule map entry. Its setter continues working if both endpoints are current, without requiring Bond membership.
`_add_bond` transfers that cell into the Molecule map only when it appends/registers the actual Bond; journal the old binding and new key first.
Current direct/nonmember default order0.0, raw constraint conversion and id preservation remain unchanged.
A permanently removed Bond keeps an isolated cell and its existing owner/endpoints; setters keep today's endpoint-currentness permission.
An incident Bond removed with an Atom remains readable, while writes still fail when an endpoint is no longer current.
A detached Atom retains its sealed one-row 16-field table plus isolated constraint cell; reads/exports/pickle remain inspectable and setters retain current membership rejection (`KeyError` for a removed-axis scalar write).

Prepare detached maps before publication. Structural deletion/filtering replaces owner maps once instead of deleting keys in an existing map.
Rebind every retained handle to the new maps and every removed handle to its singleton under the existing owner lock.
Cover actual removed objects, including `list.remove` endpoint-equality behavior, rather than assuming the caller's Bond object was the removed one.
This applies to replacement, Atom/Bond removal, factual reinstall and all three existing permanent hidden-clear paths.
Hide/restore are still partition moves with identical Bond identity/value; no public topology behavior is changed here.

## Sole ingress, projection and consumer cutover

Add pure private split/join/select helpers beside `_ConstraintSet`, taking explicit table/cell/UID inputs and owning no carrier reference.
At legacy ingress split values19/masks19/sources19 exactly once into table16 plus UID cells; retain raw NaN payloads, signed zero,
non-boolean float values, independent presence and every source code. Only constructor alias handling retains its present bool coercion.
Use the same helpers at constructor/import, record install, raw foreign-Atom insertion and detached restoration; no second normalization route.
`_capture_factual_record` joins RAW table facts and owner cells, never the chemical display projection (which could alter H/derived state).
`MoleculeRecord`, `_AtomFactUpdate`, their public field order/immutability, and record byte formats remain unchanged at19/3.

`_atom_projection` joins only requested fields, then runs the existing electron/valence/H display logic unchanged.
`_numeric_property` and `_atom_numeric_value` read an axis from its bound UID cell in constant work; no Atom row or content digest is built.
`Bond.attrs`, scalar getters, `_bond_order_setter`, `_store_bond_field`, metadata setter, chemistry's aromatic commit and record capture
all use the one raw three-slot export helper when they need transport. Order-only consumers continue `_attrs[0]`; id routes use physical slot1.
No direct owner/consumer may interpret `_attrs[1]` as constraint after the cutover.

Source identity `constraints` keeps precisely its current bool-value semantics, topology and ordered Angle/Torsion tuples.
`forcefield_inputs` still includes raw axis values/metadata and raw three-slot Bond exports in the same order, despite the new physical layout.
All other domains retain their current dependencies. Explicit dependency capture hashes once per operation, not per scalar access.
Current immutable source bindings, source-stale rejection and native row/UID/owner checks remain authoritative.

Copy/deepcopy/components/merge and topology publication already use the raw record boundary: remap both UID maps there, without sharing maps.
Current Molecule pickle must rekey both maps after restoring original Atom/Bond UIDs, then rebind every handle; retain UID gaps/high-water.
Detached Atom pickle carries its cell separately from table state; nonmember Bond pickle uses the full3 projection plus current extra state.
Exclude `_constraint_values` from generic Atom/Bond extras so serialization cannot retain a parallel map/whole-owner reference.
Historical core-path/private-layout asset compatibility is not invented; the existing separately gated historical-pickle decision remains open.

`clear_constraints` still writes all current Atom axes (with today's explicit metadata), active Bonds only, and clears Angle/Torsion tuples.
It must preserve hidden Bond cells when replacing sequence state. `constraint_bonds_angles` retains its existing metal-exclusion/angle-only behavior.
Delete unused `atom_update`/`bond_entries` branches of `_commit_constraints` after confirming no callers; keep the sequence commit path.
This removes the dormant slot2/id bug without making that unused branch a second owner writer.

## Native writer and rollback

Keep native normalization over the full19 schema so current scalar coercion, aliases and internal mixed `_AtomFactUpdate` fields remain accepted.
Freeze shared `_configure` additions as `table_width=16` and `constraint_fields=("x_constraint", "y_constraint", "z_constraint")`;
existing `fields`/`indices`/`presence` remain full19. Native derives axis index0–2 from that order and looks up `mol._constraints.atom_axes`.
The Python/native cell contract is exactly values bytes0:24, local mask byte24, source bytes25:28; derive offsets from3 axes×8 bytes.
Native touches existing UID keys only; Python registration installs complete cells first. The `_cells` capsule journal/API stays unchanged.
Configure existing `Field`/`Configuration` roles, without a new callback/native entry point or exported private signature change.
`_write_prepared(table, ...)` is strictly the low-level table writer and rejects owner-axis fields; owner-aware commits accept them.
Extend existing `install`/`prepare_undo` and `Undo`, used by scalar/fields/facts commits, to partition touched numeric columns and owner axes.
Use the validated current UID to look up its existing cell, prepare one replacement bytes28 per touched row, and enroll actual map/key/old/new cells
in the SAME existing native undo capsule before any publication. Preserve explicit-producer inheritance and named producer precedence.
Use bit copies/byte-order conversion only; do not cast raw constraint values to bool. Numeric cells and dependency seed/status behavior stay unchanged.
The existing no-outer-transaction native scalar path remains atomic; no Python wrapper/source rehash/full-map clone is introduced.

Owner-cell publication replaces an existing builtin-int key in place. Undo retains actual map and exact old immutable bytes, restoring by key replacement;
this path allocates no new dictionary key/bucket, calls no user hash/destructor, and needs no copy of unrelated entries.
Retain native `value`/`source`/`presence`/`enrollment` fault coverage for owner-cell publication; metadata is one atomic cell, not three durable mirrors.
Python lifecycle/Bond writes extend the existing transaction with touched map/key/old-value entries and binding snapshots, without a new journal class.
Capture Atom binding alongside `_detached_table` in topology undo and Bond binding in `touch_bond`; preserve original owner/map identities on failure.
During rollback restore original topology/roots/bindings, replay native capsules in reverse, THEN undo Python map insertions/replacements in reverse.
That ordering leaves newly registered keys present until any later native update to them is undone; no removed old-map key needs reallocation.
Topological owner replacement must never structurally mutate an old map captured by native undo. Nested-failure poisoning stays unchanged.

## Cost, exact scope and validation release

New Atom/Bond scalar reads/writes are O(1) in molecule size and O(touched fields) in work/undo; inherited metadata copying is unchanged.
Batch updates/projections are O(selected rows × requested fields), check supplied dependencies once and allocate only requested exports.
Structural filtering/publication is one O(N+B+sequence-members) pass, matching existing topology work; isolated Atom registration must not scan existing members.
Dense maps add pointer/dictionary cost while eliminating three float/source table columns and the Bond slot; memory and usable cost need measurement,
not an assumed improvement. No revision counter, digest cache, second topology index, native ABI prefix or restraint-enforcement algorithm is added.

`file_scope.json` fixes exclusive implementation ownership: 8 production files and 8 existing test files; no production/test edit has occurred here.
`validation_plan.json` gives exact file selections, prospective build argv and evidence boundaries; meaningful new assertions cover lifecycle/raw metadata,
source domain separation, native mixed undo and current constructor behavior. Reuse root's independent public test preparation rather than duplicate it.
The current nine-file baseline is185/0/0 both CP311 source and installed outside checkout; it qualifies pre-change behavior only.
After root releases code, rebuild ONLY `_fact_native` for CP311 with the located Stage3 helper and qualified toolchain, then validate changed source,
current installed-wheel tests outside checkout and actual rebuilt-extension origins. Reused old native bytes cannot qualify this migration.
Run affected storage/native/dependency/builder/permissions/lifecycle/source/constraint tests; include working-copy/native publication and H rollback nodes.
Measure equivalent scalar-axis/aggregate/Bond loops, mixed prepared batches and full exports over increasing N, retaining all adverse observations;
separate setup, peak/RSS owner cost, primitive scaling and integrated repeated-workflow cost. No universal5× or historical-hash chemical verdict.
Root adoption plus a separate explicit implementation/build release is the remaining entry gate; later all-version/native/scientific/global gates stay open.
Hidden-selection retirement/Decision6 consumer cutover and Stage7E H/isotope/stereo normalization remain later partitions, not prerequisite edits here.

## Evidence actually obtained for this candidate

Read current completion/entry records, prior full normative alignment, actual owner/writer/consumer source and affected tests using `rg`/static file reads.
Inspected root's baseline report/argv, constructor observations and independent preparation review:11 pass/1 fail; its removed-Atom writable-axis expectation conflicts with current `KeyError` behavior. Preserve that rejection in this slice; retain the failed attempt. No runtime import/test/build/install/environment/remote/Git mutation ran here.
Main-plan changes are only authorized top status, short Stage7A/7B pointers and mechanical Decision6 synchronization to root's documented adoption.
This design is reviewable now; performance, new native undo and complete connected migration remain implementation obligations, not claimed passes.
