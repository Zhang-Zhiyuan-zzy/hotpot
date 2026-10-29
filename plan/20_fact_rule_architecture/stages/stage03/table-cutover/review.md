# Stage3 sole-table and connected-writer cutover design

Status: architecture design with the subsequently assigned Core partition B
in source validation; neither a Stage3 completion nor a performance pass.
The initial design assignment owned only this review. The later implementation
assignment owns `core.py` and its new carrier test file as recorded below.
The parent owns project-wide decisions, budget and final evidence.
The subsequent native-entry adjudication below refines the private execution
boundary only. Its implementation has not started at this review checkpoint;
the Python engine remains the actual installed implementation.

## Entry, authority and bounded work

Entry HEAD is `ce416297167e735c3961f75bcd04621d9746fde1`, synchronized with the
implementation-branch tracking ref. N02 `_builder.py`, `test_builder.py`, its
review and the N02 budget edit were still uncommitted. Atom's parent is already
readonly, UIDs/source identities and N05 exist, and Atom still owns a raw array.
No public `updata_atoms` is connected. The isolated N02 work is not a builder.
Other workers own its final validation and records; their files are untouched.

Following `hotpot-stage`, the full main plan1–2448 and four normative companions
were reread before this design: protocol1–284, scope audit1–521, alignment1–177,
Plan21 1–68; 3498 lines. Scope-audit420–480 was repeated to recover truncated
appendix output. The Stage1 concrete decisions, affected reviewed contracts,
previous builder review, current storage/update/identity code, actual writers,
UID/pickle tests and FF commit/restore code were also inspected.

| Normative source | SHA-256 |
|---|---|
| main plan | `afb13f1a62f7d8db8a0fa8e893c4d1322e18d73edb785b0ad649c1eb02621f4b` |
| chemical-type protocol | `d0ac2cff4c15f525236b00e254211020423e90f7687c1a640bfe28c7fee45bfc` |
| data-object scope audit | `72f935922c623948cc29d781c2eb77e38bcc8562cf368496a5cd7fb0599f1638` |
| Plan17/20 alignment | `e5549116f1409fb46b5fd2bdc82c0b20a34297989475eb66a119ac221a9d4a0b` |
| Plan21 | `b423d77b2ea290e847725570cf8023a90ddbbcca7e14d1796ce4a1ec3f6be83d` |

Inspected production identities at entry:

| Source | SHA-256 |
|---|---|
| `core.py` | `abfeaeb002e263bcbd384ddeec3ce83f3e0bc44ffda1430a66507f903a5e9b56` |
| `structure/_atom_table.py` | `f74ff06a61fb23ddd616b04db1dca2af0482de49573da5436267b18ed6b8b668` |
| `structure/_identity.py` | `8030773255500980c33a66eea55cc38730c688837ae6f0d5a39a4ba201cd84af` |
| `structure/_mutation.py` | `83f061036ce2a37a74e5964529fa90c622cb3855979b09fe5ccd42d0bcb56d24` |
| installed N02 before its concurrent validation refinement | `9fec55f5c6fabc773441efe582f112843ac6ce7f0ddbe76fb6db479c7f6aca05` |

Shell Python is3.11.16 at
`/home/zhangzhiyuan/usr/conda3/envs/hp-usage/bin/python`. The initial design pass
executed no chemistry tests. During review the parent reported the refined N02's111 cases,
394 affected cases on each Python3.10–3.14 and4217 default passes. Its final
source/report hashes must come from the parent-owned evidence, not the earlier
N02 hash above. No native/wheel or scientific tier is inferred from those runs.
During this review N02 was committed as `caf959e`, the first commit after the
push through `ce41629`. Its later validation refinements preserve22 fields.

The authorized route remains Plan20 first, current sole Python-RDKit MCA,
unchanged standalone wrappers, and Stage13 only after validated Stage0–12 on a
new branch. A2 retains the old raw constructor parameters until Stage5 and
defers partial-charge result ownership/NaN encoding to Stage7C. Isotope/stereo
ownership remains in its existing holders until Stage6. This cutover changes
authority, controlled writes and transport, not scientific algorithms.

The bounded implementation list is: optimize the validated numeric primitive;
introduce one controlled transaction; connect Atom/Bond access and constructors;
connect factual build/topology/lifecycle; migrate imports and FF rollback;
make source capture/serialization actually detached from live buffers; complete
the corresponding gates. Do not mix in view extraction, cache deletion,
hidden-Bond removal, scientific-model replacement or final result/frame owners.

## One coherent authority change

Install one `_AtomFactTable` on every Molecule. At the same production checkpoint,
remove every Atom-owned numeric array and all writes that depend on writable
`Atom.attrs`. Public row/matrix access becomes an independent projection. All
controlled writers, restoration, metadata and source capture switch together.
An unwired private optimization checkpoint may precede this; a committed tree
with both active storage layouts may not.

Keep the measured two-block float64 layout and19 fields for this stage. No
integer narrowing, new public isotope column, twentieth MCA slot, result NaNs
or change to scalar int/bool read codecs is needed. Bond retains its existing
three-slot physical array; its owner is renamed privately to `_attrs` so a
public projection cannot remain the write channel. This is buffer isolation,
not an unrelated Bond table rewrite.

## Exact additional state and record budget

| Owner | Added/changed private state | Invariant |
|---|---|---|
| Molecule | `_atom_table` | Only active Atom numeric/presence/source owner. No retained complete matrix or second row map. |
| Molecule | `_charge_value`, `_charge_present`, `_charge_source` | Public default remains0; absence is metadata, not an authoritative invented zero declaration. Existing charge assignment records explicit presence. |
| Molecule | `_spin_value`, `_spin_present`, `_spin_source` | Explicit declaration is distinct from the existing selected-frame fallback. The public dynamic spin name is retained. |
| `_AtomFactTable` | `_group_status`, `_dirty_seeds` | Fixed group columns plus bounded UID seeds, detailed below; no calculated values or duplicate presence/source. |
| detached Atom only | `_detached_table` | Same-schema immutable one-row table, including presence/source/status; no row remains in the active table. Active Atoms do not own this value. |
| Bond | `_attrs` replaces the public array owner | Same raw three values. `attrs` projects them independently; source metadata remains its existing sole Bond owner. |
| Conformers | `_mutation_lock` | Existing nine-field owner gains isolation/locking, not a second collection, parent reference, role/result owner or new frame schema. |
| one operation only | `_FactTransaction` workspace, or the bounded native root-scalar stack execution below | One lock/undo/publication contract; cleared on exit. Never stored on Molecule or serialized. |

Keep existing `_owner_token`, `_mutation_lock`, non-reusing UID high-water
marks and sole `_topology_index`. No `_row`, UID-to-row dict, revision counter,
Molecule-keyed global cache, persisted digest, per-Molecule mode or array proxy
is added. Existing `_model/_environ`, caches, hidden lists, whole stereo payload
and result stores remain recorded legacy debt for their assigned later stages;
this design neither creates another copy nor prematurely removes their owners.

Use at most one new ordinary behavior class, `_FactTransaction`, in
`structure/_commit.py`; book it separately before implementation. It is an
operation workspace, not a dataclass envelope or public Operator. A ContextVar
may hold the current transaction stack only while a call is active and must
reset in `finally`; it is not retained Molecule state. Pure constant mappings,
generated property closures and at most one finite status enum need no row
record classes. Update the actual class/alias/enum census honestly.

| Reservation | Decision for this batch |
|---|---|
| N01 | Reuse `_SourceIdentity`; include actual installed table metadata in relevant domains. |
| N02 | Reuse the22-field complete finite import. Do not add a per-row import class. |
| N03 | No new snapshot class now. Eager calculations still use the one carrier body under a transaction; native transport keeps `NativeMoleculeBuffers`. A field-selective rule snapshot is created only with its first real extracted consumer. |
| N05 | Keep the one seven-field `_AtomFactUpdate` for prepared/escaping candidates and batch preparation. The native-entry refinement permits its nonescaping synchronous scalar execution representation; it adds no second accepted record or weaker normalization. |
| N06 | No new `TopologyChange` dataclass needed for these synchronous edits. One named topology commit composes N01/N02 plus explicit retained-member bindings and dependency state. Do not replace an existing useful DTO with an untyped patch dict. |
| N07 | Coordinates are the XYZ subset of N05; `_commit_coordinates` delegates to the same Atom installer. No duplicate record. |
| N08 | No `ConstraintUpdate` placeholder. Named constraint commit takes the closed Atom update, Bond entries and ordered existing angle/torsion entries. ConstraintSet ownership changes at Stage4/7B. |
| N09 | `_commit_stereo(mol, source, payload)` composes N01 and the protected existing `StereoPayload`; no second wrapper class. |

No `_commit_representation` or `_attach_result` placeholder is created merely
to match a conceptual signature. Stage5/7 add their real consumers and reuse
audited assignments/results. This batch does not consume N10–N18 reservations.

## Scalar reads, normalization and one transaction

Generate Atom numeric properties/accessors from the existing shared schema.
A scalar getter resolves the member once through `_topology_index`, reads the
one addressed block slot under the owner lock and applies its existing codec.
It must not call `Atom.attrs`, `table.matrix`, scan members or compute a digest.
Molecule matrix/coordinate access packs blocks once under one lock, and Atom
row access returns one independent19-slot row. Empty matrix remains `(0,19)`.
Keep Atom.coordinates' existing three-value expression and assignment; no new
coordinate alias is published.

The common prepared write boundary is `_commit_atom_facts(mol, update)`. The
native-entry refinement below adds immediate scalar/column ingress into that
same installation implementation, without a second write policy. Membership
selection occurs under the lock: each requested handle must belong to this
owner, be current, have the selected row/UID and occur once. `atoms=None`
selects current order; explicit Atom selections retain their order. N05 holds
the normalized values/metadata, not a writable carrier view. Capture owner-only
N01 for ordinary writes; use named dependencies only for an operation that
actually depends on earlier facts.

Freeze these private entry points for the next implementation assignment:

```python
_FactTransaction(mol, *, producer: int)  # context manager; nested same-owner joins
_normalize_atom_scalar(source, row: int, atom_uid: int, field: str, value) -> _AtomFactUpdate
_commit_atom_facts(mol, update: _AtomFactUpdate) -> None
_commit_coordinates(mol, update: _AtomFactUpdate) -> None  # canonical XYZ fields only
_commit_atom_sparse(mol, atom, *, isotope=_UNCHANGED,
                    frac_valence=_UNCHANGED) -> None
_commit_bond_facts(mol, bond, binding, *, raw_values, kind, direction,
                   source_namespace, metadata) -> None
_commit_molecular_declarations(mol, binding, *, charge=_UNCHANGED,
                               spin_mult=_UNCHANGED) -> None
_commit_stereo(mol, source, payload) -> None
_commit_constraints(mol, source, *, atom_update=None, bond_entries=(),
                    angle_entries=(), torsion_entries=()) -> None
_commit_topology(mol, source, record, *, atom_bindings, bond_bindings,
                 dependency_state=None, bond_metadata=None) -> None
Molecule._build_molecule(record, *, bond_metadata=None) -> Molecule
mol._install_molecule(record, *, bond_metadata=None) -> Molecule
```

`raw_values` is the complete raw Bond triple; the other required Bond arguments
make its supplied metadata complete. The Bond commit does not calculate H.
Declaration inputs are the N02 presence/value/source triples; `_UNCHANGED` is
one private sentinel, so absent/None is not confused with omission. Constraint
entries refer to the current Bond or existing ordered angle/torsion membership
and their values. Dependency-state input is the closed status-byte/UID-seed
snapshot, remapped with membership; an omitted value initializes a fresh
import only and cannot reset retained member state. The opaque metadata rule
is specified below. No function accepts arbitrary carrier attributes.

Core owns the Stage3 write controller around these value-only commits: scalar
numeric access and `updata_atoms` prepare N05, enter the transaction, call the
same commit, and invoke the existing eager body when that public operation
requires it. The storage/commit module does not import Core's chemistry to
discover callbacks. Primitive stores used inside the eager body call the
value-only commit and cannot recursively invoke that eager controller.

The production pipeline must not normalize the same candidate repeatedly:

```text
user conversion/alias checks -> N05 -> row/UID/source checks
    -> one prepared table writer -> dependency marking -> return None
```

The table's validating `update` method and molecular commits share one private
prepared writer. That writer installs raw values, presence and producer codes
together, including rollback of the currently interrupted write. A trusted
prepared path is justified by completed validation at its caller, not by
trusting arbitrary caller data. Keep standalone table validation tests.

The Python checkpoint provides a one-row/one-field N05 factory using direct
float/shape normalization,
cached schema indices and one eight-byte pack, without NumPy arange/full,
field-dict construction or whole-table copies. Larger candidates use the
existing column preparation and vectorized blocks. Both reach the same named
commit, same field schema and same prepared writer; the native refinement
reuses that closed candidate meaning and one installer without requiring an
escaping Python envelope for every immediate scalar. There are no separate
scalar scientific algorithms. Public error/alias/member behavior remains the
exact A4 contract. A repeated setter may not skip a raw write merely because
`0.0 == -0.0` or two values share a lossy scalar projection.

`_FactTransaction` holds the Molecule lock, captures old touched values and
metadata once per prepared write, and coalesces nested named commits. One
transaction-owned undo capture replaces duplicate transaction/writer copies.
The final journal uses ordered operation-local physical-buffer undo records
for distinct bulk and scalar commits. Reverse application restores overlapping
selections and storage replacement exactly. Each record covers only its touched
cells/slices; journal work scales with the performed writes. The shared capture
is used by the transaction or by standalone table mutation. Deferred dependency
marking coalesces independently; the numerical journal has this ordered policy.
This changes no fact owner and retains the complete BaseException guarantee.
Known physical write sets
are declared before execution. Conversion and possible user callbacks precede
publication. The journal has closed Atom/Bond/topology/declaration/stereo/frame
domains; it is restoration data, not an arbitrary setattr/FactPatch interface.
On BaseException restore all changed values, metadata, membership and pending
groups. Never decrement UID allocators. Controlled getters/source snapshots
take the same lock, so no other thread can observe speculative intermediate
stores. Logical publication is the outer transaction's successful completion.

For synchronous eager legacy calculations, touched writes may be staged in
the held store while the old body reads its own proposed inputs. They remain
unpublished and rollback protected. This avoids an Atom proxy, a second
scientific body or an N-sized working Molecule per scalar edit. Nested setters
join the transaction rather than publish independently. This temporary undo
mechanism is distinct from the complete immutable proposals used by extracted
rules later. Tests must interrupt each relevant write/read boundary; a lock
without an undo set does not satisfy atomicity.

## Preserve eager chemistry through Stage3

`_atomic_number_setter` currently writes Z, six electron slots, valence and
implicit H. Keep its existing electron reference lookup, `get_valence()` and
`_calc_implicit_hydrogens()` bodies and order. Replace their raw-array stores
with named internal field stores inside one transaction. The undo set is those
nine slots and their metadata/status, not all N atoms. Numeric Z remains
explicit; electron/valence/H outputs receive their existing named producer
codes. Do not remove eager behavior or silently preserve/replace a different
chemical representation before Stage5.

`_bond_order_setter` keeps its existing conditional kind mapping: normal kinds
and unknown follow order, whereas dative/zero are not newly normalized. Keep
the two endpoint H calculations in their existing order. Its transaction
covers Bond order/kind and both endpoint H values/metadata. An endpoint failure
restores the Bond and the already updated endpoint. Do not calculate H once
through a new duplicate formula or change directional Bond semantics.

Public constructors retain `attrs_array` and `update_electron_config` until
Stage5. Raw input is copied into the sole table, marks all supplied slots
present/raw-array, then runs the existing optional electron overwrite when
requested. Ordinary kwargs retain the characterized defaults and current
Stage3 accepted inputs. Electron construction still runs when the existing
flag requests it. Constructor failure no longer leaves a registered partial
Atom: roll back the row/membership while retaining consumed UID high-water.
The existing test explicitly characterizing partial registration must be
rewritten to assert complete failure plus non-reuse, with that mapping logged.

The new batch API has no pre-existing setter-order contract. Normalize aliases
first, stage all requested facts together, run existing eager element bodies
once for selected element edits in selection order, and reinstall explicitly
requested downstream fields last. Thus an explicit H count in the same batch
is not lost to a default calculation. Computed and explicit slots retain their
different producer codes. Record this concrete Stage3 batch behavior in the
interface ledger before publication; Stage5 later replaces the eager body
with the approved mark/ensure route. Do not infer this ordering from kwargs
dict iteration or repeat whole-molecule normalization for each selected Atom.

Electron_configuration, raw constraint axes and other still-writable legacy
numeric projections use closed internal schema updates until their owning
permission cutover. Public A4 continues to reject its declared readonly fields.
Do not delete the old constructor parameters or introduce Stage7C NaNs early.

## Metadata, finite dependencies and source checking

The table owns raw values, per-row19-bit presence and per-field producer codes.
Only an actual producer supplies presence. Default zero is absent; supplied
zero is present; raw arrays supply all slots; imported formats provide their
known coverage; rule outputs are named inferred producers. Copy/remap/rollback
preserve heterogeneous row metadata instead of replacing it by one code.

Keep codes0–9 unchanged. Declare additional actual writers as fixed private
codes, with no runtime registration:10 cclib,11 conformer_load,12 geometry,
13 forcefield,14 xtb,15 aromaticity,16 kekule,17 partial_charge,18 tmqm.
An operation supplies its producer to the transaction; nested legacy setters
inherit it except for explicitly named electron/valence/H computations.
Existing standalone wrappers still use their public contracts; Hotpot-owned
workflow calls tag their own working-copy updates. These are private producer
codes, not new public record fields or standalone ABI changes. If a listed
producer is not connected in the batch, do not claim its writes are attributed.

The finite Atom groups are `VALENCE`, `IMPLICIT_H`, `AROMATIC_REPRESENTATION`
and `FORMAL_CHARGE_COVERAGE`. `_group_status` has one uint8 per row/group, with
CURRENT/NEEDS_RECALC/ABSENT/UNRESOLVED/CONFLICT. `_dirty_seeds` contains one set
of affected UIDs per group, bounded by current/changed membership, not a log of
revisions. Group edges/scope/read policy are private fixed declarations in
`_dependencies.py`. No new scalar status object is allocated on each edit.

Writes record affected seeds in O(touched entities × fixed groups). A read
that requests a group checks pending seeds in its full upstream closure before
looking at its local CURRENT byte. The first such read expands required
components through the sole topology index, changes affected statuses and
consumes/coalesces seeds once; unrelated groups/other components do no such
work. Breaking an edge seeds both old endpoints before removal so both
resulting sides are invalidated. Removal/remap preserves surviving statuses
and remaps seeds by UID; copy never resets a dirty/conflicting state to CURRENT.

Stage3 establishes and verifies that status/seed engine, owner checking,
positive/negative propagation, remote-first expansion and restoration. It does
not replace legacy public eager/derived read behavior by a new chemical
inference route. Stage5 connects the named pure ensure operations, explicit
source conflict handling and one-time materialization after extracting the
existing bodies. A Stage3 private ensure with no installed evaluator reports
the actual unresolved/stale state; it cannot label a stale diagnostic value
CURRENT. Legacy raw-slot projections remain explicitly staged behavior, not
evidence that final lazy semantics already pass. Stage3 completion records
must distinguish its migrated mechanism gates from the assigned Stage5
scientific-read gates. This follows the companion's staged ownership; it does
not waive final source-aware reads or the remote-first requirement.

N01 capture reads table columns directly once under the same consistency
boundary, not one copied Atom row per requested field. Representation captures
the representation fields and their relevant presence/source; geometry captures
XYZ and its metadata; constraints capture their current raw owner/projections.
The current scientific consumers of mutable legacy electron/valence/partial-
charge inputs must be included in the applicable operation's input check until
their derived/result cutovers, rather than omitted because the final owner is
different. Display/source IDs alone do not invalidate geometry. The actual field
metadata policy is `(selected_presence_mask, selected_producer_columns)` beside
the selected raw values. Representation captures that tuple for its four Atom
fields; geometry captures it for XYZ. A coordinate producer/presence replacement
therefore conservatively changes the geometry source binding even when numeric
coordinates compare equal. This is separate from changing `Atom.id`/`Bond.id`.
Ordinary setters
never call `_capture_source` with all domains or digest the full Molecule.

Sparse existing isotope/frac_valence writes are serialized through the Atom
fact boundary while retaining the existing sparse holder and missing-attribute
behavior. They do not gain a public property, table duplicate or new absence
API. Stereo stores one protected payload through `_commit_stereo`, retains
class identity and source/current H fields, and does no chemical interpretation.
Full decomposition into canonical isotope/radical/source-ledger owners remains
Stage6. Existing MCA/partial-charge attachment and frame-result validity are
not given shadow status owners before Stage7C/7D.

## Construction, topology and factual installation

Raw handle initialization uses the one existing Atom/Bond definitions, binds
owner/UID/endpoints and attaches no Atom-owned numeric row. It does not allocate
another UID or invoke setters. Public isolated Atom registration appends one
prepared row and one empty topology-index entry in amortized O(1); it must not
build a complete N02 or rescan all N members. Direct Bond construction retains
its characterized non-registration and one UID allocation.

`Molecule._build_molecule(record)` allocates one destination and calls the sole
`mol._install_molecule(record)`. For complete imports, prepare all buffers and
handles, reserve UID ranges once, build/validate the deterministic topology
index once, and publish table/membership/index/declarations/cell/stereo/frames
together. Build the index independently of `_update_topology`'s legacy cache
and frame-clearing side effects. No per-Bond adjacency update followed by a
second full rebuild is allowed on this path. The installer does no electron,
valence/H, aromaticity, perception, force-field or toolkit calculation.

The named `_commit_topology` takes N01, the complete factual record, aligned
`atom_bindings` and `bond_bindings` (existing handles or None for new members),
and the prepared dependency-state remap as explicit closed inputs. Existing
handles must match owner/UID and keep identity. New bindings use reserved UIDs.
This composes existing values; it is not an arbitrary dict or new universal
patch. Small isolated registration uses the same validation/publish primitives
with its O(1) local index addition. Bulk remove/replace/reorder builds one
complete candidate index and remaps table values/metadata/status together.

For detachment, freeze a same-schema one-row table before removing the active
row. Detached Atom parent/UID stay inspectable, bonds/neighbours stay empty,
idx retains its existing membership failure, and fact writes reject the
detached handle. Restoring a failed edit restores the old binding and removes
the uncommitted detached state; successful compaction cannot retarget an old
handle to another row. Replacement allocates a new Atom UID but preserves
retained Bond identities while remapping active and hidden endpoints.

Public `add_component` retains every existing destination handle/UID. Copy and
components create fresh local handles/UIDs and explicit source-row maps. Raw
Atom19/Bond3, sparse input values, stereo, current geometry and already retained
constraints/metadata survive. No `attr_dict` codec is used for raw Bond slots.
Use the reviewed declaration/cell/annotation policy and record its actual
implementation step; do not adopt charge/spin by summing incomplete Atom facts.
Whole-molecule spin is not combined by an invented rule.

The full source-versus-generated frame copy policy cannot be inferred from
today's mixed Conformers owner or the presence of energy. Stage3 must preserve
all nine fields on factual imports and FF restoration and may not lose any
previously retained values. It must not claim the final Decision19/Stage7D
copy classification until source/result roles exist. Keep that named remaining
gate; neither copying all generated frames nor treating every frame as P1 is
an acceptable shortcut. Existing supplied pair observations remain with their
current owner until their own complete migration.

## Converter and opaque-metadata boundaries

`obmol2mol(obmol, mol)` and `from_rdmol(rdmol, mol)` keep their exact signatures
and return `mol` itself. First reject any nonempty Atom/complete-Bond target
before changing it: current nonempty calls append rows but connect from row0,
which is a technical corruption/partial-mutation bug. In-repository callers
and tests use new targets. This early ValueError is the concrete minimal bug
fix under the owner's continuing Plan20 authorization; it creates no new
append capability and does not require another generic permission question.
Document/test the precise trigger. Composition stays at `add_component`.

For structurally empty preconfigured targets, preserve observed complete
effects explicitly: OB supplies charge, replaces cell only when supplied and
does not supply stereo; RD supplies charge and its complete extracted payload,
including an explicitly empty payload. Both currently clear legacy frames at
topology refresh. Apply those facade intentions in the one transaction, rather
than letting a generic installer accidentally clear unspecified state. The
target token, lock and UID high-water survive. CCDC's missing return and
`c_mol.atom` use need a concrete licensed/fake-boundary characterization before
their factual import is claimed; an unexecuted external tier is not passed.

External toolkit decoders create the finite N02 with real presence/source and
call one operation. Keep RD's extraction-before-Kekulize route and current
algorithms. cclib first prepares its supplied declarations/units/all nine frame
values and selected current coordinates; existing link/aromatic/order/valence
calls remain explicit subsequent operations. MCA DTO reconstruction keeps its
existing schema and the separate RDKit binary's scientific purpose.

N02 deliberately supports finite immutable metadata trees. Public Bond
metadata currently also preserves arbitrary deepcopy-compatible Python values.
Do not silently replace such values with None, invent an opaque-pickle tag or
narrow public copy semantics merely because N02 preparation rejects them.
Refine the private lifecycle installation input with one optional
`bond_metadata` tuple of operation-owned, deep-copied mappings. The record's
metadata entries must all be None when this complete tuple is provided, so
there are not two competing supplied values. A normal finite import omits it
and uses the record. Both feed the same raw Bond initialization and final
private metadata owner; no sidecar is retained on Molecule/Bond. This is one
installer with an explicit carrier-copy input, not a storage/profile switch.
The lifecycle copy channel is not represented as an immutable process message;
it retains the existing deepcopy guarantee and unsupported-object failures.
Source-dependent scientific hashing may still reject unsupported opaque
values as it does today. No `repr`/Python-hash stand-in is introduced.

This private input refinement is required for compatibility, not a new public
DTO or extra fact owner. Record its exact signature and mutual-exclusion tests
before implementation. Annotated metadata uses its separately accepted
container-copy/value-reference contract; do not deep-freeze arbitrary user
annotations into N02. The finite record's class count remains one.

## FF, serialization and cross-field consistency

Keep `_WorkingCopyCommit` and `_MoleculeCommitSnapshot` as their real operation
records. The candidate must carry original member UID/owner checks, full raw
rows with heterogeneous presence/source, added rows/Bonds and all nine frame
domains/selection/stereo. Capture required N01 dependencies when making the
working copy, pass them through the operation, and recheck before commit.
Capturing only after optimization cannot detect an intervening caller edit.
Do not store the original-operation context on the working Molecule.

FF prepares a new complete table and candidate topology without touching the
original live graph. Publish one swap while retaining all original Atom/Bond
handles; newly added H/Bonds get new local UIDs. The old table is task-local
rollback state, not a second active owner. A rollback restores original table,
ordered members/endpoints, dependency state, stereo, full frame owner/selection,
constraints and the still-existing cache/annotation containers needed by the
old failure fixtures. Do not rerun public setters or `_update_topology` during
restoration and erase constraints/frames. Retain the old prepared index when
restoring. Burned UID ranges remain burned.

`Molecule.__getstate__` currently shallow-copies dict state under the lock, and
the table returns live slices. This is insufficient after activation. Capture
the full chemical serialization state while holding the same transaction
lock, copying unique live table blocks/metadata/status and member/Bond/frame
values before release. Never serialize spare uninitialized capacity or column
views as independent owners. Standalone table state is likewise an owned
copy, and reconstruction recreates its unique column views. Copying a dict
that still points at mutable Atom/Bond/table/frame containers does not count.

Use a private current-version Molecule reduction that restores the complete
captured graph through the same raw initialization/index-building primitives,
with recorded UID arrays/high-water marks and one opaque owner token. Keep
that token as a shared pickle argument so jointly serialized N01/N05 source
bindings retain memo identity. No carrier lock is serialized. Recreate locks
and reconstruct rows/views exactly once; no old-core alias module is added.

Atom/Bond root reductions bind to the restored owner/member using row hints
and expected UIDs; stable ordinary paths are O(1). A mismatched concurrent
member snapshot is rejected explicitly instead of binding a different row or
creating a second active member. Detached/direct non-member values carry their
own captured fact state and retain their proper owner relationship. Test root
identity itself, not just another member reached through `restored.mol`.
Serialization of an arbitrary outer graph is not claimed to be one global
multi-owner transaction. The Molecule's own numeric/membership/metadata
snapshot must always be internally coherent.

Conformers keeps its current owner/role until Stage7D but gains one lock and
copy-on-read numeric projections so public indexing/iteration cannot mutate a
captured frame behind the transaction. Mol-to-frame lock order is fixed;
standalone frame edits acquire only the frame lock. I/O and FF replace all
nine prepared fields under that lock. The current empty-iteration/add defects
remain explicitly characterized for Stage7D; this storage batch does not
pretend its role split and scientific workflow rewrite have already happened.
Opaque annotation values retain their user-owned lifetime, and the still-
unmigrated independent Crystal internals retain their Stage7A contract gate;
neither is falsely certified by a table snapshot test.

## Connected writer inventory and exclusive implementation partitions

The executed AST/rg inventory found raw Atom array stores only in Core and FF
working-copy restoration/commit. Model extraction also reads `attrs` slices.
Other chemical writers already pass through properties, except explicit
membership/frame/stereo/private-metadata assignments listed below. Unrelated
ML objects with fields named x/y/attrs were excluded after inspecting owners.
Static scans must be repeated on the final candidate, including dynamic
setattr/getattr and tests, rather than treating this table as a permanent
exception list.

| Writer/consumer | Required connection |
|---|---|
| Core MolBlock default stores, `_atomic_number_setter`, Atom init/electron setter/coordinates/constraint/symbol | Generated projections and one scalar/batch commit; no read-modify-write through `attrs`. |
| Core `_set_coordinates`, coordinate setter, translation, conformer_load | One XYZ commit; selected frame and coordinates change in the same transaction. |
| Core `_bond_order_setter`, Bond init/metadata, Ring aromatic setter | Same raw Bond owner, coupled eager transaction; Ring/legacy aromatic algorithms remain until Stage5. |
| Core `_add_atom`, `_add_bond`, `_replace_atom`, remove helpers, create_atom/add_bonds, hide/restore/clear, hydrogen helpers | Shared registration/topology commit/remap; active and both hidden partitions covered, not removed early. |
| Core copy/components/add_component/AtomSeq.to_mol | Explicit record/member maps, fresh or retained handles as appropriate; no attr_dict loss or shared mutable destination facts. |
| Core charge/spin/create_crystal/refresh IDs/setattr/ID-dict updaters | Existing fields route through controlled boundaries. Generic public-surface retirement waits for its owning-stage consumer migration. |
| `obconvert.py`, `rdconvert.py`, CCDC converter, `_io/_io.py`, MCA serialization | N02 plus one installer, target identity, presence and source coverage. cclib unit/scientific sequence unchanged. |
| `forcefields/working_copy.py` | Remove attrs assignment/object.__setattr__ restoration and shallow numerical undo; preserve original live identities and full metadata/frames. |
| `forcefields/{coordinates,ligand,native_adapters,optimizer,repair,workflows}.py`, `conformers/adapters.py` | Existing property calls become atomic via facades; operation provenance/source snapshots and coupled frame writes join the same boundary where required. No duplicated backend. |
| `stereo.py::set_stereo_payload`, conformer projection and remap callers | Seal/store existing payload atomically; preserve protected source fields and one class identity. |
| `kekulize/aromaticity.py`, assembly `action_func.py` | Preserve existing Ring/Atom/Bond calculations and controlled setters; coalesce complete legacy writes, do not extract a second chemistry algorithm. |
| Core assign_partial_charge and `_io/g16.py` prewriter | Whole-set raw charge/declared charge updates are atomic; return/model semantics remain current until Stage7C. Existing cached OB synchronization remains its assigned legacy debt. |
| xTB `commit_xtb_coordinates`, tmQM provider | Controlled coordinate/provided-value writes with source attribution; preserve Wiberg/frac_valence observations. Do not move result ownership early. |
| `AImodels/data_extract.py`, PyG data utils, `_native/buffers.py`, calculator facts extraction | Direct requested-field/table reads with one source check and unchanged feature/ABI ordering. Atomic-number-only is one column, not19 snapshots. |
| standalone OB wrappers | Existing direct public calls still mutate the passed carrier through its controlled coordinates API; Hotpot operations supply working copies. Component APIs/ABIs stay intact. |

The following are proposed ownership partitions for the next explicit
assignment. They are implementation work packages, not permission for this
design writer to edit them. Do not assign multiple writers different portions
of Core simultaneously. Use at most the configured parent plus two children,
and sequence dependent integration.

| Partition | Exclusive files/responsibility |
|---|---|
| A: storage/commit engine | `structure/_atom_table.py`, `_fields.py`, `_mutation.py`, `_identity.py`, new `_dependencies.py`, `_commit.py`; tests for table/update/dependencies/atomicity/buffer ownership. Freeze prepared-writer and transaction signatures before B/C/D integrate. |
| B: carrier/lifecycle/serialization | `core.py` only plus UID/matrix/lifecycle/permission/transport tests. Own all descriptors, constructors, eager wrappers, member remaps, declaration access, frame lock plumbing and root reductions. Integrate A's engine without copying it. |
| C: factual decoders/readers | `_builder.py`, `obconvert.py`, `rdconvert.py`, `_io/_io.py`, `_io/g16.py`, `plugins/ccdc_api/convert.py`, `AImodels/mca/serialization.py`; builder/conversion/cclib/MCA tests. Own record-to-installer preparation and finite/opaque input distinction. |
| D: FF and outer consumers | `forcefields/working_copy.py` and the explicitly affected FF controller files; `conformers/adapters.py`, `stereo.py`, `kekulize/aromaticity.py`, assembly action, model/PyG/native fact readers, xTB/tmQM files only where the final inventory proves a needed connection. Own source capture, metadata-preserving rollback and corresponding regression/feature tests. |
| Parent integration/evidence | Interface ledger/debt/verification map/budget/log, reproducible benchmark runner and stage reports; review all changed public effects and execute source/wheel/matrix evidence. No child edits these records implicitly. |

A and the independent decoder preparation can start before production cutover.
B owns the final Core integration; D depends on its private API and must not
invent a parallel installer. Source and field freezes are reviewable
checkpoints; the active table switch is committed only when every connected
writer/consumer/rollback path passes. Three-commit push cadence remains in
force without manufacturing preparation commits.

The parent has prepared the public-only permission fixture draft at
`/tmp/plan20-interface-tests-draft/test_interface_permissions.py` for partition B.
Its reported source baseline is21 failed/2 passed out of23, recorded in that
draft's `baseline-checked.log`: missing batch support and writable/aliased attrs
are expected failures. It does not impose Stage5 readonly derived setters or
Stage7C None/NaN behavior. This is a useful failing specification, not candidate
validation or evidence executed by this design writer.

## Evidence and remaining gates

Required regression evidence includes: raw19/three-slot byte preservation;
alias isolation and fixed public errors; explicit zero/absence/source;
constructor partial-failure repair; scalar eager element and Bond endpoint
failure rollback; same-target import and nonempty early rejection; opaque
metadata deepcopy compatibility; all hidden partitions; removal/detach/replacement
and non-reusing UID maps; FF original identity/full restoration; nine-domain
scalar/shape frame preservation; payload identity/round trips; current-version
Molecule/Atom/Bond/joint N05 pickle and deepcopy; concurrent frozen serialization;
remote-first dirty propagation; and one actual topology build/no chemistry in
factual build. A mock topology counter alone does not establish value coverage.

Run the Stage3 exact selection in `../../../stage_verification_map.json`, the full new
affected contract set on Python3.10–3.14, the default suite and the appropriate
installed-wheel/native transport tests. Inspect both fresh-process import
orders and the architecture ratchet after the new private imports. Keep
scientific reports separate from technical invariants. No skipped licensed,
native, wheel or scientific tier is described as passed.

Performance evidence is presently unfavorable/incomplete. Prior private
metadata scalar writes are about11x bare-array writes, N05 scalar preparation
about11us, and the accepted diagnostic construction point estimate regressed
about6%. The parent reports new-only full N02 preparation medians for N16/256/
4096 of0.261/2.624/60.751ms and pickle round trips0.181/1.844/30.568ms over78
attempts with no failures. Those are codec diagnostics, not paired B20 results.

Optimize before acceptance: direct addressed scalar projections; one-field
trusted preparation without a generic NumPy/dict pipeline; one validation per
boundary; one owned numerical block copy per bulk operation; amortized row
append; reuse already sealed record bytes instead of freeze/canonical/decode
cycles; and one coarse source check for model/native tasks. Never route each
isolated Atom or scalar setter through N02. A trusted fast factory can avoid
revalidating its already sealed internal representation while the ordinary
constructor and pickle path retain full hostile-input validation and tests.

Measure the actual facade plus chemistry/commit/metadata work against frozen
B20 scalar read/write, complete write-plus-derived-read, construction, removal/
remap, matrix/selected export, I/O and FF/rollback workloads. Include retained
and peak memory, warmups/samples, shared-load notes and paired uncertainty.
Whole work is compared; bypassing checks or timing fewer fields does not count
as improvement. New-only mechanism timings stay separate. If the integrated
ratios remain outside the plan's policy, the performance gate remains open;
do not retain an unsafe alias or a second layout to disguise the cost.

Two cross-stage limits must stay explicit: final lazy source/conflict reads
belong to Stage5, and source/generated/result lifecycle classification belongs
to Stage6/7 as specified above. The table authority checkpoint may advance
without implementing those later algorithms, but it does not certify their
final behavior. If a claimed Stage3 completion would require those outstanding
gates, record that exact remaining scope rather than weakening them or starting
Stage4 on a false completion record.

The preceding design checkpoint records source inspection, not implementation
validation. Its AST inventory, full reads, source/environment and SHA commands
were executed. The subsequent bounded implementation evidence follows below.

## Implementation handoff and concrete hook freeze

The parent subsequently assigned production partition B exclusively to this
Sol-max task: `core.py`, the new
`tests/test_cheminfo/test_plan20_carrier_cutover.py`, and this review. The
configured role is GPT-6.1 Sol at max effort. No child delegation or commit is
part of this assignment. The main plan and all four normative companions were
reread at this architecture entry; their identities above remain unchanged.
The parent owns project-wide decisions, budget/log/evidence and outer FF
integration; the engine writer owns its six structure modules, including
`_identity.py`. These implementation edits do not certify Stage3 completion.

The callback-free Core hooks are now concrete:

```python
_new_atom_handle(mol, uid)
_new_bond_handle(mol, uid, atom1, atom2, raw_values, *, kind,
                 direction, source_namespace, metadata)
_build_topology_index(atoms, bonds, hidden_metal=(), hidden_covalent=())
mol._capture_factual_record(*, rows=None, include_frames=True,
                            copy_metadata=True)  # -> (record, bond_metadata)
mol._capture_frame_state(*, rows=None)     # -> (nine_values, selected_index)
mol._install_frame_state(nine_values, selected_index)
mol._install_factual_record(record, *, atom_bindings=None,
                         bond_bindings=None, dependency_state=None,
                         bond_metadata=None)
```

`_commit_topology` checks its source, enrolls the closed rollback domains, and
calls the sole raw installer on the owning Molecule. The private
`Molecule._build_molecule`/`mol._install_molecule` use that operation. Keeping
these builders with the actual carrier avoids adding a new dependency on the
still-legacy Core module; `_builder.py` remains the pure record/codec owner.
Factories neither allocate a UID nor invoke a public setter.
The installer reserves new UID ranges once, validates bindings using temporary
identity sets, builds the index once, and installs all three raw Bond slots on
retained handles as well as their endpoints and metadata. It retains the
existing Conformers and atom-pair owner objects. Scalar/isolated registration
has a separate *undo enrollment*, `touch_registration(atom)`, that captures
owner references and a tail position without copying all existing members;
it is not a second scientific or installation implementation.

The transaction additionally has closed `touch_sparse(atom)`,
`touch_constraints()` and dependency enrollment. Sparse isotope/frac_valence
writes use the named sparse commit above, retain their current `__dict__`
holders and missing-attribute behavior, and add neither a twentieth Atom column
nor a public isotope property. Cell, declarations, frame selection and legacy
cache/constraint restoration are included in complete topology enrollment.

One source detail corrects a mechanical interpretation of schema generation:
`Atom.n` already has an explicit principal-quantum-number calculation. Its
existing getter is retained. The raw first electron slot remains part of
`attrs`/`electron_configuration`, and its legacy scalar write remains routed
through the table. Generated numeric getters do not overwrite existing
scientific properties. Canonical setter dispatch is decided before conversion;
a ValueError from an eager calculation cannot fall back to writing a shadow
instance attribute.

FF consumes full legacy numeric inputs that ordinary representation identity
intentionally does not include. The engine's closed `forcefield_inputs` domain
covers those actual inputs without canonicalizing arbitrary opaque Bond
metadata. The FF operation can retain operation-local references to the sole
Bond metadata mappings and check replacement by object identity under the
commit lock. Public metadata access returns deep copies and controlled writes
replace the mapping; this detects supported concurrent metadata replacement
without a repr/hash surrogate, a revision counter or a retained second owner.
The existing explicitly requested representation digest may retain its current
unsupported-opaque-value error. The parent records the actual final FF domain
selection and replacement checks in its evidence.

Serialization is two phase. The reduction's arguments contain the frozen raw
chemical graph, UID/high-water state, dependency state and shared owner token,
so the restored Molecule and its members exist before graph-bearing state is
unpickled. Annotations, opaque metadata and member extras are restored as the
reduction state after pickle has memoized that graph. This preserves current
self/member references and root Atom/Bond identity without putting live numeric
owners in a shallow dict. The numeric/frame capture still occurs under the
same locks. Arbitrary outer-graph multi-owner atomicity is not claimed.

The Stage3 copy/component paths retain their existing frame-dropping behavior:
the current owner cannot identify which frames are explicit P1 versus generated
P3. Full factual imports, serialization and FF restoration preserve all nine
fields. This is an explicit outstanding Decision19/Stage7D source-frame copy
gate, not a claim that dropping explicit source frames is the final contract.
Copy remaps existing status/seeds into fresh local UIDs; add-component retains
destination handles and merges remapped incoming state. Component subsets do
not fabricate molecular declarations from incomplete raw formal charges.

At the first connected checkpoint, the executed command

```text
python -m pytest -q --import-mode=importlib tests/test_cheminfo/test_plan20_carrier_cutover.py
```

reported **14 passed in0.95s**. Coverage includes alias-free sole storage,
addressed reads, element and Bond eager failure restoration, constructor UID
burn, explicit downstream batch values, detachment, opaque metadata lifecycle,
one real factual index build/no chemistry, root/member/self-referential pickle
and deepcopy, joint N05 source binding, nine frame fields and concurrent
coordinate/declaration serialization. This is source-worktree evidence only.

The subsequent existing UID/matrix/source/stereo working-copy selection
reported **49 passed,3 failed in1.14s**. One failure is the old intentional
partial-constructor-registration characterization and needs its documented
old-to-new assertion mapping. Two failures require exact StereoPayload identity
through the public setter and FF commit; they must be fixed by retaining an
already sealed payload rather than weakening those assertions. The parent and
engine owners were notified. Later final-source reruns, default/Python-matrix,
wheel/native, scientific and performance evidence remain the parent's gates.

The architecture ratchet exposed that moving cache assignments into a new
helper creates new debt even when the fields existed before. The correction
keeps their actual ownership in the existing
`_update_topology(clear_conformers=True, *, _prepared_index=None,
_constraint_rows=None)` method. A factual installer supplies its one prepared
index and explicit ordered angle/torsion constraints there, requests no frame
clear, and executes no second topology build. The current legacy cache owner
performs its own clearing/restoration; no scanner exclusion or allowlist growth
is used. The four capture/install/frame implementations are actual private
Molecule methods, so engine and FF consumers need no new Core import. Tests
instrument the module reported by the sole public class's `__module__` and do
not freeze a legacy source path.

The next source checkpoint expanded the carrier selection to16 tests and
reported **16 passed in0.98s**, adding isolated-registration no-copy coverage
and complete topology BaseException rollback of sparse facts, endpoints and
identifier. With the shared Stereo sealing repair, the combined carrier,
Core/Bond/hidden/lifecycle/remap/geometry command reported **70 passed in1.36s**.
Those results precede the final method-ownership repair; its rerun and the
parent's complete final-source matrix remain required. Primary member,
index, endpoint, numeric and selected-frame projections now acquire the owner
lock. A further explicit concurrent failed-topology test checks that those
readers wait for restoration.

## Final bounded carrier checkpoint and remaining gates

The parent approved the private builder spelling as actual carrier methods:
`Molecule._build_molecule(record, *, bond_metadata=None)` creates a new carrier;
`mol._install_molecule(record, *, bond_metadata=None)` returns that same target.
Both use `_commit_topology` and `mol._install_factual_record`; neither imports a
second carrier definition or adds a Core consumer. External converters retain
their own early nonempty-target rejection and supplied/omitted metadata policy.
The private installer is a complete factual replacement operation, not a new
public append converter. `_builder.py` owns only the finite immutable codec.

The source-field tuple policy above is implemented in `_identity._atom_metadata`.
`source_frames` deliberately retains its coordinate-only legacy identity.
`forcefield_inputs` adds all non-XYZ Atom numeric fields/presence/producers,
retained isotope/frac_valence, raw Bond3/kind/direction/source, declarations,
selected spin/stereo, the other eight frame domains and selection. FF captures
that domain together with the actual topology/geometry/cell/constraint/source-
frame domains and checks opaque Bond mapping replacement separately. It does
not hash arbitrary deepcopy-compatible metadata or invent a scientific digest.

A4 now reuses the already approved `NotInSameMolecule` for a foreign selection
and `ObjNotInMolecule` for a detached selection; duplicates retain ValueError.
Detached scalar writes retain KeyError. The existing exception constructors
previously formatted Atom repr, which reads `idx` and itself fails on a detached
Atom. Their messages now use type names, keeping their classes, signatures and
object fields. This message-text change is the concrete technical repair,
not an additional exception type.

The existing writable Angle/Torsion `.constraint` is now lock controlled while
retaining its existing local `__dict__['constraint']` owner and permissions.
The setter enrolls that one view in the same undo transaction, including
standalone noncached views. Constructors/raw restoration initialize the value
before publication. No ConstraintSet, new view role or Stage4 extraction was
introduced. Serialization reconstructs constrained ordered views only from the
frozen record; live cache extras cannot overwrite those values. Other derived
caches are reconstructed, while supplied pair observations are copied with
memoized graph references. A deterministic capture-then-clear regression
retains Atom/Bond/Angle constraints from the captured chemical state.

Ordinary fresh-builder Bond roots use UID only as a speculative row hint,
validate that the addressed member is the same object, and retain an actual
row plus expected UID for restoration. Active/hidden segment indexing restores
in O(1), with a membership/UID mismatch rejected. A changed order or burned UID
may require a fallback scan while capturing; there is no universal O(1) claim
and no added row map or persistent hint. The fresh path has a no-iteration test.
Detached one-row tables retain only their own UID's pending seeds, preventing
an N-sized seed set on each removed Atom.

The final executed bounded selection was:

```text
python -m pytest -q --import-mode=importlib tests/test_cheminfo/test_plan20_carrier_cutover.py tests/test_cheminfo/structure/test_uid_lifecycle.py tests/test_cheminfo/structure/test_builder.py tests/test_cheminfo/structure/test_commit_atomicity.py tests/test_cheminfo/structure/test_dependencies.py tests/test_cheminfo/structure/test_atom_fact_table.py tests/test_cheminfo/structure/test_atom_fact_update.py tests/test_cheminfo/structure/test_factual_imports.py
```

It reported **322 passed in1.76s**, using the shell Python3.11.16 in the current
dirty source checkout. Core SHA-256 after this run was
`466ec0a4c86830d7155f9e9b9b07c4ad27b97ddd413bbf6816e2451bf03cab7e`;
the new carrier test SHA-256 was
`f8b5f008dd0641b4a363303255a558e7f96b3eef97aa34814caf39bdf48f1d78`.
The engine was then handed to the parent's separately bounded performance
assignment. Those subsequent edits require rerunning their affected checks;
322 is not final-source or all-version acceptance. An earlier attempted
selection named nonexistent `test_fact_commit.py`/`test_fact_dependencies.py`
and exited4 before collection; no tests from that command passed.

Two retained diagnostic batches used the frozen B20 source at
`/tmp/hotpot-plan20-20261028/source-b20`, commit
`e6082f5fd67cb1054726ddcf43db7f963f7091ed`, and the same
`/tmp/hotpot-plan20-env/bin/python`, CPU0 and
`LD_PRELOAD=/tmp/hotpot-rdkit-toolchain/lib/libstdc++.so.6` on both sides.
They invoked the existing `run_performance_baseline.py --worker <id>` with the
frozen manifest, `--repository <side> --phase <baseline|candidate>
--cpu-affinity 0`; side order alternated by workload. Each batch contains16
worker executions,160 measured attempts plus48 warmups, all completed with no
failure/no-output/process error, and16 separate untimed memory probes.

| Frozen workload | First candidate/B20 median | After bounded Sol-max refinement |
|---|---:|---:|
| construct_8 |14.014|3.521|
| construct_64 |8.208|1.316|
| scalar_read_64 |1.761|0.979|
| scalar_write_64 |151.039|68.633|
| batch_facts_64 |12.170|8.073|
| coordinates_64 |0.749|0.744|
| write_read_64 |35.105|19.471|
| insert_delete_64 |7.247|4.754|

Raw commands/samples and summaries are retained under
`/tmp/plan20-core-diagnostic-20261029` and its `-v2` sibling, driven by
`/tmp/plan20-core-diagnostic.py` and `/tmp/plan20-core-diagnostic-v2.py`.
Their `diagnostic.json` SHA-256 values are respectively
`430766fd2ca1aaf3dc0c3178bfdba82ad5f7a25b257441d1c0db4bc6640eee6e` and
`5e0342c5ffca5f839be966aa93dd0a731d35b8431938ddc52a3ef87b210ed689`.
Runner SHA-256 is `e6e2e5e5a82147399a545e7bb0b770bde2d653d7f9a984aaf44b0664a66177e4`;
manifest SHA-256 is `ed5bba12ac6ae0305f6c85f7af3aa8fa824a3278fe12aad537a2f0c13768571b`.
Core stayed unchanged during each batch, at
`501d35f3d17a79205e107679dc05dbea8fc1953b9aafebf4abc27368dcc52fe4`
and `b0a8f9a778bbffd08318961e5e7685387b8f231200afadb1bfaf95a475840bec`.
V2 also records and matches pre/post structure-module source hashes. V1 does
not have that complete source map. The whole production worktree was not
frozen; other development/default tests and shared host load were present.
These are checkpoint diagnostics, not paired-bootstrap/global acceptance.

The separate executed cProfile command ran the actual frozen
`scalar_write_64` prepared workload10 times (12,800 writes), using the same
interpreter/preload; its retained profile is
`/tmp/plan20-core-diagnostic-20261029/scalar-write-profile.prof`. It identified
repeated dirty bookkeeping, array-sized scalar undo and nested transaction/
source work. Refinements retained raw writes and BaseException undo while
removing duplicate scalar context/selection work, scalar array temporaries,
redundant materialized-group seeds and unnecessary full graph enrollment for
isolated registration/local Bond append. Getter point estimates improved to
near1, but the large write/construction costs remain a performance failure.
The parent recorded the bounded escalation evidence separately; this review
neither approves an exception nor describes the later engine as validated.

Before Stage3 completion or Stage4 entry, the parent still needs the following
actual evidence, read against the current verification map and main-plan
Sections10–12 rather than this checkpoint's passing subset:

1. Freeze production bytes after engine/decoder/outer integration, repeat the
   writer/read/restore inventory, public-interface/record budget and architecture
   ratchet, and update stale preparation-only verification-map statuses only
   when final evidence exists. No new debt/allowlist exemption is authorized.
2. Execute the map's exact five existing Stage3 targets (`test_core.py`,
   `test_molecule_structural_lifecycle.py`, `test_native_molecule_buffers.py`,
   `test_stereo_level1_lifecycle.py`, `test_topology_index.py`) plus the complete
   new/affected contract set, then the default suite. Carry constructor/assertion
   migrations and genuine defects as explicit old-to-new mappings. Include all
   newly affected cases in Python3.10–3.14; the old394-case preparation matrix
   and this322-case Python3.11 run do not establish that matrix.
3. Close the migrated storage/transaction gates on final bytes: raw Atom19/
   Bond3 preservation, buffer isolation/presence/provenance, scalar/batch error
   semantics and eager rollback, remote-first dependencies, hidden partitions,
   UID/remap/detachment, complete factual builder with one actual index build
   and no implicit chemistry, same-target imports, opaque metadata, FF original
   identities/stale rejection/full restoration, all nine frame domains, and
   coherent Molecule/Atom/Bond/joint-source pickle/deepcopy. Retain the concrete
   constraint snapshot and mixed local append failure regressions.
4. Run the affected installed-wheel/package/native packing and fresh-process
   import-order checks outside the checkout with recorded wheel/Python/native
   identities and harness flags. New runtime modules must be packaged; a source
   pass is not installed evidence. Any licensed/external/SDK tier is reported
   by its actual execution/skip/absence, never inferred from Python tests.
5. Rebind the frozen performance tasks to the actual sole implementation and
   measure equivalent scalar/batch, complete write-plus-derived-read, factual
   construction/I/O, remap, full19 float64 and separately selected export,
   native packing, FF/rollback and memory work with the frozen allocation,
   warmups/samples and outcome accounting. Retain all failed/no-output attempts
   and uncertainty. The costs above cannot fit the existing rare-exception
   policy. Required affected-family parity remains open; final global/per-family/
   end-to-end and Python-matrix performance acceptance still follows the main
   plan and frozen manifests, including the independent71-workload weights.
6. Record affected scientific observations under the selected unchanged route
   and the verification map's declared tiers; final O0 seeds42/43/44, dynamic
   eligibility and MCA/site/geometry/energy reporting remain separate from
   technical tests. No fixed187/182 success criterion, algorithm substitution
   or reduced work is introduced to close a technical/performance gate.

Final lazy scientific reads/conflict handling, immutable-view/ConstraintSet
ownership, canonical stereo decomposition, Crystal isolation and source-versus-
generated frame/result lifecycle still belong to their already assigned
Stages4–7. Stage3 must accurately certify its own migrated mechanism and
current staged semantics; this checkpoint neither implements those later
stages nor waives their final gates. Independent Crystal internals and arbitrary
outer-graph lifetime remain outside the table snapshot proof as stated above.

A final P0 review found that dynamic Bond endpoint access allowed
`bond.atom1 = value`/`atom2 = value` to create a public instance shadow outside
the private endpoint tuple. Explicit read-only data-descriptor properties now
retain the same names, owner lock and sole `_atoms` tuple. Same-owner and
foreign assignment both raise AttributeError before changing membership,
topology or facts. This is the intended existing read-only endpoint contract;
other AtomSeq/view interfaces remain with their assigned later-stage work.

The subsequent executed command

```text
python -m pytest -q --import-mode=importlib tests/test_cheminfo/test_plan20_carrier_cutover.py tests/test_cheminfo/test_bond_metadata.py tests/test_cheminfo/test_topology_index.py tests/test_cheminfo/test_stereo_level1_lifecycle.py tests/test_cheminfo/test_stereo_level1_working_copy.py
```

reported **71 passed in1.28s**. It includes the four endpoint rejection cases
and earlier carrier/constraint snapshot tests. At that checkpoint Core SHA-256
was `e6474fdc8e61e0a85c3b91069f045f18af672d81fe8c2b47173bd27919ec0648`
and the carrier test SHA-256 was
`413263b0317ca4c06797bd10741618ad2e7125cfc2eaa69c9ba57a56d08b1da9`.
The later engine journal
refinement must additionally retain mixed scalar/bulk overlap, storage growth,
remap/removal and currently interrupted write restoration tests. The parent
owns its final frozen-source and matrix evidence; neither this71-case run nor
the earlier322-case run certifies that later engine implementation.

## Native ingress architecture adjudication — 2026-10-29

This is the bounded Sol-max architecture review requested by the parent after
the measured Python-engine refinement. The reviewer owns only this document
and `../native/entry_review.json` at this entry. No production, main-plan,
build or test source is changed by the review. The full current main plan and
four normative companions were reread, all3498 lines, including repeated
bounded reads where combined tool output was truncated. Their five digests
still match the entry table above. The new JSON binds the actual dirty-source,
build, test and diagnostic identities; it also records the parent-owned gates.

HEAD at review is `caf959eda9c68b5ef1d2cbd60dc570d0a273ab18` on
`implement/plan20-fact-rule-architecture`. Stage0–2 are complete and Stage3 is
incomplete. The Python table/commit implementation is the active checkpoint;
there are no fact-native source or extension files yet. The parent inspected
and froze all production Python bytes for the default run: **4349 passed,
123 skipped,3 xfailed,49 subtests,10 warnings in244.26s**, runner246.11s.
Its exact command/environment and before/after hashes are in
`.plan20-artifacts/20261029/stage3/production-exploratory/default-python-engine-frozen/command.json`.
The log and record were inspected in this review. This is Python3.11 source
evidence before native implementation, not native/matrix/wheel acceptance.

### Decision and measured reason

Accept one private `hotpot.cheminfo.structure._fact_native` extension as the
implementation of invariant-preserving storage ingress, preparation and
installation. Its sources belong in `structure/_native`, use the existing
pybind11 build, and directly address the existing sole NumPy owners. It is not
a scientific rule, graph/geometry kernel, generic transport, native Molecule
mirror or alternate layout. It adds no public symbol, field, setter, result,
storage mode, backend selector or compatibility path. The current scientific
algorithms and their Stage5/6/7 cutovers remain unchanged.

The boundary must cover normalization and owner/row checks as well as stores.
Astra's two retained paired diagnostics give scalar8/64/256 ratios of
27.065/28.816/27.937, batch8/64/256 of14.368/3.328/2.001 and
write-plus-derived-read64 of11.600 against B20. All140 measured attempts and
42 warmups completed; the separate invalid `scalar_write_4096` worker request
produced no output and remains recorded. Shared-host diagnostics do not prove
global acceptance, but these large regressions remain unresolved outliers.

The inspected scalar/batch profiles locate repeated Python preparation,
N05 allocation/validation, row loops, transaction wrappers and physical undo;
writer-only compilation retains those costs. A new fresh-process isolation
probe on the current bytes measured10.712us per actual formal-charge write
and0.218us with only the engine replaced by a process-local no-op. The latter
still executes `MolBlock.__setattr__`, dict construction and dispatch; it
already consumes about56% of the paired B20 scalar64 budget of0.388us.
This is a non-equivalent diagnostic lower bound, not a benchmark gain or proof
that native code will meet parity. It justifies including compiled canonical
Atom dispatch in the bounded migration and measuring the complete public call.
Raw profiles,10 samples after3 warmups, source hashes and import tracing are
retained at `/tmp/plan20-native-entry-review-20261029/`.

### N01/N05 and the exact private entry boundary

N01 keeps its one physical `_SourceIdentity(owner, dependencies)` definition.
Prepared or long operations check the captured owner and every named content
domain using the existing canonical implementation. An ordinary immediate
scalar has the same owner-only N01 meaning: capture the token under the owner
RLock and check that identical token after conversion. No full-molecule digest,
cached digest, revision or additional owner field is introduced for that case.

N05 keeps the actual seven-field `_AtomFactUpdate` in the same order:
`source, rows, atom_uids, fields, value_bytes, presence, sources`. Existing
construction, immutable bytes, `values`, pickle/deepcopy and joint source-token
semantics remain. Batch preparation returns that actual record so Core can
reuse it for the existing element loop and explicit downstream reinstall.
Native preparation may use a trusted constructor after it has validated and
sealed the values; it must not repeat Python validation loops afterward.

For synchronous immediate one-field ingress only, a const eight-byte value
plus row/UID/owner/metadata on the native call stack is the execution
representation of that same closed N05. It cannot escape, be attached to a
carrier, become another accepted update type or substitute for an externally
supplied N05. This is a private allocation refinement, not a new scalar schema.
The actual scalar normalizer still returns N05 and reuses the same converter.
If the implementation cannot preserve those distinctions, retain the actual
N05 object and report its measured cost; an unsafe bypass is not an alternative.

The frozen entry contract is:

```python
_commit_atom_scalar(atom, field_index: int, value, *,
                    producer: int, present: bool = True) -> None
_commit_atom_facts(mol, update: _AtomFactUpdate) -> None
_commit_atom_columns(mol, atoms, columns) -> _AtomFactUpdate
_AtomFactTable._write_prepared(selection, prepared, *, transaction=None) -> None
```

These are private entry functions, not additional user-facing methods.
`_commit_atom_columns` is called under the existing same-owner outer
`_FactTransaction`. Core continues realizing an explicit `atoms` iterable
before entering that transaction; the native function receives its tuple or
None. It selects current members, normalizes only the approved aliases/shapes,
seals bytes once, checks the complete selection after all conversions, and
uses the one installer. `atoms=None` uses current ordered membership, not a
second row map. Public duplicate/foreign/detached, readonly/unknown, empty,
broadcast and overlapping-alias behavior stays as already approved.

All three molecular entries and standalone table `update` reach one native
physical installation/restoration implementation. `_write_prepared` becomes
a thin call; its Python raw-store/undo loop is removed in the same cutover.
Standalone table validation retains its own existing input/metadata contract;
it gains no artificial Molecule identity or molecular dependency behavior.
The prepared N05 boundary never round-trips sealed bytes through scalar read
codecs. Float64 raw values, NaN payloads, signed zero, heterogeneous metadata
and writes that compare equal retain their existing meaning. There is no
raw-equality skip, integer narrowing, NaN canonicalization or optional-result
encoding change.

Canonical field indices, block addresses, presence bits, producer codes,
permissions/aliases, dependency effects and status values come from the one
`_fields.py`/`_dependencies.py` declarations. A one-time immutable compiled
lookup of those declarations is code configuration, not a new schema owner or
runtime registry. It may retain no Molecule, Atom, table, ndarray, UID mapping
or result reference. Direct writes respect actual dtype/shape/strides and
writeability of every touched numeric and metadata/status buffer. Resolve the
current table and hold strong references after conversion and before stores;
a callback-induced growth must not leave the writer pointing at an old block.

### One lock, journal and publication contract

1. Capture current Molecule/token, member UID/UID owner, row and sole-index
   binding under the same RLock used by readers. Conversion may invoke Python.
   After the last such callback, recheck `atom._mol`, token, `_uid_owner`, UID,
   row membership, topology-row correspondence and all requested N01 content
   dependencies before binding writable buffers or making any store. Reusing
   an equal UID from another owner never establishes membership.
2. With a same-owner outer transaction, enroll one complete physical undo
   entry in its existing ordered journal before the first store. Include raw
   values, sources, presence and every changed status/seed membership. Native
   failure restores the interrupted entry and marks that outer transaction
   failed, even if an enclosing caller catches the exception. The original
   journal still governs final rollback of prior writes and publication.
3. Without such an outer transaction, an ordinary immediate scalar may use
   the same install/restore primitive with a call-local fixed-size undo guard
   and the same RLock. The bounded group count is obtained from declarations;
   no Python transaction object, ContextVar cycle or carrier-owned journal is
   needed merely to store one field. This is the same transaction contract
   with a shorter lifetime, not a second transaction implementation.
4. Preflight readonly storage and allocate/capture all necessary undo before
   stores. Handle Python `BaseException`, `KeyboardInterrupt`, `MemoryError`
   and translated native allocation failures. Preserve the original exception
   after restoration. Restoration uses the captured physical owners and the
   same native restorer, not a second Python store loop or public setters.
   Seed add/remove restoration is part of that proof, including failures
   during enrollment; a const numeric value alone is not sufficient undo.
5. Keep the GIL for this first native commit slice and use the RLock as the
   consistency boundary. No user callback runs between the first direct store
   and completion of an immediate native install. The existing eager element
   and Bond bodies may still run Python inside their outer transaction and
   read proposed values; they retain their present ordering and complete undo.
   Bulk/eager/topology operations keep their outer transaction. UID allocators
   are never rewound. Overlap, growth/remap and repeated writes retain ordered
   physical-buffer replay with work proportional to the touched write set.

### Core dispatch, loading and packaging

Compiled Atom canonical dispatch is within this bounded design. Prefer a
native implementation of Atom's existing `__setattr__(self, key, value)`
dispatch over introducing a descriptor class for every field. It calls the
same immediate scalar entry directly for ordinary numeric fields, so no
one-key dict, Python `_store_atom_fields` layer or repeated local import remains
on that steady-state route. Keep the explicit `Atom.n` getter and all existing
getters. `atomic_number` still enters the existing `_atomic_number_setter`
transaction and scientific body. Sparse isotope/frac_valence and nonnumeric
attributes keep their current controlled/default behavior; Bond keeps its
existing dispatch. Descriptor-based realization is acceptable only if it
preserves these same semantics and does not add schema or record types.
The installed dispatch must bind `self` correctly: an ordinary extension-module
function assigned to a Python class does not automatically supply method
binding. A CPython method descriptor such as `PyDescr_NewMethod` can provide
that binding without an additional runtime class. Test both direct and normal
attribute assignment on every supported Python version.

The first factual operation lazily imports the one module and caches only
module/function/class bindings. A bootstrap dispatch performs no numeric write
itself, then delegates to the sole implementation. Missing or incompatible
extensions raise an explicit ImportError with build/install guidance before
facts change; no Python fallback or selectable old path remains. Loading the
structure modules themselves must not import `_fact_native`.

A fresh-process trace of `import hotpot; import hotpot.cheminfo.structure`
on the reviewed source recorded **zero Molecule/Atom/Bond constructions**.
Thus no current module-level molecule indirectly triggers the proposed loader.
The trace did observe the pre-existing geometry extension/eager toolkit debt;
it does not certify the final all-native/all-toolkit lazy-import gate. Repeat
the fact-native absence/presence test after the cutover and from installed
wheels. If a changed importer starts constructing facts, repair that concrete
module-level consumer; do not special-case import stacks or grow the allowlist.

The minimum build addition is a C++17 `Pybind11Extension` for
`hotpot.cheminfo.structure._fact_native` in the existing `setup.py` list, with
one initial `structure/_native/facts.cpp` translation unit and a private
`structure/_fact_native.pyi`. Add their source/stub paths to MANIFEST/package
data. `pybind11/numpy.h` needs no NumPy C-header build dependency or new CMake
driver. No RDKit/Open Babel link, standalone SDK, Graph storage or shared
`NativeMoleculeBuffers` rewrite belongs in this extension. Record any actual
C++ helper structs separately in the budget; no new exported pybind class or
Python data envelope is needed by this design.

### Failure evidence and execution ownership

| Existing evidence / risk | Required native equivalent |
|---|---|
| `test_interrupted_write_restores_current_earlier_and_uninstalled_columns` and `test_metadata_write_interruption_restores_numeric_presence_and_source` in table tests | Deterministic interruption after each relevant value/source/presence store; retain every original raw/metadata restoration assertion for scalar and bulk standalone writes. |
| `test_commit_restores_write_that_raises_after_mutation` in molecular atomicity tests | Value/source/presence/status failpoints through actual public scalar and batch ingress, plus both seed add/remove boundaries; KeyboardInterrupt and MemoryError. |
| ndarray subclass `__setitem__` instrumentation | Native pointer stores bypass this hook. Map it explicitly to operation-local native failpoints; do not delete the underlying evidence or claim the old hook was exercised. Readonly preflight must still precede even the first store. |
| caught nested failure, producer inheritance, overlap/growth/removal tests | Same journal enrollment/poisoning, named producer precedence, original physical owners, membership/topology restoration and burned UIDs after native writes. |
| conversion/reentrant source changes | Mutating `__float__`/`__array__`/iterable inputs, foreign/equal UID, stale domains/rows, detachment and callback-induced growth/replacement; reject or bind the valid current owner exactly as specified before any native store. |
| serialization, aliases and concurrency | Existing immutable/joint N01/N05 round trips, NaN/-0/raw fractional values, approved shape/error/permission cases, all-reader/source capture lock tests; no writable owner alias or speculative publication. |
| extension availability/build | Fresh processes before/after first factual operation; missing/wrong-ABI extension fails explicitly; source and installed-wheel coverage for every supported Python ABI. |

Failpoints are private, deterministic and operation-local, disabled in normal
calls. They invoke no arbitrary Python callback after a store and introduce no
retained per-Molecule test mode. Both native and Python errors must traverse
the real installer/restorer. The explicit test mapping is reviewed before
replacing subclass hooks, and full affected tests run after the final bytes
are frozen. The existing466 focused passes precede this implementation.

Implementation partitions, to be activated by the parent's next assignment:

- The existing bounded engine-performance task owns the new native source/stub,
  `_commit.py`, `_mutation.py`, `_atom_table.py`, required declaration plumbing
  and `tests/test_cheminfo/structure/test_fact_native.py` plus the affected
  engine/table/dependency/atomicity tests. It must not edit Core or build
  files. Keep the complete immutable prepared contract and one installer.
- The assigned Sol-max Core owner coordinates `core.py`, carrier/permission
  tests and dispatch/normalizer integration after the private hook freeze.
  Preserve the scientific bodies, batch element order and explicit downstream
  facts. This review itself does not activate production editing.
- The parent owns `setup.py`, MANIFEST/package data, five-ABI builds,
  installed-wheel/import tests and the maintained budget/interface/verification/
  implementation-log records. No child edits those owners implicitly.

The parent first records the reliable Python-engine checkpoint with its
unresolved performance results, then coordinates native integration. No final
checkpoint may retain two active normalizers/installers, a Python fallback,
new architecture debt or an unbuilt extension. Required final evidence still
includes the five existing Stage3 targets and all new/affected contracts on
Python3.10–3.14, default suite, wheel/sdist/native identities outside checkout,
the unaffected standalone boundaries and the declared scientific reports.
The71 frozen performance entries, their individual outliers, each family and
integrated workflows remain authoritative. Primitive ingress/store timings
diagnose cost; they neither replace the global comparison nor excuse these
large regressions. Stage4 and all later-stage work remain outside this task.
