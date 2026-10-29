# Stage3 factual import record and builder review

Status: private design frozen at `5a5006f6c9964e6106fdce16e4bd3b56f3651345`;
record/codec implementation is a separately testable next checkpoint. No
production builder, table installation, converter cutover or Stage3 completion
is established by this document. It changes no public signature. The public
names remain exactly `updata_atoms` and `atom_matrix`.

## Entry assessment and scope

Before this architectural revision, the main plan and all four normative
companions were read completely in bounded ranges: main1–2448, protocol1–284,
scope audit1–521 including the appendix, alignment1–177, Plan21 1–68; 3498 lines
in total. Oversized main-plan output was recovered by rereading281–450,
451–610 and1291–1395. The accepted concrete decisions and affected semantic
entries of `reviewed_interface_contracts.json` were also inspected.

| Record | SHA-256 at this review |
|---|---|
| `fact_rule_architecture_implementation_plan.md` | `afb13f1a62f7d8db8a0fa8e893c4d1322e18d73edb785b0ad649c1eb02621f4b` |
| `chemical_type_interface_contract.md` | `d0ac2cff4c15f525236b00e254211020423e90f7687c1a640bfe28c7fee45bfc` |
| `data_object_scope_audit.md` | `72f935922c623948cc29d781c2eb77e38bcc8562cf368496a5cd7fb0599f1638` |
| `plan17_plan20_execution_alignment.md` | `e5549116f1409fb46b5fd2bdc82c0b20a34297989475eb66a119ac221a9d4a0b` |
| `../21_metal_cluster/metal_cluster_implementation_plan.md` | `b423d77b2ea290e847725570cf8023a90ddbbcca7e14d1796ce4a1ec3f6be83d` |
| `stage1_contract_decisions.md` | `fa647efe5e827cb16c207b70312bf6a708659589b41d5e87caaec7b6d0ed07e9` |
| `reviewed_interface_contracts.json` | `9a15c89feba940294200c78b6deb076682dc38aa01d9a9c4d0c4d66233bfabb8` |

The implementation branch is `implement/plan20-fact-rule-architecture`.
HEAD was the full SHA above and the worktree was clean at entry. Concurrent
parent-owned `Atom.mol`/UID-test edits subsequently appeared and are outside
this writer's ownership. The inspected shell Python was3.11.16 at
`/home/zhangzhiyuan/usr/conda3/envs/hp-usage/bin/python`; this is environment
identification, not a chemistry-validation run. At entry the last recorded push was
`40312df7c678dc193a9104248f54819ac1dc062a`, followed by `d7989b3` and `5a5006f`.
The next coherent commit triggers the three-commit push instruction; no
artificial commit is needed for this review. The concurrently completed
readonly-parent checkpoint is `ce41629`; its execution and push evidence belong
to the parent-owned implementation log.

Stage0–2 are complete. At HEAD, each Atom still owns its legacy19-slot array.
The private table, N01 source identity and N05 immutable update candidate exist,
but complete molecular commits and the builder do not. Prior4096 default
passes and273/version source passes are predecessor evidence in
`stage3_atom_update_evidence.json`; they do not validate the proposed installer.
Table/writer/rollback integration, readonly detached rows, dependency reads and
integrated performance remain open. No Stage4, Plan17 backend replacement,
Stage7 result/frame ownership cutover or Stage13 protocol is started here.

## Selected migration

Freeze and test the complete immutable import record and its pure codecs first.
Connect it to production only with the sole `_AtomFactTable` and the connected
writer/rollback migration. The existing per-row arrays cannot retain the
record's per-field presence and producer codes. Accepting those fields and
discarding them, or retaining a temporary metadata side owner beside the old
arrays, would break the reviewed ownership contract.

A narrower builder accepting only legacy-representable facts could be a valid
different preparation task. It is not selected for complete N02: it would
require a second input-schema transition and would leave the exact source
presence problem unsolved. There is no per-Molecule storage mode or selectable
old/new installer. An isolated record module has useful executable codecs and
tests; it must contain no unimplemented installer placeholder.

## N02 exact private contract

One `MoleculeRecord` is defined in `structure/_builder.py`, with
`@dataclass(frozen=True, eq=False)` and empty public exports. Its bytes
constructor takes all22 fields below in this exact order. It is a task-local
factual import, not a general scientific-result DTO, rollback image or carrier.
It has no source Molecule reference, source-owner token, UID table or persistent
index. New-member UIDs belong to the destination allocator.

| Field | Exact content and meaning |
|---|---|
| `atom_count` | Nonnegative integer N; one row order shared by all Atom fields. |
| `atom_values` | Immutable little-endian float64 bytes, C order `(N,19)`, in `ATOM_FIELDS` order. Preserve raw slots, including fractional values and wide float64 display IDs; do not apply scalar read codecs. |
| `atom_presence` | Immutable little-endian uint32 bytes `(N,)`; only the existing19 presence bits. Explicit zero and absent remain distinct. |
| `atom_sources` | Immutable uint8 bytes `(N,19)` using exactly `ATOM_SOURCE_CODES`. Producer code is independent of presence. |
| `atom_isotopes` | Sparse tuple of `(row, int_or_None)` for the existing standalone legacy isotope attribute; absence from the tuple means no such attribute. No new public isotope property or duplicate table column. |
| `atom_frac_valences` | Sparse tuple of `(row, float)` for the existing supplied `frac_valence` observation. No new result class or getter. |
| `bond_endpoints` | Tuple of B ordered `(first_row, second_row)` integer pairs. Keep donor/direction order; do not sort endpoints. |
| `bond_values` | Immutable little-endian float64 bytes, C order `(B,3)`: raw `bond_order, constraint, id` slots. |
| `bond_partitions` | B integers:0 active,1 hidden-metal,2 hidden-covalent. Preserve order within each current partition; removal of hidden storage remains Stage7B. |
| `bond_kinds` | B strings from the existing `BondKind.value` domain: single/double/triple/aromatic/zero/dative/unknown. No new enum. |
| `bond_directions` | B optional strings; retain existing direction/source semantics. |
| `bond_sources` | B optional source-namespace strings. |
| `bond_metadata` | B immutable encoded values, each None or a dict tree. Preserve None versus an empty dict, nested container kinds and source data. See codec boundary below. |
| `charge` | `(present, real_scalar_or_None, source_namespace_or_None)`. Canonical absence is `(False,None,None)`; explicit zero stays present. No formal/partial-charge summation. |
| `spin_mult` | The same declaration triple for an explicitly supplied molecular spin. Do not copy a selected frame's spin into a declaration. |
| `cell` | None or the six values `(a,b,c,alpha,beta,gamma)` in existing angstrom/degrees order and characterized convention. Reuse the one existing Crystal implementation at installation; no matrix property or second cell type. |
| `stereo_payload` | None if not supplied, otherwise the existing protected `StereoPayload` class and its exact nested schema. No new stereo envelope/class identity. |
| `legacy_frame_values` | Exactly nine immutable encoded values, in the order listed below. Every field retains its scalar/array/None form, dtype and shape. This does not classify legacy frames as generated or explicit. |
| `legacy_frame_index` | Integer selected index, including existing negative indexing; no codec normalization to a different index. |
| `angle_constraints` | Ordered Atom-row triples for already persisted true angle constraints. No generated full angle cache. |
| `torsion_constraints` | Ordered Atom-row quadruples for already persisted true torsion constraints. No generated full torsion cache. |
| `identifier` | `(present, str_or_None)` for the existing molecular source label; `(False,None)` is absent and `(True,None)` is an explicit None. Atom display IDs remain in their numeric slot. |

All B columns have B entries. Sparse Atom rows are unique within each field;
the same row may have both isotope and frac_valence. Endpoint/sparse/constraint
indices must address this record's Atom order. The record checks byte lengths,
tuple widths, ranges, legal producer/partition/kind codes and structural schema.
Charge/spin scalar types are Python bool/int/float and NumPy bool/integer/real
floating scalars, preserving values without narrowing an extended-precision
scalar. Complex values and ndarrays are not declaration scalars. A present
declaration's source is a string or None when unknown. No source is fabricated.
It does not impose valence, charge conservation, positive scientific values,
finite-energy acceptance or a new interpretation of source chemistry.
Self-edge/parallel-edge and destination membership validation belongs to the
one later topology installer, rather than a second graph implementation here.

The exact pure factory is `_make_molecule_record`, with keyword-only parameters
matching the field names above except `atom_count`, which it derives from
`atom_values.shape[0]`. All parameters are required. The four numeric inputs
`atom_values`, `atom_presence`, `atom_sources`, `bond_values` are arrays at this
boundary; the factory validates their domains before encoding their bytes.
`bond_metadata` and `legacy_frame_values` are raw supported nested values at the
factory boundary and canonical encoded values in the record. No constructor
defaults invent presence, source or Bond kind. Empty/absent inputs are supplied
explicitly by the producer. No matrix convenience-method family is added.

Presence and producer information is never derived from numeric nonzero tests.
RDKit, Open Babel, CCDC and cclib adapters know what their source actually
supplied. Electron/reference columns calculated by an existing adapter remain
attributed to that existing producer; moving the calculation into the builder
is forbidden. A legacy raw-array API supplies all19 slots as raw input, while
omitted public constructor defaults remain absent under the accepted A2 policy.
An old array alone cannot recover lost source presence; do not fabricate it.

### Nested immutability and serialization

Two private helpers, `_freeze_value` and `_thaw_value`, encode/decode only the
following value grammar, using tagged tuples rather than new record classes:

- None and Python bool/int/float/str/bytes;
- list, tuple and dict with recursively encoded entries, each retaining its
  own tag; dict insertion order and supported key/value contents survive;
- NumPy scalar with dtype string and bytes, retaining its scalar tag;
- ordinary non-object ndarray with dtype string, original shape and C-order
  bytes. A zero-dimensional array remains an array, not a Python scalar.

The current grammar explicitly rejects object dtype, structured fields,
subdtype and dtype metadata. `dtype.str` loses structured field names; it must
not be called a lossless encoding for those values. No current inspected
scientific/import producer requires them. A real future requirement must
extend the explicit dtype descriptor before its dependent path is connected.
Unknown Python objects have no opaque-pickle or mutable-leaf fallback.

The bytes constructor accepts canonical tagged values and validates/rebuilds
their tuples. It does not encode those tuples a second time. Stereo values are
reconstructed with the existing protected classes, original field order and
tuple sequences, without interpreting descriptors or changing source codes.
Frame/Bond metadata decoding returns fresh containers and ndarray headers;
an ndarray header's shape/dtype mutation cannot mutate the saved record.
Mutable arrays needed for installation are explicit operation-owned copies.
Setting an ndarray write flag alone is insufficient evidence of isolation.

`__reduce__` reconstructs through the normal validating bytes constructor;
pickle and deepcopy must retain the same canonical field/byte content and
immutability. A fully sealed record may return itself from `__deepcopy__`.
There is no opaque owner token to rebind in N02. This current-version private
pickle contract introduces no historical core-pickle compatibility promise.

Arbitrary caller objects currently accepted inside `bond_source_metadata`
are not silently made supported by this limited codec. Before migrating a
public copy/import path, establish that all of its supported metadata can be
encoded without loss or obtain a concrete boundary decision. Failing early in
an isolated private codec does not approve a new restriction on public Bond
construction/copy. Annotations, opaque runtime contexts and backend objects
are outside this chemical record. Copy controllers separately preserve the
accepted annotation-container/value-reference behavior and install that
prepared container atomically; the builder does not freeze user annotations.

### Raw Bond and legacy frame coverage

`Bond.attr_dict` uses bool/int read codecs for constraint/id. A raw constraint
slot such as2.5 becomes True and a fractional raw id is narrowed. It is useful
for ordinary public projections but is not N02's raw codec. Capture the three
slots directly, and carry kind/direction/source/metadata separately. Keep
explicit aromatic/dative/zero/unknown combinations exactly as represented;
do not repair them by rerunning `_bond_order_setter` or inferring kind anew.

The fixed nine-frame order is:

```text
coordinates, energy, partial_charge, gibbs, force,
zero_point, spin_mult, thermo, capacity
```

Current coordinates live in `atom_values` XYZ. The frame coordinate array is
the distinct existing collection, not another stored current frame. The cclib
reader writes, among others, scalar Gibbs/zero-point/spin and array energy,
force/thermo/capacity; copying only coordinates and energy is insufficient.
Preserve `None`, Python/NumPy scalar, `(K,)`, `(K,N)`, `(K,N,3)` and any other
actually supported source shape without flattening, stacking, broadcasting or
inventing a K axis. The selected index is retained separately. The current
`_partial_charges` constructor typo is not a tenth logical field; no producer
inspected writes a non-None plural field. A discovered non-None case requires
explicit characterization rather than silent loss. Source/generated role
separation and the known add/empty-iteration defects remain Stage7D.

## Future initialization and single installation boundary

The ordinary constructors keep their characterized registration behavior.
Bulk import must bypass them: `Atom.__init__` registers immediately and may
compute electron configuration, while Bond construction requires an already
published membership index. Calling either repeatedly is not the one-shot
builder described by Stage3.

Use small private initialization primitives for raw handles. An Atom primitive
uses the one physical Atom class and binds `_mol`, `_member_uid`, `_uid_owner`.
Rows are resolved through the one topology index; no per-handle row cache,
table pointer or second UID-row map is added. This primitive does not append
membership, allocate a second UID, invoke setters or install an ndarray owner. A Bond
primitive binds existing local endpoints, one reserved UID, the three raw
slots and its complete metadata without invoking chemistry. Public direct
constructors call the same binding primitive where applicable and perform
their normal registration once. These are functions, not per-row DTO classes.

`Atom.mol` can become a readonly property independently before N02 integration.
The inspected production Atom assignment occurs only in `Atom.__init__`;
foreign insertion/replacement already copy rather than steal. Constructor and
raw initialization bind `_mol` directly. Other classes' `self.mol` assignments
are unrelated. Existing detached-parent inspection and current-version
pickle/deepcopy must remain covered; no writable reparenting API is introduced.

The private signatures retain the plan's fresh-build entry and add one
target-preserving internal installation entry:

```python
def _build_molecule(record: MoleculeRecord) -> Molecule: ...
def _install_molecule(mol: Molecule, record: MoleculeRecord) -> Molecule: ...
```

The first allocates one destination and calls the second. The second keeps
`mol`, its owner token and lock. All failure-prone decoding, source-value
translation and candidate storage preparation precede publication. Reserve
monotonic UID ranges under the destination lock; failed reservations need not
be reused. Build local handles/Bonds and one deterministic topology index from
the complete candidate membership. Validate ownership/endpoints there once,
including active versus hidden semantics; retain no helper UID-row map.
Publish table, members, index and supplied molecular facts together, preserving
or restoring the previous committed state on any BaseException. Readonly
detached rows and all metadata remaps must use the same schema.
Allocated UID high-water marks are the exception to restoration: they never
decrease, including after failure.

The pure record factory never reads a carrier. Its source-extraction caller
captures values, presence/source, relevant stereo/frames and any required N01
dependency identity under the same owner consistency boundary. Representation
identity must include the installed table's relevant presence/source metadata
when authority switches. Long operations recheck their captured source before
committing; N02 itself retains no live owner binding or mutable status owner.

The topology builder must be separable from `_update_topology`'s current cache
and conformer-clearing side effects. One final adjacency build is allowed;
per-Atom placeholder registration, per-Bond adjacency edits followed by a
second rebuild, and frame installation followed by an implicit clear are not.
No element/valence/H/aromatic/bond-perception/force-field/toolkit operation is
called by installation. Requested chemistry follows as an explicit named
operation. Ordinary scalar construction remains O(1), without routing every
isolated Atom through a whole-Molecule record.

Copy/components select retained facts and remap record rows before building
new local members. An add-component/topology transaction retains existing
destination handles/UIDs and uses the same raw initialization/table/index
primitives for new members. It must not rebuild the entire destination via a
fresh Molecule and replace existing handles. Likewise FF commit/rollback keeps
its real operation/snapshot records, existing Atom/Bond identities and owner;
it consumes these same preparation primitives rather than treating N02 as a
second rollback snapshot. Numeric values, presence, source, table layout,
constraints, frames and stereo are restored together. N02 does not precreate
N03–N09 or later result owners merely to name these operations.

## Converter identity, append scope and actual consumers

`obmol2mol(obmol, mol)` and `from_rdmol(rdmol, mol)` currently return that exact
`mol` object. Replacing the return with a newly built Molecule would change
observable identity. Keep their signatures and return identity by using the
target installation entry when they migrate.

All inspected in-repository converter callers and direct tests supply a new
Molecule. Nonempty-target behavior is a current defect: Atom rows append but
new Bond endpoints are interpreted from row0. It can connect old members or
raise after partial mutation. That defect establishes no correct append
contract. The first implementation/evidence cohort is structurally empty
targets. Before connecting a public converter, record its concrete
nonempty-target disposition and evidence in the interface decision ledger.
The minimum candidate is atomic rejection before mutation, directing
composition to the existing add-component operation. The owner's continuing
authorization to complete Plan20 applies to correcting endpoint corruption and
partial mutation; historical pending wording alone does not require another
permission request. At the dependent cutover, check the frozen supported uses
and treat a justified early rejection as the concrete technical bug fix. This
review does not establish an implemented exception contract, clearing of an old
structure or a new correct append capability. A genuinely new append contract
must resolve offset, declarations, cell, stereo and frames explicitly before
its implementation, without pausing unrelated authorized work.

Even a structurally empty target may carry declarations/cell/frame metadata.
Record absence and installer intention must be explicit. The observed legacy
effects below are the compatibility inputs, not permission to erase all target
state. Source facts must never be accepted and dropped.

| Actual writer/consumer | Required coverage before its production migration |
|---|---|
| `obconvert.obmol2mol` | One source decode; explicit numeric presence/source, raw Bond source data and existing electron columns. It sets source net charge, changes cell only if supplied, does not set stereo, and currently clears frames through topology refresh. Preserve target return identity and decide preconfigured-target effects explicitly. |
| `rdconvert.from_rdmol` | Preserve existing toolkit copy, stereo extraction before Kekulize, existing Kekulize/scientific route, source Bond semantics, optional Gasteiger-charge presence and conformer-coordinate presence. It sets net charge and replaces the whole payload; None versus an explicitly empty payload must remain distinct. |
| `plugins/ccdc_api/convert.py` | Raw19 fields, labels/ordering, supplied charge/coordinates/aromatic values and all source Bond facts. The current `c_mol.atom` use and absent return are existing adapter defects to characterize; no unexecuted licensed CCDC tier is called passed. |
| `_io/_io.py` cclib reader | Preserve declarations, all nine frame values, unit alignment, selected current frame and subsequent explicit link/aromatic/order/valence operations. Do not move those scientific algorithms into the builder. |
| `Molecule.copy/components/add_component`, insertion/replacement/removal and `AtomSeq.to_mol` | One explicit row/member remap; raw Bond/Atom slots, sparse attributes, stereo, constraints, source labels and retained frames. Apply the reviewed lifecycle policy at its owning step; do not reset validity or manufacture charge/spin/cell applicability. |
| `forcefields/working_copy.py` | Preserve original handles and complete failure restoration while replacing direct attrs assignments; keep all nine frame domains, selection and source values. Existing rollback records remain operational records. |
| `AImodels/mca/serialization.py::hotpot_mol_from_dto` | Decode the existing MCA DTO into the private record without another public DTO or a forced toolkit/chemical recomputation. Preserve the separate RDKit binary's scientific purpose and the sole selected MCA algorithm. |
| Model/native/PyG readers | Retain approved19-column and task-selected feature order/encoding. Stage3 does not introduce A2's Stage7C NaNs into an unchanged model, and CBond atomic-number-only extraction must not project all19 columns. |

The common-contract entries checked for this mapping are `carrier`,
`identity`, `coordinates`, `net_declaration`, `bond_representation`,
`constraints`, `copy_import`, `conformers`, `pair_observation` and
`result_projection`, including the supplemental `mol`, `identifier` and
`frac_valence` entries. The source Wiberg pair observations remain with their
current owner until their own complete migration; N02 neither transports P4
path caches nor grants permission to discard those scientific observations.
Existing `attrs_array` and `update_electron_config` stay until the A2 Stage5
cutover. Selected partial-charge ownership/absence encoding remains Stage7C.

## Reviewable checkpoints and acceptance boundaries

1. **Readonly parent:** independent constructor/back-reference change, with
   UID/detached/pickle evidence. No record or numeric ownership change required.
2. **Isolated N02 and pure codecs:** one new dataclass; no carrier writer or
   installer. Validate byte/value preservation, nested isolation, malformed
   schema rejection, deepcopy and current-version pickle. This is the next
   bounded implementation task and does not pass the builder gate.
3. **Coherent production authority and imports:** integrate sole table,
   all scalar/batch/raw/coordinate writers, lifecycle remaps, detached rows,
   converters, FF commit/rollback and one topology build. Remove their old
   direct-array writes together; preserve Stage5 scientific bodies until their
   assigned extraction. Public batch publication waits for the real commit.
4. **Stage3 closure:** complete its remaining dependency/interface/import
   gates and integrated performance/source/Python evidence. Record actual
   commands and results before marking it complete or entering Stage4.

Isolated N02 tests must include all19 distinct raw Atom columns, raw fractional
Bond constraint/id slots, explicit-zero versus absent masks, heterogeneous
producer matrices, directed/hidden Bonds, nested dict/list/tuple metadata,
scalar and differently shaped frame values across all nine fields, sparse
attribute row maps, existing stereo classes/field order and selected negative
frame index. Mutating every original input and every decoded container/header,
then pickle/deepcopy/redecode, must leave canonical record bytes unchanged.
No test can substitute a full logical-codec comparison for raw Bond bytes.

The integrated builder must additionally demonstrate one topology build and
zero chemistry calls; same-target converter identity; no state changes on
decode/ownership/endpoints/interrupted-install failure; non-reusing UIDs;
metadata remap and numeric/stereo/frame/constraint rollback; legacy constructor
behavior until Stage5; and unchanged model input encoding. Cover current and
detached handles plus repeated independent imports. The nonempty/preconfigured
target and opaque-metadata gates above must be resolved at that concrete step.

Use the Stage3 command selection in `stage_verification_map.json` and add the
actual builder/record tests to the complete affected Python3.10–3.14 selection.
Run the default suite with
`python -m pytest -q --import-mode=importlib tests` in the selected environment.
The five listed baseline files alone do not cover the new contract. Required
installed-wheel/native evidence remains governed by the affected integration;
source tests are not a wheel/build certificate. Isolated-record test success
does not validate converter scientific behavior or complete molecular
concurrency. Scientific changes, if observed, require their named reports;
no new scientific pass-count rule is introduced.

Measure complete factual construction/import and rollback against the frozen
B20 workloads, including decoding, UID allocation, presence/source, locking,
one topology build, numeric copies and peak memory. Keep standalone codec cost
as new-only evidence where no equivalent B20 entry exists. Scalar construction
and writes must not acquire N-sized record/table copies. Prior unfavorable
table/candidate measurements remain retained; there is no integrated
performance pass yet.

This writer performed document/source inspection and schema consistency checks
only. Proposed tests/installer steps above are not executed evidence. Any
record implementation, API disposition and actual test result is recorded in
the implementation log and verification map by its owning implementation task.

Executed review checks:

- `python -m json.tool plan/20_fact_rule_architecture/data_object_budget.json > /dev/null`
  exited0.
- `git diff --check -- plan/20_fact_rule_architecture/data_object_budget.json plan/20_fact_rule_architecture/stage3_builder_review.md`
  exited0. The Markdown file is new, so its whitespace is checked separately.
- A JSON comparison with `git show 5a5006f:plan/20_fact_rule_architecture/data_object_budget.json`
  verified that replacing just the edited N02 entry yields the original whole
  ledger. A Python AST comparison with
  `/tmp/plan20-molecule-record-draft/_builder.py` reported
  `22 N02 fields match draft`; every field also appears in the table above.
  This checks the proposed schema, not the draft's behavioral correctness.
