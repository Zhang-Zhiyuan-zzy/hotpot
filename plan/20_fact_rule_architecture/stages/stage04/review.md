# Stage4 mathematical extraction and view wiring

Status: implementation handed off with frozen production/tests; full Stage4
validation is still open. This implements the already authorized Stage4 scope.
The [entry record](entry_review.json) binds the complete3498-line normative
reread, clean production checkpoint `bdc6a8066095e81c415e1152b0f3e4717583ac83`,
accepted interfaces, predecessor evidence and file ownership. This record
changes no main-plan requirement, scientific algorithm or measurement policy.
Stage4 is not complete.

## Stage and path boundaries

Main-plan Sections6/9/10, the Stage1 migration inventory and accepted interface
decisions distinguish these changes:

| Work | This stage | Later owner |
|---|---|---|
| Existing graph/geometry Python algorithms | One mathematical implementation and atomic consumer updates | Stage10 moves native sources/build paths |
| Carrier-to-geometry conversion | `adapters/structure/geometry.py` | No duplicate geometry contracts |
| Graph spectrum | Chemical descriptor policy separated from reusable matrix mathematics | Stage5 owns the elemental-rule extraction |
| Core mathematical queries | Thin facades, explicit numerical inputs, on-demand non-Ring views | Stage9 relocates remaining carrier/view class definitions |
| Angle/Torsion constraints | Real Molecule-owned ConstraintSet before cache deletion | Stage7B finishes coordinate/Bond constraint ownership |
| Ring | Extract pure calculations while retaining the existing writer/cache boundary | Stage5 migrates writers and removes Ring caches together |
| Crystal | Parameter/matrix calculation only, exact characterized convention | Stage7A removes carrier/toolkit/back-reference state |

The existing `graph._graph_native` and `geometry._geometry_native` import names,
C++ source locations, shared CSR definition, stubs, namespaces and build recipes
remain unchanged. The old package directories remain necessary native package
homes. Their superseded pure Python algorithms are not copied or re-exported
as a compatibility tree. New mathematical modules address the retained native
entry points explicitly. No native binary or `_fact_native` change is planned.

## One implementation and explicit consumers

The existing pure modules move to `rules/mathematics/graph/` and
`rules/mathematics/geometry/`. `graph/spectrum.py` moves to
`rules/chemistry/descriptors/graph_spectrum.py`; its current scientific
calculation remains intact and imports the one mathematical matrix helper.
The mixed `geometry/convert.py` moves to the structure adapter. Mathematical
packages export numerical objects/functions only; carrier-aware conversion
functions are imported from their adapter by actual callers.

`InternalCoordinates` becomes the one implementation in
`rules/mathematics/internal_coordinates.py`, using the extracted graph helper
and explicit arrays. Crystal's `_matrix_to_params` delegates to
`rules/mathematics/cell.py`. The existing alpha convention is the angle
between the first two matrix rows; no conventional crystallographic correction
or new public matrix property is included.

Core query methods enter named operations, which prepare owner facts under the
existing lock, invoke pure rules and materialize caller-owned results. Until
Stage9 moves the remaining classes, facades pass their existing view or
Molecule materializer explicitly; operations add no imports back to Core.
Coordinate mutation such as
translation enters `operations/geometry.py` and the existing named coordinate
commit. Pure rules never receive a mutable carrier to edit. Existing graph,
cycle, path and geometry ordering and numeric behavior remain the reference.

## Source-bound views without repeated molecular digests

Reuse N01's owner binding and existing non-reused Atom/Bond UIDs. The sole
`_topology_index` already supplies a source capability: complete topology
installation replaces its dictionary; O(1) isolated-Atom registration changes
its membership size. A private task binding can retain that actual index
object and its captured membership sizes alongside N01. It introduces no
second index, molecular digest cache, mutation counter or public record.
Topology consumers recheck the owner, index identity/membership size and the
fixed member UIDs under the same owner lock. Rollback restores the original
index and membership, so an aborted change does not invent a new source.

Angle/Torsion/AtomPair membership uses fixed ordered references/UIDs; detached
membership remains inspectable. Geometry-dependent calculations additionally
check only the captured member coordinate values and applicable metadata.
Their relevant changed source raises the existing RuntimeError. Checking a
bounded view costs its member count, not the whole molecule. A coarse table
or iterator prepares shared topology/coordinate input once. Carrier facts
and dependency status remain owned by their current owners.

The public name remains `AtomPairs`; no AtomPairTable alias or exception class
is added. Pair mathematics and supplied scientific observations are separated:
removing a carrier's generated pair cache must preserve existing tmQM Wiberg
zero/nonzero observations and pair-feature order. Their source-observation
preservation is keyed by non-reused member UIDs independently of regenerated
views. It does not publish a new P3 result schema or assume that arbitrary
generated results survive copying. The existing vertex-count meaning of
`length_shortest_path` is preserved. The Stage7B/7C pair ownership and complete
collection/consumer disposition remain recorded at their assigned owners.

The actual active observation owner is `Molecule._pair_wiberg_bond_orders`,
one dictionary keyed by ascending UID pairs. The parent authorized the finite
private N02 extension `pair_wiberg_bond_orders=()`, appended as field23:
each supplied entry is `(first_row, second_row, float_value)`. Missing pairs
read mathematical zero without storage; a supplied zero is stored and survives
copy, subset, merge and current pickle. Complete FF input identity includes
these observations, preventing a stale working copy from overwriting them.
The [contract review](contracts/review.md) and [schema](contracts/schema.json)
record the complete table API, row/UID transport and remaining later-stage
result-owner work.

Current-version pickle/deepcopy reconstructs views against the restored
owner's actual sole index, in either graph-root order and when serialized
alone. It preserves stale topology/member/geometry state instead of rebinding
an old calculation as current. It does not pickle a second index dictionary.

## Constraint ownership precedes cache removal

At Stage4 entry there was no ConstraintSet implementation. Angle/Torsion
constraints lived in cached view dictionaries; the complete factual record and
closed `_commit_constraints` seam are available but do not constitute the
target owner.

Install one private Molecule-owned ConstraintSet for ordered Atom-UID angle
and torsion entries. Existing `.constraint` inputs project that owner; views
retain no independent constraint value. Coordinate constraints stay in the
Atom table and Bond constraints stay at their existing owner until Stage7B;
this change claims no early migration of those remaining fields.

Prepare replacement constraint content before installation. The existing
transaction journals the owner state and restores original references on
failure, preserving exception and outer-transaction behavior. Topology
record capture/install, copy, components, merge, current pickle/deepcopy,
source-identity capture and FF working-copy capture/rollback consume the same
entries and ordered UID/row remap. Entries whose required members disappear
are explicitly subset or rejected by the existing complete operation, never
silently replaced by another row. Hidden topology partitions remain within
their existing Stage7B policy. No previously absent native force-field angle
constraint behavior is added.

The actual owner is the frozen private `_ConstraintSet(angles=(), torsions=())`.
It is one added owner dataclass, counted separately from N01-N18; moving
existing mathematical definitions and removing a snapshot field retire no
dataclass. The current AST census is779 classes,351 dataclasses,13 Protocols
and2 TypedDicts. The data-object budget records the additional owner cost and
the corresponding adjustment to its historical forecast.

Undo retains the original pair dictionary and topology-index references.
Only scalar pair writes and actual fast index stores enroll per-key undo;
existing keys are restored and newly appended keys removed in reverse order.
Complete publication uses prepared replacements. Neither canonical pair
observations nor the topology index use `clear()`/`update()` reconstruction
during rollback. Real allocator rejection tests cover both owners in
mixed topology and FF publication failures, including the original exception
identity and restored old view capability. Other Python rollback work still
allocates; these tests do not establish failure-free rollback under every OOM.

Only after that connected writer/consumer migration are `_angles`, `_torsions`
and the generated `_atom_pairs` cache removed. `angles`/`torsions` materialize
views on request; `update_angles`/`update_torsions` are removed without aliases.
Ring storage/writers are deliberately left to their already assigned Stage5.

## Parallel file ownership

The entry record contains exact existing/new kernel files and48 initially
identified consumer files. Parent may independently assign either partition:

- Kernel/adapter partition: existing graph and geometry Python modules;
  `rules/mathematics/{__init__,graph/,geometry/}`;
  `rules/chemistry/{__init__,descriptors/__init__,descriptors/graph_spectrum.py}`;
  `adapters/structure/{__init__,geometry.py}`. No C++/.so/.pyi/build edits.
- Consumer partition: only the explicit `consumer_imports.files` list in the
  entry record, for path/import wiring. Preserve native import strings/source
  paths in native-specific assertions and do not weaken behavior assertions.

This Sol-max owner retains Core, Crystal, structure identity/commit/builder
and the new ConstraintSet, views helpers, InternalCoordinates/cell mathematics,
geometry operations, FF working-copy integration, new Stage4 contract tests,
Stage4 records and retirement of real allowlist entries. Parent owns global
log/map/index, validation and final Git submission. Workers share the
workspace and must accommodate concurrent changes without reverting them.

## Verification and public effects

Public effects are the authorized mathematical import migration, immutable
source/member behavior, removal of angle/torsion cache-update methods and the
constraint-owner projection. Keep `updata_atoms`, `atom_matrix`, existing
Angle/Torsion constructors, AtomPairs name, constraint payload/order and all
standalone native signatures. No chemistry/model/resource algorithm changes.

Use the existing Stage4 technical/native selection and its planned
`test_view_identity.py`, `test_constraints.py`, `test_crystal_convention.py`,
plus actual affected consumers, default tests and focused equivalent-work
performance. Record old cache assertions against their retained behavioral
purpose. Retire only proved-eliminated allowlist entries; record any new
implementation/record definitions as additions, not consolidation gains.
Stage3's observed performance/memory regressions and complete final efficiency
acceptance remain open at their owners. No new independently gated sub-stage
or final71-entry Stage4-entry gate is introduced here.

## Implementation evidence and closure limits

The [implementation handoff](implementation/evidence.json) retains all local
attempts, including two allocator-harness failures and the reproduced
branched canonical-tree extraction error. The latter was fixed by renaming
the recursive path parameter that shadowed the moved helper; AST comparison
confirms the original algorithm after explicit-input substitution. The last
large owner checkpoint passed450 tests; the changed view/graph selection
passed72. Final read-container and allocator repairs pass80 focused tests on
each of Python3.11/3.13/3.14 and eight allocator cases with actual
`PYTHONMALLOC=debug` on3.13/3.14. Their observed dict-bucket domains are OBJ
on3.11 and MEM on3.13/3.14; the helper wraps both actual previous allocators.
Freeze v2 preserves `bonds` as an independent list and pair-table `copy()` as
an independent plain dict. The parent owns the complete frozen default, supported-version,
wheel, native and performance gates and final Stage4 report.

The [architecture evidence](architecture/evidence.json) retires only13
actually disappeared entries, reducing debt281→268. Three exact numerical
value/snapshot initializer sites are recognized by receiver, path, class,
method, field and operation. Foreign, live-carrier, dynamic, buffer and
wrong-scope writes remain checked. No debt allowance was added.
The actual new pair owner is also registered in the canonical write check,
with mapping-store, mutator and owner-replacement regressions. All95 final
architecture checks pass. The original missing-guard diagnostic is retained.

The entry's document hashes remain historical identities of the full3498-line
read. After that read the parent mechanically bound the data-object audit's
90 source links to audit baseline `d9eb80e239f24a694a8cb23fd412eb6baf030cbb`;
its classification, line references and normative semantics were preserved.
The closure evidence records that later link-only document identity without
rewriting the entry record or claiming another stage-entry read.
