# Stage8B — Force-field ownership and connected workflow contracts

## Entry and complete owning scope — 2026-11-03

Root aligns Stage8B at `ff4373890356cd53d95e0ae57b115038bc540096`, after the
accepted and pushed [Stage8A CP311 owning completion](../../scientific-state/completion_20261103.json).
The effective [goal](../../../../goal.md), main plan Sections6/9.2/Stage8B/16,
Stage1 exact proposals, data-object audit/budget, current verification map and
integrated Plan12 requirements govern this work. Focused alignment follows the
owner's explicit direction; current companion decisions override historical
proposals. No user-owned untracked review or Task168 material is in scope.

The full Stage8B objective remains: pure physics/acceptance rules; carrier
capture and field-selective transport; force-field/trajectory/repair/atomic-commit
operations; thin Open Babel/native adapters invoking the one standalone
obWrappers engine; commands; connected xTB/plugin/pipeline finding-control,
method transport and sample containment. Native source ownership follows
semantics even when rules/controllers/bindings link into the same `_ob_native`.
No alternate engine, graph, model, trained-feature interpretation or scientific
algorithm is introduced. Later slices must finish these obligations before
Stage8B completion; the first slice does not substitute for the whole stage.

## B1 — Accepted D04/D05/FF-CLI consolidation

Root accepts these exact existing-type migrations before production authoring,
under the owner's full-authority mandate and effective goal. The earlier Stage1
D04/D05 paths and delegated-factory suggestion are superseded by this decision;
the objective and public semantic quantities are unchanged.

| Item | Sole final definition | Connected behavior |
|---|---|---|
| D04 | `rules/physics/forcefields/contracts.py::OptimizationStoppingCriteria` | Retain the five current fields, order, units, defaults, frozen behavior and existing validation. Retire only the Python `OptimizationStoppingOptions` and workflow's five-field Python copier. Native translation directly consumes Criteria; the distinct real C++/pybind Options type and scalar standalone wrapper API remain. |
| D05 | `structure/forcefields.py::BondTopology` | Retain the existing frozen/order three-field record, endpoint normalization and `from_bond` factory; retire `BondTopologySignature`. Move existing `AtomTopologySignature` and `TopologyReference` into this module unchanged except reference bond annotation. No D06 consolidation. |
| FF-CLI | Existing `ForceFieldRouteOptions`, ultimately `operations/forcefields/stage.py` | Retire `_RunOptions`; pass the existing route record directly. Existing `_WorkItem(record, options, trajectory_path, output_format)` carries required output format after its existing first three fields. No new wrapper. |

D04 constructor remains `window: int = 5`,
`maximum_energy_change_kj_mol: float = 1e-4`,
`maximum_atom_displacement_angstrom: float = 1e-4`,
`maximum_rms_gradient_kj_mol_angstrom: float = 1.0`,
`maximum_gradient_kj_mol_angstrom: float = 5.0`.
`ComplexOptimizationOptions.stopping` becomes Optional[OptimizationStoppingCriteria]
with default None. None remains opt-out throughout native and organic routes;
no default stopping object is invented. Callers replacing the retired unvalidated
Python Options encounter Criteria's existing constructor validation. That earlier
validation boundary is intentional. The real bound constructor keeps its exact
five arguments; no native signature/default/layout or numeric policy changes.

D05 keeps `BondTopology(atom_indices, bond_order, bond_kind)` with required
fields, ascending endpoints, equality/order/hash and three-field asdict schema.
Manual reversed endpoints now normalize consistently with the retained record;
normal collection capture already normalized them. `TopologyReference` retains
`atoms, bonds, allow_added_hydrogens=True`; atom signatures retain
`index, atom_id, atomic_number, formal_charge`. Move carrier collection helpers
and `capture_topology(mol, *, allow_added_hydrogens=True)` from the retired
`forcefields/topology.py` to `adapters/structure/forcefields.py`. Consumers import
types from structure and capture from the adapter. Complete Bond truth, active
masks, original-atom identity, H-only additions, row-index mapping, native packing,
trajectory revision pooling and format5 JSON/SDF projection remain.

`BondTopology.from_bond(cls, bond)` retains direct read-only field capture without
an adapter/operation import. Atom.idx is the current topology row; there is no
diagnosed UID defect. Forcing a lower value to call an upper collection operation
would create the dependency this refactor removes. Pure rules may consume the
record; they neither invoke its carrier factory nor capture carriers.
New snapshot/capture leaves retain the existing quoted forward annotations without
TYPE_CHECKING imports of legacy core or the upward public facade. Runtime hint
resolution retains the earlier missing-name behavior unless an explicit namespace
is supplied. Retire only the deleted old topology module's three observed import
allowlist entries; add no import-debt exception or checker relaxation.

CLI preserves flags, route defaults, optional forcefield/seed/trajectory meaning,
format inference, result ordering, existing local outcomes/JSON and exit codes.
Output format moves only from duplicated options to the existing work item.
The current route owner remains `forcefields/stage.py` until its B3 operation
move; this is one definition, not an alternate implementation.

### Imports, exports, persistence and accounting

The existing forcefields aggregate remains its current public facade during B1,
pointing at the actual canonical types/capture. Remove the retired names from
exports and all current consumers; add no alias for them. The old topology
implementation is deleted. Other unmoved forcefield modules remain sole owners
until their declared Stage8B cutover. Final controllers/facade move to operations
and CLI to commands; remove the old package at that coherent dependent step.

Canonical classes use their real module paths; no __module__ spoof, compatibility
module or migration loader is added. Current objects/classes must cold-pickle
under source and installed CP311; current CLI spawn payloads must resolve their
actual owners. Retired-name Python-object pickles lose those symbols. Historical
retained-name paths may still resolve through imports needed by unmoved consumers;
that incidental behavior is neither forced to fail nor a compatibility promise.
Full old-package removal has its own explicit path/pickle migration record.
Existing explicit trajectory/report JSON schema/version/units remain unchanged.
No new class or schema is authorized: measured B1 target is three Python
dataclass definitions retired, with real native Options retained. Other reserved
consolidations are not claimed as achieved.

### B1 validation, release and rollback

Before evaluation, require: identical stated fields/defaults/units, known-zero
and None distinctions, direct criteria identity to native translation, one
canonical topology class, sorted row endpoints, stable acceptance/trajectory
values and JSON, correct CLI output-format composition and spawned worker state.
Allowed changes are the approved defining/import paths, retired symbols,
reference annotation and constructor normalization/validation differences above.
Reject lost topology/charge/coordinates, broken row joins, duplicated schemas,
changed numerical algorithms, stopping enabled from None, broken scalar wrapper,
JSON loss, new compatibility path, or source/installed origin leakage.

Run affected criteria/topology/acceptance/trajectory/CLI/API/native-Python and
wrapper contracts first, followed by complete affected CP311 source and installed
selections, current cold imports/pickles, and real CBond→FF→GFN-FF/GFN2 plus MCA
attachment at both boundaries. Bind exact commands/inputs/tests/models/native
binaries and current runtime bytes; preserve failures, numerical variation and
original denominators. Unchanged native implementation/build inputs may reuse
the six accepted Stage7E binaries with attribution; this is no clean build claim.
No new speculative guards or broad exception fallback is authorized.

Root reviews integration and releases runtime after a source freeze. Workers
cannot close gates. Rollback is the coherent B1 code/tests/documentation commit;
restore all connected owners/imports together, without touching unrelated work.
No stage acceptance follows B1 alone. B2 acceptance separation, B3 operation/
backend/command/native ownership, B4 connected finding-control/sample containment,
and full Stage8B parity/rollback/trajectory/standalone/FF benchmark evidence remain.
Other-version adaptation, clean-native/ABI/SDK/sdist, optional activation, broad
science/GPU/performance and Plan20/Plan12 final gates remain separate. Usable
bounded performance is judged under goal.md, not a forced historical ratio.


## B2 — Accepted pure acceptance and scoped capture boundary

B1 is committed and pushed as71c456a. Root aligns this second coherent slice
against the effective goal, main-plan Sections6/9.2/Stage8B, the33-file Stage8B
verification selection, B1 source/installed/actual evidence and the read-only
acceptance lookup. This is the existing Stage8B route, not a new scientific
algorithm or whole-stage acceptance. Native source and runtime engine stay fixed.

### Sole owners and existing contracts

Move the existing AcceptanceCheck, StructureAcceptanceThresholds,
ForceFieldAcceptanceEvidence, related AcceptanceLevel/ForceFieldStage/
ForceFieldDiagnosticValue aliases, and the three private normalized acceptance
issue/ring-evidence dataclasses into rules/physics/forcefields/contracts.py.
These are the existing scoped rule inputs/findings, not a second molecular
snapshot or new public report schema. Preserve each field/default/frozen behavior,
threshold, message, severity and enum; no new class, DTO or confidence field.
The existing mixed contracts module imports actual public definitions downward
until its B3 cutover. No retired implementation, private facade bridge, module
spoof or compatibility loader is created. Current classes cold-pickle from the
new true owner; incidental historical imports receive no compatibility promise.

New rules/physics/forcefields/acceptance.py owns carrier-free comparison and
numerical policy. It imports only lower immutable/value contracts, pure math,
NumPy and stdlib; no carrier, adapter, backend, workflow, runtime callback or I/O.
Existing forcefields/acceptance.py remains the one operation/controller until
B3 relocates that package. It assembles the existing ForceFieldValidationReport,
its fresh metrics dictionaries, warnings/control and lazy completed-evidence
normalization. Do not move its shallowly frozen mutable report into pure output.
The retained _CoordinationMetrics TypedDict stays with that operational assembly.

Extend adapters/structure/forcefields.py for actual carrier reads used by the
policy. Move the existing _iter_metal_donor_pairs there once; coordination.py
and acceptance capture import its one definition. Preserve its actual element-
classification semantics regardless of BondKind. Do not replace it with a new
coordination perception or the richer chelation/component capture. Existing
geometry adapter measurement and ring workspace remain sole algorithms.

### Value boundary and immutable outputs

Use existing records and scoped immutable tuples, without a new snapshot class:

| Pure work | Prepared inputs | Return and assembly |
|---|---|---|
| Coordinate checks | coordinate ndarray plus expected atom count | tuple of existing checks and finite flag; failure indices are row positions. Operation maps only returned nonfinite rows to current Atom.idx and owns atom/bond-count metrics. |
| Topology count and comparisons | scalar actual/original counts, then existing captured TopologyReference and reference | immutable check tuples. First call the count-only predicate; when original atoms are missing, return its existing failure without capturing fields/bonds. Otherwise reuse capture_topology and compare original IDs/Z/charge, H-only additions, exact bond quantities and added-H degree. |
| Force-field evidence | existing optional evidence mapping, level, limits and candidate/final stage | existing ordered immutable check tuple, with all current None/empty/zero/history/convergence distinctions. |
| Pair criteria | ordered tuples (endpoint-key pair, bonded flag, measured float, optional captured radius sum), level and limits | existing immutable issue/check tuples. Operation retains one geometry measurement and minimum-distance metric; off does not request pair-policy capture. Radius sums are read only for standard/strict nonbonded, nonoverlap rows that consume them. None means unconsumed radius input, never a fabricated radius. |
| Explicit-bond geometry | first coordinate snapshot and ordered tuples (bond-sequence index, endpoint rows, endpoint labels, optional radius sum, optional metal-ligand flag), level and limits | immutable check tuple plus maximum-length scalar; operation builds fresh metrics. Basic does not capture radius/metal-ligand inputs; no stale carrier or callable crosses the rule boundary. |
| Coordination metrics | first coordinate snapshot, metal rows including isolated metals, oriented metal/donor row pairs and row labels | immutable tuples (metal label, count, donor-label tuple, distance tuple, angle tuple). Operation renders the existing dictionary schema. Keep sorted rows, pair-angle order, zero-denominator skip and cosine clip. |
| Ring criteria | existing normalized immutable ring facts plus captured endpoint-key/bond-position values | existing immutable check tuple; operation copies actual scope/count/completeness and coordination metrics into its report. Missing bond keys/unknown native scopes still fail naturally. |

Local rule dictionaries/lists may implement comparisons; returned values contain
only immutable records/scalars/tuples and no source objects, mappings or aliased
numeric buffers. Do not recursively freeze or change the public operational
report schema. Coordinate inputs are not written and no arrays escape in results.
Capture/normalization is not a new scientific decision. The operation retains
completed geometry/native normalization at its current boundary until B3 moves
external translation to its final adapter owner; no rule imports it upward.

### Preserved execution and diagnostic meaning

Validate level and candidate/final stage before capture. Keep shape/finite checks,
optional topology and backend checks before finite early return; malformed/nonfinite
coordinates must not trigger pair/bond/ring/coordination work. Preserve shape-
mismatch avoidance of Atom.idx reads. Finite coordinates still allow later checks
when topology/backend findings already fail. Keep the existing ordered check list:
coordinates, topology, backend, overlaps, close pairs, explicit bonds, ring checks.
Keep overlap <= and closeness <, basic/standard override order, row labels versus
bond-sequence positions, per-bond failure order and success placeholders.

Public standard/strict performs exactly one existing ligand_skeleton/max16 ring
scan; public off/basic performs none. Explicit completed checkpoint reports are
consumed at every level, including off/basic, but only after finite checks.
Normalization/callbacks remain late; mandatory native evidence is reused without
rescans or per-epoch acceptance. Keep piercing/undetermined/scope order and actual
scan counts/completeness. Preserve current `.passed`/severity/public warning and
fallback policy; B4 owns its separately required finding/control migration.

This split does not establish one evaluator-wide concurrent mutation transaction.
It preserves current section sequencing and the existing operation-owned working-
copy route. Each pure invocation consumes prepared values only. Add no speculative
locks, extra molecular hashes, new guards, retries or fallback results.

### Validation, delivery and rollback

Before execution root reviews source and freezes the exact B2 runtime/test bytes.
Meaningful new tests call pure rules with prepared values while carrier/adapter
entry points are unavailable; demonstrate immutable/source-independent outputs,
non-mutation, scalar/zero/None/history behavior, early-return diagnostics, radii
laziness and actual scan/checkpoint reuse. Preserve and retarget existing tests to
real production lookup sites, including exact public serialization/check order.
Current moved classes/instances must cold-pickle at source and installed boundaries.

Run focused acceptance/checkpoint/native-checkpoint/scan-boundary/new-pure/contract
cases first, then the complete affected CP311 selection including all33Stage8B
files and connected contract/import/pipeline/xTB consumers. Compare exact ordered
source/installed nodes; run actual3pipeline+3MCA at both boundaries. Reuse six
accepted unchanged native binaries only with bound unchanged native/build inputs;
no C++ build or scientific algorithm change is claimed. Preserve failures, actual
numerical variation and original sample denominators. No cost or all-version gate
is created for this slice. Full Stage8B, native/standalone/FF/scientific benchmarks,
B3/B4 and later/global delivery/adaptation obligations remain open.

Rollback is one coherent B2 code/test/record commit with connected imports/captures.
Workers have exclusive files, no nested delegation and runtime HOLD until root
release; root retains design, scientific judgment, acceptance and Git delivery.


## B3a — Accepted native ownership cutover

Root/max accepts this bounded native split after B2 acceptance and pushed commit
13a23ce28a7bd1aaf16f825ef31711809baa61e7. Source/test authoring is released;
compilation, product execution and all gates remain on HOLD until root review and
an exact build/runtime release. B3 Python/backend/CLI, B4 connected control,
scientific/standalone evidence and later/global gates remain required.

### Objective and invariants

Cut over the current36native files once, by semantics, without a second engine or
an old/new runtime route. Keep one `_ob_native`, the existing `hotpot::forcefields`
names, real C++/pybind classes/functions/defaults/exceptions/layouts, numeric rules,
proposal ordering, budgets, coordinate/mask/revision behavior and standalone wrapper.
New internal functions/header paths are private implementation, not public schema.
No new class or generic molecule schema, renamed bound class, speculative guard,
retry or fallback is authorized. Approved old native include paths are hard moves.

The existing read-only native owner/shared-input/session-mechanics reports are the
source inventory; do not repeat it. Rebind their selected input hashes at entry.

### Canonical owners

All short names below were in `forcefields/_native`. Cross-owner quoted includes
must spell the actual relative path; never use ambiguous basename include search
or old-path forwarding headers. Preserve within-owner quoted local includes.

| Destination under cheminfo | Existing definitions / bodies |
| --- | --- |
| structure/_native/forcefield_input.hpp/.cpp | Existing ComplexSessionInput, count/validation methods and intrinsic ligand_molecule_view; preserve all fields, order, errors and metal-first intended endpoints. Include canonical `_native/include/hotpot/native/molecule_buffer_view.hpp`, alias canonical BondIndex/BondKind/Coordinate, with identical concrete types. No wrapper or rule dependency. |
| rules/physics/forcefields/_native | placement_candidates, placement_policy, target_selection, radii, pure topology_workspace; retain existing names/bodies except approved dependency extraction. |
| rules/physics/forcefields/_native/placement_evidence.hpp | Existing proposal/status/donor/candidate records only. |
| rules/physics/forcefields/_native/defaults.hpp | Ring limits and placement constants from maximum_actionable_ring_size through duplicate_tolerance; same detail namespace/values. |
| rules/physics/forcefields/_native/bond_keys.hpp | Existing internal::canonical_bond_key only. |
| operations/forcefields/_native/session_contracts.hpp/.cpp | Existing PerturbationOffsetBatch and validation, StructureSnapshot. Include the sole structure input. No pure consumer imports operation snapshots. |
| operations/forcefields/_native/ring_workspace_cache.hpp/.cpp | Existing RingWorkspaceCache, same_options and revision/options/preparation counters; depend downward on pure topology workspace and operation snapshot. |
| operations/forcefields/_native/placement_reports.hpp | Existing MetalPlacementResult/MetalPlacementReport, include pure placement evidence. |
| operations/forcefields/_native/defaults.hpp | Existing forcefield, stopping, attempts/steps/epochs, perturbation, VDW and energy defaults; include lower rule defaults when needed. |
| operations/forcefields/_native/warning_codes.hpp | Existing internal::append_unique only. |
| operations/forcefields/_native | Existing placement_engine, structure_session/internal, session_optimization, coordination_stage, optimization_stage, untangling_engine, workflow_stage, trajectory, stage_contracts; existing rule/checkpoint-to-operation report bridges stay with their consumer. |
| adapters/native/forcefields/bindings.hpp/.cpp | Existing complete pybind conversion/registration and exception translation, one registration called from retained wrapper initializer. |
| adapters/native/forcefields/openbabel_session.hpp/.cpp | Prepared external mechanics extracted from current session/session_optimization, described below. |
| reference/_native/openbabel_radii.hpp/.cpp | One raw immutable OpenBabel covalent-radius table lookup, described below. |

Do not duplicate generic OwnedMoleculeData or add radius/session fields to
ComplexSessionInput. Its existing owning vectors remain; const APIs and Python
read-only copied properties retain current behavior. No general C++ deep
immutability or borrowed-view lifetime guarantee is added. Workspaces borrow
const pointers within the current operation lifetime; outputs own their values.

### Immutable radius reference seam

The sole raw reference function is
`hotpot::reference::openbabel_covalent_radius(unsigned int atomic_number)` returning
`double`, defined in reference/_native/openbabel_radii.cpp using the exact current
`OpenBabel::OBElements::GetCovalentRad` call. The header has no OpenBabel include.
This is immutable vendor table access, not an optimizer/backend implementation;
retain the same linked OpenBabel table. No copied/replacement radius table.

The rule's existing `covalent_radius(int32_t)` calls this lookup with the same
unsigned cast, then preserves its finite-positive test, OPENBABEL_COVALENT tag and
0.77 Angstrom DEFAULT_COVALENT result exactly. The rule has no direct toolkit
include or adapter/controller dependency. Existing Python-bound radius names and
values remain. Negative/unknown-input behavior receives no new guards.

### Session-to-external-system boundary

Use internal namespace `hotpot::forcefields::openbabel_session_adapter`. It accepts
prepared ComplexSessionInput, old/requested masks, coordinates and wrapper-owned
native optimizer options; never a source Hotpot carrier, StructureSession or
StructureSnapshot. Do not invent a handle/DTO or move operation decisions into it.

Move the current ligand conversion/validation/allocation to
`make_session_obmol(const ComplexSessionInput&) -> unique_ptr<OpenBabel::OBMol>`.
Preserve wrapper compatibility validation before input.validate and make_obmol.
The operation keeps its one unique_ptr, lifetime and existing mutex-protected
release; complete OBMol type availability for unique_ptr destruction is type
plumbing, not external scientific implementation.

Extract the complete existing ligand and coordination edit mechanics into
prepared adapter functions taking OBMol&, input, old mask and requested mask.
Operation validates masks, takes existing mutex, handles equal-mask early return,
then calls mechanics and commits its mask/revision only at the existing boundary.
Preserve BeginModify/EndModify, exception text/order, zero-to-one-based endpoints,
ligand int(order), coordination max(1,int(order)), and the separate post-EndModify
ligand aromatic pass over every requested active bond. No coordination aromatic
pass is invented. These mutate only the operation-owned external molecule.

Adapter exposes the existing runtime mutex, coordinate extract/set, active bond
count and single/general optimizer calls by forwarding to the one wrapper engine.
Keep wrapper-native argument/result/exception types, no conversion DTO/copy path.
Operation owns coordinate equality/change decisions, before/after comparison,
mask/component/count/snapshot assembly, revisions, stopping/budget/trajectory and
rollback/selection. Retarget placement-engine mutex calls to this same adapter
mutex. Wrapper result/enums/default type dependencies remain their existing one
owner. No second optimization engine or broad exception fallback.

There is no transactional rollback in the current two bond-edit blocks; extraction
does not promise one or silently add recovery. Higher controller rollback and the
working-copy/source atomic-commit route remain their existing operation owners.

### Connected cutover and validation

Update all36old files, exact cross-owner includes, the retained wrapper initializer
binding include, setup.py's single extension source list and MANIFEST.in. Expected
20FF-related TUs after contracts/cache/reference/session-adapter splits, each once;
all16old TU paths retire. Preserve all unrelated extension sources, ABI0/C++17,
OpenBabel/dl/rpath and the standalone wrapper bodies.

Update the nine existing native_cpp driver/fixture pairs coherently, including
full source lists/include paths and lower-dependency topology/stage-contract fences.
The topology fixture keeps its cache/revision-reuse case at the operation owner.
No stale production path or forwarding header remains. Add narrowly meaningful
fences for lower-rule dependency isolation, canonical shared concrete aliases,
reference-radius equivalence and session-mask/aromatic/order mechanics; retain
all current actual behavior/rollback/trajectory tests. Do not merely assert source
spelling or change expected chemistry to make the cutover pass.

Before runtime root inspects the final split and binds source/tests/build/native
inputs. Require an actual rebuilt single _ob_native; other accepted unchanged
extensions may be reused with explicit evidence. Start focused compile/contracts,
then all affected CP311 source and installed Python/native workflow/standalone
coverage, all nine FF source/header drivers plus connected wrapper/shared fences,
actual3pipeline+3MCA at both boundaries. Bind wheel/source origin and meaningful
scientific geometry/state/finite-energy/lineage/persistence observations, preserving
numerical variation and first failures. Header-only/compile-only fences are not
execution, native ABI parity or full clean-build claims.

B3a acceptance covers this physical native cutover only. Python operation/backend/
CLI/packing migrations, B4 controls and full owning/global gates remain open.
Rollback is one coherent source/test/build/record commit, retaining unrelated work.


## B3b — Accepted force-field command ownership cutover

At `5b5615f`, B3a is accepted within its recorded CP311 boundary. Main-plan
Section9.2 already assigns the FF CLI to `commands/ff.py`. Move the sole existing
implementation and its unchanged Markdown to `hotpot/commands/ff.py` and `ff.md`;
create only the real command package initializer. Update the lazy `ff` command
registration, package-resource declaration and connected test imports. Retire the
old CLI/document paths without aliases, forwarding modules or a compatibility
loader. Existing force-field operation/backend owners remain until their coherent
subsequent B3 cutover. No other command is migrated in this slice.

Preserve command name/flags/defaults, argument errors, route selection, local
outcomes/exit codes, serialization, input/output cardinality and order, format
inference, report/trajectory behavior, serial/parallel workers and the current
scientific algorithms. The private command payloads acquire their actual new
module owner; current spawned workers must resolve them. No historical private
CLI pickle compatibility is promised. No new class, schema, guard or fallback is
authorized. Imports required by the relocated command use the existing sole
owners; this does not authorize moving the coupled Python backend piecemeal.

Root owns this bounded design, maintained records, integration, acceptance and
Git delivery. A local Sol/medium writer owns the command/resource/registration,
package metadata and connected test migration. An independent local Sol/medium
worker owns only temporary CP311 validation preparation; neither accepts the
change, commits, pushes or runs candidate business before root release.

Validate the complete affected CLI, lazy-loading and packaging selections on
source and a newly built/installed CP311 wheel outside the checkout. Preserve the
existing real serial SMILES and parallel `CC`/`CO` order/SDF tests; exercise actual
command dispatch and packaged `--doc`, and record emitted chemistry and source/
installed origins. Reuse unchanged accepted B3a natives after current native/
build rebinding; do not rebuild unchanged C++ or rerun unrelated MCA/full-stage
work merely for a command move. Package-data-only build edits must be explicit.
Source/installed membership, actual outputs, failures and limitations remain
attributable. This is no complete Stage8B/global, controlled-performance,
clean-native/ABI/SDK/sdist or all-version acceptance. Rollback is the coherent
command/test/package/record commit.


## B3c — Accepted Python native transport and capture cutover

At `0a0290c`, B3a native ownership and B3b command delivery are accepted in their
recorded CP311 scopes. Existing B3 Python preparation and the targeted closure
lookup were rebound to current bytes. Section9.2 already assigns carrier capture
to the structure adapter and external/native translation to adapters. This slice
closes that dependency boundary before moving the coupled Python controllers;
it does not move a mixed controller wholesale into a lower adapter.

| Existing responsibility | Sole target owner |
| --- | --- |
| `forcefields/native.py`: current request/snapshot classes and raw native calls | `adapters/native/forcefields/native.py` |
| `forcefields/native_reports.py`: existing transport results and raw converters | `adapters/native/forcefields/reports.py` |
| Existing `ComplexSessionInput` definition | leaf `adapters/native/forcefields/contracts.py` |
| Carrier `_coordination_bonds` and `pack_complex_session_input` | existing `adapters/structure/forcefields.py` |
| Array/mask-only `_bond_topologies`, `_BOND_KIND_NAMES` and `unpack_native_topology_revisions` | native `reports.py` beside the raw translation |
| Existing TrajectoryStart/Stage/Event, three frame-evidence records and FrameEvidence union | existing `structure/forcefields.py` |
| Two existing ring-policy constants | `rules/physics/forcefields/settings.py` |

Create only the real native-forcefields initializer. Retire the four old native,
native_reports, native_packing and settings modules, with no shims or old/new
switch. The leaf transport contracts import no operation, legacy forcefield
package or adapter implementation. Structure values import no native contract,
adapter or operation; native translation imports their unchanged enum/evidence
values. Keep AtomIdentity/from_atom, BondTopologyRevision, ForceFieldFrame,
_TRAJECTORY_STAGE_ORDER, mutable trajectory/archive/controller behavior and D06
consolidation at their current owners for the later coherent operation cutover.
No new class/Protocol/DTO/schema, scientific algorithm or confidence is added.

Preserve fields/defaults/signatures, native option mapping, units, row/atom IDs,
sorted directed metal-donor rows, ligand-then-coordination masks, topology order,
source versus session revisions, native class-name/enum-name conversions,
selected/terminal frame indices, format5 tags/pooling, seed/draw order, copied
readonly result arrays versus writable packed inputs, and forwarded placement
reports. Existing wrapper packing remains the sole field schema/bond-code map;
consumers import its `_BOND_KIND_CODES` directly instead of recreating an
incidental alias. Capture lock scope and unavailable partial-charge task encoding
remain unchanged. No source carrier publication is added to an adapter.

Migrate all actual consumers, including private packing references, lazy trajectory
ingestion, TYPE_CHECKING imports, benchmark native lookup and source-path fences.
The ordinary FF facade keeps current public identity by importing actual owners.
Do not expand the structure-adapter `__all__` with internal packing helpers or
add accidental FF exports. Current objects pickle through their actual new owners;
old retired-native-module pickles may now fail ModuleNotFoundError instead of the
previous AttributeError. No spoofed __module__, compatibility alias or migration
loader is authorized. Existing eager parent-package imports remain; no cold
package-import-isolation claim follows from closing direct dependency edges.

The retained TYPE_CHECKING-only core.Molecule import carries exactly its former
native.py debt slot to the actual adapter owner until Stage9. Also reconcile the
already accepted B3b cli.py-to-commands/ff.py annotation with its own former slot.
Remove the two obsolete native_packing core-import entries. These fixed transfers
preserve both existing ceilings and multiplicity and grant no new import slot or
unrelated debt credit. Each retained annotation remains due for Stage9 retirement.

Root owns this global-boundary design and all maintained decisions/releases.
Local Sol/medium owns canonical production extraction first, with exclusive files;
connected consumer/test migration follows after the canonical writer stops.
Source syntax/static ownership checks may run during authoring. Candidate runtime
stays HOLD until root reviews the completed connected cutover and freezes it.
Require meaningful current owner/pickle/array/schema and native workflow tests,
complete affected CP311 source and installed selections, actual representative
FF/connected pipeline behavior and attributable package/native origins. Reuse
unchanged accepted B3a C++/binaries after rebinding; no unchanged C++ recompilation
or unrelated workflow rerun is a prerequisite. Numerical equality and historic
pass counts are not chemical or coverage oracles. Preserve original failures.

Acceptance is this Python transport/capture boundary only. Coupled Python
operation/backend separation, B4 finding/control/sample containment, full Stage8B
scientific/standalone/benchmark gates and later/global delivery/adaptation remain
open. Rollback is the coherent source/test/package/record commit.


## B3d — Accepted coupled Python operation and backend cutover

Root aligns at pushed `8001127`, with B3c accepted at its CP311 boundary. The
remaining legacy Python forcefield package mixes real workflow controls with
external invocation and carrier-derived values. Main Section9.2's destinations
remain; this decision resolves those mixed bodies before their coherent move.

| Current responsibility | Sole target owner |
| --- | --- |
| Remaining forcefield controllers, acceptance composition, mutable trajectory/archive, worker lifecycle, reports/errors, facade | `operations/forcefields/`, preserving the existing module names |
| Existing `AtomIdentity`, `BondTopologyRevision`, `ForceFieldFrame`, `CoordinationEnvironment` | existing `structure/forcefields.py`, retaining fields/factories exactly |
| Existing `collect_coordination_environments` carrier capture | existing `adapters/structure/forcefields.py`; facade imports this actual definition |
| Raw wrapper `build`, `single_optimize`, `optimize`, native lookup and existing seed call | thin `adapters/openbabel/forcefields.py`, invoking only retained standalone obWrappers |
| Existing coordinate copy and clipping/addition arithmetic | `rules/physics/forcefields/coordinates.py`; caller supplies displacement |
| RNG creation, `normal` draws/schedules and coordinate publication | `operations/forcefields/coordinates.py` and their existing workflow callers |

Move all remaining Python files out of `cheminfo/forcefields`; remove that old
package without compatibility shims or old/new runtime switches. The real
`operations/forcefields/_native` sources already belong here and remain unchanged.
Use explicit facade imports/exports with the same supported names, except the
plan-required deletion of the unimplemented `prepare_coordination_geometry` hook
and its dormant `CoordinationGeometryCandidate`/`CoordinationGeometryResult`.
Their unsupported NotImplemented entry is not replaced by a guessed algorithm.
Retire the connected `coordination_geometry=None` option from `build_complex3d`,
`complexes_build`, `build_and_optimize`, `auto_optimize`, `Molecule.build3d` and
private forwarding/trajectory-parameter records together. Non-None previously
reached only NotImplementedError; the removed keyword now fails through the
ordinary signature/unknown-option contract. Existing default behavior stays.
Do not silently accept/ignore the obsolete option or leave a dead forwarding call.
Current public objects pickle at actual new owners; old-package object pickles
may fail ModuleNotFoundError. Keep explicit trajectory JSON/NPZ/format5 unchanged;
no spoofed module, migration loader, extra DTO/Protocol/schema or class is added.
D06 remains an assessed optional consolidation: trajectory AtomIdentity retains
its distinct stored symbol and format contract; no reduction is booked for it.

Backend selected-graph preparation/currentness/atomic publication, setup-error
translation, forcefield selection and the one worker lifecycle lock remain in
operations/backend.py. `optimizer` and `native_adapters` retain RNG scheduling,
trajectory ingestion, report assembly and publication as operations. These
modules must not be moved wholesale into an adapter. Thin external adapters may
import the actual standalone wrapper functions/lookup; they add no second engine,
carrier selection, seed lifecycle lock, report construction or numerical policy.
Standalone public wrapper contracts and bond-code constants remain their existing
single owners. Preserve production lookup sites for monkeypatch tests at their
actual new modules, not aliases at retired paths.

The coordinate rule consumes explicit coordinates/displacement/sigma and performs
only the existing NumPy clipping/addition; the operation draws from the existing
RNG once in the existing order and calls it. This clarifies Section9.2's former
`_perturbed_coordinates` example: accepting/advancing a mutable Generator is
operational scheduling, not a pure rule. Pure copy/arithmetic has one definition.
Retain all budgets/defaults, seed/draw order, lock scope, selected versus direct
mutation behavior, double source rechecks, rollback, atom handles, units, row/mask
mapping, frame/pool ordering, warnings and existing local failure behavior.
This move does not preempt B4's finding/control/sample-containment review.

Remove obsolete TYPE_CHECKING-only core imports rather than transferring their
debt slots. Quoted/postponed carrier annotations remain unchanged; ordinary runtime
hint resolution already lacks these names, and explicit namespaces remain usable.
Migrate actual package/dynamic imports, core facade, conformer default builder,
command/registry/benchmark/tests and maintained current paths together. Update
architecture allowlist only by retiring actually removed entries; no inherited
exception, new core import slot or wildcard credit is granted. New leaf/structure
owners import no operation or legacy facade.

Root owns this global-boundary design, maintained records and acceptance. One
local Sol/medium writer owns the canonical package/value/capture/adapter/rule
cutover; connected consumer/test migration follows after it stops. Source-only
AST/ownership/equivalence checks may run while authoring. Candidate runtime stays
HOLD until root reviews the whole connected cutover and freezes source. Require
focused affected workflow/trajectory/pickle/RNG/rollback/seed/worker cases, complete
affected CP311 source and installed selections, actual representative FF/pipeline
and affected MCA/conformer integration with attributable origins. Reuse unchanged
accepted B3a native source/binaries after rebind; no unchanged C++ rebuild is a
prerequisite. Compare chemical facts/units/currentness and allowed numerical
variation rather than forcing identical float values or historical pass counts.
Retain failures. B4, complete Stage8B scientific/standalone/FF benchmarks and all
later/global gates remain open. Rollback is the coherent source/test/record commit.


## B4 — Accepted failure and sample-control boundary

B3d is accepted and pushed as9ac9680. Root aligns effective goal, focused main
Stage8B/Sections2,11–12, integrated Plan12 requirements and HARD-012 against the
actual FF/xTB adapters and unchanged generic engine. The read-only discovery
finds broad stage catches conflating known sample rejection with worker, mapping,
source and commit integrity errors. Only identified bounded causes may continue.
No scientific algorithm, numerical threshold, trained feature or generic engine
redesign is authorized. Root owns these global-boundary decisions.

First authoring partition establishes attributable local failure identity at the
actual producer, before any adapter continuation. Existing FF error classes gain
optional `sample_failure_code` metadata; existing worker result/error transport
forwards that exact optional code. Missing/unknown codes do not acquire a local
meaning from exception class names or messages. Ordinary molecular/report values
are unchanged. One private ValueError subclass in the route owner identifies the
two existing request-domain checks, preserving ValueError compatibility; no new
public exception export, DTO, success envelope or generic status is introduced.

| Actual bounded trigger | Code | Origin and limit |
| --- | --- | --- |
| Existing optimize-only request without usable coordinates | `FF-REQUEST-COORDINATES` | Route precondition only |
| Existing complex route with unsupported selected field | `FF-REQUEST-METHOD` | Route precondition only |
| Existing translated native setup failure | `FF-SETUP-UNAVAILABLE` | Existing setup error only |
| OBBuilder returns unsuccessful | `FF-BUILD-UNAVAILABLE` | Exact existing false-result branch |
| Existing isolated ordinary/complex worker timeout | `FF-BUILD-TIMEOUT` / `FF-COMPLEX-BUILD-TIMEOUT` | Existing bounded timeouts only |
| No usable finite/topology frame | `FF-NO-USABLE-FRAME` | Existing GeometryQualityError |
| Exhausted existing ligand candidates with no usable fallback | `FF-LIGAND-CANDIDATES-EXHAUSTED` | Exact terminal branch; prior unknown errors stay unknown |
| Official xTB native unsuccessful process | `XTB-NATIVE-EXECUTION` | Existing XTBExecutionError with raw report |
| Required xTB result absent/invalid before commit | `XTB-RESULT-UNAVAILABLE` | Existing XTBResultError; raw artifacts remain |
| Existing method element-domain rejection | `XTB-ELEMENT-UNSUPPORTED` | Existing XTBApplicabilityError |

The first ligand sampler error wrapper forwards a code only if its actual cause
already carries one. Worker EOF/shutdown/exit/protocol/invalid-success coordinates
and arbitrary child errors retain None. Do not infer safety from a worker error
name. Stale source, carried-state/provenance mismatch, unknown ValueError and
programming exceptions remain fatal. A validated-coordinate commit integrity
error retains XTBInputError and original cause; remove its old reclassification
as ordinary XTBResultError. Parsing-invalid isolated output remains a named
unavailable result. No new retry, fallback or guessed guard is added.

The existing ligand candidate and refinement catches also require a concrete
correction before adapter continuation: both catch every ForceFieldError and can
turn an unknown/protocol failure into named exhaustion or silently retain a
candidate. Their actual recoverable producer is the translated
ForceFieldSetupError from _single_ob_optimization, including the same call inside
ring repair. Narrow both catches to that existing setup exception. Preserve the
current bounded attempt/fallback and diagnostic behavior for setup rejection;
propagate other errors unchanged, without another attempt or candidate fallback.
The first sampler wrapper still retains its original cause and optional identity.
This corrects observed loss of failure identity; it adds no speculative guard,
algorithm, retry budget or scientific policy. Focused tests must prove both known
setup behavior and unchanged unknown failure identity at candidate and refinement.

Local Sol/medium owns this exact code/transport/test partition. Root then reviews
it before authorizing the connected adapter-control partition. Until that
partition's concrete reporting/denominator/break contract is recorded, it may not
be implemented. Authoring AST/static checks only; candidate imports/tests/models/
build/native execution remain HOLD until root review and the bounded freeze.
Focused tests must distinguish actual known codes from unknown/protocol failures,
prove worker transport and commit rollback/original error, and retain messages,
reports and diagnostics. B4 acceptance additionally requires actual mixed-sample
continuation and finite-observation tests at source/installed CP311; code metadata
alone does not close B4, Stage8B or later/global gates.


### B4 connected authoring — complete sample reports and explicit control

Root settles the connected partition at9ac9680 plus the reviewed first identity
changes, before its authoring. This is the Stage8B/Plan12 operation-adapter
boundary, not a redesign of the generic pipeline. Scientific algorithms,
thresholds, FAST's documented candidate/fallback method, numerical budgets,
trained features and ordinary molecular return types remain unchanged.

**Known unavailable results.** FF adapters continue only for the exact codes in
the preceding producer table. xTB additionally names its actual typed
XTBStateResolutionError as `XTB-STATE-UNAVAILABLE`, retaining the scientific
bundle/projection findings. This is distinct from carried-state, expected-identity
and commit mismatches, which remain untagged/fatal. At molecule_to_xtb_geometry,
the existing explicit-H rejection is `XTB-EXPLICIT-H-REQUIRED` and the existing
geometry validator's XTBInputError is `XTB-INPUT-GEOMETRY`. Attach that geometry
code only at this input call site; the same validator's failure during validated
output commit remains an untagged integrity error. Do not translate arbitrary
ValueError, backend/infrastructure error, state-extraction exception or error text
into a known local cause. Preserve raw errors/reports/causes and actual attempts.

**Finding classification.** All existing AcceptanceCheck names, measured values,
thresholds, severity, order, subjects and profile computation remain intact.
Pipeline findings use `FF:<check.name>` or `XTB:<check.name>`; only failed checks
are listed, including warning checks. The domain prefix scopes an existing rule,
not a scientific grade. Exact name sets are documented here:

| Names | Defined value / default control | Commit/fallback consequence |
| --- | --- | --- |
| coordinate_shape, finite_coordinates | Required Cartesian frame unavailable; named local result failure | No downstream output; retain available reports/native evidence |
| topology_atom_count, topology_atom_identity, topology_added_atoms, topology_original_bond, topology_added_bond, topology_added_hydrogen_degree | Required topology/row contract unavailable; named local result failure | No downstream output; integrity exceptions from source/commit/worker remain fatal |
| forcefield_report, forcefield_setup, finite_final_energy, finite_rms_gradient, finite_max_gradient | Required FF evidence/value unavailable; named local result failure | No downstream output; retain actual finite observations and report; no invented numeric value |
| atom_overlap, atom_too_close, bond_distance, short_bond, bond_length_ratio, bond_ring_piercing, bond_ring_scope_coverage | Finite geometry remains an observation; report and continue | Retain selected result, existing attempts and findings; no new repair |
| backend_explosion, forcefield_convergence, rms_gradient, max_gradient, energy_change, max_displacement, stability_observations | With required finite fields available, report method/convergence/stability finding and continue | Retain finite result and native report; do not redefine numerical convergence |

`topology` is the current success-only check. xTB's `XTB:convergence` additionally
reports the actual false converged process fact without claiming an FF method.
Distance/coordinate units remain angstrom, FF units remain the selected FF
report's kJ/mol and kJ/(mol*angstrom), xTB energy remains hartree. These are
method/profile-domain findings, not universal chemical impossibility claims.

xTB post-checks in both pipeline and standalone CLI use a private geometry-only
operation in the existing acceptance owner, reusing the same coordinate, pair,
bond and ring rules/scans. A private core switch excludes FF evidence checks on
that route. Public evaluate_structure_acceptance(strict, no FF report) keeps its
existing missing-report behavior. No fabricated FF report and no second geometry
algorithm or extra scan is introduced.

**Per-input report contract.** Keep MolecularRecord/MolecularPayload schemas and
Molecule state unchanged. FF/xTB report adds `sample_outcomes_version=1`, counts
`input_count, attempted_count, output_count, unavailable_count, sample_break_count,
not_attempted_count`, and ordered `sample_outcomes`. Every input has exactly one
row with `input_index`, `output_index` (integer or null), and `outcome` from
`returned, unavailable, sample-break, not-attempted`. Add `report_index` when a
method report exists, `findings` when discovered, `failure` for an unavailable
exception, and `break_scope` when an explicit rule matched. Findings contain
`code` plus unchanged check content; failure retains code/type/message and
available structured diagnostics, worker traceback or native report. Method
reports remain in existing forcefield/xtb locations; output_index joins their
retained molecular output, report_index joins method reports. Actual error
trajectories/native files remain persisted as stage artifacts.

For contiguous FF/xTB stages, complete positional edges and persisted adjacent
outputs preserve original ordinals/denominators, even for equal molecules. The
cohort anchor is nonempty initial input or persisted CBond-generated output/rank.
Do not claim automatic generic lineage through arbitrary unrecorded reordering or
generating stages. No molecule/content hash is used as a substitute for position.

**Explicit control.** The pipeline FF/xTB stage parsers accept repeatable
`--sample-break-on CODE` and `--batch-break-on CODE`, both default empty. Match
exact emitted code strings; no message inference or catch-all promotion. Known
codes and the complete named finding sets above are parser choices, with xTB
restricted to its actual geometry/process/state codes. With no override, known
unavailability removes only that downstream sample; non-hard finite observations
remain output. A sample-break omits that finite sample from downstream output
but persists its molecular observation at `observations/<input_index:04d>.sdf`,
alongside the report. An unavailable value remains unavailable; a sample-break
cannot manufacture an output. A batch-break stops attempts after the matching
input, preserves every earlier/current defined observation and all remaining
not-attempted rows. If both scopes match, batch control wins; the current finite
observation stays in stage output because the whole pipeline is stopping.

Append `break_pipeline: bool=False` to existing StageResult after stderr. This
is expected execution control on the existing pipeline result, not a molecular
success wrapper. Persist it in stage manifest. The existing runner stops only
after normal result persistence when break_pipeline is true, or when status is
QUALITY_FAILED and (final stage or empty output). Return existing FAILED with
failed_stage_index and retained stage payload/report; no exception and no
final.sdf. Existing intermediate nonempty QUALITY_FAILED from other stages may
still continue. No new status/engine/registry/retry is introduced. FF/xTB return
SUCCEEDED for a nonempty surviving cohort with default findings/local failures;
counts disclose failures. All-unavailable/sample-broken output is QUALITY_FAILED,
terminating at its actual origin so a later generator cannot replace the cohort.

**Standalone command boundary.** After the pipeline partition is reviewed, audit
and migrate the two existing CLI local catches to the same precise known causes;
unknown/integrity errors must propagate. Keep actual emitted observations and
reports. Finite non-hard quality findings default to reporting with usable-output
exit0 rather than an implicit final failure. Local unavailable samples still
produce existing error outcomes/exit1 while later independent samples run.
Explicit per-code break flags initially belong to controlled pipeline stages;
standalone batch scheduling/flags are not silently redesigned by this partition.
Record that scope in command help/docs. The separate CLI authoring assignment
must name its files before writing them.

Root retains design/records/freeze/runtime release/acceptance. Parallel local
Sol/medium writers may own disjoint adapter/report and pipeline/geometry/producer
files; no nested delegation. Authoring remains static until both stop and root
reviews source/tests. Required focused CP311 tests cover known versus unknown,
worker transport/commit rollback, middle rejection with trailing success,
duplicate-content composed ordinals and CBond anchor, explicit sample/batch
breaks with not-attempted rows and finite artifact retention, all-unavailable
persistence/no downstream generator, default non-hard observations and xTB strict
geometry without fabricated FF failure. Obtain actual mixed FF/xTB backend
workflows, source and installed origins, and rebind unchanged native evidence.
Keep full Stage8B scientific/standalone/FF cost and later/global gates open until
their actual evidence/owning review; no historical float/hash equality oracle.


### B4 standalone continuation contract

The subsequent exclusive CLI slice migrates commands/ff.py and plugins/xtb/cli.py
with their focused tests/docs after the adapter writer stops. Reuse the stage
owner's exact known-code and unavailable-check sets and canonical FF error-evidence
helper; remove the command's former duplicate helper. No extra inference or
per-command scientific thresholds. In FF, _run_forcefield no longer turns an
arbitrary ValueError into _CLIUsageError. Only the exact existing private route
request error becomes its InputError outcome; retain its actual code. Unknown
FF errors, ValueError and xTB stream/commit/provenance errors propagate rather
than becoming local outcomes. Known typed xTB input/state/process/result failures
continue at the sample boundary with their precise code and existing native
report/logs. Add the code to existing error evidence / xTB error JSON (schema2
additive field); do not introduce a new ordinary result or compatibility route.

A successful call with failed required-result checks is an unavailable sample:
preserve its report, produce no molecular payload and report the exact scoped
finding code. A returned finite result with non-hard findings retains the existing
quality-failed diagnostic outcome and molecular stream; that diagnostic alone
no longer makes command exit nonzero. Commands return0 when every sample produced
a usable result, including non-hard findings, and1 when any known sample result
is unavailable. Existing initial syntax/input-path errors remain2. Preserve actual
reports/warnings and denominator/order; no finite finding is rewritten as passed.
Parallel command scheduling remains the existing implementation: an unknown
error propagates and aborts aggregation, with no promise to cancel already-running
independent processes. Controlled pipeline flags supply explicit per-code breaks;
no standalone scheduling redesign or extra flags are added here.

Required tests distinguish request-domain errors from arbitrary ValueError,
known from unknown same-text FF/xTB errors, commit/stream integrity from known
input geometry, hard unavailable result from finite quality findings, and actual
serialized output/report/exit behavior. Explain profile findings, partial failure
and controlled per-code breaks in the existing command/pipeline documentation.


### B4 root integration correction before runtime

Root review of the first adapter authoring found an unwanted dependency: the
independent xTB stage imported forcefield.stage merely for generic report/control
helpers, eagerly loading the FF operation/backend graph even with post-check off.
Place only the shared positional counts, exact-code break selection, finding-row
conversion and observation persistence functions in a small private
pipeline/_sample_control.py utility. It imports no FF/xTB/scientific algorithms
and accepts each caller's serializer explicitly. Each adapter retains its actual
domain code/check-name sets; xTB required checks are its coordinate checks only.
This is helper ownership for the two existing adapters, not a generic engine,
registry, status schema or pipeline redesign.

Use argparse's existing append action and convert parser lists to immutable
prepared tuples; no custom Action class is needed. Retain the existing explicit
FFError/ValueError and XTBError/ValueError translation boundaries. Other unknown
exceptions propagate unchanged; no new catch-all is needed to continue known
samples. Unknown caught errors still become fatal StageExecutionError with the
original cause. No unknown exception becomes local merely by message or type.

Observation persistence returns a relative path recorded in the sample row.
Sample-break preserves its finite observation there; a hard missing FF metric
also preserves the defined geometry when its existing coordinate/topology checks
pass. Invalid frame/mapping observations rely on their retained raw evidence;
do not serialize them as a usable molecule. Keep original first-authoring
receipts and record these corrections separately before source freeze.


### B4 standalone input-reader correction before runtime

Root CLI review found one remaining broad translation in xTB's non-SDF
input reader: arbitrary RuntimeError/ValueError from MolReader became invocation
usage errors. Narrow this to the actual input I/O exception OSError, matching
the existing FF reader. Preserve explicit missing-path/format/empty-input usage
checks and exit2. Unknown parser/programming exceptions retain their original
identity and cause; SDF stream integrity errors also propagate. This is the same
accepted unknown-error boundary, with no new parser, guard, fallback or scheduling
change. Focused tests must prove unknown reader propagation and known input I/O
usage handling before freeze; retain original authoring evidence separately.


### B4 CP311 root acceptance

The validation/b4/checkpoint.json accepts the settled failure/control/sample and
standalone contracts at source/installed CP311:269focused,1408affected passed and
2explicit scientific skips per boundary,1410matching orderednodes,3actual official
pipelines and real UFF/GFN2 mixed cohorts per boundary with original3/surviving2.
Root checked atoms/bonds/charges, finite values/units, singlepoint nonmutation,
legal projection, persistence and origin evidence; numerical variation is retained.
The actual middle failures are known request/state rejection, not simulated native
failure. Original attempt01/three new legacy-core test imports remain archived;
only public-export imports corrected them, without weakening architecture checks.
Fresh478Python-body wheel reuses6unchanged B3a natives;41unchanged omissions are
P20-D008. Unchanged B3d MCA source/evidence is reused without redundant inference.
No native rebuild, all-version, full corpus/cost or global acceptance follows.
Whole Stage8B still needs O0 scientific observations/comparison, explicit
standalone-wrapper evidence/disposition and comparable FF cost or an attributable
owner-authorized disposition before Stage8C entry. Later/global gates stay open.


### Whole Stage8B scientific/standalone/cost evidence after B4

B4 is accepted and pushed as eceb034. Continue the existing whole-stage owning
gate under Sections11/12 and the effective goal; this is not Stage8C entry.
Read-only inventory identifies a single retired-package import in
run_scientific_report.py, used only to attribute acceptance.__file__. Replace
that import with operations.forcefields.acceptance, with no numerical, budget,
seed, algorithm, roster or monitor change. Record the original tool source and
AST-only import difference before the new actual scientific execution.

Release the existing full standard O0 Am corpus with all187 source records,
rootseeds42/43/44,32spawn workers, affinity0-31 and BLAS/OpenMP threads1. Retain
all561requested trial slots with current rule-derived eligibility, actual
provider seeds (including explicit ignored/unavailable seed evidence), attempts,
frames or absence, energy/gradient units, monitors/findings and errors. Compare
against B20 as observations, not fixed182 success or float/hash equality. Unknown
execution/integrity errors remain unresolved; heuristic findings alone are not
scientific impossibility. Explain differences without relaxing observed rules.

Execute the two explicit deep fixtures (full phosphorus corpus and full standard
Eu workflow) with their scientific/high-compute flags at source and the already
accepted B4 outside installed boundary. Rebind the unchanged wrapper sources,
six binaries/build inputs and accepted B3a12native fences; no redundant C++build
or broader ABI/SDK qualification is claimed. Preserve failures and raw outputs.

Run the existing organic_forcefield CPU cost recipe with the current manifest,
source mode, one warmup/five measured attempts, affinity0-63/threads1 in a quiet
window. All71workloads and measurement settings match B20; only policy/bookkeeping
manifest fields differ. Preserve the existing comparator's manifest-identity
limitation, raw durations/work counters/failed attempts and uncertainty. Judge
comparable semantic work and usable cost under current Section12.2, not a forged
manifest hash or superseded acceptance quotas. This one workload does not close
later/global cost or complex workload requirements; index remaining gaps openly.

Root retains review, any bounded route correction, gate closure and Git delivery.
Use fresh lifecycle-managed artifact outputs. No business-source edit is authorized
by this evidence slice beyond the settled one-line report-tool import; all other
validated B4 source/test bytes remain fixed (including its documented EOF-only
normalization). No external helper, new model, dependency or environment change.


The two deep fixtures currently retain mostly assertions/JUnit properties. Before
running them, add only evidence persistence under their pytest tmp_path: actual
input/output structural values, raw coordinate arrays, returned method/quality
reports and elapsed values before scientific assertions. Keep actual parameter
budgets, assertion thresholds, APIs, sample membership and algorithms unchanged.
This is the existing fixture's result recording, not another calculation or a
parallel validator. Preserve any failed observation; do not make a finite finding
pass by changing a check. Only those two test files and the one report-tool import
are authorized maintained edits for this owning-evidence slice.


### Observed whole-gate recorder/preparation failures and bounded corrections

Released O0 attempt01 preserved all187inputs/561trial slots but performed no FF
trial:182CBond-returned molecules failed graph serialization with ValueError
"graph serialization requires present implicit-H facts"; fivecases were
ineligible. The raw reports and unchanged1484-file release remain retained.
Inspection shows build_validation_cohort already explicitly calls the existing
operations.chemistry._require_implicit_hydrogens before its raw graph capture,
but the dynamic O0 report's _case bypassed that preparation after the carrier
cutover. Correct _case to use that same finite-H consumer boundary before capture,
without changing the graph schema, H rules, CBond eligibility policy or algorithms.
Add focused actual-Molecule regressions for supplied ligandH/new metalH0, stale
inferredH refresh and genuinely unresolvedH remaining unknown rather than0.
Original deterministic/raw-graph rejection remains; no catch/default/schema waiver.
Only run_scientific_report.py and its existing scientific report-contract test
receive this correction. A new source freeze/released O0 attempt is required.

Deep fixture attempt01 source phosphorus actually passed all15cases with
28rule applications and retained outputs. Its subsequent provenance postcheck
failed because three imported tests package initializers were omitted from the
copy/provenance map; all observed initializer hashes match the released source
and no product leak occurred. Preserve the original KeyError and passed scientific
artifacts. Correct only the temporary recipe's exact input map and reconcile that
completed run; do not repeat it solely for the broken postcheck. Root will release
only the remaining sourceEu/installedP/installedEu commands after final preparation
review and current source binding. No evidence/gate is approved by this correction.


### Observed ring-open aromatic failure — operation-buffer correction

The saved case68 deterministic reproduction confirms the six O0attempt02
failures at common aromatic packing of a projected open Molecule. Intact source
and ligand graphs are cyclic and valid; removing edge2-3 makes supplied aromatic
rows0,1,2,3,21,22 noncyclic. Strict AROMATIC.NONCYCLIC_SOURCE correctly rejects
that ordinary chemical graph. Original source facts/state and all1475release
bindings remain unchanged; full traceback and observations are retained under
/tmp/hotpot_plan20/long/stage8b-gate-20261104/aromatic-runtime-repro/attempt01.

Root selects a bounded correction at the existing operation/adapter boundary,
superseding only Stage7B's ring-repair use of an independent Molecule for native
optimization. A temporary open force-field network is an operation input, not a
new molecular chemical interpretation. Under the original owner's lock, validate
the selection and pack the intact source through existing strict aromatic/native
preparation. Apply the immutable mask only to the four aligned native Bond arrays
(indices, orders, kinds, aromatic flags) and their count, preserving complete Atom
order, typing flags, charges, coordinates and cell. No aromatic flag is cleared or
recomputed on the cut network. The original source remains authoritative. Ordinary
Molecule projection/components and full-graph standalone packing stay strict.

Reuse existing NativeMoleculeBuffers and MoleculeData; add no DTO, schema, native
code, public selector/signature or persistent graph. Extract one private
_single_optimize_data native-call/report translator from standalone single_optimize
and use it for both full and masked data; public single_optimize keeps its existing
coordinate-write contract. The Open Babel adapter supplies these private data seams
to the operation, without another force-field engine. The selected backend captures
its existing bonding_inputs/geometry/cell source after strict preparation, computes
outside the source lock, then checks the selection and source inside the existing
fact transaction before publishing only returned coordinates. Unknown/native/setup
failures, stale source and transactional rollback keep their existing behavior.
Open-network energy remains attributed only to its matching open trajectory frame.
No repair budget, perturbation, selection rule, geometry monitor, denominator,
scientific algorithm or ordinary source fact is changed.

Release authoring only in obWrappers/{packing,native,operation}.py,
adapters/openbabel/forcefields.py and operations/forcefields/backend.py, with
connected existing packing/selected-backend tests and a focused real-native aromatic
regression. Root owns versioned method documentation and these records. Validate
full-source rejection, exact ordered masks/dtypes/isolation, same-work native calls,
coordinate-only commit, stale geometry/topology/representation/cell rejection and
original-error propagation before successor source/installed/native-backed evidence.
No candidate calculation runs until root review/freeze. Existing six native binaries
remain reusable after rebind; required wheel/source/affected scientific evidence must
use the successor Python bytes. Historical failures remain retained.


### Whole Stage8B CP311 acceptance — 2026-11-04

Root accepts the observed ring-open Bond-buffer correction and complete owning
Stage8B CP311 scope. [Checkpoint](../validation/whole-stage/checkpoint.json),
[report](../validation/whole-stage/report.md) and its evidence index bind current
source/installed1449passed+2skipped of1451, focused66source, retained installed
focused65/1contextfailure plus successful failed-node continuation, deep2 and
official3 per boundary, and two real mixed cohorts per boundary. No passed
unchanged source/focused65 work was repeated for the artifact Git-index defect.
Current1482bindings/model/native/wheel/origins match. SourceO0all187inputs/561
slots yields546completed/15ineligible/0executionerrors; all historical failures
and finite concerning observations remain. Sixrepaired open-network frames match
B20 and do not transfer their energy onto restored topology. Root reviewed worst
arrays, high/nonconverged energy and monitor changes; later matching-topology
trajectory variation remains P20-D022, without RNG attribution. The phosphorus
fixture remains finite-only evidence (P20-D023). Current small organic FF work is
usable; the original manifest-hash comparator inconclusive status and absence
of separate memory probes remain. Twelve unchanged native fences and actual
unchanged MCA evidence are rebound. This permits Stage8C alignment; no later/global
scientific/performance/native-ABI/package/version/Plan20/Plan12 acceptance.
