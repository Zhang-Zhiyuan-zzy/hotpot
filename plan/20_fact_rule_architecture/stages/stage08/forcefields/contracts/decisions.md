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
