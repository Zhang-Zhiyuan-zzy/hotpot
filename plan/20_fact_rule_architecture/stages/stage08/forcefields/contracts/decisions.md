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
