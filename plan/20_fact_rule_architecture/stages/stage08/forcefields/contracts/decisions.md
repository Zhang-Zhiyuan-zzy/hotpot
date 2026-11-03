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
