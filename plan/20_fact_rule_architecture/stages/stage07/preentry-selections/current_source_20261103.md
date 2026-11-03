# Current-source rebind — 2026-11-03

Factual discovery only; this note changes no plan, design, public contract or gate.
Inspected source HEAD: `48780a669c9b1ca5b0e6c2cadf63365579ad3ccf` (`48780a6`).
Historical inventory remains `review.md`/`evidence.json`, source `979455ed`.
Root's later Stage7A closure/private-constraint design-entry decision is separate;
this discovery does not authorize implementation or hidden-topology migration.
Read governance skill/references, relevant mainplan scope/Decision6/17, and the
historical inventory's root qualifications. All source references are relative
to `hotpot/cheminfo/`; line numbers identify the inspected source bytes.

## Hidden topology and current production consumers

- Owners remain `core.py:153`: `_bonds`, `_hided_metal_bonds`,
  `_hided_covalent_bonds`; `_hided_bonds` at `:243` is a concatenated projection.
- `_build_topology_index` (`core.py:4645`) validates complete membership while
  indexing active incidence. Hide/restore preserve Bond objects/UIDs.
- Public family survives: clear-metal `core.py:821`, clear-hidden-covalent `:841`,
  clear-hidden-all `:850`, restore `:859`, recover-metal/covalent `:889/:911`,
  hide-selected `:1422`, hide-metal `:1427`; `_hide_bond` at `:1411` is private.
- Internal family calls: clear-metal→hide-metal `:832`; recover→restore
  `:908/:914`; hide-selected→private hide `:1424`. No other production clear
  caller was found. All family methods remain transaction-wrapped (`:4866`).
- External-to-Core production calls are exactly: `operations/chemistry.py:289`
  (default charge clone), `forcefields/ligand.py:131` (ligand clone),
  `forcefields/working_copy.py:116/:122` (hydrogenation hide/recover),
  `forcefields/repair.py:378/:406` (one-edge hide/finally restore), and
  `forcefields/native_adapters.py:362/:364` (native selection publication).
- Connected state consumers: factual capture/install `core.py:2149/:2203`,
  copy/merge/replace/remove/pickle; `structure/_builder.py:225/:259`,
  `structure/_commit.py:259`, `structure/_identity.py:75`,
  `views/_source.py:14`, `operations/geometry.py:15/:33`,
  `forcefields/working_copy.py:214`, `forcefields/native_packing.py:59`, and
  active-only common packing `_native/buffers.py:438/:449`.
- Historical import-check paths changed: current checks are
  `operations/conversion.py:91/:151`, not old obconvert/rdconvert bodies.
  Current stereo collection capture/checks are
  `adapters/structure/stereochemistry.py:13/:52`.
- Hidden permanent clear reaches `_remove_stereo_bond_rows` (`core.py:317`),
  `operations/chemistry.py:569`, then `structure/_commit.py:465`; that last
  commit seeds only changed implicit-H transport inputs. A general topology
  invalidation call is absent on this traced path; runtime harm was not tested.

## Constraint ownership, correction and enforcement boundary

- Atom axes remain table facts (`structure/_fields.py:30`); aliases/constructor
  metadata remain at `core.py:2526`. Aggregate constraint is all three axes and
  writes all axes (`:2739`); scalar writes reach `:4547` and native fact storage
  `structure/_native/facts.cpp:824/:997`.
- Correcting the historical inventory: Bond row order is bond_order, constraint,
  id (`core.py:3580`), so constraint is zero-based slot **1**, not slot2.
  Public setter→`core.py:4559`→`structure/_commit.py:356` uses the proper index.
- Dormant hazard: `_commit_constraints(..., bond_entries=...)` writes `_attrs[2]`
  (`structure/_commit.py:507`), the id slot. Search found no production/test
  caller supplying bond_entries. This is not a public Bond-setter failure;
  the branch must not be reused unchanged during consolidation.
- Angle/Torsion sole persisted owners remain ordered UID tuples in frozen
  `structure/constraints.py:20`; views project with stale-source checking
  (`core.py:4825/:4880`). Existing Stage4 ownership/science is not reopened.
- UID subset/remap stays in `core.py:460/:2168/:2283/:4746`; rollback retains
  table/Bond values and constraint owner (`structure/_commit.py:214/:323`).
- `clear_constraints` explicitly covers axes, active Bonds and owned sequences
  (`core.py:811`). Active-only Bond scope is characterized, not declared defective.
  `constraint_bonds_angles` (`:1243`) does not promise torsion assignment.
- Constraint source identity (`structure/_identity.py:209`) is consumed by FF
  working-copy stale-publication checks (`forcefields/working_copy.py:33/:38`).
- No carrier/view Atom/Bond/Angle/Torsion constraint enforcement reader was found
  in searched FF/native optimization paths. Common/complex packing has no such
  fields. `obWrappers/_native/optimization_operation.cpp:132` creates empty
  OBFFConstraints then Setup; `forcefields/_native/session_optimization.cpp:42/:72`
  delegates there. Native fact storage writes are not optimization enforcement.
- Separate explicit RDKit CoordinationEmbeddingConstraints (`rdWrappers/contracts.py:42`)
  reaches native bounds matrices (`rdWrappers/_native/src/sampler.cpp:65`);
  it does not read carrier/view constraints. Do not silently add restraint science.

## Scope and evidence limits

`AtomPairs` is already a fixed caller-owned mapping (`core.py:3477`). The map's
constraint/view-test existence labels are corrected; topology-selection test
remains absent (`stage_verification_map.json:3271/:3281`), with incomplete current
technical selection. Preserve accepted Angle/Torsion/ring science and public
signatures; public hidden-family removal/changed semantics still needs Decision6.
Commands were read-only sed/cat/rg/rg --files/sha256sum and Git identity/status.
Search scope included production Python and native sources under cheminfo plus
connected tests; broad constraint names include unrelated query/state constraints.
No runtime imports, tests, builds, package/environment/remote/Git mutations ran.
Source/installed-wheel, supported-version, performance and runtime rebinds remain
owning-stage obligations. Forbidden Task168 failed-successor1 data was not used.
