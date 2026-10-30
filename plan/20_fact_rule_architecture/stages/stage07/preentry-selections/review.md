ANALYSIS_RESULT

task_id=p20-s7b-hidden-constraint-inventory-704283-36  
revision=1  
verification_code=704283  
task_hash=d4cfb178387db81924e5e03ea3ce739dc4145d328acd2072b4f0a5583d1adcd0  
source_commit=979455ed8bada61f4d44038935e7f02c1b204c86  
diff_or_tree_hash=82d24bc5b463bd011683d2cc322d4ceb858e42d5680db0980c67a15d659401f6  
capture_identity_kind=SHA256 canonical capture-manifest.json raw bytes; NOT Git tree  
status=PREENTRY_INVENTORY_ONLY

## Hidden-bond ownership and selection matrix

| Area | Current factual state | Completion / remaining scope |
|---|---|---|
| Persisted owners | `Molecule` owns `_bonds`, `_hided_metal_bonds`, and `_hided_covalent_bonds`. `_hided_bonds` constructs a new concatenated list and is not an owner. `core.py:145-173` SHA `17f0c5db…`; `254-256` SHA `3941101e…`; source SHA `bc44ba23…`. | Persistent hidden collections remain. Stage7B’s operation-local-selection migration is not implemented in this capture. |
| Active topology | `_build_topology_index` validates endpoints and active/hidden disjointness across all collections, but builds incidence only from active `_bonds`. `core.py:4697-4728` SHA `7995f5c8…`. | Active adjacency and complete represented topology are distinct current concepts. |
| Hide writers | `_hide_bond` moves an object from `_bonds` into the metal or covalent hidden list; `hide_bonds` and `hide_metal_ligand_bonds` then call `_update_topology`. `core.py:1457-1481`, SHAs `f0fb890d…`, `514decef…`, `ea9ccb85…`. | These are persisted carrier mutations, not task-local selections. |
| Restore writers | `restore_bonds` validates the entire selection, preserves the same Bond objects and UIDs, removes them from hidden collections, appends them to `_bonds`, and updates topology once. `core.py:874-902` SHA `7fa7a85d…`. The two recover methods delegate to it (`904-930`, SHAs `363ccf8a…`, `1077d055…`). | Public-family removal remains Decision 6 and is not approved by this inventory. |
| Permanent clear writers | `clear_metal_ligand_bonds`, `clear_hided_covalent_bonds`, and `clear_hided_bonds` remove represented bonds and target their stereo rows. `core.py:836-872`, SHAs `77b2eb07…`, `ff810b32…`, `e9b61eee…`. | No explicit `_update_topology` or `_mark_topology_dirty` occurs in the two direct hidden-clear bodies. Any transitive invalidation through the unsupplied stereo helper is unproven. Focused dependency-invalidation coverage for permanent hidden clearing was not supplied. |
| Transaction enrollment | All public hide/recover/clear methods are dynamically wrapped with `topology=True`. `core.py:4921-4927` SHA `cf7c3a5f…`. `touch_topology` snapshots the three Bond collections, constraint owner, complete Bond state, table, declarations, stereo, frames and dependencies; rollback restores original owners and contents. `_commit.py:262-353`, SHAs `fad082d6…`, `7998b226…`; source SHA `48eeb8a6…`. | Static rollback path is connected; no current runtime execution was performed. |
| Factual transport | `_capture_factual_record` serializes active/hidden partition numbers, complete endpoints, values, kinds, directions, sources, aromatic flags and constraints. `_install_factual_record` reconstructs all three partitions and one topology index. `core.py:2202-2356`, SHAs `4c824d41…`, `976dc05c…`. `MoleculeRecord` validates aligned partitions `0/1/2`; `_builder.py:215-338`, SHAs `b8f7d24…`, `b8e73d66…`. | Complete represented-Bond persistence is established; it is the storage that Stage7B must later cease using for transient selection. |
| Copy/component/merge | `copy()` remaps dependency state and installs fresh members; `add_component()` concatenates complete partitions and allocates destination UIDs. `core.py:375-387` SHA `5fb811d7…`; `745-801` SHA `8f5b7be0…`. Atom replacement retargets active and hidden endpoints while preserving Bond identities (`618-648`, SHA `8e633a59…`); atom removal filters all three collections (`1720-1743`, SHA `07e86d94…`). | Current ownership/remap semantics are covered statically and by existing tests; migration must preserve them without reopening geometry/ring algorithms. |
| Pickle | Molecule pickle captures complete active/hidden order but excludes those live containers from generic state; raw restore reinstalls factual partitions and original UID/high-water values. Bond member restore searches active, hidden-metal, then hidden-covalent order. `core.py:4764-4840`, SHAs `fa427f25…`, `e33ea332…`, `60fab820…`, `659ac010…`. | Current-version persistence is mapped. Historical cross-version behavior is outside this inventory. |
| Source/view invalidation | Topology identity includes complete active and hidden bonds (`_identity.py:75-76`, SHA `76e1070c…`; `173-244`, SHA `4ac0337c…`). View freshness additionally binds the actual topology-index object and all three membership counts (`views/_source.py:14-30`, SHAs `b3e60278…`, `45fe65c3…`). | Hidden membership affects source identity even though hidden bonds do not enter active incidence. |
| Factual import | RDKit/OpenBabel imports reject targets containing any active or hidden bonds, then install active partition-zero records. `obconvert.py:86-125` SHA `ed9a7adb…`; `rdconvert.py:709-773` SHA `eecb136c…`. | Import boundaries currently depend on the hidden collections’ existence. |

## Current selection consumers

- Ligand building persists a metal-excluded topology by cloning and calling `hide_metal_ligand_bonds`; its components, candidate optimization and coordinate publication operate on that modified clone. `forcefields/ligand.py:113-545`, SHA `03277bcf…`.

- Working-copy hydrogenation hides and later restores metal-ligand bonds. Publication captures and compares complete active/hidden inputs, prepares complete bindings, and commits under rollback. `working_copy.py:38-52`, SHAs `c005ca8b…`, `f3fe9159…`; `96-138` SHA `c8b00ff8…`; `149-280`, SHAs `867d8a8b…`, `d7ae460a…`, `a3e1163a…`, `21e141e1…`.

- Ring repair temporarily persists one open edge with `hide_bonds` and restores exactly that edge in `finally`. `repair.py:361-415`, SHA `a79b48c…`. Ring geometry/science is accepted Stage5 scope and was not re-audited.

- Native structure application converts a native active-coordination mask back into persisted hide/restore mutations, followed by a coordinate commit inside one factual transaction. `native_adapters.py:332-365`, SHA `c775c1e6…`.

- Common `pack_molecule` copies only `mol.bonds`, so its bond arrays are active-only. `_native/buffers.py:438-583`, SHA `f32b89a1…`. OpenBabel packing requests that field set without structural constraints. `obWrappers/packing.py:1-50`, source SHA `e90afbd7…`.

- Complex native packing separates active non-metal ligand bonds from intended coordination bonds. Intended coordination is deduplicated from active metal-ligand bonds plus `_hided_metal_bonds`; hidden covalent bonds are not packed. `native_packing.py:59-69`, SHA `c4a8b827…`; `89-177`, SHA `5a44b775…`.

- Geometry views derive adjacency and ordinary angle/torsion rows from active bonds, append persisted constrained UID sequences when absent, then resolve captured Bond references from active followed by both hidden collections. `operations/geometry.py:15-40`, SHA `63fa7fbd…`.

Thus current task-local native masks already exist, but Python ligand, repair, working-copy, validation, chemistry and native-application paths still express exclusions by mutating persisted hidden collections. Those are the factual Stage7B consumer gaps. No replacement DTO or public API is approved here.

## Constraint ownership matrix

| Constraint kind | Current owner and writers | Reads, identity, preservation and gaps |
|---|---|---|
| Atom coordinates | Three Atom-table fields: `x_constraint`, `y_constraint`, `z_constraint`; aliases are declared in `_fields.py:1-87`, source SHA `8dff7260…`. `Atom.__init__` accepts axis fields and `constraint`, records explicit presence/source, and rejects conflicting overlapping aliases (`core.py:2569-2605`, SHA `f94c988c…`). `Atom.constraint` reads `all(...)`; its setter commits all three axes through `_store_atom_fields` (`2790-2806`, SHAs `9be752c5…`, `767e7893…`; `4599-4608`, SHA `cab2917f…`). | Stored in the sole Atom fact table, not `_ConstraintSet`. Factual records, copy, import, merge and pickle carry these columns. They participate only in `ConstraintIdentity`, not representation identity (`_identity.py:34-36`, SHA `01b1b099…`; `173-244`, SHA `4ac0337c…`). Native/OpenBabel field selection contains no constraint fields. |
| Bond | Numeric slot 2 of `Bond._attrs`; schema is `bond_order`, `constraint`, `id`. `core.py:3628-3637`, SHAs `91dca851…`, `ed388e46…`, `663d562b…`. Generic Bond assignment calls `_store_bond_field`, which commits a complete raw row through `_commit_bond_facts` (`4611-4617`, SHA `b9d829ea…`; `_commit.py:380-405`, SHA `e6ca229c…`). | Complete factual Bond transport and transaction snapshots preserve it. Constraint identity reads every active and hidden Bond’s UID/value. `clear_constraints` iterates only active `self.bonds`, so hidden-Bond constraints are not explicitly cleared by the supplied body. No focused hidden-Bond-clear constraint test is supplied. |
| Angle | Sole persisted owner is frozen `Molecule._constraints.angles`, containing ordered Atom-UID triples. Fresh Angle properties validate the view source and call `_commit_constraints`. `constraints.py:1-34`, SHA `c68efd36…`; `core.py:4880-4897`, SHA `7dc0a736…`. | Completed in accepted Stage4: no view-local value, UID remap through copy/pickle/component/merge, stale-view checking and rollback. Current source still reflects that contract. |
| Torsion | Same owner model with ordered four-UID entries; property route parallels Angle. | Same completed Stage4 ownership. Repeated-member torsions and constrained rows are represented by ordered tuples rather than a second cache. |
| Bulk clear/default assignment | `clear_constraints` clears every Atom axis, active Bond constraints, then replaces `_ConstraintSet`; `constraint_bonds_angles` sets active non-metal Bonds and fresh non-metal Angles. `core.py:826-834`, SHA `b50aff24…`; `1258-1277`, SHA `9fa99565…`. Both are dynamically transaction-wrapped (`4933-4934`, SHA `3cab12d5…`). | Neither supplied body clears hidden-Bond constraints; the convenience assignment covers neither torsions nor a distinct hidden-Bond selection. This is a static coverage gap, not a proposed behavior change. |
| Unified private commit | `_commit_constraints` validates current Atom UIDs, can commit Atom-axis updates, accepts Bond entries from any complete active/hidden partition, and publishes updated Angle/Torsion sets. `_commit.py:506-534`, SHA `a4d7395e…`. | Current public scalar routes remain split between the fact engine, Bond-fact commit and `_ConstraintSet` commit. |
| Rollback | Transactions journal Atom fact cells, complete Bond rows, and the `_ConstraintSet` reference. Nested setters join the parent transaction; rollback restores each owner. `_commit.py:133-160`, SHAs `8129fe5a…`, `822cd0ff…`; `217-219` SHA `52d45eac…`; `286-353` SHA `7998b226…`. | Static rollback wiring is complete. Runtime failure claims rely only on accepted earlier evidence, not this turn. |
| Topology/remap | `_update_topology` subsets Angle/Torsion entries to retained UIDs or constructs them from import rows. `_capture_factual_record` maps UID entries to rows; `_install_factual_record` maps rows back to current UIDs. Raw pickle later restores original UID gaps. | Atom/Bond constraints travel in their respective factual rows. Angle/Torsion migration is completed; consolidation of the remaining coordinate/Bond owners is Stage7B scope under the existing plan. |
| Native/force-field packing | `NativeMoleculeField` and `NativeMoleculeBuffers` contain atom/bond chemistry, coordinates, cell and stereo transport, but no constraint field. `_native/buffers.py:28-43`, SHA `7fb6f9ba…`; `151-170`, SHA `a231d746…`. | No supplied packing path proves native enforcement of Atom/Bond/Angle/Torsion constraints. This is missing evidence, not a runtime failure claim. |

Constraint source checks are content-based. `ConstraintIdentity` includes topology, all three Atom-axis booleans, complete active/hidden Bond constraints, and both `_ConstraintSet` tuples. `_check_source` rejects changed captured domains without a carrier revision counter. `_identity.py:173-277`, SHAs `4ac0337c…`, `e4cb469f…`, `caf7fee8…`.

## Completed evidence and test coverage

Accepted Stage4 records establish Angle/Torsion ownership, fresh immutable identity-bound views, constraint copy/pickle/remap, rollback and Wiberg pair-observation preservation. Relevant immutable records are:

- `stage04/contracts/review.md`, SHA `326db981…`
- `stage04/contracts/schema.json`, SHA `cc465b8a…`
- Stage4 gate review, SHA `772793d5…`
- Historical implementation evidence, SHA `808452d2…`
- Historical validation evidence, SHA `239e9383…`

Those records are accepted completed evidence, not fresh execution against source commit `979455ed…`. Stage5 ring ownership/science remains accepted and was not reopened.

Current connected tests include:

- Hidden lifecycle/restoration: `test_hidden_bond_restoration.py` SHA `7dd82fb6…`, `test_molecule_structural_lifecycle.py` SHA `10935917…`, `test_uid_lifecycle.py` SHA `47b30d52…`, and `test_topology_determinism.py` SHA `6026b5b7…`.
- Native selection and rollback: `forcefields/test_forcefield_native_adapters.py` SHA `991a5e9a…`, `test_plan20_outer_cutover.py` SHA `89d05701…`, and working-copy rollback in `test_complex_hydrogens.py` SHA `90def710…`.
- Angle/Torsion ownership: `structure/test_constraints.py` SHA `f886015c…`, `structure/test_view_identity.py` SHA `6dfa93ae…`, `structure/test_source_identity.py` SHA `d0b003c4…`, and builder/pickle cases.
- Hidden represented-bond consumers: stereo lifecycle/projection test identities and limited boundary excerpts were supplied, but their completed scientific algorithms were not re-audited.

The maintained Stage7B map contains two stale labels:

- `tests/test_cheminfo/structure/test_constraints.py` exists, SHA `f886015c…`, despite `planned_not_created`.
- `tests/test_cheminfo/structure/test_view_identity.py` exists, SHA `6dfa93ae…`, despite `planned_not_created`.
- `tests/test_cheminfo/structure/test_topology_selection.py` is genuinely absent.

The maintained four-file technical command omits several presently connected constraint, source-identity, native-adapter, rollback and UID tests. No collection or execution was performed, so neither that command nor the indexed test definitions establish current pass status.

Missing focused evidence includes:

- operation-local topology-selection contract tests;
- migration parity after removal of persisted hidden collections;
- permanent hidden-clear dependency invalidation;
- hidden-Bond constraint behavior under `clear_constraints`;
- constraint propagation or deliberate exclusion at common/native packing boundaries;
- current source-versus-installed-wheel parity, cold imports, supported-version execution and performance comparison.

## Required rebind and authority boundary

Stage6 and Stage7A may alter production bytes, imports, transitive consumers and installed-wheel contents. Before any future Stage7B design or implementation ownership release, root must rebind:

1. exact production source and import graph;
2. hidden-bond/constraint consumer searches and selected bodies;
3. maintained test identities and existence labels;
4. installed-wheel selections and cold-import evidence; and
5. the actual exact-turn review/audit record.

This inventory is not a Stage6 acceptance dependency. It does not authorize Stage7B entry, design, implementation, public hide/recover removal, a new topology/constraint DTO, scientific reinterpretation, or gate closure. Root retains all substantive decisions and project-record ownership.