# Stage7B complete Bond topology and operation selections

Status: **DESIGN ACCEPTED (HT1–HT5); IMPLEMENTATION RELEASE SEPARATE**. Root acceptance is recorded in [design_acceptance.json](../design_acceptance.json); [entry_review.json](../entry_review.json) retains the original entry snapshot. This document specifies the remaining connected Stage7B cutover after the [accepted constraint checkpoint](../../constraints/validation/checkpoint_20261103.json). C0 records HEAD `63043dc766d27087bcc3aba3c75367d403b2f3d0` with production/test bytes matching business source `6d1ea01f4e8d354318a00633a43b85b92cb68e7b`. Earlier runtime evidence retains its recorded parent-plus-overlays identity. The owner-approved, activated [goal.md](../../../../goal.md) governs this design. Design acceptance does not release source authoring or runtime, or close Stage7B/global gates.

## 1. Entry, authority and bounded objective

The complete current main plan, effective goal, four normative companions, affected verification-map/budget entries, accepted Stage7A completion, constraint entry/release/checkpoint/report and relevant current implementation-log decisions were reread before this design. Entry alignment was appended to the owning log before substantive authoring. Reuse the maintained [current-source inventory](../../preentry-selections/current_source_20261103.md); its pre-constraint observations are superseded only by the accepted constraint checkpoint. Scoped native and pickle scouts identify current boundaries; their paths/hashes are in the entry record. Static source reads rebind the affected owners to current bytes. This design task executed no product import, calculation, test, build, install or Git command.

Root already adopted Decision6: one complete ordered authoritative Bond collection, operation-local exclusions, and connected retirement of eight public hide/recover/clear methods with no aliases or replacement public selector. Root also directs that native completion retain all factual Bonds; native masks describe an operation/session graph or trajectory observation, not chemical validity or discovery. Root accepts the exact dependent HT1–HT5 contracts below; this document records that decision, while implementation release and candidate evidence acceptance remain separate.

For Hotpot-initiated external working projections, Hotpot retains original fact authority. Apply the registered adapter's documented graph/bond translations with minimum loss, quietly for expected projections; a computed return does not authorize reverse-importing its temporary graph as new Hotpot membership. Independently supplied native/toolkit inputs instead use the documented adapter-authoritative import mapping. No universal OB metal/single-to-coordination or hapticity rule, extra object label or support claim is inferred from an example. Unexpected or unmapped material loss and integrity failures require precise adapter-scoped diagnostics; detached carriers cannot recover absent upstream facts or relations.

The completed constraint slice remains accepted: existing `_ConstraintSet` is the sole Atom-axis/Bond/Angle/Torsion owner, public Atom19/Bond3 projections preserve raw values and metadata, and the native touched-key undo implementation remains. Its active-only `clear_constraints` behavior changes to complete-Bond coverage only at this connected cutover. Chemical semantics take priority, then usable cost, then version adaptation; CP311 leads. Known H/isotope/stereo normalization and canonical storage remain Stage7E. No FF restraint enforcement, new aromatic/H/charge rule, model feature/weight change, Plan17 backend switch or later-stage extraction is authorized here.

## 2. Authoritative membership, order and public effects

`Molecule._bonds` becomes the sole complete ordered Bond list. `Molecule.bonds` remains a fresh list containing its existing Bond handles; `bond(i,j)`, `c_bonds`, Bond indices, Atom incidence/neighbours, public components, rings, geometry views, copy, merge and default export see that complete graph. `_complete_bonds(mol)` remains the existing private helper and returns `tuple(mol._bonds)`. Remove `_hided_metal_bonds`, `_hided_covalent_bonds`, `_hided_bonds` and `_hide_bond`, including rollback/pickle-owned-field assumptions. No empty compatibility lists, carrier mask, context-variable graph override, revision switch or saved native Graph remains.

Order is insertion/import record order. Temporary selections never move or reorder a Bond. Permanent removal retains the relative order and UID of surviving handles; addition appends and allocates a fresh UID. Copy/subset remaps independent handles in source order; merge uses destination order followed by source order. Working-copy publication reuses every original Atom/Bond handle and its UID; newly added H atoms/Bonds append. Transport may sort coordination rows for its existing `(metal_row, donor_row)` mapping without sorting the carrier list. UID high-water marks never decrease after failed transactions.

The complete graph is a simple graph under the existing active-graph rule: endpoints are current owner members, self edges and repeated unordered endpoint pairs are rejected, and a Bond handle occurs once. The old installer allowed duplicate pairs across active/hidden partitions; the new installer rejects the complete invalid record atomically rather than selecting a winner, collapsing facts or implementing multigraph science. This is an explicit root disposition (HT4), including the currently characterized active-first/hidden-fallback fixture. Such fixtures become rejection tests; valid unique-pair constraint/view cases retain their assertions.

Retire these exact Molecule entries together:

| Entry | Current arguments after `self` | Cutover |
|---|---|---|
| `hide_bonds` | `*bond: Bond, clear_conformers: bool = False` | Remove |
| `hide_metal_ligand_bonds` | `clear_conformers: bool = False` | Remove |
| `restore_bonds` | `*bonds: Bond, clear_conformers: bool = False` | Remove |
| `recover_hided_metal_ligand_bonds` | `clear_conformers: bool = False` | Remove |
| `recover_hided_covalent_bonds` | `clear_conformers: bool = False` | Remove |
| `clear_metal_ligand_bonds` | none | Remove |
| `clear_hided_covalent_bonds` | none | Remove |
| `clear_hided_bonds` | none | Remove |

Permanent edits use existing `remove_bond(bond)` / `remove_bonds(bonds: Iterable[Bond])`, which retain complete invalidation, stereo filtering, constraint-cell detachment and transactional rollback. No public mask, `TopologySelection` export, new selector argument, exception type or no-op alias is introduced. `clear_constraints(self) -> None` keeps its signature and clears axes, every complete Bond cell and owned Angle/Torsion entries. Remaining public methods keep their signatures; complete topology, the accepted HT1 H/Bond ordering change and the native generic-adapter disposition below are intentional observable effects. Old hide/recover tests are not a parity requirement for a retired API.

## 3. Exact private selection and projection contract

Consume reservation **N04**, at most one new private dataclass, in `structure/_selection.py`:

```python
@dataclass(frozen=True)
class _TopologySelection:
    source: _SourceIdentity
    atom_uids: tuple[int, ...]
    bond_mask: bytes

_capture_topology_selection(mol, *, excluded_bonds=(), ligand_skeleton=False)
_check_topology_selection(mol, selection) -> None
_selected_bonds(mol, selection) -> tuple
```

The result of `_capture_topology_selection` is `_TopologySelection`. `_selected_bonds(mol, None)` is the complete tuple; `None` is only a private call default, not stored state. There is one 0/1 byte per complete Bond row, including excluded rows; Atom UIDs retain complete current Atom order. The source reuses N01, which already carries owner identity: full/explicit-handle selections bind `topology`; predicate-derived ligand selections also bind `representation` so element/classification changes cannot silently change the intended scope. No separate owner field, UID lookup table, selected-Bond mirror or mutable ndarray is stored. Bytes/tuples are immutable by content, including nested buffers.

Factory/check boundaries hold the owner lock. Excluded handles must be current complete members by object identity and owner, with no duplicates; endpoint-equal or equal-UID foreign handles do not qualify. Cardinality and binary mask validation use existing `ValueError`; foreign/stale source uses existing `RuntimeError`. Check the captured topology/Atom order before interpreting a mask or publishing output. A same-length Bond permutation, endpoint replacement, remove/re-add or owner change is stale. Coordinates alone do not invalidate a topology-only exclusion; each numerical operation separately binds the geometry/representation/cell inputs it actually consumes. Capture/check a whole-source identity once per coarse boundary, never per Atom/Bond scalar access. Selections do not mark owner dependencies dirty or invalidate live views merely by being constructed or discarded.

Use the existing capture/installer, not another DTO, for independent task graphs:

```python
Molecule._capture_factual_record(
    self, *, rows=None, include_frames=True, copy_metadata=True, selection=None)
_components(mol, molecule_type, *, with_bindings=False, selection=None)
_project_molecule(mol, molecule_type, *, selection, rows=None)
```

The last two functions live in `operations/topology_queries.py`; `_project_molecule` returns one independent Molecule. `_capture_factual_record` retains its existing `(MoleculeRecord, metadata)` result. Its default always captures complete facts. With an explicit private selection it filters Bond rows in complete order and then by requested Atom rows; all aligned raw Bond columns/cells follow exactly the same row selection. Public `copy`, `components`, conversions and persistence always use the full default. Component connectivity and component record projection use the same selection, never a full record accidentally reintroducing an excluded chord.

Projection copies/remaps the existing table/dependency state, constraints, cell, pair observations and transitional stereo transport through the current builder. For whole-Atom projections retain the existing whole-source stereo payload convention; operation exclusions do not delete source stereo history. For Atom subsets keep the existing row/allowed-edge filtering. Recompute no supplied fact during capture. Seed the existing topology dependencies at retained endpoints of excluded edges before any derived read on the independent task graph, as old hide did on its working clone. Do not copy inferred `CURRENT` across changed inputs. Charge/spin, annotations and source-frame lifecycle follow existing whole/subset rules; this cutover does not implement Stage7D/E policy changes. `_project_molecule` uses no legacy frames, matching current `copy()`/components; H caller metadata is copied by the existing helper.

Only an operation that needs ordinary Molecule/Atom chemistry on a reduced graph constructs this independent graph. Pure graph/snapshot consumers may continue to use their existing explicit filtered edge tuples, captured once. This is the same rule/algorithm, with the operation preparing its input. There is no ambient selection that alters ordinary getters.

## 4. Every operation mask and connected consumer

| Owner/consumer | Selected graph and exact migration | Preserved scientific/control meaning |
|---|---|---|
| Default molecular charge, `operations/chemistry.py::_calculate_default_molecular_charge` | Preserve initial preparation and H materialization. For a non-organic clone call private `_components(..., selection=ligand_selection)` instead of hide + public components. | Same `_default_molecular_charge`, organic classification, per-component H counts and isolated-metal formal-charge rule. No source fact rewrite. |
| Ligand proxies, `forcefields/ligand.py::_build_ligand_proxies` | Independent clone plus selected `_components(..., with_bindings=True)`; return coordinates through the original row bindings. | Same deterministic component enumeration, skip metal-containing components, sampler, attempts, FF, seed, warmup/refinement, warnings and best-candidate policy. |
| Ligand H working copy, `forcefields/working_copy.py::_hydrogenated_working_copy` | Keep the current complete working clone and its original handles; capture its complete record/mappings/donors, then create one additional all-Atom ligand projection. Run existing donor/H code with the same RNG and assemble one complete update on the retained working clone. | Same neutral N/O/P/S/As/Se donor condition, declared charges, explicit H, bidentate coordination, H counts and unique H labels. HT1 deliberately places the complete original Bond prefix before appended H, changing old hide/recover order; no new H counting convention. |
| Ring repair, `forcefields/repair.py` and `backend.py` | Select the same exact eligible Bond in the current operation Molecule; create an exclusion for just that handle. Optimize an independent projection through existing `_single_ob_optimization`; atomically copy accepted coordinates to the complete working Molecule. | Same distance/tie break, eligibility, watch batch, perturbation, short steps and outer best closed-frame selection. Opening is an operation graph, not carrier removal. |
| Native selected publication, `forcefields/native_adapters.py` | Validate intended-row/mask/coordinate mapping; publish selected coordinates only in the existing fact transaction. Never hide, delete, restore or reorder complete Bonds. | Mask observations stay in native result/session/trajectory; forced edges are not a new validity judgement. HT2 explicitly changes synthetic zero/mixed publication. |
| Complex native packing, `forcefields/native_packing.py` | Under one owner lock, collect complete rows once; ligand rows are complete rows failing `is_metal_ligand_bond`, intended coordination is every complete row satisfying it, sorted by existing directed row key. Remove hidden union/dedup. | Same `ComplexSessionInput` field order, dtype, contiguous layout, orders/kinds/aromaticity, charges, cell, metal rows and trajectory mapping. |
| Existing aromatic/Kekule/bond-order/H rule snapshots | Keep existing explicit ligand edge/ring selection and the same pure rules. Full carrier local-input reads use the complete index; operation-specific reads run on the projection above. | Covalent contribution, coordination number, heavy-order sum and valence are not equated. No new BondKind classifier or numerical rule. |
| Common packing, toolkit conversions, RD sampling, MCA/toolkit consumers | Existing default `mol.bonds` capture now means complete graph. No selector is added to their public signatures. The backend receives a projected Molecule when a reduced graph is needed. | Existing dative/unsupported/lossy adapter rules and protected public/native/stereo schemas remain. No fallback projection or normalization is added. |
| Native trajectory reconstruction | Keep ligand and coordination masks independently filtering their existing parallel row arrays. | Preserve dative/original order/kind evidence without reconstructing from lossy OB edges. Selected and terminal frames retain their own topology. |
| Scientific validation helpers | Migrate the three coordination-complex helper consumers to the same private ligand selection/components. Graph documents enumerate the current complete record rows. | Preserve sources, eligibility rules, case IDs, seeds, proposal counts, atom/bond mappings and outcome denominators. No old/current comparison runs against different graphs by accident. |

`ligand_selection` means **`not bond.is_metal_ligand_bond`**, exactly the existing XOR predicate: one endpoint is metal and the other is not. It does not mean “remove every metal-containing edge”, “only BondKind.SINGLE/DOUBLE”, or “reject every dative bond”. Metal–metal edges remain in this selection; ligand proxy code still skips components containing metal. A non-metal dative/zero/unknown edge retains the same current treatment. “Covalent-only” in valence rules remains their existing coefficient/scope calculation; no universal new topology mask is inferred from that word. Metal-free Atom subsets and ring scopes use existing explicit caller selections. Full is all bits one; ring-open is the current operation mask with exactly the chosen Bond bit cleared.

### H working-copy reassembly

First create the same complete `working_mol = copy(mol)` as today. Retain all of its original Atom/Bond handles, UIDs and order, and copy caller metadata through the existing helper. From that complete clone capture the original record, a source binding and selected-to-complete Bond row map, then create one independent ligand projection with all original Atom rows in order. The projection initially contains only selected Bond rows; its local UIDs do not replace the retained working clone's UIDs. Run the existing donor/H code there. Validate that all original projected Atom and selected original Bond identities remain and that every added atom/bond is permitted by the existing H/working-copy contract.

Construct the final complete record in original full Bond order: use updated projected rows for retained original selected Bonds, saved raw rows/cells/metadata for excluded intended coordination, then append newly created H Bond rows in their creation order. Apply projected Atom values/presence/source, real current-H payload edits and remapped dependency state. Do not append coordination after H, sort all Bonds by endpoint, rewrite original Bond identity from an OB return, or overwrite H edits with an old full payload.

Check the complete working clone's saved source before installing this assembled record through its existing transaction. Bind every retained complete-clone Atom/Bond handle; append fresh H handles allocated at that clone's existing high-water marks. Under accepted HT1, original complete Bond handles/UIDs/order remain the prefix and fresh H Bonds append after the original full Bond allocation. This intentionally changes the old hide/recover result order; it is not historical ordering parity. The temporary projection's handles never become aliases in this owner. Re-seed existing dependencies at reinstated coordination endpoints, just as old recovery did; do not label a ligand-context inference unconditionally current in the full graph. A failed projection/assembly discards the unfinished work and returns no modified source/caller. The existing `_prepare_working_copy_commit` / `_commit_working_copy` then checks every complete original Bond and publishes one factual batch to the caller, retaining its original handles/UIDs and full failure restoration. The extra independent H projection is a declared linear allocation/copy cost to measure, not a second persistent fact source or an unmeasured optimization claim.

C0 `/tmp/hotpot_plan20/long/stage7b-topology-20261103/baseline/receipt.json` and `summary.md` record the old working-copy Bond UID sequences: phenol `[0,1,2,3,4,5,6,8,9,10,11,12,13,7]`; OCCO `[0,1,2,5,6,7,8,9,10,3,4]`. Recovery appended coordination after H. Root explicitly accepts complete original Bond order followed by H instead. Check original membership/handles/UIDs, kinds, endpoints, donor chemistry, six added H per fixture, donor H counts `[1]` / `[1,1]`, zero donor charges and unique H labels separately from this allowed ordering change. Complete-list Bond indices and aligned transport/trajectory rows follow the accepted new order; do not hide the old observation or require its sequence as the candidate oracle. These fixture observations do not establish universal chemical truth.

## 5. Ring observations, trajectories and rollback boundaries

The later [Stage8B observed ring-open correction](../../../stage08/forcefields/contracts/decisions.md#observed-ring-open-aromatic-failure--operation-buffer-correction)
supersedes only this section's independent-Molecule native FF route and the
ring-repair row above: validate the intact source, filter task-local Bond buffers,
then keep the same source checks and coordinate-only atomic publication. Ordinary
independent projections/components and all membership/trajectory contracts remain.
The following paragraph records the original Stage7B route.

Extend only private seams:

```python
_single_ob_optimization(mol, forcefield, steps, *, selection=None)
_RingTrajectoryRecorder.record(..., selection=None)
```

For an explicit selection the backend builds `_project_molecule`, runs the existing public/native single optimization once on it, then checks the original operation source (`bonding_inputs`, `geometry`, `cell`) before copying coordinates to the complete working Molecule in a fact transaction. The native calculation and its existing coordinate validation remain one implementation. The default call continues to optimize the full graph. Public standalone OB signatures, common buffer types and C++ declarations stay unchanged. No selection is stored on the backend, carrier or record.

For RING_OPENED, PERTURBED and OPTIMIZED events, the recorder calls existing `trajectory.record(coordinates, bonds, ...)` with selected `BondTopology` rows. For RING_CLOSED/checkpoints/selected outputs it records the complete closed graph. Keep `ForceFieldTrajectory.record_molecule`'s public signature; its default now records complete factual topology. No duplicate Python bond topology DTO or Stage8B D05 consolidation is introduced. Every open energy is attached only to its open graph frame. Closed watch/checkpoint frames retain existing NaN/None energy behavior until actually evaluated; no finite value is fabricated or relabelled.

The inspected current repair helper restores its exact Bond in `finally` but does **not** restore coordinates; outer `_resolve_ring_piercings` owns global/watched coordinate snapshots and closed-frame rollback. Preserve that boundary. The new helper has no carrier membership to restore: discard the local exclusion in `finally`, record closure on the same failure path, and rethrow the original error. A perturbation already committed before backend failure may remain on the working Molecule, as today; a failed projected optimization must not partially publish its coordinates. Root/public outer working-copy failure still preserves the caller. Selection/recording failure before the old helper's `try` is not retroactively called a successful atomic repair. No broad catch, automatic retry or new coordinate rollback algorithm is added.

The fixed watch is scanned only on closed complete topology. Preserve no-edge/no-attempt behavior, attempts/budgets, minimum counts, original piercing reports, unavailable energy, last finite observations and selected-versus-terminal archive content. Retire the “unrelated hidden edge stays hidden” synthetic membership case. Retain its applicable isolation coverage with independent/coexisting selections and a projected operation graph: creating, using and discarding one exclusion must not alter another selection or any source facts. If a repair runs on an already projected ligand component, that component is its explicit input graph; the repair excludes exactly its chosen edge there. No new ambient or nested-mask API is required.

## 6. Native adapter disposition and scientific boundaries

`apply_native_selected_structure(mol, session_input, result) -> None` retains its exported name/signature and updates coordinates atomically. Under HT2, a valid synthetic zero or mixed `final_active_coordination_mask` **does not** reduce caller membership. It remains a valid operation observation; do not reject it merely to preserve old all-one assumptions, silently replace it with ones, or reinterpret its bit as chemical confidence. Validate the mask's binary cardinality against intended coordination rows and coordinate/mapping integrity before publication. Bad mapping/shape/mask is an existing `ValueError`/`RuntimeError` integrity failure, with no mutation; there is no new exception type or schema. Preserve current native/report validators and add the needed generic-adapter checks rather than relying on truncating `zip`.

Normal native Stage2 forces pending edges active and retains `coordination_bonds_forced` findings; a one bit is not successful screening. Stage3 requires all intended edges at entry and restores temporary openings, including failure paths. Leave these algorithms, topology revisions, selected/terminal policies and warning contents unchanged. Generic adapter coordinate publication is therefore the concrete observable difference, while production completion continues to retain all original Bonds. A terminal zero-mask frame may remain archived while selected coordinates are committed to the caller's full graph. Do not assume a terminal mask belongs to selected coordinates. The generic adapter does not attach an open-graph energy as current full-graph energy.

| Boundary/finding | Defined value and default consequence | Cutover policy |
|---|---|---|
| `coordination_bonds_forced` in Stage2 | Coordinates/operation observations may remain; warn/continue under current workflow policy | Preserve original warning and attempted/forced counts; full factual Bonds remain |
| Existing ring-screening indeterminacy/coverage findings | Finite attributed frames/reports remain observations | Preserve messages, thresholds and existing control; no new success rating or hard gate |
| Existing no-eligible-edge / exhausted repair reason | Best closed frame and available observations remain | Preserve outer selection/rollback and unavailable closed energy |
| Existing typed FF setup/native integrity failure | No accepted numerical publication for the failed operation | Preserve typed failure, diagnostics and original caller; no retry |
| Invalid/stale selection or record/mapping | No valid publication | Reject locally and atomically with existing exception classes; do not fabricate topology or values |

This cutover changes no warning-to-break policy, geometry threshold, hard chemistry rule, scientific success denominator or ordinary result envelope. Record any newly discovered algorithm defect separately before expanding scope. Known H/isotope/source gaps remain open, not “fixed” by a topology selection.

## 7. Factual record and current-version pickle cutover

Remove `bond_partitions` from N02 `MoleculeRecord`, `_make_molecule_record`, complete capture, builder validation/installation, component/merge and every producer: OB import, RD import, MCA DTO reconstruction and CCDC conversion. `operations/io.py` consumes the single OB record path and needs verification, not a second record factory. All new records describe one complete ordered Bond table; installing a record never classifies membership. Aligned fields are endpoints, raw Bond3 bytes, kinds, directions, sources, metadata and independent aromatic flags. Raw Atom19 values/presence/source, declarations, cell, constraints, pair observations, stereo and nine legacy frame domains keep their current encoding and lifecycle. No scalar codec round-trip of raw values is introduced.

The accepted ordinary current field order is exactly:

```text
atom_count, atom_values, atom_presence, atom_sources,
atom_isotopes, atom_frac_valences, bond_endpoints, bond_values,
bond_kinds, bond_directions, bond_sources, bond_metadata,
charge, spin_mult, cell, stereo_payload, legacy_frame_values,
legacy_frame_index, angle_constraints, torsion_constraints,
identifier, pair_wiberg_bond_orders, bond_aromatic
```

Current records have 24 positional pickle fields including partitions; removing the middle field without a codec boundary is unsafe. **Root accepts HT3's explicit hard current-record format cutover:** make the same private dataclass constructor keyword-only; `_make_molecule_record` constructs by keyword; current pickle reduction uses a new private `_restore_molecule_record(record_type, schema_version, field_values)` with literal version **2** and exact current field cardinality/order. The helper validates the version/cardinality and re-enters the keyword validating constructor. Keep the existing type argument so this reducer does not gratuitously change its type-preservation behavior. No stored `schema_version` field, new class, old partition reader, compatibility module, old/new runtime profile or dual record implementation is added. The version is a private serialization tag, not a native/stereo/MCA schema change.

Old unversioned record pickles calling the class positionally fail before installation; they are not silently shifted, normalized to active or migrated. Before source edits, freeze small 6d1ea01 Molecule/Atom/Bond/record pickle fixtures with original UID gaps/order/metadata/frames and source identity, solely to prove the declared rejection boundary. These are previous-current-version assets, not proof about arbitrary historical assets. If root requires old asset loading, stop this dependent cutover and approve an explicit separate migration policy rather than retaining hidden partitions. Decision10's later core-module relocation/historical-asset review remains open; this bounded choice neither claims all old assets have been audited nor creates a future core-path bridge.

Keep current Molecule/Atom/Bond reducer factory names/arguments and the seven-argument `_restore_molecule_raw` boundary. New complete record restoration precedes restoration of saved UIDs, constraint maps, pair keys, dependency state, allocator high-water and extra annotation/member graphs. `_restore_bond_member` resolves a complete-list row plus saved UID; `_bond_member_at` uses one list. Preserve current root-member/self references, detached/unregistered values and opaque metadata graph, fresh lock/owner, all supported pickle protocols, annotations, nine frame domains and selected index. A genuine fresh-process **new writer -> new reader** round trip is required. Existing tests/scout show current round trips, not old-version compatibility; old evidence cannot be relabelled as that proof.

## 8. Source, view, transaction and constraint coverage

The one `_topology_index` indexes incidence over all `_bonds`; retain its sorted-neighbour deterministic traversal and stable handle checks. No extra permanent row/adjacency index is added. `_SourceIdentity` topology includes complete Atom/Bond UIDs/order/endpoints. Representation, geometry, boolean constraint semantics, raw `forcefield_inputs`, bonding inputs, source frames and cell keep their existing declared content; remove partition concatenation and partition commentary only. Same owner/UIDs with a changed complete Bond order remains a source change. A temporary selection creates neither a new topology fact nor a source-frame selection. An operation projection has its own owner/source.

`views/_source.py` captures/checks the existing owner/index plus Atom/complete-Bond lengths, without hidden-length fields. Geometry and ring-representation checks remain; pure selection construction leaves existing views current, actual coordinate/factual changes still stale them as declared. `operations/geometry.py` resolves constrained sequence edges from the complete list; arbitrary declared Angle/Torsion paths still allow missing edges and repeated members under existing rules. The stereo capture adapter keeps the same N09 three fields and explicit referent/endpoints/raw checks over one complete Bond tuple; canonical payload decomposition is still Stage7E.

`_FactTransaction._TOPOLOGY_FIELDS`, `touch_topology`, working-copy snapshots and restoration retain the complete list/index/table roots and all member state, constraints, declarations, cell, stereo, pair observations and frames. Remove hidden-list roots; preserve exact list/container identity on failure and reverse touched native/Python constraint undo. Permanent removals detach raw Bond cells and filter unique-owner maps; failed installs/publications restore bindings and source currentness, preserving burned UID ranges. `_prepare_working_copy_commit` compares every original complete Bond, preserves its handle and rejects changed original order/kind/endpoints or a missing original atom. No facts are written by a rule or view.

## 9. File ownership, checkpoints and acceptance

[File scope](file_scope.json) is the accepted exact connected writer partition; root controls its separate release. Two production partitions own disjoint files; one validation partition owns tests/helpers. `core.py` has one writer. Source/test authoring follows root's separate [implementation release](../implementation_release.json); this design grants no runtime release. Keep partial source unpublished/unexecuted while record and consumers are inconsistent; do not ship a half migration or public aliases as an intermediate checkpoint.

| Checkpoint | Reviewable result | Next gate |
|---|---|---|
| C0 | HT1–HT5 accepted; frozen receipt records exact 6d1ea01/current source, 24 previous-current pickle fixtures and 23 comparable cost cells. Root independently verified all 43 artifact hashes, 1398 source/test bindings and six CP311 natives. This is baseline evidence, with the explicit HT1 ordering difference retained | Separate authoring release recorded; C1 readiness/review follows |
| C1 | Connected source/tests ready; all eight public entries and hidden owners/record fields removed, all listed consumers migrated, unchanged science/native schema explicitly checked | Root static source review/freeze before any product import/build |
| C2 | Prospective CP311 technical, real FF/backend/model and outside-checkout current-wheel checks in [validation plan](validation_plan.json); original failures/observations and exact origins retained | Root semantic/rollback/record/public acceptance; native rebuild only if native source/ABI actually changed |
| C3 | Bounded same-work cost and allocation observations with every adverse result; usable-cost decision plus maintained source/API/budget/test/report/map records | Root Stage7B owning completion decision and coherent business delivery |

N04 adds one private dataclass; N02 is modified in place; N01/N03/N05/N09, existing constraints/working-copy/trajectory/native records are reused. New Protocol/TypedDict/Enum/exception/native-ABI-mirror budgets are zero. Do not spend D05 consolidation credit or create N06/N07/N08 placeholders. The implementation record must update the owning budget ledger with the actual N04 fields/+1 and no unrelated deletion credit.

No C++/header/stub/CMake change is required by this design. Common/native transport layouts and existing six CP311 extension inputs remain. Python-only packer/adapter/storage changes require actual native-backed tests and current wheel assembly with exact binary/source bindings, not an automatic all-native rebuild. If implementation reveals a needed native change, root expands that file scope and rebuilds the affected extension after writer/source freeze before its evidence is accepted. Clean-all-native/ABI/SDK/sdist/manylinux and other Python versions retain later/global gates.

Root has accepted HT2's generic-adapter behavior, HT3's private pickle break, HT4's former cross-partition duplicates and N04's concrete budget, alongside the exact mask/H/order route (HT1/HT5), including the C0-observed intentional H/Bond ordering change. Design approval is separate from implementation release and from evidence acceptance. Actual checks run for this document are static document/source identity checks only; prospective validation has no fabricated pass count. Stage7C/D/E and later stages, Plan12 integration, broader science/GPU/external work, global performance/final acceptance, and Stage0–12 -> B_exp -> separately gated Stage13 remain open.

The 2026-11-03 alignment author ran a read-only `python -B -` stdlib check against the pre-alignment working documents: 27 assertions passed, all four assigned JSON documents parsed, 51 local Markdown link paths resolved, and all 71 performance workload recipes remained unchanged. Checks preserved accepted topology code/API/record field blocks, the eight-method table, exclusive file partitions/N04/native budgets, exact technical/business argv, installed/native/reuse boundaries, frozen sampling, historical constraint/preentry records, full scope and Stage13 sequencing. Scoped `rg -n` searches found no live reader of the three removed slowdown-quota fields; the historical Stage5 copy remains untouched. These are document/preservation checks, not candidate validation or root acceptance. No product import, test, build, install or calculation was run by this alignment task; C1 and subsequent evidence/release boundaries remain in force.
