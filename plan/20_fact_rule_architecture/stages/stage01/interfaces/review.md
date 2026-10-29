# Stage1 common interface semantic review

This review completes the finite interface/mutation/read table required by the
main plan's Stage1 item 9 and the chemical-interface companion Section6. The
machine-readable table is [reviewed_interface_contracts.json](../../../reviewed_interface_contracts.json).
It supplements the captured API inventory with ownership, dependency,
presence, permission, operation and lifecycle semantics. It adds no production
protocol, registry, descriptor, common base class or chemical algorithm.

The baseline remains source commit
`e6082f5fd67cb1054726ddcf43db7f963f7091ed`, production tree
`4a2086e717327c99056ef8cda200b0786e688219`. The JSON binds the inventories,
103 Python source modules, native ABI evidence, selected import consumers and
the governing design documents by SHA256. Exact signatures, defaults,
annotations, dataclass field order and inheritance remain in those bound
sources; the semantic table does not duplicate or silently rewrite them.

## Scope and meaning of review

| Coverage | Count / disposition |
|---|---|
| Repository Python class definitions | 782, unchanged baseline |
| Definitions under `hotpot/cheminfo` | 485, all explicitly dispositioned |
| Named types assigned to a contract/scope family | 384, including 20 supporting task/resource types and 8 retirement candidates |
| Private implementation or deployment graph definitions | 101, migration tracked; not claimed public chemical contracts or dead code |
| Captured interface entries / owning types | 1,837 / 318 |
| Entries assigned to explicit scope families | 1,818, including 95 support entries and 19 retirement entries |
| Implementation entries separately tracked | 19 |
| Specific carrier member mappings | 204 unique `(type, entry)` overrides |
| Supplemental ordinary/dynamic member groups | 15, including 181 Thermo forwarding names |
| Enum types with captured value expressions | 62 |

`scoped_semantics_reviewed` means that the existing schema, role and applicable
ownership/lifecycle policy were reviewed. It does not mean that the planned
implementation exists, every dynamic consumer was discovered, every returned
array is immutable, or a new public API was approved. Supporting resource
types and retirement candidates have distinct status labels. Original
`domain_contract_or_implementation_to_review` labels are preserved as source
observations; the new table supplies their reviewed disposition without
changing historical evidence.

An entry resolves first through `member_contract_overrides`, otherwise through
its explicit `type_groups` family. The original source supplies its exact
schema; an override's `kind_policies` further distinguishes a retained
property read from its retired setter. `supplemental_member_contracts`, `ordinary_fields_from_class_inventory`,
`declared_class_data_names`, enum values and inheritance notes cover interfaces
that a properties/annotations-only scan misses. General Python inheritance
and unchanged stable DTOs use their bound schemas; this does not require a new
runtime object for every entry. Existing top-level exports remain determined
by the separate export/runtime inventory, not by class-name capitalization.

Unless a statement explicitly says **baseline/current**, ownership and
mutation policies describe the planned target and require their named stage
gate. Main-plan, companion and explicit owner decisions retain precedence;
this table cannot authorize a new signature by inference.

## Substantive ownership and signal decisions

| Interface family | Reviewed boundary |
|---|---|
| Molecule / Atom / Bond | One factual owner, scalar aliases and complete batch writes, isolated numeric exports, coupled bond representation and complete source-preserving import. UID, current row and public source ID remain distinct. |
| Charge, H, electron and aromatic values | Explicit facts retain source/presence. Pure derived reads use declared dependencies; complete assignments install once. Scientific uncertainty is not itself a stale-source or execution exception. |
| Current results / history | Current selected projections check required source validity; returned historical results keep their content. Absent partial charge, absent MCA and stale selected data retain their separately specified behavior. |
| Views / constraints | Member identities and source-dependent calculations are distinguished. Views do not own another dirty state; constraints project the one Molecule owner. Existing AtomPairs name and RuntimeError route remain; immutable collection migration must account for inherited dict mutators. |
| Crystal / MolBundle | Six-parameter cell convention retained; old independent values do not follow host replacement. MolBundle owns an independent list containing shared Molecule handles; no invented custom iteration/length API. |
| Conformers | Source insertion order, then result attachment order and original frame order. Load changes only current coordinates; clear removes both owners. Nine-key target and atomic failure are distinct from known baseline defects. |
| Calculator science | Facts snapshot, rule evidence, candidate proposal, selection, backend projection and historical evidence are separate roles. None/uncovered, explicit zero, uncertainty, admissibility and executability are not collapsed. |
| Geometry / search / assembly | Independent numerical values, source-associated measurements, mutable query specifications, hit views and task factories retain their distinct lifecycles. Template schemas remain tied to the asset review. |
| Protected stereo / native / wrappers | Preserve exact schema, status codes, unit/order and buffer ownership. Frozen dataclass shells do not imply immutable ndarray/mapping contents. No new protocol changes these protected seams. |
| MCA / CBond | Public scientific results, versioned process snapshots and short-lived scheduling messages remain separate. CBond path and aggregated structure probabilities are not merged by similar names. |

`Molecule.get_partial_charge()` is an explicit Open Babel calculation returning
`tuple[float, ...]`; it does not assign Atom partial charges. Its existing
model arguments and failure behavior remain. `assign_partial_charge()` performs
the explicit complete calculation/assignment. These methods are not ordinary
current-result properties. Likewise `cid`, `name` and `cas` are existing
PubChem query conveniences.

The empty `Molecule.assign_aromatic()` is scheduled for deletion;
`determine_rings_aromatic()` is the retained Molecule aromatic operation.
The explicit removal family also covers generic public mutation, angle/torsion
cache-update methods, Ring/JointRing mutation and hidden-bond APIs at their
respective migration gates. Capturing an old method does not declare it a
retained facade. Ring's aromatic read remains separate from its removed setter.

Copy/component/merge follow the main plan's Decision19 proposal: preserve
attributable facts and the specified P1 state, drop generated P3/runtime state,
and apply the explicit annotation-container, cell compatibility and conditional
charge/spin rules. `decision19_lifecycle` records the exact policy. A complete
import preserving supplied values does not authorize keeping all generated
results on copy.

`Thermo` has 181 declared forwarding names plus broader `__getattr__`
forwarding to `Chemical`, including executable external objects. The adapter
move retains this caller-owned proxy. Replacing it with a small frozen record
would remove real interfaces. `GraphSpectrum` likewise currently retains its
input ndarray and exposes `vectors` as its alias; it is not falsely recorded
as a deeply immutable result.

## Values and interfaces missed by the first inventory

- `Molecule.charge`, `properties` and consumer-written `identifier` are real
  ordinary/dynamic state. Metadata and net declaration require explicit
  migration; neither is an Atom row or a hidden selected-result owner.
- Atom and Bond inherit mapping/ring access from MolBlock; Bond also inherits
  AtomSeq endpoint aliases. Angle/Torsion constraints and Crystal's six
  writable ordinary fields cannot disappear because they lack property
  annotations. Bond `attr_dict` additionally preserves kind, direction,
  source and copied source metadata.
- tmQM writes `Atom.frac_valence` and `AtomPair.wiberg_bond_order`. Existing
  model consumers read pair columns in the order
  `(wiberg_bond_order, length_shortest_path)`. The baseline path length here is
  **number of path vertices**. A default Wiberg zero cannot distinguish
  absence from a supplied real zero. Values must survive the first Stage4
  pair/cache extraction; Stage7B/7C then completes pair consumers and selected
  result ownership. Exact P1/P3 classification, absence/setter and copy policy
  remain explicit gates before the corresponding old owner is removed.
- `BackendIdentity` is a frozen digest dataclass; `BackendContext` is the
  actual backend Protocol. Facts/state/backend identities retain distinct
  digest domains. Existing atom maps are not yet implemented private UIDs.
- Worker/GPU types ending in `Result` are often short-lived messages.
  `CPUPreprocessTask.sites=None` triggers detection; `()` means already
  detected with no sites. GPU request/ordinal IDs correlate communication,
  not molecular identity. Mutable feature arrays remain mutable inside frozen
  messages. `_PendingRequest` is a frozen shell despite broader audit prose
  grouping pending task state as mutable.
- Stereo current/source H, string absence/empty text and native integer
  presence masks remain distinct. Native transport buffers are owned writable
  task arrays. RDKit Python outputs copy native buffers before native result
  destruction; C API callers retain explicit destruction obligations.

## Remaining owning-stage gates

The user's **2026-10-28 A2 acceptance** covers the exact constructor removal,
independent numeric snapshots, missing-result NaN encoding and unchanged model
input encoding in [stage1_contract_decisions.md](../contracts/decisions.md).
It does not approve every other new public signature or combine the
Stage3/5/7C changes into one early cutover.

| Gate | Required before its implementation |
|---|---|
| Stage3 N01–N09 / A4–A5 | Final private layouts/dtypes/budget and review of the concrete new batch/export interfaces before publication. No generic Atom-record system or extra authoritative table. |
| Stage5 / Stage7C A7–A9 | Migrate every derived/result writer with permissions, source presence, missing/current behavior and unchanged model encoding. |
| Stage4 then Stage7B / Stage7C pair science | Preserve supplied Wiberg and frac_valence from first extraction; complete importer/result ownership and reviewed P1/P3 lifecycle with zero/nonzero and model pair columns. No new public DTO is implicitly approved. |
| Stage3/4/5 views, Stage7B completion / Stage7D conformers | Identity/buffer contracts precede mathematical views and angle/torsion constraints in Stage4; Ring writers/caches are removed in Stage5. Stage7B completes remaining views/constraints; Stage7D completes frame transactions and known-defect repair. |
| Stage8A D01–D03 | Review exact constructors, codec compatibility, namespace/digest behavior and consumers before publishing consolidated result/finding/activation types. |
| Stage8B D04 / D05 | The frozen data-object budget assigns both force-field consolidations here. Keep native ABI and historical topology schema while removing Python duplicates with their consumers; Stage3 shared contracts do not count as completed removals. |
| D07 / Stage8D D08 | Reserved public dormant records are not automatically dead; retain actual MCA DTO/message and pickle semantics. Cosmetic message renaming remains outside scope. |
| Future MetalCluster | Preserve the user-requested design registration in Plan21; unspecified constructors/members/matching/export details are not newly implemented here. |
| Stage13A / Stage13B experiment | Complete Stage0–12 with required verification, scientific reports, final efficiency comparison, final report and Git commit `B_exp`; create `experiment/plan20-kekulize-aromatic-protocol`. Define/test/commit `D_exp` without production integration, then integrate/test/compare/report/commit `I_exp`. Existing Decision12/13 deferral rules remain the only applicable exceptions. |

These later gates do not become new Stage2 blockers. Stage2 still performs its
own consumer/import/build and applicable public-deletion checks. The interface
review does not mark Stage1 complete without the independent testing and
performance deliverables.

Known baseline defects are recorded, not normalized into intended behavior:
scientific-facts extraction currently turns implicit-H zero into None;
empty Conformers iteration raises TypeError; an energy-length failure can
occur after coordinate mutation; and the Conformers constructor initializes
`_partial_charges` while its access schema uses `_partial_charge` until
`clear()` creates that name. The last finding is static and requires
owning-stage characterization; it does not retract the nine-key target.

## Stage alignment audit

The final table was cross-checked against the complete main-plan Chapters9–17,
the chemical-interface companion, the full data-object audit including its
appendix, the Plan17/20 alignment and Plan21 registration. The stage map has
the following boundaries; no later stage delays an earlier assigned migration:

- Stage3 splits immutable element/ideal-bond reference loading and establishes
  common identity, buffer and commit contracts. Stage5 owns elemental,
  classification, electron and chemical descriptor rules.
- Stage4 owns mathematical views and carrier-to-geometry adapters, including
  initial pair extraction; Stage5 completes Ring's read-only transition and
  its writer/cache removal. Stage7B finishes remaining pair/hidden-topology
  and constraint consumers. Search-specific migration remains Stage8C.
- GraphSpectrum's chemical descriptor policy belongs under chemistry,
  with reusable matrix mathematics split out; it is not wholly a mathematics
  object merely because it contains a numerical spectrum.
- Stage8A owns scientific-state consolidation; Stage8B owns D04/D05
  force-field consolidation. Stage8D relocates the selected sole MCA route;
  the aligned `B20`/`F20` route remains Python-RDKit ETKDG plus MMFF/UFF.
- Stage11 is the separately governed outer-command/data-ownership tail.
  Stage12 supplies final main-scope hardening and reporting. Only the validated
  Stage0–12 commit `B_exp` permits Stage13A/B's separate experiment. Existing
  aromatic/Kekule extraction at Stage5 is not delayed or repeated there.

## Verification

Static checks passed for the 485-type disjoint partition, all 1,837 entry
dispositions, unique/existing member overrides, supplemental type references,
source/inventory/document hashes, enum source declarations and JSON parsing.
The review preserves rather than overwrites the original inventory. This
documentation-only task reran no production tests or benchmarks; their
existing evidence and separately recorded Stage1 matrix remain separate deliverables.

The review's status can therefore be stated narrowly: the common semantic
table now exists and exposes the remaining decisions at their proper stages.
It provides no evidence that the production ownership migration has already
landed.
