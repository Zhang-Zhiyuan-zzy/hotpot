# Stage1 API and consumer review record

Baseline source: `e6082f5fd67cb1054726ddcf43db7f963f7091ed`.
Scope: the latest Plan20 implementation instruction, the main plan's
Sections7/9/16, and the chemical-type and data-object companion records.
Production code is unchanged by this inventory capture.

The exact A2 constructor/projection proposal and its staged consumer migration
are recorded in [stage1_contract_decisions.md](stage1_contract_decisions.md).
The owner accepted that previously unspecified parameter decision on
2026-10-28; its cutovers retain the Stage3/5/7C implementation gates.

The documented implementation direction is executable. The exact differences
below identify the consumer work and any still-unspecified detail before its
owning stage. An open detail does not block unrelated inventory or baseline
work. This record does not approve a new public symbol merely because a private
implementation would find it convenient.

## Reproducible records and interpretation

Run from the checkout with the selected baseline interpreter:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/hotpot-plan20-inventory-mpl \
  /tmp/hotpot-plan20-env/bin/python \
  plan/20_fact_rule_architecture/capture_baseline_inventory.py \
  --source-root /tmp/hotpot-plan20-20261028/source-b20 \
  --output plan/20_fact_rule_architecture/baseline_inventory --runtime
```

The caller can select another output directory. The tool writes only there;
runtime probes use new temporary molecules and do not edit production files.

| Record | Evidence and limits |
|---|---|
| `manifest.json` | Source commit/tree, every production Python digest, counts, output digests and tool interpreter |
| `runtime_surface.json` | Actual root/cheminfo visible names, signatures, physical definitions, loaded native-module digests, class-identity checks, dynamic Atom/Bond fields and root CLI command table |
| `classes.jsonl`, `functions.jsonl` | Every production class definition, class methods/fields, top-level function signatures and explicit-import/call consumer candidates |
| `interface_inventory.jsonl` | Declared chemical-type constructors/properties/methods/collection access and dynamic fact projections, with source read/write observations |
| `declared_exports.jsonl`, `cli_arguments.jsonl` | Explicit export declarations and parser argument definitions; dynamic expressions remain visible rather than guessed |
| `imports.jsonl`, `documentation_imports.jsonl` | Exact AST imports and documented direct paths; historical/planning examples are marked separately |
| `private_writes.jsonl`, `canonical_property_write_candidates.jsonl`, `architecture_debt.jsonl` | Exact matched source sites and required removal stages; item writes into authoritative buffers are included; property-write receivers still require ownership review |
| `native_struct_candidates.jsonl`, `explicit_type_aliases.jsonl` | Separately counted native named-struct and explicit TypeAlias syntax; macro/anonymous schemas and implicit aliases still need component review |
| `migration_inventory.json` | All 173 Section9 table rows, owning stages, and explicit subdivisions of mixed rows |
| `../stage_verification_map.json` | Exact existing test selections and commands, plus explicitly marked future contract tests |

Static imports and call expressions are consumer evidence, not proof that a
public name lacks dynamic, pickle, plugin or external users. Name-based role
labels identify the initial domain; ambiguous classes are explicitly marked
for review. Generated dataclass constructors and nested-buffer guarantees must
be checked with their actual field/inheritance contracts before schema changes.

The first baseline capture found 403 production Python files, 782 total class
definitions, and exactly 347 dataclasses, 3 TypedDicts and 13 explicit Protocols.
The latter 363 candidates reproduce the companion audit. They are not all pure
data objects and are not all consolidation targets. No reduction is claimed.
The AST inventory found 67 production and 74 test files directly importing the
core module or its symbols. The old 72/81 text counts are historical context;
the captured path list and final zero-reference scan determine completion.

The root and chemistry facades currently expose 75 and 66 non-private names,
respectively, and neither declares `__all__`. This includes incidental imported
modules and typing names; visibility alone does not establish a supported API.
The generated list freezes the old surface for the explicit-export review.

## Carrier and value changes

| Item / owning stage | Current concrete interface or observation | Documented target and consumer work | Remaining decision |
|---|---|---|---|
| A1 / Stage3–5 | `Atom.atomic_number` and `symbol` are writable; `_atomic_number_setter` writes electron configuration, valence and implicit H immediately (`core.py:3056`) | Keep existing names; one atomic-number owner and cheap dependency marking; lazy chemistry through named operation | Exact affected-field/source table is implemented and tested in Stage3/5; no extra public alias |
| A2 / Stage3,5,7C | `Atom(mol=None, *, attrs_array=None, update_electron_config=True, **kwargs)` (`core.py:3189`); supplied ndarray is retained by reference | Private builder imports complete facts; isolate buffers in Stage3, remove old constructor parameters with their Stage5 consumers, move partial-charge ownership/absence encoding in Stage7C | Owner accepted the exact signature, fixed snapshot and model-encoding contract in `stage1_contract_decisions.md`; do not discard later-stage facts during initial storage extraction |
| A3 / Stage3 | Atom and Molecule coordinate getters currently produce arrays that do not share the Atom attrs buffer in the captured probe; setters and x/y/z already exist | Keep the existing setters and shared current-frame fact; return independent buffers; batch and scalar writes use one commit path | Preserve existing shape/order/unit behavior; immutable result buffers require separate alias validation |
| A4 / Stage3 | No current `Molecule.update_atom_facts` method | Exact candidate `update_atom_facts(*, atoms=None, **columns) -> None`; Atom handles select rows; broadcast only approved fields, coordinate shape N×3, consistent overlapping aliases, one commit | Accepted fields, selection, broadcasting and error behavior are specified in `stage1_contract_decisions.md`; apply the owning-stage public-interface review before exposure; no parallel edit API |
| A5 / Stage3 | No current `Molecule.atom_feature_matrix` method | Exact candidate `atom_feature_matrix(fields, *, dtype="float32") -> numpy.ndarray`; explicit order, schema units, independent task array, reject unknown values | Supported numeric projections, duplicate/empty columns, unknown-value and dtype errors are specified in `stage1_contract_decisions.md`; no invented features or mask return schema |
| A7 / Stage5 | Dynamic row includes writable `n/s/p/d/f/g`, `valence`, `implicit_hydrogens`, `is_aromatic` | Electron configuration/valence/aromatic projection become readonly; explicit implicit-H setter remains; complete aromatic assignment starts from Molecule | Internal array writers and imported derived values must migrate in the same owner stage |
| A8 / Stage7C | `Atom.partial_charge` is a writable dynamic float column; newly created atom reads `0.0` without a calculated result | Readonly `Optional[float]`; `None` means no selected value; existing complete partial-charge calculation/assignment supplies the sole record | `stage1_contract_decisions.md` distinguishes absence, stale RuntimeError and historical results; migrate all full-set assignment consumers and preserve explicit zero |
| A9 / Stage7C | `Atom.mca` and `Molecule.mca_sites` raise `AttributeError` before calculation (`core.py:3741`, `1975`) | Keep these names and absent-result exception; project one selected identity-bound record, remove `_mca`, `_mca_sites` and `properties["mca"]` copies | Use the recorded stale RuntimeError policy; getters never launch model/GPU work |
| Bond / Stage5 | `Bond(atom1, atom2, **kwargs)` and existing kind/order/source/direction surface; kind is not automatically writable | One coupled kind/order representation; source/direction retain their own facts; complete assignment is atomic | Preserve characterized zero/dative/unknown representations; do not infer a new kind setter |
| Views / Stage4–5 | Angle/Torsion instances own mutable constraints; Ring has aromatic/Kekulé write paths | Move angle/torsion constraints before cache removal; make Ring readonly in the same commit as assignment-consumer migration | Arbitrary constraint selections must use an adequate existing Molecule operation or an exact separately listed new method |
| View failures / Stage3–4 | No existing `StaleStructureViewError` implementation | Relevant source changes invalidate a source-bound calculation while detached membership stays inspectable | Use RuntimeError as recorded in `stage1_contract_decisions.md`; no new public exception and no scientific-uncertainty rejection |
| Crystal / Stage4,7A | `Crystal(a, b, c, alpha, beta, gamma, *, mol=None)` (`crystal.py:16`); public `from_matrix`, `obcell`, `space_group`, `lattice_group` | Sole six-parameter value; remove `mol` back-reference and toolkit/placeholder surface; adapter makes fresh toolkit values; preserve characterized matrix convention | Exact removal and first-two-row alpha convention are recorded in `stage1_contract_decisions.md`; `Crystal.matrix` remains unestablished |
| Thermo / Stage7A | `Molecule.get_thermo(temp=298.15, pressure=101325.0) -> Thermo`; `Thermo` is a dynamic external-property proxy (`call_thermo.py:246`) | Isolate thermo/RDKit bridge and typed result ownership | Preserve all 181 declared properties and dynamic Chemical forwarding as a caller-owned adapter result; no replacement with a reduced DTO |
| Conformers / Stage7D | Current/source/generated arrays are mixed behind one public access surface | Preserve indexing/array-access contract as a deterministic projection; add records for explicit source frames; generated ensembles are P3; load copies current frame; clear covers both owners | Nine-key integer/slice projections and ordering are recorded in `stage1_contract_decisions.md`; empty-iteration and partial-mutation defects require Stage7D fixes; no second persisted frame copy |
| Lifecycle / Stage3,7,9 | Constructors, foreign insertion, copies/components and hidden-bond copies have existing tests | Use main-plan Decisions16/19: copy/remap factual rows and constraints, independent components, drop generated P3 results/runtime contexts, preserve presence/source/dirty status | Exact regression cases accompany each owner migration; these are not new generic cloning semantics |

`attrs_array` has live production consumers in `obconvert.py:90`,
`plugins/ccdc_api/convert.py:47`, and `core.py:1628`. The latter explicitly passes
`update_electron_config=False`. This is enough to rule out deleting those
constructor parameters without a coupled consumer migration. Source references
also include non-public row-building helpers; private status does not authorize
discarding their imported facts.

The following optional proposals are deliberately not added by this record:
`Atom.isotope`, `Crystal.matrix`, `Atom.new_attrs`, a public fact-table/schema/
dirty-state API, and the optional `AtomPairs` rename. Existing stereo transport
continues to convey isotope facts. The existing `AtomPairs` name can identify
the immutable implementation without an alias class or duplicate export.

## Record-family consolidation and protected boundaries

| Item / stage | Concrete baseline | Recommended minimum change | Required closure before implementation |
|---|---|---|---|
| D01 / Stage8A | Five result classes have `identity, facts_identity, candidates, admissibility, resolution, confidence, findings` | One same-domain result family with explicit quantity identity and typed candidates | Exact class/path, quantity encoding, constructor field order, return and serialized schema; retain all candidate/authority semantics |
| D02 / Stage8A | Finding/reason records share code/message/related identities; StaticEligibilityFinding lacks the third field | One finding schema with explicit namespace and role carried by its containing context | Preserve absence/default distinction and each consuming reason/eligibility/projection role |
| D03 / Stage8A | ProviderToggle/RuleToggle are structurally identical | One activation value with provider/rule ID domains kept distinct | Exact constructor/tag and enabled-policy permissions; sharing storage does not unify policies |
| D04 / Stage8B | `OptimizationStoppingCriteria` and Python `OptimizationStoppingOptions` have five matching fields; workflows copies them one by one | Prefer retaining existing public domain name `OptimizationStoppingCriteria` and removing the Python-only transport copy | Preserve all field defaults/units and validation; native C++/pybind `OptimizationStoppingOptions` remains an ABI value, not a duplicate to delete |
| D05 / Stage8B | `BondTopologySignature` and trajectory `BondTopology` describe ordered endpoints, order and kind | Reuse one immutable record and an adapter factory | Select the existing retained name, endpoint normalization and unchanged historical JSON projection; index/UID mapping belongs in a complete snapshot |
| D06 / Stage8B | Atom topology and trajectory rows differ by stored symbol and historical schema | Assess composition only after consumers/serialization are characterized | No reduction is booked for this optional consolidation |
| D07 / Stage2 or owner migration | AtomPairKey, RingEdgeDistance and dormant coordination candidate/result records have differing public/placeholder history | Delete only after explicit public/consumer disposition; existing documented Triangle and meaningful narrow report subtypes stay | Zero construction count alone is insufficient evidence of zero public or reserved-contract use |
| D08 / Stage8D | MCA task messages and public prediction DTOs have real multiprocessing/serialization consumers | Retain prediction DTO names/schema; use existing records and explicit ownership during move | No cosmetic MCA message rename until supported imports/pickle consumers are identified |
| D09 / Stage9 outer-consumer work | Harness/XTB process records have distinct provenance/path scopes | Update chemistry imports; independent plugin cleanup stays outside the critical path | No unsolicited process-report schema rewrite |
| Stable stereo / Stage6 | Five facade functions and full Level1 transport schema are fixed by Plan17 | One set of physical definitions and a maintained stereo facade | Preserve class identity, fields, ABI order and exact round trips |
| Stable wrappers / Stage10 | Independent rdWrappers and obWrappers public Python/C/C++ contracts | Preserve component paths and identities; adapters invoke the one implementation | Remove superseded bodies only; do not delete a retained wrapper as an “old backend” |

The companion's candidate 16→5 consolidation would remove 11 dataclasses only
if all required public/schema decisions and consumer migrations actually land.
New table, identity, update, selection and result definitions must be booked
on the additive side. The captured 164 ordinary named native-struct candidates
are separate from Python counts; they are not all public ABI types.

## Import, serialization and verification disposition

The root classes move once to `hotpot.cheminfo.structure`, with ordinary root
exports preserved. Existing advanced package paths follow the main plan's
hard-move inventory; standalone wrappers and the maintained stereo facade are
explicit retained paths. The final import gate checks real consumers, not a
historical numeric target.

Existing pickle evidence includes the force-field public function/contract and
spawn-target tests (`test_forcefield_package.py:133,150,188,196`), MCA DTO tests
(`tests/mca/test_serialization.py:38,147`), and conformer-adapter spawn tests
(`forcefields/test_conformer_sampler_seam.py:157`). GPU preprocessing also
spools prepared values with pickle (`AImodels/mca/gpu_backend.py:109,118`).
These in-process/current-version contracts must survive migration. They do not
prove that no user owns older `cheminfo.core` pickles. Historical cross-version
pickle compatibility remains an explicit Decision10 disposition; no alias
module or hidden migration fallback is introduced.

`stage_verification_map.json` lists actual existing test files, separates native
high-compute tests, and records the planned structure-contract filenames for
Stages3 onward. Its placeholders for a selected wheel/artifact directory are
resolved and logged at execution. Planned tests and unexecuted matrix tiers
are not counted as passed. API/schema changes cannot advance solely because
this inventory tool ran successfully.

All 173 migration rows are assigned an existing owner stage. Mixed cache,
constraint, class/storage, native/build and adapter/workflow rows have explicit
subtasks, so Stage10 cannot absorb an unassigned semantic refactor. Conditional
Stage11 asset/command work retains its existing decision and deferral rules;
Plan21 remains documentation-only in this delivery. The Stage13 interface is
defined against actual `B_exp` only, and its future tests are intentionally not
invented during this baseline capture.
