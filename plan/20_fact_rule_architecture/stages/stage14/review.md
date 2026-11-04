# Stage14 implementation plan — General chemical information ↔ PyG Data

Current completion overlay, 2026-11-05: root accepts Stage14A–E within the
declared useful general Molecule/Crystal transport domain. The [Crystal delivery
report](crystal/validation/report.md) and [checkpoint](crystal/validation/checkpoint.json)
bind source02 1880 passes, installed CP310–314 1880 passes each, documentation,
fresh sdist, attributed wheel repacks, actual mixed Crystal CUDA and accepted
bounded current-only costs. Original review/test/comparator failures are retained.

The [current Crystal/Data contract](crystal/contracts/decisions.md) publishes
`hotpot.molecule/2` as a hard cut with ordinary Data; no schema1 loader remains.
Historical14A/schema1 and molecular checkpoints retain their original scopes.
The original plan body below preserves its inspected-source context and design;
current runtime/interface status follows this overlay and the current contract.
Stage0–12 acceptance and `B_exp`, and Stage13's `D_exp`/`I_exp`, are unchanged.
Final combined13/14/15 union validation/delivery and root review/application of
the related business commit remain pending; overall Plan20 is not closed.

## 1. Required outcome and present boundary

Provide general conversion between `hp.Molecule` and
`torch_geometric.data.Data`, then Stage15's `hp.Crystal` and Data, with distinct
purposes:

- `to_pyg()` gives PyTorch/PyG the maximum **useful**, chemically interpretable
  training information for new models: structure, available scientific
  observations and supported annotations/targets with their actual mappings.
  Select fields for that utility and usable batching/cost, rather than copying
  every internal object field.
- `from_pyg()` structures the chemical information actually encoded in Data into
  Molecule/Crystal so it can be queried and reviewed. Preserve edited supported
  fields and minimize information loss; missing chemistry is not reconstructed
  from a trained embedding or guessed column meaning.

Bidirectional preservation applies to the documented useful-field contract,
not a complete clone of runtime state. Locks, caches, private identifiers and
transaction machinery are outside the contract. A native third-party Data
object supplies only information it contains; conversion cannot recover an
unknown bond kind, missing stereochemistry or absent lattice image.

At inspected source `ec4d61889a7262df765da29082e405b3c612cd15`:

| Current owner | Observed behavior | Stage14 consequence |
|---|---|---|
| `structure/molecule.py::to_pyg_data` | Delegates to `plugins/PyG/data/utils.py::mol_to_pyg_data` | Replace this public output with the general training/review conversion under the owner's explicit instruction. |
| `plugins/PyG/data/utils.py` | Atom float32 features, directed bond arcs, pair/ring tensors and optional batch | Keep necessary model feature mathematics at the actual model-input owner; migrate its callers and never infer original facts from rounded features. |
| `operations/collections.py`, `structure/bundle.py` | Ordered dataset export, ordinal filenames for unnamed members | Export the new transport Data and migrate actual dataset readers; retain only the collection ordering/naming behavior still required by the new contract. |
| `structure/_builder.py::MoleculeRecord` | Factual values, presence/source, bonds, stereo, frames, constraints and partial-charge record | Reuse applicable codecs and atomic reconstruction for the useful-field contract; do not require export of the whole private record or add another molecular fact owner. |
| MCA/result owners | MCA attachment is not a field of that complete builder record | Inspect available typed observations and their atom/site mapping as useful training information; do not assume the builder record includes every result. |
| `adapters/pyg.py` | Lazy Torch serialization access only | Place PyG-specific encoding here; core imports remain independent of Torch/PyG. |

The current production models do **not** call public `to_pyg_data()`, directly
or transitively. CBond follows `cbond/apply.py` → `operations/inference/features.py`
→ NumPy encoders in `adapters/structure/inference.py`; its atomic-number graph
input is not the public sixteen-column float32 `x`. MCA follows its workers →
`pipeline.build_features` → RDKit/Uni-Mol preparation and ONNX session input
casting. Shared feature helpers and separate training/ComplexFormer consumers
still require explicit impact review. The concrete public-method caller in
current product code is `operations/collections.py`; migrate it to `to_pyg()`.
There is no model-driven need for a legacy public-output compatibility path.

Scope is useful general chemical information for training and inspection.
Deployed models retain their customized inputs; this stage does not redesign
or retrain them. It is not a generic object serializer, arbitrary Python-object
pickle, or a new chemical ontology. Unknown facts and explicit zero remain
distinct. General conversion does not launch chemical/model calculations to
fill missing data; any included derived projection must have a documented
meaning and its actual available source.

## 2. Interfaces and separation of purposes

The owner selects the following public names; Stage14A settles the exact mapping arguments:

```python
Molecule.to_pyg(self) -> Data
Molecule.from_pyg(cls, data, *, field_map=None) -> Molecule
# Added only after the Stage15 aggregate exists:
Crystal.to_pyg(self) -> Data
Crystal.from_pyg(cls, data, *, field_map=None) -> Crystal
```

The owner explicitly selects `to_pyg()` and `from_pyg()` and authorizes replacing
`to_pyg_data()` directly. Remove that method and its obsolete arguments in the
connected cutover, migrating general export/dataset callers to `to_pyg()`.
Add no compatibility switch, legacy-output path or legacy-use documentation.

Every deployed model, including CBond/MCA, owns its Molecule/Crystal-to-model
feature interface at its corresponding model module. Design that interface
on demand for the model's actual inputs and maximum efficiency; no general
Data intermediate or common all-feature schema is mandatory. Preserve existing
trained widths, feature order, H semantics, dtype and padding. Shared pure
encoders may be reused, but the model owner selects their use and owns the
contract. General `to_pyg()` supplies information for new-model training without
redefining deployed-model inputs. Retire a Data-wrapper helper only after its
actual remaining consumers have migrated or no longer need it.

A Hotpot-authored transport Data has one documented schema marker and the
necessary field/units contract. This is a transport schema, not a success or
confidence envelope. A foreign Data needs a selected documented `field_map`
unless it already carries that schema. Specify exact mapping arguments and
units at Stage14A; do not introduce general plugin discovery or guess that the
first `x` column is atomic number. Ordinary returns are Data or the requested
chemical object; diagnostics use the existing finding/error channels.

## 3. Field ownership and loss contract

Stage14A classifies candidate fields by useful new-model training information,
chemical query/review needs, units/meaning, mapping and export/batch cost. Cover
the available useful chemical information rather than a topology-only stub;
justify omissions and document supported loss. Internal persistence details are
not mandatory fields merely because the private builder contains them.

Each chemical fact has one authoritative location in Data. Reassemble the
applicable existing builder fields from it; do not hide a second whole Molecule
or duplicate full record beside editable tensors. Training targets and caller
annotations remain identifiable annotations and do not silently become
chemical facts or a freshly calculated Hotpot result.

| Information | Required transport and reconstruction |
|---|---|
| Atoms and membership | Explicit `num_nodes`, transport-local row mapping, element/isotope and the supported useful factual columns, presence and source meaning. Preserve isolated atoms, empty inputs and original order. Mint destination-local runtime identities; never reuse process addresses/private UIDs as public IDs. |
| Chemical bonds | One authoritative column per chemical bond in transport `edge_index`, preserving bond order, endpoint mapping, kind, direction and metadata. The feature adapter remains responsible for model-directed duplicate arcs. A foreign two-arc encoding needs an explicit agreement/deduplication rule; opposite arcs cannot silently overwrite one another. |
| Molecular declarations | Explicit supplied charge/spin and presence, identifier and registered portable metadata; do not conflate formal-charge sum with molecular net charge. |
| Coordinates | Angstrom Cartesian `pos` with original precision and presence; useful conformer/trajectory information has explicit frame and atom maps, units and selection meaning. Use existing codecs without requiring all private frame bookkeeping. No fabricated zero coordinates. |
| H/aromatic/stereo | Preserve supplied facts, unavailable states, complete stereo records and reference-atom permutations. Transport does not force perception or silently normalize an authoritative aromatic representation. |
| Constraints | Supported bond/angle/torsion constraints useful to training or review retain their units and row maps; no graph reconstruction from distances or private lifecycle-state export requirement. |
| Observations and targets | Include available supported partial-charge/MCA and other useful scientific observations/targets with atom/site/frame/graph joins and required interpretation. Preserve their distinction from supplied facts and caller labels. Restore a current Hotpot attachment only when actual destination dependencies establish that meaning; otherwise keep an attributable annotation/historical observation. Export/import performs no new inference and edited Data cannot make old results current. |
| Crystal data | Stage15 Crystal owns the oriented cell and chemical association; its SpaceGroup owns group/setting/operations. Crystal.symm_units is a read-only derived view. Stage14 transports the one group definition plus independent contents, site relations and periodic bond images through the published codec, without a second group/operation fact store or independent inference. |
| Other Data attributes | Preserve the explicitly supported portable primitive/tensor metadata. Leave caller-only labels outside chemical truth. Unmapped material loss is reported specifically; unknown arbitrary objects are not serialized through unsafe fallback pickle. |

Categorical values, numeric precision, presence masks and mapping tensors must
remain directly useful to PyTorch/PyG; do not collapse everything into an opaque
snapshot or one lossy floating-point table. Stage14A settles exact fields and
dtypes. Deployed-model projections retain their own precision/encoding contract. Exact comparison is appropriate for discrete facts and lossless
unmodified fields; geometry/derived observations use declared tolerances.
Hashes identify artifacts, not chemical correctness.

## 4. Operations, mutation and batch behavior

1. Capture the molecular factual closure once under its existing owner boundary;
   encode independent tensors/primitive metadata after release. Export must not
   expose writable aliases or leave long-lived locks around Torch callbacks.
2. Decode with explicit row/edge mappings. Build one complete candidate and use
   the existing topology/coordinate/result commit path; failure publishes no
   partial molecule. A schema/mapping failure is an actual boundary failure,
   not a reason to synthesize defaults or catch an unknown exception.
3. Import uses current edited transport fields. No hidden pristine payload may
   overwrite an intentional `pos`, charge or connectivity edit. Existing derived
   facts/results become dirty or historical according to actual dependencies.
   Metadata still referring to removed/reordered members needs an explicit map;
   it must not be blindly carried forward.
4. Exercise actual `Batch.from_data_list`, `to_data_list`, DataLoader,
   CPU/device transfer and supported serialization. Explicitly classify every
   field as node-, edge-, graph- or ragged-scoped. PyG's default `index` increment
   and concatenation rules cannot accidentally shift source IDs, symmetry IDs,
   periodic vectors or ragged stereo/constraint references.
5. Prefer ordinary Data. Add one minimal adapter-owned Data subclass only if
   real batching requires explicit `__inc__`/`__cat_dim__`; settle its exact
   field rules in Stage14A. This must remain a genuine Data accepted by standard
   PyG batching, not a project-only loader or a second Molecule class.
6. Importing one graph and splitting a batch retain graph order, atom counts,
   isolated nodes, per-graph charge/cell and independent numeric buffers. A
   legitimate empty graph cannot disappear. No conversion-driven model call.

## 5. Implementation sequence and ownership

| Slice | Work and exclusive responsibility | Exit evidence / next release |
|---|---|---|
| 14A — contract | Inventory live feature consumers and candidate chemical/observation fields; justify training/review utility and cost, distinguish deployed-model ownership, and settle exact signatures, Data fields, dtype/units, batching rules, external maps and loss diagnostics. Inspect current package versions with small real batching probes. | Root records concrete public/schema decision under main-plan §16.4 and updates the interface/verification records; no production API is published beforehand. |
| 14B — Molecule transport | Adapter encoders/decoders in `adapters/pyg.py` or one focused child module; domain orchestration in `operations/conversion.py`; thin carrier methods only. Reuse builders, stereo/result/frame codecs. | CP311 source + outside-checkout installed actual round trips, edited-Data imports and local-failure/atomicity checks. |
| 14C — batch and existing consumers | Correct actual PyG field batching, general collection/dataset exports and portable serialization; preserve separate model-owned, demand-driven feature interfaces. | Real multi-graph batch/unbatch; actual CBond/MCA feature/inference checks where shared code or chemistry is affected. Source ownership and buffer isolation hold. |
| 14D — Crystal transport | Use the Stage15A settled schema for preparation; execute round trips only after the implemented Stage15C domain publication (15B/C tests and 15E migration closure for any touched external format). Include all required image/origin relations. | Crystal ↔ Data ↔ Crystal and mixed molecular/crystal batches preserve the declared geometry/periodicity; no duplicated crystal implementation. |
| 14E — delivery | Current interface documentation/examples and field/loss table, affected package support, representative complete-work timings/resources and report/commit. | All applicable transport gates close; source/installed distinctions, losses and unverified external schemas remain explicit. |

Candidate files are `structure/molecule.py`, `structure/crystal.py`,
`operations/conversion.py`, `adapters/pyg.py`, existing feature consumers only
where actually affected, and focused transport tests/docs. Assign exact files
exclusively per writer at entry. Other active agents share the workspace and
must not revert one another. No rule algorithm, native ABI, trained model or
MetalCluster implementation change is authorized by this stage plan.

## 6. Required business checks

| Group | Cases and assertion |
|---|---|
| Structural facts | Empty/isolated/disconnected/charged/isotopic molecules; absent versus explicit zero; unknown versus declared bond kind; dative direction and atom/bond order. Whole factual round trip uses the original independent source fixtures as well as equality checks. |
| Chemical semantics | Aromatic/Kekulé, explicit/materialized/implicit H and stereochemical reference permutations; preserve declared conventions without using feature extraction as the sole oracle. |
| Geometry and lifecycle | Missing coordinates, multiple source/conformer frames, frame selection, constraints, copy/pickle and mutation after export/import. Stale attached results must not become current. |
| Real PyG | Heterogeneous sizes, no edges, graph-level scalars, batch/unbatch, permitted reordering, CPU/device movement, independent buffers and saved/loaded Data. Foreign declared schemas, one/two directed arcs and actual mapping errors. |
| New-model training and review | Use exported tensors in an actual small PyTorch/PyG training-data path, with an explicit target and atom/edge/graph mapping; import supported native/edited Data and query the resulting chemical objects. Information must be accessible without decoding a hidden full-object snapshot. No new production model/retraining is required. |
| Deployed models | Check actual model-owned feature/inference paths only where shared code or chemistry is affected, including CBond/MCA atom/site joins. They need no route through general `to_pyg()`. No trained-feature redesign or retraining. |
| Periodic structures | After Stage15: skew lattice, nontrivial space group, special-position relations, boundary-crossing and periodic edges, generated-cell/supercell origins. |
| Cost/support | Complete export→batch→unbatch→import work and bounded memory at representative sizes; CP311 first, then affected declared package versions. Do not compare a richer transport to a feature-only baseline as equal work. |

Document and test exact diagnostics for an unmapped required field, inconsistent
edge identity, lost periodic image or irreconcilable stereo/member map. A useful
partial external projection may return only under its declared map with precise
loss findings. Do not add blanket NaN/Inf checks, a defensive framework or
hypothetical fallback routes.

## 7. Dependencies and completion

Planning/probes may proceed without changing frozen mainline product bytes.
Production authoring starts from the validated Stage0–12 `B_exp`; Stage13's
private rule experiment is not an unrelated dependency. Stage14B/C and Stage15
may use separate bounded branches. Stage14D can prepare against the Stage15A schema, but real Crystal round trips
wait for the Stage15C implemented domain publication and affected 15E format
migration. Stage15 publishes and closes its domain without waiting for Stage14D,
so there is no cycle. The shared Crystal↔Data integration fixture belongs to
Stage14D/overall-task integration, not Stage15's standalone exit gate.

Stage14 cannot close after molecule-only export: both Molecule and Crystal
bidirectional scopes, useful general training information, actual batching,
chemical query/review, mapping/currentness, documentation and applicable
installed delivery must be demonstrated. Complete internal object-state cloning
is not an acceptance target. Further foreign schemas
may remain explicit independent integration limitations, without pretending
that unknown facts were recovered. Overall current-task completion also requires
the separately recorded Stage15 scope; neither side branch changes `B_exp`.

## Primary technical references

Read the current installed-version behavior before implementation. These sources
support boundary design, not a claim that Hotpot has implemented it:

- [PyG Data contract](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html)
- [PyG batching and increment/concatenation behavior](https://pytorch-geometric.readthedocs.io/en/latest/advanced/batching.html)
- [Stage15 crystal contract and scientific boundaries](../stage15/review.md)
- [Effective goal](../../goal.md), [chemical-type interface protocol](../../chemical_type_interface_contract.md), [main plan](../../fact_rule_architecture_implementation_plan.md)
