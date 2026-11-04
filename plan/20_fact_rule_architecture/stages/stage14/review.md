# Stage14 implementation plan — Molecule and Crystal ↔ PyG Data

Status: owner-directed scope registered on 2026-11-04; concrete implementation
plan prepared, runtime authoring not started. This is additional current-task
scope. Stage0–12 acceptance and `B_exp`, and Stage13's `D_exp`/`I_exp`, retain
their existing meanings. The main agent owns entry decisions and acceptance.

## 1. Required outcome and present boundary

Provide reversible, editable structural transport between `hp.Molecule` and
`torch_geometric.data.Data`, then between Stage15's `hp.Crystal` and Data. Minimize
information loss within an explicit field contract. A native third-party Data
object can supply only information it actually contains; conversion cannot
recover an unknown bond kind, missing stereochemistry or absent lattice image.

At inspected source `ec4d61889a7262df765da29082e405b3c612cd15`:

| Current owner | Observed behavior | Stage14 consequence |
|---|---|---|
| `structure/molecule.py::to_pyg_data` | Delegates to `plugins/PyG/data/utils.py::mol_to_pyg_data` | Replace this public output directly with reversible transport under the owner's explicit instruction. |
| `plugins/PyG/data/utils.py` | Atom float32 features, directed bond arcs, pair/ring tensors and optional batch | Keep necessary model feature mathematics at the actual model-input owner; migrate its callers and never infer original facts from rounded features. |
| `operations/collections.py`, `structure/bundle.py` | Ordered dataset export, ordinal filenames for unnamed members | Export the new transport Data and migrate actual dataset readers; retain only the collection ordering/naming behavior still required by the new contract. |
| `structure/_builder.py::MoleculeRecord` | Factual values, presence/source, bonds, stereo, frames, constraints and partial-charge record | Reuse its codecs and atomic reconstruction; do not add another molecular fact owner. |
| MCA/result owners | MCA attachment is not a field of that complete builder record | Explicitly inspect and transport the actual typed attachment; do not assume one snapshot already includes every scientific result. |
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

Scope is structural transport, not a feature redesign, model retraining, generic
object serializer, arbitrary Python-object pickle, or a new chemical ontology.
Existing unknown facts and explicit zero remain distinct. Conversion does not
run aromaticity, hydrogen, charge, geometry or model calculations to fill gaps.

## 2. Interfaces and separation of purposes

The proposed public spelling to settle in Stage14A is:

```python
Molecule.to_pyg(self) -> Data
Molecule.from_pyg(cls, data, *, field_map=None) -> Molecule
# Added only after the Stage15 aggregate exists:
Crystal.to_pyg(self) -> Data
Crystal.from_pyg(cls, data, *, field_map=None) -> Crystal
```

The owner explicitly selects `to_pyg()` and `from_pyg()` and authorizes replacing
`to_pyg_data()` directly. The new interface has one reversible-transport output;
remove `to_pyg_data()` and its obsolete arguments/assumptions in the connected
cutover, migrating actual callers to `to_pyg()`. Add no compatibility switch, second public feature
export, legacy-output path or legacy-use documentation. Migrate actual model
and dataset callers rather than retaining the public projection for them.

CBond/MCA feature preparation stays at the actual model-input owner with its
existing trained widths, feature order, H semantics, dtype and padding. Reuse
that one calculation where needed; public Data transport does not redefine a
trained model's mathematics. Retire a Data-wrapper helper if no real consumer
remains after this cutover.

A Hotpot-authored transport Data has one documented schema marker and the
necessary field/units contract. This is a transport schema, not a success or
confidence envelope. A foreign Data needs a selected documented `field_map`
unless it already carries that schema. Specify exact mapping arguments and
units at Stage14A; do not introduce general plugin discovery or guess that the
first `x` column is atomic number. Ordinary returns are Data or the requested
chemical object; diagnostics use the existing finding/error channels.

## 3. Field ownership and loss contract

Each transported fact has one authoritative location in Data. Reassemble the
existing builder record from those fields; do not hide a second whole Molecule
or duplicate full record beside editable tensors.

| Information | Required transport and reconstruction |
|---|---|
| Atoms and membership | Explicit `num_nodes`, stable transport-local row keys, element/isotope and all supported factual columns, presence and source meaning. Preserve isolated atoms, empty inputs and original order. Mint destination-local runtime identities; never reuse process addresses/private UIDs as public IDs. |
| Chemical bonds | One authoritative column per chemical bond in transport `edge_index`, preserving bond order, endpoint mapping, kind, direction and metadata. The feature adapter remains responsible for model-directed duplicate arcs. A foreign two-arc encoding needs an explicit agreement/deduplication rule; opposite arcs cannot silently overwrite one another. |
| Molecular declarations | Explicit supplied charge/spin and presence, identifier and registered portable metadata; do not conflate formal-charge sum with molecular net charge. |
| Coordinates | Angstrom Cartesian `pos` with original float64 precision and presence; optional source/generated frames and selected-frame identity use existing codecs and atom maps. No fabricated zero coordinates. |
| H/aromatic/stereo | Preserve supplied facts, unavailable states, complete stereo records and reference-atom permutations. Transport does not force perception or silently normalize an authoritative aromatic representation. |
| Constraints | Existing complete bond/angle/torsion constraints, units, row maps and lifecycle semantics; no graph reconstruction from distances. |
| Selected results | Preserve actual partial-charge/MCA values, atom/site joins, method identities and existing dependency bindings. Rebind only when destination facts establish the same dependency meaning. Edited Data cannot make old results current. Historical observations remain historical; unavailable source bindings are not replaced with a fresh success claim. |
| Crystal data | Stage15 owns lattice, space group/setting, independent contents, site relations and periodic bond images. Stage14 transports that contract without independently inferring or owning it. |
| Other Data attributes | Preserve the explicitly supported portable primitive/tensor metadata. Leave caller-only labels outside chemical truth. Unmapped material loss is reported specifically; unknown arbitrary objects are not serialized through unsafe fallback pickle. |

Tensor precision belongs to the transport contract. Model float32 projections
remain separate. Exact comparison is appropriate for discrete facts and lossless
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
| 14A — contract | Inventory live feature consumers and every builder/result field; settle exact new signatures, Data fields, dtype/units, batching rules, external maps and loss diagnostics. Inspect current package versions with small real batching probes. | Root records concrete public/schema decision under main-plan §16.4 and updates the interface/verification records; no production API is published beforehand. |
| 14B — Molecule transport | Adapter encoders/decoders in `adapters/pyg.py` or one focused child module; domain orchestration in `operations/conversion.py`; thin carrier methods only. Reuse builders, stereo/result/frame codecs. | CP311 source + outside-checkout installed actual round trips, edited-Data imports and local-failure/atomicity checks. |
| 14C — batch and existing consumers | Correct actual PyG field batching, collection exports and portable serialization; keep model-feature extraction singular. | Real multi-graph batch/unbatch and actual CBond/MCA feature/inference checks; source ownership and buffer isolation hold. |
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
| Actual models | Compare original and new code's existing model features and real CPU CBond/MCA atom/site joins on representative molecules. New transport adds no algorithm or retraining. |
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
bidirectional scopes, actual batching, mapping/currentness, documentation and
applicable installed delivery must be demonstrated. Further foreign schemas
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
