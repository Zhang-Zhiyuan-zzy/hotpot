# Stage14A — accepted molecular PyG transport contract

Root decision: 2026-11-05, under main-plan §16.4 and the owner's full-authority
mandate. Entry is validated `B_exp=5a7b19396328eee964a8f3c65bc3345c013eaec6`.
This branch starts there independently of the completed private Stage13 experiment
(`D_exp=5dc0443de17439f0c4d74151e2427a3a834c899c`,
`I_exp=cc9470e19f2fac786138d5af0986d726d5e793c7`, both pushed).
Stage14D remains gated on Stage15's implemented domain; this decision publishes
no new Crystal domain or chemistry algorithm.

## API and ownership

```python
Molecule.to_pyg(self) -> Data
Molecule.from_pyg(cls, data, *, field_map=None) -> Molecule  # classmethod
MolBundle.to_pyg_dataset(self, dataset_root) -> None
to_pyg_dataset(list_mol, dataset_root) -> None
```

Remove `Molecule.to_pyg_data` and the general collection `prefix` argument
atomically with their actual callers/tests. No alias, legacy output or legacy
usage documentation. Keep separate specialized training helpers and unchanged
CBond/MCA model encoders; they are not general transport consumers.

Carrier methods remain thin. `operations/conversion.py` owns one locked capture
and fresh reconstruction; `adapters/pyg.py` (or its one focused private child)
owns Torch/PyG encoding and decoding. Core import does not import Torch/PyG.
Reuse the current complete builder, stereo and source-frame codecs. Data must
not contain a second complete MoleculeRecord, Atom table, original Molecule,
locks, private identities, dependency engine or hidden pristine restoration path.
No runtime registry, new exception class, result envelope or confidence label.

## Molecular schema `hotpot.molecule/1`

Use ordinary `torch_geometric.data.Data`. `hotpot_schema` is a one-element list
containing the schema string. Node and edge tensors are authoritative for the
facts they represent; metadata never overrides edited tensors. All buffers are
independent of the source. Integers/bools are never encoded in float features.
Missing values have false presence; their numerical storage is not a fact.

| Field | dtype / shape | Meaning |
|---|---|---|
| `num_nodes` | int | Explicit N, including empty and isolated nodes. |
| `atom_key` | int64 / N | Unique graph-local transport keys, initially 0..N-1; not source UID/id. |
| `z` | int64 / N | Atomic number. |
| `formal_charge` | int64 / N | Atom formal-charge declaration, distinct from graph charge. |
| `total_h` | int64 / N | Existing site-total H target T; includes graph H for ordinary atoms. |
| `aromatic` | bool / N | Raw atom aromatic representation. |
| `atom_id` | int64 / N | Existing public source/display id. |
| `pos` | float64 / N×3 | Cartesian angstrom coordinates; no fabricated coordinate presence. |
| `atom_present` | bool / N×8 | z, formal_charge, total_h, aromatic, atom_id, x, y, z-coordinate. |
| `atom_source` | uint8 / N×8 | Same column order; existing finite ATOM_SOURCE_CODES, documented by name. |
| `representation_available` | bool / N×2 | T/aromatic availability at capture. Inferred values require CURRENT and no pending upstream invalidation; a conservative false is permitted. Raw supplied declarations remain present regardless of rule usability. |
| `edge_index` | int64 / 2×M | One ordered column per complete chemical Bond; endpoint orientation is meaningful. |
| `bond_key` | int64 / M | Unique local keys, initially 0..M-1. |
| `bond_order` | float64 / M | Existing numeric order, including supported fractional/unknown-kind representations. |
| `bond_kind` | int64 / M | unknown=0, zero=1, single=2, double=3, triple=4, aromatic=5, dative=6. |
| `bond_direction` | int8 / M | 0 absent, +1 atom1→atom2, -1 atom2→atom1. Other existing direction spellings remain in structured bond metadata without loss. |
| `bond_aromatic` | bool / M | Independent representation flag; alternating kinds do not reconstruct it. |
| `bond_id` | int64 / M | Existing public id. |
| `bond_constraint` | bool / M | Existing bond constraint flag. |
| `atom_constraint` | bool / N×3 | Existing x/y/z constraint flags. |
| `angle_index`, `torsion_index` | int64 / 3×A, 4×T | Constrained member tuples, not target angle/torsion values. |
| `wiberg_index`, `wiberg_value` | int64 / 2×P, float64 / P | Sparse supplied pair values; no quadratic all-pair generation. |
| `net_charge`, `spin_mult` | float64 / 1 each | Existing graph declaration values, interpreted only when corresponding presence is true. Preserve accepted scalar values without new inference. |
| `net_charge_present`, `spin_mult_present` | bool / 1 each | Declaration presence, including explicit zero. |
| `cell_parameters`, `cell_present` | float64 / 1×6, bool / 1 | Existing six-scalar attached cell only; no new Stage15 inference. |

Use one-element list wrappers around graph-ragged structures so ordinary PyG
batching does not concatenate their graph-local references. Fixed `*_index`
tensors use PyG's ordinary node increment; `atom_key`, `bond_key` and wrapped
references do not. `hotpot_graph=[{...}]` carries declaration sources,
identifier presence/value and portable caller annotations. `hotpot_bonds=[...]`
carries ordered source namespaces, exact other direction values and portable
metadata joined by bond_key. `hotpot_stereo=[...]` is a **structured primitive**
encoding of the sole existing StereoPayload schema, with atom references as
transport keys, preserving record/occurrence order, duplicates, all anchors and
carriers, groups, isotope/radicals, current I/J/no-implicit H, immutable source H
and arbitrary signed Python integers. T is not copied into I or the source
ledger. No duplicate isotope/radical authority is added as a narrowed tensor.
Fractional-valence annotations are a sparse wrapped atom_key/value structure.

`hotpot_frames=[...]` retains source coordinates and subject keys, source versus
generated grouping, provider ordinals, available eight-slot observations,
selectors, units, methods, parameters and portable provenance in structured
primitive/tensor form. Actual removed subjects stay in an explicitly historical
namespace with unavailable current join. Never leak private source tokens or
claim an old UID is a current transport key. Source frames may be reconstructed
through their existing codec; import selects neither old observations nor a
generated ensemble and does not replace edited main pos with an old frame.

`hotpot_observations=[...]` retains selected or stale partial-charge and MCA
values with original subject namespaces, actual joins where recoverable,
method/version/parameters/units and finite invalidation markers. MCA's original
rows join current atoms only when the retained/current mca_representation digest
matches under the same owner boundary; counts, element strings and SMILES are
not substitutes. Partial-charge UIDs map only extant subjects; preserve removed
subjects as historical. No blanket finite-value check or model call is added.

## Import, editing and historical observations

Import reads current tensor rows and edges, validates their actual schema/maps,
and constructs one candidate via the sole builder. Mint fresh destination
identities. The existing builder marks destination rule dependencies; export
never runs a rule/getter to populate absent H/aromatic information. Raw inferred
values and original source remain attributable but do not bypass destination
preparation. Independently edited T and I/J declarations can conflict; preserve
supplied facts and let the existing named chemistry/adapter rules diagnose them.

Computed observations are always **historical annotations** on this generic
import, even when tensors happen to be unchanged. Store their structured value
and original joins under reserved `properties['pyg_observations']`; do not install
current MCA, selected frame or partial-charge attachments. Clear raw partial-
charge presence bits before the builder so it cannot fabricate a current supplied
charge result. Truly foreign supplied charges are also explicitly historical
annotations under this generic map, not a calculation or assignment operation.
Source-frame observations may remain historical source-frame records with no
selected result. Generated coordinates remain useful historical data.

Historical annotations retain their original transport subject namespace and
initial import order. They do not promise a live join after topology edits,
copy/components/merge or re-export. Re-export retains that history separately
and never relabels it as the new graph's current keys. Current source attachments
captured during export have their actual current joins; historical namespaces
remain independently interpretable. No new mutation-aware annotation engine.
Caller labels remain annotations; they never become chemical facts.

Reordering nodes means reordering node fields/keys and updating row-index edge,
constraint and pair tensors. Wrapped references follow atom_key/bond_key, not
row position. Missing referenced members yield a named local mapping failure,
not the existing remapper's silent drop. Edge edits cannot leave stereo anchors
on nonexistent Bonds. Other supported fact edits are imported as provided.

## Foreign field map

Foreign Data requires an explicit mapping unless it carries this exact schema.
The map is a plain dict with `fields`, `edge_mode` and `coordinate_unit`:

```python
field_map = {
    'fields': {'z': ('x', 0), 'pos': 'pos', 'edge_index': 'edge_index',
               'bond_order': ('edge_attr', 0)},
    'edge_mode': 'single',  # or 'paired'
    'coordinate_unit': 'angstrom',  # or 'nanometer', 'bohr'
}
```

A field selector is an attribute name or (attribute, integer column). Supported
names are the fixed fields above plus explicit portable `annotations`; z is
required and edge_index is required when edges exist. N is explicit num_nodes
or the mapped z length. Unmapped facts are absent, never guessed from x. Units
are required if pos is mapped; conversion constants are documented. A mapped
bond_order without bond_kind uses the explicit numeric map 0/1/1.5/2/3 to
zero/single/aromatic/double/triple; other orders retain unknown kind. Missing
both yields unknown kind and no invented chemical claim.

`paired` requires exactly one opposite arc per chemical edge, exact agreement
of nondirectional mapped fields and inverse direction signs; preserve first
encounter order. Reject self/parallel/inconsistent or incomplete paired mappings
that cannot be represented by the current Molecule Bond domain. `single` does
not deduplicate. Reject a multi-graph Batch as a single molecule: callers first
use standard to_data_list. Explicitly mapped annotation targets remain labels.
Unmapped foreign attributes produce precise `pyg_unmapped_field` RuntimeWarning
with attribute paths; they do not silently become chemistry.

Portable annotations/metadata use the existing closed value-codec domain
(primitives, lists/tuples/dicts and ordinary non-object numeric arrays encoded
as independent tensors where useful). Unsupported actual values fail locally
with `pyg_unsupported_metadata` and their exact path; no unsafe fallback pickle,
catch-all or speculative recovery. Successful supported conversion is silent.
Other local ValueError/TypeError messages name `pyg_schema`, `pyg_mapping` or
`pyg_stereo_reference` and the concrete field. No partial object is published.

## Evidence, cost and exit

Actual ordinary-Data probe: Torch2.7.1+cpu/PyG2.6.1, 12/12 checks including
empty graph, ragged wrapper, large Python integers, DataLoader, transfer and
trusted-local save/load; this is not converter evidence. Feasibility review is
read-only, not a gate or scientific approval. Roots:
`/tmp/hotpot_plan20/medium/stage14a-{contract-probes,feasibility-review}-20261105/`.

14B/C requires CP311 source and outside-checkout installed independent factual
fixtures, edited/reordered imports, honest missing/stale history, constraints,
full stereo/H meanings, empty/disconnected/dative/fractional cases, actual
Batch/DataLoader/device/save-load, buffer independence and a small real training
path. Preserve separate CBond/MCA feature tests if affected. Measure representative
complete export→batch→unbatch→import work and resources, including failed attempts,
without claiming equality to the old feature-only output. CPU support is tested
first; real GPU and remaining affected supported versions remain explicit gates.
No fourth attempt on the same bounded failing issue; record dependency/impact
and advance independent work. Root owns code acceptance, commits and pushes.
