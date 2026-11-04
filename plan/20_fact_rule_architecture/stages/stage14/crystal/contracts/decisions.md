# Stage14D — oriented Crystal/Data transport

Root design decision2026-11-05 under§16.4; sourcec0eb6c4 plus the recorded CUDA
observations. Stage15A–F domain and15E format closure are accepted and pushed4407db0.
The preparation-only gate is now satisfied; actual shared Crystal/Data
roundtrips use that implemented sole record and published portable/stereo codecs. Reuse retained full-plan/companion
alignment and current effectivegoal/governance. No scope or chemistry change.

## One current schema and thin facades

Use hotpot.molecule/2 for the coherent molecular/crystal Data hard cut. Remove
the earlier1x6cell_parameters projection; provide no old schema loader/alias.
Existing molecular fields,ordinary Data,source facts,absence,methods,historical
observations and deployed-model independence remain unchanged. Current source
records use Stage15's sole oriented cell codec. Crystal.to_pyg()/from_pyg(...,
field_map=None) reuse the same molecular capture/builder and one cell adapter;
no second Crystal graph,group algebra,codec or fact owner. A standalone Crystal
has zero nodes and an explicit contained-Molecule presence bit in wrapped graph
metadata; an empty attached Molecule remains distinct. Crystal.from_pyg requires
an actual cell; it never creates P1 or a cell from positions.

| Current Data field | dtype/shape | Authority/meaning |
|---|---|---|
| cell_matrix,cell_present | float64/1x3x3,bool/1 | Oriented row basis and actual attached-cell presence. |
| bond_image,bond_image_present | int64/Mx3,bool/M | Actual directed edge image; unknown differs from explicit zero. |
| occupancy,occupancy_present | float64/N,bool/N | Current site occupancy; absent differs from explicit zero. |
| space_group_R,space_group_t | int64/Gx3x3,float64/Gx3 | Useful derived numeric training projections of the exact current group. |
| hotpot_crystal | one wrapped mapping orNone | Exact group/setting,relations,disorder,representation and historical source binding/maps. |

cell_matrix,images and occupancy occur only in their authoritative tensors;
remove them from the wrapped portable record. Group exact rational xyz/setting
is authoritative for group meaning; numeric group views must agree on import.
No approximation of arbitrary float translations to guessed rational groups.
Unknown group has zero operations,not an identity/P1 fabrication. Full original
and historical operations/carries remain exact through the shared codec.

Wrapped current_atom_keys/current_bond_keys join row-aligned disorder and
origin-map sequences to editable tensor order. Remap current equivalent-image
references through atom_key; historical source indices stay in their historical
frame. Metadata never overrides edited tensor facts. Missing/duplicate/stale
joins or an invalid historical basis relation reject locally; callers may make
explicit supported context edits,including marking a historical basis mapNone.
Edge images follow current edge_index orientation; a user reversing an edge
must reverse its directed image consistently. New/deleted context members need
complete updated mappings. Queries after import use current owners,not a hidden
pristine source. All tensor/storage buffers are independent.

## Foreign mapping and edits

The existing explicit field_map remains the foreign-data boundary. Replace its
cell_parameters selection with cell_matrix and add bond_image/presence and
occupancy/presence. An optional fields.crystal_context selects an inspectable
complete shared portable-cell record from the foreign Data. If selected,it
supplies exact group/setting/relations/origins; explicit selected numeric fields
must agree for their duplicated projection. Otherwise a supplied matrix creates
only an independent unknown-group context with the actually supplied images/
occupancies. Missing images stayNone. No connectivity,periodicity,symmetry,
H,formal-charge or model-feature inference. Contradictions reject rather than
silently pick an authority. Foreign unmapped material remains attributable.

## Evidence and gate

Use actual implemented15B/C/D and affected15E closure. Cover oriented/skew/
changed-origin/centred/screw structures,independent versus generatedP1/history,
explicit equivalence/disorder/occupancy,unknown versus zero images,finite and
periodic components,standalone/empty values,editable node/edge order and facts,
precise broken maps and no source mutation. Mixed molecule/crystal DataLoader
and ordinary Batch,CPU and actual CUDA,unbatch,fresh reconstruction and trusted
save/load must preserve current tensor and exact context meanings. Check the
same source/installed CP311 and affected delivery boundary. Keep broader ABI/
dependency claims honest. Shared model encoders remain unchanged and do not
route through this interface. Root reviews code/business evidence and owns
acceptance/integration/Git; neither branch alone closes overallPlan20.
