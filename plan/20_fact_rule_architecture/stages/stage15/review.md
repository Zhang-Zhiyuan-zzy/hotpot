# Stage15 implementation plan — Crystal spatial symmetry and molecular expansion

Status: owner-directed planning revision, 2026-11-04; concrete Stage15A choices
accepted 2026-11-05 in [the domain contract](domain/contracts/decisions.md).
Standalone exact symmetry and connected cell/lifecycle are delivered at the
checkpoints below; actions, findings and format closure remain open. Crystal's primary
job is to apply crystallographic spatial symmetry
and copy/expand molecular units, with useful geometry/topology observations.
The following surface is a concrete **proposal for Stage15A**, not a released
API or scientific approval. The parent owns acceptance, main-plan registration,
interface/schema decisions and runtime entry under §16.4.

## Current implementation checkpoint

Root accepted the standalone exact domain on2026-11-05; see
[the report](domain/validation/exact_symmetry_report.md). Connected cell/lifecycle
15B is accepted at [559 source/installed CP311 checks](domain/validation/connected_cell_report.md).
Actions15C, observations15D, formats15E and delivery15F remain open. The accepted
[15A decision](domain/contracts/decisions.md) resolves the proposal choices below.

## 1. Complete proposed public tree — before implementation

This is one coherent surface. `Crystal`, immutable `SpaceGroup`, and the abstract
`SymmUnit` are the proposed public types. `Array`, `Int3`, `Q3`, `Setting`,
`ImageKey`, `Candidate` and `Finding` below denote value layouts, not additional
exported classes. Stage14 owns its selected `to_pyg`/`from_pyg` names/encoding.
The owner's latest directions fix fresh-placement `operate`, group magic
methods, the single SpaceGroup owner and the complete 230-type catalogue;
the exact new signatures/exports below remain concrete §16.4 proposals.

```text
Crystal
├── construction
│   ├── Crystal(a, b, c, alpha, beta, gamma, *, molecule=None,
│   │           space_group: SpaceGroup | None=None,
│   │           site_relations=None, bond_images=None)
│   └── from_matrix(matrix, *, molecule=None, space_group=None,
│                   site_relations=None, bond_images=None) -> Crystal
├── contained chemistry and active cell facts
│   ├── molecule -> Molecule | None                         [read-only association]
│   ├── a, b, c -> float                                    [Angstrom; legacy setters]
│   ├── alpha, beta, gamma -> float                         [degrees; legacy setters]
│   ├── matrix -> independent float64 Array(3,3)             [read-only; basis rows]
│   ├── volume -> float                                    [Angstrom^3; derived]
│   ├── space_group -> SpaceGroup | None                    [read-only association]
│   ├── symm_units -> tuple[SymmUnit, ...] | None            [derived: space_group.symm_units]
│   ├── site_relations -> immutable mapping                 [actual image/site relations]
│   ├── bond_images -> tuple[Int3 | None, ...]              [complete Bond order]
│   ├── representation -> "independent" | "cell" | "supercell"
│   └── image_map -> tuple[tuple[ImageKey, ...], ...] | None [all origins per output row]
├── complete, explicit changes
│   ├── set_cell(matrix, *, preserve="cartesian") -> None
│   ├── set_space_group(space_group: SpaceGroup | None,
│   │                   *, site_relations=None) -> None
│   └── set_site_relations(relations) -> None
├── coordinate and molecular actions
│   ├── to_fractional(cartesian) -> independent float64 Array(...,3)
│   ├── to_cartesian(fractional) -> independent float64 Array(...,3)
│   ├── operate(operation: SymmUnit) -> Crystal              [one fresh transformed placement]
│   ├── unit_cell() -> Crystal                              [fresh periodic cell content]
│   ├── unit_molecules() -> tuple[Molecule, ...]             [fresh finite unwrapped units]
│   ├── expand(repeats: Int3) -> Crystal                     [e.g. (2,3,2)]
│   └── copy() -> Crystal                                   [independent mutable facts]
├── requested relations and observations
│   ├── same_molecule_candidates(*, tolerance) -> tuple[Candidate, ...]
│   ├── molecular_symmetry(molecule=None, *, tolerance) -> tuple[SymmUnit, ...]
│   ├── linking_number(first_cycle, second_cycle, *, tolerance) -> int | None
│   └── geometry_findings(*, coincidence_tolerance, close_contact_cutoff,
│                        ring_scope="full_graph", max_ring_size=None)
│                        -> tuple[Finding, ...]
└── Stage14D consumer additions, after the domain is published
    ├── to_pyg() -> Data
    └── from_pyg(data, *, field_map=None) -> Crystal          [classmethod]

SpaceGroup                                            [immutable group/setting value]
├── construction
│   ├── SpaceGroup(number: int)                            [selected reference, 1..230]
│   └── from_operations(operators, *, reference_number=None,
│                       to_reference=None) -> SpaceGroup    [complete exact cell representatives]
├── crystallographic facts and read-only derived views
│   ├── number -> int | None                               [ordinary type, when supplied/verified]
│   ├── hermann_mauguin -> str | None                       [current-setting label if tabulated]
│   ├── hall -> str | None                                 [current-setting label if tabulated]
│   ├── setting -> Setting | None                          [(reference_number,P,p), f_ref=P f+p]
│   ├── crystal_system -> str | None                       [when known from the declared type]
│   ├── point_group -> str | None                          [named type, when known]
│   ├── symm_units -> tuple[SymmUnit, ...]                   [ALL conventional/current-cell reps]
│   ├── centering_translations -> tuple[Q3, ...]            [all pure-translation reps, incl. zero]
│   ├── translation_basis -> exact rational 3x3 tuple      [primitive basis columns]
│   ├── point_rotations -> tuple[exact integer 3x3, ...]    [distinct R; quotient by full T]
│   ├── cell_order -> int                                  [finite |G/T_cell|]
│   └── point_group_order -> int                           [finite |G/T|; G itself is infinite]
├── group membership and exact lifts
│   ├── identity() -> SymmUnit
│   ├── lift(index: int, image: Int3=(0,0,0)) -> SymmUnit   [T_image r_index]
│   ├── contains(operation: SymmUnit) -> bool
│   └── __contains__(operation: SymmUnit) -> bool           [g in space_group]
├── explicit coordinate-setting change
│   └── change_setting(P, p) -> SpaceGroup                 [fresh; f_old=P f_new+p]
└── __eq__ / __hash__                                     [exact immutable context value]

SymmUnit                                              [ABC; immutable group element]
├── construction returning the one concrete affine implementation
│   ├── from_affine(R, t, *, context: SpaceGroup) -> SymmUnit
│   ├── from_xyz(symbol: str, *, context: SpaceGroup) -> SymmUnit
│   └── identity(*, context: SpaceGroup) -> SymmUnit
├── R -> exact integer 3x3 tuple                           [read-only; preserves cell Z^3]
├── t -> exact rational length-3 tuple                     [read-only; full, never wrapped away]
├── context -> SpaceGroup                                 [same immutable group/setting owner]
├── xyz -> str                                            [full exact reconstructible operation]
├── kind -> identity / translation / rotation / screw /
│           reflection / glide / inversion / rotoinversion [derived classification]
├── operate(fractional) -> independent float64 Array(...,3) [no wrapping, no mutation]
├── compose(other: SymmUnit) -> SymmUnit                    [self AFTER other]
├── __matmul__(other: SymmUnit) -> SymmUnit                 [g @ h delegates compose]
├── inverse() -> SymmUnit
├── __invert__() -> SymmUnit                               [~g delegates inverse]
├── power(n: int) -> SymmUnit                              [negative, zero, positive]
├── __pow__(n: int) -> SymmUnit                            [g ** n delegates power]
├── __eq__ / __hash__                                     [exact full R,t AND group/setting]
└── reduce_to_cell() -> (SymmUnit, Int3)                    [representative AND full carry]

Molecule association boundary
└── create_crystal(a, b, c, alpha, beta, gamma, *, space_group=None,
                   site_relations=None, bond_images=None) -> Crystal
    └── delegates to the same complete association/cell operation
```

All new spelling, keywords, properties, return layouts, permissions and exports
need the parent's concrete decision before dependent code. `SpaceGroup` and
`SymmUnit` are proposed at `hotpot.cheminfo.structure.symmetry`; top-level
`hotpot.SpaceGroup` and `hotpot.SymmUnit` exports are separate explicit decisions.
Existing `hotpot.Crystal` keeps its ordinary name and six positional arguments.
The former **unreleased proposal** for Crystal-level `setting`, `operators`,
`symmetry_context`, string-valued `space_group` and `set_symmetry` is replaced by
this single-owner design. There is no production compatibility cutover for those
unimplemented names, no `mol` alias, generic registry or result envelope.

| Value / entry | Exact proposed meaning and ownership |
|---|---|
| `SpaceGroup` | One immutable canonical record of the complete current-cell exact `(R,t mod Z³)` representatives and its exact setting binding; derived labels/lattice/views read that record. No mutable external crystal object is retained. |
| `operators` in `from_operations` | Complete exact `(R,t)` data normalized to one half-open cell list, including centring, identity and inverses. Explicit integer/fraction inputs, not arbitrary floats guessed into rationals. A generator-only list is insufficient. Validate full lifted closure before constructing the group. |
| `Setting` / `to_reference` | `Setting=(reference_number,P,p)` with exact rational `P,p`, `f_reference=P f_current+p`. The numbered constructor selects the catalogue's `(I,0)` reference setting. `from_operations` supplies both a declared reference number and its exact map, or neither; an anonymous complete affine group remains usable without an invented type/name. Supplied type/map must agree with the reference full group. |
| Names / context equality | H–M and Hall labels describe the **actual current setting** when tabulated; otherwise they are `None`, while the exact operations/setting remain available. Never show a reference Hall label as an unverified alternate-setting label. Context equality/hash uses the exact setting binding and canonical representative table; aliases/display spelling do not create a second group. |
| Lattice views | `centering_translations` is the pure-translation portion of that same table. `T=⋃_c(Z³+c)`; `translation_basis` is the deterministic rational primitive column basis derived by column Hermite normal form of denominator-cleared translation generators. `T_cell=Z³` is the chosen cell-image sublattice, not a second numeric cell. No independent lattice setter or mutable lattice object. |
| `SymmUnit.context` | The immutable SpaceGroup itself, with no physical lengths/orientation, molecule or second active-cell record. Full element `(R,t)` is a verified member `T_n r_i`. Same-context algebra is closed; a different context needs an explicit conversion/rebind and membership check. |
| `Crystal.symm_units` | Exactly `None if space_group is None else space_group.symm_units`, read-only and derived. No `_symm_units`, setting/operator/context shadow fact, setter or duplicate mutable list on Crystal. Explicit P1 has the identity tuple; unknown group is not P1. |
| `site_relations` | Immutable mapping with `equivalent_images` (tuples of explicitly identified image keys), `occupancies` (one optional supplied value per site), and `disorder` (one optional supplied assembly/group pair per site). Missing values remain unknown; relations bind actual members and the relevant group/source frame. |
| `ImageKey` / `image_map` | `ImageKey=(source_atom_row,source_operation,integer_image)`, operation an immutable full `SymmUnit`. Every generated row has a nonempty tuple of contributing keys. Source row order, source content and its original **cell matrix plus group value** are captured once in an immutable origin binding. This is historical attribution, not a second active cell/group owner. Original input has `image_map=None`. |
| `bond_images` | Per existing directed Bond endpoint order, an exact integer vector `s` for the second endpoint's cell image; `None` means unavailable. Ordinary finite Hotpot Bond input has zero images by its existing unwrapped-coordinate meaning. An importer may not treat missing periodic information as that finite convention. The self/parallel-edge gate in §3 applies. |
| `Candidate` | Immutable tuple `(operation,atom_pairs,maximum_distance)`: one full lifted SymmUnit, a complete tuple of paired ImageKeys, maximum Cartesian residual in Angstrom. This possible relation requires complete coordinate AND labelled graph/site matching; it grants no permission to erase sites. |
| `Finding` | Existing applicable domain finding records, or a scoped immutable `(code,subjects,measurements)` tuple where no existing record fits. Exact image/edge/cycle subjects and immutable code-specific measurements from §4; no new public finding base/export. Freeze actual layouts at 15A. |
| Coordinates / cell reads | Independent float64 arrays; coordinate conversions retain `(...,3)` shape. Methods requiring absent coordinates/content report exact local unavailability, without fabricated zeros. SpaceGroup's exact matrices/vectors are genuinely immutable tuples. |
| Closed-cycle linking | Explicit oriented ordered `(current_atom_row,integer_image)` sequences, each closing with zero image voltage. Return the integer for supported separated simple closed polygonal curves; `None` means unavailable with a precise finding, never an inferred zero. |
| Cell changes | One oriented row-basis `matrix`. `preserve` is exactly `"cartesian"` (default) or `"fractional"`. Legacy scalar setters use that complete operation, preserving global orientation and Cartesian atom positions. Cell changes revalidate group metric/applicable identifications; rejection leaves all input facts intact. |
| Group replacement | `set_space_group` replaces the immutable group association, with cell/Cartesian chemistry retained. `site_relations=None` preserves existing relations **only if compatible**; an explicitly supplied mapping replaces the complete relation set in the same commit. Incompatible required maps reject atomically, rather than disappear. Historical origins remain historical. |
| Association / returns | `crystal.molecule is molecule` and `molecule.crystal is crystal` for one active association. Construction cannot steal another molecule's active association. Existing `Molecule.create_crystal` explicitly replaces/detaches the prior association atomically; the old Crystal keeps its cell/group value. Structural actions/copy isolate all writable chemistry, coordinates, cell and bond buffers; immutable SpaceGroup/SymmUnit values may be shared safely. Queries do not mutate facts or policy; complete setters return `None`. |

`unit_molecules()` first obtains the active cell projection, then returns its
complete finite lifted components in stable order, with fresh associated P1
cell contexts and remapped origins. An incomplete fragment is not completed
from distances. A periodic component stays in the Crystal and is reported as
outside finite extraction. `molecular_symmetry(None,...)` uses contained content
only when it is one complete finite molecular unit; otherwise the caller
supplies the supported finite unit with an explicit map into this context.

## 2. Crystallographic meanings and one affine algebra

The project name `SymmUnit` means an individual **symmetry operation**, a group
element acting on coordinates. In IUCr usage a *symmetry element* is the
geometric element together with its associated operations; a screw axis or
mirror plane is not itself a molecule or complete operator list. [S1, S2]

| Concept | Meaning used here |
|---|---|
| Asymmetric unit | A fundamental region for the specified group action; it need not contain a whole molecule. Input can be larger, smaller, disconnected or on special positions, rather than a minimal asymmetric unit. [S3] |
| Molecular unit | A complete finite connected chemical object reconstructed from its actual bond/image relations. A crystallographic fragment, finite molecule and infinite network are distinct domains. |
| Crystallographic orbit / stabilizer | An orbit contains all images; spatially distinct images need not be the same chemical record. A site's stabilizer fixes that site in a chosen Euclidean lift; a whole-molecule stabilizer is another subgroup. [S4, S5] |
| Full space group `G` / translations `T` | Infinite crystal isometries with full Bravais translation subgroup `T`; `G/T` is the point group. The chosen conventional cell has `T_cell=Z³`, possibly a proper sublattice of `T`; `G/T_cell` retains centring cosets. [S5] |
| Space-group type / setting | The catalogue has 230 ordinary 3D **types**, one explicitly selected reference setting/origin each. Alternate axes, cell choices, origins or Hall settings are not new types. Exact operators always refer to the current setting. [S10, S15] |

Take public row basis `M=(a;b;c)` and column basis `H=M.T`; positions satisfy
`x=H f`, or `x_row=f_row @ M`. The origin is the selected fractional frame
origin. Lengths are Angstrom, angles degrees: `alpha=angle(b,c)`,
`beta=angle(a,c)`, `gamma=angle(a,b)`. `from_matrix` preserves orientation.
Scalar construction chooses a documented right-handed canonical frame. Reject
zero/negative signed volume under that contract or an incompatible metric,
without silently repairing orientation. Scalar shape changes retain the rigid
orientation relating the current basis to its canonical shape. [S7, S8]

Every SymmUnit uses one exact affine law:

```text
g.operate(f)   = R_g f+t_g
g.compose(h)   = g @ h = (R_g R_h, R_g t_h+t_g)      # g AFTER h
g.inverse()    = ~g    = (R_g^-1, -R_g^-1 t_g)
e                      = (I,0)
g.power(n)     = g ** n; g ** 0 = e; negatives use the same inverse
```

The ABC has abstract read-only `R`, `t`, `context` primitives; sole private
`_AffineSymmUnit` supplies them. The common named methods delegate to one pure
affine mathematics owner, and magic methods call those named methods. Equality
and hashing use exact **full** `R,t` plus equal immutable group/setting values.
Thus `T_(1,0,0) != e` even though they have the same cell representative. There
is no `*`/`+` alias or parallel type-specific algebra. [S7, S14]

The [supported-operation table](operation_table.md) specifies all eight forms,
actual `R,t`, exact reconstructible `xyz`/numeric Seitz symbols, axis/plane/
origin/sense/pitch parameters, and full screw/glide powers. It also distinguishes
an off-origin rotation's nonzero `t` from an invariant screw translation. Kind
names describe the same values; they never restrict a composition result to its
input kind or introduce eight separate coordinate/graph implementations.

`from_affine`/`from_xyz` now construct **members of the supplied SpaceGroup**,
including arbitrary integer lifts. This tightens the earlier unimplemented
free-isometry proposal: supported same-group operations remain closed under
all named/magic operations. Different groups can be combined only after an
explicit validated common-group rebind; familiar crystallographic kinds or a
compatible metric alone do not prove common-lattice/group membership. Arbitrary
mixed isometries have no promised crystallographic closure.

Owner-approved abstraction (2026-11-04): a `SymmUnit` is one **complete group
element** `{R|t}` in its declared context. It is neither one of 786 fixed
operation types nor restricted to an elementary generator. A generator is an
ordinary `SymmUnit` selected as part of a group-level generating set; generating
sets need not be unique. Decomposing an affine operation in a larger group does
not imply that its factors belong to the selected SpaceGroup. For example, in
`P2_1`, `g={2_y|(0,1/2,0)}` is a member and `g**2=T_(0,1,0)`, while the pure
rotation `{2_y|0}` and the half-cell translation `T_(0,1/2,0)` are not separately
members. Keep the full element and translation carry in the common algebra.

Complete representative lists enumerate separate orbit images. For a fourfold
axis, the identity, 90°, 180° and 270° representatives all act on the input;
composing two rows gives another group element rather than replacing their two
images. SpaceGroup owns membership, setting, lattice and complete cell
representatives; a generator-based construction is another group-level input or
derivation route, not a second fact owner or SymmUnit family. The 786-pair
cross-group catalogue union is neither closed nor a generator/type limit.

For declared representatives `r_i`, preserve exact carries:

```text
r_i @ r_j = T_c(i,j) @ r_k,                   c(i,j) in Z^3
(T_n @ r_i) @ (T_m @ r_j) = T_(n+R_i m+c(i,j)) @ r_k
reduce_to_cell(g) = (r_k,n),                   g = T_n @ r_k
```

Finite representatives are **not** the complete infinite group. The selected
cell-image lattice must be normal/preserved (`R Z³=Z³`). A conventional centred
cell's quotient retains its centring translations; do not identify it with the
point quotient `G/T`. A `2_1` screw squares to a nonzero lattice translation;
a glide can likewise have infinite order as a full affine element. Wrapping a
coordinate or looking up its representative never erases that translation.

SpaceGroup construction checks exact closure/identity/inverses of the complete
lifted representation and preserves full translation data. Crystal attachment
and cell changes additionally require `R.T @ (H.T@H) @ R=H.T@H` to the declared
numeric cell tolerance. A group can serve several compatible physical cells;
its immutable context does not capture one cell's numeric dimensions. Broken
membership/setting/metric rejects only the affected operation/context; no P1
fallback or scientific truth claim follows from a valid group table.

`change_setting(P,p)` means `f_old=P f_new+p`, hence
`R_new=P⁻¹R_old P`, `t_new=P⁻¹(R_old p+t_old−p)`. It returns a fresh immutable
SpaceGroup, composes its reference map exactly, transforms the **full**
translation lattice and rebuilds all finite representatives for the new cell.
For a changed conventional/primitive index, transforming only the old finite
list is insufficient. The new cell must be a finite-index periodic cell of that
transformed lattice, with `R_new Z³=Z³`; otherwise the requested finite-cell
contract is locally unsupported. There is no implicit basis/origin guess.

This method transforms group data only. Reframing an actual Crystal also needs
the correct cell, Cartesian origin/coordinate transformation and all images,
prepared by one construction/import operation. `set_cell` changes the physical
cell in its current fractional convention; it does not silently reinterpret
operators as a coordinate-setting change. `Crystal.operate` likewise leaves the
active setting/cell/group unchanged.

The [indexed complete catalogue](space_group_catalogue.md) and
[exact data](space_group_catalogue.json) already supply every actual finite
representative for all 230 selected reference settings, not just generators or
a promise of a future table. Current IUCr H–M names, source legacy aliases,
explicit Hall/axis/origin choices, immutable source hashes/licences and the
handling of alternate settings are recorded there. The catalogue is design
reference data; runtime lookup/CIF dependency selection remains a 15A decision.

## 3. Source baseline, ownership and connected migration

The inspected Crystal-related source and test observations are attributed to
`ec4d61889a7262df765da29082e405b3c612cd15`. The concurrent parent business commit
`436e027690a0b0f4125c11e3bf3259423c716573` concerns other work; it releases no
Stage15 runtime. The scoped evidence manifest verifies the inspected source
bytes against the earlier baseline. The latest exact-file attribution also
checks 22 selected source/test files against frozen product
`436e027690a0b0f4125c11e3bf3259423c716573`: all match, including the inspected
identity/commit/dependency owners. These observations establish missing
capability and migration boundaries, not completed Crystal functionality.

| Actual owner / consumer | Inspected behavior and required migration |
|---|---|
| New `structure/symmetry.py` value module and pure affine/lattice mathematics | SpaceGroup alone owns exact group/setting data; SymmUnit references that immutable value. Crystal owns its active association, never a parallel setting/operators list. Canonical translation/point views and membership indexes derive from one group record. This module is a proposal, not current product code. |
| `structure/crystal.py`, `rules/mathematics/cell.py` | Six writable scalars; `from_matrix` reduces to lengths/angles and loses orientation. The current row conversion calls angle(a,b) alpha and angle(b,c) gamma. Replace once through this stage, with standard-angle release notes and independent oblique/rotated fixtures. Do not automatically swap historical scalar data of unknown construction convention. |
| `Molecule.create_crystal`, `_capture_factual_record`, `_install_factual_record` | Create/compare/capture/install six scalars only. Migrate the actual association, complete oriented cell, one nested SpaceGroup value and relations in the same cutover. No adapter call is required for reconstruction. |
| `_builder.MoleculeRecord.cell`, `_state` pickle, `Molecule.copy` / `components`, `operations/topology.py` merge | Record validates exactly six scalars; graph restoration recreates a fresh scalar Crystal; component merge compares scalar cells. One current-version crystal codec must cover the basis, one SpaceGroup/setting record, site/bond images, historical source-cell/group origins and both association directions. Operator references decode through that record; no independently editable Crystal-level duplicate operators or setting. Settle the private record/pickle cutover explicitly; no indefinite old/new loader. |
| `_identity.py` cell domain, `_commit.py`, dependency/result owners | Cell identity captures six scalars; rollback retains `_crystal` as a topology field. Migrate the cell identity to the oriented numeric basis and introduce only the closed symmetry/site/image dependency inputs actually consumed by Stage15. A group replacement does not dirty a calculation that consumes only unchanged cell/Cartesian facts. Rollback restores cell, immutable group association, chemical association and affected relations atomically; an old pointer to mutated cell contents is insufficient. |
| `operations/conversion.py`, `adapters/openbabel/crystal.py` and Open Babel factual import/export | Existing separate Molecule integration projects/extracts six scalars and invokes Open Babel. This is a reverse consumer to inventory, not an implementation route for Crystal. §5 replaces the conflicting old 15E direction; no Crystal operation may delegate to it. |
| `adapters/native/molecule_buffers.py`, native molecule-buffer headers and force-field boundary | Existing `unit_cell` transport is shape `(6,)`; it cannot encode orientation, group or periodic edge identity. Preserve its reviewed ABI and scientific algorithms. Its documented cell-parameter projection is not a full Crystal round trip; any new native periodic contract needs its own concrete decision. |
| `operations/geometry.py`, `adapters/structure/geometry.py`, `rules/mathematics/geometry` | Existing finite pair/segment/cycle predicates and indeterminate surface states already have one Hotpot mathematics/native owner. Prepare correct periodic lifts and reuse them; do not replace their scientific algorithms or call toolkit chemistry. |
| Existing Crystal convention/transport tests | Freeze the old angle convention, absence of matrix/molecule and six-scalar persistence. They are migration evidence, not the oracle for the corrected design. The transport test imports Open Babel and uses toolkit-backed construction; both tests were inspected only, without execution in this planning assignment. |
| Stage14D | Consumes the published domain, image maps and codec. `to_pyg` serves maximum useful new-model training information; `from_pyg` serves chemical query/review. Efficient deployed-model inputs remain at each model owner. Stage15 introduces no general feature schema or model-input redesign. |

A newly exposed entry gate is `structure/_topology.py::_build_topology_index`:
it rejects self-bonds and parallel Bonds. An image-labelled quotient graph can
need a self-edge or multiple edges between the same site rows. Adding one image
vector to each current Bond does not, by itself, support every periodic network.
Stage15A must record the supported molecular/periodic domain and the exact
representation of those cases. An extension, if needed, belongs in the sole
existing topology foundation with all writers/consumers migrated; a separate
Crystal chemical graph is forbidden. No general-network support, silent edge
collapse, automatic cell enlargement or accepted scope deferral follows from
this document. An unrepresentable imported network retains its source record
and an explicit scoped limitation until the parent resolves this gate.

The contained Molecule/Atom/Bond owners remain authoritative for elements,
isotopes, H, charge/spin, bond kind/direction, stereo and Cartesian coordinates.
Crystal owns the one oriented active-cell matrix, its SpaceGroup association and
site/image relations. SpaceGroup owns the immutable fractional group/setting;
it has no molecule, numeric cell matrix or backlinks. Fractional coordinates,
metric, neighbors and matching are derived; no second mutable
coordinate block, topology index, global equivalence label or persistent generic
cache is added. A cell-only six-argument construction remains possible; missing
molecule/group is not an empty molecule or automatically known P1.

One exact group record can be safely shared by several compatible Crystals and
returned operations. Replacing a group creates a new value/association; old
SymmUnits and historical origins keep their original group. Changing only a
physical cell retains the same fractional group when metric/relations remain
valid, so old operations do not become stale merely because lengths changed.
Changing basis/origin needs the explicit transformation in §2. Derived member
views/indexes have no independent writer or lifecycle. A table lookup materializes
only the requested group; no getter preloads all 230 groups or runs perception.

Lifecycle migration covers isolated Crystal, Molecule, and a combined reference
graph. Deep copy/pickle must preserve intended internal aliases while isolating
writable facts; equal immutable SpaceGroup/SymmUnit values may be shared.
Component extraction subsets and remaps site/image relations;
merge requires an explicit compatible context and image map or a selected target,
with atomic rejection otherwise. A cell change preserves Cartesian positions by
default; the fractional-preservation option installs cell and coordinates in one
transaction. Cell/coordinate facts, molecule association, group association and
dependent site/image relations share that publication/rollback boundary. Complete group
replacement preserves compatible relations or installs the supplied complete
mapping; it never silently clears an explicit relation. Historical origins
retain their captured cell/group; they are not an assertion that an edited
current placement still equals its original source operation. Transport between
frames uses an explicit exact map, never a copied context pointer. Pure reads
never move atoms, repair topology or rerun a model.

## 4. Molecular actions, generation and geometry observations

### 4.1 One action on coordinates, graphs and molecule images

`SymmUnit.operate` acts once on batched fractional coordinates and never wraps.
`Crystal.operate(g)` returns **one fresh, independently owned transformed
placement of the current content**. It preserves the exact active cell matrix
and active SpaceGroup value/setting; it does not append `g` to the group, change
its operations, or union the original placement with an image. Molecular/cell
replication belongs to `unit_cell`/`expand`.

Its domain is a known active `space_group`, with `g.context` equal to it and
full membership `g=T_n r_i` for one accepted representative. Its linear action
preserves the cell-image lattice (`R Z³=Z³`) and the physical metric. Unknown
group or incompatible membership is locally unavailable; matching names/metric
alone cannot establish membership. Generated P1/supercell content uses its
**active** group, so an operation retained only in original-source origins is
not automatically accepted. It needs an explicit conversion and active-group
membership check. The public group-bound `SymmUnit.operate` also supplies the
same fractional action without copying chemistry.

Within this domain, capture the actual chemical closure once, apply the common
action, remap existing facts/topology/stereo, and publish fresh contained
chemistry with isolated writable buffers and correct associations. Each source
atom/bond has one transformed counterpart; retained origin aliases are remapped
rather than added as extra chemistry. No source Crystal, group, coordinates or
relation is mutated. Exact full translations are retained even when a separate
generated-cell projection wraps coordinates.

The mandatory operation fixture captures cell matrix and complete exact active
group before `operate`; checks unchanged source state, equal output cell/group,
exactly one output copy, transformed whole coordinates/bond images and isolated
mutable buffers; then calls `operate` again with a same-group composition.
An operation outside the active group rejects atomically. This fixture makes the
unchanged-group invariant observable independently of a method description.

For the Cartesian linear part use `Q=H R H^-1`; improper actions transform the
orientation-dependent stereo representation under its existing conventions,
rather than copy an incompatible label or infer a new chemical state. No
scientific perception or external molecule API is needed.

For a Bond from site `i` to image `s` of site `j`, displacement is
`H(f_j+s-f_i)`. Under `(R,t)`, let `w_i,w_j` be the integer vectors subtracted
when wrapping its transformed endpoints. The output image is
`s'=R s+w_j-w_i`. The wrapping-only case has `R=I`. This relation, endpoint
orientation and all dative/supported haptic meaning travel together. Ordinary
contact distance never creates a covalent/coordination bond or MetalCluster.

Lift a component along its actual bond/image relations. A finite molecular lift
must close every cycle with zero net image translation. A nonzero cycle voltage
reveals a periodic component, not a finite molecule to unwrap atom by atom.
Missing cross-image connectivity cannot be inferred merely from proximity.
Retain periodic topology and precise unsupported/unknown observations rather
than inventing finite chemistry.

### 4.2 Cell filling, whole units and expansion

Use stable source-site, exact representative, then integer-image order. A
half-open cell and preregistered numeric boundary convention determine site
placement. Crossing a face must not truncate a molecule: keep periodic edges
and use the finite lift for molecular queries/extraction. Occupancy/disorder
are transported as supplied, not converted into guessed alternate molecules.

Keep every generated source/operator/image origin. Only an **explicitly
validated site/image identification** can share a generated row; retain every
contributing alias and all compatible Bond origins. A declared site stabilizer
can justify repeated images of that same site. Numeric near coincidence or a
possible whole-molecule relation cannot authorize destructive identification.
Distinct supplied source atoms, bonds, charge and occupancy remain intact.

The proposed generated-cell representation is active **P1**, containing the
complete generated image projection and its captured original context/origins.
`space_group` is a P1 SpaceGroup value and `symm_units` delegates to it; its
setting/operations describe that active representation. The original cell/group
remains once in the immutable origin binding. This is a representation choice, not a newly inferred crystal space group. Reapplying `unit_cell()` to
it must not apply the original symmetry again. `representation` and the single
codec make this distinction portable without an old/new runtime switch.

For `D=diag(n1,n2,n3)`, form `H'=H D` and `f'=D^-1(f+k)` for the requested
integer cell images `k`, transporting boundary-crossing Bonds to that cell.
`expand((2,3,2))` therefore accounts for exactly twelve active image cells before
any separately justified identification. If starting with independent content,
first obtain the one-cell projection; if already generated, expand that active
cell. Unsupported parent operations need not preserve an anisotropic supercell;
never relabel them as active symmetry without a correct basis/translation test.

Current geometry/topology queries address `(current_atom_row, integer_image)`
in the one active cell frame. `image_map` is a join to captured source origins,
not a substitute coordinate frame. A source-frame operator cannot act in a
generated/supercell frame without the explicit conversion.

Replication creates no new MCA/partial-charge/model observation. Transport a
supported site/frame result only through its actual mapping/currentness rules;
otherwise retain it as an attributable historical observation or omit its live
attachment with the appropriate finding. No writable input/output buffers alias.

### 4.3 Possible same molecule requires both complete tests

Compare two specified generated placements `A'=g(A)+n` and `B'=h(B)+m` in
one active frame, after reconstructing each whole finite lift from its actual
bond images. Seek possible identification only where those **already placed**
units overlap; do not apply a fresh symmetry operation to bring distinct
placements together. A symmetry operation relating spatially distinct copies
establishes an orbit relation, not coincidence or molecular identity.

Require **one bijection satisfying both conditions together**:

1. Every already transformed atom of `A'` overlaps its paired atom of `B'`
   within the stated Cartesian Angstrom tolerance. Each placement uses one
   affine action and one common integer lift for the whole molecule; the chosen
   physical image/anchor remains fixed during the comparison.
2. That same bijection is a complete molecular graph/site isomorphism: members,
   element/isotope and declared presence/facts, all chemical connections and
   their kind/order/dative direction, supported stereo orientation, occupancy
   and disorder semantics are preserved in the declared matching domain.

An isolated pair match, centroid/RMSD match, partial overlap, matching formula,
or an isomorphism with a different incompatible coordinate permutation is
insufficient. Do not independently minimum-image-wrap every atom or optimize a
new free rotation when testing a declared crystallographic operation. Unknown
required information yields an unresolved scoped observation, rather than a
complete relation. Partitioning by graph labels/site facts is a cheap candidate
filter; it is not the final matching test.

Use finite unit bounds and the stated overlap tolerance to find candidate
placements, with a stable cell anchor and retained integer-image identities;
do not enumerate infinitely translated copies or erase distinct requested
supercell images. The returned full operation is the placement/relative affine
map explaining the paired origins, not an extra alignment fitted during matching.
Return the complete candidate map and measured residual. Even a complete match
is a **possible same-molecule relation** in the stated input/representation;
finite coordinates and graph equivalence do not determine scientific identity,
disorder interpretation or which supplied records the researcher intends to
identify. The query never deletes/averages/merges either observation. Explicit
site/image identification is a separate caller decision, validated atomically
against all affected facts and edge maps before generation uses it.

For `molecular_symmetry`, retain the full lifted operations mapping a selected
finite molecule to itself. For its fixed centre `f0`, the point action is `Q`
about `H f0`; the affine origin term need not be zero. Nonzero pure translations
cannot stabilize a finite molecule. The resulting matches are crystal-supported
molecular symmetry candidates, not a search for the full isolated-molecule point
group. A mathematical stabilizer/group claim additionally requires identity,
inverse and composition closure of the complete maps. Numeric tolerance can
produce a nonclosed set; preserve those matching observations and report that
limitation, rather than adding/removing matches to fabricate a subgroup.

### 4.4 Triclinic neighborhoods, crowding and periodic topology findings

Distances use the actual triclinic metric. For `d=f_j-f_i`, examine integer `n`
with `||H(d+n)|| <= r`. If `b_i` is row `i` of `H^-1`, Cauchy–Schwarz gives
`|d_i+n_i| <= r ||b_i||`; enumerate that finite integer box and compare actual
Cartesian vectors. A nearest-image query can start from an actual candidate
upper bound. Pair this bound with spatial/neighbor lists and reuse prepared
local image neighborhoods. Per-component rounding and a fixed 27-image stencil
are not correct for arbitrary skew bases. Do not call Gemmi/toolkit nearest-image
functions as a substitute for this Hotpot algorithm.

A close expanded pair is classified against the complete coincident-placement
relation, not just atom coincidence or common orbit membership. A symmetry
relation between neighboring molecular copies never exempts their actual close
contacts from observation. If the full possible same-molecule relation fails,
its unusually close independent images are a **crowding observation** under the
caller cutoff. If matching is unresolved, report both the proximity and the
unresolved relation; do not assert independence. Even accepted molecular copies
can contain local contact observations worth retaining; relation findings and
contact-control policy remain separate.

Periodic ring/bond analysis uses complete lifted cycles and necessary nearby
segments, with canonical image-aware keys to avoid duplicated reports. Reuse
the existing Hotpot segment/cycle predicates and their surface/indeterminacy
semantics. A boundary-crossing ring must not be split into artificial fragments.
A reported bond–ring piercing is intersection of a segment with the selected
ring surface; it is different from two disjoint closed loops being linked.
Neither observation establishes chemical impossibility. Catenanes and linked
crystal structures can be legitimate objects for this harness. [S13]

The concrete proposed linking tool is `linking_number`, limited to two selected
oriented, disjoint, simple closed polygonal curves in one finite Euclidean lift.
Use a pure Hotpot polygonal Gauss-linking segment-pair/angle sum with an explicit
numerical error bound sufficient to resolve its integer. Preserve numerical
indeterminacy rather than blindly round an uncertain sum. The two-loop input,
zero-voltage closure and separation conditions are mathematical applicability
requirements, not arbitrary chemical exclusions. [S11]

A supported nonzero linking number establishes nontrivial linking of that pair;
zero is only that invariant's value and does **not** prove the pair unlinked
(the Whitehead link is a counterexample). [S12] Reversing one loop changes the
sign; a cyclic vertex shift, subdivision or a common proper rigid motion must
not change the value. Report the integer and exact loop/image subjects, without
calling it an error or automatically repairing a valid catenane. [S11, S13]

Open chains, touching/intersecting or degenerate loops, and nonzero-voltage
periodic paths are outside this finite closed-loop contract. Return unavailable
with the exact cause; retain the geometry and any separate contact/piercing
observation. Do not close periodic paths artificially or equate a finite-image
linking number with an infinite network's complete entanglement classification.
Periodic geometry can still select and lift actual finite zero-voltage loops.
No general knot classification, untangling optimizer or structural repair
framework is added.

| Proposed code / trigger | Available value and default control | Break / commit consequence |
|---|---|---|
| `CRYSTAL-POSSIBLE-SAME-MOLECULE`: complete coordinate + graph/site bijection | Candidate image pairs, lifted operation and maximum residual remain available; report and continue. | Per-code operation/sample/batch break may be selected; never commits identification. |
| `CRYSTAL-COINCIDENT-IMAGES`: local image coincidence without a complete relation | Retain all source/image observations and actual distance; report coincidence or unresolved matching precisely. | Same selectable break scopes; no deduplication. |
| `CRYSTAL-CROWDING`: close expanded images outside a complete same-molecule relation | Report affected images, actual distance and the named cutoff/method; retain geometry. | Default continue; caller can break at the declared boundary, without changing finding content. |
| `CRYSTAL-PERIODIC-BOND-RING` / `CRYSTAL-INTERPENETRATION`: existing geometric predicate on a specified periodic lift | Preserve affected edge/cycle images, predicate result and necessary intersection/indeterminacy measurements. No global truth/entanglement verdict. | Default continue; per-code break only, no implicit repair. |
| `CRYSTAL-CLOSED-LOOP-LINK` / `CRYSTAL-LINKING-UNAVAILABLE`: nonzero resolved pair linking, or the finite closed-loop invariant cannot be evaluated | Preserve the integer for a supported pair, or the exact open/degenerate/intersecting/periodic/numerical limitation. A zero value is not an unlink certificate. | Nonzero linking is an observation, with default continuation even for a catenane; unavailable is local to this invariant. No deletion, implicit repair or general network judgment. |
| `CRYSTAL-MOLECULAR-LIFT-UNAVAILABLE` / `CRYSTAL-STABILIZER-NONCLOSURE` | Retain periodic/incomplete content or numerical match observations; only the requested finite lift/group claim is unavailable. | Local unavailable result with an exact reason; independent queries/samples continue. |
| `CRYSTAL-CONTEXT-INTEGRITY` / `CRYSTAL-IMAGE-MAP-INTEGRITY`: invalid cell, incompatible operator/setting/closure, broken members/images | Affected context or proposed structural result cannot be published; preserve the input and diagnostic measurements/cause. | Local rejection and complete rollback; no silent group replacement. Fatal escalation requires shared execution integrity actually lost. |
| `CRYSTAL-UNSUPPORTED-PROJECTION`: missing/unrepresentable topology, disorder or format information | Preserve available input/observations and identify the exact supported projection/loss. A missing required map is not a usable full result. | Reject only the dependent full projection, or return the expressly requested supported partial projection; no fabricated bonds/equivalence. |

These codes and payloads are explicit proposals to freeze at 15A, reusing
existing geometry codes/records where their meaning is unchanged. Record the
actual `code | domain/trigger | available value | default control | allowed break
scope | commit/fallback consequence`. Numerical boundary/matching tolerance,
contact cutoff and integrity/identification conditions are different contracts,
fixed before observing results. No energy/spin/FF/xTB/model is run by a geometry
or symmetry query. Findings do not change with warning/break policy.

## 5. Linear implementation sequence and non-toolkit boundary

Production starts only from the validated Stage0–12 `B_exp`, with the parent's
explicit Stage15 source boundary, accepted §16.4 decisions and affected
verification map. Planning/source inspection and small mathematical probes do
not release that gate. Stage13 retains its independent `D_exp`/`I_exp` sequence.

| Slice | Coherent work and owners | Required exit / next release |
|---|---|---|
| 15A — decide the tree and domain | Freeze §1's complete Crystal/SpaceGroup/SymmUnit signatures/layouts/permissions, SpaceGroup-only group/setting/lattice authority, group-bound algebra/magic methods and unchanged-group `operate`, exact frame/carry rules, matching versus identification, generated P1, and the self/parallel periodic-edge gate. Review the actual 230-reference catalogue, current/legacy H–M labels, alternate-setting coverage and licences. Inventory all scalar/record/identity/native/I/O consumers and the supported scientific domain. Select the bounded CIF/table dependency and its distribution implications. | Parent records exact new public/schema decisions, source identity, prospective checks and cost baseline under §16.4. No unresolved convention or unrepresentable edge reaches production. Stage14D may prepare against this settled schema. |
| 15B — one carrier/context/lifecycle migration | `structure/crystal.py` plus the one immutable SpaceGroup/SymmUnit module; pure cell/affine/lattice mathematics; one crystal operation owner; actual Molecule association, builders, codec/pickle, copy/components/merge, identity/dependencies and transaction rollback. Assign exclusive files before writing. | Standard angles/orientation, same-group exact/magic algebra, derived Crystal.symm_units, no second group/setting/chemical/cell truth, fresh outputs, complete rollback and alias/remap checks. Preserve native ABI and existing scientific algorithms. |
| 15C — operate, reconstruct and expand | The one-placement, unchanged-cell/group `operate` action; exact full group membership and cell representative/carry handling; supported site/image relations and bond images; cell filling, finite molecular-unit extraction, and general integer repeats. | Actual P1, inversion, centring, ordered special positions, screw/glide, crossing-cell finite units and declared network limits; exact origins/counts and no double symmetry application. Publish the implemented domain to Stage14D after 15B/C evidence, without waiting for PyG. |
| 15D — relations and periodic findings | One full coordinate + graph/site matching implementation for candidate relations and molecular-symmetry queries; metric-bounded neighborhoods; lifted existing geometry predicates, bounded finite closed-loop linking and precise controls. | Independent complete/partial/improper/ambiguous matches, crowding, periodic ring/bond and finite loop-linking cases; observations, source state and local failure semantics preserved. No destructive deduplication or global scientific verdict. |
| 15E — pure persistence, CIF and consumer closure | Extend the one pure Hotpot codec and existing data-capture consumers with one group/setting record and explicit source-frame bindings; one justified lazy Gemmi table/CIF adapter if selected, or a separately approved bounded pure CIF path. Close every affected file-format migration before its consumer round trip. Document separate legacy toolkit/native projection limits using prepared values. | Real supported CIF/file/current-version pickle round trips and outside-checkout installed domain evidence, with actual setting/origin, precision, supported site/bond/image relations and loss coverage. Publish each touched format's migration closure to Stage14D. No OB/RDKit/CCDC call is required or permitted in this Crystal route. |
| 15F — delivery | Complete tree/examples, connected writer/consumer and removal tables, versioned scientific observation report, representative scaling/time/RSS/completion/failure records, applicable source/installed/package/version evidence and coherent delivery. | Parent reviews the concrete domain deliverable and remaining limitations/gates. Stage15's own exit is independent of the Stage14D PyG fixture. No overall completion follows automatically. |

15B/C provides the common validator for an already supplied site/image map,
including the affected coordinates, facts and edges. 15D adds candidate search
and public relation/stabilizer queries using that same validator. Generation
does not depend on 15D search or infer an identification from it.

**15E correction:** the earlier direction “CIF and existing OB/RDKit projections”
as Crystal implementation/acceptance is superseded. No Crystal method,
operation, geometry/topology rule, import/export orchestration or indirect
Molecule helper may call OpenBabel, RDKit or CCDC Python APIs. Learning from their
documented methods is allowed; depending on them for Crystal behavior is not.
Copy/build/query/expansion must remain usable without those Python packages.
Existing separate `Molecule.to_obmol`/`to_rdmol` integrations remain owned by
their adapters; their reverse consumption of cell parameters must be inventoried
and documented or migrated at the pure capture boundary. Their external-object
round trips cannot be presented as Stage15 Crystal capability or silently
replace missing periodic facts. This revision adds no Crystal toolkit façade.

The official Gemmi **0.7.5** table/Hall generator was used only as a justified
standalone documentation-data tool for the complete catalogue, with immutable
source commit/hash/licence and independently retrieved COMCIFS reference labels.
This does **not** select Gemmi as a runtime dependency. It remains a concrete
candidate for lookup/CIF parsing only; before adoption settle version/API,
licence/distribution, lazy loading, setting/origin behavior and supported fields.
Convert selected lookup/CIF values into one exact Hotpot SpaceGroup record.

Gemmi's documented `Op` multiplication wraps translations into `[0,1)`, so it
cannot implement Hotpot's full affine composition with retained carry. Hotpot
exact rationals are not restricted to its denominator 24. Algebra, coordinate/
molecule actions, matching, metric neighbors and topology predicates remain
Hotpot's own implementations. The versioned catalogue is an actual attributed
reference derivation, not an unmaintained hand-transcribed runtime table; its
promotion into runtime data would itself need the 15A distribution/lookup
choice and affected checks. Retain no external crystal object as a fact owner
and substitute no prohibited/paid backend. An unavailable lookup is precise
local unavailability, never a P1 fallback. [S9, S10, S15]

Stage14's dependency remains acyclic: 15A's accepted schema supports preparation;
15B/C's implemented domain plus **15E closure for any touched external format**
support actual 14D round trips. Stage15 publishes and closes its own domain
without 14D. The shared Crystal↔Data fixture belongs to Stage14D/overall
integration. Stage14 maximizes useful training information and chemical review,
while each deployed model retains its efficient custom feature interface.
Neither branch changes main-scope acceptance or retroactively redefines `B_exp`.

## 6. Required business evidence and remaining decisions

| Fixture family | Independent expected facts / rejection cases |
|---|---|
| Cell and lifecycle | Orthogonal/triclinic and rotated full bases; standard alpha/beta/gamma; Cartesian↔fractional round trips; both cell-preservation modes; atomic failed updates; combined Crystal/Molecule copy/pickle aliases and remapped components/merge. |
| Group/affine ownership | SpaceGroup immutable and sole group/setting/lattice owner; Crystal.symm_units delegates with no shadow fact; exact full `@`, `~`, `**`, equality/hash and named-method agreement, including negative/zero powers. Off-origin rotation, full screw/glide carries, centring versus point quotient, setting conjugacy including changed translation index, stale old-group operations and incompatible/nonclosed sets. No grouping by kind name alone. |
| Operation invariant | A known active group, a whole finite unit and an unwrapped active-group operation: source unchanged; output cell/group value unchanged; exactly one transformed copy; coordinates/stereo/directed bond images correct; mutable buffers independent. A second same-group composition stays supported; an outside-group operation rejects. No group extension or original+image union. |
| Catalogue | All 230 types and selected settings, exact rational operators, unique identities/inverses, every cell pair product with full integer carry, symbol decoding, full centring lattice and independent named/centred/screw examples. Metadata/current-name checks are separate from molecular/scientific runtime evidence. |
| Site/cell generation | P1, general and ordered special positions, ambiguous/unapproved coincidence, explicit image identification, occupancy/disorder preservation, boundary coordinates, full crossing-cell units and no repeated application of origin symmetry. No assumption of a minimal molecular asymmetric unit. |
| Periodic topology | Correct `R*s+w_j-w_i`, directed edges, finite zero-voltage cycles and nonzero-voltage periodic components; self/parallel quotient-edge cases at the accepted representation boundary. No silent edge loss, second topology graph or fake finite extraction. |
| Molecule relations | Complete overlap + same bijection isomorphism; complete coordinate overlap with nonisomorphic graphs; isomorphic graph with partial coordinate overlap; one coincident atom; incompatible isotope/site/disorder/stereo; improper/chiral action; one common lift versus per-atom wrapping; spatially distinct orbit-related copies remain distinct and can crowd; map-closure limits. |
| Geometry/control | Skew nearest image outside rounding/fixed-27 choices, exact/near coincidence, independent-image crowding, unresolved relations, periodic ring/bond and interpenetration cases with indeterminacy retained; default continuation and explicit per-code breaks checked separately. |
| Finite loop topology | Analytic separated unlink and Hopf-link fixtures; reviewed Whitehead-link coordinates with linking number zero but no unlink assertion; legitimate molecular catenane as a retained observation; orientation reversal, cyclic reindexing, segment subdivision and common rigid motion; crossing-cell finite zero-voltage loops; open, touching/intersecting, degenerate and nonzero-voltage periodic cases unavailable under the finite invariant. |
| Persistence/integration | Actual supported CIFs and exact group settings, independent expected images/topology, pure current-version codec and installed examples, all actual scalar writers/readers migrated. Guard against direct and indirect prohibited API calls. Stage14D separately checks useful Crystal/Data fields and real batching. |
| Cost/support | Representative finite molecular crystals and the accepted periodic domain at increasing sizes; source/expanded counts, neighbor/image work, complete-work time/RSS and real failures. CP311 first, then applicable declared versions and installed package gates. No richer-work versus six-scalar “speedup” comparison. |

Freeze expected facts, allowed numerical changes, tolerance and rejection
conditions before execution. Analytic constructions and independently reviewed
source structures provide the oracle; two paths sharing one implementation or
historical equality alone do not. Real-data reports retain ambiguous relations,
unsupported input, failed attempts and original denominators. Unavailable
scientific corpora/dependencies get explicit placeholder reports with blank
results, not confidence grades or a claimed pass.

The parent still needs concrete decisions on (a) the proposed complete
Crystal/SpaceGroup/SymmUnit tree, new keyword/value/export contracts, group-bound
`from_affine`/`from_xyz`, explicit setting transformation and complete group
replacement, including `close_contact_cutoff` replacing the earlier
underspecified `close_contact_policy`; (b) the periodic self/parallel-edge domain
and representation; (c) the one private record/pickle cutover and supported CIF
loss map; and (d) dependency selection and preregistered numerical conventions.
These are bounded implementation-entry decisions, not a request to approve an
unspecified framework. This plan neither approves production nor removes scope.

The two independent local planning assignments used **GPT-6.1 Sol / max** as
assigned by the parent; runtime model metadata is not independently verified.
Prior mathematical/source evidence is indexed in the maintained
`.plan20-artifacts/20261104/stage15/symmetry-plan-review/INDEX.md`. The latest
SpaceGroup/catalogue revision is indexed at
[the durable evidence index](../../../../.plan20-artifacts/20261104/stage15/spacegroup-plan-review/INDEX.md).
Its actual isolated standard-library check reports **230/230** selected types
and current COMCIFS Hall/name matches; **4,425** exact operator/symbol round trips
and full inverses; **284,089** complete pair products plus the same number of
positive/negative lifted-pair products, all with retained carries; **167,201**
nonzero-carry products; full centring lattice actions; all complete Markdown
lists; three independently published small-group lists and **15** analytic
screw/glide/inverse/order cases. The source table and its Hall generator are
one route, not independent scientific oracles; shared historical ancestry of
reference metadata is also stated. The evidence index records source/artefact
hashes, exact commands, retrieval failures and scoped document checks.

These are document/data/algebra design checks, not product tests or a chemical,
metric-tolerance, CIF, installed-package, performance or scientific-acceptance
gate. No Hotpot/toolkit import, forbidden API, engine/model, remote/helper job,
production/test edit, staging or commit is part of this assignment. Parent
log/index/main-plan/Git and every Stage0–12, Plan12, Stage13/14/15 and overall
gate retain their existing owners.

## Primary references and limits

The following primary sources support terminology/conventions, not a completed
Hotpot implementation. They were consulted on 2026-11-04; runtime dependency
behavior must be verified separately at the accepted implementation entry.

- **S1** — [IUCr: symmetry operation](https://dictionary.iucr.org/Symmetry_operation).
- **S2** — [IUCr: symmetry element](https://dictionary.iucr.org/Symmetry_element).
- **S3** — [IUCr: asymmetric unit](https://dictionary.iucr.org/Asymmetric_unit).
- **S4** — [IUCr: crystallographic orbit](https://dictionary.iucr.org/Crystallographic_orbit).
- **S5** — [IUCr: space group](https://dictionary.iucr.org/Space_group).
- **S6** — [IUCr nomenclature report: Symbols for symmetry elements (1992), Acta Cryst. A48, 727–732](https://www.iucr.org/who-we-are/commissions/commission-on-crystallographic-nomenclature/published-reports/symsym), including the distinction and notation for symmetry elements and operations.
- **S7** — [IUCr: matrices, mappings and crystallographic symmetry](https://www.iucr.org/what-we-do/education/pamphlets/matrices-mappings-and-crystallographic-symmetry).
- **S8** — [IUCr core CIF cell definitions](https://www.iucr.org/resources/cif/dictionaries/browse/cif_core1/2), together with the [COMCIFS core dictionary](https://github.com/COMCIFS/cif_core/blob/master/cif_core.dic). The IUCr page was search-accessible but direct retrieval failed during this review; that limit is recorded in the evidence index.
- **S9** — [Gemmi official unit-cell and symmetry documentation](https://gemmi.readthedocs.io/en/stable/cell.html).
- **S10** — [Gemmi official symmetry/table documentation](https://gemmi.readthedocs.io/en/stable/symmetry.html).
- **S11** — [Bertolazzi, Ghiloni and Specogna: Efficient computation of Linking number with certification, arXiv:1912.13121](https://arxiv.org/abs/1912.13121), especially the separated closed-polygon domain and segment-pair angle sum. This is a method reference, not a copied implementation or completed Hotpot validation.
- **S12** — [Kauffman: A Self-Linking Invariant of Virtual Knots, arXiv:math.GT/0405049](https://homepages.math.uic.edu/~kauffman/SelfLink.pdf), which explicitly notes the classical Whitehead link as a nontrivial two-component link with linking number zero. No virtual-knot machinery is adopted.
- **S13** — [O’Keeffe and Treacy: Isogonal weavings on the sphere: knots, links, polycatenanes, Acta Cryst. A76 (2020)](https://journals.iucr.org/a/issues/2020/05/00/ug5014/index.html), supporting the distinction between legitimate linked structures and geometric crossing.
- **S14** — [IUCr Commission report: Seitz symbols for crystallographic symmetry operations (2014)](https://journals.iucr.org/a/issues/2014/03/00/es0405/index.html); the [supported-operation companion](operation_table.md) links its exact notation and bounded independent examples.
- **S15** — [Complete attributed 230-type catalogue and provenance](space_group_catalogue.md), using pinned official Gemmi 0.7.5 and COMCIFS Enumeration Templates 1.4.9/current core-dictionary sources; exact data and preserved licences are linked there. All 230 selected Hall symbols agree; this supplies reference-setting coverage, not all Hall alternatives or a runtime dependency approval.
- Governing local records: [effective goal](../../goal.md), [chemical-type interface protocol](../../chemical_type_interface_contract.md), [Stage14 plan](../stage14/review.md), [main plan §18](../../fact_rule_architecture_implementation_plan.md#18-additional-owner-directed-branches-stage14-and-stage15).
