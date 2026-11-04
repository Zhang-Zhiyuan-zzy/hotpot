# Stage15 implementation plan — Crystal spatial symmetry and molecular expansion

Status: owner-directed planning revision, 2026-11-04. Runtime authoring has not
started. Crystal's primary job is to apply crystallographic spatial symmetry
and copy/expand molecular units, with useful geometry/topology observations.
The following surface is a concrete **proposal for Stage15A**, not a released
API or scientific approval. The parent owns acceptance, main-plan registration,
interface/schema decisions and runtime entry under §16.4.

## 1. Complete proposed public tree — before implementation

This is one coherent surface. `Crystal` and the abstract `SymmUnit` are the
proposed public types; notation such as `Array`, `ImageKey`, `Candidate` and
`Finding` below describes values, not additional exported wrapper classes.
Stage14 owns its already owner-selected `to_pyg`/`from_pyg` names and encoding.

```text
Crystal
├── construction
│   ├── Crystal(a, b, c, alpha, beta, gamma, *, molecule=None,
│   │           space_group=None, setting=None, operators=None,
│   │           site_relations=None, bond_images=None)
│   └── from_matrix(matrix, *, molecule=None, space_group=None, setting=None,
│                   operators=None, site_relations=None, bond_images=None) -> Crystal
├── contained chemistry and crystallographic facts
│   ├── molecule -> Molecule | None                         [read-only association]
│   ├── a, b, c -> float                                    [Angstrom; legacy setters]
│   ├── alpha, beta, gamma -> float                         [degrees; legacy setters]
│   ├── matrix -> independent float64 Array(3,3)             [read-only; basis rows]
│   ├── volume -> float                                    [Angstrom^3; derived]
│   ├── space_group -> str | None                          [resolved name or unknown]
│   ├── setting -> str | None                              [basis AND origin choice]
│   ├── symmetry_context -> immutable frame binding         [read-only; see §2]
│   ├── symm_units -> tuple[SymmUnit, ...] | None            [cell-coset representatives]
│   ├── site_relations -> immutable mapping                 [contextual, see below]
│   ├── bond_images -> tuple[Int3 | None, ...]              [complete Bond order]
│   ├── representation -> "independent" | "cell" | "supercell"
│   └── image_map -> tuple[tuple[ImageKey, ...], ...] | None [all origins per output row]
├── complete, explicit changes
│   ├── set_cell(matrix, *, preserve="cartesian") -> None
│   ├── set_symmetry(*, space_group=None, setting=None, operators=None) -> None
│   └── set_site_relations(relations) -> None
├── coordinate and molecular actions
│   ├── to_fractional(cartesian) -> independent float64 Array(...,3)
│   ├── to_cartesian(fractional) -> independent float64 Array(...,3)
│   ├── operate(operation: SymmUnit) -> Crystal              [fresh; active-group member]
│   ├── unit_cell() -> Crystal                              [fresh periodic cell content]
│   ├── unit_molecules() -> tuple[Molecule, ...]             [fresh finite unwrapped units]
│   ├── expand(repeats: Int3) -> Crystal                     [e.g. (2,3,2)]
│   └── copy() -> Crystal                                   [independent chemistry/context]
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

SymmUnit                                              [ABC; immutable operation]
├── construction returning the one concrete affine implementation
│   ├── from_affine(R, t, *, context) -> SymmUnit
│   └── identity(*, context) -> SymmUnit
├── R -> exact rational 3×3 tuple                         [read-only linear part]
├── t -> exact rational length-3 tuple                    [read-only; NOT wrapped away]
├── context -> immutable crystallographic frame binding   [read-only]
├── kind -> identity / translation / rotation / screw /
│           reflection / glide / inversion / rotoinversion [derived classification]
├── operate(fractional) -> independent float64 Array(...,3) [no wrapping, no mutation]
├── compose(other: SymmUnit) -> SymmUnit                    [self AFTER other]
├── inverse() -> SymmUnit
├── power(n: int) -> SymmUnit                              [negative, zero, positive]
├── equality / hash                                      [exact value AND context]
└── reduce_to_cell() -> (SymmUnit, Int3)                    [representative AND carry]
    └── identity / translation / rotation / screw / reflection / glide /
        inversion / rotoinversion are mathematical forms of this same family,
        not eight independent coordinate/molecule algorithms or toolkit classes

Molecule association boundary
└── create_crystal(a, b, c, alpha, beta, gamma, *, space_group=None,
                   setting=None, operators=None, site_relations=None,
                   bond_images=None) -> Crystal
    └── delegates to the same complete association/context operation
```

All new spelling, keywords, properties, return layouts, permissions and exports
in this tree need the parent's concrete decision before their dependent code.
In particular, `unit_molecules`, `operate`, `same_molecule_candidates`,
`linking_number`, the three complete setters, `copy`, `volume`, context/image
access and the `SymmUnit`
construction/algebra are intentional new proposals. `SymmUnit` is proposed at
`hotpot.cheminfo.structure.symmetry`; a top-level `hotpot.SymmUnit` export is a
separate explicit decision. Existing `hotpot.Crystal` remains its ordinary name.
There is no `mol` alias, generic registry, success envelope or confidence grade.

The value layouts and permissions are:

| Value / entry | Exact proposed meaning and ownership |
|---|---|
| `operators` | A finite sequence of exact fractional `(R,t)` data, normalized into `SymmUnit` values in this Crystal's frame. Rational input is explicit integer/fraction data, not an arbitrary floating-point matrix guessed into rationals. A supplied named group/setting must agree with the explicit operators. |
| `symmetry_context` | One immutable binding of the oriented cell basis, selected fractional basis/origin, full translation lattice where known and any declared group/setting; the integer cell-image lattice remains distinct. It can be a private value shared by Crystal and its operations; this proposes no public context class, global ID registry or second cell owner. Old returned operations retain their original immutable context after a cell/group update. |
| `symm_units` | Stable, exact representatives for one declared cell quotient, including centring cosets. `None` means unavailable group information; explicit P1 has the identity tuple. These representatives do not enumerate the infinite space group. |
| `site_relations` | Immutable mapping with `equivalent_images` (tuples of explicitly identified image keys), `occupancies` (one optional supplied value per site), and `disorder` (one optional supplied assembly/group pair per site). Missing values remain unknown. Relations bind actual members and the specified operator context, rather than global labels on Atom. |
| `ImageKey` / `image_map` | `ImageKey=(source_atom_row, source_operation, integer_image)`, where the operation is an immutable `SymmUnit` in its captured source context. Every generated row has a nonempty tuple of contributing keys. Source row order and one internal source-content binding are captured once, not private UIDs or repeated full source records per atom. Original input has `image_map=None`. |
| `bond_images` | Per existing directed Bond endpoint order, an exact integer vector `s` for the second endpoint's cell image; an explicit `None` entry means unavailable. Ordinary finite Hotpot Bond input has zero images by its existing unwrapped-coordinate meaning. An importer may not turn missing periodic information into that ordinary finite convention. The self/parallel-edge gate in §3 still applies. |
| `Candidate` | Immutable tuple `(operation, atom_pairs, maximum_distance)`: one full lifted `SymmUnit`, a complete tuple of paired `ImageKey` values, and the measured maximum Cartesian residual in Angstrom. This is the requested possible relation, with complete coordinate AND labelled graph/site matching; it is not permission to erase sites. |
| `Finding` | Existing applicable domain finding records, or a scoped immutable `(code, subjects, measurements)` tuple where no existing record fits. Subjects are exact image/edge/cycle keys; measurements are immutable named values defined by the code in §4. No new public finding base class/export is implied. Stage15A freezes each actual layout before authoring. |
| Coordinates / cell reads | Independent float64 arrays; `to_fractional` and `to_cartesian` preserve the input's `(...,3)` shape. Methods requiring absent coordinates or content report the exact local unavailability; they do not fabricate zeros. |
| Closed-cycle linking | `first_cycle` and `second_cycle` are explicitly oriented, ordered sequences of `(current_atom_row, integer_image)` pairs in the active frame, each closing with zero image voltage. Return the integer linking number for supported separated simple closed polygonal curves. `None` means the requested invariant is unavailable with a precise finding, never an inferred zero. |
| Cell changes | `set_cell` takes the whole oriented row-basis matrix. `preserve` is exactly `"cartesian"` (default) or `"fractional"`. The retained six scalar setters use the same complete operation, preserving global orientation and Cartesian atom positions. No independent scalar/matrix truth or writable matrix view. |
| Association | `crystal.molecule is molecule` and `molecule.crystal is crystal` for one active association. Construction cannot silently steal a molecule from another context. Existing `Molecule.create_crystal` is the explicit replacement entry: detach the prior active association and bind the new one atomically, retaining the old cell/context value. No new direct parent setter. |
| Returns / mutation | Structural actions and `copy` return fresh content with isolated numeric/bond buffers. Queries never change coordinates, chemistry, relations or warning policy. The three complete setters return `None`; context/chemistry changes publish only after validation. |

`unit_molecules()` first obtains the active cell projection, then returns its
complete finite lifted components in stable order, with fresh associated P1
cell contexts and remapped origins. An incomplete fragment is not silently
completed from distances. A periodic component is retained in the Crystal and
reported as outside finite-molecule extraction, rather than cut into an invented
finite molecule. `molecular_symmetry(None, ...)` uses the contained content only
when it is one complete finite molecular unit; otherwise the caller supplies
the intended supported finite molecule. That selection must have an explicit
map into this context; an unrelated molecule is not implicitly reframed.

## 2. Crystallographic meanings and one affine algebra

The project name `SymmUnit` means an individual **symmetry operation**, a group
element acting on coordinates. In IUCr usage a *symmetry element* is the
geometric element together with its associated operations; a screw axis or
mirror plane is not itself a molecule or a complete operator list. [S1, S2]

| Concept | Meaning used here |
|---|---|
| Asymmetric unit | A fundamental region for the specified group action; the term does not promise a complete molecule. Supplied chemical content may be larger, smaller, disconnected or on special positions, and is not assumed to be a minimal asymmetric unit. [S3] |
| Molecular unit | A complete finite connected chemical object reconstructed from its actual bond/image relations. A crystallographic fragment, a finite molecule and an infinite coordination/polymer network are different domains. |
| Crystallographic orbit | All images of a site under the group. Related images can be spatially distinct; orbit membership does not identify their chemical records. [S4] |
| Site stabilizer | Operations fixing a specified point/site in the selected Euclidean lift. Its point action is the site's symmetry; this differs from the subgroup preserving a whole finite molecule. [S4, S5] |
| Space group `G` | The infinite group of crystal isometries, containing the full Bravais translation subgroup `T`. `G/T` is the crystallographic point group. The chosen cell may have a smaller integer translation subgroup `T_cell`; its cell quotient retains centring cosets. [S5] |

Take the public row basis `M=(a;b;c)` and column basis `H=M.T`; positions satisfy
`x=H f`, or `x_row=f_row @ M`. The origin is the explicitly selected fractional
frame origin. Lengths are Angstrom and angles degrees, with standard
`alpha=angle(b,c)`, `beta=angle(a,c)`, `gamma=angle(a,b)`. `from_matrix` preserves
orientation. Scalar construction chooses a documented right-handed canonical
frame. Reject zero/negative signed volume under this right-handed contract and
incompatible metrics locally, without silently repairing orientation. Scalar
shape changes retain the rigid orientation relating the current basis to its
canonical shape. A changed setting/origin requires its actual transformation,
never a comparison of just space-group names. [S7, S8]

Represent every operation by exact affine fractional data `g=(R,t)`:

```text
g.operate(f)       = R f + t
g.compose(h)       = (R_g R_h, R_g t_h + t_g)       # g after h
g.inverse()        = (R_g^-1, -R_g^-1 t_g)
e                   = (I, 0)
g.power(0)         = e; negative powers use the same inverse/composition
```

These are one implementation of the standard affine group law. [S7]
The `SymmUnit` ABC has abstract read-only `R`, `t` and `context` primitives; its
sole private concrete `_AffineSymmUnit` supplies them, with all common methods
delegating to the one pure affine-mathematics owner. Named forms validate
or classify that same `(R,t)`; they introduce no independent rotation/screw/glide
coordinate or graph engines. Centring is a translation in this algebra.
A nonzero `t` alone does not classify an operation as a translation or screw:
`(R,(I-R)c)` can be an ordinary rotation about an off-origin axis. Classification
uses the linear part and its invariant translation after the origin shift.
Rotations use crystallographic orders; mirror/inversion are the corresponding
special improper forms, without duplicated algorithms. [S1, S6]

A `from_affine` value supplies an explicit mathematical isometry in its frame;
frame compatibility alone does not establish membership in a declared crystal
group. Cell filling and crystal-supported molecular symmetry use only the
accepted group representatives and their lifts.

Composition, equality, inverse and powers require the same basis/origin/lattice
context. Coordinates in a different setting must first undergo the explicit
registered change. If `f=P u+p`, then
`R_u=P^-1 R P`, `t_u=P^-1(R p+t-p)`; a Cartesian orientation change is accounted
for in `H` as well. This is an explicit conversion in the operation owner, not
an automatic guess or a new general frame framework. [S7]

For declared cell representatives `r_i`, preserve exact closure and the carry:

```text
r_i r_j = T_c(i,j) r_k,        c(i,j) in Z^3
(T_n r_i)(T_m r_j) = T_(n + R_i m + c(i,j)) r_k
reduce_to_cell(g) = (r_k, n),  g = T_n r_k
```

The cell quotient is used only when its integer translation subgroup is
preserved by the operations; validate that requirement. In a centred conventional
cell, `T_cell` is a proper subgroup of the full Bravais translations, so its
finite quotient must not be equated to `G/T` or stripped of centring operations.
Exact carries are retained even if coordinates are later wrapped for display.
A `2_1` screw squares to a nonzero lattice translation in the full group; its
cell representative can have order two. A glide likewise need not have finite
order as a full affine element. [S5, S6]

Require metric compatibility `R.T @ (H.T@H) @ R = H.T@H` to the declared numeric
cell tolerance, and exact group/translation closure, identity and inverses for
an explicit operator set. Metric compatibility alone does not prove that the
set is the declared space group. Broken cell/setting/closure contracts reject
the affected operation locally; they never select P1 as a fallback.

## 3. Source baseline, ownership and connected migration

The inspected Crystal-related source and test observations are attributed to
`ec4d61889a7262df765da29082e405b3c612cd15`. The concurrent parent business commit
`436e027690a0b0f4125c11e3bf3259423c716573` concerns other work; it releases no
Stage15 runtime. The scoped evidence manifest verifies the inspected source
bytes against the earlier baseline. These observations establish missing
capability and migration boundaries, not completed Crystal functionality.

| Actual owner / consumer | Inspected behavior and required migration |
|---|---|
| `structure/crystal.py`, `rules/mathematics/cell.py` | Six writable scalars; `from_matrix` reduces to lengths/angles and loses orientation. The current row conversion calls angle(a,b) alpha and angle(b,c) gamma. Replace once through this stage, with standard-angle release notes and independent oblique/rotated fixtures. Do not automatically swap historical scalar data of unknown construction convention. |
| `Molecule.create_crystal`, `_capture_factual_record`, `_install_factual_record` | Create/compare/capture/install six scalars only. Migrate the actual association, complete oriented context and relations in the same cutover. No adapter call is required for reconstruction. |
| `_builder.MoleculeRecord.cell`, `_state` pickle, `Molecule.copy` / `components`, `operations/topology.py` merge | Record validates exactly six scalars; graph restoration recreates a fresh scalar Crystal; component merge compares scalar cells. One current-version crystal codec must cover basis, context, site/bond images, generated origins and both association directions. Settle the private record/pickle cutover explicitly; no indefinite old/new loader. |
| `_identity.py` cell domain, `_commit.py`, dependency/result owners | Cell identity captures six scalars; rollback retains `_crystal` as a topology field. The new basis, actual symmetry, membership and image relations must enter only the identities/dependencies that consume them. Rollback must restore the relation graph and cell/coordinate state, not just an old pointer whose interior was mutated. |
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
Crystal owns only the context and relations needed for spatial use. Fractional
coordinates, metric, neighbors and matching are derived; no second mutable
coordinate block, topology index, global equivalence label or persistent generic
cache is added. A cell-only six-argument construction remains possible; missing
molecule/group is not an empty molecule or automatically known P1.

Lifecycle migration covers isolated Crystal, Molecule, and a combined reference
graph. Deep copy/pickle must preserve intended internal aliases while isolating
writable facts. Component extraction subsets and remaps site/image relations;
merge requires an explicit compatible context and image map or a selected target,
with atomic rejection otherwise. A cell change preserves Cartesian positions by
default; the fractional-preservation option installs cell and coordinates in one
transaction. Both molecule/context associations, currentness and rollback share
that boundary. Pure reads never move atoms, repair topology or rerun a model.

## 4. Molecular actions, generation and geometry observations

### 4.1 One action on coordinates, graphs and molecule images

`SymmUnit.operate` acts once on batched fractional coordinates and never wraps.
`Crystal.operate` has the narrower fixed-context domain: a validated active
operator set must be available, and the operation must exactly equal `T_n r_i`
for one accepted representative and `n in Z^3`, in that same immutable context.
Its linear action must preserve the cell-image lattice (`R Z^3=Z^3`), as must the
declared group. Unknown group/membership is locally unavailable; a compatible
metric or group name alone does not establish membership. Generated P1/supercell
content uses its active group, so an operation retained only in source origins
is not automatically accepted. The result retains that active cell/group context
with exact integer bond images; this action neither changes basis nor extends
the group. Arbitrary frame-compatible mathematical isometries remain usable
through `SymmUnit.operate` on coordinates.

Within that domain, `Crystal.operate` captures the actual chemical closure, uses
the common action, remaps
existing facts/topology/stereo and publishes fresh contained chemistry. For the
Cartesian linear part use `Q=H R H^-1`; improper actions must transform the
orientation-dependent stereo representation under its existing conventions,
not copy an incompatible label or infer a new chemical state. No scientific
perception or external molecule API is needed.

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
`space_group`, `setting` and `symm_units` describe that active representation;
the original group remains in the origin binding. This is a representation
choice, not a newly inferred crystal space group. Reapplying `unit_cell()` to
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
| 15A — decide the tree and domain | Freeze §1's signatures/layouts/permissions, exact frame/setting and affine/carry rules, matching versus explicit identification, generated P1 representation, and the self/parallel periodic-edge gate. Inventory all scalar/record/identity/native/I/O consumers and the supported scientific domain. Select the bounded CIF/table dependency and its distribution implications. | Parent records exact new public/schema decisions, source identity, prospective checks and cost baseline under §16.4. No unresolved convention or unrepresentable edge reaches production. Stage14D may prepare against this settled schema. |
| 15B — one carrier/context/lifecycle migration | `structure/crystal.py` plus one immutable symmetry-value module; pure cell/affine mathematics; one crystal operation owner; actual Molecule association, builders, codec/pickle, copy/components/merge, identity/dependencies and transaction rollback. Assign exclusive files before writing. | Standard angles/orientation, common-context algebra, no second chemical/cell truth, fresh outputs, complete rollback and alias/remap checks. Preserve native ABI and existing scientific algorithms. |
| 15C — operate, reconstruct and expand | One coordinate/graph action, exact cell representative/carry handling, supported site/image relations and bond images; cell filling, finite molecular-unit extraction, and general integer repeats. | Actual P1, inversion, centring, ordered special positions, screw/glide, crossing-cell finite units and declared network limits; exact origins/counts and no double symmetry application. Publish the implemented domain to Stage14D after 15B/C evidence, without waiting for PyG. |
| 15D — relations and periodic findings | One full coordinate + graph/site matching implementation for candidate relations and molecular-symmetry queries; metric-bounded neighborhoods; lifted existing geometry predicates, bounded finite closed-loop linking and precise controls. | Independent complete/partial/improper/ambiguous matches, crowding, periodic ring/bond and finite loop-linking cases; observations, source state and local failure semantics preserved. No destructive deduplication or global scientific verdict. |
| 15E — pure persistence, CIF and consumer closure | Extend the one pure Hotpot codec and existing data-capture consumers; one justified lazy Gemmi table/CIF adapter if selected, or a separately approved bounded pure CIF path. Close every affected file-format migration before its consumer round trip. Document separate legacy toolkit/native projection limits using prepared values. | Real supported CIF/file/current-version pickle round trips and outside-checkout installed domain evidence, with actual setting/origin, precision, supported site/bond/image relations and loss coverage. Publish each touched format's migration closure to Stage14D. No OB/RDKit/CCDC call is required or permitted in this Crystal route. |
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

Gemmi is a concrete **candidate for lookup/parsing only**, not an already selected
or tested runtime dependency. Before adoption inspect its actual version/API,
license/distribution cost, named setting/origin behavior and supported CIF fields.
Convert its table/CIF values into the exact Hotpot representation once. Gemmi's
documented `Op` multiplication wraps translations into `[0,1)`, so it cannot
implement Hotpot's full affine composition with retained integer carry. Hotpot's
exact rational values are not restricted to Gemmi's denominator 24. Algebra,
coordinate/molecule actions, matching, metric neighbors and topology predicates
remain Hotpot's own implementations. Do not copy an unmaintained 230-group table,
retain an external crystal object as a fact owner, or substitute a licensed
backend. An unavailable lookup is an explicit local unavailability, not P1.
Official documentation describes possible table/CIF boundaries only. [S9, S10]

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
| Affine group | Analytic identity/composition/inverse/negative powers; off-origin rotation with nonzero `t`; screw/glide powers with translation carry; centring retained in a conventional-cell quotient; explicit basis/origin conjugacy; incompatible metric and nonclosed operator-set rejection. |
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

The parent still needs concrete decisions on (a) this proposed tree and exact
value/export changes, including `close_contact_cutoff` replacing the earlier
underspecified `close_contact_policy`; (b) the periodic self/parallel-edge domain
and representation; (c) the one private record/pickle cutover and supported CIF
loss map; and (d) dependency selection and preregistered numerical conventions.
These are bounded implementation-entry decisions, not a request to approve an
unspecified framework. This plan neither approves production nor removes scope.

The independent revision was assigned **GPT-6.1 Sol / max** by the parent; no
independent model-metadata verification is claimed. Its evidence index is
`/tmp/hotpot_plan20/long/stage15-symmunit-plan-review-20261104/crystal_plan_solmax/INDEX.md`.
It records read-only source attribution, primary-document research, 12/12 small
standard-library mathematical planning probes, and the scoped document/link/
whitespace checks. These are design evidence only: no product test, toolkit API,
engine/model, build, performance, scientific-acceptance or Stage15 runtime gate
has been executed or passed by this revision. Parent log/index/main-plan/Git and
all Stage0–12, Plan12, Stage13/14/15 and overall-task gates retain their owners.

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
- Governing local records: [effective goal](../../goal.md), [chemical-type interface protocol](../../chemical_type_interface_contract.md), [Stage14 plan](../stage14/review.md), [main plan §18](../../fact_rule_architecture_implementation_plan.md#18-additional-owner-directed-branches-stage14-and-stage15).
