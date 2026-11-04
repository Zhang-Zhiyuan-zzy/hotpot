# Stage15 implementation plan — Crystal, symmetry and periodic geometry

Status: owner-directed concrete implementation scope registered on 2026-11-04;
plan prepared, runtime authoring not started. Build a usable Crystal aggregate
under the existing fact/rule/operation architecture. A CSD-like object relation
is the modeling direction, not a requirement for licensed CCDC software or an
assertion that every CSD API detail is reproduced.

## 1. Required outcome and current source

A Crystal contains chemical content plus crystallographic parameters. Supplied
atoms are treated as distinct independent content unless an explicit contextual
site relation says otherwise. It can generate unit-cell contents, construct a
`2 × 3 × 2` supercell (and general positive integer dimensions), and report
coincident/very close periodic images with source-atom/operator/translation maps.
It can test which actual crystal operations preserve an entire finite molecule.

At `ec4d61889a7262df765da29082e405b3c612cd15`, `structure/crystal.py` has only six
ordinary writable parameters and `from_matrix`; there is no contained molecule,
space group, cell filling or periodic geometry implementation. Molecule attaches
that cell through `create_crystal` and serializes six scalars in `MoleculeRecord`.
Open Babel gets a fresh six-scalar projection. Current tests deliberately retain
the old row-angle convention and absence of `matrix`/`mol` attributes. These are
real migration boundaries, not evidence that the desired Crystal already exists.

In particular, `_matrix_to_params` currently calls angle(a,b) `alpha` and
angle(b,c) `gamma`. Stage15 adopts standard crystallographic definitions:
`alpha=angle(b,c)`, `beta=angle(a,c)`, `gamma=angle(a,b)`. This is an explicit
business correction with new oblique-cell fixtures, consumer migration and
release notes. Old scalar data do not prove which construction route supplied
them: do not automatically swap historical alpha/gamma without known source
convention. Retain the old observations as history, not as the new oracle.

## 2. Object and ownership contract

The concrete public proposal to settle at Stage15A is:

```python
Crystal(a, b, c, alpha, beta, gamma, *, molecule=None,
        space_group=None, setting=None)
Crystal.from_matrix(matrix, *, molecule=None, space_group=None, setting=None)
Crystal.molecule                 # supplied independent chemical content
Crystal.matrix                   # independent array; lattice vectors are rows
Crystal.space_group              # explicit group/setting or unavailable
Crystal.unit_cell()              # generated Crystal with image-to-source mapping
Crystal.expand(repeats)          # generated supercell Crystal; e.g. (2, 3, 2)
Crystal.geometry_findings(*, coincidence_tolerance, close_contact_policy)
Crystal.molecular_symmetry(molecule=None, *, tolerance)
```

Exact return shapes, mutation permissions, mapping access and tolerances must be
recorded before their dependent production step under main-plan §16.4. The list
is a focused chemical interface, not a public registry or generic result-wrapper
framework. The ordinary returns are the requested structure, mappings or actual
findings; do not attach scientific confidence grades.

| Fact / relation | Sole owner and lifecycle |
|---|---|
| Element, bond kind/direction, H, charge/spin, stereo and Cartesian coordinates | The contained Molecule and existing Atom/Bond owners. Crystal does not retain a second chemical table. |
| Oriented cell basis | Crystal owns one authoritative complete basis. Six lengths/angles are projections and their existing setters, if retained, update that one basis atomically. No stored matrix plus independently mutable six-scalar truth. |
| Space group, setting/origin and actual affine operators | Crystal owns the declared crystallographic context; registered adapters provide named-table lookup, not authoritative molecule facts. A point-group name alone cannot supply translations, screw/glide operations or setting. |
| Supplied independent content | `Crystal.molecule` references its actual Molecule. It may contain disconnected components or a supplied finite molecule on a special position. Do not claim every supplied fragment is a mathematically minimal asymmetric unit or a completed molecule. |
| Site equivalence, occupancy/disorder if supplied | Crystal-context relations keyed to contained atom identities; do not add global equivalence labels to Atom or duplicate elemental facts. Missing occupancy/disorder information remains unknown. |
| Generated sites and periodic bonds | Generated Crystal owns its new content and immutable source-atom/operator/integer-image maps. Original source facts remain unchanged. |
| Fractional positions, cell metrics, neighbor lists and symmetry findings | Read-only derived values with finite source dependencies; no second mutable coordinate truth or permanent global cache. |

A six-positional-argument legacy cell can remain cell-only: unknown chemical
content or group does not imply an empty molecule or automatically known P1.
A real P1 model is explicit. Linking a Molecule and Crystal must settle a single
context identity: the planned direct relationship is
`crystal.molecule is molecule` and `molecule.crystal is crystal`. Attaching the
same molecule to a second independent context requires an explicit copy or
replacement operation; do not silently create competing active parents.
Existing `Molecule.create_crystal` delegates to this one association operation.

## 3. Coordinate and symmetry conventions

Use a documented row-vector public matrix `M=(a;b;c)` and column basis
`H=M.T`. Cartesian column positions satisfy `x=H f`; row positions satisfy
`x_row=f_row @ M`. Lengths are Angstrom and angles degrees. Preserve the full
orientation of a `from_matrix` input; six scalar lengths/angles alone cannot
encode that orientation. Scalar construction chooses a documented canonical
right-handed frame. Negative/zero volume or an incompatible explicit metric is
an actual cell-domain error, not a heuristic chemical judgment.

If scalar setters remain, update the complete cell through one named operation:
retain the current global orientation while constructing the new canonical
shape. Coordinate policy is explicit: changing lattice parameters preserves
Cartesian atom positions by default; callers can instead request preservation
of fractional positions through a complete cell+coordinate operation. There is
no implicit atom movement from a view getter. Cell/context changes invalidate
only dependent periodic/transport results through the owning relation.

A space-group operation acts on fractional coordinates as `f' = R f + t`.
Named group import must resolve the actual setting and origin, retain the exact
rational operator data where supplied, and validate operators against the cell
metric `G=H.T@H`. Preserve operations modulo integer translations without
confusing the infinite space group with its finite quotient for one cell.

The preferred implementation uses Hotpot's pure cell/periodic mathematics and
one lazy crystallographic adapter (Gemmi is the first concrete candidate) for
space-group tables/CIF interpretation. At entry verify its installed API,
license/distribution impact and a maintained dependency path, then record the
selection. Do not hand-copy an unmaintained 230-group table or make CCDC a
mandatory dependency. Explicit operator input must remain attributable; there
is no silent fallback to P1 when group resolution fails.

## 4. Independent sites, coincident images and molecular symmetry

Three concepts remain separate:

1. **Same symmetry orbit:** sites are related by an operation and lattice image;
   they can occupy different Cartesian positions.
2. **Coincident generated images:** transformed coordinates meet a declared
   distance tolerance. This supplies a candidate same-site relation, not
   permission to merge two supplied chemical identities.
3. **Molecular symmetry:** one operation preserves every atom and chemical
   connection of one complete finite molecule under a single bijection.

By default, different supplied Atom records remain different sites. Symmetry
images of the same declared site can be deduplicated under the operator/site-
stabilizer rules with all origins retained. Distinct supplied sites may be
identified only through a validated explicit equivalence relation; coordinate
proximity by itself cannot erase atoms, bonds, charge or occupancy. Contradictory
chemical identities at a proposed same site produce a precise finding or a
relation-commit rejection, as appropriate; no averaging or guessed disorder.

For molecular-symmetry testing, first reconstruct one finite unwrapped molecule
using its bond/image graph. For each candidate affine crystal operation, permit
one common integer lattice translation for the whole molecule and seek a
complete atom bijection preserving element/isotope, declared atom/bond facts,
dative direction and orientation-aware stereo. Compare coordinates with a
stated Angstrom tolerance. Do not independently wrap each atom and mistake the
result for a whole-molecule operation. A pair `g(A)=B` is only a candidate seed.
If the known crystal operation preserves the complete molecular connectivity
and maps a connected independent molecule back into itself, the full mapping
establishes the corresponding molecular symmetry.

For an accepted operation and fixed molecular center `f0`, the Cartesian linear
part is `Q=H R H^-1`. Shift the origin to the fixed center before interpreting
it as a molecular point operation. A nonzero pure translation cannot stabilize
a finite molecule; a nonzero affine `t` by itself does not imply a translation,
since rotations about an off-origin center also have such a term. The retained
operations form the crystal-supported molecular stabilizer/point subgroup,
not necessarily the molecule's full isolated point group. General-position
molecules need not have nontrivial internal symmetry. Disorder/partial occupancy
cannot be passed off as ordered whole-molecule symmetry.

## 5. Cell filling, periodic bonds and supercells

Generate in a stable documented order: supplied site order, normalized operator
order, then integer image order. For each generated site keep its source atom,
operator and translation; preserve contributing aliases when deduplicated.
Boundaries use one half-open fractional cell with a preregistered numerical
boundary tolerance; tests cover zero/one boundaries. Do not lose a complete
molecule merely because its coordinates cross a cell face.

Periodic topology needs endpoint rows plus an integer image vector `s`, so the
bond displacement is `H(f_j+s-f_i)`. Atom-row equality alone cannot represent a
bond into another cell. If wrapping subtracts integer vectors `w_i,w_j`, update
the bond image by `s'=s+w_j-w_i`. Ordinary, dative and supported haptic relations
retain their meaning; contact geometry does not create metal coordination bonds
or introduce Plan21 MetalCluster runtime work.

For repeat dimensions `N=(n1,n2,n3)`, form `H'=H diag(N)` and image fractional
positions `f'=diag(N)^-1(f+k)` for integer cell images `k`. A `2×3×2` expansion
has exactly twelve image cells before any explicitly justified site relation.
Internal atom/bond counts, image multiplicities and boundary-crossing relations
must be accounted for. A generated P1 cell representation is a complete image
projection with its original symmetry/provenance retained as origin metadata;
it is not an independently inferred new space group. The implementation must
choose and document a representation that cannot apply the original symmetry
again accidentally and multiply cell contents twice.

Produce fresh content by default. Share no writable coordinate/bond/result
buffers with the source. Site-specific MCA/partial-charge and conformer results
are transported only through their actual supported mapping/currentness rules;
replication is not a new inference or a silently valid model observation on a
changed periodic structure.

## 6. Periodic geometry findings and control

Compute distances using the actual triclinic metric. Simple per-component
fractional rounding or a fixed 27-image search is not sufficient for arbitrary
skewed bases. Use metric-aware finite image bounds (or a validated lattice
reduction/closest-vector implementation) and compare candidate Cartesian
vectors. For cutoff `r`, reciprocal-basis row norms bound each image-coordinate
range; use those bounds with a cell/neighbor list rather than unbounded images
or an all-pairs expansion for routine large structures.

| Observation / failure | Required interpretation and default response |
|---|---|
| Coincident images within explicit tolerance | Report source sites, operation/image and distance as possible same-site relation; continue and preserve both source facts unless a validated relation authorizes identification. |
| Very close independent sites | Report the method/radius or caller cutoff, units and affected images; useful geometry remains available. It can suggest placement problems but is not universal chemical impossibility. |
| Periodic short bonds or ring/bond piercing | Reuse existing named geometry rules on necessary periodic neighborhoods, avoiding artificial split rings or duplicated findings. Preserve finite structures/findings; declared per-code user breaks remain available. |
| Invalid cell, inconsistent operator metric, broken member/image map | Reject the affected context/operation with its exact cause; do not publish partial state or silently substitute another space group. |
| Unsupported disorder, symmetry relation or adapter information | Preserve available input and issue a scoped loss/unsupported finding. Do not guess equivalence, occupancy, connectivity or molecular identity. |

Thresholds are method/context-specific and fixed before observations. Numerical
coordinate tolerance, a chemical close-contact cutoff and a relation-commit
integrity condition are different contracts. Do not impose energy, spin or
universal bond-length truth on crystal placement. Geometry reports do not run
an FF, xTB or model without the caller requesting that calculation.

## 7. Connected migration and implementation slices

| Slice | Implementation and owners | Required exit |
|---|---|---|
| 15A — exact contracts | Resolve public signatures, row/angle/orientation conventions, cell update policy, contained-Molecule association, site/image relation and generated-cell representation. Inventory every six-scalar record/copy/adapter consumer. | Root decision under §16.4, updated interface and dependency records, independent positive/negative/skew fixtures; no unresolved convention reaches production. |
| 15B — aggregate/cell | `structure/crystal.py` becomes a fact/relation carrier; operations own coordinated writes; mathematics owns conversion/metric rules. Migrate `Molecule.create_crystal`, builder/state/pickle, source identity/invalidation and conversion together. | Atomic complete updates; no duplicate cell/chemical facts or backend object; orientation, copy and alias identities hold. |
| 15C — symmetry and expansion | Lazy named-group adapter, exact operators, site relations, unit-cell/supercell generation and periodic edges. | Actual P1, inversion, special-position, nonsymmorphic, skew-cell and boundary fixtures; correct origins/multiplicities and no double application. Publish the implemented domain contract to Stage14D after 15B/C evidence, without waiting for PyG. |
| 15D — geometry/symmetry | Metric-aware periodic neighbors and findings, finite whole-molecule unwrapping/stabilizer matching. | Independent coincident/near-contact and complete/partial symmetry cases; all controls preserve observations and caller state. |
| 15E — I/O and Stage14 | CIF and existing OB/RDKit projections through documented maps; update copy/components/merge and periodic serialization; publish each affected format's migration evidence to Stage14D. | Actual file/object/installed round trips preserve declared supported facts; unsupported group/disorder/periodic topology losses remain explicit. |
| 15F — delivery | Documentation/examples, migration tables, affected core workflows, representative scaling/resources and applicable package/version evidence. | Detailed report, final API and coherent Git delivery; all required actual functionality verified. |

Candidate owners include `structure/crystal.py`, `structure/molecule.py`,
`structure/_builder.py`, `_state.py`, `_identity.py`, existing commit/dependency
owners, `rules/mathematics/cell.py` and focused periodic mathematics,
`operations/conversion.py`/one crystal operation owner, crystallographic and
Open Babel/I/O adapters, and focused tests/docs. Choose exact file partitions
before authoring; writers share the workspace and cannot revert others.

Lifecycle migration must specifically demonstrate: deep-copy/pickle of Crystal,
Molecule and their combined reference graph preserve the intended aliases;
Molecule copies attach a fresh context instead of pointing to the original;
component extraction subsets site relations; merge either has an explicit
compatible context/map or requires a caller-selected target rather than silently
dropping a lattice. Group/cell/coordinate changes invalidate only actual
periodic/geometry dependencies. Six-scalar legacy private storage has an
explicit documented cutover; do not create an indefinite compatibility loader.

## 8. Acceptance fixtures and delivery

| Fixture family | Required independent business facts |
|---|---|
| Cell mathematics | Orthogonal and triclinic cells, rotated full basis, standard alpha/beta/gamma, Cartesian↔fractional round trips and explicit setter policies. Include a skew case where component rounding picks the wrong nearest image. |
| Site generation | P1, inversion/general positions, special positions, screw/glide and centering, boundary coordinates, supplied equivalence versus proximity, occupancy/disorder preservation where supported. Named group lookup binds exact settings. |
| Molecules | General-position asymmetric molecule; ordered special-position molecule; one-pair coincidence without full symmetry; complete symmetric molecule; chiral/improper operations, crossing-cell finite molecule and periodic network explicitly distinguished. |
| Expansion | `2×3×2` and asymmetric repeat dimensions, exact twelve-cell accounting, bonds crossing faces, stable atom/image joins and alias-free source isolation. |
| Geometry | Exact/near coincidence, close independent contact, periodic ring/bond case, incompatible metric/operator and invalid relation rejection; findings and default controls tested separately. |
| Integration | Current builders/copy/pickle/components/merge and cell adapters, actual CIF fixture round trips and installed package examples. Stage14D owns the shared Crystal↔Data fixture; it is not a reverse prerequisite for Stage15. Keep supported chemistry and trained model meaning intact. |
| Cost/support | Representative finite and periodic structures at increasing sizes, bounded image/neighbor work, actual time/RSS and completion/failure counts. CP311 first; finish applicable declared-version/installed evidence before support claims. |

Geometric tolerance and expected mappings are set before running. Fixture
truth uses analytic cells/operations and reviewed source structures; comparing
two paths that share the same implementation is insufficient. Real-data
observations retain failures, ambiguous relations and placement findings.

Completion requires usable cell filling, supercell expansion, periodic findings,
whole-molecule symmetry checking and an accepted connected ownership/lifecycle
migration, not just a Crystal class or a cell matrix. Stage15 does not depend on
Stage14D: it publishes the domain contract consumed there. Both side branches
start production from the validated `B_exp` and are part of overall current-task
completion; they cannot retroactively waive Stage0–12 or replace Stage13's
separately gated experiment.

## Primary references and limits

- [IUCr crystallographic symmetry and coordinate transformations](https://www.iucr.org/education/pamphlets/9/full-text)
- [IUCr special-position molecular symmetry examples](https://journals.iucr.org/e/issues/2019/12/00/su5523/index.html)
- [Gemmi unit-cell, affine symmetry and nearest-image API; note nearest-image caveats](https://gemmi.readthedocs.io/en/stable/cell.html)
- [CSD Crystal API: molecule, asymmetric-unit molecule, packing and symmetry](https://downloads.ccdc.cam.ac.uk/documentation/API/descriptive_docs/crystal.html)
- [CSD Crystal class reference](https://downloads.ccdc.cam.ac.uk/documentation/API/modules/crystal_api.html)
- [Effective goal](../../goal.md), [Stage14 plan](../stage14/review.md), [main plan](../../fact_rule_architecture_implementation_plan.md)

References describe source conventions/APIs, not an already completed Hotpot
capability. Confirm installed backend/version behavior at implementation entry.
