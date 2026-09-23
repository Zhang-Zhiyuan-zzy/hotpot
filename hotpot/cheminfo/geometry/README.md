# `hotpot.cheminfo.geometry` API Reference

[Chinese version](README.zh.md)

This document describes every public interface in `hotpot.cheminfo.geometry`,
including its mathematical semantics, numerical settings, and rules for
converting chemical objects. It is aligned with the package-level `__all__`;
all public names can be imported from one entry point:

```python
from hotpot.cheminfo import geometry as geo
```

## 1. Package introduction

`geometry` is Hotpot's three-dimensional Euclidean-geometry fact layer. It
provides four groups of capabilities:

1. immutable representations of points, infinite lines, finite segments,
   planes, triangles, and ordered closed cycle boundaries;
2. measurements of planarity, line relationships, distances between finite
   objects, and the projected location of a point relative to a cycle;
3. a three-state relationship between a finite segment and a cycle under an
   explicitly stated spanning-surface model: `PIERCES`, `DOES_NOT_PIERCE`, or
   `UNDETERMINED`;
4. conversion of Hotpot Atom, Bond, Ring, and Molecule objects—or objects that
   satisfy the same structural protocols—into pure geometry objects while
   retaining source-object references and stable keys.

The package reports spatial facts only. It does not decide whether a structure
is chemically reasonable, physically stable, or suitable for a particular
force field. Decisions such as whether to accept a conformer, break a ring
bond, or repair an `UNDETERMINED` case belong to callers such as
`hotpot.cheminfo.forcefields`.

Module responsibilities and dependency direction are:

```text
settings.py  ─┐
object.py    ─┼─> relation.py ─┐
              └───────────────┼─> convert.py
                               └─> package __init__.py
```

- `settings.py`: numerical tolerances and nonplanar-cycle surface-enumeration
  budgets;
- `object.py`: immutable geometric value objects without relationship
  algorithms;
- `relation.py`: measurements and relationship classifications that accept
  only pure geometry objects;
- `convert.py`: the thin adapter layer that understands chemical-object
  structure.

### 1.1 Units and input conventions

- `CLU` denotes the consistent coordinate-length unit selected by the caller.
  Hotpot molecular coordinates normally use Å as the `CLU`.
- All coordinates and length thresholds within one calculation must use the
  same unit.
- Numerical kernels calculate in `float64`.
- Geometry objects may contain non-finite coordinates so that relationship
  functions can explicitly return `NaN` or `UNDETERMINED`.
- A `Cycle` represents only a one-dimensional ordered closed boundary; it does
  not assume a unique interior surface.

### 1.2 Quick examples

#### a. Determine whether a cycle is planar

```pycon
>>> from hotpot.cheminfo import geometry as geo
>>> cycle = geo.Cycle([
...     (0.0, 0.0, 0.0),
...     (2.0, 0.0, 0.0),
...     (2.0, 2.0, 0.0),
...     (0.0, 2.0, 0.0),
... ])
>>> result = geo.measure_planarity(cycle)
>>> result.kind.value, result.maximum_deviation
('planar', 0.0)
```

#### b. Determine whether a segment pierces a nonplanar cycle

```pycon
>>> from hotpot.cheminfo import geometry as geo
>>> cycle = geo.Cycle([
...     (0.0, 0.0, 0.0),
...     (2.0, 0.0, 0.0),
...     (2.0, 2.0, 0.4),
...     (0.0, 2.0, 0.0),
... ])
>>> segment = geo.Segment((0.6, 0.8, -1.0), (0.6, 0.8, 1.0))
>>> result = geo.determine_segment_cycle_relation(segment, cycle)
>>> (
...     geo.measure_planarity(cycle).kind.value,
...     result.state.value,
...     result.surface_model.value,
...     result.surface_evidence.embedded_surface_count,
...     result.surface_evidence.intersecting_surface_count,
... )
('nonplanar', 'pierces', 'vertex_triangulation_family', 2, 2)
```

The output says that the cycle is nonplanar and that the finite segment
pierces both confirmed embedded candidate surfaces, producing `PIERCES`.

#### c. Scan a chemical object

```python
from hotpot.cheminfo import geometry as geo
report = geo.scan_bond_ring_relations(
    mol,
    ring_scope="ligand_skeleton",
    max_ring_size=16,
)

for finding in report.piercings:
    print(finding.target.bond.bond, finding.target.ring.ring)
```

## 2. Public API overview

### 2.1 Settings (4 names)

| Name | Kind | Meaning |
|---|---|---|
| `NumericToleranceSettings` | frozen dataclass | Dimension-aware numerical tolerances and guard-band configuration |
| `SurfaceEnumerationSettings` | frozen dataclass | Budgets for nonplanar-cycle candidate surfaces and predicate calls |
| `GeometrySettings` | frozen dataclass | Aggregate numeric and surface settings |
| `DEFAULT_GEOMETRY_SETTINGS` | instance | Default configuration for every interface with a `settings` parameter |

### 2.2 Geometric value objects (6 names)

| Name | Meaning |
|---|---|
| `Point` | A point in three dimensions |
| `Line` | An infinite line defined by an origin and a direction |
| `Segment` | A finite closed segment with two endpoints |
| `Plane` | An infinite plane defined by a point on it and a unit normal |
| `Triangle` | A triangle formed by three ordered vertices; degeneracy is allowed |
| `Cycle` | A closed one-dimensional boundary formed by at least three ordered vertices |

### 2.3 Relationship vocabulary and result records (16 names)

| Name | Kind | Meaning |
|---|---|---|
| `PlanarityKind` | Enum | Planarity classification |
| `LineRelationKind` | Enum | Spatial relationship between two infinite lines |
| `PointCycleLocation` | Enum | Location of a point in the planar projection of a cycle |
| `SurfaceEmbeddingState` | Enum | Embedding state of a candidate triangulated surface |
| `SurfaceSegmentState` | Enum | Relationship between a segment and one embedded surface |
| `PiercingState` | Enum | Final three-state segment–cycle result |
| `SegmentCycleFeature` | Enum | Confirmed piercing or contact fact |
| `SegmentCycleIndeterminacy` | Enum | Reason a result cannot be classified as piercing or non-piercing |
| `CycleSurfaceModel` | Enum | Surface model used to define the interior of a cycle |
| `PlanarityMeasurement` | frozen dataclass | Planarity classification and SVD measurements |
| `LineRelation` | frozen dataclass | Line relationship, distance, and parallelism measure |
| `PointPairDistance` | frozen dataclass | Distance between one pair in an input point sequence |
| `ClosestCycleEdge` | frozen dataclass | Nearest cycle edge and its distance |
| `SurfaceFamilyEvidence` | frozen dataclass | Counts from nonplanar candidate-surface enumeration and intersection tests |
| `SegmentCycleRelation` | frozen dataclass | Segment–cycle state, evidence, intersections, and numerical configuration |
| `SegmentCycleScreening` | frozen dataclass | State-only segment–cycle result with optional complete relation evidence |

### 2.4 Relationship functions (12 names)

| Name | Purpose |
|---|---|
| `measure_planarity` | Measure how an ordered cycle departs from its best-fit plane |
| `determine_line_relation` | Classify two infinite lines |
| `line_distance` | Return the distance between two infinite lines |
| `point_segment_distance` | Calculate the shortest distance from a point to a finite segment |
| `segment_segment_distance` | Calculate the shortest distance between two finite segments |
| `point_pair_distances` | Enumerate distances for all unordered point pairs |
| `find_point_pairs_below_distance` | Filter point pairs by a caller-supplied threshold |
| `locate_point_in_planar_cycle` | Classify a point in the planar projection of a cycle |
| `iter_segment_cycle_relations` | Lazily classify multiple segments against one cycle |
| `iter_segment_cycle_screenings` | Screen multiple segments with strict AABB broad-phase exclusion |
| `determine_segment_cycle_relation` | Determine whether one finite segment pierces a cycle |
| `closest_cycle_edge` | Find the cycle edge nearest to a target segment |

### 2.5 Conversion vocabulary and result records (12 names)

| Name | Kind | Meaning |
|---|---|---|
| `PairScope` | Literal alias | Atom-pair scope: all, bonded, or nonbonded |
| `RingScope` | Literal alias | Ring scope: full graph or ligand skeleton |
| `AtomGeometry` | frozen generic dataclass | Mapping among an Atom, a `Point`, and an atom key |
| `AtomPairTarget` | frozen generic dataclass | Two atom mappings and their bonded status |
| `BondGeometry` | frozen generic dataclass | Mapping among a Bond, a `Segment`, and a bond key |
| `RingGeometry` | frozen generic dataclass | Mapping among a Ring, a `Cycle`, and a canonical ring key |
| `BondRingTarget` | frozen generic dataclass | One candidate Ring × Bond combination |
| `AtomPairDistance` | frozen generic dataclass | Source atom pair and pure-geometry distance record |
| `BondRingFinding` | frozen generic dataclass | Source Ring × Bond pair and segment–cycle relation |
| `RingEdgeDistance` | frozen generic dataclass | Source ring bond and nearest-edge distance record |
| `BondRingScanReport` | frozen generic dataclass | Dense scan report for a selected ring scope |
| `BondRingScreeningReport` | frozen generic dataclass | Sparse all-pair piercing screen with coverage and AABB counters |

### 2.6 Conversion functions (13 names)

| Name | Purpose |
|---|---|
| `point_from_atom` | Convert an Atom-like object to a `Point` |
| `segment_from_bond` | Convert a Bond-like object to a `Segment` |
| `cycle_from_ring` | Convert a Ring-like object to a `Cycle` |
| `iter_atom_geometries` | Yield source-to-geometry mappings atom by atom |
| `iter_atom_pair_targets` | Yield atom pairs according to `PairScope` |
| `iter_ring_geometries` | Yield ring mappings according to `RingScope` and ring size |
| `iter_bond_ring_targets` | Yield candidate Ring × Bond pairs |
| `measure_atom_pair_distances` | Measure atom-pair distances while retaining source objects |
| `determine_bond_ring_relation` | Classify one chemical Ring × Bond pair |
| `iter_bond_ring_findings` | Lazily scan Ring × Bond pairs in a requested scope |
| `scan_bond_ring_relations` | Return a complete dense scan report for a requested scope |
| `screen_bond_ring_relations` | Screen a complete scope while retaining only actionable findings |
| `determine_bond_ring_piercing_state` | Aggregate a whole molecule to a three-state piercing result with early exit |

## 3. Mathematical symbols and numerical conventions

The following symbols retain the same meaning throughout this document:

| Symbol | Meaning |
|---|---|
| `CLU` | The consistent coordinate-length unit selected by the caller |
| $\mathbf p$, $\mathbf q$, $\mathbf x$ | A point or coordinate vector in three dimensions |
| $\mathcal L_i$ | The $i$th infinite line |
| $\mathbf o_i$, $\mathbf d_i$ | Origin and direction vector of $\mathcal L_i$ |
| $S=[\mathbf a,\mathbf b]$ | A finite segment with endpoints $\mathbf a$ and $\mathbf b$ |
| $\Pi(\mathbf q,\hat{\mathbf n})$ | The plane through $\mathbf q$ with unit normal $\hat{\mathbf n}$ |
| $T=(\mathbf a,\mathbf b,\mathbf c)$ | An ordered triangle |
| $\mathcal C=(\mathbf v_0,\ldots,\mathbf v_{m-1})$ | A cycle boundary with $m$ ordered vertices |
| $e_i=[\mathbf v_i,\mathbf v_{(i+1)\bmod m}]$ | The $i$th cycle edge |
| $L$ | Local length scale of the current predicate |
| $\epsilon_L,\epsilon_A,\epsilon_V$ | Length-, area-, and volume-dimensional tolerances |
| $\epsilon_u$ | Dimensionless parameter tolerance |
| $g$ | Numerical guard multiplier `predicate_guard_factor` |
| $\mathbf x_\cap$ | A calculated intersection point |

### 3.1 Local scale

For the point set $X$, segment set $E$, and optional cycle $\mathcal C$
actually used by the current predicate, the local scale is:

$L=\max\left(\operatorname{diam}(X),\max_{e\in E}\lVert e\rVert,\operatorname{median}_{e_i\in\partial\mathcal C}\lVert e_i\rVert\right)$

Terms that do not exist are omitted. The scale does not inspect the bounding
box of the entire molecule, so distant unrelated atoms cannot change a local
classification.

### 3.2 Derived tolerances

Let $\epsilon_{\mathrm{machine}}$ be float64 machine epsilon:

$\epsilon_r=\max(\epsilon_{\mathrm{rel}},k_{\mathrm{mach}}\epsilon_{\mathrm{machine}})$

$\epsilon_L=\epsilon_{\mathrm{abs}}+\epsilon_rL$

$\epsilon_u=\epsilon_{\mathrm{param}}+\epsilon_L/L\qquad(L>0)$

$\epsilon_A=\epsilon_LL$

$\epsilon_V=\epsilon_LL^2$

$\epsilon_{\mathrm{AABB}}=k_{\mathrm{AABB}}\epsilon_L$

$\epsilon_{\mathrm{merge}}=k_{\mathrm{merge}}\epsilon_L$

Here $\epsilon_L$, $\epsilon_A$, and $\epsilon_V$ are compared only with
quantities having dimensions of length, area, and volume, respectively;
$\epsilon_u$ is compared only with parameters or normalized quantities.

### 3.3 Guard bands and three-state predicates

For a residual $r$ and a tolerance $\epsilon$ of the same dimension:

- $|r|\le\epsilon$: contact or a value on the defined boundary;
- $\epsilon<|r|\le g\epsilon$: the numerical indeterminacy guard band;
- $|r|>g\epsilon$: a sign or separation may be classified stably.

When the $[0,1]$ parameter domain must be separated into endpoint, interior,
and exterior regions, $g\epsilon_u<1/2$ is also required. Otherwise, the
relationship is `UNDETERMINED` and records `TOLERANCE_DOMAIN`.

## 4. Settings API

### 4.1 `NumericToleranceSettings`

```python
NumericToleranceSettings(
    absolute_length: float = 1.0e-8,
    relative_length: float = 1.0e-10,
    parameter: float = 1.0e-10,
    machine_epsilon_factor: float = 64.0,
    predicate_guard_factor: float = 4.0,
    planarity_factor: float = 1.0,
    winding_residual: float = 1.0e-10,
    intersection_merge_factor: float = 4.0,
    aabb_padding_factor: float = 4.0,
)
```

This frozen configuration object defines the numerical tolerances used by all
relationship kernels.

| Parameter | Symbol | Default | Unit/valid range | Purpose |
|---|---|---:|---|---|
| `absolute_length` | $\epsilon_{\mathrm{abs}}$ | $10^{-8}$ | `CLU`, $>0$ | Absolute length resolution and degeneracy baseline |
| `relative_length` | $\epsilon_{\mathrm{rel}}$ | $10^{-10}$ | dimensionless, $\ge0$ | Relative resolution that grows with local scale |
| `parameter` | $\epsilon_{\mathrm{param}}$ | $10^{-10}$ | dimensionless, $0<\epsilon_{\mathrm{param}}<1/(2g)$ | Tolerance for segment parameters, barycentric coordinates, and angular measures |
| `machine_epsilon_factor` | $k_{\mathrm{mach}}$ | $64$ | dimensionless, $\ge1$ | Multiplier for the float64 round-off lower bound |
| `predicate_guard_factor` | $g$ | $4$ | dimensionless, $>1$ | Guard multiplier separating determined and indeterminate regions |
| `planarity_factor` | $k_{\mathrm{plane}}$ | $1$ | dimensionless, $>0$ | Multiplier of $\epsilon_L$ for the planarity-deviation threshold |
| `winding_residual` | $\epsilon_{\mathrm{winding}}$ | $10^{-10}$ | dimensionless, $0<\epsilon_{\mathrm{winding}}<1/2$ | Allowed winding-number residual from $0$ or $\pm1$ |
| `intersection_merge_factor` | $k_{\mathrm{merge}}$ | $4$ | dimensionless, $\ge1$ | Merge numerically identical intersection points |
| `aabb_padding_factor` | $k_{\mathrm{AABB}}$ | $4$ | dimensionless, $\ge g$ | Safe padding for stable AABB separation checks |

Every parameter must be finite. Construction with invalid settings raises
`ValueError`.

### 4.2 `SurfaceEnumerationSettings`

```python
SurfaceEnumerationSettings(
    maximum_cycle_vertices: int = 8,
    maximum_surface_count: int = 132,
    maximum_segment_triangle_tests: int = 792,
    maximum_triangle_pair_tests: int = 1980,
)
```

This frozen configuration object bounds the cost of complete candidate-surface
enumeration for nonplanar cycles.

| Parameter | Symbol | Default | Valid range | Purpose |
|---|---|---:|---|---|
| `maximum_cycle_vertices` | $B_{\mathrm{vertex}}$ | $8$ | integer, $\ge3$ | Largest cycle for which complete vertex triangulation is allowed |
| `maximum_surface_count` | $B_{\mathrm{surface}}$ | $132$ | positive integer | Candidate-surface budget; $132=\operatorname{Cat}_6$ |
| `maximum_segment_triangle_tests` | $B_{S\triangle}$ | $792$ | positive integer | Segment–triangle predicate budget for one segment–cycle relationship |
| `maximum_triangle_pair_tests` | $B_{\triangle\triangle}$ | $1980$ | positive integer | Triangle-pair predicate budget while constructing candidate surfaces |

Exhausting a budget never produces a definite conclusion from partial
evidence. The relationship records incomplete evidence and returns
`UNDETERMINED`.

### 4.3 `GeometrySettings`

```python
GeometrySettings(
    tolerance: NumericToleranceSettings = NumericToleranceSettings(),
    surface: SurfaceEnumerationSettings = SurfaceEnumerationSettings(),
)
```

Combines numerical tolerances and surface budgets in one immutable
configuration. Both fields are created with `default_factory`, so instances do
not share mutable state.

### 4.4 `DEFAULT_GEOMETRY_SETTINGS`

```python
DEFAULT_GEOMETRY_SETTINGS: GeometrySettings
```

The module-level default instance. Every public relationship and conversion
interface with a `settings` argument refers to this object by default. Custom
configuration normally uses `dataclasses.replace()`:

```python
from dataclasses import replace
from hotpot.cheminfo import geometry as geo

settings = replace(
    geo.DEFAULT_GEOMETRY_SETTINGS,
    tolerance=replace(
        geo.DEFAULT_GEOMETRY_SETTINGS.tolerance,
        absolute_length=1.0e-7,
    ),
)
```

## 5. Geometric value objects

All classes below are frozen dataclasses. In signatures, `PointInput` means
`Point | Iterable[float]`; a three-dimensional coordinate is converted to
three `float` values. A coordinate sequence whose length is not 3 raises
`ValueError`.

### 5.1 `Point`

```python
Point(coordinates: Iterable[float])
Point.from_coordinates(coordinates: Iterable[float]) -> Point
```

Represents a point $\mathbf p=(p_x,p_y,p_z)$ in three dimensions.

- Field: `coordinates: tuple[float, float, float]`.
- Properties: `x`, `y`, and `z`.
- Iteration yields `x, y, z` in that order.
- `from_coordinates()` has the same semantics as the constructor and provides
  an explicitly named constructor.

### 5.2 `Line`

```python
Line(origin: PointInput, direction: Iterable[float])
Line.from_points(first: PointInput, second: PointInput) -> Line
```

Represents the infinite line:

$\mathcal L(\mathbf o,\mathbf d)=\{\mathbf o+t\mathbf d\mid t\in\mathbb R\}$

- Fields: `origin: Point` and
  `direction: tuple[float, float, float]`.
- `from_points(first, second)` sets $\mathbf o=\mathbf p_1$ and
  $\mathbf d=\mathbf p_2-\mathbf p_1$.
- The constructor neither normalizes the direction nor rejects a zero
  direction. Relationship functions report a zero direction as `DEGENERATE`.

### 5.3 `Segment`

```python
Segment(start: PointInput, end: PointInput)
```

Represents the finite closed segment:

$S=[\mathbf a,\mathbf b]=\{\mathbf a+t(\mathbf b-\mathbf a)\mid t\in[0,1]\}$

- Fields: `start: Point` and `end: Point`.
- `direction = end - start`.
- `length = ||end - start||`.
- Identical endpoints are allowed, but relationship functions classify the
  resulting segment as degenerate.

### 5.4 `Plane`

```python
Plane(point: PointInput, normal: Iterable[float])
```

Represents the plane:

$\Pi(\mathbf q,\hat{\mathbf n})=\{\mathbf x\mid(\mathbf x-\mathbf q)\cdot\hat{\mathbf n}=0\}$

- Fields: `point: Point` and
  `normal: tuple[float, float, float]`.
- A finite nonzero normal is normalized to $\hat{\mathbf n}$ during
  construction.
- A zero normal raises `ValueError`.

### 5.5 `Triangle`

```python
Triangle(first: PointInput, second: PointInput, third: PointInput)
```

Represents the ordered triangle $T=(\mathbf a,\mathbf b,\mathbf c)$.

- Fields: `first`, `second`, and `third`.
- `vertices -> tuple[Point, Point, Point]`.
- `edges -> tuple[Segment, Segment, Segment]` in the order $ab,bc,ca$.
- Collinear or coincident vertices are allowed; downstream predicates report
  degeneracy.

### 5.6 `Cycle`

```python
Cycle(vertices: Iterable[PointInput])
```

Represents an ordered closed boundary
$\mathcal C=(\mathbf v_0,\ldots,\mathbf v_{m-1})$ whose closed edges are:

$e_i=[\mathbf v_i,\mathbf v_{(i+1)\bmod m}]$

- Field: `vertices: tuple[Point, ...]`.
- `edges -> tuple[Segment, ...]`.
- `len(cycle)` returns the vertex count $m$.
- Iteration yields vertices in boundary order.
- Fewer than three vertices raises `ValueError`.
- A `Cycle` defines only the boundary. Relationship functions separately
  construct a planar polygon or nonplanar candidate surfaces.
- A `Cycle` neither perceives rings from a graph nor stores whether its
  boundary came from Relevant Cycles, a cycle basis, or another source.

## 6. Relationship vocabulary and result structures

### 6.1 `PlanarityKind`

| Member | `.value` | Meaning |
|---|---|---|
| `PLANAR` | `"planar"` | The maximum out-of-plane deviation lies stably within the planarity threshold |
| `NONPLANAR` | `"nonplanar"` | The maximum out-of-plane deviation lies stably beyond the guard band |
| `DEGENERATE` | `"degenerate"` | Point-set scale is too small or approximate rank is below 2 |
| `UNDETERMINED` | `"undetermined"` | Input is non-finite or the result lies in a numerical guard band |

### 6.2 `LineRelationKind`

| Member | `.value` | Meaning |
|---|---|---|
| `INTERSECTING` | `"intersecting"` | Two nonparallel infinite lines intersect |
| `PARALLEL` | `"parallel"` | Directions are parallel and the lines are separated |
| `COINCIDENT` | `"coincident"` | The two infinite lines coincide |
| `SKEW` | `"skew"` | The two three-dimensional lines are neither parallel nor intersecting |
| `DEGENERATE` | `"degenerate"` | At least one direction vector is zero |
| `UNDETERMINED` | `"undetermined"` | Input is non-finite or the relationship lies in a guard band |

### 6.3 `PointCycleLocation`

| Member | `.value` | Meaning |
|---|---|---|
| `INTERIOR` | `"interior"` | The projected point lies stably inside the simple polygon |
| `BOUNDARY` | `"boundary"` | The projected point lies within tolerance of the cycle boundary |
| `EXTERIOR` | `"exterior"` | The projected point lies stably outside |
| `UNDETERMINED` | `"undetermined"` | The projected cycle is not simple, input is non-finite, or the point lies in a guard band |

### 6.4 `SurfaceEmbeddingState`

Public vocabulary for construction of a candidate nonplanar surface. No public
function currently returns this Enum directly; its statistics are aggregated
in `SurfaceFamilyEvidence`.

| Member | `.value` | Meaning |
|---|---|---|
| `EMBEDDED` | `"embedded"` | The triangulated surface has no extra self-intersection and its boundary equals the original cycle |
| `PROVEN_NON_EMBEDDED` | `"proven_non_embedded"` | A degenerate triangle, extra overlap, or self-intersection has been proved stably |
| `CONSTRUCTION_UNDETERMINED` | `"construction_undetermined"` | Numerical evidence is insufficient to decide whether the candidate surface is embedded |

### 6.5 `SurfaceSegmentState`

Public vocabulary for the relationship between a segment and one embedded
surface. Public functions aggregate it into the final `PiercingState`.

| Member | `.value` | Meaning |
|---|---|---|
| `INTERSECTING` | `"intersecting"` | The open segment is confirmed to cross the surface interior |
| `NON_PIERCING` | `"non_piercing"` | Required predicates completed and no strict crossing occurred |
| `EVALUATION_UNDETERMINED` | `"evaluation_undetermined"` | At least one required intersection predicate is indeterminate |

### 6.6 `PiercingState`

| Member | `.value` | Meaning |
|---|---|---|
| `PIERCES` | `"pierces"` | The finite segment is confirmed to cross the interior under the stated, completely checked surface model |
| `DOES_NOT_PIERCE` | `"does_not_pierce"` | No strict interior crossing exists under the stated, completely checked surface model |
| `UNDETERMINED` | `"undetermined"` | Degeneracy, a tolerance guard band, surface construction, or lack of candidate-surface consensus prevents a binary result |

This state is not a conclusion about chemical reasonableness.

### 6.7 `SegmentCycleFeature`

`SegmentCycleRelation.features` is a set of coexisting spatial facts. Endpoint
or boundary contacts can therefore be present even when the final state is
`PIERCES`.

| Member | `.value` | Meaning |
|---|---|---|
| `TRANSVERSE_INTERIOR` | `"transverse_interior"` | The open finite segment crosses the surface interior |
| `LINE_EXTENSION_INTERIOR` | `"line_extension_interior"` | At least one evaluated surface or triangle has an interior intersection only on the segment's infinite extension; the aggregate result can contain this together with other features |
| `CYCLE_EDGE_CONTACT` | `"cycle_edge_contact"` | Contact occurs in a cycle-edge interior |
| `CYCLE_VERTEX_CONTACT` | `"cycle_vertex_contact"` | Contact occurs at a cycle vertex |
| `SEGMENT_ENDPOINT_CONTACT` | `"segment_endpoint_contact"` | An intersection lies at an endpoint of the target segment |
| `COPLANAR_CONTACT` | `"coplanar_contact"` | The finite segment has coplanar contact with the cycle region or a triangle |

### 6.8 `SegmentCycleIndeterminacy`

| Member | `.value` | Meaning |
|---|---|---|
| `NONFINITE_INPUT` | `"nonfinite_input"` | Input contains `NaN` or an infinite value |
| `NUMERIC_BAND` | `"numeric_band"` | A required predicate lies in a numerical guard band |
| `TOLERANCE_DOMAIN` | `"tolerance_domain"` | The derived parameter tolerance cannot divide $[0,1]$ stably |
| `DEGENERATE_CYCLE` | `"degenerate_cycle"` | The cycle's local scale or rank is degenerate |
| `DEGENERATE_SEGMENT` | `"degenerate_segment"` | The target segment length is degenerate |
| `DEGENERATE_TRIANGLE` | `"degenerate_triangle"` | A candidate surface contains a degenerate triangle |
| `SELF_INTERSECTION` | `"self_intersection"` | A planar-cycle projection is confirmed to self-intersect |
| `SURFACE_DISAGREEMENT` | `"surface_disagreement"` | Valid candidate surfaces disagree about piercing, or no embedded surface can form a consensus and no more specific cause applies |
| `INCOMPLETE_SURFACE_FAMILY` | `"incomplete_surface_family"` | A vertex limit or computation budget makes enumeration/evaluation incomplete |
| `SURFACE_CONSTRUCTION` | `"surface_construction"` | At least one candidate surface has an undetermined embedding state |

### 6.9 `CycleSurfaceModel`

| Member | `.value` | Meaning |
|---|---|---|
| `PLANAR_POLYGON` | `"planar_polygon"` | The simple polygon interior on the best-fit plane |
| `VERTEX_TRIANGULATION_FAMILY` | `"vertex_triangulation_family"` | Consensus of all valid vertex triangulations of a nonplanar cycle |

### 6.10 `PlanarityMeasurement`

```python
PlanarityMeasurement(
    kind: PlanarityKind,
    centroid: Point,
    normal: tuple[float, float, float] | None,
    singular_values: tuple[float, float, float],
    maximum_deviation: float,
    rms_deviation: float,
    length_scale: float,
    length_tolerance: float,
)
```

| Field | Meaning |
|---|---|
| `kind` | Planarity classification |
| `centroid` | Cycle-vertex centroid; coordinates are `NaN` for non-finite input |
| `normal` | Unit normal of the best-fit plane; may be `None` for degenerate/indeterminate input and its sign is not fixed |
| `singular_values` | Three singular values $\sigma_1,\sigma_2,\sigma_3$ in descending order |
| `maximum_deviation` | Maximum absolute out-of-plane vertex deviation; `NaN` if undefined |
| `rms_deviation` | RMS out-of-plane vertex deviation; `NaN` if undefined |
| `length_scale` | Local scale $L$ of the current cycle |
| `length_tolerance` | Derived $\epsilon_L$ of the current cycle |

### 6.11 `LineRelation`

```python
LineRelation(
    kind: LineRelationKind,
    distance: float | None,
    parallel_measure: float,
)
```

| Field | Meaning |
|---|---|
| `kind` | Classification of the two infinite lines |
| `distance` | Shortest distance when determined; `None` when degenerate or indeterminate |
| `parallel_measure` | $q_{\parallel}=\lVert\hat{\mathbf d}_1\times\hat{\mathbf d}_2\rVert$; `NaN` if it cannot be calculated |

### 6.12 `PointPairDistance`

```python
PointPairDistance(first_index: int, second_index: int, distance: float)
```

`first_index` and `second_index` are positions in the input
`Sequence[Point]`, always satisfying `first_index < second_index`; they are not
the `idx` values of chemical Atom objects. `distance` is `NaN` for a
non-finite pair.

### 6.13 `ClosestCycleEdge`

```python
ClosestCycleEdge(edge_index: int, edge: Segment, distance: float)
```

Records the zero-based index of the nearest cycle edge, the corresponding
`Segment`, and its shortest distance to the target segment.

### 6.14 `SurfaceFamilyEvidence`

```python
SurfaceFamilyEvidence(
    enumeration_complete: bool,
    enumerated_surface_count: int,
    embedded_surface_count: int,
    proven_non_embedded_surface_count: int,
    construction_undetermined_count: int,
    intersecting_surface_count: int,
    non_piercing_surface_count: int,
    evaluation_undetermined_count: int,
    segment_triangle_tests_used: int,
    triangle_pair_tests_used: int,
)
```

| Field | Meaning |
|---|---|
| `enumeration_complete` | Candidate-family construction/enumeration and all segment evaluations required for this result completed without exceeding a budget |
| `enumerated_surface_count` | Number of candidate surfaces submitted to construction classification |
| `embedded_surface_count` | Number of confirmed embedded candidate surfaces |
| `proven_non_embedded_surface_count` | Number of confirmed non-embedded surfaces excluded from consensus |
| `construction_undetermined_count` | Number of candidate surfaces whose embedding state is undetermined |
| `intersecting_surface_count` | Number of embedded surfaces the segment is confirmed to pierce |
| `non_piercing_surface_count` | Number of embedded surfaces the segment is confirmed not to pierce |
| `evaluation_undetermined_count` | Number of embedded surfaces with an indeterminate segment relationship |
| `segment_triangle_tests_used` | Number of segment–triangle predicates used |
| `triangle_pair_tests_used` | Number of triangle-pair embedding predicates used |

A complete nonplanar enumeration satisfies:

$n_{\mathrm{enum}}=n_{\mathrm{emb}}+n_{\mathrm{nonemb}}+n_{\mathrm{construct\_undetermined}}$

$n_{\mathrm{emb}}=n_{\mathrm{hit}}+n_{\mathrm{miss}}+n_{\mathrm{eval\_undetermined}}$

### 6.15 `SegmentCycleRelation`

```python
SegmentCycleRelation(
    state: PiercingState,
    features: frozenset[SegmentCycleFeature],
    indeterminacy_causes: frozenset[SegmentCycleIndeterminacy],
    surface_model: CycleSurfaceModel | None,
    intersection_points: tuple[Point, ...],
    closest_boundary_edge: ClosestCycleEdge | None,
    surface_evidence: SurfaceFamilyEvidence,
    settings: GeometrySettings,
)
```

| Field | Meaning |
|---|---|
| `state` | Final three-state relationship |
| `features` | All confirmed piercing, line-extension, and contact facts |
| `indeterminacy_causes` | Indeterminate or incomplete evidence observed during classification; it may be nonempty even if `state` has a definite consensus |
| `surface_model` | Selected surface model, or `None` when no model can be selected |
| `intersection_points` | Discrete finite-segment intersection/contact points explicitly returned by kernels and merged within $\epsilon_{\mathrm{merge}}$; excludes coplanar contact regions and line-extension intersections |
| `closest_boundary_edge` | Nearest cycle edge, or `None` if input is non-finite or scale cannot be compared |
| `surface_evidence` | Candidate-surface and intersection-budget evidence; always present |
| `settings` | Configuration actually used for this classification |

### 6.16 `SegmentCycleScreening`

```python
SegmentCycleScreening(
    state: PiercingState,
    relation: SegmentCycleRelation | None,
    aabb_separated: bool,
    surface_complete: bool,
)
```

A state-only result used by the broad-phase API. `relation` is omitted only
when guarded finite-segment and cycle AABBs strictly separate after the cycle
surface model has been validated. Such separation proves
`DOES_NOT_PIERCE`; boundary and tolerance-band cases use the complete kernel.

## 7. Relationship functions

### 7.1 `measure_planarity`

```python
def measure_planarity(
    cycle: Cycle,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> PlanarityMeasurement
```

Fits a best plane to the cycle vertices by SVD. The centroid and centered
matrix are:

$\mathbf c=\frac1m\sum_{i=0}^{m-1}\mathbf v_i$

$X=[\mathbf v_i-\mathbf c]$

$X=U\Sigma V^{\mathsf T},\qquad \sigma_1\ge\sigma_2\ge\sigma_3$

The right singular vector associated with $\sigma_3$ is the best-fit-plane
normal $\hat{\mathbf n}$. Out-of-plane deviations are:

$h_i=|(\mathbf v_i-\mathbf c)\cdot\hat{\mathbf n}|$

$h_{\max}=\max_i h_i$

$h_{\mathrm{rms}}=\sqrt{\frac1m\sum_i h_i^2}$

Define the rank ratio $q_{\mathrm{rank}}=\sigma_2/\sigma_1$. The decisive
classification rules are:

- $L\le\epsilon_{\mathrm{abs}}$: `DEGENERATE`;
- $\sigma_1\le\epsilon_L$: `DEGENERATE`;
- $\epsilon_L<\sigma_1\le g\epsilon_L$: `UNDETERMINED`;
- $q_{\mathrm{rank}}\le\epsilon_u$: `DEGENERATE`;
- $\epsilon_u<q_{\mathrm{rank}}\le g\epsilon_u$: `UNDETERMINED`;
- $h_{\max}\le k_{\mathrm{plane}}\epsilon_L$: `PLANAR`;
- $h_{\max}>gk_{\mathrm{plane}}\epsilon_L$: `NONPLANAR`;
- $k_{\mathrm{plane}}\epsilon_L<h_{\max}\le gk_{\mathrm{plane}}\epsilon_L$:
  `UNDETERMINED`.

Returns `PlanarityMeasurement`; it makes no aromaticity or ring-reasonableness
decision.

#### Example

```pycon
>>> from hotpot.cheminfo import geometry as geo
>>> cycle = geo.Cycle([(0, 0, 2), (2, 0, 2), (2, 2, 2), (0, 2, 2)])
>>> result = geo.measure_planarity(cycle)
>>> result.kind.value, result.maximum_deviation
('planar', 0.0)
```

### 7.2 `determine_line_relation`

```python
def determine_line_relation(
    first: Line,
    second: Line,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> LineRelation
```

For $\mathcal L_i=\mathbf o_i+t\mathbf d_i$, the implementation first obtains
unit directions $\hat{\mathbf d}_i$ by a scale-safe method and then calculates:

$q_{\parallel}=\lVert\hat{\mathbf d}_1\times\hat{\mathbf d}_2\rVert$

The distance for parallel directions is:

$d_{\parallel}=\lVert(\mathbf o_2-\mathbf o_1)\times\hat{\mathbf d}_1\rVert$

The three-dimensional shortest distance for nonparallel directions is:

$d_{\mathrm{skew}}=\frac{|(\mathbf o_2-\mathbf o_1)\cdot(\hat{\mathbf d}_1\times\hat{\mathbf d}_2)|}{\lVert\hat{\mathbf d}_1\times\hat{\mathbf d}_2\rVert}$

The angular and distance tolerances for this operation are:

$\epsilon_\theta=\max(\epsilon_{\mathrm{param}},k_{\mathrm{mach}}\epsilon_{\mathrm{machine}})$

$\epsilon_d=\epsilon_{\mathrm{abs}}+\epsilon_r d$

The actual classification boundaries are:

- either direction is zero: `DEGENERATE`;
- $q_{\parallel}=0$ and $d_{\parallel}\le\epsilon_d$: `COINCIDENT`;
- $q_{\parallel}=0$ and $d_{\parallel}>g\epsilon_d$: `PARALLEL`;
- $0<q_{\parallel}\le g\epsilon_\theta$: `UNDETERMINED`;
- $q_{\parallel}>g\epsilon_\theta$ and
  $d_{\mathrm{skew}}\le\epsilon_d$: `INTERSECTING`;
- $q_{\parallel}>g\epsilon_\theta$ and
  $d_{\mathrm{skew}}>g\epsilon_d$: `SKEW`;
- any remaining distance guard band or non-finite input: `UNDETERMINED`.

Returns `LineRelation`.

#### Example

```pycon
>>> from hotpot.cheminfo import geometry as geo
>>> first = geo.Line((0, 0, 0), (1, 0, 0))
>>> second = geo.Line((0, 1, 1), (0, 1, 0))
>>> result = geo.determine_line_relation(first, second)
>>> result.kind.value, result.distance, result.parallel_measure
('skew', 1.0, 1.0)
```

### 7.3 `line_distance`

```python
def line_distance(
    first: Line,
    second: Line,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> float
```

A scalar projection of `determine_line_relation()`: returns
$d_{\parallel}$ or $d_{\mathrm{skew}}$ when determined and `NaN` when
degenerate or indeterminate.

#### Example

```pycon
>>> from hotpot.cheminfo import geometry as geo
>>> first = geo.Line((0, 0, 0), (1, 0, 0))
>>> second = geo.Line((0, 1, 0), (1, 0, 0))
>>> geo.line_distance(first, second)
1.0
```

### 7.4 `point_segment_distance`

```python
def point_segment_distance(
    point: Point,
    segment: Segment,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> float
```

For point $\mathbf p$ and segment $S=[\mathbf a,\mathbf b]$:

$t=\operatorname{clip}_{[0,1]}\frac{(\mathbf p-\mathbf a)\cdot(\mathbf b-\mathbf a)}{\lVert\mathbf b-\mathbf a\rVert^2}$

$d(\mathbf p,S)=\lVert\mathbf p-[\mathbf a+t(\mathbf b-\mathbf a)]\rVert$

If segment length does not exceed the current $\epsilon_L$, the function
returns $\lVert\mathbf p-\mathbf a\rVert$. Non-finite input returns `NaN`.

#### Example

```pycon
>>> from hotpot.cheminfo import geometry as geo
>>> point = geo.Point((1, 1, 0))
>>> segment = geo.Segment((0, 0, 0), (2, 0, 0))
>>> geo.point_segment_distance(point, segment)
1.0
```

### 7.5 `segment_segment_distance`

```python
def segment_segment_distance(
    first: Segment,
    second: Segment,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> float
```

For $S_1=[\mathbf a,\mathbf b]$ and $S_2=[\mathbf c,\mathbf d]$:

$d(S_1,S_2)=\min_{s,t\in[0,1]}\lVert\mathbf a+s(\mathbf b-\mathbf a)-\mathbf c-t(\mathbf d-\mathbf c)\rVert$

Degenerate segments are handled as point–segment or point–point distances.
Non-finite input returns `NaN`.

#### Example

```pycon
>>> from hotpot.cheminfo import geometry as geo
>>> horizontal = geo.Segment((0, 0, 0), (1, 0, 0))
>>> vertical = geo.Segment((0.5, -1, 0), (0.5, 1, 0))
>>> geo.segment_segment_distance(horizontal, vertical)
0.0
```

### 7.6 `point_pair_distances`

```python
def point_pair_distances(
    points: Sequence[Point],
) -> tuple[PointPairDistance, ...]
```

Returns all unordered pairs in stable $i<j$ order:

$d_{ij}=\lVert\mathbf p_i-\mathbf p_j\rVert$

There are $n(n-1)/2$ records. This function does not read
`GeometrySettings`; a non-finite pair has distance `NaN`.

#### Example

```pycon
>>> from hotpot.cheminfo import geometry as geo
>>> points = tuple(geo.Point((x, 0, 0)) for x in (0, 1, 3))
>>> result = geo.point_pair_distances(points)
>>> [(item.first_index, item.second_index, item.distance) for item in result]
[(0, 1, 1.0), (0, 2, 3.0), (1, 2, 2.0)]
```

### 7.7 `find_point_pairs_below_distance`

```python
def find_point_pairs_below_distance(
    points: Sequence[Point],
    threshold: float,
) -> tuple[PointPairDistance, ...]
```

Calls `point_pair_distances()` and retains records that strictly satisfy
$d_{ij}<d_{\mathrm{caller}}$. The caller owns the interpretation of
`threshold`; geometry does not treat it as a chemical threshold.

#### Example

```pycon
>>> from hotpot.cheminfo import geometry as geo
>>> points = tuple(geo.Point((x, 0, 0)) for x in (0, 1, 3))
>>> result = geo.find_point_pairs_below_distance(points, threshold=3.0)
>>> [(item.first_index, item.second_index, item.distance) for item in result]
[(0, 1, 1.0), (1, 2, 2.0)]
```

### 7.8 `locate_point_in_planar_cycle`

```python
def locate_point_in_planar_cycle(
    point: Point,
    cycle: Cycle,
    plane: Plane,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> PointCycleLocation
```

Constructs an orthonormal basis $(\mathbf u,\mathbf v)$ on `plane` and
projects three-dimensional points as:

$\pi(\mathbf p)=((\mathbf p-\mathbf q)\cdot\mathbf u,(\mathbf p-\mathbf q)\cdot\mathbf v)$

The implementation first checks every nonadjacent pair of projected edges for
a stable self-intersection or an indeterminate numerical result. It then
computes distance to the boundary and winding number. This is not a general
proof that a polygon is simple: callers must still provide vertices ordered
along one simple boundary. Inputs with adjacent-edge overlap, reversal, and
similar defects are outside the complete validation scope of this interface.

$d_{\partial\mathcal C}(\mathbf p)=\min_i d(\pi(\mathbf p),\pi(e_i))$

$w(\mathbf p)=\frac1{2\pi}\sum_i\operatorname{atan2}\left((\mathbf z_i\times\mathbf z_{i+1})_z,\mathbf z_i\cdot\mathbf z_{i+1}\right)$

where $\mathbf z_i=\pi(\mathbf v_i)-\pi(\mathbf p)$.

- $d_{\partial\mathcal C}\le\epsilon_L$: `BOUNDARY`;
- boundary distance in the guard band: `UNDETERMINED`;
- $||w|-1|\le\epsilon_{\mathrm{winding}}$: `INTERIOR`;
- $|w|\le\epsilon_{\mathrm{winding}}$: `EXTERIOR`;
- otherwise: `UNDETERMINED`.

This interface classifies the point's projection on the supplied plane. It
does not verify that `point` itself lies in the plane or that `plane` is the
cycle's best-fit plane. The numerical scale $L$ is derived from the
three-dimensional `point` and cycle vertices before projection, so moving an
otherwise identical projected point far along the plane normal can alter the
derived tolerance.

#### Example

```pycon
>>> from hotpot.cheminfo import geometry as geo
>>> cycle = geo.Cycle([(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)])
>>> plane = geo.Plane((0, 0, 0), (0, 0, 1))
>>> geo.locate_point_in_planar_cycle(geo.Point((1, 1, 4)), cycle, plane).value
'interior'
```

### 7.9 Shared segment–cycle mathematical model

Planar cycles use `PLANAR_POLYGON`. Let the segment be
$S=[\mathbf s_0,\mathbf s_1]$ and the best-fit plane be
$\Pi(\mathbf c,\hat{\mathbf n})$:

$h_0=\hat{\mathbf n}\cdot(\mathbf s_0-\mathbf c)$

$h_1=\hat{\mathbf n}\cdot(\mathbf s_1-\mathbf c)$

When the segment stably straddles the plane:

$t=\frac{h_0}{h_0-h_1}$

$\mathbf x_\cap=\mathbf s_0+t(\mathbf s_1-\mathbf s_0)$

`TRANSVERSE_INTERIOR` and `PIERCES` are confirmed only when $t$ lies stably in
$(0,1)$ and $\mathbf x_\cap$ lies stably inside the simple polygon. Endpoint,
cycle-edge, cycle-vertex, and coplanar contact, as well as an intersection
found only on the infinite line extension, are not strict piercings; they are
recorded in `features`.

Nonplanar cycles use `VERTEX_TRIANGULATION_FAMILY`. A simple boundary with $m$
vertices has:

$n_{\mathrm{surface}}=\operatorname{Cat}_{m-2}=\frac1{m-1}\binom{2m-4}{m-2}$

combinatorial vertex triangulations. A candidate surface is counted as
`EMBEDDED` only if its triangle topology is valid, its unique boundary equals
the original cycle, and no pair of triangles has a geometric intersection
beyond the combinatorially permitted shared part. For triangles $T_i,T_j$,
the embedding condition is:

$f(T_i)\cap f(T_j)=f(T_i\cap T_j)$

The triangle normal and intersection-point barycentric coordinates are:

$\mathbf n_\triangle=(\mathbf b-\mathbf a)\times(\mathbf c-\mathbf a)$

$\lambda_a=\frac{((\mathbf b-\mathbf x_\cap)\times(\mathbf c-\mathbf x_\cap))\cdot\mathbf n_\triangle}{\lVert\mathbf n_\triangle\rVert^2}$

$\lambda_b=\frac{((\mathbf c-\mathbf x_\cap)\times(\mathbf a-\mathbf x_\cap))\cdot\mathbf n_\triangle}{\lVert\mathbf n_\triangle\rVert^2}$

$\lambda_c=1-\lambda_a-\lambda_b$

All $\lambda_i>g\epsilon_u$ denotes the strict triangle interior. Boundary
and guard-band values are recorded as contact or indeterminate, respectively.

Let $K$ mean that surface construction/enumeration and every segment
evaluation required for this result completed without a budget interruption.
Define:

- $n_{\mathrm{emb}}$: number of confirmed embedded candidate surfaces;
- $n_{\mathrm{construct\_undetermined}}$: number whose construction is
  indeterminate;
- $n_{\mathrm{hit}}$: number of embedded surfaces the segment is confirmed to
  pierce;
- $n_{\mathrm{miss}}$: number of embedded surfaces the segment is confirmed not
  to pierce;
- $n_{\mathrm{eval\_undetermined}}$: number of embedded surfaces with an
  indeterminate intersection result.

Final consensus is:

$\mathrm{PIERCES}\iff K\land n_{\mathrm{construct\_undetermined}}=0\land n_{\mathrm{eval\_undetermined}}=0\land n_{\mathrm{emb}}>0\land n_{\mathrm{hit}}=n_{\mathrm{emb}}$

$\mathrm{DOES\_NOT\_PIERCE}\iff K\land n_{\mathrm{construct\_undetermined}}=0\land n_{\mathrm{eval\_undetermined}}=0\land n_{\mathrm{emb}}>0\land n_{\mathrm{miss}}=n_{\mathrm{emb}}$

Every other case returns `UNDETERMINED`. The conclusion is relative to this
vertex-triangulation family; it does not claim to cover every continuous
spanning surface.

### 7.10 `iter_segment_cycle_relations`

```python
def iter_segment_cycle_relations(
    segments: Iterable[Segment],
    cycle: Cycle,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> Iterator[SegmentCycleRelation]
```

A lazy batch interface. It measures planarity or prepares nonplanar candidate
surfaces once for `cycle`, then yields a `SegmentCycleRelation` for each
segment in input order. Every result is equivalent to an independent
`determine_segment_cycle_relation()` call. Prefer this interface when several
candidate bonds share a cycle.

#### Example

```pycon
>>> from hotpot.cheminfo import geometry as geo
>>> cycle = geo.Cycle([(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)])
>>> segments = (
...     geo.Segment((1, 1, -1), (1, 1, 1)),
...     geo.Segment((3, 3, -1), (3, 3, 1)),
... )
>>> [result.state.value for result in geo.iter_segment_cycle_relations(segments, cycle)]
['pierces', 'does_not_pierce']
```

### 7.11 `determine_segment_cycle_relation`

```python
def determine_segment_cycle_relation(
    segment: Segment,
    cycle: Cycle,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> SegmentCycleRelation
```

The single-segment entry point. It is equivalent to calling
`iter_segment_cycle_relations()` for a one-element segment sequence and taking
the first result. Returns the complete state, contact features, indeterminacy
causes, surface model, intersection points, closest cycle edge, and
enumeration evidence.

#### Example

```pycon
>>> from hotpot.cheminfo import geometry as geo
>>> cycle = geo.Cycle([(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)])
>>> segment = geo.Segment((1, 1, -1), (1, 1, 1))
>>> result = geo.determine_segment_cycle_relation(segment, cycle)
>>> result.state.value, [point.coordinates for point in result.intersection_points]
('pierces', [(1.0, 1.0, 0.0)])
```

### 7.12 `closest_cycle_edge`

```python
def closest_cycle_edge(
    cycle: Cycle,
    segment: Segment,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> ClosestCycleEdge | None
```

Calculates for every cycle edge:

$d_i=d(e_i,S)$

$d_{\min}=\min_i d_i$

$i^*=\min\{i\mid d_i\le d_{\min}+\epsilon_L\}$

Ties within tolerance select the smallest edge index, giving stable output.
Returns `None` for non-finite input, degenerate local scale, or when no finite
distance exists.

#### Example

```pycon
>>> from hotpot.cheminfo import geometry as geo
>>> cycle = geo.Cycle([(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)])
>>> segment = geo.Segment((-1, -1, -1), (-1, -1, 1))
>>> result = geo.closest_cycle_edge(cycle, segment)
>>> result.edge_index, round(result.distance, 6)
(0, 1.414214)
```

### 7.13 `iter_segment_cycle_screenings`

```python
def iter_segment_cycle_screenings(
    segments: Iterable[Segment],
    cycle: Cycle,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> Iterator[SegmentCycleScreening]
```

Prepares the cycle once and screens segments in input order. It never uses
AABB separation to hide an invalid or incomplete cycle-surface model.

## 8. Chemical-object conversion API

### 8.1 Structural protocols and generic source types

The conversion layer uses structural protocols rather than importing Core at
runtime. It therefore accepts native Hotpot objects as well as custom objects
with the same minimum attributes:

| Source object | Minimum structural requirements |
|---|---|
| Atom-like | `coordinates: Sequence[float]` and `idx: int` |
| Bond-like | `atom1` and `atom2`, both Atom-like |
| Ring-like | ordered `atoms: Sequence[Atom-like]` |
| Structure-like | iterable `atoms` and `bonds` |
| Molecule-like | Structure-like requirements plus `rings_for_scope(ring_scope)` |

Generic results retain the concrete source type and original object
references. Conversion copies current coordinates into immutable geometry
value objects but neither copies, freezes, nor modifies the source chemical
objects.

### 8.2 `PairScope`

```python
PairScope = Literal["all", "bonded", "nonbonded"]
```

- `"all"`: every unordered atom pair;
- `"bonded"`: only explicitly bonded atom pairs;
- `"nonbonded"`: only atom pairs without an explicit bond.

### 8.3 `RingScope`

```python
RingScope = Literal["full_graph", "ligand_skeleton"]
```

The source object's `rings_for_scope()` implements the actual ring-selection
behavior. In Hotpot, `full_graph` uses the complete molecular graph, while
`ligand_skeleton` observes ligand-skeleton rings without metal-coordination
connections.

### 8.4 `AtomGeometry`

```python
AtomGeometry[AtomT](atom: AtomT, point: Point, key: int)
```

`atom` is the original Atom-like object, `point` is a coordinate snapshot,
and `key=int(atom.idx)`.

### 8.5 `AtomPairTarget`

```python
AtomPairTarget[AtomT](
    first: AtomGeometry[AtomT],
    second: AtomGeometry[AtomT],
    bonded: bool,
)
```

Records an atom pair in stable order and whether the source graph contains an
explicit bond between the atoms.

### 8.6 `BondGeometry`

```python
BondGeometry[BondT](
    bond: BondT,
    segment: Segment,
    key: tuple[int, int],
)
```

The bond key is the ascending pair of endpoint atom keys:

$k_{\mathrm{bond}}=\operatorname{sort}(k_{a_1},k_{a_2})$

### 8.7 `RingGeometry`

```python
RingGeometry[RingT](
    ring: RingT,
    cycle: Cycle,
    key: tuple[int, ...],
)
```

The ring key is the lexicographically smallest tuple among every cyclic shift
of the atom-key sequence in both traversal directions. It is therefore
invariant to the starting vertex and traversal direction.

### 8.8 `BondRingTarget`

```python
BondRingTarget[RingT, BondT](
    ring: RingGeometry[RingT],
    bond: BondGeometry[BondT],
)
```

Represents one source Ring × Bond combination awaiting a segment–cycle
classification.

### 8.9 `AtomPairDistance`

```python
AtomPairDistance[AtomT](
    target: AtomPairTarget[AtomT],
    measurement: PointPairDistance,
)
```

Combines the source atom pair with a pure-geometry distance record.

### 8.10 `BondRingFinding`

```python
BondRingFinding[RingT, BondT](
    target: BondRingTarget[RingT, BondT],
    relation: SegmentCycleRelation,
)
```

Combines the source Ring × Bond pair with its complete segment–cycle
relationship.

### 8.11 `RingEdgeDistance`

```python
RingEdgeDistance[BondT](
    bond: BondT,
    measurement: ClosestCycleEdge,
)
```

Combines a source ring bond with a nearest-edge measurement. No public
function currently constructs this record directly. Callers may use it when
mapping a geometric edge index back to a chemical Bond.

### 8.12 `BondRingScanReport`

```python
BondRingScanReport[RingT, BondT](
    findings: tuple[BondRingFinding[RingT, BondT], ...],
    ring_scope: RingScope,
    max_ring_size: int,
    selected_ring_count: int,
    excluded_ring_count: int,
    piercing_pair_count: int,
    does_not_pierce_pair_count: int,
    undetermined_pair_count: int,
)
```

| Field/property | Meaning |
|---|---|
| `findings` | Complete record for every evaluated candidate pair |
| `ring_scope` | Ring scope requested for this scan |
| `max_ring_size` | Maximum atom count of rings included in the scan |
| `selected_ring_count` | Number of rings satisfying the size limit |
| `excluded_ring_count` | Number of perceived Relevant Cycles larger than `max_ring_size` and therefore omitted from relationship evaluation |
| `candidate_pair_count` | Read-only derived number of Ring × Bond findings after excluding each ring's own edges |
| three `*_pair_count` fields | Number of final results in each of the three states |
| `scan_complete` | Read-only derived fact that every selected finding completed its candidate-surface enumeration |
| `piercings` | Read-only derived tuple containing only `PIERCES` findings |
| `undetermined` | Read-only derived tuple containing only `UNDETERMINED` findings |

`scan_complete=True` applies only to candidates inside the selected,
size-bounded scope. Also inspect `ring_scope`, `max_ring_size`, and
`excluded_ring_count` when interpreting overall coverage.
An empty selection may produce a complete empty report. Force-field policy may
report excluded larger rings without treating them as a geometry failure.

### 8.13 `point_from_atom`

```python
def point_from_atom(atom: AtomT) -> Point
```

Reads `atom.coordinates` into an immutable coordinate snapshot without
modifying `atom`.

### 8.14 `segment_from_bond`

```python
def segment_from_bond(bond: BondT) -> Segment
```

Converts the current coordinates of `bond.atom1` and `bond.atom2` into a
finite segment. It does not inspect bond order or chemical type.

### 8.15 `cycle_from_ring`

```python
def cycle_from_ring(ring: RingT) -> Cycle
```

Creates a closed `Cycle` in the existing order of `ring.atoms`. Input order
defines boundary connectivity. This adapter does not perceive rings; its input
must already describe an ordered closed boundary.

### 8.16 `iter_atom_geometries`

```python
def iter_atom_geometries(
    structure: _StructureLike[AtomT, BondT],
) -> Iterator[AtomGeometry[AtomT]]
```

Lazily yields `AtomGeometry` in the source order of `structure.atoms`.

### 8.17 `iter_atom_pair_targets`

```python
def iter_atom_pair_targets(
    structure: _StructureLike[AtomT, BondT],
    pair_scope: PairScope,
) -> Iterator[AtomPairTarget[AtomT]]
```

Yields atom pairs in source combination order $i<j$ and obtains `bonded` from
`structure.bonds`. An unsupported `pair_scope` raises `ValueError`.

### 8.18 `iter_ring_geometries`

```python
def iter_ring_geometries(
    mol: _MoleculeLike[AtomT, BondT, RingT],
    *,
    ring_scope: RingScope,
    max_ring_size: int,
) -> Iterator[RingGeometry[RingT]]
```

Calls `mol.rings_for_scope(ring_scope, max_size=max_ring_size)`, sorts the
returned Relevant Cycles by canonical ring key, and then yields them lazily.

`max_ring_size` is required and is passed into native Relevant Cycle
enumeration. This avoids enumerating larger cycles merely to discard them.
Relevant Cycles are not the set of all simple cycles. Hotpot Core retains its
10,000-result safety limit, so this call can raise
`RelevantCycleLimitExceeded` rather than return a partial family.

### 8.19 `iter_bond_ring_targets`

```python
def iter_bond_ring_targets(
    mol: _MoleculeLike[AtomT, BondT, RingT],
    *,
    ring_scope: RingScope,
    max_ring_size: int,
) -> Iterator[BondRingTarget[RingT, BondT]]
```

For every selected ring and chemical bond, ordered by canonical ring key and
bond key respectively, yields a candidate combination except when the bond is
an edge of that ring. In contrast, a direct
`determine_bond_ring_relation(ring, bond)` call does not perform this
exclusion.

### 8.20 `measure_atom_pair_distances`

```python
def measure_atom_pair_distances(
    structure: _StructureLike[AtomT, BondT],
    pair_scope: PairScope,
) -> tuple[AtomPairDistance[AtomT], ...]
```

Calculates Euclidean distances for the selected atom pairs while retaining
source Atom references, atom keys, and bonded status. Returns a dense tuple.

### 8.21 `determine_bond_ring_relation`

```python
def determine_bond_ring_relation(
    ring: RingT,
    bond: BondT,
    *,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> BondRingFinding[RingT, BondT]
```

Adapter for a single chemical-object pair: converts `ring` and `bond` to a
`Cycle` and `Segment`, calls `determine_segment_cycle_relation()`, then wraps
the sources and result in `BondRingFinding`.

### 8.22 `iter_bond_ring_findings`

```python
def iter_bond_ring_findings(
    mol: _MoleculeLike[AtomT, BondT, RingT],
    *,
    ring_scope: RingScope,
    max_ring_size: int,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> Iterator[BondRingFinding[RingT, BondT]]
```

Lazily yields results in canonical ring-key and bond-key order. Each ring uses
one `iter_segment_cycle_relations()` call, so all candidate bonds for that ring
share its planarity measurement or surface preparation.

### 8.23 `scan_bond_ring_relations`

```python
def scan_bond_ring_relations(
    mol: _MoleculeLike[AtomT, BondT, RingT],
    *,
    ring_scope: RingScope,
    max_ring_size: int,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> BondRingScanReport[RingT, BondT]
```

Consumes the complete lazy stream and returns a dense report. This interface
is intended for diagnostics, auditing, and business logic that requires
per-candidate evidence.

Here too, `max_ring_size` is required and bounds Relevant Cycle enumeration at
the source.

### 8.24 `determine_bond_ring_piercing_state`

```python
def determine_bond_ring_piercing_state(
    mol: _MoleculeLike[AtomT, BondT, RingT],
    *,
    ring_scope: RingScope,
    max_ring_size: int,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> PiercingState
```

A fast entry point for callers that need only the aggregate three-state result:

1. return immediately on the first `PIERCES` result;
2. if no piercing exists but at least one candidate is `UNDETERMINED`, return
   `UNDETERMINED`;
3. otherwise return `DOES_NOT_PIERCE`.

The result covers only the declared `ring_scope` and `max_ring_size`. This
scalar interface does not carry ring-selection metadata; use
`scan_bond_ring_relations()` when the selected family and bound must be audited.

### 8.25 `BondRingScreeningReport`

```python
BondRingScreeningReport[RingT, BondT](
    actionable_findings: tuple[BondRingFinding[RingT, BondT], ...],
    ring_scope: RingScope,
    max_ring_size: int,
    selected_ring_count: int,
    excluded_ring_count: int,
    candidate_pair_count: int,
    aabb_separated_pair_count: int,
    exact_pair_count: int,
    piercing_pair_count: int,
    does_not_pierce_pair_count: int,
    undetermined_pair_count: int,
    scan_complete: bool,
)
```

The report covers every selected pair but retains complete findings only for
`PIERCES` and `UNDETERMINED`. The three state counts always sum to
`candidate_pair_count`; `aabb_separated_pair_count + exact_pair_count` does as
well.

### 8.26 `screen_bond_ring_relations`

```python
def screen_bond_ring_relations(
    mol: _MoleculeLike[AtomT, BondT, RingT],
    *,
    ring_scope: RingScope,
    max_ring_size: int,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> BondRingScreeningReport[RingT, BondT]
```

Screens the complete requested Ring × Bond scope with a guarded AABB broad
phase, falling back to the complete relation kernel whenever separation is not
strictly proven. Use `scan_bond_ring_relations()` when complete evidence for
every non-piercing pair is required.
