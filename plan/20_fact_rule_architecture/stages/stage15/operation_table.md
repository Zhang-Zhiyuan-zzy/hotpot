# Stage15 — supported exact symmetry operations

Status: concrete design companion, 2026-11-04. The [complete public tree](review.md#1-complete-proposed-public-tree--before-implementation)
owns the proposed API. `SymmUnit` is one immutable affine group-element family;
these rows describe mathematical forms of that same structure. A kind name or
an axis/plane symbol alone never supplies a complete operation.

## Exact notation and geometric parameters

Use fractional columns `f`, oriented Cartesian column basis `H=M.T`, and
`x=H f`. An exact operation is `{R|t}` with `f′=R f+t`. Equivalently, its
Jones coordinate expression is the three rows of `R f+t`, such as
`-x,y+1/2,-z`. Either the **full numeric Seitz pair** or the exact `xyz`
expression reconstructs the operation in the specified group/setting. The
full numeric matrix supplies axis/sense information that a bare `2`, `m` or
`4_1` cannot. IUCr's recommended **symbolic** Seitz notation additionally
specifies the orientation/sense of the linear operation; it must not be
confused with just its type symbol. [O1, O2]

For the geometric descriptions below:

- `c` is a fractional point on the axis/plane, or the inversion/rotoinversion
  centre; its Cartesian location is `H c`.
- `u` is an oriented Cartesian unit axis/normal. `Q(u,θ)` is the proper
  Rodrigues rotation; `F(u)=I−2uu.T` is reflection in a plane through zero.
  The fractional linear parts are `R=H⁻¹ Q H` or `H⁻¹ F H`.
- `n∈{2,3,4,6}` and `θ=±2πk/n` identify crystallographic rotation order/sense
  (reduce the order if `k,n` have a common divisor). `l` is a specified
  fractional lattice vector parallel to the axis, `R l=l`.
- `v` is the **full** invariant translation: along an axis for a screw,
  or in a plane for a glide. It includes every integer translation. All
  displayed `R,t` are exact; they must belong to the supplied `SpaceGroup`.
  Floating Cartesian `u,θ,H` describe geometry, not an instruction to guess
  exact rational operators by rounding.

| Derived kind / conventional shorthand | Exact `R` and **full** `t` | Required geometric information | Reconstructible example in its stated axes |
|---|---|---|---|
| Identity `e`, type `1` | `R=I`, `t=0` | Specified group/setting | `x,y,z` = `{I|0,0,0}` |
| Translation `T_v`, including centring | `R=I`, `t=v≠0` | Exact vector `v`; integer versus fractional centring is contextual | `x+1/2,y+1/2,z` = `{I|1/2,1/2,0}`; its square is `x+1,y+1,z` |
| Rotation `n^±` | `R=H⁻¹Q(u,θ)H`, `t=(I−R)c` | Axis line `c+sH⁻¹u`, order, angle/sense | Twofold about `y` through `c=(1/4,0,0)`: `-x+1/2,y,-z` |
| Screw `n_p^±` | Same proper `R`; `t=(I−R)c+v`, `R v=v`; conventionally `v=(p/n)l` **plus any retained axial lattice carry** | Axis line, sense, exact `l`, pitch and full translation; `p mod n` alone is insufficient | Along `y`: `-x,y+1/2,-z`; along tetragonal `z`: `-y,x,z+1/4` |
| Reflection `m` | `R=H⁻¹F(u)H`, `t=(I−R)c` | Plane point `c` and normal `u` | Plane `y=1/4`: `x,-y+1/2,z` |
| Glide reflection `a,b,c,n,d` (contextual glide-plane symbols) | Same reflection `R`; `t=(I−R)c+v`, `R v=v`, `v≠0` | Plane, full in-plane vector `v`; the glide letter alone omits position/orientation/carry | Plane `y=1/4`, glide `c/2`: `x,-y+1/2,z+1/2` |
| Inversion `−1` | `R=−I`, `t=2c` | Centre `c` | Centre `(1/4,0,0)`: `-x+1/2,-y,-z` |
| Rotoinversion `−n^±`, `n∈{3,4,6}` | `R=−H⁻¹Q(u,θ)H`, `t=(I−R)c` | Axis/sense/order and centre; `−1` and `−2=m` are the previous special forms | Tetragonal `−4⁺` about `z` at zero: `y,-x,-z` |

These forms cover all ordinary 3D crystallographic operations in the declared
lattice/group. `e`-glide denotes a **geometric element with multiple glide
operations**: each associated operation still has its own exact row above.
A centred lattice can likewise support several glide vectors for one plane.
No extra algebra or coordinate engine is attached to these names. [O1–O4]

For completeness, these are the axial proper-rotation/screw templates used by
the table; changing an axis/origin uses exact conjugation, not another engine:

| Axes / linear order | Exact `R` (rows) | `t` for on-origin axis | Exact `xyz` |
|---|---|---|---|
| Axes admitting `2_z` | `(-1,0,0);(0,-1,0);(0,0,1)` | `(0,0,p/2+q)` | `-x,-y,z+p/2+q` |
| Hexagonal axes, `3_z⁺` | `(0,-1,0);(1,-1,0);(0,0,1)` | `(0,0,p/3+q)` | `-y,x-y,z+p/3+q` |
| Tetragonal axes, `4_z⁺` | `(0,-1,0);(1,0,0);(0,0,1)` | `(0,0,p/4+q)` | `-y,x,z+p/4+q` |
| Hexagonal axes, `6_z⁺` | `(1,-1,0);(1,0,0);(0,0,1)` | `(0,0,p/6+q)` | `x-y,x,z+p/6+q` |

Here `p=0,…,n−1`, `q∈Z`, and a specific example is supported only when it is a
member of the active group. `p=q=0` is a rotation; nonzero full axial translation
is a screw form even when its cell representative looks like a rotation. For
an `n_p` element in this template, `g**n=T_(0,0,p+nq)`; replacing its full power
by identity would lose a real translation. A pure rotation translated along
its invariant axis can have an integral screw pitch; a fractional type suffix
alone cannot represent that full element.

Classification is derived exactly: for a linear part of order `N`,
`v=(I+R+…+R^(N−1))t/N` is the invariant translation. `t≠0` by itself does not
make an off-origin rotation a screw or an off-origin mirror a glide. Removing
`v` leaves the origin term `(I−R)c`. The classification is descriptive; every
result continues to be stored as the same exact `(R,t,context)` value.

## One algebra and Python operators

`context` is the immutable **SpaceGroup value** in the proposed tree; it owns
the selected coordinate setting, complete representatives and lattice data.
It contains no physical cell dimensions or chemical content. Constructors check
that a `SymmUnit` is a member of that full group. Every accepted same-context
operation stays supported after composition, inverse and any integer power.

| Spelling | Exact meaning / single implementation |
|---|---|
| `g.compose(h)`, `g @ h` | `g` after `h`: `(R_g R_h, R_g t_h+t_g)`; `(g@h).operate(f)=g.operate(h.operate(f))` |
| `g.inverse()`, `~g` | `(R_g⁻¹,−R_g⁻¹ t_g)` |
| `g.power(n)`, `g ** n` | Integer repeated composition; `0` gives identity; negatives use the same inverse; no modular power argument |
| `g == h`, `hash(g)` | Exact full `R,t` and equal immutable group/setting context; never comparison modulo cell translations |
| `g.reduce_to_cell()` | `(r_i,n)` satisfying **exactly** `g=T_n r_i`; this explicitly returns the carry |

The magic methods delegate to the named methods and their one pure affine
owner. No second implementations, `*`/`+` aliases or array-action overloads are
proposed. Different contexts need an explicit exact rebind/conjugation whose
membership is checked; equality is false across unequal contexts, while algebra
with them is locally incompatible. Hashing uses exact values, not mutable cell,
Molecule identity or process IDs.

## Closure, lattices and the finite catalogue

For an accepted group, `R Z³=Z³`, its complete cell representatives `r_i`, and
`T_n(f)=f+n`:

```text
r_i @ r_j = T_c(i,j) @ r_k                 c(i,j)∈Z³, retained exactly
(T_n @ r_i) @ (T_m @ r_j)
           = T_(n+R_i m+c(i,j)) @ r_k
G          = {T_n @ r_i : n∈Z³, every listed i}
T          = {T_(n+c) : n∈Z³, c a listed pure-translation coset}
```

This guarantees closure of supported members of **one common group**. The
finite representatives alone need not be closed under full affine composition:
`2_1**2=T_l` and a glide square can be a nonzero lattice translation. Their
finite quotient product is useful only together with its carry. Centring must
remain in `G/T_cell`; removing centring to form `G/T` changes the quotient.

A common positive metric and shared invariant lattice, or an explicitly
validated common supergroup, can justify combining isometries. Arbitrary
rotations/reflections from different metrics, lattices or groups do not acquire
crystallographic closure merely because each has a familiar kind name. A
mixture can have no accepted common group or produce a noncrystallographic or
infinite point action. Do not repair such a mixture by wrapping away integer
translations or relabelling its powers. The public group-bound `SymmUnit`
construction is the supported path; pure Cartesian mathematics stays at its
existing mathematical owner.

The [complete catalogue](space_group_catalogue.md) gives every actual
conventional-cell representative for every ordinary 3D type in one explicit
reference setting. Each literal `xyz` there decodes to the exact `R,t` in
[the data file](space_group_catalogue.json), using the universal notation above.
An `ImageKey` must retain the full operation and integer image; a displayed
wrapped position is not the group element.

## Primary references and independent examples

- **O1** — [IUCr Commission report: Seitz symbols for crystallographic symmetry operations (2014)](https://journals.iucr.org/a/issues/2014/03/00/es0405/index.html),
  especially Tables 1–3 and the need for orientation/sense with symbolic Seitz notation.
- **O2** — [COMCIFS core dictionary, pinned source](https://github.com/COMCIFS/cif_core/blob/93b39f69529f489815dd1020376ce1df267f2eee/cif_core.dic),
  `_space_group_symop.operation_xyz`, `R`, `T`, `Seitz_matrix`, and the glide
  `x,1/2-y,1/2+z` through `y=1/4` with glide `c/2`.
- **O3** — [IUCr symmetry operation](https://dictionary.iucr.org/Symmetry_operation),
  [symmetry element](https://dictionary.iucr.org/Symmetry_element) and
  [space group](https://dictionary.iucr.org/Space_group).
- **O4** — [IUCr nomenclature report: Symbols for symmetry elements (1992)](https://www.iucr.org/who-we-are/commissions/commission-on-crystallographic-nomenclature/published-reports/symsym).
- **O5** — [IUCr checkCIF P21/c complete operator example](https://journals.iucr.org/services/cif/checking/SYMMG_02.html)
  independently specifies all four reference-setting operators, including
  screw/glide terms; [official OPTIMADE v1.3 examples](https://schemas.optimade.org/defs/v1.3/properties/optimade/trajectories/space_group_symmetry_operations_xyz)
  specify full primitive `P 2` and centred `C 2` lists. These bounded examples,
  exact arithmetic and separately retrieved COMCIFS reference metadata check
  the catalogue without claiming independent validation of all operators.
