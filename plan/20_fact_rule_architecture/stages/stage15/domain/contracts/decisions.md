# Stage15A — accepted exact symmetry and molecular-crystal domain

Root decision 2026-11-05 under §16.4; source starts from validated
`B_exp=5a7b19396328eee964a8f3c65bc3345c013eaec6` on independent
`feat/plan20-crystal-symmetry`. Stage13 is complete separately at pushed
`I_exp=cc9470e19f2fac786138d5af0986d726d5e793c7`. Stage14 molecular work is
independent. This document resolves the entry choices in review.md; its existing
mathematical and chemistry constraints remain normative.

## Accepted tree and incremental publication

Accept review.md §1's Crystal, SpaceGroup and SymmUnit tree, permissions,
exact meanings and ordinary returns. Publish `SpaceGroup` and `SymmUnit` at
`hotpot.cheminfo.structure.symmetry`, `hotpot.cheminfo.structure`,
`hotpot.cheminfo` and top-level `hotpot`. There is one public abstract SymmUnit
and one private concrete affine implementation, not separate operation classes
or 786 enumerated types. A complete affine element is the abstraction level.
Group-bound constructors/magic methods retain full integer carries.

Add exactly the following pure persistence facade methods for §15E:

```python
Crystal.from_cif(cls, source, *, block=None) -> Crystal  # classmethod
Crystal.to_cif(self) -> str
```

A source is CIF text or a pathlib/os path; a selected block is its exact name.
Multiple blocks without a selection reject locally. There is no reader registry,
model launch, scientific perception, permissive fallback or new result type.
`Crystal.operate` returns one fresh placement, leaving its active group/cell
unchanged. `unit_cell`/`expand` own replication and produce active P1. Unknown
group remains unknown, never defaults to P1. Legacy six-scalar construction is
retained; standard alpha=angle(b,c), beta=angle(a,c), gamma=angle(a,b) supersedes
the old from_matrix angle bug. Do not reinterpret historical scalar facts.

The first coherent production checkpoint may publish the complete standalone
exact group/affine value domain and attributed lookup. It does not publish an
unimplemented expanded Crystal facade or close Stage15. Crystal publication
follows one connected association/codec/lifecycle migration; subsequent actions,
findings and formats have their separate evidence. No general network or PyG
capability is inferred from exact-table tests.

## Exact group and setting contracts

The sole immutable SpaceGroup record contains canonical complete half-open
(R,t) representatives and exact setting binding; labels are reference/current-
setting metadata. No retained Gemmi object. SymmUnit holds one full immutable
R,t and that context. Fractions use stdlib Fraction; explicit int/Fraction or
exact rational strings are accepted, arbitrary floating rationals are not
silently guessed. R is integer, unimodular and of finite crystallographic order.

`SpaceGroup.from_operations` accepts a complete iterable of (R,t) pairs, xyz
strings, or existing SymmUnit values whose matrices/vectors are revalidated.
It is not a generator enumerator. Normalize representatives, require identity,
full quotient closure/inverses and lattice preservation. The finite rotation
orders are 1,2,3,4,6; compatible metric is checked only on Crystal attachment.
`reference_number` and `to_reference=(P,p)` must be supplied together or neither;
verify the full transformed affine group against that reference setting, including
the full translation lattice. Only an exact reference setting has its tabulated
H–M/Hall labels; unlabelled changed settings retain None, not invented names.

Named/magic algebra is exactly:
`(R,t)@(S,u)=(R*S,R*u+t)`, `~g=(R^-1,-R^-1*t)`, full integer powers.
Same-context equality/hash includes exact group/setting, full R and full t.
`reduce_to_cell` returns (representative, integer carry), with g=T_c@rep.
`lift(i,n)` retains integer n. Distinct groups reject composition until callers
explicitly rebind into a valid shared group.

`change_setting(P,p)` means f_old=P*f_new+p. Conjugate operations and transform
the full Bravais lattice; rebuild representatives by exact finite translation
cosets when index changes. Reject a requested cell whose Z^3 is not a sublattice
of the transformed translations or whose rotations do not preserve Z^3. Pure
rational matrix/lattice arithmetic belongs to one Hotpot mathematics module;
translation_basis is deterministic positive-diagonal column Hermite basis.
No external symbolic package is required.

Derived kind uses the invariant translation over the finite R order. An
off-origin pure rotation/reflection is not a screw/glide merely because t!=0.
Identity/nonzero translation, determinant, rotation order and the full affine
power determine the eight named kinds. Parsing xyz is a closed arithmetic
parser for linear rational combinations of x/y/z and rational constants, never
eval. xyz must reconstruct the exact full element, including negative/large t.

## Reference asset and optional CIF dependency

Promote the pinned attributed catalogue's 230 reference groups, 4425 complete
representatives, symbols and exact metadata to the sole runtime asset
`hotpot/cheminfo/reference/data/space_groups.json`. Preserve both original
licence texts under that reference owner and ship them. The maintained planning
catalogue remains documentation/evidence, not a second runtime owner. Lookup
materializes only the requested immutable group. The shipped attributed table is
qualified at release; only externally supplied complete operator lists need
quadratic closure validation at runtime. This avoids repeating release arithmetic
for every numbered lookup. No network lookup at runtime,
no whole-group global cache and no runtime Gemmi lookup dependency.

Use a lazy optional Gemmi0.7.5-compatible CIF parser/writer only, via one adapter;
record exact actual version. It may read/write CIF scalar/loop syntax, never
perform symmetry algebra, molecule manipulation, neighbor/matching/expansion or
chemistry. Add one optional `crystal` package extra as needed. Its absence is an
identified local unavailable CIF dependency and does not prevent mathematical
or molecular Crystal operations. Preserve MPL/CC-BY attribution/distribution.
No direct or indirect OpenBabel/RDKit/CCDC Python API call is permitted on any
Crystal path. Existing separate Molecule/toolkit projections consume prepared
six-parameter values only and are explicitly lossy for orientation/group/images.

## Molecular and periodic domain

Molecule remains the one chemistry/Cartesian/topology owner. Accept explicit
bond-image vectors on the complete simple Bond graph. Zero denotes the existing
finite unwrapped Bond convention; an unknown imported periodic image is None.
No self/parallel edges, silent edge collapse, synthetic atoms, guessed bonds,
automatic supercell repair or second Crystal chemical graph is introduced.

A quotient requiring self/parallel periodic edges is outside the present simple
Molecule domain. Its source CIF text/record is retained and a precise
CRYSTAL-UNSUPPORTED-PROJECTION rejects only that requested full projection;
independent molecular structures continue. This is the explicit resolution of
the current topology gate, not a claim of arbitrary polymer/network support.
Track it as a bounded deferred cross-stage topology extension with responsibility
and reopening evidence. Infinite components representable by the simple
image-labelled graph remain periodic; nonzero cycle voltage is not a finite
molecule. unit_molecules rejects/excludes only their finite extraction with the
named unavailable finding; never force cycle closure.

`site_relations` has exactly equivalent_images, occupancies and disorder keys;
values are immutable. Image keys/layouts and whole-bijection validation are those
in review.md. Only explicit validated identification may merge generated sites.
Numeric coincidence and possible-molecule findings never deduplicate. Generated
P1 active representation retains one immutable historical source basis/group
binding, complete contributing origins and all compatible bond origins. No second
active cell/group facts. Expand preserves independent copies and periodic edge
images in requested repeat order. Local unresolved origins stay explicit.

## Codec, lifecycle and numerical policy

Use one private immutable tuple crystal codec with schema marker
`hotpot.crystal/1`: oriented row basis, one exact SpaceGroup record, site relations,
bond images, representation and historical origin binding/maps. Atom/Bond facts
remain only in the enclosing MoleculeRecord. Referenced operations decode through
the one group value. No duplicate operations/setting on Crystal. The main
MoleculeRecord/pickle gets one current-version hard cut; update all actual
builders/capture/copy/components/merge/identity/rollback readers and writers.
No old/new loader or six-scalar compatibility runtime path. Existing scalar
construction still encodes into this same new record.

A source Crystal and active contained Molecule have bidirectional identity.
Construction may not steal an active association. create_crystal explicitly
replaces it atomically; the detached old Crystal retains independent cell/group
facts. Copy/pickle of combined references preserves aliases and isolates mutable
facts. Cell/scalar setters and preserve='fractional' coordinate changes share
one validate/commit/rollback operation. Group/site updates preserve compatible
relations or reject completely. Cell identity uses the oriented basis; symmetry
changes invalidate only actual symmetry consumers.

Cell and metric integrity uses relative1e-10/absolute1e-10 in angstrom^2, evaluated
against the cell metric. Fractional half-open wrapping snaps only within1e-12
of an integer before floor; it never merges sites. Public tolerance/cutoff arguments
are explicit caller values in angstrom, strictly positive by the actual query
domain. Coordinate reconstruction checks use1e-10 angstrom for analytic fixtures;
CIF precision reports use emitted precision and input uncertainty. Exact affine
rules, maps and discrete facts use exact equality. No arbitrary historical SHA
or numerical identity requirement for chemistry.

Triclinic image search uses inverse-basis metric bounds, not rounding or a fixed
27-image cube. Whole-molecule candidates require one complete labelled graph/site
bijection satisfying all fixed placed-coordinate residuals simultaneously.
No free alignment or per-atom lift. Exact equality fixtures and explicit tolerance
boundary fixtures are independent oracles. Finite closed-loop linking uses the
referenced polygonal Gauss formula with an explicit numerical resolution bound;
None is unavailable, zero is not an unlink certificate. Query findings use the
existing applicable records or immutable (code,subjects,measurements) tuples,
with exact review.md codes, no generic severity/confidence wrapper. Warnings use
RuntimeWarning with the code; no automatic scientific repair or destructive merge.

## Entry evidence and remaining gates

Full Stage15 review, operation/catalogue definitions and actual current consumer
inventory identify the six-scalar orientation/angle bug and self/parallel graph
limit. Prior exact catalogue checks (230groups/4425ops/284089pair products) are
reference derivation evidence, not product acceptance. Root reread governance,
stage and validation skills and retained valid full-plan/companion alignment.

Each incremental implementation checkpoint needs actual CP311 source and outside-
checkout installed tests, applicable affected supported versions, independent
analytic positive/negative/boundary fixtures, prohibited-API isolation and honest
cost/resources. Full 15B–F requires actual finite molecular expansion, special
positions, skew/centred/screw cases, periodic limitations, relations/geometry/
linking, copy/pickle/merge/rollback, CIF settings and installed delivery. Stage14D
waits on the published domain and affected format closure; Stage15 does not wait
on Stage14D. Preserve failures and three-attempt bounds; independent slices may
advance only after checking upstream dependencies. Root retains acceptance,
commit/push and overall completion decisions.
