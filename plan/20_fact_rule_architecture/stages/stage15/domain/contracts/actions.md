# Stage15C — exact action and generation decisions

Root accepts these bounded implementation decisions on 2026-11-05 after the
pushed connected checkpoint `3d3c5ef`. The existing accepted public tree and
scientific meanings remain unchanged. Deliver the one-placement action first,
then cell projection/finite lifting/expansion using that same action algebra.
The valid prior full-plan/companion alignment is retained; root reread Stage15's
complete action, lifecycle, origin, generation and acceptance contracts against
actual current owners before this dependent step.

## Exact historical frame binding

The sole historical origin tuple is now exactly
`(source_matrix, source_group, source_sites, source_bonds, basis_map)`.
`basis_map` is an explicitly supplied exact rational 3x3 matrix P, or None when
that frame relation is unavailable. It maps active fractional axes to the
historical source axes: `H_active = H_source @ P`. Both cells have the existing
zero Cartesian origin. There is no second active basis, group setting used as
provenance, floating reconstruction, extra coordinate graph or generic status.
The physical basis relation is checked numerically at the existing cell
precision; its exact rational entries are never guessed from floating values.
Historical site/Bond keys describe lineage, not equality of edited current
coordinates with old source coordinates.

Birth from independent content uses P=I; expansion by D=diag(repeats) uses
P_new=P_old@D. An active action g transports old historical operations by
h=(P R_g P^-1, P t_g), checked against the actual source group. Apply h to every
retained full origin, keeping representative and integer carry. Wrapping by w
subtracts P@w in the historical frame. Existing incomplete row origins remain
explicitly unavailable. A missing required basis map rejects only a dependent
full origin projection; it never invents a source operator.

A manually changed physical cell retains all historical descriptors/keys but
sets basis_map=None in the same complete publication. An identical cell change
is a no-op. No unrelated coordinate edit discards the actual unchanged axis map.
This is the current closed unreleased generated-record grammar, not a second
loader or a compatibility schema; obsolete four-entry origins reject.

## One fresh placement

`operate(g)` requires a current associated Molecule, complete supplied XYZ and a
member of the active known group. Capture the closure under that owner lock,
then build one fresh whole copy with the same oriented cell and active group.
Do not wrap coordinates. Transform each directed known Bond image by R; unknown
images stay unknown. Conjugate supplied active image relations by g. For
generated content, compose every historical Atom/Bond origin through h above.
No union, group extension, source mutation, toolkit or model inference occurs.

Use the existing stereochemistry algebra for the exact orientation sign of R.
Proper actions keep reference-relative records. Improper actions flip TH/AX,
retain DB, and reflect coordination ideal sites through the existing canonical
rotation tables. Preserve reference/occurrence/group/source-ledger order.
The current native absence map is explicit: RDKit's `CHI_UNSPECIFIED` and
`STEREONONE` retain identical descriptor/source codes and empty carriers;
display token `NONE` (like None) declares no directional display. These carry no
orientation and remain unchanged. Actual extraction emits these entries even
for ordinary achiral content, so they must not block a reflection. Other opaque
descriptors, specified foreign replay records and uninterpreted directional
display tokens have no approved improper map and reject that requested action
with CRYSTAL-UNSUPPORTED-PROJECTION before publication. Source codes remain
historical; they are not rewritten to invent an adapter interpretation.

Cartesian axis constraints travel as their actual constrained subspace. A
signed-axis permutation transports values/presence/source per axis. A more
general rotation transports a fully supplied axis-aligned constraint only when
the transformed subspace is also axis aligned; absent/free/full isotropic cases
remain themselves. An unrepresentable oblique partial constraint rejects the
requested full projection, without dropping restrictions. Bond/angle/torsion
constraints follow the actual atom mapping.

Whole-copy selected MCA/partial-charge/source-frame observations retain their
existing immutable payload and are invalidated through the actual coordinate/
stereo dependency owners. They are never presented as newly calculated current
observations. The source remains untouched. Repeated actions must agree with
same-group composition for chemical facts, coordinates, images and stereo,
within the accepted coordinate tolerance.

## Checkpoint evidence and remaining work

Check independent P1 full translation, inversion/proper rotations, skew/rotated
bases, directed images, conjugated relations, composition, improper stereo,
axis constraints, known/unknown group and missing coordinates, copy aliases,
source/result currentness and prohibited-API isolation. Run affected CP311
source and real outside-checkout installed evidence. Keep failed attempts and
three-attempt bounds; no overall Stage15 claim follows from this action slice.
Cell projection, explicit identification validation, finite-voltage extraction,
supercell expansion, observations/CIF and Stage14D retain their required gates.
