# Connected Stage15B lifecycle decisions

Root accepts these private implementation choices on2026-11-05 from pushed
standalone checkpoint`bcc199b355183f857b8784b267cf4c53b2eaa21a`.
The accepted15A tree/meaning remains the public contract; this section resolves
its concrete connected record, ownership and rollback mechanics before code.
Stage14 remains independent until its explicit Crystal consumer migration.

## One active owner and one closed record

Crystal retains one immutable private state tuple and one Molecule association.
Its oriented row basis, immutable SpaceGroup association, supplied site relations,
Bond-image bindings, representation and historical origins live only there.
No external object, extra coordinate/chemical graph, generic dependency engine
or retained mutable view is introduced. Live member bindings use existing private
Atom/Bond UIDs; public reads and record capture translate to current ordered rows.
Removal subsets/remaps applicable relations; new sites have unknown occupancy/
disorder and newly supplied ordinary finite Bonds use the existing zero-image
convention. Removed bindings are pruned through the actual topology boundary.

The sole enclosing MoleculeRecord.cell becomes a tuple:
`(hotpot.crystal/1, matrix, group, relations, bond_images, representation,
origin_binding, image_map, bond_origins)`. Group/operation entries reuse the
immutable SpaceGroup/SymmUnit value domain and its sole existing reducer.
Relations are exactly`(equivalent_images, occupancies, disorder)` in canonical
current-row order. Schema/map validation belongs to the private crystal codec;
there is no second six-scalar loader. MoleculeRecord's current pickle generation
moves6→7 with the same24fields; old generation6 rejects. Existing scalar
construction encodes directly into this one record.

Independent content has no historical origin/image map. Generated content retains
one immutable historical source binding (source basis/group and useful source
site/Bond descriptors), plus all actual contributing image/Bond origins. This is
historical attribution, never a hidden pristine reconstruction path or a second
active chemical graph. Repeated actions compose/remap that history. A later
manual added subject has an explicit unavailable origin entry, not a fabricated
old source row; a dependent full-origin projection rejects locally if it needs
that missing map. Existing subject origins remain historical after edits.

## Publication, aliases and currentness

A complete cell/group/relation operation prepares and validates a candidate, then
swaps the one state under the contained Molecule's factual transaction. A narrow
closed Crystal undo entry restores state, both association directions and any
new/detached associations. Topology journaling enrolls that entry; restoring only
an old pointer to a mutated Crystal is insufficient. Coordinate-preserving cell
changes use the same transaction as the coordinate write. Cell-only construction
uses the same mathematical validation and atomic tuple publication.

Direct construction cannot steal an active association. Molecule.create_crystal
explicitly replaces/detaches the previous association; the old standalone Crystal
keeps its independent cell/group and no longer claims the Molecule's contents.
Copy/component/merge and complete-record installation translate real members and
images once. Compatible merge checks full basis/group/representation and historical
frame, then concatenates/remaps site/Bond relations; no silent choice of one
incompatible context. A context-free nonempty component cannot silently become
periodic content with invented images.

An attached Crystal's reducer resolves the reconstructed Molecule's active Crystal,
and the Molecule pickle extra retains that reference. This preserves combined
`(molecule,crystal,atom,bond)` aliases without a second serialized cell record.
Detached cells serialize the same closed codec. UID restoration installs cell
bindings after restoring actual member IDs, like existing frame restoration.
Copy/pickle isolate writable numeric/chemical state; immutable group values may
be shared. Arbitrary ordinary Crystal labels are not chemical facts.

Cell identity hashes the oriented basis only. Group/site changes invalidate no
calculation that consumes unchanged cell/Cartesian facts. Symmetry queries read
the actual current group/relations and do not cache results. Supplied equivalence
relations remain inspectable assertions after ordinary coordinate edits; operations
that rely on identification revalidate current coordinates/facts/edges before use.
Copy/pickle do not turn an old assertion into a newly verified scientific result.
Cell/group/relation setters reject incompatible explicit relations atomically.

## Mathematics and connected consumers

Pure helpers in`rules/mathematics/cell.py` own positive oriented3×3 basis validation,
standard six parameters, canonical scalar basis, metric compatibility, orientation-
preserving scalar replacement and Cartesian/fractional conversion. Constructors
retain full orientation; alpha=angle(b,c), beta=angle(a,c), gamma=angle(a,b).
Canonical rows use a along+x, b in+xy and positive volume. Scalar setters preserve
the rigid orientation and Cartesian positions. Existing15A tolerances apply.

The private crystal codec owns canonical maps, capture/install/subset/merge only;
`operations/crystal.py` owns complete changes; carrier methods are thin.
Open Babel/native reverse consumers receive a prepared six-parameter projection,
explicitly losing full orientation/group/images. Native ABI stays unchanged.
External objects with no periodic-image information import that information as
unknown, rather than inventing zero images. No Crystal route calls OB/RDKit/CCDC.

Before the connected checkpoint, exercise standard/rotated/skew bases, both
preservation modes, active association replacement, member reorder/removal,
copy/components/merge, combined cold pickle aliases, source identity/invalidation,
actual failure rollback and prohibited-API isolation at source and actual installed
CP311. Broaden for the schema's actual affected business, not private checker
status. Molecular actions/findings/CIF and Stage14D still have their owning gates.
