# Molecular crystals and exact spatial operations

`Crystal` owns one oriented cell and its association with a contained `Molecule`.
The Molecule owns atoms, Cartesian coordinates, chemistry and its simple Bond
graph. An immutable `SpaceGroup` owns the current setting and complete affine
operations; `Crystal.symm_units` is a derived view. Missing group information is
`None`, rather than an assumed P1 group.

The row-basis convention is `cartesian = fractional @ crystal.matrix`. Lengths
are angstrom and angles degrees: alpha is angle(b,c), beta angle(a,c), gamma
angle(a,b). `matrix` returns an independent array. Its orientation is preserved
through the current Hotpot record, copying, pickle and the Hotpot CIF extension.
The six conventional cell parameters alone cannot preserve that orientation.

`mol.coordinates` reads and writes the sole Cartesian Atom facts, returning an
independent array. For a Molecule paired with a Crystal,
`mol.fractional_coordinates` converts through that Crystal's current oriented
cell; assignment converts back through the same cell and uses the atomic
Cartesian setter. Values are dimensionless and unwrapped, with no fractional
store or cache. Reading requires complete supplied Cartesian coordinates;
reading or assigning without a Crystal raises `CRYSTAL-CONTEXT-INTEGRITY`.

## Exact operations and one placement

```python
from hotpot import Crystal, Molecule, SpaceGroup

mol = Molecule()
mol.create_atom(atomic_number=6, formal_charge=0, implicit_hydrogens=3,
                coordinates=(1., 2., 3.))
mol.create_atom(atomic_number=8, formal_charge=0, implicit_hydrogens=1,
                coordinates=(2.4, 2., 3.))
mol.add_bond(0, 1, bond_order=1)
group = SpaceGroup(4)  # shipped reference setting of P 1 21 1
crystal = Crystal(8, 9, 10, 90, 90, 90, molecule=mol,
                  space_group=group, bond_images=((0, 0, 0),))
g = group.lift(1, (1, 0, 0))
placed = crystal.operate(g)
assert len(placed.molecule.atoms) == 2
assert placed.space_group == group
assert g @ ~g == group.identity()
```

A SymmUnit is one complete exact affine element `{R|t}` in a bound group context.
Fractional column action is `R*f+t`; `g @ h` means g after h. `~g` and integer
`g**n` retain full integer translations. `reduce_to_cell()` separates a canonical
representative and its integer carry. Named `compose`, `inverse` and `power`
methods have the same meaning. Operations from different contexts require an
explicit valid rebind. Finite-order point operations do not imply that their
screw/glide affine elements have finite order.

`SpaceGroup(1..230)` selects the shipped reference setting for each ordinary
3D type. It does not enumerate every origin/axis/Hall alternative. An external
complete operation list uses `SpaceGroup.from_operations`; it must include the
whole finite cell quotient, not just generators. `change_setting(P,p)` means
`f_old=P*f_new+p` and transforms the full translation lattice. Exact rational
strings/Fraction values preserve the intended map; arbitrary binary floats are
not silently converted to guessed crystallographic fractions.

## Cell filling, expansion and finite molecules

```python
filled = crystal.unit_cell()
assert filled.space_group == SpaceGroup(1)
assert len(filled.molecule.atoms) == 4
assert [len(part.atoms) for part in filled.unit_molecules()] == [2, 2]
supercell = crystal.expand((2, 3, 2))
assert len(supercell.molecule.atoms) == 48
```

`operate` returns one fresh placement. `unit_cell` and `expand` replicate current
content and publish active P1, retaining immutable historical source binding
and all source image maps. They do not apply historical symmetry again to
already generated content. Supplied contents need not be a minimal asymmetric
unit. Numeric overlap or a possible relation never authorizes deduplication.
Only an explicit validated equivalent-images relation can identify generated
sites; occupancy/disorder is retained without choosing a chemical model.

Each Bond has a directed integer image in complete Bond order. `(0,0,0)` means
the actual unwrapped endpoint row, while `None` means unavailable image
information. Finite extraction needs a consistent zero-voltage lift. Periodic
components with nonzero cycle voltage are retained as periodic and excluded only
from the dependent finite-molecule extraction, with a precise warning. The
current simple Molecule graph cannot represent self/parallel periodic quotient
edges: the dependent full projection rejects rather than collapsing edges,
fabricating atoms or automatically enlarging the cell.

## Relations, geometry and finite links

`same_molecule_candidates(tolerance=...)` requires one complete common placed
coordinate and chemical/site bijection. It does not fit an arbitrary rigid
alignment or wrap each atom independently. Unknown required chemistry remains
unresolved. `molecular_symmetry(tolerance=...)` asks which supported full
operations stabilize the fixed finite unit, including its actual placement.

```python
findings = crystal.geometry_findings(
    coincidence_tolerance=1e-8, close_contact_cutoff=0.7)
```

Tolerances and cutoffs are positive angstrom values. Metric-bounded periodic
neighbors handle skew cells, without a fixed fractional 27-image restriction.
Actual bonded image pairs are excluded from ordinary crowding, while bonded
coincidence remains observable. Finite selected cycles use the existing native
surface predicates and retain intersections, surface assumptions and numeric
indeterminacy. A source row at another image is a different geometric vertex.

`linking_number(first_cycle, second_cycle, tolerance=...)` takes two explicit
oriented sequences `(current_atom_row, integer_image)`. Every edge, including
the closing edge, must be present in the actual directed Bond/image graph.
A supported separated simple finite pair returns an integer whose numerical
enclosure isolates exactly that integer. An unavailable case returns `None`
with its precise reason. Nonzero linking, including a catenane, is an observation;
zero is not an unlink certificate. These finite queries do not classify all
infinite-network topology or decide a structure's experimental truth.

Findings are immutable `(code, subjects, measurements)` tuples. Non-hard
observations emit coded RuntimeWarnings and return available values by default;
Python warning filters can promote a selected code to a break. Findings are
prepared before emission. No query mutates, repairs or deduplicates source facts.

## CIF and persistence

`Crystal.from_cif(source, block=None)` accepts CIF text or a pathlib/os path;
multiple data blocks require explicit selection. `Crystal.to_cif()` returns text.
Only CIF calls require the optional `crystal` extra (`gemmi>=0.7.5,<0.8`). Gemmi
handles scalar/loop syntax; Hotpot owns symmetry, coordinates, topology and all
chemical meaning. No Crystal path calls Open Babel, RDKit or CCDC Python APIs.

Bare CIF imports only the supplied supported facts. A missing Bond loop gives
no bonds; missing periodic-image information stays unknown. Explicit operator
lists determine the active group, independent of their source IDs/order. A
number or symbol without a verifiable setting/origin does not fabricate a group.
Element/label conventions and type mappings are bounded; oxidation text is not
silently treated as formal charge. No H, connectivity or disorder perception is
run. Unsupported full projections retain their source text and exact cause.

The standard CIF view contains conventional cell, site and supported Bond data.
A closed `_hotpot_state_json` extension retains the oriented current record,
exact group/setting, source image maps and otherwise unexpressed current Hotpot
facts. Its standard and extension views must agree within the declared emitted
precision; a conflicting edited pair rejects rather than restoring hidden stale
values. Numeric output uses 17 significant digits. Input esds and unowned
experimental/refinement fields remain in the archived source text; they are not
invented precision or active modeled facts.

CIF is a current structural projection, not full Python-session persistence.
Additional conformer observations, MCA attachments and arbitrary runtime extras
are outside this format; actual omitted material is reported precisely. The
current-version Hotpot pickle remains the existing complete lifecycle route.
Historical pickle/schema loaders are not supplied.
