# Stage15 — Crystal spatial symmetry and molecular expansion

Owner-directed additional Plan20 branch. Root accepted the concrete
[Stage15A contract](domain/contracts/decisions.md) on 2026-11-05; standalone exact
SpaceGroup/SymmUnit production is root-accepted at the
[297-test source/five-version installed checkpoint](domain/validation/exact_symmetry_report.md).
Connected cell/lifecycle migration is root-accepted at the
[559-test source/installed CP311 checkpoint](domain/validation/connected_cell_report.md).
One-placement action is root-accepted at the
[874-test source/installed CP311 checkpoint](domain/validation/action_report.md).
Cell generation is root-accepted at the
[1,038-test source/installed CP311 checkpoint](domain/validation/generation_report.md).
Placed relations have a bounded
[1,068-test CP311 checkpoint](domain/validation/relation_report.md).
Remaining Stage15D periodic findings and15E–F remain open. The
[detailed plan](review.md) starts with the complete proposed **Crystal,
SpaceGroup and SymmUnit** tree, then one exact affine algebra, molecular
copying/cell expansion, complete possible-molecule relations and periodic
geometry/topology observations.

`SpaceGroup` is the proposed immutable authority for group operations, exact
setting and translation lattice. `Crystal.space_group -> SpaceGroup | None`
is its active association; `Crystal.symm_units` is a read-only derived view
which delegates to it. Crystal owns its physical cell and Molecule owns all
chemistry/Cartesian coordinates/topology. There is no independent Crystal-level
setting/operators/context fact. Immutable group values may be shared safely.

`Crystal.operate(g)` returns **one fresh transformed placement** of its current
content with the same active cell and group. It leaves the source intact;
replication is performed by `unit_cell`/`expand`. It does not extend the group
or return an original-plus-image union. SymmUnit's `@` (self after other), `~`
and integer `**` delegate to `compose`, `inverse` and `power`, preserving every
lattice translation. Equality/hash uses the full element and immutable context.
The [operation table](operation_table.md) specifies actual `R,t`, exact
reconstructible `xyz`/Seitz forms, axis/plane/sense/pitch and closure limits.

The [indexed complete catalogue](space_group_catalogue.md) gives all **230
ordinary 3D space-group types**, their current H–M and explicit Hall/setting/
origin, and every actual conventional-cell representative. The
[exact data](space_group_catalogue.json) includes all **4,425** representatives.
Reconstruct the infinite group with integer translations and retain carries;
230 types do not enumerate all Hall/axis/origin alternatives. Pinned official
source provenance, legacy-name aliases and preserved licences are linked in
that catalogue. Its document/data generation selects no runtime dependency.

Supplied content need not be a minimal asymmetric unit. Complete transformed
overlap **and the same graph/site bijection** support a possible relation;
queries do not authorize destructive deduplication or decide scientific truth.
Finite closed-loop linking is distinct from piercing; catenanes remain useful
observations and zero linking number is not an unlink certificate. Crystal
behavior uses Hotpot math/topology, with only a separately justified table/CIF
lookup candidate. No direct or indirect OpenBabel/RDKit/CCDC Python API route.

Standalone exact symmetry and connected cell/lifecycle are delivered. Molecular
actions, findings and format closure remain accepted implementation scope. The
connected migration replaces the six-parameter ordinary cell record with one
oriented record; six parameters remain only at declared external projections. Stage15A settles
the exact public/export/codec/CIF/dependency/tolerance choices and the existing
topology foundation's self/parallel periodic-edge limit in its accepted contract. The group catalogue's
exact arithmetic checks do not pass molecular, numerical or runtime gates.

Stage14D prepares against 15A's accepted schema, then consumes 15B/C's implemented
domain and affected 15E format closure. Stage15 publishes/closes its own domain
without waiting for the shared PyG fixture; Stage14 owns that integration.
All runtime, scientific, package and overall-task gates remain with the parent.

## Complete record index

- [Full proposed API tree, ownership/lifecycle and linear implementation/acceptance plan](review.md)
- [Supported operations, magic methods, full affine closure and primary examples](operation_table.md)
- [All 230 types with full finite operator lists, exact data and source/licence provenance](space_group_catalogue.md)
- [Main-plan registration](../../fact_rule_architecture_implementation_plan.md#18-additional-owner-directed-branches-stage14-and-stage15)
- [Stage14 useful training information and chemical query/review plan](../stage14/review.md)

[Stage catalog](../README.md) · [Project index](../../README.md)

## Additional record links

- [space_group_catalogue.COMCIFS-LICENSE.txt](space_group_catalogue.COMCIFS-LICENSE.txt)
- [space_group_catalogue.GEMMI-LICENSE.txt](space_group_catalogue.GEMMI-LICENSE.txt)
