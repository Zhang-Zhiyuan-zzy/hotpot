# Stage3 private storage review

This is preparation within the Stage3 entry recorded in
`stage3_entry_review.json`, after foundation commit `f9f87da`. It does not
complete Stage3 or start a later stage. The main plan and four companions were
fully read at entry and again for the revised interface review; the immutable
read records are indexed by `stage3_foundation_evidence.json`.

The separate `8a7c6a6` coverage fix includes package-root Python files and
restores three never-retired original debt entries. Its correction record
supersedes the foundation's282 count; the correct remaining count is285.

## Current implementation boundary

`structure/_fields.py` is the sole declaration of the existing 19 numeric
slots and scalar read codecs. The current Atom facade derives its existing
dict from this declaration, preserving field order, dict behavior and codecs.
The approved batch fields and aliases are private declarations; they do not
publish new methods or change the permissions of existing setters.

`structure/_atom_table.py` implements the private numeric candidate. No
Molecule stores it yet. Current Molecules continue to own their existing Atom
arrays until the complete writer migration is ready. The candidate has two
float64 blocks: 16 non-coordinate slots and a contiguous `(N,3)` XYZ block.
Its 19 column views address those blocks without owning duplicate values.
Capacity growth does not create a second topology or UID index.

All raw slots remain float64, including values such as `2.5` in a Boolean
projection slot and source IDs beyond int64. Logical int/bool conversion
belongs to scalar projection, never the stored value. Row, column, coordinates,
full matrix and privately selected exports are independent arrays. No public
field-selective matrix API, MCA column, isotope column, or new result owner is
introduced here.

The table accepts canonical numeric columns. Scalar and batch updates share
`update`; conversion and shape/row checks precede installation. Molecular
owner/source checks, presence/provenance, dependency state and concurrency
control belong to the pending complete commit boundary. Tests of this numeric
primitive cannot establish those later contracts.

## Evidence and limits

All storage experiments keep their exact runner and measured leaf-module
bytes, validation, raw attempts, statistics and retained-memory accounting.
The private table experiment adds its real validation/update costs to the
same eight workloads used for the earlier layout comparisons. It still
excludes Atom dispatch, UID lookup, molecular commit locks, presence/source
metadata and scientific calculations. Current-array timings are storage-only
comparators, not full public-property timings.

The first private-table run completed all 1560 attempts and 39 validation
checks. It exposed approximately 8x scalar-write overhead and 16.5x small
remap overhead against bare per-Atom numeric arrays. Those findings prompted
optimization before integration. Independent review also found a write-then-
interrupt rollback gap; an interrupted current column must be restored along
with previously written columns. The original run remains available under
`stage3/storage-private-table`; revised source must use a different output
directory. The refined run also completed all 1560 attempts and 39 validation
checks. It reduced scalar-write overhead to roughly 4.7–4.8x and N16 remap
overhead to 8.6x, while scalar reads remained about 1.5x. Full-matrix export
ratios for N16/256/4096 were 0.762/0.209/0.159. These are mixed results, not a
production performance pass; scalar dispatch and molecular transactions still
need a measured integrated design. Retained-memory figures describe freshly
constructed stores, excluding temporary growth/remap peak allocations.
Final measured results and tests are indexed by `stage3_storage_evidence.json`.

## Required complete production cutover

Before replacing authoritative arrays, prepare the following together:

1. Closed Atom update normalization, including alias agreement, selected
   membership, presence/source and all touched fields.
2. Owner binding, stable Atom/Bond UIDs and non-reusing high-water marks,
   detached-row behavior, and one shared row remap for values and metadata.
3. The complete factual import record and builder, allocating known rows once
   and constructing topology exactly once without implicit chemistry.
4. The one commit/rollback boundary for scalar, batch and coordinate writes.
   Stage5 has not yet removed eager chemistry: the existing atomic-number
   setter's element/electron/valence/H writes and the bond-order setter's
   order/kind/two-endpoint-H writes must remain complete transactions until
   that stage. Do not copy those chemical algorithms into the table.

The same production change must migrate Atom reads/writes and constructor
imports, Molecule coordinate/matrix access, copy/components/merge/replacement/
removal, converter imports and force-field working-copy rollback. Existing
failure fixtures must retain their object identity, topology and scientific
state assertions; old writable-array identity assertions become independent
snapshot and complete restoration assertions. A writable array proxy or a
per-Molecule old/new-storage switch is not an acceptable transition.

`updata_atoms` and the full float64 `atom_matrix` property retain the owner's
approved spelling and shape. The earlier `atom_attr_matrix` name's removal
versus same-source alias remains under explicit clarification. No old name is
removed and no duplicate public alias is introduced by this checkpoint.

GitHub cadence remains one push per three commits. Preparation commits do not
change the production cutover boundary or relax any Stage3 gate.
