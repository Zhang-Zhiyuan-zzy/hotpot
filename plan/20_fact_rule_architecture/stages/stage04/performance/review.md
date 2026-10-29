# Stage4 local view and Crystal optimization review

This is the approved local optimization of the implemented Stage4 checkpoint
`0bc19f824a193b60b3ef93d89cd852079494baa8`, whose production tree is
`7cd4f7ad4c7451ec555210b456e5dba661b5c0da`. It remains inside the existing
Stage4 entry and [view contracts](../contracts/review.md). It neither opens a
new stage nor changes the main plan. [Profiling evidence](profiling/evidence.json)
binds the actual commands, inputs, source/native hashes, failures and results.

The recorded full-entry read's main-plan identity is still current. The scope
audit companion has since received90 appendix permalink repairs and an added
explanation; normalization against the recorded source proves its policy text
unchanged. Evidence preserves both document hashes and that check rather than
claiming a new full read of the changed appendix.

The original formal comparison remains in [performance evidence](evidence.json):
graph/search/geometry was4.053 times B20, numerical screening0.997 and Crystal
conversion1.103 under the frozen three-warmup/ten-sample policy. These unfavorable
measurements remain visible. The diagnostic runs below are not replacement
samples, an exception request or performance acceptance.

## Diagnosis and approved boundary

Read-only profiles verified the unchanged B20 source (`e6082f5`,523 reference
files) and Stage4 checkpoint (578 reference files) before and after execution.
They used the original workload binding, input and loops, three warmups and one
instrumented invocation on CPU0–63 with one BLAS/OpenMP thread.

The graph fixture actually constructs840 fresh views in one invocation:
20 queries times18 angles plus24 torsions. Previously each view reacquired the
owner lock, reread UIDs, selected and validated its own table rows, and captured
the same coordinate/provenance source separately. Instrumented cumulative
`_sequence_views` time was59.045ms;840 `_geometry_snapshot` calls accounted for
29.458ms inside it. B20's former carrier caches were already populated after
warmup; those caches cannot be restored under the accepted architecture.

For Crystal, the extracted mathematical body and B20 body had similar diagnostic
costs. The additional per-call local import and wrapper were the local target.
The main plan's Section6.1 permits function-local imports when needed for cycles;
binding this existing downward pure helper at module initialization creates no
new dependency family or architectural exception.

Exclusive implementation ownership was limited to the four production files
below, `test_view_identity.py`, and the necessary Crystal convention test.
The parent additionally approved extracting the existing pair-table slice
expression into one shared private helper and correcting first-match Bond
selection. Reports, formal performance records, shared helpers, the verification
map and final gate execution remain with the project owner.

## Implemented behavior

- `operations/geometry.py` captures Atom membership, UIDs, source capability and
  geometry once under the collection's existing owner lock. It reads raw Bond
  endpoints and the actual sole topology index, preserving rule input order.
- Core's private `_new_sequence_view(cls, atoms, rows, complete_paths, *, source,
  uids, geometry)` consumes those prepared values. A closed initializer writes
  only the existing five fields: `_atoms`, `_bonds`, `_view_source`, `_view_uids`
  and `_view_geometry`. Every requested Angle/Torsion remains a fresh object.
- `views/_source.py` owns the one existing three-component encoding and the
  extracted `_member_geometry` byte-selection helper. Batch sequence views and
  `AtomPairs.__getitem__` use that same helper. Direct constructors and restored
  views retain their existing source-binding path and share the private field
  initializer. No per-view table selection or repeated lock is needed in the
  already locked batch path.
- `crystal.py` binds the existing pure `_matrix_to_params` function directly as
  its private staticmethod. `from_matrix` still calls `cls._matrix_to_params`,
  including subclass overrides. Mathematical source and arithmetic are unchanged.

The Bond lookup also fixes a real read-view selection gap, not just runtime
cost. Complete imported records permit hidden Bond endpoints to overlap active
or other hidden endpoints. The prior Stage4 batch dictionary overwrote an
earlier active Bond with a later hidden one. The temporary lookup now keeps the
first match while traversing active, hidden-metal, then hidden-covalent members.
Generated active paths therefore retain the same Bond as `Molecule.bond()`.
Constraint-only extra paths retain their existing first hidden fallback and
missing-path `None`; legal imported records are not rejected or rewritten.

No public signature, class name, permission or return-container type changes.
The private Crystal method now refers to the pure helper itself rather than a
facade wrapper; its public constructor and `from_matrix` signature are preserved.
Geometry remains member-ordered float64 bytes, masked native uint32 presence
bytes, and three uint8 producer bytes per member. Signed zero, NaN payload,
missing presence, explicit zero and every existing producer code are retained.
Only relevant member bytes enter each view; batch scratch data creates no
unrelated-coordinate dependency or mutable alias. Repeated first/last Torsion
members, captured Bond inspection, stale-source rejection, ordered UID
constraints, pickle/deepcopy and transaction rollback retain their contracts.

No carrier cache, second index, revision, record-layout field, native change,
scientific algorithm, benchmark input, loop or sampling change was introduced.
Later carrier/result, Ring writer/cache, physical relocation and native-build
stages remain outside this work.

## Executed focused checks

The second actual invocation passed320 tests in16.26s using
`/tmp/hotpot-plan20-env/bin/python -m pytest -q --import-mode=importlib`, the eight
exact target paths and `-ra` recorded in the evidence. Coverage includes view
freshness, collection/standalone byte equality, all producer codes, presence and
NaN/signed-zero inputs, relevant/unrelated source changes, provenance/constraint
rollback, current/stale serialization, repeated-member torsions, imported
overlapping Bond partitions, builder parity, topology determinism, hidden-bond
restoration, pair observations and the architecture ratchet. The Crystal test
also preserves subclass conversion dispatch. All578 source/native reference
files and all selected tests were stable during this run.

The first invocation was319 passed and one failed architecture-ratchet test in
16.29s. Three newly added direct-Core test imports caused the failure. The tests
now obtain standalone constructors from existing view types, adding no import
path or public export. Production bytes and the allowlist did not change between
attempts. Both the original failure and its detailed architecture artifact are
retained. Six pre-existing plugin escape-sequence warnings occurred in each run.
`git diff --check` also passed.

## Diagnostic result and remaining gates

The optimized diagnostic repeated the same three manifest workloads with three
warmups and one instrumented invocation. All source/native bytes were stable;
all warmup and profiled counters matched: graph20 calls with2 rings/5 path
vertices/18 angles/24 torsions/78 pairs/2 hits, screening100 calls with the same
two outcomes, and Crystal200 calls returning8.0.

| Diagnostic call | Before | Optimized |
|---|---:|---:|
| Fresh sequence objects |840 |840 |
| Geometry snapshots and table selections |840 each |40 each |
| Table row normalization |860 |60 |
| Sequence batch cumulative instrumented time |59.045ms |24.875ms |
| Private Crystal facade-wrapper calls |200 |0 |

These are call attribution observations. The optimized graph profile still
contained50.813ms of instrumented work versus24.956ms in the earlier B20 profile;
the remaining search cost and unavoidable fresh-view work have not been accepted
as parity. Instrumented runs do not establish a median, confidence interval,
memory result or numerical speedup. They were run on a shared host without an
exclusive allocation.

The project owner must revalidate the affected technical/source/wheel gates on
the frozen changed bytes and repeat the unchanged official three-warmup/ten-sample
comparison. This focused source run does not replace the five-version matrix,
installed-wheel/native evidence or full default suite. Global71-workload,
end-to-end, scientific and memory acceptance remain open. Stage4 stays in
progress until its required gates actually pass.
