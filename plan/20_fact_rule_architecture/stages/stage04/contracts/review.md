# Stage4 implemented owner and view contracts

These are implemented details of the accepted Stage4 mathematical extraction
and common view permissions. The parent additionally authorized one finite
private N02 field to preserve existing scientific pair observations. This
review does not approve a new public pair-result DTO, move a later-stage
boundary or mark Stage4 complete. Exact fields are in [schema.json](schema.json).

`Molecule._constraints` owns one frozen `_ConstraintSet` with ordered UID
tuples for true angles and torsions. Fresh views project that owner; no view
has a local constraint value. Record capture/install, subsets, merge, copy,
working-copy publication and current pickle use the same entries. Removed
members cannot acquire a replacement row's constraint. Atom coordinate and
Bond constraints remain with their current owners until Stage7B.

`Molecule._pair_wiberg_bond_orders` stores only explicitly supplied values,
including zero, keyed by ascending non-reused Atom UID pairs. N02 field23,
`pair_wiberg_bond_orders=()`, carries row/row/float triples. Its constructor
normalizes unordered row pairs and rejects malformed widths, self-pairs,
duplicate keys and invalid rows. Copy and merge allocate destination UIDs;
subsets drop entries whose members disappear. Current pickle restores UID
gaps before reconstructing this owner. Complete FF source checks include the
observations so a concurrent supplied value cannot be overwritten.

This temporary preservation owner is separate from generated pair mathematics.
Stage7B/7C still own the complete pair/result attachment and provenance policy.
The existing Wiberg scalar setter remains a named factual commit; it checks
the member identities rather than treating an unrelated topology change as a
change to a supplied observation. Path and distance calculations check their
own topology and geometry sources. `attrs` remains Wiberg then path length;
path length remains the number of vertices, including both endpoints.

`angles`, `torsions` and `atom_pairs` return fresh caller-owned objects.
`update_angles`, `update_torsions`, `_retrieve_torsions` and the three retained
carrier caches are removed. `AtomPairs` remains the public spelling and is a
read-only `Mapping`; no alias or new stale-view exception is added. Inherited
dict writes, deletion, `clear`, `update`, `setdefault`, `pop`, `popitem`,
`fromkeys`, dict unions and `clear_not_exist_pairs` are absent. The existing
`copy()` read helper returns an independent plain dict; `AtomSeq.bonds` keeps
its independent-list return contract over captured fixed members. Standard
`copy.copy`, deepcopy and current pickle use the source-preserving codec.
`getdefault` is lookup; `update_pairs` validates the fixed source and adds no
members. Existing keys/items/values/get/iteration/length, index matrix,
condensed distances and path projections remain available. No current source
consumer uses the removed dict-only API. The default table generates pair
views lazily and prepares its coordinate source once, without an O(N²) object
cache on the Molecule.

Views retain fixed Atom and captured Bond references for inspection after a
source change. Calculations and row-dependent operations check N01 owner,
the actual sole topology-index reference, captured membership counts and
member UIDs. Geometry checks only the relevant coordinate rows plus their
presence/source metadata. No per-view whole-molecule digest, carrier revision
counter, new exception or second index is introduced. Pickle/deepcopy covers
each view/table alone and both graph-root orders; a stale source stays stale.

Scalar pair updates enroll old value or absence against the actual dictionary.
Complete topology installs a prepared replacement and retains the original
owner reference. Fast Atom registration and Bond append enroll actual index
entry deltas before their stores, even after a complete topology enrollment.
Undo replays these deltas in reverse without rebuilding either dictionary.
Real allocator tests establish the tested dict-bucket boundary, original
exception identity and restored view capability. They explicitly observe
other Python allocations and make no claim of arbitrary-OOM immunity.

`InternalCoordinates` has one implementation at its mathematical path; its
complete class AST is unchanged. Crystal's formula keeps the characterized
first-two-row alpha convention and adds no matrix property. Core query
facades pass their existing materializers into operations until Stage9's
physical relocation, avoiding new imports back to Core. Native names, source
and build paths, ABI, scientific algorithms and the sole MCA Python-RDKit
route remain unchanged.
