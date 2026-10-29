# Stage3 native factual commit and seed undo review

Status: the parent accepted this bounded private execution revision on
2026-10-29. The revision is ready for the engine owner to implement; its
correctness, supported-Python and performance gates remain open. It does not
complete Stage3 or authorize Stage4. The original [native entry review](entry_review.json)
remains a historical entry snapshot. The current reread, source identities,
allocation evidence and bounded assignment are in
[undo revision entry review](undo_revision_entry_review.json).

## Accepted boundary

Keep the sole Atom fact table, existing Python `set` seed containers, existing
`_dirty_seeds` tuple, one `Undo` capsule and one ordered transaction journal.
Root and outer commits continue to share the installer and restorer. No new
public interface, canonical container, persistent owner, parallel writer,
fallback or scientific algorithm is introduced. Actual N01 and the seven-field
immutable N05 retain their existing meanings. Operation-local undo storage is
not a second current fact store.

The change replaces membership-boolean restoration with exact physical seed
entry restoration and retained bucket storage at clear/rehash boundaries.
Restoration must perform no allocation and no `PySet_Add`, user conversion,
hash/equality callback or other fallible Python operation. Saving the original
exception and poisoning the matching outer transaction remain mandatory even
when a caller catches the failed inner commit.

## Why membership restoration is insufficient

The archived native-v1 binary has SHA-256
`7c66215f15b3399d310f17edc1b396e721383bbd3a629528a4e697f1fe9b4a57`;
its C++ source has SHA-256
`cdf3de18d2e02748a7622ee08285b384b3949a214fa3552ad8927accffde6bd9`.
Fresh CPython3.11 subprocesses loaded that actual archived binary. The four
live Python owners used by the probe matched the v1 source snapshot before
each run and retained their hashes through each run.

A task-local C helper wrapped `PYMEM_DOMAIN_MEM` and rejected only the selected
allocation size while real rollback ran. Both original exceptions were
precreated objects. These were heap-allocation failures, distinct from the
private deterministic post-store exception checkpoints.

| Scenario, each with original KeyboardInterrupt and MemoryError | No-rejection control | Allocation rejected during rollback |
|---|---|---|
| Clear64 seeds: mask127 becomes mask7 | Original exception identity and all values/status/seeds restored; allocations512/2048 bytes | Reject512 bytes: only5/64 seeds restored, status undo skipped, original exception replaced by MemoryError |
| Seed76;75 scalar formal-charge writes discard UIDs; Python `_seed` adds a new UID and shrinks mask127 to mask15 | Original exception identity and all values/status/seeds restored; allocations1024/4096 bytes | Reject1024 bytes: only8/76 seeds restored,68 numeric writes remain, original exception replaced by MemoryError |

Actual command:

```text
/tmp/hotpot-plan20-env/bin/python /tmp/plan20-undo-allocation-review-20261029/build_and_run.py
```

The driver records the compiler argv, eight fresh-process argv, affinity24–31,
environment, interpreter, native/source/helper identities and results under
`/tmp/plan20-undo-allocation-review-20261029`. The maintained entry JSON binds
the exact artifact hashes and embeds the eight outcomes. All subprocesses
exited0 because the probe reports the caught exceptions and state; the four
allocation-rejection outcomes are confirmed defects, not passing atomicity
checks. They cover CPython3.11 native-v1 only.

Retaining capacity alone is insufficient: reinsertion may choose different
tombstones and then consume a previously unused bucket. Even without a forward
rehash, undo using `PySet_Add` can request a rehash. The accepted restorer
therefore restores physical entries rather than attempting to make membership
reinsertion safe by copying only the active members.

## Preparation, installation and restoration

1. Prepare all fallible conversions, row/UID/owner/source rechecks, array
   dtype/shape/stride/writeability checks, seed hashes, candidate storage and
   complete undo entries before the first authoritative store. Rebind the
   current table after conversion callbacks. Retain the actual physical NumPy,
   seed-set and UID/key owners.
2. For ordinary non-reshaping seed changes, capture the exact old bucket entry
   (key and cached hash), its physical set/storage and the required `fill`/`used`
   counters. Use one closed UID lookup and one installer. A batch uses an
   operation-local delta overlay of changed entries to prepare colliding
   insertions without copying the complete set. Reserve this metadata before
   stores. Entry/counter writes and reference ownership form one undo action.
3. Before a clear, prepare an empty set. Before an addition that actually needs
   rehashing, prepare its after-candidate with the same logical membership
   semantics. Swap the set bodies during installation, preserving the original
   active Python set object's identity. The inactive operation-local candidate
   then owns the actual old bucket storage, including tombstones and capacity.
   A compact copy alone does not preserve that old physical storage.
4. Enroll the complete capsule in the existing journal before any store. Root
   scalar commits use the same local `Undo` without creating a Python outer
   transaction. Numeric bits, sources, presence, statuses and seed entries/body
   transitions remain one complete write set.
5. Restore entry beforeimages/counters or swap the saved physical body back;
   never rebuild membership. Restore actions that depend on each other replay
   in reverse installation order. Keep the existing transaction order around
   topology/registration restoration; seed undo addresses its captured physical
   set, rather than whichever table is currently attached to the Molecule.
   Track whether each prepared body swap actually installed: enrollment or an
   earlier numeric store can fail before any seed swap. An uninstalled
   after-candidate must never be swapped into active storage during restore.
   Set the applied state in the uninterrupted native swap segment before its
   post-store fault checkpoint, and restore only the applied transitions.
6. Make a successfully restored `Undo` cancelled/restored exactly once. An
   `install` failure can restore locally and leave the cancelled capsule in the
   outer journal. Later outer replay skips it; later writes have their own undo
   entries. Do not swap a body twice or allocate another snapshot for replay.

Expected work is proportional to the touched entries: ordinary scalar O(1)
expected, a k-entry batch O(k) expected, and a real rehash O(bucket capacity).
Clear prepares one empty set and swaps bodies; it does not copy all members.
There is no full-set copy for every scalar, full-molecule digest, raw-equality
short circuit or performance waiver. Collision behavior and auxiliary memory
must be measured honestly; the stated expected bounds are not benchmark results.

## CPython ABI and lifetime constraints

This is a deliberate dependency on each target CPython's `PySetObject` and
`setentry` representation. Compile against the matching3.10–3.14 headers and
inspect the corresponding probing, dummy-entry, load/rehash and body-layout
rules. Do not claim abi3 or cross-minor binary compatibility. The small closed
lookup belongs only to private builtin-integer UIDs; it must not admit arbitrary
user hash/equality behavior or become a general set implementation.
Record the exact CPython source/version/digest and required attribution when
adapting its probing algorithm; equal header offsets do not establish equal
algorithm behavior.

Body swaps preserve Python object headers, identity and weak-reference state.
They account for `fill`, `used`, `mask`, the active table, inline smalltable,
cached hash and pop finger as required by that interpreter. Bucket ownership
and references transfer exactly once; active keys are pinned through prepare,
install, rollback and disposal. Dummy entries are not ordinary owned keys.
GC tracking remains correct. Reference disposal during restore must be limited
to the closed builtin UID/key domain and must not invoke user finalizers.

Heap bucket allocations retained by an inactive candidate stay alive until
their earlier entry undo is no longer needed. For inline storage, the captured
set object stays alive and the saved inline contents return to that object's
own smalltable before earlier entry addresses are dereferenced. Never retain a
pointer into a temporary set's inline array as though it belonged to the active
set. Across successive rehash/clear transitions, reverse body swaps must restore
each prior storage generation before replaying entries targeting it. This must
also hold when topology growth/remap restores old table owners first.

The parent's standalone header-layout probe compiled on all five versions;
its exact argv/source/binary observations are bound in the revision entry.
This establishes header-layout observations only. Do not hardcode the observed
offsets or treat the probe as a rebuilt engine, refcount/GC, algorithm or runtime
atomicity result.

All active mutations occur under the existing owner lock and the extension's
CPython execution assumptions. Do not advertise free-threaded support without
the needed object synchronization. Other interpreter implementations and an
unbuilt ABI are unvalidated, not fallback execution routes.

## Complete writer and consumer migration

| Current owner/consumer | Required connected treatment |
|---|---|
| Native fact `prepare_undo` / `Undo.install` / `Undo.restore` | One seed preparation/installer/restorer for root, prepared, scalar, columns and multi-field writes |
| `_dependencies._seed` | Replace Python `set.update` on active seeds with the same native seed ingress, including all downstream groups |
| `_dependencies._expand_dirty` | Replace active `seeds.clear`; retain existing scope/row/status semantics and capture status before its stores |
| Core `Molecule._add_atom` pending-group copy | Replace active `set.add/discard` after `touch_dependency_seeds` with that same ingress |
| `_FactTransaction.touch_dependencies` and dependency capture helpers | Reconcile broad captures with exact entry/body undo; no membership-only restore or bypass writer may remain |
| New table, codec, `take`, dependency-state installation, detach and topology remap | Keep fresh-owner preparation and existing complete storage/topology restoration; verify no hidden in-place active-set mutation escapes the ingress |
| Dependency reads, serialization and copies | Continue consuming the actual existing set/frozenset values; no new canonical representation or reader fallback |

An implementation is not complete while an ordinary external Python
`add/discard/update/clear` can still mutate active seed storage. Read-only
frozensets on sealed detached tables remain read-only. Source searches must
separate fresh-container construction from mutations of an existing owner.

## Adjacent approved ingress refinements

The private native `_commit_atom_fields(atom, columns, *, producer, present=True)`
reuses the current normalizer and complete checks, constructs actual seven-field
N05 once, invokes the same installer/journal and returns None. Only Core's
existing multi-field branch is connected. The scalar branch, eager scientific
bodies, getters and constructors retain their semantics.

The actual `_AtomFactTable` class's `PyMemberDescr(T_OBJECT_EX)` code offsets
may be bound with strong descriptor/type references and compatibility checks.
Every call reads the current instance slot and still validates arrays. No
table, array or data pointer is retained as a cache. These are private execution
refinements; their cost still requires actual performance evidence.

## Required validation and remaining gates

Rebuild and run the complete affected-contract selection on CPython3.10–3.14,
including actual allocator-failure regressions equivalent to all eight cases
above. Verify original exception identity, full facts/metadata/status/seeds,
outer poison, root behavior, preparation/enrollment failures and local-restore
followed by cancelled outer replay. A deterministic after-store MemoryError
alone does not establish allocation-free restoration.

Exercise collisions, overlapping membership changes, repeated clear/rehash,
heap↔smalltable transitions, multi-generation pointer lifetime, GC/reference
ownership, batch preparation and topology grow/remap/removal. Include mixed
Python dependency/native fact operations and complete copied pending scopes.
Debug-allocator or sanitizer evidence must identify the actual binary and
coverage; unexecuted native ownership checks are recorded as missing.

Preserve the existing raw NaN/-0/fractional bits, immutable N05, post-conversion
source/member checks, readonly/strided arrays and non-reused UIDs. Complete
source/native builds, installed-wheel checks, focused/default tests and the
stage's scientific reporting and affected storage/ingress scaling comparisons.
The final global comparison still covers all71 frozen performance entries,
outliers, family/integrated results and memory under the existing policy;
its unmet items remain open at their owning gates. This revision neither
changes the frozen measurement policy nor adds a new Stage3 exit gate.
Microbenchmarks and prior Python/native-v1 tests do not close final acceptance.

The engine owner retains production-engine/test ownership. The Sol-max author
owns this bounded design record and the assigned Core/carrier wiring. New C++
undo/preparation helper structs must be counted as actual internal implementation
types in evidence; moving code or replacing booleans is not DTO consolidation.
If the implementation requires a new persistent container/owner, per-scalar
full-set copying, a second writer or different journal semantics, report that
scope change before proceeding.
