# Stage5 F1–F3 correction handoff

Status: the bounded structural corrections are complete and production/tests are
frozen for root handoff. Stage5 remains incomplete. The original Stage5 review,
477-pass checkpoint and its evidence are unchanged. This record continues the
existing stage; it does not approve or alter scientific H defaults.

The entry snapshot is `.plan20-artifacts/20261030/stage5/remote-checkpoint01/`
`source-snapshot.tar.gz`, SHA-256
`d7433533eb891364900c6666e9138d55fee8d45ece73c5540b00484ca857fa0c`.
Root released the source freeze for these three findings only. The existing
entry review and normative-document read remain the stage-entry record; no new
stage, DTO, public result or architectural route was introduced. Both Hotpot
skills were read for this continuation. [evidence.json](evidence.json) binds the
current seven changed source/test files, all 71 checkpoint identities, exact
commands, logs, native/interpreter/dependency identities and remaining gates.

## Corrected behavior

F1: VALENCE and IMPLICIT_H pending seeds now expand to the written Atom and its
immediate neighbours through one topology helper. Both scalar and batch writes
use that same lazy dependency path. H preparation also retains this expanded
UID set so relevant new input retries UNRESOLVED rows once. The existing
component scope for aromatic representation and whole-molecule charge coverage
remain. Explicit/imported H declarations, including zero, remain protected.
The focused tests verify remote-first reads, one rule call per affected Atom,
unrelated-component reuse, and no two-hop H invalidation. The pre-fix N-C to
N-O example observed old inferred N2 while the existing forced rule proposed
N4. That is a consistency observation under the retained algorithm. Final
regressions compare with the current pure rule and do not prescribe N4 as a
scientific default.

F2: the exact private signature is now:

```python
_commit_aromatic_assignment(mol, assignment, *, snapshot, preserve_sources=False)
```

The operation supplies its captured `_BondingSnapshot`; commit never captures
another one and has no compatibility default. Before any mutation, it checks
that proposal and snapshot share their captured source, that the source and
Atom membership are current, and that aligned/distinct Bond UIDs equal exactly
the snapshot Bonds whose two endpoint rows belong to `covered_rows`. This
handles both whole and partial component proposals. Metal/ligand selection
stays in operations: the default snapshot excludes metal-ligand Bonds, while
an explicitly captured full snapshot includes them. Structure adds no chemical
policy import. The trust boundary is the operation-owned captured input, not a
general hostile-caller validator.

All four real producers pass the input: explicit aromatic/Kekule assignment,
lazy aromatic preparation, Open Babel bond-order perception, and assembly flag
restoration. The OB path previously captured a source and selected Bond map;
it now captures the existing snapshot once under the same pre-backend lock,
with the source that also binds the cell. No new snapshot class is added.
Direct tests and the legacy molecular facade are migrated; the bounded fixture
already uses the named facade and needs no change. Positive whole, default
ligand, full-snapshot and partial component cases pass; missing, extra ligand,
wrong-component, source and Atom-membership cases reject without changing
facts or dependency state. Existing atomic rollback remains tested.

F3: `Ring.to_mol()` checks its representation binding under the owner lock
before the inherited factual export. Element or Bond-order changes now reject
through the existing RuntimeError, matching the Ring aromatic read. Held Atom
and Bond identities remain inspectable. Other AtomSeq exports retain their
existing topology policy; a current Ring and an Angle after an element change
both export successfully in the positive regressions.

No native source or binary changed. All six bonding scientific rule files and
the three-strict-xfail test file retain their original checkpoint hashes. This
correction adds no class or carrier definition. The root-owned ledger had
already been reconciled separately; this task makes no new global ratchet,
class-budget or performance claim.

## Actual validation

All paths below are under `.plan20-artifacts/20261030/stage5/corrections/`.
The recorder preserves exact argv/cwd/environment/source/test/native hashes,
stdout, elapsed time and before/after equality for each attempt.

| Attempt | Actual result | Coverage |
|---|---|---|
| `repro-attempt1` | 7 failed, 43 deselected; rc1 | Scalar/batch stale neighbour H; missing/extra/wrong-component Bond coverage; stale Ring element/Bond-order export |
| `focused-attempt1` | 80 passed, 1 deselected; rc0 | Dependency, bonding and legacy facade owning tests |
| `focused-attempt2` | 82 passed, 1 deselected; rc0 | Same affected selection plus real OB perception and assembly producer migration |
| `bound-fixture-attempt1` | 7 completed, 0 execution errors, 0 compared differences; rc0 | Existing B20-bound private inputs; all factual/Bond bindings preserved |
| `neighbour-invariant-attempt1` | 2 passed, 62 deselected; rc0 | Final F1 tests compare with the active rule instead of asserting scientific N4 |
| `bindings-attempt1` | rc1 | Evidence helper assumed absent `openbabel-wheel` metadata; original helper/log retained |
| `bindings-attempt2` | rc0 | Seven owned files parsed, 11 snapshot-bound calls, historical test names/parameter decorators retained across three files, scientific/native hashes unchanged |

Every run reports `frozen_unchanged: true`. No production edit followed the
82-pass run. Only the F1 regression function changed afterward, to remove its
fixed N4 scientific assertion; those final two parameter cases passed. The
82-pass and original seven-failure test bytes are separately saved and match
their command records. The failed metadata helper is corrected by recording
the actual Conda OpenBabel package/module/library identities. It is not a
production failure or a missing-extension success claim.

The executed environment is current source with Python3.11.16, NumPy2.4.6,
RDKit distribution2026.9.1, OpenBabel3.2.1, pytest9.1.1 and periodictable2.1.0.
CPU affinity is24–31, four BLAS/OMP thread overrides are1, the existing
libstdc++ preload is retained, and no model artifact is used. The existing
rebuilt cp311 fact native SHA-256 remains
`77473313ae381af6818a5d4b5c2e940d54a44cc0edf36855a68cdf03aa80327a`.
This is not installed-wheel, five-version, final science or benchmark evidence.
The one deselected historical scalar-H node is recorded exactly in the pytest
argv; the other pending scalar nodes were outside this focused selection.

## Remaining gates and ownership

The latest user direction preserves explicit counts and requires inference for
unspecified H using scientifically justified valence/bond defaults. A separate
Sol-max task owns that scientific review, `contracts/decisions.md` and its main
plan amendment. These corrections preserve the existing scientific functions.
The pending public H reader and genuinely unavailable scalar/numeric-export
branches remain untouched; no optional-return approval is inferred.

Root owns indexing/global records/Git, the unfiltered affected/default suites,
Python3.10–3.14 source/debug/installed-wheel rebuilt-native checks,
packaging/native closure, final science and frozen lazy9/graph measurements.
The prior Stage4 performance blockers remain open without a waiver. The
additional operation input capture in OB perception has no measured cost claim.
No child delegation or new experiment occurred.
