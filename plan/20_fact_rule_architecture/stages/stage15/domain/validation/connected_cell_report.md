# Connected Crystal cell/lifecycle checkpoint

Root accepts the connected Stage15B slice on 2026-11-05 from standalone
`bcc199b355183f857b8784b267cf4c53b2eaa21a`. Crystal owns one immutable
oriented-cell/group/relation record, attached to the existing Molecule owner.
The [private lifecycle contract](../contracts/lifecycle.md) governs UID-bound
relations, complete publication and rollback, copy/component/merge, restored
pickle aliases and external six-parameter projections. Standard angle semantics
are alpha=(b,c), beta=(a,c), gamma=(a,b). The current MoleculeRecord pickle
is generation 7; the superseded six-scalar ordinary cell record is removed.

The actual affected CP311 source and outside-checkout pip-installed wheel each
pass **559 tests, zero failures/errors/skips**. Source elapsed time is 6.7304 s;
installed elapsed time is 17.7937 s. These are test durations, not performance
benchmarks. Installed evidence observes 196 correct product origins and zero
violations, with 912 installed files and 21 support files unchanged. All six
accepted native payloads remain unchanged: this wheel is a repack, not a fresh
native build. The unchanged exhaustive standalone symmetry suite retains its
separate source/five-version qualification and was not needlessly repeated.

Review found and repaired three concrete defects: missing/partial XYZ could
become fabricated zero facts during fractional preservation; generated content
could claim a non-P1 active group; association replacement while acquiring a
lock could use a detached owner. Six targeted regressions pass, including the
real-thread association interleaving. The independent script reports two
additional witnesses passed and one checker error: its coordinate equality
assertion compared identity-only MoleculeRecord objects after the correct local
business rejection. Its subsequent subcases did not run. That script is not
reported as an overall pass; direct source/installed coordinate-presence and
atomicity regressions supply the required evidence.

Original generation-schema, external-import and obsolete assertion failures
remain archived. The external-import repair preserves the explicit basis/group
while leaving unavailable site facts and Bond images unknown. It never invents
periodic images. All final source/test inputs match their recorded tested bytes.

[The checkpoint](connected_cell_checkpoint.json) records source/wheel inputs,
results and coverage. Exact commands, environments, JUnit, guards, failed
attempts, review scripts and wheel payload are in
`.plan20-artifacts/20261105/stage15/connected-cell-lifecycle/evidence.tar.gz`:
7,736,642 bytes, SHA256
`242b8630d62aad845dd018bf489ffaf027007ecdd98447ff3b8960b9e39aa322`,
153 verified regular members. Temporary execution root:
`/tmp/hotpot_plan20/medium/stage15b-connected-20261105/`.

Molecular actions/expansion, periodic matching/findings, CIF, current integrated
costs/support/delivery and Stage14 Crystal transport remain open. This coherent
cell/lifecycle checkpoint is not Stage15 or overall completion.
