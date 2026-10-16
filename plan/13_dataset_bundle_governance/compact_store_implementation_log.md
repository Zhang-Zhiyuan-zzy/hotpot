# Compact formal-store implementation log

## Scope and isolation

- Date: 2026-10-16.
- Feature branch: `feat/dataset-compact-release`.
- Isolated worktree: `/tmp/hotpot-ds-compact`.
- Source generation: `stage13-efficiency-current`.
- Candidate generation: `stage13-compact-current`.
- The source generation and the three source Bundles were opened read-only.
  Candidate data was built beside them and activation was a separate step.
- The pre-efficiency archive remains subject to its existing 2026-11-01
  retention date. This implementation did not delete it.

## Execution record

1. Measured all 252,665 source Cases before changing the target layout. The
   complete fact JSON occupied 3,207,002,168 logical bytes; removing only
   implementation-owned migration mirrors reduced it to 1,171,057,125 bytes,
   or 392,432,311 bytes with deterministic zlib level 6.
2. Defined the permanent information boundary before editing storage code.
   Original scientific facts, evidence, attestations, classifications,
   structures, Gate policy, final labels, and concise reasons remain durable.
3. Replaced the target schema with the single current
   `hotpot-ds-store/compact-current` layout. Facts are compressed and
   digest-bound; complete Gate traces and reproducible archive tables are not
   stored.
4. Added target-owned Evidence references to `Data.info` and verified that a
   formal release remains usable after its Bundle root is absent.
5. Added `hotpot ds bundle rm --all`. Cleanup is receipt-gated, uses an exact
   deletion allowlist, revalidates the target on every crash-resume, records
   every deleted object, and preserves unknown and original files.
6. Restricted removable migration controls to the generated
   `identity-migration` and `legacy-identity-relationships` maps. The small
   `legacy-source-inventory` and `migration-provenance` controls are retained
   in the formal CAS as provenance.
7. Adapted the one-time owner rebuild program. No old reader was added to the
   package: the program reads the declared schema-v2 source intake and sealed
   schema-v1 assessment generation locally, while the runtime accepts only the
   current schemas.
8. Added exact source-to-candidate checks for retained facts, identifiers,
   structures, scientific assets, final quality/evidence/archive/benchmark
   values, policy, receipts, and permanent controls. Added a hard 1 GiB
   activation gate for formal SQLite plus CAS objects.
9. Locked both source and candidate SQLite writers during publication and kept
   activation as an explicit, atomic directory-symlink replacement.
10. Ran the complete dataset regression suite: 368 passed and 1 skipped.
11. Built the three candidate releases and completed full parity. The candidate
    contains 252,665 Cases and occupies 959,831,795 bytes (0.894 GiB):
    951,156,736 database bytes plus 8,675,059 CAS object bytes.
12. Revalidated every completed phase with the final code, then repeated full
    parity under source/candidate SQLite write locks and atomically activated
    `stage13-compact-current` at 2026-10-16T11:08:32.289731Z. The previous
    `stage13-efficiency-current` generation remains the direct rollback target.

## Unexpected findings and required adjustments

### A false full-facts claim

The first parity draft compacted source facts before comparison but reported
that original facts were compared in full. This was incorrect. The report now
states explicitly that the declared retained projection is compared in full,
lists every omitted implementation-owned field, and binds the untouched source
intake separately. This wording and implementation are required to avoid
overstating the proof.

### Provenance controls were classified too broadly

Treating every `migration` control as disposable would have removed the source
inventory and owner/tool migration decision. Those files are small, original
classification provenance. They are now normalized to permanent `provenance`
controls; only the two exact generated mapping names are removable.

### Crash-resume cleanup did not repeat target validation

The initial cleanup resume path trusted its prior plan. It now repeats
certificate, receipt, logical release, SQLite, compressed-fact, and CAS checks
before deleting another byte. This is required because a new CLI process or an
external edit can invalidate the earlier observation.

### Evidence bytes were durable but not discoverable

The first compact reader retained Evidence CAS objects without exposing their
target-owned locations. `Data.info["evidence_assets"]` now reports the formal
CAS path and identity metadata, without adding another first-level public API.

### The real source Bundle is intentionally an older working schema

The first rebuild parity function attempted to open the schema-v2 source with
the current Bundle runtime and would have failed only after the expensive
build. The owner script now has a narrow, read-only migration-local reader for
that exact source. Runtime compatibility code was not introduced.

### The legacy assessment manifest needed a logical digest proof

Matching a manifest's self-reported digest to the source control row was not
enough. The rebuild program now recomputes the logical generation digest,
checks the digest-derived generation ID, hashes every declared artifact, and
then reads the final scalar labels. This closes a same-manifest tampering gap.

## Verification state

- Active generation: `stage13-compact-current`.
- Build phases and semantic parity: complete.
- Formal size gate: passed (`959831795 <= 1073741824`).
- SQLite integrity and foreign keys: passed by the rebuild parity gate.
- CAS inventory: 2,289 objects, 8,675,059 bytes, no missing or orphan object.
- Public API/CLI and Bundle-deleted end-to-end tests: passed.
- Atomic activation record SHA-256:
  `9ecfde80201f3ddf3f099e4bcc7a82dafca36fee1bd368e21fef48f350b28435`.
- Cleanup was not run against the real candidate Bundles during the rollback
  acceptance window. The command and its destructive boundary are tested; the
  retained source and candidate Bundles remain available for revalidation.

