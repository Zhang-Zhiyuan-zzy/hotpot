# M_17-19 implementation log

## Baseline

| Item | Value |
|---|---|
| Integration branch | `merge/m17-19-native-graph-rdwrapper` |
| Integration worktree | `/tmp/hotpot-m17-19` |
| Current parent | `6639754` |
| Graph parent | `clean/networkx@b912c8a` |
| Merge base | `2704f3c` |
| Started | 2026-10-26 |

## Stage 0 — Freeze plan, current assets, and disposition matrix

Status: complete

Intent:

- preserve the current main worktree exactly as found;
- create a separate branch/worktree for all integration work;
- carry the uncommitted Plan 17 assessment, benchmarks, and proposed organic
  native conformer extension into the integration branch;
- freeze merge precedence, the final API tree, conflict semantics, deletion
  policy, validation gates, and the complete API/implementation disposition
  matrix before running `git merge`.

Performed:

- created branch `merge/m17-19-native-graph-rdwrapper` at `6639754`;
- created worktree `/tmp/hotpot-m17-19`;
- copied the three modified and five untracked Plan 17 files without changing
  the source worktree;
- recorded both parents and merge-base;
- created this integration plan and the normative migration matrix;
- expanded the deletion audit beyond Plan 19 to include the constant-empty
  traversal compatibility API, the disconnected SymPy search implementation,
  GraphMatcher-shaped native-search adapters, missing-stereo-field fallback,
  RDKit `hotpot-legacy/1` transport, and proposed old-prefix ABI handling;
- explicitly retained RI candidate-domain `compatibility.hpp`, opaque stereo
  source facts, the current exact-version C API/SDK target, and the private
  conformer sampler because they are active business mechanisms rather than
  compatibility layers; and
- corrected the proposed Plan 17 organic-native plan so a future ABI extension
  accepts only its one current full structure/version.

Conflict/deviation:

- none; merge has not started.

Validation:

- `git diff --check`: passed;
- Python syntax compilation for all three copied benchmark drivers: passed;
- relative Markdown link resolution for Plan 17, M_17-19, and the Plan index:
  zero missing targets;
- stale legacy-prefix wording scan: only the intentional deletion row in the
  migration matrix remains.

Git checkpoint: pending.
