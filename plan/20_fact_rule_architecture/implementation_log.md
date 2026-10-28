# Plan20 implementation log

Execution was requested on 2026-10-28 against the latest reviewed plan.
Entries are append-only. A stage starts with a complete plan/companion reread
and a current-project assessment; its completion requires its named evidence
and focused Git commit. Scientific observations and technical pass/fail results
are recorded separately.

## 2026-10-28 — Stage0 entry: freeze the implementation direction

- Source commit: `c4eeac97b0ad983f2b00949b77d32d2235e6d333`.
- Starting branch: `docs/plan20-kekulize-protocol-experiment`.
- Implementation branch created from that commit:
  `implement/plan20-fact-rule-architecture`.
- Worktree: `/home/zhangzhiyuan/hotpot`; no additional worktree.
- The starting worktree contained reviewed but uncommitted Plan17/20/21 and
  plan-index documentation. Those documents are part of the Stage0 freeze;
  production code, tests and build definitions were unchanged.
- Full reread completed: all 17 main-plan chapters, the execution alignment,
  chemical-type interface contract, data-object audit including its complete
  definition index, and Plan21 framework. Document identities and current
  source/environment assessment are recorded in `stage0_review.json`.

### Current project and selected route

1. The production MCA pipeline calls
   `hotpot/cheminfo/AImodels/mca/conformer.py::ensure_3d_conformer`: reuse usable
   existing 3D coordinates, otherwise AddHs, single ETKDGv3 embedding with the
   existing seed, MMFF when parameterized or UFF otherwise, then RemoveAllHs.
2. The standalone native RDKit sampler still performs ETKDG sampling without
   the 17-MCA optimization/result-transport closure. Its historical 256-case
   assessment is not evidence that this closure was implemented.
3. Select the plan-permitted **Plan20-first** route. `B20/F20` use the current
   sole Python-RDKit MCA algorithm. Stage8D moves its ownership/call boundary;
   it does not replace the algorithm. Defer 17-MCA until after `F20` and record
   its own `B17/F17` then. Plan17 continuation, Plan20 and the alignment record
   are synchronized before this planning freeze.
4. Existing `core.py`, carrier caches, mutable fact arrays, mixed stereo and
   adapter modules remain the starting implementation. No extraction or
   production deletion is authorized to jump ahead of the Stage1 baseline.
5. The source corpus exists at `molecules/extractant/extractants.smi`, SHA256
   `e67ae3167a1a96d3c6a4c6df3ce12b70abfaf479bdd89bb156bfaefa771795a7`.
   Eligibility and scientific reports still need execution under current rules;
   neither historical 187 nor 182 supplies a scientific pass criterion.

### Fixed goals, decisions and boundaries

- Follow the final ownership tree in main-plan Section8: structure/views,
  rules, operations, adapters and reference, with one implementation per role.
- Keep P0–P4 ownership, the sole Atom fact table, one topology index, finite
  dependency metadata, source-aware atomic commits, immutable rule results,
  and the approved standalone wrapper/stereo boundaries.
- Record the exact public surface and consumer differences in Stage1. The
  current implementation instruction permits the documented implementation
  work; unspecified public constructors, result schemas and read semantics
  must first become concrete reviewable entries. No public change is inferred
  from a private helper or an illustrative name.
- Keep optional unestablished APIs, MetalCluster runtime and unrelated
  training/plugin rewrites outside current production work. Decisions12/13
  retain their existing implementation/explicit-deferral rules.
- The experimental rule protocol starts only after validated main delivery
  `B_exp`, on its separate branch; definition `D_exp` and integration `I_exp`
  remain separate commits. Only kekulize/aromatic adopts it.
- At every stage and separately gated sub-stage, repeat the full-document
  read and assess goals, source state, predecessor evidence, environment and
  exact scope before starting its work. This requirement is now in Sections10
  and13 and the Plan20 README.

### Environment and verification status at entry

- Local interpreter:
  `/home/zhangzhiyuan/usr/conda3/envs/hp-usage/bin/python`, Python3.11.16.
- Existing runtime packages include RDKit2026.9.1, OpenBabel3.2.1,
  NumPy2.4.6, pytest9.1.1 and ONNX/ONNX Runtime; CPython3.11 native extension
  files are present. File/package presence does not establish ABI test success.
- The current environment needs isolated build/test tooling: `build`,
  `pybind11`, coverage, and the project-constrained setuptools version. A
  task-specific environment will be prepared before baseline/build execution.
- The local GPU driver is unavailable. Read-only access to ai-14 succeeded;
  eight A800 GPUs are visible with existing shared allocations. Controlled
  CPU/1-GPU/4-GPU timing and exact installed-wheel identities remain pending.
  Remote task files stay under `~/mca_test` or `~/mca_study`; environments stay
  under `~/.local/conda`.
- MCA artifacts are locally available; CBond has a candidate artifact tree
  under `/tmp/hotpot-test-models` to validate explicitly. Cached native build
  archives also exist and require digest verification before reuse.
- This entry records read-only code/document/environment assessment. No
  default suite, native/wheel validation, scientific report or performance
  baseline is recorded as passed.

### Stage0 deliverable and next-stage scope

The Stage0 delivery is a documentation-only commit containing the complete
reviewed plan and normative companions, selected route, stage-entry requirement
and this log. Its resulting SHA will be appended in the Stage1 entry.

Stage1 first rereads those committed records, then captures API/consumer and
architecture-debt inventories, the stage-to-verification map, scientific
runner/comparator, and executable technical/scientific/performance baselines.
All observations, absent evidence and raw report paths remain explicit.

## 2026-10-28T07:06:20Z — Stage0 complete; Stage1 entry assessment

- Stage0 planning commit: `e6082f5fd67cb1054726ddcf43db7f963f7091ed`.
  `git diff --check` passed before commit; the commit contains documentation
  and review records only. The worktree was clean at Stage1 entry.
- Production subtree: `4a2086e717327c99056ef8cda200b0786e688219`, unchanged
  from Stage0 entry. This is the source implementation selected for `B20`;
  actual baseline measurements and installed-wheel identities are still pending.
- A new complete reread was performed for Stage1 by the architecture reviewer:
  all 17 main-plan chapters (2434 lines), the 280-line interface contract,
  519-line data-object audit including its full appendix, 177-line execution
  alignment, and 68-line Plan21 framework. The implementation log, review JSON
  and both relevant READMEs were also read. The five normative SHA256 values
  match the committed `stage0_review.json`; this was a new full read, not a
  diff-only check or a reused Stage0 reading claim.
- Assessment: Stage1 may start. Stage0 scope and route are frozen; tests,
  scientific reports, installed-wheel validation and efficiency measurements
  have not yet supplied Stage1 completion evidence.
- Scope: exact API/type/consumer and architecture-debt inventories; Section9
  migration responsibilities and verification commands; data-object ledger;
  scientific report runner/comparator; reproducible technical, scientific and
  performance baselines. Production extraction/deletion, MCA cutover,
  MetalCluster runtime and Stage13 protocol implementation remain later work.
- Parallel ownership: inventory/API map, scientific reporting, and performance
  tooling have separate file owners. The parent owns environment preparation,
  test/build/GPU execution and this log.
- Task environment: `/tmp/hotpot-plan20-env` created from the current Python
  with system-site runtime packages; task-local build/test tooling is installed
  there. Existing `hp-usage` package contents are not modified.
- Raw local evidence root: `/tmp/hotpot-plan20-20261028`; remote evidence will
  use `~/mca_test/plan20-20261028`. Each measurement records its exact command,
  allocation, package/model/input identity and any missing evidence.

## 2026-10-28 — Stage1 baseline tooling and technical execution checkpoint

- The baseline executes from the detached, frozen checkout
  `/tmp/hotpot-plan20-20261028/source-b20` at Stage0 commit `e6082f5`.
  Production sources remain unchanged. New private reporting tools are
  independently identified by their content digests; copying those tools into
  the baseline checkout does not change the selected production tree.
- Evidence root: `/tmp/hotpot-plan20-20261028`. Task interpreter:
  `/tmp/hotpot-plan20-env/bin/python` (CPython3.11.16). The environment record
  is `baseline/environment-local.json`. The task environment adds build/test
  tooling without modifying the original `hp-usage` environment.

### Completed technical evidence

| Execution | Result | Raw evidence relative to the evidence root |
|---|---|---|
| Frozen source default suite: `python -m pytest -q --import-mode=importlib tests` | 3822 passed, 123 skipped, 3 xfailed, 49 subtests passed; 247.56s | `baseline/default-source.log`, `.xml`, `default-source-cases.json` |
| Explicit MCA/CBond models: `python -m hotpot models verify --all` | Both verified | `baseline/models-verify.log` |
| Fresh cp311 wheel content audit and installed core smoke, outside checkout | Passed | `baseline/wheel-audit.log`, `installed-core.log` |
| Installed inference smoke; installed forcefield smoke, separate invocations | Passed; 6.61s and 5.43s | `baseline/installed-inference.log`, `installed-forcefield.log`, `installed-model-smoke.json` |
| Native C++/Open Babel/RDKit standalone tier, high-compute and external enabled | Initially 100 passed, 1 skipped, 2 SDK consumer failures | `baseline/native-standalone.log`, `.xml` |
| The two SDK consumers with the SDK's pinned C++ toolchain/runtime | 2 passed, 2 deselected | `baseline/native-sdk-correct-toolchain.log`, `.xml` |
| Reporting regressions and one actual bounded O0 run | 22 passed, 1 warning; 2.71s | `baseline/report-tools.log`, `.xml` |

The default-suite XML includes subtests and xfails in its counts; the table
uses pytest's actual terminal summary. These elapsed times are functional
execution observations, not the formal uncontended performance baseline.
`installed-all-smoke.log` contains only the core invocation: that script takes
one mode, so inference and forcefield evidence comes from the separate runs.

The initial SDK consumer failures were a system-compiler/libstdc++ mismatch
(`GLIBCXX_3.4.31`), resolved by selecting
`/tmp/hotpot-rdkit-toolchain/bin/x86_64-conda-linux-gnu-g++` and its runtime
library directory. They remain recorded rather than overwritten. The one
native skip is the explicitly disabled scientific-boundary phosphorus test
(`test_all_extractant_phosphorus_centers_build_and_optimize_finitely`); the
standalone invocation did not enable that tier.

### Build and input identities

- Fresh wheel:
  `build/wheels/hotpot_zzy-0.5.4.5-cp311-cp311-linux_x86_64.whl`, SHA256
  `a7a4c1580881be42a3651a92660efdc934164a4d47a386ffe584b4ef0d4c9a97`.
  Build log: `build/cp311-wheel.log`. This is a local probe-profile Linux
  wheel, not a manylinux release certification.
- Fresh SDK prefix: `build/sdk`; log: `build/sdk-build.log`. Both use the six
  verified locked source archives under `/tmp/hotpot-plan17-release-cache`
  and the pinned probe toolchain under `/tmp/hotpot-rdkit-toolchain`.
- The five wheel native modules were installed into the frozen source
  checkout; `baseline/built_native_files.json` records their hashes. The
  obsolete preexisting `_relevant_cycles` binary was moved outside its import
  tree into `baseline/legacy-native-not-in-wheel`. Historical binary hashes
  remain in `baseline/existing_native_files.json`.
- Corpus snapshot: `molecules/extractant/extractants.smi`, SHA256
  `e67ae3167a1a96d3c6a4c6df3ce12b70abfaf479bdd89bb156bfaefa771795a7`.
- MCA model directory:
  `/home/zhangzhiyuan/.cache/hotpot/models/artifacts/mca/1/adf525b6a95d6d296ca0d5c68995b1a0ce14d5fe/mca/v1`;
  CBond directory:
  `/tmp/hotpot-test-models/artifacts/cbond/1/adf525b6a95d6d296ca0d5c68995b1a0ce14d5fe/cbond/v1`.
  Derived model/cache files use the task's selected model-cache directory.

### Inventory and reporting changes

- Captured 403 production Python files, 782 classes, 347 dataclasses,
  3 TypedDicts, 13 Protocols and 1837 interface entries. All 173 migration rows
  have owner stages. Direct-core imports occur in 67 production and 74 test
  files; the concrete path roster supersedes historical text-count estimates.
  `baseline_inventory/manifest.json` identifies every captured artifact.
- `stage1_api_review.md` records constructor/buffer alias observations,
  existing absence behavior and precise remaining public/schema decisions.
  `stage_verification_map.json` distinguishes existing and future tests.
  This inventory does not claim that a public removal or DTO reduction has
  occurred, or that static searches establish absence of external consumers.
- Added private scientific execution/comparison tools. The former fixed-182
  test is replaced by record-completeness and artifact-integrity assertions.
  Completed monitor observations use `scientific_status="observed"`; monitor
  flags do not decide success. Execution or artifact-write failures remain
  explicit and unresolved. OBBuilder's existing ignored provider seed is
  recorded, while requested root/case seeds remain exact.
- Frozen performance manifest: 71 entries, SHA256
  `ed5bba12ac6ae0305f6c85f7af3aa8fa824a3278fe12aad537a2f0c13768571b`;
  three new-only measurements have no fabricated old baseline. The original
  common technical selection contains 3948 node IDs at
  `perf-common-test-selection/test_nodeids.json`.
- Independent review caught omitted actual force-field work counters,
  conversion output counts and teardown failures. The tools now preserve
  epochs/steps/attempts (including unknown step counts), OB/RD atom counts,
  and exclude cleanup failures from comparability and claimed speedups.
  Eleven performance regressions passed within the 22-test report check.
  These corrections do not change workloads, weights or sampling policy.

Stage1 remains open: full scientific execution, formal comparable local and
remote performance baselines, final tool review and the required concrete API
entries still need closure. No production refactor or Stage2 work has started.

## 2026-10-28T07:50:23Z — Stage1 scientific baseline and A2 decision

- The complete frozen-source O0 standard run finished with root seeds
  42/43/44, 32 spawned workers, CPU affinity 0–31 and OMP/BLAS/MKL/NumExpr
  threads fixed at one. Raw report: `science-b20-standard/report.json` under
  the evidence root; runner log: `baseline/science-b20-standard.log`.
  Wall time was 181.6331s, and the sum of all per-trial times was 4861.7257s.
  These are scientific execution observations, not formal speed measurements.
- All 187 source records and all 561 case/seed combinations are retained.
  Current CBond eligibility yielded 182 eligible and 5 ineligible cases:
  546 trials completed, 15 were explicitly not run for ineligibility,
  zero execution errors, zero unavailable elapsed-time records. The count
  resulted from the current rule; no fixed population criterion was applied.
  Of the completed observations, 545 had the existing aggregate quality
  monitor flag true and one false. That false flag and all individual
  findings remain in the report; all 546 are completed observations.
- The self-comparator passed artifact/roster/coordinate integrity on the full
  report. Evidence: `science-b20-selfcheck/comparison.json` and
  `baseline/science-b20-selfcheck.log`. Self-comparison establishes internal
  integrity only; it is not an old/new scientific or performance comparison.
- The Stage1 native test map ran with the pinned SDK toolchain/runtime and
  high-compute/external enabled: **26 passed**, 222.98s. Evidence:
  `baseline/native-stage1-map-command.json`, `.log`, `.xml`.
- The frozen-source model-artifact tier ran separately:
  **42 passed, 1 skipped, 3905 deselected**, 38.36s. The skip is the separately
  governed full scientific test, whose replacement report was executed above.
  Evidence: `baseline/model-tier.log`, `.xml`.
- A fresh architecture review read the entire main plan and all four
  normative companions, including the full data-object appendix, then
  examined the still-unspecified constructor/read/result contracts. The
  status-header update was read as well. The review identified additional
  raw-array import, working-copy, model-feature and thermo-proxy consumers;
  no production code changed.
- The owner explicitly accepted the concrete A2 proposal in
  `stage1_contract_decisions.md`: remove `attrs_array` and
  `update_electron_config`, retain independent fixed numeric snapshots,
  encode absent selected results as NaN, and preserve existing model tensor
  encoding. The companion and API register are synchronized. Stage3 first
  isolates the sole table/builder, Stage5 makes the constructor cutover with
  its consumer migration, and Stage7C changes partial-charge ownership.
  This decision does not authorize losing imported values in an earlier stage.

Formal efficiency evidence is still pending. A further performance-tool audit
found that GPU UUID/model/driver were not checked for comparability; its fix
and installed-package/wheel identity binding precede formal measurements.
