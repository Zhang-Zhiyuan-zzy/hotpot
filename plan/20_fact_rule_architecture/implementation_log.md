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

## 2026-10-28T08:49:26Z — Stage1 performance and inference checkpoint; Python matrix open

Earlier pending statements remain as dated history. This checkpoint records
completed evidence from frozen source `e6082f5` and production subtree
`4a2086e717327c99056ef8cda200b0786e688219`; no production refactor has started.

### Completed formal performance and identity checks

- Local source-mode execution completed all **64 workloads: 605 measured
  samples and 178 warmups**, all 783 attempts completed without execution
  errors. Affinity was CPU0–63 with helper threads fixed at one; no concurrent
  local task build/science/test work ran during these measurements. The five
  common-suite measurements had median **242.979444s**, MAD **0.353841s** and
  p95 **243.277620s**. Test collection median was **9.408003s**.
- ai-14 installed-mode execution completed **7 workloads: 27 measured samples
  and 7 warmups**, all 34 attempts completed, with no errors/no-output attempts
  or discarded reruns. Total run wall time was **82.433329s**. CPU affinity
  was 0–63; GPU visibility was 0,1,2,3 with `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
- The actual union is **68 existing workloads**. Follow
  `stage1_sampling_plan.json`: use the 64 local entries plus the four remote
  GPU entries once each; remote `mca_cpu_8`, `mca_cpu_64` and `cbond_cpu` are
  supplemental host observations. The three new-only entries retain no B20
  counterpart. The original manifest and weights are unchanged.
- GPU comparability now checks selected UUID, model, driver, total memory and
  actual allocation. ai-14 used shared A800-SXM4-80GB devices and driver
  525.105.17; per-attempt load/memory snapshots are retained. Dynamic load and
  unrelated GPUs do not supply device identity, and shared-load review remains
  necessary for a later paired candidate comparison.
- Installed-wheel verification bound all **761 files** to the supplied cp313
  wheel and its RECORD; all seven timing-worker loaded-package checks passed.
  Wheel SHA256: `5e9d55297dd4d15814e7814dd3055b88ee84d2d8656312677c9a663792546672`.
  Installed RECORD SHA256:
  `3a4c6a654613e0bc375a304ee9403619f9022695c92dd39f0cfac7dd8989a892`;
  wheel RECORD SHA256:
  `69dd61c01f9651d1ac9c368f34a6d3b5ff16b427e6506be070ec16d9477f31fb`.
  Local formal timing deliberately remains source-mode evidence with its
  separately recorded cp311 wheel/native provenance.
- Self-comparisons found **64/64 local** and **7/7 remote** executed entries
  comparable, with no environment mismatch. These establish report consistency;
  neither is a candidate comparison or completed whole-project acceptance.

### Untimed inference observations and saved-output comparison

- Separate local capture retained **72 MCA molecules, 306 ordered atom
  predictions/flags, 108 sites and 20 CBond calls**, with zero execution errors.
  It ran after formal local timing; its elapsed values are excluded from
  performance evidence. It retains 156 artifact references and their hashes.
- Separate remote capture retained **1104 MCA molecules, 4692 ordered atom
  predictions/flags, 1656 sites and 40 CBond calls**, with zero execution errors.
  All 1272 saved artifacts matched recorded hashes; no nonfinite observations
  were reported. The 120 CBond output arrays and 8 input arrays remain saved.
  Reporting-process wheel bindings passed; spawned MCA worker modules/sessions
  were explicitly not directly introspected by this capture.
- Read-only analysis of the saved remote output found CPU8/fallback8 and
  CPU8/CPU64-first8 identical; 1GPU/4GPU agreed across all 512 molecules
  (2176 atom values and 768 site values). Comparing CPU64 with either GPU's
  first64 gave atom MAE **0.04825367647**, RMSE **0.10298425110** (272 pairs),
  and site MAE **0.07421875**, RMSE **0.13049447725** (96 pairs), in kJ/mol.
  Both p95 and maximum absolute differences were **0.25kJ/mol**. Pairing had
  no identity/nonfinite exclusions and `is_Nuc_site` flags were unchanged.
- All captured MCA DTOs record `model_variant="fp16"`. This label does not
  measure individual operators' arithmetic precision; cohort size/execution
  mode also differ. No CPU-fp32/GPU-fp16 cause or scientific pass threshold is
  inferred. This numerical analysis did not rerun inference.

### Object budget and remaining Python matrix gate

- `data_object_budget.json`, SHA256 `9d5fb507d704050c2697feec8b527c3b24a24dff2e73511090b52eb949048946`,
  records **347 current dataclasses** and zero measured production changes.
  The audited consolidation is 16 old definitions to five retained/replacement
  definitions: retire14, retain2, add3, net−11. Reservations allow at most18
  further mechanism records, giving **347−14+3+18 = 354 (net+7)** if all
  consolidations land. This is a conditional planning ceiling; without the
  still-required D01–D03 public/schema approval the ceiling is362. No whole
  project record-count reduction or early placeholder creation is claimed.
  Stage3 must freeze N01–N09 fields/reuse before implementation; Protocol,
  TypedDict and native-ABI-mirror additions have separate zero budgets.
- Review identified the still-missing Python3.10–3.14 baseline required by
  Sections10/12. The supplemental manifest selects **2689 common nodes**
  (2408 inference-contract plus281 smarts-core), one warmup plus five measured
  rounds per version, with disjoint eight-CPU allocations and fixed helper
  threads. This mandatory matrix remains separate from the 71-workload
  aggregate; build/install/model setup is excluded from its timed work.
- Matching B20 wheels/environments are prepared and matrix execution is in
  progress. The first preflight retained its collection failures under
  `/home/zhangzhiyuan/hotpot/.plan20-artifacts/20261028/python-matrix-b20`;
  pytest9.2 shortened long parameter IDs, so no measured rounds ran there.
  Explicit `parametrize_long_str_id_strategy=legacy` preserves the original
  frozen node IDs without changing test scope or weights. The current manifest
  SHA256 is `7c6ed13718cfa1563f65c99c8f3ac8ff94b77eddf43aaf5e0ff27ac54de3a107`. The follow-up run uses
  `.plan20-artifacts/20261028/python-matrix-b20-legacy-ids`; its result is not
  asserted in this checkpoint.

Stage1 remains open until this matrix and the remaining Stage1 records are
validated and committed. Stage2 must begin with its own full-document reread
and state assessment after that gate.

### Exact evidence bindings at this checkpoint

| Report path | SHA256 |
|---|---|
| `/tmp/hotpot-plan20-20261028/performance-b20-local/report.json` | `ad460fe264a6418379893f69fb918becd0ae5428b20494e9c8ebab1458906cf5` |
| `/tmp/hotpot-plan20-20261028/performance-b20-local-selfcheck/comparison.json` | `d2aaac94cf2080abde2092db3b9d99b1df7f6ae43adce176dab62a513d8ae473` |
| `/tmp/hotpot-plan20-20261028/remote-build/performance-evidence/performance-b20-remote/report.json` | `692034b778c5c88c909ffa3b8ad9d6934a94f0a5f9d4d6510829cab2d17aa49b` |
| `/tmp/hotpot-plan20-20261028/remote-build/performance-evidence/performance-b20-remote-selfcompare/comparison.json` | `803087b368682166e1090441a993c77ab5d382c58898dab493eaacecd7ee4e9d` |
| `/tmp/hotpot-plan20-20261028/inference-b20-local/report.json` | `d4dfa4e1cdd446c5a3855d908ae270bd587980e12e1236bd10726486ae167691` |
| `/tmp/hotpot-plan20-20261028/remote-build/inference-evidence/inference-b20-remote/report.json` | `522173ba898885f952f4e7742c957dd9f2af7a66c53455db93f78b8e4f2b4eaa` |
| `/tmp/hotpot-plan20-20261028/remote-build/numerical-analysis-evidence/inference-b20-numerical-analysis.json` | `0f7a33058fe26fb821bad2c7cf1470f1878ea2a3cc495a4bff3eaad28b437e42` |
| `/home/zhangzhiyuan/hotpot/.plan20-artifacts/20261028/python-matrix-b20/report.json` | `5036b838a480b1701d4e99fd362956d361ce1c6498f86d7a81544ff13c21c2e0` |

## 2026-10-28T09:21:24.010085+00:00 — Stage1 complete: supported-Python matrix and semantic closure

The full Stage1 deliverable is now verified. This checkpoint contains reporting,
review and baseline records only; the production subtree remains
`4a2086e717327c99056ef8cda200b0786e688219`. The focused Git commit containing
this entry is the Stage1 close; its exact SHA belongs in the Stage2 entry.
`stage1_execution_evidence.json` binds all principal reports, policies, logs,
review records and retained failed attempts by SHA256.

### Python matrix and reporting verification

- All five matching B20 wheels passed preparation and installed-byte preflight.
  The successful matrix used separate eight-CPU allocations, helper threads1,
  2689 frozen nodes, one warmup and five measured rounds per version. All
  **30 attempts (25 measured +5 warmups)** completed, with zero execution errors
  or missing rounds. Each attempt reports **2647 passed, 39 skipped, 3 xfailed,
  and 49 passed subtests** across its two pytest invocations. The skip/xfail
  identities and actual execution events are retained, not inferred from only
  collection totals. This matrix is independent of the original71 weights.
- Per-version measured medians/MADs in seconds:

| Python | Median | MAD |
|---|---:|---:|
| 3.10 | 198.601123 | 0.118640 |
| 3.11 | 94.459424 | 0.228534 |
| 3.12 | 206.607143 | 0.322156 |
| 3.13 | 215.484424 | 0.175484 |
| 3.14 | 230.213605 | 0.222707 |

- Raw matrix: `.plan20-artifacts/20261028/python-matrix-b20-verified/report.json`
  under the project root. Its paired selfcheck is
  `python-matrix-b20-selfcheck-subtests/comparison.json` under that artifact root:
  all five versions verified with all five measured pairs. This establishes
  internal consistency and retained artifact integrity, not F20 improvement.
  Different versions' package environments remain independently frozen.
- The initial preflight failed because pytest9.2 shortened long generated IDs;
  the frozen legacy ID option restored the identical selection. The next run
  was interrupted during warmup (exit130) after review found the missing
  outside-source subprocess workdir. The final runner fixes
  `HOTPOT_TEST_WORKDIR` and `PYTEST_ADDOPTS`, records actual execution events,
  rejects reduced JUnit work, and propagates worker-exit errors. Original
  reports, partial artifacts and interruption reason remain intact.
- The first comparator selfcheck correctly withheld a conclusion, but its
  one-testcase-per-summary-count assumption was too strict for pytest subtests.
  Actual B20 evidence has2408 XML cases and2457 summary outcomes, including49
  extra passed call reports. Comparison now verifies the terminal-event count
  and preserves per-node/phase/outcome and XML-case identities. No timing sample,
  test selection or source assertion changed, and no matrix rerun was needed.
- Reporting regressions: **59 passed**,5.49s, in
  `report-tools-complete.log/.xml`; after the subtest correction the complete
  comparator selection passed **14 tests**,3.93s, in
  `matrix-comparison-subtests.log`. The latter overlaps13 prior tests and adds
  one regression.

### Finite interface and asset review

- The reviewed semantic table dispositions all485 chemistry classes and1837
  captured entries. It includes204 exact carrier/member overrides,15 ordinary
  or dynamic supplements,62 enum domains and source/schema bindings. Public
  contracts, support types, retirement candidates and private implementations
  remain distinct. Original raw inventory observations are preserved.
- Parent review corrected erroneous experiment-stage references in the draft
  table: **Stage0–12 → committed B_exp → new experimental branch → Stage13A
  D_exp → Stage13B I_exp**. It also corrected active-versus-deleted aromatic
  entry points, reference/view stages and D04/D05 ownership. The new review
  reread the implementation/acceptance/decision/experiment chapters and the
  normative companions; the main route itself did not change.
- A2 remains explicitly accepted. N01–N09/A4–A5 layouts/new public interfaces,
  D01–D03 schemas and pair-result details retain their later-stage gates.
  New findings include tmQM Wiberg/frac_valence writers and model pair-column
  order, shallow frozen wrapper/native/message buffers, and additional baseline
  conformer/presence defects. Earlier table/cache extraction must preserve
  those scientific values before the later ownership cutover.
- The assembly audit compared all four files in the three named locations.
  They are not one interchangeable template schema. The sole active default
  is absent from the existing wheels; Stage8E/10 must close its packaging gap.
  Legacy fragment definitions and the independent36-entry framework collection
  retain their explicit Stage11/Decision13 disposition. No asset was deleted.
- The measured production record delta remains zero. The same-method budget's
  conditional ceiling is354 (net+7) only if all reviewed consolidations land;
  without D01–D03 approval the conservative ceiling is362. No net whole-project
  reduction or early record creation is claimed.

### Evidence retention and next-stage boundary

- Copied **7946 files /478398549 bytes** of completed raw scientific, technical,
  inference, performance, wheel and preparation evidence into
  `.plan20-artifacts/20261028/b20-archive`. Every copied file matched its original
  SHA256. `archive-index.json` maps original paths to the durable copies;
  original files remain in place and historical paths inside reports are not
  rewritten. The matrix already resides in the same project artifact area.
- All ten Stage1 responsibilities now have recorded deliverables: explicit
  surface/consumer inventories; type-role/alias and semantic tables; ratcheted
  debt/migration-stage maps; frozen deterministic technical fixtures and actual
  test outcomes; O0 roster/report/comparison; source/native/installed validation;
  complete existing performance families plus the Python matrix; frozen
  sampling/work policies; concrete interface review; and the net-object budget.
- README/main-plan status and the verification map now reflect this close.
  Stage2 has not begun. It requires a fresh full plan/companion/decision read,
  current-project/predecessor assessment and its own zero-reference/build audit
  after the Stage1 Git checkpoint. No production deletion or extraction is
  bundled into this commit.

## 2026-10-28T09:34:42.351595+00:00 — Stage2 entry: confirmed dead/placeholder cleanup

- Predecessor Stage1 commit: `2aca9f1df18d7821f8a7ccefc873d949317d6c9f`.
  The worktree was clean before the entry review. Production subtree remains
  `4a2086e717327c99056ef8cda200b0786e688219`.
- A fresh full read covered all17 main-plan chapters/all2434 lines, the four
  normative companions including the full data-object appendix, and the latest
  A2 decisions, API/semantic reviews, log, README and execution evidence.
  `stage2_entry_review.json` records each file/range/hash, predecessor verification
  and attributed source findings; entry SHA256 `f4e976ed05ce7e70fd7194e360aa5faf47864d454aee52bb30405a28f11802b4`.
  This was a new read, not a reused Stage1 reading or diff-only check.
- The parent completed exact source/export/consumer/build review. Remove the
  named unused parser, empty Calculator, old Cython/SWIG implementation,
  duplicate ob2chem layer and broken spreadsheet-driven APIs. Migrate the two
  actual test consumers through existing interfaces. Canonical obconvert
  keeps its nine-atom/three-bond-column arrays; aromatic transport assertions
  move to its existing full molecule conversion, not a new array schema.
- Remove the connected Crystal space/lattice placeholders together, retaining
  its mol/obcell and six-parameter mathematics for their later stages. Remove
  the empty Molecule.assign_aromatic and joint-ring assignment/facade/export
  chain, retaining every implemented aromatic/Kekule algorithm and check.
- The initial conservative hold on AtomPairKey was resolved by a separate
  explicit public/consumer/history review in the entry record. Section9.1
  already names its deletion under the user-authorized concrete plan. It is a
  directly visible core symbol but absent from the captured root/cheminfo
  surfaces, with no repository consumer, documented usage, codec or native ABI.
  Remove that exact wrapper; preserve AtomPair/AtomPairs and their Wiberg values,
  and every other D07 reserved/documented contract. No absence of unknown
  external users or approval of an unlisted new schema is inferred.
- Existing solvent/other molecule assets stay unchanged for Stage11 governance.
  Production extraction, fact storage, chemistry algorithms, MCA route,
  wrapper/ABI source and Stage13 protocol work remain their assigned stages.
- Test environment: use the Stage1 CPython3.11 interpreter and B20 native bytes.
  Archive the different existing RD extension and retired cp311 relevant-cycle
  binary before installing the verified B20 RD extension; no C++ rebuild or
  alteration of other ABI files. Record original hashes and paths.
- Gates: complete default suite, explicit fresh-process removed-module/import
  smoke, supported conversion/aromatic/Crystal behavior, and final source/build
  zero-reference checks. Record old-to-new test assertions and all outcomes
  before the focused Stage2 commit.

## 2026-10-28 — Stage2 validation and cleanup close

- Removed the six named legacy Python/Cython modules and the eight tracked
  `_clib` files. Removed Calculator, AtomPairKey, LatticeGroup and their named
  placeholder facades/exports. Kept actual solvent assets, Crystal's existing
  mol/obcell/math behavior, AtomPair/AtomPairs scientific values, and every
  implemented aromatic/Kekule operation. No scientific algorithm changed.
- Migrated two existing tests, without adding Stage2 tests: the Calculator
  export expectation no longer names the empty class; the historical
  `test_ob2chem_compatibility_contracts` node now checks canonical obconvert,
  read_mol and Molecule.to_obmol. Its old four-column duplicate bond array
  assertion becomes the canonical three-column array plus aromatic-source
  metadata on the complete imported molecule. The atom shape, mappings,
  round-trip and delegation evidence remain. Test node identity is retained
  for the frozen Python matrix.
- Focused calculator/conversion/bond-metadata/import-safety selection:
  **27 passed**,3.69s. Complete default suite: **3882 passed,123 skipped,
  3 xfailed,49 subtests passed**,4 warnings,234.10s; process exit0.
  The 60 additional passed nodes compared with the initial B20 default suite
  are the Stage1 reporting tests, not new Stage2 chemistry coverage.
- Fresh-process smoke verified seven absent module paths, eight absent
  members, nine retained members, Crystal matrix/obcell behavior and all five
  loaded cp311 native hashes against B20. The previous RD binary and obsolete
  relevant-cycle extension were archived before binding the B20 environment;
  no modern C++ source/build/ABI change or rebuild occurred. The retired
  `_clib` directory, including ignored caches, is preserved in stage artifacts.
- Bound mutable Stage1 review/budget/manifest references to committed-byte
  copies from `2aca9f1`, preserving original_path and original SHA256. All43
  Stage1 Plan20 files are retained in stage1-records with their snapshot index.
  Rechecking all45 indexed path/digest references, resolving wheel bindings
  relative to their matrix report, found zero mismatches. Later stage edits
  therefore cannot silently alter historical baseline records.
- Final tracked-production AST count:403→396 Python files,782→773 classes;
  the nine removals are ordinary classes. Dataclasses remain347, Protocols13
  and TypedDicts3. No data-record reduction or addition is claimed. Retained
  function ASTs and modern native/wrapper sources are unchanged.
- Stage2 execution evidence is indexed in `stage2_execution_evidence.json`.
  The final source/build audit and counts are bound there before commit.
  Scientific/performance tiers were not rerun for this deletion-only stage;
  their Stage1 baselines remain intact. Stage3 requires its own new full read,
  predecessor assessment, contract/layout review and storage measurements.

## 2026-10-28 — Stage3 entry: contracts, declarations and storage assessment

- Predecessor Stage2 commit: `c6bfcd4906254ee113c14103009f071e91f60368`;
  clean tracked worktree. Production subtree:
  `85f86bbf8a3b83b1322f20181c2f9996bb3c41d3`.
- Newly read all2434 lines/all17 main-plan chapters and all four normative
  companions, including the complete data-object definition appendix. An
  oversized migration-table output was reread in a smaller range. Exact hashes
  and immutable read copies are recorded in `stage3_entry_review.json`.
- Rechecked Stage1/2 indexed evidence; previous technical/import/source gates
  passed. Stage2 source cleanup did not change scientific algorithms. Current
  class definitions, topology index, mutable arrays and legacy setter behavior
  still reside in Core; there is no `_topology.py` implementation yet.
- Entry permits bounded private layout/reuse review and equivalent-work storage
  measurements before production choice. N01–N09 reservations are upper
  bounds; retain existing scientific/stereo/native schemas. No new wrapper for
  each field, no universal patch or early experimental rule protocol.
- A2 is accepted and remains staged: private ownership/import preparation now,
  constructor cutover in Stage5, selected-result/NaN policy in Stage7C. Keep
  still-needed Wiberg, partial-charge, constraint and model-input values during
  each earlier migration. A4/A5 new methods retain their specific decision gate.
- Required Stage3 evidence includes builder one-shot topology/no chemistry,
  UID non-reuse/owner checks, buffer isolation, atomic writes, role/permission
  declarations, positive/negative dependency propagation and scalar/batch
  scaling, ratchet/import contracts, appropriate Python coverage and default
  suite. None of those new gates is yet claimed complete.

### Stage3 owner updates and GitHub cadence

- The owner approves the two proposed capabilities with revised public names:
  `Molecule.updata_atoms(*, atoms=None, **columns) -> None` (exact spelling), and
  `Molecule.atom_matrix` as a property returning the complete float64 array,
  without public field selection. These supersede the proposed
  update_atom_facts/atom_feature_matrix names. Existing atom_attr_matrix already
  implements the19-column projection; its precise rename/alias disposition is
  being clarified against the earlier A2 preservation decision. No second
  fact table or independent matrix-generation implementation is authorized.
- New instruction: push GitHub after every three commits. Initial catch-up push
  succeeded for implementation branch through
  `c6bfcd4906254ee113c14103009f071e91f60368`, now tracking
  `origin/implement/plan20-fact-rule-architecture`. Count subsequent commits
  from this pushed SHA; uncommitted Stage3 work was not included.

### Stage3 foundation checkpoint — verified, stage still in progress

- Created the six empty-export ownership packages, without moving carriers.
  Element and ideal-bond JSON assets now have unique, deeply immutable cached
  loaders. The existing public element dict/list is an independent snapshot.
- Added the architecture ratchet against the frozen inventory: 287 initial
  entries, 282 remaining, no new debt. Review repaired checks for the approved
  `updata_atoms` name, imported module aliases, and direct numeric fact writes.
  Source checks complement, rather than replace, runtime ownership tests.
- Full default suite: **3917 passed, 123 skipped, 3 xfailed, 49 subtests**, in
  242.15 seconds. Production code did not change afterward. Final focused run,
  including the 15 additional ratchet cases: **74 passed** in 9.10 seconds
  (46 architecture, 4 reference, 24 performance-report tests). The six focused
  warnings concern existing unrelated invalid escape sequences in source ASTs.
- Two retained storage experiments cover 936 and 1248 attempts, with zero
  execution failures and 33/36 validation checks. Typed columns improve larger
  exports but regress small-molecule matrix export/remap. Grouped float64
  blocks improve complete matrix export while showing scalar/remap overhead.
  Neither experiment includes production handle, presence, status or commit
  costs; neither selects or validates the production layout. Wide existing
  charge/H/id and raw numeric slot values remain required.
- The revised interface review included another fresh complete read of the
  main plan and four normative companions. Its exact retained manifest and
  checkpoint evidence are indexed by `stage3_foundation_evidence.json`.
- Table/schema integration, UID lifecycle, builder, atomic commits, dependency
  propagation, production scaling and candidate Python coverage remain open.
  The verification map explicitly retains those gates as uncompleted. The
  frozen five-column float32 private benchmark awaits the actual projector;
  it is not substituted by the public complete19-column float64 property.

### Correction — Stage3 architecture scan must include package roots

- Reviewing the complete source/class census exposed a Git pathspec error in
  the foundation scanner: `hotpot/**/*.py` and `tests/**/*.py` omitted Python
  files directly under those roots. The Stage1 frozen inventory included them.
  The scanner now enumerates both directories and filters Python filenames;
  a regression fixture covers package roots, test roots and nested modules.
- The earlier **282** remaining-debt count was incomplete. Three original
  `hotpot/__init__.py` records (eager RDKit import, wildcard export, sys.path
  mutation) were never retired. Its bytes still equal the last pushed
  `c6bfcd4` checkpoint. Restoring those original records gives **285** actual
  remaining entries out of the frozen287; no new production violation is added.
- This explicit coverage repair is checked against `c6bfcd4` and the original
  inventory, then ordinary HEAD/merge-base ratcheting resumes on the corrected
  list. No permanent exception or growth override is introduced. Existing
  tests continue to reject new debt and reintroduced retired entries.
- Exact correction evidence is in
  `stage3_architecture_coverage_correction.json`. Historical foundation evidence
  remains retained; this record supersedes its scan-coverage/count conclusion.
  Reference results, technical outcomes and measured storage data are intact.

### Stage3 shared Atom schema and private numeric table checkpoint

- Current Atom's 19-field dict now derives from `_fields.py`; order, scalar
  codecs and dict semantics are identical. A source audit confirms all330
  existing Core function ASTs are unchanged. No chemical algorithm or current
  Molecule storage changed. `_AtomFactTable` is one private implementation
  class; the full production census is774, with no new dataclass, Protocol,
  TypedDict or native mirror.
- Implemented two owned float64 blocks, scalar/raw projections, independent
  exports, selected-row updates, capacity growth, remap and independent copies.
  Tests preserve fractional raw Boolean slots and IDs beyond int64. Scalar
  and batch fault injection verifies restoration of the interrupted column
  as well as earlier writes. Exact private scope and complete future writer
  cutover are recorded in `stage3_storage_review.md`.
- Initial and refined real-table experiments each completed1560 attempts and
  39 equivalence/ownership checks. All unfavorable results are retained. The
  refined table improves bulk exports, but still has roughly4.8x scalar-write
  and8.6x N16 remap overhead against bare arrays. Full facade/commit performance
  is not measured and production parity is not claimed.
- Complete default suite: **3970 passed, 123 skipped, 3 xfailed, 49 subtests**,
  in236.22 seconds. Production modules have not changed afterward. The separate
  scanner-coverage correction then received final focused checks on Python
  **3.10–3.14:89 passed per version** (38 table, 47 architecture, 4 reference).
  These are current-source checks, not a new installed-wheel/native matrix.
- Corrected architecture scan includes roots, leaves285 original debt entries,
  and passes ordinary HEAD ratcheting after the dedicated `8a7c6a6` repair.
  `stage3_storage_evidence.json` binds commands, versions, source identities,
  raw experiments and the distinction between primitive and molecular gates.
- Stage3 remains in progress. Owner/UID/presence/source, complete import
  builder, closed molecular commits, lifecycle migration and real integration
  scaling remain open. The approved public methods are not yet published;
  old `atom_attr_matrix` disposition remains under the existing clarification.
- This checkpoint is the third commit after the last push through `c6bfcd4`,
  following `f9f87da` and `8a7c6a6`; push the implementation branch immediately
  after committing it, then restart the three-commit counter.
