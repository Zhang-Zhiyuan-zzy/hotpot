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

### Stage3 member/source identity and private metadata checkpoint

- Read the full current main plan and four normative companions again before
  this continuation; hashes and complete retained read copies are recorded in
  `stage3_continuation_review.json`. This checkpoint continues Stage3 only.
- Molecule owns the lock, opaque owner token and non-reusing Atom/Bond UID
  high-water marks. Copy/insertion/merge/replacement create local member
  handles; hidden Bond endpoints and single-Atom transport records survive
  replacement. Raw fractional/electron slots survive copy/components. The
  current-version pickle restores the lock and existing topology-index keys.
- One private `_SourceIdentity(owner, dependencies)` record captures only
  requested domains; owner-only checks do not hash facts. Legacy writable
  arrays/writers still bypass the new lock, so concurrent molecular snapshots
  and result validity are not claimed. Source frame provenance remains Stage7D.
- The private numeric table adds per-row presence bits and aligned producer
  codes, preserving explicit zero and numeric/metadata rollback together. It
  remains a private candidate, not a second authority installed on Molecule.
  The full production census is775 classes/348 dataclasses/13 Protocols/
  3 TypedDicts; N01 is the only new record. Architecture debt stays285.
- The later owner instruction to rename and avoid duplication resolves the
  earlier matrix clarification: directly rename `atom_attr_matrix` to
  `atom_matrix`, migrate live docs/tests and retain no old-name alias. It
  returns an independent complete19-column float64 array, including `(0,19)`
  for empty input. This uses existing explicit authorization, not a new answer
  to the optional clarification. `updata_atoms` awaits the closed commit path.
- Accepted-source final default run: **4045 passed,123 skipped,3 xfailed,
  49 subtests**,10 warnings,239.84 seconds. Python **3.10–3.14:222 passed per
  version** on the affected source selection. Hashes bind the exact frozen
  production bytes. Matching graph binaries were missing in four source
  environments initially; SHA-verified original B20 wheels supplied unchanged
  binaries. This is not a new candidate installed-wheel/native build.
- Retained failed attempts include the missing-binary matrix and a shared-empty
  StereoPayload optimization that violated per-Molecule payload ownership.
  Unique payloads were restored before accepted validation. Its earlier
  approximately0.633x constructor timing is withdrawn as acceptable evidence.
- The thirteen-workload final B20 diagnostic has small-molecule construction
  at1.065x (95% CI1.036–1.127); other unmeasured families remain inconclusive.
  New private metadata writes cost about11.4–11.7x bare-array writes and small
  remaps about13.8x. Bulk matrix export improves. All attempts are retained;
  non-interleaved shared-host diagnostics do not establish global acceptance.
- `stage3_identity_metadata_evidence.json` binds commands, logs, original and
  final attempts, binary preparation, source audits and performance records.
  A separate read-only review found no new checkpoint blocker and verified
  current pickle/deepcopy ownership. Production table/writer cutover, complete
  factual builder, atomic commits, parent/detached readonly, finite dependency
  reads and integrated scaling still block Stage3 completion.
- Last push is`40312df`; this checkpoint is the first subsequent commit.
  Push after two more commits, without manufacturing commits for cadence.

### Stage3 closed update preparation and unique table serialization

- Again read the full2448-line main plan and four normative companions
  (3498 lines total) before this connected-writer preparation. Exact hashes,
  read copies, synchronized predecessor`d7989b3`, project-state assessment and
  bounded tasks are in`stage3_cutover_entry_review.json`. This stays Stage3.
- Froze N05 fields/consumers before code. One private `_AtomFactUpdate` owns
  source, aligned row/UID tuples, canonical fields, immutable numeric bytes and
  per-field presence/producer codes. Approved scalar/batch columns use the
  same normalizer; overlapping aliases must agree. No molecular installer or
  public batch method is published before the complete owner/writer cutover.
- An independent review caught retained ndarray header mutability despite
  read-only elements, and Unicode casing that could turn unknown spellings
  into real elements. Replaced array authority with bytes/transient headers
  and restricted symbol casing to ASCII; regression cases cover both.
  Joint/standalone pickle checks preserve/reject owner bindings as appropriate.
- Shared the existing Element.symbols tuple with the reference module, leaving
  all Element scientific-function ASTs and Core bytes unchanged. The property
  JSON has different atomic numbers for In/Tb/Yb/Bk; it was not substituted
  for existing behavior or corrected in this refactor. The full118-element
  spelling comparison and D/T/unknown cases match current Open Babel.
- Reproduced a private-table pickle/deepcopy defect: separately serialized
  column views could update raw reads while coordinates/matrix stayed old.
  Serialize unique live owner blocks/metadata, preserve capacity without
  reading spare slots and reconstruct views. Table coverage is81 cases.
- Accepted final source: **4096 passed,123 skipped,3 xfailed,49 subtests**,
  10 warnings,241.07 seconds. Affected source selection on Python
  **3.10–3.14:273 passed per version**, including42 update candidate cases.
  This is source coverage, not a new installed-wheel/native rebuild.
- The initial candidate's271/version and4094 default passing tests are
  retained as superseded evidence: review found the two gaps above despite
  those passes. A new owner-pickle fixture briefly used an unsupported
  positional create_atom call; it was corrected to the existing keyword API.
  No production behavior was changed to accommodate that fixture.
- New-only preparation measurements retain104 attempts and no failures:
  one-row x~11.1us, coordinate aliases~24.0us;4096 rows~0.787/0.814ms.
  These exclude member selection, installation and dirty work and have no
  fabricated B20 comparator. Their scalar cost remains an integrated
  efficiency concern. Prior unfavorable table results remain retained.
- Census:776 classes,349 dataclasses,13 Protocols,3 TypedDicts. N05 is the
  only added record this checkpoint; architecture debt stays285. Evidence and
  exact frozen source/commands are in`stage3_atom_update_evidence.json`.
- Production table/connected writers/FF rollback, complete builder and
  metadata installation, parent/detached readonly, dependency reads and real
  facade scaling remain open. Subsequent builder work may first unify current
  representable factual imports using the existing unique row owner, then
  change that same installer during table cutover. It must not accept and
  discard metadata or create a temporary shadow owner/mode.
- This checkpoint is the second commit since last push`40312df`, following
  `d7989b3`. Push after the next coherent commit.

### Stage3 readonly Atom parent invariant

- Continued Stage3 from clean`5a5006f` with the latest Sol-first delegation
  policy and both project stage/validation skills. The bounded Sol-max builder
  review performs the fresh full main/companion read and freezes the next
  import-record design; this local change implements the already specified
  readonly-parent lifecycle without changing the table/builder route.
- The repository's only Atom-parent assignment is its constructor. It now
  binds private `_mol` directly; existing `mol` is a readonly property and
  `molecule` still projects it. Copy/insertion/replacement already create local
  handles, so no additional reparenting writer or old/new mode is introduced.
  Other types' parent properties and chemical implementations are unchanged.
- Added attached/detached × same/foreign assignment rejection and verified
  owner/UID/topology retention, plus Atom/Bond/Molecule-root pickle/deepcopy.
  Current-version serialization succeeds; no historical-pickle fallback is
  added. UID/lifecycle coverage is22 cases.
- Default: **4106 passed,123 skipped,3 xfailed,49 subtests**,10 warnings,
  238.97 seconds. Python **3.10–3.14 each283 passed** on the affected source
  selection. These do not assert a new installed-wheel/native build.
- Thirteen frozen B20 diagnostic tasks retain all attempts. Construction is
  about1.063/1.057/1.067x at N8/64/256 and insert-delete about1.044–1.050x.
  These shared-host, non-interleaved interim regressions remain unresolved;
  no global parity, improvement or rare exception is approved. Actual table,
  scalar/batch and final workflow performance still require integration.
- `stage3_parent_evidence.json` binds frozen source, commands, results and
  measurement limits. The complete Stage3 owner/builder/commit/dirty gates
  remain open. No new class or public constructor signature was added.
- This is the third coherent commit since last push`40312df`, following
  `d7989b3` and`5a5006f`; push the implementation branch immediately afterward.

### Stage3 complete immutable import-record preparation

- The third-commit push succeeded through`ce41629`; the three-commit counter
  restarts there. Continued the owner's instruction to finish all Plan20 under
  the latest explicit Sol/effort policy; no later-stage gate is bypassed.
- Sol-max read the full3498 lines of main/normative companions again and
  assessed clean`5a5006f` before the bounded architecture design. Hashes,
  consumer evidence, exact22 N02 fields and installation boundaries are in
  `stage3_builder_review.md`. The separate readonly-parent work subsequently
  reached`ce41629`. Normal implementation used an isolated Sol-medium draft;
  the loader was not installed into the project.
- Added one private `MoleculeRecord` and finite lossless codecs, preserving
  raw Atom/Bond numbers, explicit-zero versus absent masks, heterogeneous
  producer rows, directional/hidden Bonds, source metadata, sparse inputs,
  cell/declarations, protected stereo classes and all nine old frame domains.
  Immutable bytes/tuples are authoritative; decoded headers/containers are
  independent. No generic opaque-pickle fallback, public export, extra DTO
  family or unimplemented installer placeholder is added.
- An initial75-case implementation passed all358/version and4181 default
  tests, but independent probes found wrong-position stereo node types and
  NumPy shapes that accepted bytes could not decode. Retained source/probe
  artifacts document both gaps. Added exact nested-type checks and a temporary
  NumPy header check, preserving arbitrary version/descriptor/scientific
  values. Final record coverage is111 cases; original75 fixtures remain.
- Accepted source: default **4217 passed,123 skipped,3 xfailed,49 subtests**,
  10 warnings,239.65 seconds. Python **3.10–3.14 each394 passed** on the full
  affected source selection. Independent adjacent probes confirm both fixes,
  six pickle protocols and88 array/scalar dtype round trips. These are not
  production installer or rebuilt-wheel/native evidence.
- All pre-existing production files remain unchanged from`ce41629`.
  Census is777 classes/350 dataclasses/13 Protocols/3 TypedDicts; N02 is the
  only added record this checkpoint. Architecture debt stays285.
- New-only record preparation/pickle measurements retain78 attempts, zero
  failures. At N16/256/4096 with B=N-1, four metadata entries per Bond and
  two source frames: preparation~0.261/2.624/60.751ms; pickle roundtrip
  ~0.181/1.844/30.568ms. Installation/UID/topology/rollback are excluded;
  no old-baseline speedup or integrated parity is claimed.
- `stage3_record_evidence.json` binds exact sources, commands, all-version/
  default results, rejected attempt and diagnostic costs. The complete
  production table/writer/lifecycle/FF rollback migration is the next coherent
  batch, with Sol-max independently rereading/designing that bounded step.
  Existing opaque metadata behavior, converter identity and technical defects
  require their concrete handling; isolated-codec restrictions do not approve
  silent fact loss or a new restriction on public copy/import behavior.
- This checkpoint is the first commit after push`ce41629`; push after two
  further coherent commits. Stage3 remains in progress.

### Stage3 production authority integration — exploratory checkpoint

- Continued from `caf959eda9c68b5ef1d2cbd60dc570d0a273ab18` under the owner's
  request to finish all Plan20. The full3498 lines of the main plan/four
  normative companions were reread before this architectural continuation;
  hashes, source/worktree/environment and bounded ownership are recorded in
  `stage3_production_entry_review.json`. This is not a Stage4 entry.
- Sol-max owns Core/global cutover design; an explicitly spawned
  `gpt-6.1-sol/high` worker implemented the settled engine then factual
  decoders sequentially, and performed a read-only cross-review. The parent
  owns FF/outer consumers and evidence. No child was authorized to delegate.
  The stale parent-only agent declined the new task because its model metadata
  could not be verified; it made no new edits. No Astra escalation has occurred.
- The uncommitted candidate now connects one Molecule-owned Atom table,
  controlled row/matrix projections, approved `updata_atoms`, private factual
  build/install methods, complete imports and FF publication/rollback. The
  actual private builder methods reside on the still-single carrier so that
  Stage3 does not introduce new direct-Core consumers; the record/codec stays
  pure in `_builder.py`. This follows the settled cutover review, with no
  registry, storage mode or extra public signature. Stage5 eager-chemistry
  removal and Stage6/7 final ownership remain at their assigned steps.
- The source architecture scan found no new debt after owner repair. Four
  actual old writes are retired (two hidden-list removal assignments and
  runtime-context copying in FF); the remaining allowlist is281. Its source
  gate must be rerun on frozen final bytes. No scope/ceiling exemption was added.
- Intermediate focused source outcomes reported/executed: engine183 passed;
  record/import selection221 passed; broader affected selection503 passed with
  one then-unretired ratchet failure; FF/native/xTB213 passed,8 skipped.
  Licensed CCDC and official external xTB tiers were not executed. The CCDC
  fake characterization rerun with explicit `PYTHONPATH` verified the baseline
  `.atom` typo and missing return against HEAD's converter source; the current
  fake boundary verifies `.atoms`, return identity and factual values.
- Exploratory default command and per-file source-before/source-after hashes
  are retained in `.plan20-artifacts/20261029/stage3/production-exploratory/`
  under `default-attempt1`:4299 passed,8 failed,123 skipped,3 xfailed,
  49 subtests,10 warnings,245.16 pytest seconds. Production bytes changed
  during execution, so this is not accepted frozen-source evidence.
- Old-to-new assertions retain their original technical purposes: attrs object
  identity becomes exact numeric/isolation/owner restoration; malformed
  Gate6 transport is injected at the actual private raw owner rather than a
  failed public setter that previously left partial data; SMARTS permutation
  remaps table/UIDs through the closed factual commit instead of reordering
  membership alone; stereo storage fixtures use the real locked carrier;
  FF fakes accept the new private source binding. Worker-only copies explicitly
  clear annotation containers before spawning, preserving their structure-only
  payload guarantee. A recheck of the eight failing areas plus new outer
  regressions yielded64 passed; source was still changing, so final evidence
  remains outstanding.
- Independent reproductions found genuine gaps despite focused passes: live
  extra angle constraints could overwrite a frozen serialization record, and
  a replaced working-copy Atom with duplicate source labels/chemistry could
  pass FF's old identity test. The candidate now reconstructs fact-bearing
  constraints from the frozen record and validates fresh-copy row/UID/owner
  membership, with regressions. Bond member-root lookup's ordinary-path cost
  and lock-controlled existing angle/torsion constraints are also under repair.
- Real B20/current diagnostics expose large unresolved costs (initial scalar
  write64 about151x, construction8/64 about14x/8.2x; coordinates64 about0.75x).
  These use equivalent frozen tasks but shared load/concurrent source and are
  diagnostics, not acceptance. Sol-max is optimizing the measured commit,
  dependency and constructor paths without bypassing raw writes, undo or
  source checks. No global performance pass or exception is asserted.
- No production commit has been made at this exploratory point. Last push is
  `ce41629`; `caf959e` remains the first following commit. The next coherent
  commits continue the every-three-commits push cadence, without artificial
  cadence-only commits. Stage3 is incomplete until final integration,
  source/Python/affected-wheel and performance gates have actual evidence.

### Stage3 production authority integration — frozen Python checkpoint

- The complete default suite now passes on frozen production Python bytes:
  **4349 passed,123 skipped,3 xfailed,49 subtests,10 warnings**,244.26 pytest
  seconds (246.11 runner seconds). Exact argv, environment, all production
  file hashes before/after, JUnit and log are bound by
  `stage3_python_cutover_evidence.json`. The command was
  `python .plan20-artifacts/20261029/stage3/production-exploratory/run_check.py default-python-engine-frozen tests`;
  the recorded child argv uses the selected Python3.11 interpreter and CPU24–31.
  Six newly added whitespace-only lines in Core were cleaned before this run.
- The final engine/structure/carrier/outer focused selection separately passed
  **466 tests in8.75s** without production changes during execution. This is
  the Python implementation, not evidence for a future compiled writer.
  The source architecture audit passes with281 remaining debt entries and
  zero new debt, allowlist growth or layer findings. It was executed as
  `python plan/20_fact_rule_architecture/audit_architecture.py --output .plan20-artifacts/20261029/stage3/production-exploratory/default-python-engine-frozen/architecture.json`.
- A genuine profiled Sol-max attempt reduced engine cost but did not solve
  the same bounded performance problem. The owner policy therefore authorized
  one explicit Astra/xhigh escalation; this supersedes the earlier exploratory
  entry's then-current statement that no escalation had occurred. The engine
  optimization combines physical-buffer undo, vectorized bulk capture and
  finite-group dispatch while preserving sources, membership checks and
  rollback. New regressions cover post-store failure and overlapping writes
  across growth/removal, including removal followed by registration.
- Frozen-runner paired diagnostics remain blocking: scalar8/64/256 ratios
  are27.06/28.82/27.94; batch8/64/256 ratios are14.37/3.33/2.00; complete
  write/read64 is11.60. All use the frozen semantic tasks,3 warmups and10
  samples on the same core. Shared-host measurements are diagnostics, not a
  global efficiency certificate. The attempted nonexistent4096 workload and
  its no-output failure are retained. The complete237-file Astra artifact
  archive and hashes live at
  `.plan20-artifacts/20261029/stage3/commit-performance-astra/` and are bound
  by `stage3_commit_performance_escalation.json`.
- The root and Sol-max reread all3498 current main/companion lines before
  reviewing a single private compiled ingress/preparation/installer. Actual
  N05 transport/pickle, sole NumPy authority, owner/UID checks, outer undo and
  BaseException restoration remain requirements. No native production code
  was installed for this checkpoint. The final affected Python3.10–3.14,
  current-wheel/native, scientific and performance gates are still open;
  Stage3 is not complete and Stage4 has not started.
- Read-only environment probes found the five existing interpreters and old
  wheels usable, but those old wheels are not current-source evidence. The
  Python3.13 `/tmp` build environment lacked build tools; the authorized
  prerequisite command `python -m pip install 'pybind11>=3.0' setuptools build wheel ninja`
  completed there with pybind11=3.1.0,setuptools=84.2.1,build=1.6.1,
  wheel=0.48.0,ninja=1.13.2,pyproject_hooks=1.3.3. Other environment paths
  and production sources were not changed by this installation.

### Stage-record layout — mechanical organization on 2026-10-29

- The owner explicitly requested stage-based document directories while the
  main process continued implementation. Starting from
  `aeda200868209243aabf8b8c611765fdde142710`, the assigned Sol-max task moved
  28 existing records (9 Markdown,19 JSON) into `stages/stage00` through
  `stage03`, with topic directories for repeated Stage3 reviews. The complete
  old/new list and original/relocated identities are in
  `stages/path_migration.json`; every stage has a complete README index.
- `stages/README.md` defines the finite Markdown filenames `README.md`,
  `entry_review.md`, `review.md`, `decisions.md`, `report.md`. JSON names remain
  purpose-specific and each JSON is indexed by the corresponding README.
  No empty later-stage directory was created. Main/normative and genuinely
  cross-stage manifests, ledgers, baseline inventory and tools retain their
  existing ownership. The chemical-type companion has one mechanical link
  adjustment; plan objectives, stages, gates, contracts and decisions did not
  change. This move does not assert a new normative reread or test rerun.
- All19 historical JSON files retain their exact bytes. Existing Markdown
  command fences, digest-bound identity rows and all preceding implementation
  log text were preserved against the checkpoint. Active root navigation and
  maintained-ledger references were updated; historical checkout filenames
  resolve through the migration index rather than alias files or rewritten
  evidence identities.
- Executed `taskset -c 24-31 python plan/20_fact_rule_architecture/audit_stage_records.py --check-migration-bytes --output plan/20_fact_rule_architecture/stages/layout_evidence.json`.
  The read-only audit checks Markdown types, complete README indexes, JSON,
  local links, migration targets and active external paths. An isolated copied
  plan also verified rejection of an unknown Markdown type, unindexed Markdown
  and JSON records, and a missing link target. Exact results, interpreter,
  affinity and audit/probe identities are retained in
  `stages/layout_evidence.json`. This is document-layout evidence, not native,
  default-suite, scientific or performance acceptance.
- Core/native/build implementation remained outside this document change.
  Core integration resumes under its already frozen private hooks after the
  document handoff. No commit/push was performed by the document worker; the
  parent owns the next coherent documentation commit and push cadence.

### Stage3 native integration — first compiled checkpoint, performance open

- The reliable Python integration was committed as
  `aeda200868209243aabf8b8c611765fdde142710`. The owner-requested stage-record
  layout was separately committed as`ba662e3` and pushed successfully to
  `origin/implement/plan20-fact-rule-architecture`, completing the three-commit
  cadence since`ce41629`. Native/build changes were excluded from that
  documentation commit. The next implementation commit starts a new cadence.
- The sole private`structure._fact_native` now supplies scalar/column
  preparation and physical installation/restoration against the existing
  NumPy owners. Python update numeric normalizer/store/restore bodies are removed;
  actual seven-field N05 bytes/pickle remain. Native and Python dependency
  mutations enroll in one ordered physical undo journal. Core retains its
  existing eager scientific bodies and later-stage ownership cutovers.
- Four recorded Python3.11 builds were attempted. The first two failed on a
  C++ conditional pointer conversion and a pybind11 tuple conversion; the
  third and fourth compiled. Focused runs found a real NumPy1D selection
  classification bug and an extension-loading gap on first append. Both were
  repaired; first-transaction loading precedes lock acquisition and stores.
  Original compiler/test failures remain retained, rather than replaced by
  the later successful attempt.
- The first complete compiled focused selection passed **521 tests,6
  warnings in10.68s**, followed by **59 passed in0.96s** on the exact existing
  Stage3 targets and additional Bond/stereo working-copy contracts. Their
  source/native/test hashes were stable during execution. Commands, logs,
  dirty-source/binary snapshots, build identities and coverage limits are
  indexed in`stages/stage03/native/evidence.json`; the complete raw archive
  lives under`.plan20-artifacts/20261029/stage3/fact-native/native-v1-checkpoint/`.
  These are Python3.11 source/native checks, not all-ABI/wheel completion.
- The same frozen seven paired tasks still fail performance policy:
  scalar8/64/256 ratios5.756/6.129/6.030, batch8/64/256
  ratios4.424/2.313/2.059, and complete write/read64 ratio5.229. CPU0,
  three warmups and ten measured samples were used on both B20 and candidate;
  full raw attempts, load and source/native identities remain in the archive.
  Ordinary scalar64 now costs about2.40us/write against B20 about0.39us.
  This improvement over the earlier Python attempt does not establish parity.
- A subsequent public-signature review found that native special-method
  binding must retain the successful bound and unbound keyword calls of the
  frozen`MolBlock.__setattr__(self,key,value)`, including unbound`self=`.
  The Stage1 inventory explicitly covers that interface. Its repair and
  independent undo review remain in progress; no public contract was waived.
- Further engine optimization targets measured temporary allocation,
  attribute lookup and ContextVar overhead within the accepted boundary.
  Stage3 remains incomplete; the final default, five-ABI/source/wheel/sdist,
  relevant scientific and performance gates remain open. No Stage4 or
  Stage13 implementation has started.

### Stage3 native undo — frozen build8 and gate interpretation correction

- A real allocator probe demonstrated that membership-only seed restoration
  could allocate during rollback, partially restore facts/seeds and replace
  the original exception. The original eight rejection/control subprocesses
  are retained. The Sol-max-reviewed revision now captures physical bucket
  entry/counter beforeimages, and preserves actual prior set bodies at
  clear/rehash boundaries. It retains the same active Python seed sets and
  ordered journal. Applied/cancelled state prevents restoring uninstalled
  candidates and duplicate outer replay. No Python writer fallback or later
  scientific owner was introduced.
- Final build8 C++ SHA-256 is
  `482ac5feb9778ef9ebc39f7f7e40be320bd7d4a61aecde9da8b18f04465f91c1`;
  Core SHA-256 is
  `5dba619e4a46f9c5fc6faa0019fbc931b02127ba5d71d152292874e3cfcdd595`.
  The actual matching CPython3.10–3.14 extensions compiled successfully.
  The first3.13/3.14 attempt failed because `_PySet_Dummy` is no longer
  declared by their public headers; runtime sentinel binding from a disposable
  builtin set resolved that build failure. This is a per-minor CPython
  extension, not abi3 or validated free-threaded support. Matching CPython
  algorithm/source identities and the unmodified PSF license are retained.
- Build7's full focused selection passed548 tests. It is an earlier source
  checkpoint, distinct from build8. Build8 with `PYTHONMALLOC=debug` passed75
  native tests, including actual MEM allocator rejection, zero allocation
  attempts during physical seed restoration, original exception identity,
  preparation rejection, repeated body generations, collisions and lifecycle.
  Raw probes, logs, helpers and hashes are archived under
  `.plan20-artifacts/20261029/stage3/fact-native/native-v4-undo-checkpoint/`.
- Actual final default command is recorded verbatim in
  `.plan20-artifacts/20261029/stage3/fact-native/request-default-build8-attempt1.json`:
  CPU24–31, `/tmp/hotpot-plan20-env/bin/python -m pytest -q
  --import-mode=importlib tests`, with isolated JUnit/cache/model-output paths.
  `helpers/run_recorded_v2.py` measured259.05s wall time; pytest reported
  **4432 passed,123 skipped,3 xfailed,49 passed subtests**,10 warnings in257.20s.
  Production source and every checkout native binary were unchanged throughout.
  This source/default result does not substitute for five-version or installed
  wheel verification; those final checks are running separately.
- The final paired-v4 diagnostic retains scalar8/64/256 ratios
  **1.865/1.905/1.877**, batch8/64/256 **2.338/0.756/0.551**, and complete
  write/read64 **3.136**. These include real unresolved regressions. A failed
  driver call accidentally reused the v3 output directory, produced no new
  samples and invalidated its top metadata; its failure is preserved, and
  the previously archived valid v3 remains separately bound. No failed call
  is counted as a passing measurement.
- `stages/stage03/gates/review.md` corrects an execution interpretation:
  the complete71-entry final global comparison is required at Stage12/final
  delivery, while Stage3 requires actual migrated-mechanism correctness,
  source/native/wheel and storage/export/scaling evidence. Neither frozen
  workloads nor efficiency thresholds/weights/exception policy changed. This
  grants no performance waiver; the new engine's own overhead remains visible
  and cannot all be attributed to retained Stage5 eager chemistry. Stage3 is
  still in progress and Stage4 has not started.
- The owner-requested directory standard now includes navigation READMEs for
  every `stage00` through `stage13`; later-stage navigation is not an entry or
  execution record. The current layout audit
  `.plan20-artifacts/20261029/stage3/fact-native/layout-final-attempt1.json`
  passes26 Markdown records,23 JSON records and28 migration identities with
  no errors. Historical migration evidence remains unchanged.
- The frozen five-column float32 benchmark now binds its candidate to the
  existing private table selector under the owner lock. Native export still
  calls the sole current `_native.buffers` path until its Stage10 physical
  move. This only supplies owning-stage fixture/import bindings: task fields,
  dtype, work, manifest, sampling and public readers are unchanged. The full
  public19-column float64 matrix remains a separate measured operation.

### Stage3 final frozen validation and storage evidence

- Executed `PYTHONDONTWRITEBYTECODE=1 /tmp/hotpot-plan20-env/bin/python
  .plan20-artifacts/20261029/stage3/fact-native/helpers/run_matrix_final.py`.
  Python3.10–3.14 each passed1023 source tests with8 external-xTB skips,
  75 actual debug-allocator native tests,941 installed-runtime tests with the
  same8 skips, wheel content audits and installed core smoke. The full affected
  selection adds the new native test, import safety/order and changed-build
  scans.43 source targets and38 installed targets are explicitly frozen;
  architecture/four packaging source scans are not relabeled wheel tests.
- Every matrix command binds exact argv, cwd, environment, interpreter,
  source/native/tests/runner before/after hashes and elapsed time. The whole
  matrix manifest is stable. Copied test bytes match, and all installed
  Hotpot/MCA origins resolve to each isolated pip target outside checkout.
  Current Python/stub/license and each rebuilt fact extension were packaged;
  unchanged existing native members were reused only after matching their B20
  source closure/build recipe/wheel-member identities. This is current
  installed-package evidence, not the Stage10 clean all-native rebuild/SDK gate.
- The actual current sdist was built and audited: current fact CPP, stub and
  unmodified PSF license match exactly, with no prebuilt `.so`. The matrix and
  package identities are in`stages/stage03/validation/evidence.json`; the
  executed validation report is`stages/stage03/validation/report.md`.
- After all task builds/tests stopped, executed
  `helpers/run_stage3_storage_final.py` under the fact-native artifact root,
  using the same interpreter, LD_PRELOAD and single-thread environment as the
  frozen source/default runs. The script's exact commands/hashes are retained
  in`storage-final-attempt1/execution.json`.25 unchanged workload comparisons
  used CPU0–63,3 warmups/10 samples:50 workers,650 total attempts,500 measured,
  no failures/no-output, equal actual work counters, stable source/native,
  and zero comparator environment mismatches. Manifest/policy bytes did not
  change; the candidate binding changes were limited to their owning stage.
- Current/B20 scalar-write8/64/256 ratios are1.833/1.878/1.872; batch
  ratios2.339/0.760/0.563; coordinate write/read0.402/0.122/0.082;
  five-column float32 export0.227/0.033/0.008; native export
  0.398/0.126/0.077. Construction ratios3.585/1.343/0.455 and
  insert/delete7.088/3.880/2.867 remain fully visible. Complete write/read64
  is3.111(CI3.076–3.146). The entire measured Atom-storage family is
  1.037(CI1.035–1.041), not parity. No exception was requested or approved.
- Executed the supplemental`helpers/measure_stage3_marking.py`:12 new-only
  scaling cases,156 attempts/120 measured.20 sparse UID marks in one actual
  transaction cost48.1/49.9/50.6/51.0us at N8/64/256/4096. Dense marking and
  full19-column float64 exports retain their N-dependent costs and buffers.
  This is actual mechanism evidence, not a fabricated old/new counterpart.
- Executed`helpers/run_retained_final.py` with six fresh-process memory
  controls. Full19-column numeric projections match B20 byte-for-byte at
  N8/256/4096. GC-controlled Python/NumPy-tracked retained deltas are
  B20=7667/129203/2095348B and current=17187/144419/2175556B. Current retained
  increases remain open; lower observed process peak-RSS does not erase them.
  All memory/scaling source/native before-after identities are stable.
- `stages/stage03/performance/{report.md,evidence.json}` preserve all25
  confidence intervals, individual regressions, family completeness, original
  samples, GC/release observations and measurement limits. Final71-entry
  scientific/efficiency/end-to-end/memory acceptance stays open at its owning
  later stages. No inferred improvement from future eager/result changes is
  counted now. The ratchet passes281 remaining entries, with no new debt;
  the document audit now passes28 Markdown/25 JSON and all28 migration records.

### Stage3 closure

- The owning Stage3 gates are complete on the frozen build8 production bytes:
  the one factual owner/writer, connected factual importers and outer commits,
  raw value/source/UID/permission/lifecycle/buffer/dirty contracts, actual
  allocator-safe rollback, complete default, affected five-Python source and
  current installed-package checks, architecture ratchet and actual selected
  storage/export/marking/memory measurements. The final stage report is
  `stages/stage03/report.md`; the verification map now reflects these executed
  gates rather than historical pending checkpoint labels.
- This closes Stage3, not final global efficiency or scientific delivery.
  The recorded regressions and retained-memory increases remain open under
  the unchanged Stage12/final policy. Stage10 clean all-component native
  rebuild/SDK and the assigned later science/result/frame owners are not
  claimed complete. Stage4 requires a new complete main/companion reread and
  current-state assessment before its implementation starts.
- The next coherent implementation commit begins the first of three commits
  since the successful`ba662e3` push. No cadence-only commit is created.

## Stage4 entry — committed predecessor and scope assessment

- Stage3 was committed as`bdc6a8066095e81c415e1152b0f3e4717583ac83` on
  `implement/plan20-fact-rule-architecture`. The worktree was clean afterwards;
  the branch is one coherent commit ahead of the last successful push.
  `.plan20-artifacts/20261029/stage3/fact-native/source-commit-binding.json`
  verifies that all validated production-manifest bytes match the committed
  worktree; the actual production Git tree is
  `3853805935e99521727956fd5968db42317ca3f4`. No unchanged test was rerun solely
  because commit metadata changed.
- Stage4 begins with a fresh full main-plan/four-companion read and actual
  source/predecessor-gate assessment by the assigned Sol/max implementation
  owner. Its entry/review records belong in`stages/stage04/`. Production stays
  unchanged until this entry and bounded connection design are recorded.
  An independent read-only Sol/low lookup maps only current constraint owners,
  caches and consumers; it does not design, edit or run verification.
- Public spelling, staged semantic/physical boundaries and the source of
  stage gates remain unchanged. Stage5 retains Ring write-back/cache/eager
  chemistry/constructor removal; Stage7 retains Crystal isolation and
  result/frame ownership; Stage10 retains physical class/native/build moves.
  No Stage13 protocol is implemented early. The final performance and memory
  regressions recorded in Stage3 remain open, without an invented waiver.

### Stage4 source lookup — constraints and pair metadata

- Read-only lookup found no implemented`ConstraintSet`. Current Atom constraints
  live in the three table projection columns, Bond constraints in the Bond
  numeric slot, and Angle/Torsion constraints in their view dictionaries and
  cached lists. Stage3's factual record/rollback transports these values; the
  reserved owner class is a Stage4 implementation task, not an already created
  type. The current `_commit_constraints` has no production caller.
- `_refresh_topology_index`, factual capture/install, component/copy/merge and
  pickle preserve ordered constraint rows. `SourceIdentity` reads current
  constraint facts without constructing the cache. FF working-copy code
  captures this domain and restores the current caches. Native FF has no
  existing constraint field/execution path; this lookup does not authorize
  adding new scientific constraint behavior.
- `AtomPairs` also carries Wiberg/pair metadata and has independent copy/FF
  restoration consumers. Removing mathematical caches must preserve those
  actual facts. Exact source/test paths were delivered to the Stage4 Sol/max
  owner before its connection design; no lookup edits or tests were executed.

### Stage4 entry recorded and parallel extraction connected

- The Sol/max owner completed the full3498-line current main/four-companion
  read, recovering truncated outputs with smaller complete ranges. Actual
  ranges, document/source/native identities, worktree/environment, completed
  Stage3 gates and bounded scope are in`stages/stage04/entry_review.json`.
  The connected design is`stages/stage04/review.md`. No main-plan requirement,
  workload or public decision changed.
- Clarification of the earlier entry shorthand: remaining carrier/view class
  relocation belongs to Stage9; Stage10 owns native/source/build-path moves.
  Stage4 moves the actual mathematical calculator implementations and the
  specified structure adapter under their semantic owners. Angle/Torsion
  constraints move now; coordinate/Bond constraints remain at Stage7B.
- An explicitly assigned Sol/medium worker moved11 graph/geometry/descriptor/
  adapter bodies, deleted their superseded bodies and updated local README
  imports. The old graph/geometry packages remain native homes, with no
  mathematical compatibility re-export tree. Geometry exports split into
  38 mathematics and32 adapter entries; graph into6 mathematics and4 chemical
  descriptor entries. Current native module names,38 native source/stub files
  and16 matching existing binary identities stayed unchanged.
- Actual mechanical check command was
  `/tmp/hotpot-plan20-env/bin/python /tmp/hotpot-stage4-kernel-check.py`:
  exit0,19 modules compiled,11 bodies matched the entry AST after imports were
  excluded. Original helper/results are copied under
  `.plan20-artifacts/20261029/stage4/kernel-handoff/`. This is a static/code
  equivalence check, not runtime/default/native gate completion.
- Parent connected62 inventoried consumer files, including13 geometry-alias
  consumers and one dynamic monkeypatch consumer found after the initial48
  list.57 files changed; the two local ComplexFormer graph imports and actual
  native imports were left at their existing owners. Mathematical calls and
  carrier-adapter calls now address their respective real modules, including
  type annotations and string-based monkeypatch targets. Existing70-name
  coverage, parameter signatures, numeric/golden assertions and missing-native
  fences retain their behavioral purpose at the canonical paths.
- Executed
  `python .plan20-artifacts/20261029/stage4/helpers/check_consumer_rewire.py`:
  all62 owned files parse, and production AST bodies are unchanged after
  import/owner-qualifier normalization. Exact files/hashes and limits are in
  `consumer_rewire_check.json`. No runtime test was run while Core/constraint/
  view/serialization code remained under concurrent edits.
- The Sol/max owner retains the finite private record change needed to preserve
  supplied pair Wiberg observations through copy/subset/merge/current pickle.
  Its default-empty row/value field is distinct from generated mathematical
  pair defaults, preserves explicit zero, and changes no public DTO, N01 or
  seven-field N05. Precise schema/owner and connected tests belong to the
  Stage4 review and implementation, not a generic result/protocol expansion.


### Stage directory standard follow-up and historical navigation

- Reused the independent Sol/low source-lookup agent for the owner-requested
  bounded layout audit; the initial mechanical migration was already committed
  as`ba662e3`, so no duplicate move or historical evidence rewrite was made.
  The catalog defines finite Markdown meanings and stage00–13 navigation.
- The first follow-up audit found four broken source links in the historical
  data-object audit after Stage4 mathematical files moved. The parent pinned
  all90 source links in that audit to its actual original
  `d9eb80e239f24a694a8cb23fd412eb6baf030cbb` source, preserving the original
  line references and normative content. The appendix states that historical
  identity; it does not point old line numbers at a different implementation.
- The layout auditor now checks every file asset against its owning README,
  the complete14-stage range and every stage's catalog link. It retains
  Markdown type checks, JSON parsing, current local links and archival byte
  checks. The catalog now acknowledges the actual Stage4 entry/design records.
- Executed`python plan/20_fact_rule_architecture/audit_stage_records.py
  --check-migration-bytes --output
  plan/20_fact_rule_architecture/stages/layout_followup_evidence.json`:
  exit0/passed;14 stage directories,30 Markdown,27 JSON,0 other assets,
  57 indexed files and1282 external text files checked. All19 original JSON
  snapshots,9 historical Markdown command/identity records and the existing
  implementation-log prefix passed preservation checks. The earlier failed
  audit is retained in the follow-up evidence. No Stage4 execution gate is
  completed by this navigation audit.

- Evidence clarification: the follow-up JSON retains a factual transcription
  of the first failed audit and its four observed broken links. That command's
  original output was visible in the tool record only; no original artifact
  file was saved. Its artifact path/digest are explicitly null. The final
  passing audit above has an actual retained JSON record.


### Stage4 freeze-v1 executed gates and retained failures

- After the owner froze production/tests/audit/allowlist/budget, ran the actual
  complete default command through `stage4/validation-helpers/run_recorded.py`:
  `taskset -c 24-31 /tmp/hotpot-plan20-env/bin/python -m pytest -q
  --import-mode=importlib tests`, with exact environment/JUnit/temp arguments
  retained in`stage4/default-attempt1/command.json` under the dated artifact
  root. It passed4563 tests,123 skips,3 xfails and49 subtests, with10 warnings;
  wall143.603s. All recorded production/native/tests/runner bytes stayed stable.
- The independent Stage4 native command selected all7 mapped C++ source-API
  tests with`--run-high-compute`; all7 passed, no skips/failures/errors,
  wall34.294s. Their native source/recipe closure remains unchanged from
  Stage3. This is actual standalone compilation/execution, not Stage10's
  separate clean all-component build/ABI/SDK gate.
- Actual`setup.py ... sdist` and content auditing passed542 current
  Python/stub/native-source/license/documentation members, rejected prebuilt
  shared libraries and confirmed11 superseded Python modules absent. This
  freeze-v1 source archive is retained under`sdist-build-attempt1/` and is
  historical once the two read-interface fixes below change production bytes.
- The supported-version driver's first two preflight attempts failed before
  tests/builds started: first a root-relative/native-wheel path namespace
  mismatch, then its incorrect assumption that Stage3 root and installed OB
  builds must have identical bytes. Both runner snapshots and actual failure
  records are retained in`matrix-attempt{1,2}-preflight-failed/`.
  The corrected provenance independently binds unchanged root binaries and
  previously executed installed wheel/native identities, including the four
  different OB builds, rather than conflating those builds.
- The true five-version freeze-v1 run is retained in`matrix-final/` with its
  partial commands/logs/XML. Python3.13/3.14 each exposed8 failures before
  rollback execution: the new allocator calibration observed no allocations
  and attempted`max([])`. Python3.11 installed testing exposed12 fixture errors
  because its copied test support omitted`plan/17_rdkit_wrapper/run_r0_benchmark.py`.
  These are actual failed checks, not passes or an excluded test population.
- A final owner read-interface audit also found that bound sequence`bonds`
  must retain its independent-list return and the read-only pair mapping's
  `copy()` read helper must remain available. On that concrete evidence,
  reopened freeze-v1 and stopped the running matrix with unified-exec Ctrl-C
  (actual exit130).`matrix-final/cancellation.json` records that action and
  the earlier unsuccessful cross-PID-namespace process search, which sent no
  signal. The Sol/max owner owns the read-container and allocator-probe fixes;
  the parent owns missing installed test-support assembly. Final source,
  affected default/matrix/current-package evidence will use a new freeze and
  new artifact directories. Unchanged standalone C++ evidence is not discarded
  merely because unrelated Python view bytes change.


### Stage4 freeze-v2/v3 and default-gate prerequisite defect

- Freeze-v2 fixed the two existing read contracts and the actual new canonical
  pair-owner scanner coverage. Python3.11/3.13/3.14 local80-case selections,
  both3.13/3.14 debug8-case probes and the95-case architecture selection
  passed. The probes now actually observe both MEM/OBJ domains:3.11 dict
  buckets use OBJ,3.13/3.14 use MEM. No failure was skipped or inferred away.
- The complete five-version v2 source/debug/current-wheel selections passed,
  including actual outside-checkout runtime and smoke. Their unchanged source
  identities are retained in`stage4/matrix-final-v2/`; current final evidence
  supersedes their production checkpoint after the next concrete defect fix.
- The v2 complete default run failed1 DS case with4569 passes,123 skips,
  3 xfails,49 subtests. Its unchanged HEAD DS implementation and unchanged
  original test missed O-to-N same-length source rewriting when filesystem
  metadata stayed equal. An actual100-rewrite probe on this filesystem found
  98 full stat-signature collisions; the raw observations and code identities
  are in`default-attempt2/filesystem-timestamp-probe.json`.
- Preserved that failure and added a deterministic unchanged-stat/changed-byte
  case. Before the fix its exact recorded node failed; after the minimal fix
  all409 DS tests passed with30 declared skips. Cache final release now uses
  its existing stable-digest helper to reprove current source content once
  per cached present asset. Shared snapshot/parse work remains shared; error
  types and cleanup behavior remain the existing ones. No new public contract,
  architecture, rule, model route or generic protocol was introduced.
- The new collision case is an additional parametrized test. For final frozen
  common-technical benchmarking, the original one-case node binds specifically
  to`test_shared_asset_cache_rejects_source_change_before_reuse[False]`; the
  new`[True]` case remains additional work. This mapping is recorded before
  measuring that family; original node membership/weights are unchanged, and
  the focused Stage4 graph/geometry/Crystal subset does not execute this node.
- Recorded global`stage4/global_freeze_v3.json` after the DS prerequisite fix,
  preserving the Stage4 owner's v2 bytes. The complete default rerun passed
  4571 tests,123 skips,3 xfails,49 subtests and10 warnings in147.47s pytest time
  (149.456s wall), with all source/native/tests/runner hashes stable. Current
  source sdist building and542-member content auditing also passed.
- Full affected chemistry plus all DS contracts passed source checks on all
  Python3.10–3.14, along with debug allocator, current Python wheel assembly,
  native provenance, wheel-content audits and isolated installs. All five
  installed attempts then correctly failed the same DS documentation fixture
  because test support omitted its source`pyproject.toml` input. Those actual
  failed logs/XML are retained. The recovery driver copies and hashes that
  already frozen build-metadata input and reruns the complete installed
  selection and smoke without changing any production, test or wheel bytes;
  its results remain pending until the actual commands complete.


### Stage4 verified extraction checkpoint and measured regressions

- Completed`validation-helpers/recover_installed_support.py`: each
  Python3.10–3.14 reran the complete installed-runtime selection after its
  missing frozen`pyproject.toml` support input was supplied. Each passed2143
  tests with42 skips; all five actual installed core smokes passed. Production,
  native, test, wheel and matrix-runner bytes stayed unchanged. Earlier
  installed failures remain intact;`matrix-final-v3/matrix-acceptance.json`
  binds the successful recovery rather than rewriting the old failed summary.
- Final owning-stage technical evidence is in`stages/stage04/validation/`:
  default4571 passed/123 skipped/3 xfailed/49 subtests; each supported Python
  passes2273 source/162 debug/2143 installed. The42 source/installed skips are
  41 requiring`--run-external` and1 requiring`--run-scientific-boundary`;
  those tiers are not claimed executed. Seven standalone C++ builds/executables
  and the current542-member source-distribution audit have their own records.
- After all internal task tests/builds stopped, executed
  `PYTHONDONTWRITEBYTECODE=1 /tmp/hotpot-plan20-env/bin/python
  .plan20-artifacts/20261029/stage4/validation-helpers/run_recorded.py
  .plan20-artifacts/20261029/stage4/request-performance-wrapper-attempt1.json`.
  The actual interleaved B20/candidate driver ran all3 frozen workloads with
  CPU0–63 and3 warmups/10 samples:6 workers,78 attempts,60 measured, no
  failures/no-output, equal actual work counters, stable production/native
  bytes and zero comparator environment mismatches. Manifest/policy bytes
  remain unchanged; the host was shared and is not claimed exclusive.
- Current/B20 graph/search/view reads are4.05299 (95% CI4.04018–4.08116),
  geometry screening0.99693(CI0.99035–1.00588), Crystal matrix conversion
  1.10286(CI1.09772–1.11215). The complete two-entry graph/geometry family is
  2.01011(CI2.00284–2.02241). These are actual regressions/parity, not a
  performance approval; lower observed process RSS does not erase them.
  All original samples/limits are in`stages/stage04/performance/`.
- The Sol/max owner has a read-only bounded task to identify minimal local
  view-creation/Crystal-dispatch optimizations. No production edit is yet
  authorized for that next partition; existing stale/member/constraint/undo
  contracts, no-carrier-cache rule and frozen semantic work must remain.
  No Stage5 eager chemistry, future experimental protocol or workload waiver
  is used to explain away the measured4×cost.
- `stages/stage04/report.md` is explicitly an in-progress extraction checkpoint,
  and its README indexes all current records. Ran the actual layout audit
  with`--check-migration-bytes --output
  .plan20-artifacts/20261029/stage4/layout-checkpoint.json`: passed34 Markdown,
  32 JSON,66 indexed files, all14 stages and original historical identities.
  `git diff --check` passes. The next coherent extraction commit will be the
  second commit since the successful`ba662e3` push; no cadence-only commit
  is created and Stage4 is not declared complete at this checkpoint.


### Stage4 bounded optimization and frozen follow-up — 2026-10-29

- Continued the same Stage4 entry at`0bc19f8`; no new stage, scientific route,
  protocol experiment or architectural revision was entered. The Sol/max
  implementation owner confirmed the unchanged main-plan digest and preserved
  the scope-audit's link-only appendix change separately. The six dirty
  production/test files were its known exclusive work; no unknown user edit
  or competing writer was found. All owner writes stopped before final gates.
- The bounded implementation captures source/UID/geometry once under the
  existing sequence-query lock and creates the same fresh views with the same
  five fields/three byte buffers. One existing pair slice expression now has
  one private helper. The complete Bond lookup preserves first active, then
  hidden-metal, then hidden-covalent selection and missing`None`. Crystal's
  existing private staticmethod binds the sole unchanged mathematical function
  directly, retaining subclass dispatch. No carrier cache, extra index, public
  signature, native source or scientific algorithm changed.
- The owner's focused second attempt passed320 tests in16.26s. Its first
  attempt's one architecture import failure and319 passes remain preserved;
  only the new tests' direct-Core imports required correction. The actual
  profiling commands and unchanged-work counts live in
  `stages/stage04/performance/profiling/evidence.json`;840 geometry snapshots
  became40, with840 fresh views retained. These diagnostics are not formal
  performance acceptance.
- Executed the actual recorded wrappers with
  `/tmp/hotpot-plan20-env/bin/python
  .plan20-artifacts/20261029/stage4/validation-helpers-opt/run_recorded.py`
  and requests`request-default-attempt4.json`,`request-matrix-v4.json`, and
  `request-sdist-optimization-attempt1.json`. The complete default command
  remains`taskset -c 24-31 /tmp/hotpot-plan20-env/bin/python -m pytest -q
  --import-mode=importlib tests` with exact XML/cache/temp arguments recorded:
  4585 passed,123 skipped,3 xfailed,49 subtests,10 warnings;146.92s pytest,
  148.877s wall. Python3.10–3.14 each passed2287 source,175 debug allocator,
  2157 installed tests and42 source/installed skips. All five current-wheel
  audits, outside-checkout smokes and542-member current sdist audit passed.
  Frozen production/native/tests/runner bytes stayed stable. The unchanged
  seven standalone native source/API results retain their exact provenance;
  Stage10's clean native rebuild is not claimed here.
- The common technical benchmark runner now binds exactly the original DS
  node to`[False]` only in candidate mode. Four baseline/candidate and
  collection/execution command checks preserve count, already qualified and
  unrelated targets; the original function-body AST matches B20 after the
  sole new collision conditional is removed. New`[True]` work stays outside
  the frozen common set. Full common-suite timing is still pending. The first
  artifact-only helper had an incorrect argv-offset assertion; its failed
  record remains. An sdist request also stopped before build on an already
  used output directory; the passing fresh directory did not overwrite it.
- Executed the same recorded wrapper with
  `request-performance-wrapper-attempt2.json` only after all technical work
  stopped. Original three workloads/inputs,CPU0–63,one BLAS/OpenMP thread,
  three warmups/ten samples and interleaved phase ordering remain unchanged.
  Six workers/78 attempts/60 measured samples complete, no failed/no-output,
  equal work counters,stable source/native and zero environment mismatches;
  host shared, not exclusive. Graph/view current/B20 is2.05414
  (95%CI2.03997–2.06512), screening1.00568(CI0.99899–1.01255), Crystal0.99424
  (CI0.99093–0.99957). Complete graph family1.43729(CI1.43035–1.44402)
  remains a regression. Original4.053/1.103 results remain visible, no
  exception was requested/approved, and lower process RSS does not close
  global memory/performance acceptance. Exact artifacts and hashes are appended
  as`optimization_followup` in the owning validation/performance evidence.
- Helper v2 was acknowledged/transmitted. The unique inline-delta remote
  review returned no concrete correctness blocker but executed no tests;
  source/test hashes link the actual local follow-up above. That task was
  already running on its existing5.6-sol/xhigh thread when the latest model
  correction arrived, and the actual model is recorded. Later explicit6.1-sol
  dispatch did not change that existing thread; the uniquely authorized6.1
  child failed deployment with404 before analysis. No failed task is called
  completed analysis. The bounded gate-interpretation task was uniquely moved
  to local6.1-sol/high; no duplicate remote/local analysis ran. No project
  synchronization, MCP installation or permission copying occurred.

### Stage4 owning-stage closure and unchanged final acceptance

- Completed the bounded work and its frozen follow-up before changing the
  current in-progress status. The actual Stage4 gate requires focused
  comparison; numerical geometry/Crystal parity is a separate correctness
  item. Section12.2 explicitly blocks`performance acceptance` on convincing
  regressions, and Stage12 owns the complete Sections11–12 matrix. This is
  the same scope distinction already recorded at Stage3 closure, not a new
  gate, policy amendment or efficiency waiver.
- The unique local6.1-sol/high analysis read the packet and actual main/status
  documents and reported those exact controlling phrases. Root checked the
  key quotes and actual frozen execution evidence; the analysis itself ran
  no tests and supplied no approval. The formal closure, evidence map and
  unresolved results are in`stages/stage04/gates/review.md`. Main-plan edits
  only refresh its stale status header; objectives/stages/gates are unchanged.
- Explicitly closed Stage4's owning implementation/technical/current-package
  and valid-comparison obligations in its report, README and verification map.
  The2.054×graph/view and1.437×complete graph-family regressions, earlier
  Stage3 costs, scientific/all71-workload/GPU/end-to-end/memory obligations
  continue to block final main-scope acceptance. No claim says these will
  disappear automatically in later stages. No Stage13 experiment is entered.
- Stage5 entry is next, only after the coherent Git checkpoint; it must reread
  the full current main plan/four companions and assess predecessor/source/
  worktree/environment before design or implementation. The optimization
  checkpoint is the third commit since the successful`ba662e3` push and will
  be pushed under the owner's established cadence. Actual commit/push identity
  will be captured by the next stage-entry record; no cadence-only commit is
  created for that bookkeeping.
- Executed`python plan/20_fact_rule_architecture/audit_stage_records.py
  --check-migration-bytes --output
  .plan20-artifacts/20261029/stage4/layout-closure.json`:passed36 Markdown,
  33 JSON,69 indexed assets,all14 stages and historical identity checks.
  `git diff --check` passes. These navigation checks supply no performance
  or scientific approval.


### Stage5 entry, independent partitions and concrete read decision

- Entered after the actual`ba5fa6d` Stage4 commit and successful third-commit
  push to`origin/implement/plan20-fact-rule-architecture`; post-push worktree
  was clean and commits since push reset to0. The Sol/max owner completed
  the full current2448-line main and all four normative companions, recovered
  truncated aggregate reads, and recorded effective ranges/digests, actual
  project/environment, predecessor closure and bounded scope in
  `stages/stage05/entry_review.json`. This entry is actual reading/assessment,
  not production or validation completion. Only status headers changed later
  to show Stage5 active; no normative route/gate changed.
- Sol/max owns the connected chemistry/representation/readonly Ring, finite
  dependency/commit/snapshot/native-ingress implementation and real consumers.
  A separately assigned Sol/medium worker completed the bounded elemental
  reference/rule extraction and froze its files. The actual10,190 leaf
  type/value comparisons and18 focused tests are recorded in
  `stages/stage05/elements/evidence.json`; concurrent stage work means these
  are preparation evidence, not a whole-project source freeze or Stage5 gate.
- Readonly property enforcement must also reach the sole compiled fact writer:
  the owning implementer identified its direct field ingress bypass. The
  bounded fact-native permission/schema connection therefore requires five
  fresh CPython builds/debug/current-wheel evidence. Root prepared requests
  and an actual-build index checker but has not executed final builds during
  source edits. Independent graph/geometry/rd/ob dependencies retain separate
  unchanged-closure checks; old fact-native binaries cannot stand in for the
  changed component.
- At the concrete implicit-H reader step, no prior approval fixes its unknown
  scalar return. The Sol/max owner wrote an exact pending`int | None` proposal
  in`stages/stage05/contracts/decisions.md`; A2's numeric-snapshot NaN decision
  does not approve this new scalar return. Root submitted one async exact
  choice to the owner, explaining its documented authorization boundary.
  This read/export branch waits for an explicit answer; elapsed time is not
  approval. All independent authorized implementation continues.
- Root located the actual archived B20 O0 report/source identity and captured
  seven existing planar aromatic/Kekule cases against frozen`e6082f5`, with
  7 completed,0 execution errors and stable B20 source/native. The retained
  second producer adds exact argv/cwd/environment/dependency/input provenance;
  its input canonical JSON hash is
  `a781ee50b4c4bf52d8b111235b94bdd69c5924481a0b52f33746c19a4d879d2d`.
  These bounded state observations are not a truth gate, current Stage5
  evidence or the full187-source O0 corpus. Candidate execution will use the
  same frozen inputs/producer after owner handoff.


### Stage5 recovery and metadata reconciliation — 2026-10-30

- Recovered the existing root thread, actual CODEX_THREAD_ID and SESSION_ID
  both`01a18583-f506-71f2-81c9-a66ca32e3561`; no thread-ID change was detected.
  The existing communication subagent retains real UUID
  `01a186d6-d3e5-7df0-911b-449c54e0f92e` and OTP704283. The attempted
  SESSION_ROUTE_UPDATE using the required readonly queue flag failed before
  delivery:rc2,`codex queue does not support --sandbox`. There is no message ID
  or successful delivery claim. Reply/receipt/unique ledger are maintained in
  `/tmp/codex-stage4-bridge-01a18583/`; existing-route compatibility is being
  checked without dropping the readonly contract or recreating a bridge.
- The original Sol/max implementation owner confirmed all71 recorded source/
  test identities and current cp311 fact extension unchanged, and froze its
  production/tests/contracts records for an immutable helper snapshot.
  The accepted known-input checkpoint is477 passed,5 exact pending-scalar
  deselections and3 strict xfails; the seven private B20-bound observations
  have zero compared differences. These completed checks were inspected,
  not rerun. Undefined`_implicit_hydrogen_read` and unavailable scalar/export
  handling remain an explicitly non-distributable pending public decision,
  not an approved fallback or Stage5 completion.
- Root reconciled the exact12 retired Ring-cache allowlist entries:268→256,
  no added entry. Two actual frozen snapshots and five retired protocols
  change the class census779→776, dataclasses351→353, protocols13→8 and
  TypedDicts2→2. The exact original inventory marker method was imported;
  a first local census that incorrectly extended subscript marker handling
  is retained separately and superseded. No retired reservation was credited
  as an actual class removal. The private trailing`bond_aromatic` field is
  counted as a schema extension. See`stages/stage05/bookkeeping/`.
- Changed-metadata checks actually ran through the frozen recorder:
  `python plan/20_fact_rule_architecture/audit_architecture.py --base-ref HEAD
  --output .plan20-artifacts/20261030/stage5/bookkeeping-checks-architecture/report.json`
  passes890 files/256 remaining entries, no ratchet/layer/carrier findings.
  `python plan/20_fact_rule_architecture/audit_stage_records.py
  --check-migration-bytes --output
  .plan20-artifacts/20261030/stage5/bookkeeping-checks-layout/report.json`
  passes42 Markdown,39 JSON,1 other asset and82 indexed entries/all14 stages;
  all original migration and log-prefix identities pass. Actual argv includes
  the absolute output paths; exact commands/interpreter/input identities and
  before/after hashes are retained in each`command.json`. Production/test/
  native bytes remained equal. These checks cover metadata only, not the
  remaining unfiltered source/default/five-ABI/current-wheel/science/performance
  gates. No new Git commit or push occurred; cadence count remains0.

- The unique route diagnosis`p20-bridge-route-704283-04` additionally observed
  `ConnectionRefusedError(111)` at the existing socket. Global sandbox-option
  parsing alone is not enforcement evidence. No Stage5 remote task or session
  notification was delivered, no transport/configuration/contract was changed,
  and no duplicate remote review was started. The fixed43,679,523-byte archive
  is retained with source/support manifests under
  `.plan20-artifacts/20261030/stage5/remote-checkpoint01/`; archiveSHA
  `d7433533eb891364900c6666e9138d55fee8d45ece73c5540b00484ca857fa0c`,
  source-treeSHA`11663c53dca93b803c8d77445243aad8ab89e8bdf3644a463afdd1220662b697`.
  All71 owner identities and before/after repository bytes matched during
  packaging. This is snapshot preparation, not synchronization or testing.
  A single local readonly Sol/high task owns a bounded independent correctness
  fallback; it changes no files/runs no tests. The pending exact H return
  decision and all final gates remain open.

- The unique local static review completed with three actionable code-derived
  findings, not executed failures: neighbour-element H invalidation, exact
  selected Bond coverage at aromatic commit, and Ring export's representation
  source check. Root inspected the cited boundaries only and released the
  original owner for`p20-s5-known-corrections-01` to record reproducers before
  minimal fixes/focused regressions. Real ligand/partial-component selection
  must determine the coverage fix. No prior full suite/scientific run is
  duplicated; no new scalar-H approval or scientific truth claim is inferred.
  The pre-correction archive remains immutable and unsynchronized.

- Verified the actual owning test definitions and mapped the formerly planned
  `tests/test_cheminfo/structure/test_representation_commit.py` target to
  `test_stage5_bonding.py`, retaining the original planned path and explicit
  immutable/rollback/source/conflict nodes in the verification map. The actual
  file is included as an affected target. The prepared matrix driver reads
  that map dynamically; it was inspected only, not executed. This is target
  reconciliation, not a new suite-pass claim.


### Stage5 owner clarification and restored communication — 2026-10-30

- The owner's answer requires explicit/source H declarations to retain their
  supplied counts and unspecified H to receive valence-rule defaults. It does
  not mean explicit-input-only acceptance, and the old optional-scalar proposal
  is not silently marked approved. A separate bounded GPT-6.1 Sol/max task
  `p20-s5-h-default-design-01` owns scientific default design, primary-source
  verification and the precise contract record. The original implementation
  owner continues the three recorded structural corrections; scientific
  H/valence algorithm changes await that design. No model-limit escalation or
  duplicate broad review was requested.
- The user reported the existing bidirectional bridge restored. The existing
  communication child sent the requested communication-only `RECONNECT_ACK`
  once for `reconnect-20261030-01`; actual returncode0 and message ID
  `01a18ae4-fff5-7303-a393-c9c6439dae54` were reported, then the helper confirmed
  receipt. Main/communication thread IDs remain unchanged. The earlier failed
  delivery record remains historical evidence. This ACK's explicitly authorized
  no-sandbox-flag syntax does not alter substantive review enforcement. No
  snapshot synchronization, project/config change or remote Stage5 validation
  occurred as part of reconnection.


- The original owner completed `p20-s5-known-corrections-01`. The exact
  pre-fix selection actually produced7 failures; changed-boundary validation
  produced82 passed/1 pending-H deselection. The final two neighbour regressions
  then passed against the active rule after removing a fixed N4 expectation.
  Seven already-bound assignment cases completed with no compared differences.
  All attempts, including the metadata helper's original failure, remain in
  `stages/stage05/corrections/evidence.json`. Root compared the pre-fix snapshot
  with the four production boundaries, inspected the malformed/positive-scope
  and declaration/one-time regressions, and checked command/log identities.
  No test was rerun by root and no concrete correction blocker was found.
  Native/scientific-rule bytes remained unchanged. The owner froze its source
  and tests for the separate scientific/default handoff; Stage5 gates stay open.
- Restored-route preflight `p20-bridge-readonly-preflight-704283-06` used the
  actual0.166 schema's explicit per-turn `readOnly` policy over the existing
  socket. The helper audited exact turn
  `01a18aec-ba1e-7b21-b498-adfde0a18a24`: normalized `read-only`, zero filesystem
  write entries, unchanged gpt-5.6-sol/xhigh, zero tool/delegation activity and
  the expected control ACK. This establishes that exact turn's enforcement;
  every later substantive turn still requires explicit policy and its own
  audit. Queue/global flag parsing is not used as evidence. The unique next
  task is environment/destination-metadata inventory only, without source
  review or test/build execution. No current project source was synchronized.


### Stage5 default-H design and validation preparation — 2026-10-30

- The bounded Sol/max design task completed its full normative reread and
  scientific/default convention, with a seven-line main-plan amendment limited
  to the owner's requested H-default correction. Its53 local observations,
  29 RDKit references and12 aromatic-charge references are source-labelled
  design evidence, not production/default/native acceptance. Exact primary
  source/probe identities, the first import-path failure and actual document
  checks are in `stages/stage05/hydrogen-defaults/evidence.json`. The integer
  property and supplied source/model encodings are retained; imported-count
  normalization remains a separate unapproved proposal. Root assigned the
  original production owner the one-rule/default/new-input-dependency migration.
  The genuinely missing-input public read remains a concrete bounded review;
  ordinary known-element defaults need no optional-scalar decision.
- Completed remote environment task07 and narrow Open Babel task08 were
  audited as actual read-only turns with zero filesystem write entries and
  unchanged gpt-5.6-sol/xhigh. The available remote interpreter is3.10.13;
  direct `find_spec`/import confirms Open Babel absent. No compatible3.11–3.14
  runtime/project destination exists there. No sync/install/build/test was
  performed. Existing local compatible environments therefore own the required
  runtime gates; helper work stays uniquely scoped readonly analysis.
- Historical view-profile task09/revision2 completed with inline-only evidence.
  Root checked the four profile source hashes against the complete committed
  Stage4 production delta; helper corrected the source-provenance limitation.
  The batching/tuple-construction suggestions are future hypotheses, without
  new measurements or a Stage5/final performance-pass claim. Exact task/hash/
  turn identities and the narrow root check are in
  `stages/stage05/performance-analysis/evidence.json`.
- Prepared optional base-Git/model support archives are retained under
  `.plan20-artifacts/20261030/stage5/remote-support/` with their actual evidence
  manifests. They were not synchronized or used for a remote validation claim.
  The older incomplete pre-correction snapshot remains immutable.
- Unique ai4 static runner task `p20-s5-runner-review-704283-10` returned
  BLOCKED for six evidence-binding/failure-propagation issues; it executed no
  scripts or production review. The exact turn
  `01a18b07-c2f5-7dd0-9ef6-966f4d8dbc5f` was independently audited read-only,
  zero tools/delegation/write entries, unchanged model/effort. Root checked
  the reported ABI selection, performance exit and actual earlier wheel
  command provenance, then assigned one local writer to correct only the
  prepared validation scripts. A stable authorized dirty tree is valid input
  when bound to its exact manifest; a clean-worktree prerequisite is not
  introduced. Script corrections/collection/runtime gates remain unexecuted.
  No Git commit or push occurred; cadence count remains0.


- The follow-up Sol/max parent review selected the existing display behaviour
  for genuinely missing H input: scalar/permissive field14 show the historical
  zero placeholder, without committing zero, inventing source/presence, marking
  CURRENT or returning a stale derived count. Strict consumers retain the
  existing unresolved `ValueError` guard. Explicit/source zero and real inferred
  zero remain present/current. This preserves the public integer contract and
  needs no new owner approval; the earlier scalar/projection-error recommendation
  was withdrawn, never attributed to the owner. A2's optional selected-result
  NaN policy remains distinct. Exact amended records and actual document-check
  outcomes are in `hydrogen-defaults/evidence.json`; default/read implementation
  and runtime gates are pending. Independent helper task11 owns inline contract
  review only, with no overlapping source inspection or execution.


- Received `HELPER_PROTOCOL_UPDATE 704283`: the owner explicitly permits
  non-overlapping concurrent readonly helper tasks on distinct reviewer threads,
  still with per-turn explicit sandbox and exact task/hash/turn audits. Every
  envelope carries expected output and its critical wait point. Dependent
  architecture/science/acceptance/commit/stage-gate decisions wait for matching
  results; independent implementation may continue. Root ACKed and relayed the
  protocol through the existing communication child. The update starts no new
  task and changes no model/effort/config/source permissions or rate limits.


- The changed-document layout check retained two failures: the first invocation
  could not write its absent output directory; the next found two concurrently
  created owning H-implementation records missing from the stage index. Root
  created the output parent and indexed those real records. The third exact
  `python .../audit_stage_records.py --check-migration-bytes --output .../attempt3/report.json`
  passes50 Markdown/46 JSON/1 other asset/97 indexed entries, with19 original
  JSON,9 Markdown identity records and the original log prefix preserved.
  Exact argv/cwd/interpreter/elapsed/log identities and original failures are
  under `.plan20-artifacts/20261030/stage5/layout-after-design-review/`.
  This covers changed navigation only; no source/native/test/science acceptance
  is inferred. No production tests were run by root while the writer was active.


- Contract task11 revision2 returned PASS_CONTRACT for the bounded metal/raw0, mandatory effective-input and current-J clarifications. The helper corrected its earlier metal-zero interpretation; the original findings remain retained. Matching task hash0b99720b…, source ba5fa6d…, tree a9803954… and exact turn01a18b2b-4526-7150-905a-e202a7697160 passed the actual readonly audit (context503, range502–518, zero tools/delegation/write entries). Root accepted these clarified contracts only; the original implementation owner continues the required atomicity/source/rollback/idempotence tests. No production acceptance, stage closure, commit or push occurred.
- The bounded validation-runner writer completed eight corrected helpers and nine synthetic stdlib check groups, without real collection/import/build/test/benchmark. Root retained its original bytes/results and uniquely dispatched task10 revision2 to ai4 for static correction review, binding twelve inline evidence files to tree f3f46165… and task hashcce64148…. Final execution waits for the matching review/audit and the production owner's coherent handoff. No live tree was synchronized.

- Added the actual owning `tests/test_cheminfo/structure/test_stage5_hydrogen_defaults.py` to Stage5's explicit affected-contract list while the writer finishes it. The final freeze will collect its actual nodes and hash this selection; no pass or frozen-source claim is implied. Runner revision2 resolved the original six findings but exposed a non-strict-XPASS accounting hole, confirmed by root at the exact outcome branch and assigned to the original runner writer for a narrow correction.

- Runner task10 revision3 closes the single XPASS accounting defect after10 synthetic groups passed. The delta-only PASS_RUNNER_STATIC and matching exact-turn readonly audit bind task07420050…, tree3a1b88a6… and turn01a18b38-355e-7570-9948-161c6d5b0fde (context573/range572–584; zero tools/delegation/write entries). Root checked the changed wasxfail branch and verified all eight helper bytes against the already reviewed revisions. Helpers are statically accepted for later freezing; real collection/build/default/matrix/scientific/performance gates remain unexecuted pending the production handoff.


- After local child restoration, two attempts to resume the old communication child returned `agent thread limit reached` before dispatch. Root reassigned the already idle runner child to communication-only responsibility; reviewed helper bytes remain frozen and it no longer owns code edits. Helper route ACK accepts communication thread01a18b0c-e0c4-7641-aeaa-c34444b3cad9 and retires the former communication owner for future tasks. Root01a18583-f506-71f2-81c9-a66ca32e3561, reviewer01a186e8-89de-7791-8969-98b64ab572c3, socket and OTP704283 are unchanged. No link/config/model/permission/PhaseB change occurred. Task12 uses the original explicit per-turn readonly protocol and exact-turn audit; communication ACKs alone are not review enforcement.
- The H owner's third actual focused attempt reports267 passed, rc0 and unchanged source/tests during execution. Root compared its six changed production files and supplied tests/context with that exact retained snapshot and prepared a unique inline implementation-delta primary review. This does not close the final default/native/matrix/scientific/performance gates; the owner is completing static/environment identity records.

- H implementation task12 returned BLOCKED for zero-residual inferred transport cleanup. Root checked the exact num>0 branch against accepted Section7 and assigned the existing production owner the narrow correction plus Atom/Molecule and exceptional-encoding regressions. Task f21125f9…, tree eba438d6…, exact turn01a18b47-3a6d-7472-9b48-0521b5becbcc has a matching actual readonly audit (context587/range586–664, zero tools/delegation/write entries). The complete finding and audit are retained in hydrogen-source-review; enforcement PASS does not override the content blocker. Subsequent shared-extractor/retained-installer changes and cleanup will receive delta-only review and new execution evidence. Historical267 passes are not attributed to those new bytes. No Stage5 closure, commit, push or live snapshot sync occurred.

- The sole H implementation owner froze its final nine-file handoff after304 focused passes and matching AST/scoped diff checks. The zero-residual defect was reproduced with8 failures/21 passes before repair; focused-attempt5 retains1 failure/303 passes from a test spy that remained installed during the repeat call. Final attempt6 scopes that spy to its intended first pass and passes304 without exclusions. Post267 changes are exactly core.py, operations/chemistry.py, structure/_identity.py and test_stage5_hydrogen_defaults.py; the sole finite rule and native bytes are unchanged. Root prepared unique task12 revision2 hash4798a531…, evidence tree96a0beee… against the actual retained267 snapshot and final304 manifests. Dispatch awaits the existing communication owner's activity slot; no current-source acceptance is inferred.
- The owner requested a dedicated gpt-6.1-sol/medium Git agent for smaller semantic commits and every-third-commit pushes. Root created git_management with exclusive index/commit/push responsibility and root-controlled rollback. Its first task is read-only hunk/message planning; HEAD/index remain frozen through the pending source review. No cadence-only commit, reset, rebase or force-push is authorized. Future intermediate checkpoints may precede full stage closure but must state their actual coverage and unfinished gates.

- H source task12 revision2 returns PASS_IMPLEMENTATION_STATIC with no concrete defects; its matching actual readonly audit passes exact turn01a18b65-51b7-7451-9f5e-41a75338529c, context670/range666–692, zero tools/delegation/write entries and unchanged5.6-sol/xhigh. Duplicate existing audit receipts are deduplicated without another review. Root accepted only the bounded source delta and corrected eligibility/cleanup key point. Broad Stage5 gates remain open. The dedicated Git agent is authorized for the first three small element/snapshot/pure-H rule checkpoints and the every-third-commit push; HEAD will freeze again afterward for source-bound validation. Separate ai4 task14 owns the evidence-scope analysis, without duplicating source or runner review.

- Dedicated Git management completed three natural Stage5 preparation commits: e67c28ed912b37fbe6c49456fef676fa2d7d8964 (elemental references/readers), cac91164f3c054478836024b69024a09095a7e73 (immutable bonding snapshots/pure rules), and0d8c7098b375c8b3c1ab7845fb555f97c57810aa (finite H rule/scientific table). Every-third-commit push succeeded and the agent independently matched remote branch SHA to0d8c7098…. Source/test bytes across its1369-path manifest are unchanged; index matches HEAD. Staged diff/AST checks ran, no tests/builds against those partial commit trees are claimed. Remaining Stage5 work stays in the worktree; count since last successful push is0. HEAD/index now freeze at0d8c7098… for the accepted dirty-source build/matrix handoff.


- Stage5 unique gate-evidence task14 and revision2 have matching actual readonly audits. Root challenged the helper's inference that common coverage tooling was mandatory at this stage; the helper withdrew that point after checking Section11.3 and the Stage5-specific map. No normative gate or waiver changed. Original and corrected results remain in gate-evidence-review; default/matrix/native/package/science/focused performance remain pending, and later full O0/global/GPU gates retain their owners.
- Five sequential forced fact-native ABI builds completed on the 0d8c7098… dirty-source freeze. Python3.13 attempt1 failed before compilation because PYTHONSAFEPATH omitted the helper directory; fresh attempt2 with exact helper PYTHONPATH passed, as did3.10/3.11/3.12/3.14. All original logs/commands remain retained. Task15 requested configuration/dependency/rejection context; root supplied it and ran the exact-current cp311 probe. All14 private derived-field normalizations returned invalid candidates without state writes, and append VALENCE status disagreed with normalized presence. Revision2 reports BLOCKED for these same bounded defects, with matching actual readonly audit turn01a18b91-7324-7632-84ee-888072953639. Root assigned a local writer the two minimal fixes and regressions, retaining one DERIVED_FIELDS declaration and the internal aromatic assignment path. Before any source edit, final-validation/INVALIDATION.json marks that freeze superseded. No full matrix or performance run was started; changed native bytes will be rebuilt across all five ABIs. Stage5 remains open.

- The bounded storage writer completed append normalized-presence status and native derived-field rejection across normalization/prepared/store ingress, retaining internal aromatic complete assignment. Five specified files changed; AST/diff checks passed, with68 parameter cases prepared but not yet collected. Root inspected only the two correction branches and canonical/aromatic distinction, then uniquely dispatched task15 revision3 for delta review. No fresh compiled/runtime acceptance is claimed. Stage5's explicit affected list now includes the owning existing test_atom_fact_update.py so the final verifier requires its new nodes rather than permitting optional skips.
- Git management completed the next three natural documentation commits:2de8d0e7cfadb4ebf761fe190f4937db186b63c3 (accepted H/read contracts), baa5d154c01accd317e3fa0a524e86967a57d947 (owner-directed default-H plan amendment), and93457a7507539f9a731e14b487e599fdc86fd151 (historical scientific design/evidence). Push and independent remote SHA match succeeded; count resets0, index matches HEAD. Source/test bytes remained unchanged across all1369 paths; these document commits claim only actual staged/link/JSON checks, with production gates still open. HEAD/index are held for the next current-source freeze.


### Stage5 first final source-matrix failure and bounded correction — 2026-10-30

- Five current fact-native builds succeeded sequentially on the corrected storage snapshot, including the retained cp313 retry; build index dfa13416… and final matrix freeze b8e028d9… bind their exact bytes. All15 source/debug/installed-target collections passed. The first actual final source matrix then failed on the same74 nodes under Python3.10–3.14: each JUnit2949 tests/74 failures/0 errors/45 skipped entries including xfails. Source/tests/helpers/native remained unchanged. Debug/wheel/installed phases never executed. Original failures and all observations remain under `.plan20-artifacts/20261030/stage5/matrix-final-attempt1/`; no matrix/default/scientific/performance acceptance is inferred.
- Unique ai4 primary analysis task17 hash73514597…, evidence tree769da544…, source93457a75… and exact turn01a18bad-b2bb-7c20-8db9-9724c228e2d3 matched the actual readonly audit (context853/range852–892; zero writes/tools/delegation). Root checked only critical import/member-binding conclusions and supplied the requested representative named rollback diagnostic, without repeating the full74-node analysis. Prior observer task16's matching audit remains retained and is not a content PASS.
- Before edits, root verified the whole active freeze unchanged and wrote `final-validation-attempt2/INVALIDATION.json` sha b57f3a9b…. MCA's top-level import now reaches the same absolute H-readiness owner. Ligand publication uses captured source-member handles from the same existing component query rather than non-unique public IDs; a real multi-component regression covers within-component and cross-component collisions. This adds no persistent owner, public API/schema or scientific rule. The native/math/H-rule bytes remain unchanged.
- The original pair state helper saved status/seeds before a lazy public matrix read. Exact diagnostics reproduce the aggregate failure but show identical raw values/presence/source/status/seeds after the first read and after rollback. The corrected test captures projection first, then matching metadata, and adds full raw numeric bytes. Every original owner, entry, sentinel, allocator-denial and view check is retained. Native fault tests now target real materialized-H status/seed-remove, CURRENT formal-charge-coverage clear and empty-set rehash transitions. Stereo atom clones use complete factual records including sparse atom transport and clear every aligned bond column. Supplied-H and absence/default cases remain separate; no imported-count normalization was introduced.
- Actual recorded local focused commands are `python .plan20-artifacts/20261030/stage5/consumer-binding-corrections/run-focused.py` (32 passed), `python .../contract-test-corrections/run-focused.py` (30 passed/6 failed), its `focused-attempt2/run-focused.py` (6 passed/two retained partial-metal-placement warnings), and `python .../pair-rollback-correction/run-focused.py` (8 passed). Each record binds the real pytest argv/environment/interpreter/current source/native and verifies unchanged bytes. Two historical parameter IDs map to factual atomic-number/bond-order input cases; all74 failures plus two added production regressions are accounted for. The retained six local test-fixture failures were corrected without production algorithm changes. These focused results are not final gates.
- The bounded seven-case observer now pins its exact B20 bytes/ordered unique roster and checks molecular charge, bond sources/opaque metadata and independently declared candidate presence/source arrays. Task18 hash0e5067fe… is prepared for unique inline monitor/observer delta review; a communication followup and local writer spawn returned `agent thread limit reached`. A non-triggering internal message is queued; no dispatch/result/audit or acceptance is invented, and no link/config/model/permission change is used as a workaround. Main continues unrelated implementation and local evidence while dependent acceptance remains open.
- The owner's complete `/goal` attachment was read in full and bound in the new owning evidence. Scope remains Plan20 through Stage13B, with Stage5 open, original scientific/global performance debt and Decisions12/13 preserved. Git remains93457a75…, index unchanged and count since successful push0. Dedicated Git ownership and small semantic checkpoints remain the requested policy; no root reset/rebase/amend/stash/force-push, later-stage entry or completion claim occurred. New execution/review/failed-attempt bindings live in `stages/stage05/source-matrix-corrections/` and are indexed from the owning README and verification map.

- The complete current main plan (lines1–2455) and all four normative companions were read in bounded complete ranges after the latest goal steering, together with approved H/public decisions and predecessor gates. Their exact current digests/ranges, Git/worktree state and bounded next tasks are now recorded in source-matrix-corrections/evidence.json. This continuation adds no new stage entry or public/scientific design. Broadening the corrected failure cases to11 complete containing files passes394 tests/0 skips with4 retained warnings in23.58 seconds; source/tests/helpers/native remained unchanged. Command, environment, interpreter, JUnit and byte identities are retained. Historical pending children were released ACK-only so the existing communicator could resume task18; no duplicate74-node analysis, model/config/bridge change or runtime acceptance was inferred. Stage5 remains open.

- Unique task18 returned BLOCKED with a matching actual readonly audit (turn01a18bf6-8732-7742-a573-7a8cf89fe44b, context898/range894–924). Root accepted the concrete risk that projection-first inside every state capture could repair and hide rollback damage. The monitor now warms once before the transaction and captures raw numeric/presence/source/status/seeds/identity/constraints before public projection; all original8 combinations pass10.28 seconds with stable bytes. A delta-only revision2 is queued after nonoverlapping task19. The same task18 resolved all observer binding findings; no scientific runtime result is inferred. Current architecture ratchet passes891 files/256 debt; layout passes54 Markdown/50 JSON/1 other/105 indexed assets with historical identities retained. Exact commands/results/source bindings are in source-matrix-corrections/evidence.json.

- Task19 static consumer/index verdict PASS and task18 revision2 raw-first monitor verdict PASS have matching exact-turn readonly audits:01a18bfa-4df9-7752-ae17-9bd25a254f93 (context927/range926–946) and01a18bfb-b26d-78d3-81dd-f072897f38b5 (context949/range948–958). Root checked only source-row/handle publication, strict membership, raw-before-lazy capture and the single index constant. The one reviewed artifact-path change selects fact-native-builds-attempt3/build-index.json; old dfa13416… index bytes remain unchanged, all helper ASTs parse, and no new index/build pass is claimed yet. Exact results/audits and before/after helper identities are retained in source-matrix-corrections/evidence.json. Bounded static blockers are resolved; full owning validation remains open.

- Task20 original High monitor finding and Medium missing-inline-Bond-proof are retained with actual readonly audit. Root corrected _complete_state to capture raw facts/status/seeds before lazy public projection, warmed exactly two tests once before the transaction, restored explicit Bond byte/enum assertions and added same-owner checking. Both actual native-fault tests pass2.48 seconds unchanged source. Delta-only task20 revision2 passes with exact actual readonly audit turn01a18c0a-a674-7280-baa0-c015e5d5f83c (context987/range986–996). Other7 accepted migration files are hash-identical; no repeated full review.
- Current bounded scientific commands execute successfully with unchanged recorded source/test/native/input bytes: seven exact-bound B20 cases complete with0 differences/errors;90 finite-H contract cases all match, including6 expected unresolved values. Actual commands, environment, runner/source/input identities and denominators are in stage05/science/evidence.json. This is not full O0, physical truth, fresh five-ABI or final scientific acceptance.
- Dedicated Git owner committed C07 storage/native checkpoint26a92ad…, element evidencece72034… and storage-test evidencea5fc8f52…, pushed the third natural commit and independently matched remote HEAD. Count since push resets0; index/HEAD held at a5fc8f52f0028c28ec666438f7bca4d9951034a4. Selected source bytes were unchanged; the unselected root-owned commit-atomicity test updated concurrently, so no whole-tree stability claim is made for Git execution. All writers have handed off; root now prepares the new exact dirty-source owner/build/index/matrix freeze.


### Stage5 default-tier discovery and owner-goal realignment — 2026-10-30

- Five sequential forced fact-native ABI builds on HEAD a5fc8f52… all succeeded; attempt3 index a9ad1765… and actual commands remain retained. Collection preparation stopped after7 successful commands on a new goal.md dirty path. HEAD and source/test bytes were stable; no completed matrix freeze or runtime/performance acceptance existed. final-validation-attempt3/INVALIDATION.json preserves that ownership failure and later correction boundary; no historical owner/index files were rewritten.
- The user explicitly replaced the active goal with goal.md and integrated Plan12 requirements, then confirmed goal.md as Hotpot's long-term implementation constitution. Root read that file in full and reread the entire current main plan1–2455 and all four normative companions284/177/523/68 lines, correcting truncated read ranges. Exact digests, Git state and bounded next work are recorded in source-matrix-corrections/evidence.json. This remains Stage5; no later entry or gate waiver is inferred.
- Unique ai4 task21r2 is audited PREENTRY_INVENTORY_ONLY for future stereo extraction; task22's matching actual audit accepts only its bounded scientific review:7/7 B20 observation cases with0 differences/errors,90 H convention matches comprising84 known counts and6 expected unresolved outcomes. Native binaries remain those actually bound during the runs. Same-method class census executes776 classes/353 dataclasses/8 Protocols/2 TypedDicts with unchanged source; this is counting evidence, not final consolidation acceptance.
- Unique task23 inventories56 current Plan12 documents and has a matching actual readonly audit. integrated_requirements.json records its exact source/task/tree/turn and root critical limits: internal diagnostics are not automatically forbidden ordinary-success wrappers; public D01–D03 proposals and profile activation retain their concrete decision requirements. Historical explicit-state/xTB/2280 baseline/O0 reports cannot substitute for current automatic-profile/held-out evidence. Global scientific-contract assessment stays with local Sol-max; no profile is activated.
- First complete default execution after prior74-node corrections reports4932 passed/5 failed/123 skipped/3 xfailed/49 subtests,10 retained warnings in145.33s, unchanged recorded source/test/native/inputs. Unique primary task24 and actual readonly audit identify retired constructor callers plus strict-H readiness before unsupported-bond projection. Existing complete factual builder migration preserves supplied H/aromatic/ID/bond/stereo facts; supplied-H UNKNOWN and absent-H UNKNOWN retain separate native/readiness assertions. Three containing files then report45 passed/1 failed unchanged in1.91s. The remaining exact graph roundtrip crosses factual serialization into evaluated aromaticity; its original assertion is retained. Root challenged the helper's overbroad mandatory-aromatic-presence proposal using the defined false default and CBond's newly added metal with absent H, and assigned a narrow revision rather than repeating full analysis. All original failures remain; full Stage5 gates remain open.


### Effective constitution and companion alignment — 2026-10-30

- The owner's explicit “微调goal.md并正式生效” authorization is enacted. The assigned local GPT-6.1 Sol/max author revised the81-line goal (6e116730…) and2549-line main plan (fc976a8d…), then minimally reconciled the interface/Plan17/data-audit companions and Stage1 D01 clauses. Root completed the post-revision full main-plan read1–2549, final goal and all four normative companions plus current Stage1 decisions. A truncated data appendix was explicitly recovered; exact current hashes/ranges and retained author patch/check artifacts are in source-matrix-corrections/evidence.json. Earlier6195…/130f… reads remain historical and are not relabeled as these bytes.
- Adopted distinctions: ordinary requested value versus necessary internal records; same-scope factual conflict versus equivalent projection; named hard local rejection versus non-hard reporting/continuation; and unavailable integrity-broken values even when finite. D01's generic public confidence/admissibility-envelope proposal is marked superseded and never approved; specific later public design remains unapproved. No API/schema/ABI, science algorithm, frozen efficiency policy or Stage13 ordering changed.
- Sol-max found no settled Stage5 H/aromatic return-contract blocker. Its12 silent explicit-H unavailable-result probe outcomes and imprecise strict H/aromatic diagnostics are assigned concretely in integrated_requirements.json to existing Stage7C/8A owners/consumers for finding/control decisions and tests. The H integer/source/default/display contracts remain. This is tracked open global diagnostic work, not an invented Stage5 gate or a waiver.
- Stage5 remains open at a5fc8f52f0028c28ec666438f7bca4d9951034a4; source/benchmark correction proceeds under the now-aligned route. Next: focused63-node latest-byte validation, unique task28 implementation-delta primary review/audit, complete default, coherent Git/native/matrix/package/performance handoff. Stage6 has not begun. Required final scientific/performance/Plan12 gates remain open. No new bridge/OTP, model/config/permission/PhaseB action occurred.

- Companion handoff matches root's final reread hashes:295/184/543/200lines;24/24checks,17local links,layout0errors,scoped diffcheckrc0.18protected identities and44original evidence artifacts are unchanged. Stage7C/8A diagnostic debt now binds the explicit42-reference handoff; no correction route or new public schema was inferred.
- Unique task24r3 resolved source readiness/raw aromatic capture after root rejected an overbroad mandatory-presence suggestion; task25r2 confirms exact corpus/two-node installed support correction, and task26 confirms the bounded factual loader/capture delta. All matching actual readonly audits are retained. The next complete default exposed11 further evidence callers and3new direct-core imports; original4932pass/12fail remains unchanged. Public imports fix the architecture debt without allowlist expansion.
- Task27's unique primary analysis placed H readiness only before source-fixture hashing and introduced a selected-original-row observation boundary. Latest implementation captures the full raw record once, slices only original values/presence/active bonds and leaves existing stereo row filtering intact; it never prepares/repairs observed results. Original-row absentH still fails, added-H absence is excluded, and original graph/stereo/raw metadata/status/seeds are unchanged. Task28's static PASS and exact readonly audit01a18c5b-e920-73e2-b446-fecf4e78cf84 match task12a4bfe5…/treea8a4abaa…; root accepted those key conclusions only.
- Actual recorded focused attempts retain49pass/2fail,51pass,63pass(subset capture) and final63pass(full capture/prefix,1.53pytest seconds). Fresh complete default attempt3 passes4945 tests with123skips/3xfails/49subtests,10retainedwarnings in143.15pytest/145.085recorder seconds; source/tests/native/input bytes stayed unchanged. Earlier failures and the exact correction explanations/commands/environments remain in source-matrix-corrections/evidence.json. This is Python3.11 source evidence; five-version/debug/current-wheel/sdist/focused performance and global scientific/efficiency gates remain open. No later-stage entry or closure occurred.


### Stage5 current-source execution and small Git delivery — 2026-10-30

- Dedicated Git management completed all32released natural commits,with ten successful every-third-commit pushes and independent remote-SHA checks. Current HEAD is cac32af6b4001f377cf347f750bf5a73ffe2bea9; last pushed40dae100a23ff516ea8605c745fe9be754802fec leaves two natural commits since push. Its exact final-handoff.json SHA f1cfdbd096b168bb77cc6c5d9d17b7b4b75a16cc0c1010f2c825bf7d546e9919 records all mappings,108released blobs,1068watched source/native files and actual staged/AST/JSON/link checks. No filler commit,push,reset or rollback was performed. Root accepted the clean handoff before execution and holds future Git mutations pending the aggregate gate decision.
- Unique ai4 task29 PASS_DEFAULT_EVIDENCE_INTEGRITY and its matching exact-turn actual readonly audit establish the final4945pass/123skip/3xfail/49subtest Python3.11 source default. All former5and12failures actually pass; none was removed,skipped or xfailed. All5063original cases remain. Roster growth is seven first-to-second plus one second-to-final,eight first-to-final. The immutable packet's added_in_final label describes final minus union(first,second),not all eight additions. Root records this concrete limitation without rewriting or repeating the evidence.
- Local executor completed five actual forced fact-native ABI rebuilds,15collections and40matrix commands on frozen cac32af6… source/index. Each Python3.10–3.14 passes2969source/175debug/2837current-installed tests,with42sourceand42installedskips and3strictxfails in each source/installed run; total29905passed/420skipped/30xfailed/0failed. Required Stage5 nodes and all12coordination validation-evidence nodes pass. Wheel audits/smokes and548-current-member/no-prebuilt.so sdist pass. Independent native dependencies were provenance-reused,not clean-rebuilt; original orchestration failure occurred before any build and is retained. No default/focused/science rerun was performed.
- The exact execution handoff SHA6f0edf2507e6588d53dfa807b595d7d20f1525306901428a8179e231ad1d92b5 and freeze f91c33f1… bind all commands/wheels/results. Root ran only python3 .plan20-artifacts/20261030/stage5/root-gate-checks/inspect-final-evidence.py: rc0,critical4source hashes,newest-ABI3.14source/debug/installed origins,observation hashes,outcomes and totals match. Authoritative validated_outcomes data are embedded in each summary.json with actual observation/JUnit references; no standalone validated-outcomes.json is claimed.
- Focused performance completed22workers and11equal-work comparisons with10paired measured samples each and zero environment mismatches. Seven workloads regress,three improve,one is neutral. write/read ratios are13.104/9.925/9.379;clean-read30.718/33.530/34.400;graph/view3.510. Complete local lazy and graph/geometry families are4.3867and1.8796. These regressions remain open and block final performance acceptance; no waiver,full71/GPU/end-to-end or global RSS result is claimed.
- The retained seven-case/90H scientific reuse binding already passed for632source files per run and exact observer/fixture/contract inputs. Original native identities remain their historical execution identities. The check and unique task22content/audit are now indexed in scientific evidence; no scientific work was repeated.
- Unique task30 and its exact-turn actual readonly audit supply FACTUAL_RESULT_CONSUMER_INVENTORY_ONLY for later Stage7C/8A. Root records its live control/provenance readers,unproven generic-confidence necessity and missing broader-consumer scope in integrated_requirements.json. This is neither a deletion mandate nor stage/design approval. The existing communication owner deduplicates the already archived user audit receipt.
- The execution hold is released after completed unchanged-byte handoff; source implementation remains fixed while root awaits unique aggregate task31 content and its matching actual exact-turn audit. Maintained evidence records are being updated; Stage5 is not closed yet and Stage6 has not begun. Final Stage3/4/runtime/RSS,fullO0,all71/GPU,Plan12 and Stage13ordering obligations remain unchanged.


### Stage5 aggregate review and owning-stage closure — 2026-10-30

- Unique ai4 task31 hashd41d955b…/snapshot22a469ef…/sourcecac32af6…/turn01a18c83-0fe9-7c60-9c87-9a0791fd4e1d has matching actual read-only audit(context1273/range1272–1294,zero tools/commands/delegation/write entries). Substantive BLOCKED_WITH_CONCRETE_EVIDENCE_GAPS is retained verbatim;enforcement PASS is not a content pass. It supports the bounded technical/package evidence and finds a real clean-read comparison defect.
- Root confirmed only the critical three scalar-output cases:all10measured samples have equal completed_calls20 but baseline value0 versus candidate32/256/1024. The captured factory never prepares the baseline H fixture although the frozen semantic requires already-prepared values. work_counters omits scalarvalue. The earlier raw7regression/3improved/1neutral output is retained,not accepted for all11comparisons;current disposition is8valid comparisons(4regressions/3improvements/1parity)and3noncomparables. Complete lazy-family4.3867 and overall aggregates are invalidated. Graph/geometry1.8796 remains a valid regression. Exact root confirmation is root-gate-checks/clean-read-semantic-check-attempt1.json;no benchmark reran.
- Required-node wording is corrected without changing outcomes:source1334pass+3strictxfail,installed1332pass+3strictxfail,debug168pass. All required nodes are present and none ordinarily skips;JUnit codes the3xfails as skipped. Totals29905/420ordinaryskip/30xfail/0failure and120coordination validation-evidence executions agree. Only fact_native was freshly rebuilt;sdist content audit is not fresh sdist-install/runtime evidence.
- New stages/stage05/report.md and gates/{review.md,evidence.json} close the unchanged main-plan Stage5technical/default/assignment-report gate. Correctly classifying the3noncomparables resolves the unsupported reporting claim;it does not grant performance acceptance. Existing Section12.2/Stage12global gates retain measured slowdowns,missing full71/GPU/end-to-end/RSS and clean-read fixture/accounting correction. No workload,weight,threshold,denominator,scientific default or final obligation changed.
- Root completed current normative prerequisite reads before later entry:goal81/main2549/chemical-interface295/Plan17alignment184/data-audit543/Plan21draft68,Stage1decisions200,Stage5contracts99/Hdefaults421,integrated requirements and full1711-line verification map,governance plus both complete references,and relevant execution/predecessor records. The remaining data appendix291–543 and map1–1711 were read explicitly. Main status/navigation changes here are mechanical;settled architecture/scientific routes remain those reread. Stage6substantive design/implementation still awaits its separate entry record.
- Source/test/native bytes remain the validated snapshot. Git owner has a four-natural-commit evidence/navigation plan,HEAD/index unchanged and cadencecounter2. The max-effort Stage6author is reading current requirements and reusing task21r2inventory only;it is prohibited from file edits/design until root publishes STAGE6_ENTRY_READY. Root continues through Plan20/Plan12to13B;this checkpoint does not complete the active goal.

- Final owning-record layout attempt2 passes58Markdown/53JSON/112indexed targets,19historicalJSON/9Markdown identities and the original log prefix;git diff --check passes. This checks the actual new Stage5gate/report/index paths,not source or performance. Task32 causal analysis and its existing matching actual audit are accepted only as a bounded Stage12handoff. Root confirms transaction-before-clean-check,duplicate guarded H-input extraction and absent clean fixture preparation;no per-substep profiler attribution,optimization,explicit-H fixture decision or measurement is accepted. A second input read remains an integrity guard until equivalence is proved;fast-path status/dirty/rollback/native semantics and narrower-identity Sol-max review remain required. Exact artifacts and critical limits are appended to performance/evidence.json. The repeated user audit receipt was deduplicated without resend or re-audit.

- The final record-release audit invocation attempt3 stopped with rc1 before producing a report because its output parent had not been created;the wrapper then failed to parse empty stdout and did not retain the nested stderr. The exact invocation,observed missing-parent precondition and evidence limit remain in root-record-layout-attempt3/orchestration-failure.json. Corrected attempt4 creates a separate output directory before running. No source/test/scientific/performance command was repeated.


### Stage6 entry alignment and bounded stereo design — 2026-10-30

- Dedicated Gitowner completed C44–C47(4bfae48/03506fe/6c56f09/aed86ad),including the gate pair with its evidence to resolve forwardlinks;both every-third pushes match remoteHEADaed86adb601866d9870df371aef4df0812d19a97. All18released hashes matchHEAD/worktree;1068watchedsource/native files including37binaries remain unchanged;index/worktreeclean,cadencecounter0. Final stagedchecks:fourdiffchecks,sevenJSONparses,156relativelinks,blobidentitiesand original1902-line logprefix pluscontiguous1903–1928suffix. No test/build/performance/science repeated.
- Root completes Stage6entry alignment before substantive work:full current goal/main and normativecompanions,accepteddecisions/map/integratedrequirements and relevanthistory were read;currentheader/map changes are mechanical and recorded with exact prior/currenthashes. The Sol-maxowner independently completed currentfullreads and reused audited task21r2inventory;current4747-line stereoSHA4c8c66f4…matches it. entry_review.json binds allcurrentdocument hashes,actualpredecessor/sourceassessment,readcoverageandremainingglobalgates.
- Boundedcontract:physicallymoveLevel1transport/fact contracts and allGate2–7implementation bodies to single authoritative owners;retaina realfivefunction/eighteenclassidentitypublicfacade,exact schema/order/sourcecodes/nativepacking,scientificalgorithms,H/dative/hiddenbondsemantics,conditionalpreparation/errorprecedence,lock/source/atomicityandtoolkittransport. Initialexclusiveauthor /root/stage6_stereo_solmax(gpt-6.1-sol/max)owns onlycontracts/review.md andevidence.json. It mustresolve anacyclicsinglehomeforLevel2reports,pureremapvslifecycle,Gate6snapshot/nativeboundaryandsharedvalidation/assignment/alignmenthelpers. No production/testedit isreleasedyet. Rootownsindexes/map/log/architecturebudgetandGitreadiness;allGitmutationsremainwithGitownerandrollbackwithroot.
- Required completion remains the unchangedStage6gate:allLevel1/Level2Gate2–7,nativepacking,RDroundtrip,publicsignaturesandcompleteaffectedPython3.10–3.14source+installedcontracts,plusrecordedconnectedmigrationandnativeAPIevidence. TheexistingexecutorreusesactualStage5environment/evidence topreparecommandswithout execution. Stage7/8diagnostics/results,Stage10clean-native/sdk/sdist,Stage12fullscience/performance/architecture/type/API/docsandcommittedB_exp→separateD_exp→I_expremainopen. NoHdefault/importnormalization,scientificrulechange,newpublicschemaorperformancewaiverispartofStage6.

- Stage6validationpreparation uses actualStage5environment/command/nativeorigin evidence without reruns. Its31futurepayloads cover completeaffected source/installed/nativeAPIcollection+execution on3.10–3.14 and3.11default;newextractiontestsjoinatSOURCE_READY. Rootcorrected the inventory drift that existingbufferownershiptest was markeduncreated. Nativeheader/recipe cases remain source-only;twoexternalSDKABIcaseshave nofrozenprefixbinding andremainmissing untilresolved. No freshnativebuild/reuse/currentwheel/currentStage6testpassisclaimed. Mainheader/index/mapreceive mechanicalentry/designstatus only;currentrequirements unchanged.

- Git owner delivered the seven-file entry checkpoint C48 `90ff36c2c0b415bb9cf27628265bb4528e425922`; counter1, last remote `aed86adb601866d9870df371aef4df0812d19a97`, no push. Selected blobs, staged diff/two JSON/81 relative links, mechanical main-line4 change and exact1928-line prefix plus contiguous1929–1938 log append pass. The1068 source/native files and37 binaries remain unchanged. Current source still has no Stage6 implementation delta. The separately assigned local `stage6_validation` verifier is explicitly GPT-6.1 Sol/medium and reuses the completed preparation, owning only ignored runner output; source/runtime execution awaits an exact SOURCE_READY freeze.

- Unique ai4task33 has matchedtask41da58ae…/captureb81bb731…/source90ff36…/exactturn01a18c9f-bd67-7aa2-bd85-525d45710052 and actualreadonlyaudit(context1333/range1332–1370,zerotools/writeentries,established5.6-sol/xhigh). SubstantiveBLOCKED_WITH_MINIMAL_FIXESisretained. Rootkeychecks confirm unsafehistoricalhashoverride,defaultrequired-nativeerror,uncontainedsupportcopyandunprovedtwo-caseSDKsummary;all sixboundedfixesareassignedtoexplicit6.1-sol/mediumverifierinfreshrevision2. SDKenforcementisnative-tier-specific;defaultgatingskipsremainseparate. No freeze/runtime authorization occurred.
- DedicatedGitread-onlyprovenancetask proves the13missingthird_party/rdkitrecipes from actualhistoricalGitblobs and oldcp312/cp313identity/buildrecords,with46Gitcommands/123artifactchecks andfiveRDnativeABIhashchains. SupplementSHA02f81aeb…isaccepted onlyfor thathistorical omission. Rootcheckedtwooriginalartifacthashes/sourcecommit/lockkeys in native-proof-keypoints.json;no large audit reran. The actual future source/native closurestillmustmatch;SDK/version/buildacceptanceisnotinferred. All exactartifacts/limitsareboundinStage6validation-preparation/evidence.json.
- Sol-maxdelivered frozen316-line stereo extraction design(reviewSHA76f6b980…/evidenceab702cd9…),partitioning116definitions/23assignments into single owners with18classidentities/fivefunctions and an acyclicdesignedimportgraph. It requestsoneN09snapshot reservation,explicitreadlockscope/immutablegraphpackingreview,andboundedcurrentpayloadidentitywithglobalstoragedecompositionstillopen. Unique task34primarydesignreviewisdispatchedbeforedependentdecisions;rootreadthefullproposal andcriticalquestions,not a duplicate symbol audit. Production/testsremainunchanged andunreleased.

- Unique ai4 task34 has matching task21ad8134…/capture1bed0972…/source90ff36…/exact turn01a18ca8-73e4-7eb3-ab38-ab4dcbca44a7 and actual read-only audit(context1373/range1372–1410; zero tools, commands, delegation or write entries). Its substantive BLOCKED_WITH_REQUIRED_DESIGN_DELTAS_AND_ROOT_DECISIONS is preserved verbatim in Stage6 contracts/primary-review-r1.md. Root key inspection confirms inherited-lock inversion risk and Section5.3 ownership debt; the working-copy code is a risk pattern, not a proven call into alignment. Root requires late coordinate coercion after graph validation, and no blanket transaction rejection without a safe reviewed route. Sol-max owns the bounded revision2 design, exact N09 cap-one revision and named Stage7E decomposition gate. Production/test ownership and runtime execution remain unreleased; no full symbol audit or completed test was repeated.

- Git owner delivered C49 e537fad8… and C50 25735ab07ca9a2b30f5a395abe4fd10133852913 as two natural evidence checkpoints; every-third push succeeded and independent ls-remote matches C50, cadence0. Exact released blobs, staged diff/JSON/12 relative-link checks and contiguous log through1946 pass. Relevant source/test/native commit diffs are empty; no broad audit or runtime check repeated. Pending Sol-max design and runner-r2 records remained outside these commits.
- Unique task33 revision2 has matching task2def81c0…/captureec97b93f…/historical source90ff36…/exact turn01a18cac-3d1f-7682-95be-cc60483a53e4 and actual read-only audit(context1416/range1412–1440; zero write/tool/command/delegation). Substantive SIX_FINDINGS_RESOLVED_STATICALLY_FREEZE_EXECUTION_REMAINS_HELD and supplied53 deterministic checks are preserved in validation-preparation/runner-review-r2.{md,json}. Root inspected only historical disjoint attestation, per-tier nodes, exact SDK outcomes and release-path preflight; it accepts the frozen six-fix mechanism without repeating the synthetic suite. Current-source closure, Stage6 tests/wheel/native and SDK/science/performance results remain unexecuted or missing; SOURCE_READY is still required.
- Root assigned unique ai4 task35 PREENTRY_INVENTORY_ONLY for current Stage7A conversion-cache/Crystal/thermo/visualization consumers, reusing Stage1 accepted public decisions. This is independent of task34 revision2; it carries no Stage7A design/implementation release and must be rebound if connected source changes during Stage6. No source sync, environment probe or runtime command is authorized by that inventory.

- Git owner delivered C51 `e5508c6d67be9310db1a4b9a4521573f95a9b9b5` for the supplied53 synthetic checks, task33 revision2 review/audit and bounded root static acceptance; counter1, last verified push `25735ab07ca9a2b30f5a395abe4fd10133852913`, no push. Exact released hashes, two JSON parses, nine relative links and contiguous log1947–1950 passed. The verbatim review retained its intentional line22 Markdown hard break as the sole raw diff-check exception. SOURCE_READY/freeze/runtime/SDK acceptance remained absent; no runner suite or broad audit was repeated.
- Root corrects its Stage6 review filenames to the maintained stage catalog without a new content review: `contracts/primary-review-r1.{md,json}` moves to `contracts/initial-review/{review.md,evidence.json}`; `validation-preparation/runner-review-r2.{md,json}` moves to `validation-preparation/revision2-review/{review.md,evidence.json}`. Both raw review SHA256 identities remain exactly40ae276830957e41f409ede01727d2478d4a7d063115588f073f7f6d728be354 and598f97e10930d45981fa2cd3f1f0a004447954caf5eb9e5ccfa05708b5e1e2c4. Only active relative links/JSON references and the relocated disposition hash change; historical log entries remain verbatim. Actual file-layout, staged JSON/link/diff checks are recorded by Git management; pending Sol-max proposal/decisions/N09/main/source/tests and runtime status remain untouched.

- Git owner delivered C51 e5508c6d… and C52 37fd81db6229e6e4015185635a163d073c2c2aa1, cadence2 since verified push25735ab…. C52 corrects root-created review filenames to catalog topic directories, preserving both rawreview hashes exactly and historical log1–1950. Its staged diff/threeJSON/nine-link/catalog/index checks pass; no content review or runtime test repeated. The original C51 diff check flagged only an intentional two-space Markdown break in the preserved verbatim review; remaining staged content passed.
- Unique ai4 task35 task3ee1cec5…/capturea6891525…/source25735ab…/turn01a18cb6-a3f8-70e2-a66a-5dcdc88ee306 has matching actual read-only audit(context1443/range1442–1464, zero tools/commands/delegation/write entries). Root accepts only PREENTRY_INVENTORY_ONLY and preserves15421rawbytes/SHAe915fe39… in Stage7 preentry-conversion records. Root checked cached to_obmol, writer coordinate mutation, current Crystal shape and the stale map test label; it did not repeat the broad consumer inventory.543 read_mol uses are an index, not reviewed implementation bodies. Native session-owned toolkit values are distinct; Thermo/visualization/cold-wheel/memory evidence remains open. Post-Stage6 source/transitive-import/wheel rebind is mandatory before Stage7A entry or ownership release; no Stage6 dependency or Stage7A entry/design approval is inferred.

- Git owner delivered and remotely verified C53 979455ed8bada61f4d44038935e7f02c1b204c86; cadence0. Stage7A raw15421-byte inventory hash remains exact; staged diff/JSON/23 links, evidence binding, exact catalog hunks and historical log prefix pass. Pending Sol-max/main/N09 work remained uncommitted; no runtime or source review repeated.
- Unique task36 taskd4cfb178…/capture82d24bc5…/source979455…/turn01a18cc2-3eae-7c22-9e21-b55e13cc4543 has matching actual read-only audit(context1470/range1466–1494; zero tools/commands/delegation/write entries). Root accepts only PREENTRY_INVENTORY_ONLY and preserves16062rawbytes/SHAe3d66e94… in Stage7 preentry-selections. Root checked ConstraintSet shape, active-only clear and current Atom projection; no completed Stage4/5 algorithms were re-audited. Active-only clear is characterized behavior, not an established defect; absent common constraint fields do not prove every separate native path ignores constraints. Permanent hidden-clear transitive invalidation and native propagation remain unproved, not failed. Post-Stage6/7A source/import/test/wheel rebind remains mandatory.

- Git owner delivered C54 `98bcfa8dedec124e0bdc0c2e1c2f634afa38b597`, cadence1/no push/index clean; Stage7B raw16062bytes/SHAe3d66e94 preserved, JSON/25links/exact catalog hunks and append-only1957-1959 pass. Only seven intentional raw-review Markdown hard breaks triggered diff-check; no source/runtime work repeated.
- After the material Stage6 revision2/Stage7E proposal, root fully reread the current2652-line main SHA57ede191… (the first output truncation was recovered at290-420), all four normative companions, current goal/integrated requirements/full1724-line map, and both governance references. Root reinspected only tracked transaction acquisition, empty-domain owner binding and mapping ordering; no duplicate consumer/symbol audit. Exact read identities are in contracts/revision2-review/evidence.json.
- Unique task34 revision2 task2af2b832…/capture3e2476a8…/historical source979455…/turn01a18cc5-cec4-7b93-aabb-d0e509405603 has matching actual read-only audit(context1500/range1496-1538, zero tools/commands/delegation/write entries). Root verified6721 raw bytes/SHA5b44416b… and retained the reviewer status BLOCKED_PENDING_ROOT_KEY_ACCEPTANCE_AND_IMPLEMENTATION_EVIDENCE separately. Six prior findings are statically resolved. Root key accepts the two exact local RuntimeError triggers/messages/late effects, N09, ordering/introspection/remap/packing and mandatory Stage7E; author finalizes the bounded empty-source-lock correction and premeasurement adverse-cost disposition. No implementation/runtime/gate acceptance follows.
- Root explicitly releases only bounded pre-cutover source/existing-Stage5-wheel introspection, cold pickle and recursively typed Gate6 target/native-mapping fixture capture to the existing medium verifier under ignored stage6/precutover-baseline; no installation, collection/full suite/build/freeze or source/test edit. This preserves current behavior before the connected cutover. Production/test implementation remains HOLD pending final record integration and baseline completion; exact SOURCE_READY still precedes mapped runtime.
- Root mechanically integrates the accepted Stage7E owner/gate into stage_verification_map.json (predecessor7D, successor8A, explicitStage12/global block, zero-new-type budget and entry-time exact source/slot/unbound/test rebind), and corrects future planned_not_created labels only where the current file exists. Stage6 map gains exact introspection/capture/concurrency/packing/adverse-cost obligations and a planned extraction contract file. Catalog/index changes report static key acceptance only; no future-stage entry or test pass is inferred.
- Executed document-only `python plan/20_fact_rule_architecture/audit_stage_records.py --output /tmp/p20-stage6-r2-record-integration-audit.json` on C54 dirty design/integration records:65 Markdown/61JSON/127 indexed assets, passed with zero errors. This verifies layout/navigation only; report SHA11ae6e8f9ad759858b53e03fb3d87922a615314dd506bf5fb6341d7fed19222b, no runtime, scientific, build or performance acceptance.
- Unique task37 task447ea0ec…/captureb862f110…/historical source979455…/turn01a18ccd-234f-7490-9f48-aed26bb8c628 has matching actual read-only audit(context1544/range1540-1564, zero tools/commands/delegation/write entries). Root accepts only PREENTRY_GAP_INVENTORY_ONLY, preserves16879rawbytes/SHAdc0f0fc9… in Stage7 preentry-results, and extends the integrated requirement inventory. Root checked current STOP-only recovery, triple MCA attachment and reduced pipeline state hash; none proves a repository-wide unused field or authorizes schema/control changes. Complete dynamic/external/cold-wheel consumers and prior-stage source/import/test/wheel rebind remain required before Stage7C/8A ownership. No Stage6 dependency, entry or runtime claim.
- Git owner delivered C55 `19c56777cb8712245780a9577fe0e1b9e71b8478`, cadence2/no push/index clean; only task34r2 raw review/root disposition pair, its one Stage6 index bullet and contiguous1960-1964log committed. Staged diff/oneJSON/11links/raw6721byte identity passed with no whitespace exception; pending author/map/future inventory changes remained unstaged.
- Sol-max final integration preserves all five reviewed documents (398275bytes; manifest SHAdaab249c…), records root-key accepted exact errors/N09/Stage7E, corrects empty-domain source-lock wording and freezes S6-R2-09 adverse-cost disposition before measurements. Root read the final Stage6-only main delta and cost rule. `/home/zhangzhiyuan/usr/conda3/envs/hp-usage/bin/python /tmp/p20-stage6-root-integration-check.py` finally PASS_BOUNDED_ROOT_INTEGRATION_STATIC_CHECKS_ONLY, exit0 after three retained prose-checker corrections; unchanged116/23partition,18/5/23public roster,14/32DAG nodes/counts/caps/globalSection12bytes. Final main SHA f83bf4603b638222960cef341766fd6accea6929bdfe5b814abb4e4edc7bf1b6; all exact hashes and failed commands remain in owning evidence. No production/test edits or runtime gate.
- Git owner delivered C56 `579daaa94e78d5dc79161c63d52141ccf8c188b9` and independently verified the remote; cadence0/index clean. Twelve assigned files/hunks, fourJSON/111links/fiveacceptedhashes and contiguous1965-1966log pass; five-document398275-byte historical archive and globalSection12 bytes remain unchanged. Exact handoff SHAfb32696eb82edea3ebc6ad5e7cff7dd7a5b9da4b23afa96a00c167be6f636c2c; no source/runtime audit repeated.
- Root releases the already accepted Stage6 coherent cutover to existing GPT-6.1Sol/max author: exact20production/16test paths copied into stage06/implementation/evidence.json; native/build/ABI and sixread-only connected owners excluded, root retains ledgers/map/log/Git. Pre-cutover preservation report SHA0359e616…/408-fileinventory5f383081…/costprotocolab8d7228…/oldcostmanifest5b532bc6… retains14commands(12rc0/2originalrc1),10passedcapture/coldprocesses and468oldcostrecords; no pytestsession. Baseline is sufficient for implementation; candidate imports/collection/tests/builds remain STAGE6_SOURCE_READY gated, not current gate or performance acceptance.
- Git owner delivered C57 `8083c46af23b5a5bf1239ebb6d24e7628a6bede6`, cadence1/no push/index clean; exact16879-byte task37 inventory hash, twoJSON/10links/cross-bindings and contiguous1967-1971append pass. Only task37 records/integrated requirement plus bounded release history committed; Stage6 implementation pair/README and production remained outside. Handoff SHA4a380fece69fed8c6781019e3b73446344941b2c968a34dd7c256990b7cd77ad; no audit/runtime repeated.
- Root integrates the existing verifier connected-selection proposal into Stage6 map: all23original runtime files retained,22existing connected files added plus the planned extraction file for46source-and-installed runtime files; three native source-only files and exact accepted source-only node qualifications stay separate. Mechanical path-existence check found no missing connected file. Builder/commit/source/UID/facts/H/conformer/working-copy/forcefield/MCA/R0/cold-import coverage is assigned, not collected or passed. Candidate source/test/support/native/interpreter/output identities still bind only at SOURCE_READY; no extra validation executed.
- Executed document-only `python3 plan/20_fact_rule_architecture/audit_stage_records.py --output /tmp/p20-stage6-implementation-release-layout.json` after release/map integration:67Markdown/63JSON/131indexedassets, passed with zero errors; report SHAe9c7a5dbec2a6af0d3230c0f9d504404d2ba96a2bc166d40c1777ff4c26469fa. This checks actual new record naming/navigation only, not production/test/native/runtime/scientific/performance acceptance.
- Git owner delivered C58 `9e43821b3572b7adb97f1723561d4c2053ce4b5e`, cadence2/no push/index clean; seven released records/hunks plus contiguous1972-1974log pass staged diff/twoJSON/84links/exact46union/preserved23original-and3native paths/Stage6-only map changes. Root layout report retained without rerun; handoff SHA7c1c44559b904a6969a22c1be4875f3a41618a07839a92346a410139430b1a51. Worktree clean at handoff; coherent author cutover still in progress and no candidate runtime execution.
- Medium verifier completed static SOURCE_READY preparation (manifest SHAe4fed2b1796f8101b19b52fb87fb909397c0b29ced6a7a43a5d9d4afe4d3feda), confirming46source/installed paths and unchanged accepted runner. It identifies missing coordination fixture/init and two exactPlan17 runtime benchmark scripts; source-only deselection remains exactlyone. Root additionally confirms top-level mca requires installed AImodels origin equivalent to sourceparentconftest. Root releases only freshrevision3 ignored helper delta for these four hash-bound files/exactoutside-tests exceptions and installedMCApath; no selection reduction, checkoutsource injection, native/SDK policy change or candidateexecution. Revision2 retained; targeted synthetic checks and root critical delta review precede SOURCE_READY.
- Medium verifier delivers fresh runner revision3 exact support delta;32targeted synthetic/content checks passed after retained31pass attempt plus exactPlan17snapshotbinding check. Root reads entire bounded diff/supportbindings/results and accepts only support completeness/containment and installedMCAwheel-byte origin mechanism; no rerun53suite/duplicateaudit/candidate runtime. Selected runner SHAb04aca859f2150b1b9eb8fecb34b75ae972237f57d83144907a5035e1fd1c272; policye31c829b…/observer607e24e2…/support73e5f63f…; checksSHA81e86c2f30779989b2680b80b0771df5018526fc54020558ea1327399a826f07. Revision2/nativeinventory unchanged; all46files/oneinstalleddeselection/SDK/default/historical policies preserved. Original supportrelease bytes archived; SOURCE_READY/freeze/runtime still held.
- Git owner delivered and remotely verified C59 `176e3697986279a6f51a9dd3c49deaf5be79f1cc`, cadence0/index clean; supportpair/README/Stage6preparationmetadata plus contiguous1975-1977log passed twoJSON/20links/exactfouradditions/originaleightretention/helperbindings/unchanged46runtime-and3native paths. Supplied32checks not rerun; all36author source/test paths stayed dirty and unstaged. Handoff SHA010f50d060e789f6a581346ac941040d1da216fe2d661a11ec5ff3ab04c949d0; no candidate runtime/broad audit.
- Before SOURCE_READY, root intersects exact143native-inventory paths with released20production paths and identifies one overlap: hotpot/cheminfo/_native/buffers.py. Planned Pythonadapter rewire would be rejected by unconditional Stage5byte equality. Root assigns existing medium verifier READONLY build-reader classification/provenance diagnosis, preserving current revision3 and runtimeHOLD; no generic exemption, closure removal or reuse acceptance has been authorized. Any correction must keep exact owner-tree inventory(new/missing/suffixless)coverage, unchanged actualnative/build inputs, fullsource/wheelbuffer binding and native-boundary tests.
- Readonly diagnosis confirms buffers.py is packagedPythonruntime transport,not compiled/configured by tracedsetup/CMake;allfiveStage5wheelmembers bind its historicala916c72a…hash. Root accepts classification only and releases freshrevision4helper correction:retain exact143inventory/bothmembershipchecks,142strictunchangedinputs/allnativebinaries,one literal runtimeowner with historical+finalcandidate+frozenqualification binding,fullsource/wheel/origin driftguards and complete native-boundary tests. Diagnosticcurrent60202219…is transient. Revision3 remains immutable;no candidateexecution/nativecompatibility/source-ready or Stage10waiver. Exact release is validation-preparation/native-runtime-transport/evidence.json;targeted synthetic checks/rootdelta review pending.
- Mediumverifier delivers revision4 singletontransport correction with34+8newedgecases=42passed/0failed/0skipped across tworc0 stdlibattempts. Root reads allthree helperdiffs andbothactual resultrecords,accepts mechanism only and selects runnerSHAaca4a00ee35c0188e80c5bd496bcf65595110ea575d169b62098732a8236de49/policyae0254ca…/observer125b39ee…;inventorya2c35a80…/12support73e5f63f… unchanged. Exactfour-key role binds historicala916c72a…+finalcandidate+frozenqualificationb8bd1cb9…,keeps142strict/allnative equality and fullsnapshot/currentwheel/origin guards. Revision3 unchanged; no checksrerun/duplicateaudit/candidateexecution. Finalauthorbytes/task38/SOURCE_READY still required; no nativecompatibility/SDK/Stage10/global waiver.
- Executed document-only python3 plan/20_fact_rule_architecture/audit_stage_records.py --output /tmp/p20-stage6-native-runtime-record-layout.json on C59 plus native-runtime records:69Markdown/65JSON/135indexedassets,rc0/0errors;reportSHAd0ceb4689ada918b4f0461f235758909f4c6fb144325694ec1d3b575d1860a01 bound in qualification evidence. This verifies newtopic/index navigation only; no source/runtime/native/science/cost check repeated.
- Gitowner delivers C60a060900bfa3acbb7f6273b8b7d9365e8993ec904,one natural5recordqualification checkpoint,cadence1/no push/indexclean;2JSON/22links/diff/exact46runtime+3native+oneinstalleddeselection and C59log1-1977prefix/1978-1982suffix pass. All36author paths dirty/unstaged;handoffSHAb31097ad989f476ea1a46d43c3787c0d9257aab43d7057e58597db9ff28bbe36. No runtime/helper checks repeated.
- Solmax first finalhandoff0134b3fb… binds20production/16test paths; supplied25structural+19transformedAST checks pass,96originaldefinitionASTs/23constants unchanged,33newfunctions/53staticcases uncollected. Rootcritical lock/drift/order/packing/connectedrewire read finds no algorithmblocker. Actual root ASTarchitecture command against176e369… exits1 with4NEW directcoreimport debts only;raw resultSHAd59dd26ea213a41e024e99d80ebf73699f3f19198668c1ff7e2d6e3f0e5f1b76 retained in stage6/bookkeeping-attempt1. Author receives exact3file correction within existingownership,tests use currentpublic types/internaloperations avoid newcoretype dependency;allowlist growth forbidden. Originalhandoff remains historical;freshrevision2/finalsource rebind and SOURCE_READY still required.
- Solmax revision2handoff4af55362… corrects exactly3already-owned files,archives36precorrection source/test bytes plus51support artifacts intact,passes15affectedstaticchecks and actualarchitectureaudit with256remaininginheriteddebts/0new/0stale/0growth/0layers/0duplicatecarriers. Root reads exactdiff/actualresults and inherits unchanged96AST/23constant/19scientific-body evidence without rerun;allsource algorithms/costkwargs/publicfaçade remain unchanged. SOURCE_READY/runtime and unique task38 remain pending.
- Root executes PYTHONDONTWRITEBYTECODE=1 python3 .plan20-artifacts/20261030/stage6/bookkeeping-attempt1/capture_census.py /home/zhangzhiyuan/hotpot/.plan20-artifacts/20261030/stage6/bookkeeping-attempt1/census.json:rc0,exactoriginalleaf/CONTRACT_MARKERS method measures777classes/354dataclasses/8Protocols/2TypedDicts(delta+1/+1/0/0). N09capone has exactlypayload/atom_keys/bonds;27existingstereo classmoves receive0retirementcredit. Ledger/map nowrecord actualimplementation/newtestexistence;no allowlist edit or runtime/native/scientific/performance acceptance. Owningrecords:stage06/bookkeeping;singletonsupport/actualsource/primaryreview/fivePython matrix/cost/Stage7E gates remainopen.
- Document-only audit_stage_records.py --output /tmp/p20-stage6-static-implementation-record-layout.json passes70Markdown/66JSON/137indexedassets,rc0/0errors on a060900… plus newbookkeepingrecords;reportSHA6a960e16902693f62ff71ca946fba169a0b892ecaaa4a495face719a38a2c6c8 retained. Root verifies all36finalr2source/test hashes match without execution. Coherent extraction plus tests/actualbudget/staticrecords now released for one natural Gitcheckpoint;source-ready/runtime remainheld.
