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
