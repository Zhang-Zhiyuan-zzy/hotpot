# Stage 12: composable xTB workflow

Status: the execution wrapper and the canonical scientific-state architecture
are implemented through consumer migration. The screening profile remains
inactive until its reviewed scientific rules and held-out activation criteria
are approved; the large-corpus Phase 10 validation is also outstanding.

Stable-xTB scientific acceptance remains complete for the declared
Eu/GFN-FF/GFN2 execution scope. Extended-GFN-FF fragment-charge/actinide
behavior remains an explicit backend boundary.

Planning branch: `feature/xtb-workflow`

Planning baseline: `116a255`

This stage replaced the legacy xTB prototype with independent, composable
GFN-FF and GFN-xTB calculation nodes. It reuses the existing CBond and UFF
complex-building workflows without changing their computational kernels.

The design uses the official external xTB executable behind a molecular-stream
adapter, separates charge and spin inference, and treats method/element support
as a versioned backend capability. In particular, xTB 6.7.1 does not provide a
valid Am path; only a validated extended GFN-FF build may accept Am, and
GFN0/1/2-xTB remain limited to `Z <= 86`.

The reviewed revision also splits the calculator module into a focused package,
adds a controlled `hotpot run` results pipeline, and makes xTB the first
reference implementation over narrow reusable external-process primitives.

- [Detailed implementation plan](xtb_workflow_implementation.md)
- [Separate compatibility audit](compatibility_audit.md)
- [Implementation report](xtb_workflow_implementation_report.md)
- [Validation report](xtb_workflow_validation_report.md)
- [Normative scientific-state implementation plan](scientific_state_implementation.md)
- [Scientific-state implementation and validation status](scientific_state_implementation_report.md)
- [Complete scientific-state public API signature tree](scientific_state_implementation.md#111-complete-public-api-signature-tree)
- [Complete staged rule list](scientific_state_implementation.md#15-complete-rule-list)
- [Individually reviewable hard, default, empirical and AI rules](rules/)
- [Deterministic scientific-state providers](providers/)
- [Backend execution-selection policies](execution/)
- [Optional execution-recovery policies](recovery/)
- [Superseded state-inference plans and audit sources](older/)

For charge, oxidation-state and electronic-state work, the normative plan and
its indexed `rules/`, `providers/`, `execution/` and `recovery/`
specifications are the only implementation instructions. The documents under
`older/` preserve evidence and design history but are not active plans.

The implemented architecture distinguishes candidate production from decision:
hard rules define the legal space, providers and optional rules supply
attributable candidates/preferences, a factorized joint composer binds
charge/OS/electronic-state paths, and named default policies make the
deterministic workflow decision. Its profile artifact, rules and typed
abstention paths are available, but `COORDINATION_SCREENING_V1` is deliberately
not activated as a production default before scientific approval and Phase 10
held-out validation.

## Delivered public surfaces

- `hotpot.cheminfo.calculator`: canonical molecular facts, charge, oxidation,
  electronic-state, coupled-candidate, rule, policy and backend-projection
  contracts;
- `hotpot.plugins.xtb`: typed official-backend probe, runner, adapters,
  independent GFN-FF/GFN-xTB operations, strict SDF stream, CLI, and pipeline
  stage;
- `hotpot.pipeline`: typed molecular payloads, lazy registered stages, atomic
  artifact persistence, SHA-256 lineage, and failure propagation;
- `hotpot xtb`: standalone molecular-stream command;
- `hotpot run`: controlled `cbond :: ff :: xtb` workflow command.

The old `XtbCalculator`/`xtb_batch_run` prototype, mutable package cache, empty
xTB writer, and obsolete consumers have been removed rather than retained as a
compatibility path.

## Implementation commits

| Area | Commits |
|---|---|
| Planning | `4d4fbbb`, `82ff25e`, `c4fcdf8` |
| Calculator package | `27c8203`, `f13b782` |
| Electronic state | `a286e48`, `d80b69f`, `e3be153` |
| Generic process harness | `a214128`, `db9fc49` |
| xTB backend, adapters, nodes, and CLI | `439122a`, `bda6eef`, `796c597`, `b4a9761`, `ea93af4` |
| Controlled pipeline | `8ea5083`, `aa7d34c`, `03724ff`, `d58c1e3`, `ab04158` |
| Initial official validation and prototype removal | `e17c2af`, `5328071`, `64ec730` |
| Extended official parity | `b791f5e`, `652a3c7` |
| Real coordination pipelines | `9fbf984` |
| 187-structure benchmark | `837c804` |
| Python-matrix test inclusion | `8313d55` |
| Rebuild documentation and benchmark CI | `b54b59b`, `5751e00` |
| Final reports and Eu README pipeline | `d6a948f`, `8e015d3` |
| Runtime/wheel validation closure | `b4827ba` |
| README evidence and hosted CI closure | `565c639`, `bb9a737`, `1706de4`, `9477d3f`, `5c4dc8f` |

The validation report records five official direct-parity checks, three real
controlled coordination pipelines, the complete four-route 187-structure Eu
benchmark, the CPython 3.9-3.14 runtime matrix and six clean ABI-wheel checks.
Stable xTB 6.7.1 still has no valid Am path, and no result in this stage claims
otherwise.

The root README publishes a compact subset of the corpus result. Its tracked
source is `assets/readme/xtb_coordination_benchmark.json`; the full protocol,
failure analysis and applicability limits remain in the validation report.

