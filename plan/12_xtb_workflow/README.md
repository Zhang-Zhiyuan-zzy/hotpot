# Stage 12: composable xTB workflow

Status: Phases 1–18 implemented; core production and documented validation
complete, with the remaining scientific acceptance gaps listed below.

Planning branch: `feature/xtb-workflow`

Planning baseline: `116a255`

This stage will turn the legacy xTB prototype into independent, composable
GFN-FF and GFN-xTB calculation nodes. It will reuse the existing CBond and UFF
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

## Delivered public surfaces

- `hotpot.cheminfo.calculator`: the canonical calculator package, including
  independent charge and spin inference contracts;
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
| Official validation and prototype removal | `e17c2af`, `5328071` |

The validation report explicitly lists the remaining scientific acceptance
gaps; implementation completion does not imply that every optional benchmark
from the original plan has been executed.

