# Stage 12: composable xTB workflow

Status: design draft awaiting review; production implementation has not started.

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

