# Native geometry and force-field pipeline

Status: production implementation complete; in-scope Phase 10 validation
complete, with declared project-global and historical-ablation limits

Planning baseline: `9c70ef63f337`

Complete production-code commit: `2518a0c`

Validation manifest commit: `59e5741` (documentation-only; adds the native
API guide and does not change production behaviour)

This stage moved Hotpot's high-frequency geometry kernels and the
coordination-complex placement/optimization workflow to coarse-grained C++
backends while preserving the public Python API and the independent three-stage
complex workflow.

- [Approved refactor plan and historical baseline](native_geometry_forcefield_refactor.md)
- [As-built implementation report](native_geometry_forcefield_implementation_report.md)
- [Validation and benchmark report](native_geometry_forcefield_test_report.md)
- [Native API and usage guide](native_api_and_usage.md)

The approved plan remains the record of the intended architecture and the
pre-refactor baseline. The implementation and test reports are authoritative
for the resulting source layout, measured behaviour and validation evidence.

Final validation used the manifest-recorded commit `59e5741`, whose production
code is identical to `2518a0c`: 177 of 187 inputs passed, one post-CBond
structure (case 61, the declared scientific non-goal) failed the quality gate,
and nine inputs failed during CBond construction. All retained trajectories,
selected structures and final PNG artifacts passed the deep integrity checks.
