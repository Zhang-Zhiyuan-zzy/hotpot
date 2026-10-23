# Stage 17: standalone native RDKit wrapper and conformer sampling

Status: component implementation and local validation are complete; release
closure validation is now owner-approved and in progress. It covers a current
semantic 187/182 cohort, O0/F1/F2/C1/C2 execution, F1 business evidence, and
the provider decision. Recovering the old frozen manifest or graph sidecars is
no longer a prerequisite. The pinned local CPython 3.10-3.14 workflow-
equivalent wheel matrix is complete under the declared Open Babel cross-wheel
dependency model.

Planning branch: `feature/rdWrapper`

Planning baseline: `9640d906a9d13e969152d0f3a4d92a3a16b9e0ce`

The owner referred to this work as `plan16`. Stage 16 is already occupied by
the completed native-search implementation, so this directory uses the next
unambiguous archive number without replacing or duplicating Stage 16.

The normative implementation proposal is:

- [rdkit_wrapper_implementation_plan.md](rdkit_wrapper_implementation_plan.md)

The proposal fixes Hotpot to RDKit 2026.09.1, defines independently runnable
Python and C++ `rdWrappers` APIs, retains `obWrappers`, preserves the complete
current force-field workflow during component-substitution testing, and adds a
two-level Hotpot stereochemistry design (transport first, chemical
validation/assignment/alignment second).

Implementation proceeds in small verified checkpoints. The live record is:

- [implementation_log.md](implementation_log.md)

The implementation, validation, benchmark, and remaining-gate report is:

- [implementation_report.md](implementation_report.md)

The corrected force-field business-validation proposal, including the full
objective/experiment status matrices, is:

- [forcefield_component_validation_plan.md](forcefield_component_validation_plan.md)

The implementation report preserves the evidence and assumptions that existed
at handoff time. The corrected validation plan supersedes its historical-
artifact blocker after owner approval. It also separates local evidence from
GitHub-hosted CI provenance and strict standalone auditwheel manylinux
compliance, neither of which is claimed.
