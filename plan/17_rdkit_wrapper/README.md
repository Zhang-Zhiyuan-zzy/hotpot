# Stage 17: standalone native RDKit wrapper and conformer sampling

Status: owner-approved; implementation in progress.

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

The final implementation and benchmark report will be added after every
required phase and closure gate completes.
