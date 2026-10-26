# M_17-19: Native graph, stereochemistry, and RDKit integration

Status: implementation, CPython 3.10-3.14 matrix, and live GPU qualification
complete.

Integration branch: `merge/m17-19-native-graph-rdwrapper`

Integration worktree: `/tmp/hotpot-m17-19`

Initial parent: `6639754`

Graph parent: `clean/networkx@b912c8a`

Common ancestor: `2704f3c`

Semantic merge commit: `b5e04d6`

This merge stage integrates the completed Plan 19 single-source native graph
architecture with the completed and proposed Plan 17 stereochemistry,
`rdWrappers`, conformer, and force-field work. The final Graph-facing API,
topology authority, native algorithm ownership, and removal policy follow Plan
19. Stereochemical transport and interpretation, native RDKit boundaries, and
force-field working-copy semantics follow Plan 17.

The integration has three non-negotiable properties:

1. there is one authoritative topology representation and one topology refresh
   implementation;
2. permanent topology mutation preserves the Plan 17 atom, bond, axis, group,
   and transport-fact lifecycle exactly; and
3. obsolete Graph, NetworkX, MCA-adapter, updater, cache, native-source, alias,
   and fallback surfaces are deleted rather than retained for compatibility.

Documents:

- [Normative implementation plan](native_graph_rdkit_integration_implementation.md)
- [Complete API evolution tree](overview_tree.md)
- [API and implementation migration matrix](api_implementation_migration_matrix.md)
- [Implementation log](implementation_log.md)
- [Implementation report](implementation_report.md)

This directory records integration evidence. It does not replace the original
Plan 17 or Plan 19 plans, logs, reports, benchmarks, and scientific evidence.
