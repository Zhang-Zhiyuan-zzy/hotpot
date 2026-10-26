# Plan 19: Single-source topology and private native graph consolidation

Status: implementation complete on 2026-10-26.

Planning branch: `clean/networkx`

Planning baseline: `2704f3c`

This stage makes Molecule membership and Bond endpoints the sole authoritative
Python topology facts, removes duplicate Atom adjacency state, removes
NetworkX from Hotpot production code, consolidates reusable graph algorithms
under `hotpot.cheminfo.graph`, and promotes the existing native-search CSR
topology into Hotpot's one private, ephemeral native graph representation.
Molecule retains one read-only, rebuildable topology query index; it does not
cache a computational Graph or require one-shot algorithms to construct Graph
merely to conform to an interface. Legacy NetworkX cycle-basis properties are
replaced, without aliases, by deterministic `cycle_basis`/`basis_rings` and
complete bounded `all_cycle_basis`/`all_basis_rings` properties.

- [Normative implementation plan](native_graph_consolidation_implementation.md)
- [Implementation log](implementation_log.md) records incremental execution.
- [Implementation report](implementation_report.md) records final correctness,
  performance, packaging, compatibility, and residual-risk evidence.

