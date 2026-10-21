# Stage 16: Native substructure search

Status: implemented and validated.

Planning branch: `perf/native-search`

Planning baseline: `2cf0893`

This stage replaces the NetworkX-backed execution path of
`hotpot.cheminfo.search` with one in-tree C++17 deterministic RI-style
subgraph-monomorphism engine bound through pybind11. No third-party graph
matcher becomes the production executor. The public query, SMARTS, search,
result, semantics, exception, and `Molecule.search_substructure()` interfaces
are frozen explicitly in the normative plan. MCA is an acceptance consumer,
not the owner of a second search implementation.

The complete embedding set for stable predicates and every chemical/query
predicate retain the current Hotpot meaning. The legacy NetworkX enumeration
order is explicitly not retained as a permanent contract. The native
implementation instead has a documented deterministic order: identical search
inputs and deterministic predicates produce the same mapping sequence for a
given implementation.

Opaque Python callables remain supported. Only the callable predicate leaf is
evaluated in Python under the GIL; topology traversal, candidate generation,
backtracking, and all other predicates stay in the single native matcher. No
runtime backend flag, profile, or whole-search Python fallback is introduced.

- [Normative implementation plan](native_search_implementation.md)
- [Implementation log](implementation_log.md)
- [Implementation report](implementation_report.md)

The implementation log records each incremental checkpoint. The final report
summarizes correctness, packaging, performance, compatibility, and the
explicitly retained validation limitations.
