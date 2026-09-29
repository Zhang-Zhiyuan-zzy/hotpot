# Plan archive

This directory preserves Hotpot design, audit, implementation, and validation records by task stage. The numeric prefix records the working sequence; it does not define runtime dependencies.

| Stage | Topic | Primary records |
|---:|---|---|
| 01 | [SMARTS search](01_search_smarts/) | Parser and substructure-search test plan |
| 02 | [Complex build repair](02_complex_build_repair/) | Initial force-field/complex repair plan and commit retrospective |
| 03 | [Force-field review](03_forcefields_review/) | Interface, typing, compatibility, and geometry-policy audits |
| 04 | [Geometry package](04_geometry_package/) | Fact-only geometry package refactor |
| 05 | [Relevant cycles](05_relevant_cycles/) | Native relevant-cycle implementation plan |
| 06 | [Complex untangling](06_complex_untangling/) | Old/current workflow audit and staged untangling refactor |
| 07 | [Force-field abstraction](07_forcefields_abstraction/) | Utility reduction, module split, and trajectory architecture |
| 08 | [Force-field performance](08_forcefields_performance/) | Ring-scan optimization and three-stage workflow |
| 09 | [Native Open Babel force fields](09_native_openbabel_forcefields/) | Typed-buffer C++ backend implementation and validation |
| 10 | [Coordination benchmark](10_coordination_benchmark/) | Reusable 187-case benchmark and failure analysis |

Each topic README records the Git commits associated with that stage. Files under `artifacts/` are diagram sources, rendered HTML, or visual-review evidence. A `visual-check.json` whose status is `pending` is retained as provenance and is not evidence of a successful visual review.
