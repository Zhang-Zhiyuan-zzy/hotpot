# Stage 20 — Fact-Carrier and Rule Architecture

Status: implementation authorized on 2026-10-28; Stage0–2 complete;
Stage3 in progress. All public chemical
types follow a common protocol for fact ownership, interface mapping,
permissions, dependencies, buffer ownership and lifecycle. Atom's unified
fact table is one concrete implementation. Stage2 completed the approved dead/placeholder implementation cleanup;
production architecture extraction starts in Stage3. MetalCluster and its future Molecule association are registered as
Plan21 design targets; other proposed public changes and unspecified cluster
interface details remain subject to individual review.

Implementation branch: `implement/plan20-fact-rule-architecture`
Original planning branch: `refactor/fact-rule-architecture`

Original planning baseline: `0a4b0ee`; reviewed documentation: `d9eb80e`.
Stage1 must record a fresh executable/performance baseline at the selected
Plan17/Plan20 boundary before production refactoring.

## Records

- [Architecture audit and implementation plan](fact_rule_architecture_implementation_plan.md)
- [Implementation log and stage-entry assessments](implementation_log.md)
- [Stage2 execution evidence](stage2_execution_evidence.json)
- [Stage3 entry assessment](stage3_entry_review.json)
- [Stage1 API and consumer review](stage1_api_review.md)
- [Stage1 concrete interface decisions](stage1_contract_decisions.md)
- [Stage1 common interface semantic review](stage1_interface_contract_review.md)
- [Stage1 assembly asset review](stage1_asset_review.md)
- [Chemical-type public-interface protocol and fact storage](chemical_type_interface_contract.md)
- [Plan17/Plan20 execution alignment](plan17_plan20_execution_alignment.md)
- [Data-object classification and consolidation audit](data_object_scope_audit.md)
- [Plan21 MetalCluster and multi-attachment superatom draft](../21_metal_cluster/metal_cluster_implementation_plan.md)

The protocol covers carriers, independent values, collections, views,
scientific results and transport contracts. Roles determine mutability and
lifecycle; no common base class or physical storage layout is required.
MetalCluster is recorded as a registered aggregate related to Molecule,
Atom and Bond, with `Molecule.metal_clusters -> list[MetalCluster]` planned.
Plan20 registers that structure only; Plan21 defines concrete cluster kinds,
matching and superatom execution after the owner supplies their forms.

Stage0 selects Plan20 first: the minimum Plan17 MCA native closure is not yet
implemented, so B20/F20 use the current sole Python-RDKit MCA algorithm. The
17-MCA batch follows F20 with separate B17/F17 evidence. Scientific outcomes
are versioned monitoring reports; 187/182
does not impose a fixed success count. Global efficiency must be at least
comparable to the pre-refactor baseline, with improvement expected.

Every stage and independently gated sub-stage begins with a full reread of the
plan and normative companions plus a recorded project-state assessment.
Executed work has an append-only implementation log and a final report.
Planning records and proposed commands are not execution or benchmark evidence.

Stage1 measurement policies live in `performance_manifest.json`,
`stage1_sampling_plan.json` and `python_matrix_manifest.json`. The supported
Python matrix uses `run_python_matrix_baseline.py` and
`compare_python_matrix_baselines.py`; its per-version comparisons remain
separate from the original performance-family weights. The implementation log
binds executed reports and records incomplete or failed attempts explicitly.
