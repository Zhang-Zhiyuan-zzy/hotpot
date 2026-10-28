# Stage 17: standalone native RDKit wrapper and conformer sampling

Status: component implementation and automated force-field validation are
complete at authoritative checkpoint
`b0aa6c5224a97469359c53448ba48d24d8a46112`. The schema-v2 formal run covers
the recorded semantic 187/182 cohort and produces 1,092 O0/F1/F2/C1/C2 plus F1
repeat results. F1 passed 180/182 cases, below the then-used 182/182 replacement
threshold; that historical decision retained Open Babel as the default complex
conformer provider. Those counts and thresholds describe that checkpoint,
not the current scientific reporting policy.

Stage0 execution decision (2026-10-28): the native MMFF/UFF and MCA semantic
closure is not implemented, so Plan20 proceeds first using the existing sole
Python-RDKit MCA algorithm for B20/F20. The `17-MCA` batch is deferred until
after F20 and then records its own fresh B17/F17. Multi-conformer clustering
and ordinary-organic default cutovers remain separate later batches. Scientific
geometry/energy gates are monitors; current runs deliver versioned per-case
reports and do not require a fixed 187/182 success count or historical output
equality. Public API/ABI additions remain pending explicit owner review.

- [Plan17 / Plan20 execution order, baselines and shared reporting policy](../20_fact_rule_architecture/plan17_plan20_execution_alignment.md)

The exhaustive 910-cell acceptance gallery is complete with 869 selected
structures, 41 scientific-failure placeholders, and zero render errors. A
separate 41-case failure-diagnostic gallery now exposes 10 nondegenerate
rejected complex frames, the collapsed all-zero F1 case-10 attempt, and 30
labeled input graphs for routes that produced no coordinate frame. These
diagnostics do not become admitted coordinate assets. Attributable human review
is still pending. The owner-authorized benchmark records must not be published
into `hotpot/dataset` until that review is completed; failed route-cases will
remain explicit records without substituted coordinates.

The conditional local CPython 3.10-3.14 workflow-equivalent wheel matrix has
also passed at `b0aa6c5`, including installed-wheel smokes, archive-content and
integrity checks, and direct `_rd_native` ELF audits. This remains local
evidence rather than GitHub-hosted CI provenance.

Historical implementation branch: `perf/mca-resource-efficient-workflow`

Planning branch: `feature/rdWrapper`

Planning baseline: `9640d906a9d13e969152d0f3a4d92a3a16b9e0ce`

The owner referred to this work as `plan16`. Stage 16 was already occupied by
the completed native-search implementation, so this directory uses the next
unambiguous archive number without replacing or duplicating Stage 16.

The original completed-stage implementation proposal is:

- [rdkit_wrapper_implementation_plan.md](rdkit_wrapper_implementation_plan.md)

It pins Hotpot to RDKit 2026.09.1, defines independently runnable Python and
C++ `rdWrappers` APIs, retains `obWrappers`, preserves the established
force-field workflow during component-substitution testing, and specifies the
two-level Hotpot stereochemistry model.

The historical corrected business-validation contract for that checkpoint is:

- [forcefield_component_validation_plan.md](forcefield_component_validation_plan.md)

Recovering a byte-identical historical manifest or graph sidecars is not an
acceptance prerequisite. The current schema-v2 cohort preserves literal
Hotpot graph semantics, including per-atom current implicit-hydrogen counts.

The small-step execution record is:

- [implementation_log.md](implementation_log.md)

The consolidated implementation, validation, benchmark, decision, and
remaining-gate report is:

- [implementation_report.md](implementation_report.md)

The complete per-failure inventory and qualified route-efficiency report is:

- [formal_failure_analysis.md](formal_failure_analysis.md)

The former Plan-19 O0/MCA study belongs to this same native-RDKit objective and
has therefore been absorbed here. Its paired prediction, seed-diversity,
performance, and native-force-field assessment is:

- [mca_conformer_engine_assessment.md](mca_conformer_engine_assessment.md)

Its reproducible drivers are:

- [benchmark_mca_conformer_engines.py](benchmark_mca_conformer_engines.py)
- [benchmark_o0_seed_diversity.py](benchmark_o0_seed_diversity.py)
- [benchmark_rdwrapper_seed_diversity.py](benchmark_rdwrapper_seed_diversity.py)

The continuation plan separates the minimal native UFF/MMFF + MCA batch,
deferred until after Plan20 F20, from later clustering and ordinary-organic
workflow batches. Its exact new or changed public interfaces and default
cutovers remain awaiting owner review:

- [organic_native_conformer_implementation_plan.md](organic_native_conformer_implementation_plan.md)

Local evidence is kept distinct from GitHub-hosted CI provenance and strict
standalone auditwheel manylinux compliance, neither of which is claimed.
