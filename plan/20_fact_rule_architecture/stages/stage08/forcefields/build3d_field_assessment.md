# build3d field-use assessment — 2026-11-06

Bounded source review at `a9ecad5902ce49b3854b21aa330c5fe4de8a9f47`,
with the accompanying coordinate/query refinement. This records actual consumers
and cleanup candidates; it does not change public report fields or scientific
algorithms. No new calculations were needed for this inventory.

## Which interface returns which result

| Entry | Returned value | Boundary |
|---|---|---|
| `Molecule.build3d` | `BuildAndOptimizeReport` or `ComplexBuildReport` | Construction plus optimization; organic/metal routes differ. |
| `operations.forcefields.build3d` | `Build3DReport(atom_count, added_hydrogen_count, quality_report)` | Initial coordinates only. |
| `obWrappers.build` | `BuildReport(succeeded, rules)` | Separate native builder outcome and actually applied rules. |

There is no tracked Python `build_3d` alias. Public FF contracts live in
`hotpot/cheminfo/operations/forcefields/contracts.py`. Their fields are consumed
collectively: `stage.json_value` recursively serializes every dataclass field
except `trajectory`; both CLI and pipeline use it. Benchmark serializers also
retain the nested fields. Thus lack of an additional decision reader does not
establish that a field is unused. External/dynamic consumers remain outside this
bounded review.

## Fields with a concrete purpose

| Fields / family | Actual use and distinction |
|---|---|
| `effective_forcefield`, energy/gradient units | Attribute the method actually selected and the reported numerical observations. Auto selection can differ from the requested method. |
| `best_energy`, `final_energy`, `converged`, `terminal_converged`, `best_epoch` | Describe selected versus terminal frames, which can differ after rollback/recovery. |
| `setup_succeeded`, `termination_reason`, `exploded`, gradients, `energy_changes`, `max_displacements` | Acceptance rules and automatic routing consume actual availability, stop reasons and stability observations. |
| `epochs_completed`, `selected_segment_epochs_completed`, `steps_submitted`, `initialization_steps`, `attempt_count`, `attempts_completed` | Actual observations/work, including initialization and rejected work; distinct from requested budgets. |
| `routing_report`, rejected candidates, restoration/untangling outcomes, warning messages | Explain actual attempts, local failures, exceptional actions and incomplete geometric coverage. A zero confirmed-piercing count is not proof of complete coverage. |
| `quality_report` checks and measured geometry | Name affected subjects, observed values and gate conditions; pipeline/CLI and routing read these. Initial-build and final-optimization observations describe different frames. |
| `added_hydrogen_count`, accepted-candidate and relocation counts | Actual execution deltas/discoveries, not merely copies of requested inputs. Not every such count drives a subsequent decision. |
| Timings | Useful actual stage/cumulative observations where measured. Ordinary-path `elapsed_seconds=0.0` must not be interpreted as measured zero time. |
| Requested trajectory / archive | Used by persistence, frame reconstruction, galleries and scientific evidence. Retain the selected/terminal distinction and format5 topology/identity mapping. |

Core connected readers are `operations/forcefields/{attempts,auto,workflows,stage}.py`,
`rules/physics/forcefields/acceptance.py`, `commands/ff.py`, and
`tests/benchmarks/coordination_complexes/{pipeline,validation_evidence,reporting}.py`.

## Redundant or relocatable output

| Candidate | Finding | Migration consequence |
|---|---|---|
| `ForceFieldRunReport.steps_completed` | Every producer supplies `None`; no exact completed-step value is obtained. | Remove an unavailable placeholder only while preserving the explicit unknown-work meaning in evidence consumers. Never substitute zero or submitted steps. |
| `RingUntanglingReport.resolved` | Exactly `final_piercing_count == 0`. | Can be derived; preserve separate uncertainty/coverage findings. |
| Ring/restoration `attempt_limit` | Repeats an invocation budget. | Belongs with invocation/configuration, while actual attempts remain results. |
| Restoration `bond_count`; initial-build `atom_count` | Repeat input/committed topology; atom count is also in quality metrics. | Migrate public constructors/JSON and work-counter readers together. |
| `requested_forcefield`, `convergence_level`, validation `level` | Repeat known method inputs. | Store at configuration/calculation owner; effective method remains attributable. |
| Optimized workflow `quality_report` and `optimization.quality_report` | Same report object serialized twice. | Select one JSON owner while preserving existing workflow control access and build-only behavior. |
| Aggregated warning text and some history arrays | May duplicate nested diagnostics/requested movie contents. | Confirm subject/frame coverage before consolidation; no blanket deletion. |

The first five rows are concrete cleanup candidates, not a claim that the
remaining fields are scientifically necessary in every ordinary return. The
current specific public report contract and serialized readers remain intact in
this slice. Follow-up schema cleanup is indexed as **P20-D047** and must migrate
constructors, JSON, docs, tests and actual consumers coherently.

In particular, maintained `run_scientific_report.py` reads work fields with
`getattr`; `run_performance_baseline.py::_work_counters` recursively reads their
JSON names and preserves unknown `None` separately from zero. These are actual
consumers even though a simple attribute search misses them.

## Parameters are a separate question

`candidate_count` is a proven no-op: public docstrings mark it reserved and the
workflow explicitly discards it. It is an additional simplification candidate,
not a supported multiple-conformer algorithm. In contrast,
`candidate_warmup_steps`, `candidate_score_steps`,
`best_candidate_refine_steps` and `max_attempts` control real complex-route work.
Route-specific applicability does not make them dead parameters.

## Evidence boundary

The complete field/producer/reader table and spelling-discovery index are
archived with the 2026-11-06 refinement evidence. The index is discovery evidence,
not type-resolved proof of complete use/nonuse. No native report/ABI, trajectory
format, model input, gate or FF numerical algorithm was changed by this review.
