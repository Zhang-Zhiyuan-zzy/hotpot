# Hotpot controlled molecular pipeline

`hotpot.pipeline` executes an ordered sequence of registered molecular stages
without invoking a shell. It carries Hotpot molecular objects in memory and
persists an atomic, content-addressed evidence tree for every completed stage.

## Command-line use

Use the exact `::` argument token between stages:

```bash
$ hotpot run --results-dir results/eu-001 -- \
    cbond Eu 'O=C(O)C' \
    :: ff --rebuild --route complex --forcefield uff \
    :: xtb --method gfnff --task optimize \
    :: xtb --method gfn2 --task optimize
```

The force-field stage uses `--rebuild` because the CBond payload carries the
new topology with retained 2D coordinates, not a force-field-built 3D starting
structure.

The controller never evaluates shell syntax. It accepts either the inline form
above or a strict JSON workflow; see [`cli_doc.md`](cli_doc.md).

## Python use

<!-- Verified by tests/test_pipeline/test_documentation.py::test_pipeline_readme_python_example -->

```python
from pathlib import Path

import hotpot
from hotpot.pipeline import (
    MolecularPayload,
    MolecularRecord,
    StageSpec,
    run_pipeline,
)

def main() -> None:
    mol = hotpot.read_mol("CC")
    result = run_pipeline(
        (
            StageSpec(
                "ff",
                (
                    "--route", "organic",
                    "--epochs", "1",
                    "--steps-per-epoch", "5",
                    "--quality", "off",
                    "--seed", "1",
                ),
            ),
        ),
        results_directory=Path("results/run-001"),
        initial_payload=MolecularPayload((MolecularRecord(mol),)),
    )
    print(result.status.value)


if __name__ == "__main__":
    main()
```

The first stage may create its own molecular payload, as `cbond` does, or the
caller may provide `initial_payload=MolecularPayload(...)`.

## Public API

| API | Purpose |
|---|---|
| `StageSpec(name, argv)` | Immutable registered-stage name and ordered, uninterpreted arguments. |
| `MolecularRecord(molecule, scientific_state=None)` | One molecule and its optional attributable scientific-state bundle. |
| `MolecularPayload(records)` | Ordered records passed between stages. |
| `StageContext` | Run directory, private temporary stage directory, index, and specification. |
| `StageResult` | Stage status, output payload, verified artifacts, structured report, stderr, and expected `break_pipeline` control. |
| `PipelineRunResult` | Terminal status, payload, result root, ordered stage results, and optional failed-stage index. |
| `MolecularStage.prepare(spec)` | Parse a stage specification without executing it. |
| `PreparedMolecularStage.execute(payload, context)` | Execute a prepared operation without reparsing its options. |
| `run_pipeline(specs, *, results_directory, initial_payload=None)` | Preflight and execute all stages in order. |
| `register_stage(name, import_path)` | Register one lazy stage factory import path. |
| `get_stage(name)` | Resolve a registered stage only when selected. |
| `builtin_stage_names()` | Return the built-in stage names (`cbond`, `ff`, and `xtb`). |

## Result contract

The requested result root must not already exist. Each stage is first written
to a private temporary directory and renamed into place only when its artifacts
and hashes are complete.

```text
results/run-001/
├── input/
│   └── input.sdf
├── manifest.json
├── stages/
│   ├── 00-ff/
│   └── 01-xtb/
└── final.sdf
```

`manifest.json` records ordered argv, payload lineage, elapsed time, relative
artifact paths, SHA-256 digests, reports, and terminal status. `final.sdf` is
created only when the workflow reaches its successful terminal boundary.

An invalid definition fails during preflight without creating the result root.
An execution failure persists the current stage with a `.failed` suffix,
prevents downstream execution, and raises `PipelineExecutionError` through the
Python API. FF/xTB isolate only known sample unavailability; surviving samples
continue. Finite non-hard findings retain their observations and original method
reports by default, including warning checks. A nonempty surviving FF/xTB cohort
returns `SUCCEEDED`, with failures disclosed in its counts.

Each FF/xTB report records `sample_outcomes_version=1`, counts for every input,
attempt, output, unavailable sample, sample break and unattempted sample, and one
ordered row per input. Rows join inputs to outputs by `input_index` and nullable
`output_index`; `report_index` joins actual method reports. Findings use exact
`FF:<check.name>` or `XTB:<check.name>` codes. Adjacent persisted FF/xTB edges
preserve positional attribution for equal-content molecules. The cohort anchor
is nonempty initial input or persisted CBond output/rank; arbitrary unrecorded
reordering or generating stages do not acquire automatic lineage.

FF/xTB accept repeatable `--sample-break-on CODE` and `--batch-break-on CODE` with
exact parser choices and no default breaks. A sample break removes the current
finite sample from downstream input and saves it at the row's `observation_path`
(`observations/NNNN.sdf`). A batch break takes precedence, retains the current
finite output, stops further attempts, and records remaining inputs as
`not-attempted`. Unavailable values never become molecular outputs. Missing
required FF metrics may retain a defined geometry as an observation artifact.

A batch break or empty `QUALITY_FAILED` result stops the pipeline at that stage
after normal evidence persistence. The Python API returns `PipelineStatus.FAILED`
with the retained stage payload/report and failed-stage index, without creating
`final.sdf`. Other stages' intermediate nonempty `QUALITY_FAILED` results may
continue; a final quality failure also returns `FAILED`. These explicit per-code
flags belong to controlled stages; standalone FF/xTB commands retain their
existing scheduling without those flags.

## Adding a stage

Keep parsing separate from execution:

```python
class MyStage:
    def prepare(self, spec: StageSpec) -> PreparedMolecularStage:
        ...


def get_stage() -> MolecularStage:
    return MyStage()
```

Then register the factory lazily:

```python
from hotpot.pipeline import register_stage

register_stage("my-stage", "my_package.stage:get_stage")
```

A stage owns its scientific policy and artifacts. The controller owns only
ordering, typed payload transfer, atomic persistence, lineage, hashes, and
failure propagation. Arbitrary commands, Python expressions, dynamic entry
points, shell expansion, automatic entry-point discovery, implicit fallback,
resume, and parallel stage
execution are intentionally outside the contract.
