# Hotpot controlled molecular pipeline

`hotpot.pipeline` executes an ordered sequence of registered molecular stages
without invoking a shell. It carries Hotpot molecular objects in memory and
persists an atomic, content-addressed evidence tree for every completed stage.

## Command-line use

Use the exact `::` argument token between stages:

```bash
$ hotpot run --results-dir results/eu-001 -- \
    cbond Eu 'O=C(O)C' \
    :: ff --route complex --forcefield uff \
    :: xtb --method gfnff --task optimize \
    :: xtb --method gfn2 --task optimize
```

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
| `MolecularRecord(molecule, electronic_state=None)` | One molecule and its optional resolved electronic state. |
| `MolecularPayload(records)` | Ordered records passed between stages. |
| `StageContext` | Run directory, private temporary stage directory, index, and specification. |
| `StageResult` | Stage status, output payload, verified artifacts, structured report, and stderr. |
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
created only when every stage succeeds.

An invalid definition fails during preflight without creating the result root.
An execution failure persists the current stage with a `.failed` suffix,
prevents downstream execution, and raises `PipelineExecutionError` through the
Python API. A scientific quality failure retains its inspectable stage output,
returns `PipelineStatus.FAILED`, and also prevents downstream execution.

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
