# CBond, MCA and xTB result simplification — 2026-11-06

The owner-authorized cross-interface simplification is implemented for this
bounded non-FF slice. Focused current-source CPython 3.11 checks pass; root's
integrated review, installed behavior and Git delivery remain pending. The
source base is `16c01664b1fa2c6caab4e447bfba0ba421d7b7ca` on
`feat/automatic-coordination-state-workflow`, with the exact tested dirty source
bound by the evidence below.

This applies the [current goal](../../../../../goal.md), the
[owner-authorized entry](../../../../../implementation_log.md#2026-11-06--owner-authorized-cross-interface-result-and-facade-simplification-entry),
D1/D2's existing prediction/DTO/algorithm boundaries and A9's explicitly retained
detailed scientific-state contracts. It does not reopen a whole-stage gate.

## Initial four-edit interface effects

| Owner | Implemented public or persistence effect | Retained owner or observations |
| --- | --- | --- |
| `operations/inference/cbond/stage.py` | The single-path `report["cbond"]` no longer repeats `metal`. | Prepared invocation and persisted pipeline stage `argv` retain the requested metal. Donor indices, path probability and ordered steps retain their exact contents. The all-structures branch, raw scores and its structure probabilities/path counts are unchanged. |
| `plugins/xtb/stage.py` failure output | `sample_outcomes[i].failure.native_report` is removed. `sample_outcomes[i].report_index` joins the one retained native report in `report["xtb"]`. | The existing one-report dictionary and multiple-report tuple shapes remain. Failure code/type/message, native stdout/stderr/artifacts/identities, `native.log`, unavailable outcomes, positional mapping and all denominators remain. State-resolution failures still retain their scientific bundle and projection findings. |
| `plugins/xtb/stage.py` successful summary | `backend_projection.status="projected"` and its now-unused import are removed. | Projection convention, decision source, facts identity and selected-state identity remain. Actual `BackendProjectionResult.status` control and unavailable findings remain at their scientific-state owner. |
| `operations/inference/mca/results.py` | `MoleculePrediction.to_dict()` constructs its public JSON projection once instead of recursively serializing and then discarding both child collections. | The five ordered keys remain `smiles, formal_charge, sites, model_variant, atom_predictions`. Both child collections remain lists of fresh dictionaries, including empty defaults. Constructors, field order/defaults, class owners, immutable prediction values and version1 transfer DTOs are unchanged. |

No new type, alias, compatibility branch or schema-version switch is introduced.
The two stage-report key removals and failure-report nesting removal are
intentional current JSON contract changes; consumers use the retained owner.
At the initial four-edit checkpoint, the raw `XTBRunReport` was unchanged.
The approved extension below now migrates that detailed report coherently.

## Consumer migration and invariants

The official coordination pipeline test reads the requested CBond metal from
saved stage `argv` and still independently verifies the final Zn donor chemistry.
It also requires the ordinary CBond report to omit `metal`. This real-model test
was updated but is not executed by the focused selection below.

The sample-containment tests resolve native failures through `report_index`.
New single-sample controls cover both `XTBExecutionError` and `XTBResultError`,
retain the single-report dictionary shape and logs, and require the precise
failure payload. Existing multi-sample controls retain continuation and stable
input/output indices. Persisted stage JSON and run-manifest reports agree, with
one native report and no nested duplicate. Successful summaries retain their
complete state binding and projection decision; unavailable projection status
and nonconverged finite-output behavior remain covered.

The model-free MCA tests require the exact ordered nested JSON schema, empty
lists and independent mutable serialization containers across repeated calls.
The existing legacy prediction-constructor test remains selected. No scientific
model arithmetic, weights/features, state resolution, CBond thresholds or
probability algorithms, geometry/energy monitoring, coordinate commit,
trajectory, native ABI or portable SDF metadata is changed.

## Initial four-edit evidence and limits

Evidence root:
`/tmp/hotpot_plan20/short/output-simplification-20261106/nonff-implementation`.
`before.json` binds pre-edit owned source. `run_focused.py` and
`attempt01/intent.json` retain exact commands, cwd, environment overrides and
pre-run hashes. `attempt01/result.json` retains interpreter/origins, loaded
native-module paths/hashes, pre/post product and selected-test hashes, actual
elapsed time and return code. `attempt01/pytest.stdout`, `pytest.stderr` and
`junit.xml` retain the original result. No failed test attempt preceded this run.

The actual pytest command was:

```bash
/home/zhangzhiyuan/usr/conda3/envs/hp-usage/bin/python -m pytest -q --import-mode=importlib \
  tests/mca/test_result_serialization.py \
  tests/mca/test_api.py::test_existing_result_constructor_remains_compatible \
  tests/test_pipeline/test_sample_containment.py \
  tests/test_pipeline/test_builtin_stages.py \
  tests/test_plugin/test_xtb/test_native_convergence.py \
  tests/test_plugin/test_xtb/test_pipeline_integration.py \
  -k 'xtb or cbond or gfnff or prediction or existing_result' \
  --junitxml=/tmp/hotpot_plan20/short/output-simplification-20261106/nonff-implementation/attempt01/junit.xml
```

Cwd was `/home/zhangzhiyuan/hotpot`; overrides were `PYTHONPATH` set to that
checkout, `PYTHONDONTWRITEBYTECODE=1`, and `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
`OPENBLAS_NUM_THREADS` each set to1. CPython 3.11.16 returned0: **60 passed,
22 deselected**, no failures or skips, in5.91s pytest time /7.016803s child wall
time. Every frozen product/selected-test hash remained unchanged during the run.
`static-checks.json` additionally records the exact owned-path whitespace and
syntax checks.

This is source Python contract and fake-backend integration evidence using
existing native payloads. No real CBond/MCA model, official xTB calculation,
new native build/ABI, installed wheel, other Python version, GPU, scientific
corpus or comparative performance run is claimed. No speedup is measured.
Root owns those applicable integrated/delivery checks and final acceptance.
Rollback restores the three producers, connected test consumers and this
contract record together, without a second runtime route.


## Root-approved xTB provenance extension

After reviewing `nonff-implementation/xtb-followup-design.md`, root explicitly
approved this same-family contract migration on 2026-11-06. The original owner
entry and scientific boundaries remain effective; no unrelated stage or harness
redesign is introduced. Root owns the log, integrated release and final wheel.

`XTBRunReport` removes exactly eight fields: `requested_method, work_directory,
argv, return_code, stdout, stderr, elapsed_seconds, process_succeeded`. Its
existing `ProcessProvenance` is the sole process/workspace owner; readers use
`provenance.cwd/argv/return_code/stdout/stderr/elapsed_seconds`. The sole method
identifier remains `effective_method`, required by detached artifact parsing and
stream metadata. Constructors, repr/equality/field reflection and raw CLI/failure
JSON intentionally change. No forwarding properties, alias, compatibility loader,
new record, harness change or alternate schema is added.

The runner stops copying those fields. The parser reads the real provenance
return code and input filename; CLI and stage native logs read provenance text.
Benchmark readers likewise consume provenance, retaining actual work/time,
convergence, finite observations, artifacts, output/commit/workspace facts and
all failures, while emitting one method and no duplicate process-success bit.
Success pipeline summaries also remove repeated `requested_method`,
`effective_method`, `task` and `process_succeeded`; saved stage `argv` remains the
invocation owner. Direct report access replaces old fake-report fallback paths;
connected stage fixtures construct the actual current report type. Source tests,
benchmark fixtures, persisted JSON consumers, plugin guide and API inventory
migrate together. Existing slim-FF fixtures are preserved.

Expected invariants before validation: a nonzero process still fails locally
with the original code/text/artifacts; exit0 may be nonconverged yet retain finite
parsed observations. Convergence is never reduced to return code. Artifact
completeness, finite metrics, atom-order/state/source binding, source nonmutation,
coordinate transactions, cleanup/retention, GFN-FF charge-only/spin transport,
sample denominators and one-report/multiple-report persistence joins remain.
Scientific-state detailed envelopes, native ABI, model algorithms/features and
the portable SDF schema remain unchanged. Reject changed scientific values,
missing diagnostics/attempts, lost attribution or altered control/cleanup.
The affected source suite and real-backend availability are evaluated only after
the final product/test freeze; actual outcomes and remaining limits follow.


### Extended source verification and remaining delivery

The final extension product/test bytes passed the complete default xTB source
suite plus benchmark, sample-containment, builtin-stage and standalone CLI
contracts, with the MCA serialization and legacy-constructor rebind retained.
The actual command was:

```bash
/home/zhangzhiyuan/usr/conda3/envs/hp-usage/bin/python -m pytest -q --import-mode=importlib \
  tests/test_plugin/test_xtb \
  tests/benchmarks/coordination_complexes/test_xtb_benchmark.py \
  tests/test_pipeline/test_sample_containment.py \
  tests/test_pipeline/test_builtin_stages.py \
  tests/test_cli/test_xtb_cli.py \
  tests/mca/test_result_serialization.py \
  tests/mca/test_api.py::test_existing_result_constructor_remains_compatible \
  --junitxml=/tmp/hotpot_plan20/short/output-simplification-20261106/nonff-implementation/provenance-extension/attempt01/junit.xml
```

Result: **443 passed, 10 skipped, no failures/errors** out of453 collected cases,
in19.37s pytest time /20.529653s child wall time. Every skip says
`requires --run-external`; those opt-in official integrations are not passes.
No failed extension test attempt preceded this run. Existing source/native/model
history is not relabeled as fresh execution. The initial60-pass evidence remains
its earlier four-edit checkpoint; this complete selection supplies the new
xTB/report consumer evidence.

Cwd, interpreter and per-process environment overrides are the same as the
initial focused run above. The new
`nonff-implementation/provenance-extension/attempt01/intent.json`, `result.json`,
`pytest.stdout`, `pytest.stderr`, `junit.xml` and identity probe retain exact argv,
source/native/interpreter/origin identities, times, skip reasons and outcomes.
All frozen product/selected-test hashes were unchanged during execution.
`provenance-extension/schema-freeze.json` binds the final changed files and the
actual25→17 raw-report field migration. `static-checks.json` records owned-path
whitespace/syntax, unchanged harness/workflow/stream/state boundaries and the
API constructor/field inventory agreement.

The configured executable variable, current PATH, selected environment's bin and
three usual local xTB prefixes did not resolve an executable in the availability
probe; no optional real-backend smoke was run or claimed. This is not a machine-
wide absence claim. Existing fake-backend tests exercise actual subprocesses,
parsing, provenance, CLI streams, success/failure persistence and cleanup. The
full source suite retains nonzero-exit and exit0/nonconverged controls, missing
artifacts, nonfinite results, atom-order rejection, binding/rollback and
independent-sample denominators. Root owns final integrated/installed-wheel
qualification and any required official-backend/model execution. No new native
build/ABI, GPU/corpus, other-version or performance evidence is claimed.
