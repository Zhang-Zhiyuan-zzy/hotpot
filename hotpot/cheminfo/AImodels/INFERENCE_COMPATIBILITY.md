# ONNX inference compatibility

The maintained MCA and CBond ONNX compatibility matrix targets CPython
3.10--3.14. The matrix covers CPU inference, model loading,
input featurization, dynamic dimensions, shared molecule conversion,
calculator attachment, native RI-style substructure search, SMARTS parsing
and MCA site selection. It does not claim that unrelated Hotpot plugins or
optional scientific packages support every interpreter in the matrix.
All maintained jobs resolve the release dependency pinned exactly to RDKit
2026.09.1 (`rdkit==2026.9.1`).

Historical inference baseline for the interpreters that remain supported,
validated on 2026-09-14 before the RDKit 2026.09.1 pin:

| Python | NumPy | ONNX Runtime | RDKit | Tests |
| --- | --- | --- | --- | --- |
| 3.10.20 | 2.2.6 | 1.23.2 | 2026.03.6 | 149 passed + 49 subtests |
| 3.11.15 | 2.4.6 | 1.30.0 | 2026.03.6 | 149 passed + 49 subtests |
| 3.12.13 | 2.5.3 | 1.30.0 | 2026.03.6 | 149 passed + 49 subtests |
| 3.13.15 | 2.5.3 | 1.30.0 | 2026.03.6 | 149 passed + 49 subtests |
| 3.14.7 | 2.5.3 | 1.30.0 | 2026.03.6 | 149 passed + 49 subtests |

The piperidine MCA reference was exactly `503.25 kJ/mol` in every environment.
The largest cross-version difference in the fixed CBond reference output was
`5.96e-7`, below the test tolerance of `1e-6`.

The native RI search extension and its public search behavior were separately
revalidated on 2026-10-21 across the then-maintained matrix, which also
included Python 3.9. See the
[Stage 16 implementation report](../../../plan/16_native_search/implementation_report.md)
for the exact source-test, packaging and semantic matrix.

Run the complete matrix with [uv](https://docs.astral.sh/uv/):

```bash
bash tests/run_inference_compatibility.sh
```

Pass selected versions to shorten a local run:

```bash
bash tests/run_inference_compatibility.sh 3.10 3.14
```

The lower ONNX Runtime bound remains 1.19; supported Python versions resolve
to a compatible release within that range.

CI also builds a wheel, installs it without the source checkout on `sys.path`,
and runs a real MCA prediction plus public native-search smoke. This verifies
that the ONNX graph, its external weight files and the compiled search
extension are present in the distributable package.

As an additional historical forward-compatibility check, `python -m compileall
hotpot` completed under the prior six-interpreter matrix. Warnings emitted by
third-party packages do not affect the tested inference and graph-search paths.
The current supported release range is governed by package metadata and is
Python 3.10--3.14.
