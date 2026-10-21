# ONNX inference compatibility

The MCA and CBond ONNX packages and their required Hotpot integration are
tested on CPython 3.9--3.14. The matrix covers CPU inference, model loading,
input featurization, dynamic dimensions, shared molecule conversion,
calculator attachment, native RI-style substructure search, SMARTS parsing
and MCA site selection. It does not claim that unrelated Hotpot plugins or
optional scientific packages support every interpreter in the matrix.

Inference baseline validated on 2026-09-14:

| Python | NumPy | ONNX Runtime | RDKit | Tests |
| --- | --- | --- | --- | --- |
| 3.9.25 | 2.0.2 | 1.19.2 | 2025.09.2 | 149 passed |
| 3.10.20 | 2.2.6 | 1.23.2 | 2026.03.6 | 149 passed + 49 subtests |
| 3.11.15 | 2.4.6 | 1.30.0 | 2026.03.6 | 149 passed + 49 subtests |
| 3.12.13 | 2.5.3 | 1.30.0 | 2026.03.6 | 149 passed + 49 subtests |
| 3.13.15 | 2.5.3 | 1.30.0 | 2026.03.6 | 149 passed + 49 subtests |
| 3.14.7 | 2.5.3 | 1.30.0 | 2026.03.6 | 149 passed + 49 subtests |

The piperidine MCA reference was exactly `503.25 kJ/mol` in every environment.
The largest cross-version difference in the fixed CBond reference output was
`5.96e-7`, below the test tolerance of `1e-6`.

The native RI search extension and its public search behavior were separately
revalidated on 2026-10-21 with clean CPython 3.9--3.14 builds and isolated
wheel installs. See the
[Stage 16 implementation report](../../../plan/16_native_search/implementation_report.md)
for the exact source-test, packaging and semantic matrix.

Run the complete matrix with [uv](https://docs.astral.sh/uv/):

```bash
bash tests/run_inference_compatibility.sh
```

Pass selected versions to shorten a local run:

```bash
bash tests/run_inference_compatibility.sh 3.9 3.14
```

The lower ONNX Runtime bound is intentionally 1.19. Python 3.9 resolves to
1.19.2, while newer Python versions resolve to newer compatible releases.

CI also builds a wheel, installs it without the source checkout on `sys.path`,
and runs a real MCA prediction plus public native-search smoke. This verifies
that the ONNX graph, its external weight files and the compiled search
extension are present in the distributable package.

As an additional forward-compatibility check, `python -m compileall hotpot`
completed under all six interpreters. Warnings emitted by third-party packages
do not affect the tested inference and graph-search paths.

The repository README still documents Python 3.9 as the requirement for the
legacy chemical kernel. Accordingly, this matrix is not a claim that the full
hotpot dependency stack and all legacy integration tests are ready for 3.14;
that migration should be tracked separately from MCA/CBond inference.
