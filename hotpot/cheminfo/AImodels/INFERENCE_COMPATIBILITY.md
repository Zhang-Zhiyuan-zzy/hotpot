# ONNX inference compatibility

The standalone MCA and CBond inference packages are tested on CPython
3.9--3.14. This scope covers their CPU inference paths, model loading, input
featurization, dynamic dimensions and stable reference predictions. It does
not claim that every unrelated hotpot subsystem or optional scientific package
supports every interpreter in the matrix.

Validated on 2026-09-14:

| Python | NumPy | ONNX Runtime | RDKit | Tests |
| --- | --- | --- | --- | --- |
| 3.9.25 | 2.0.2 | 1.19.2 | 2025.09.2 | 15 passed |
| 3.10.20 | 2.2.6 | 1.23.2 | 2026.03.6 | 15 passed |
| 3.11.15 | 2.4.6 | 1.30.0 | 2026.03.6 | 15 passed |
| 3.12.13 | 2.5.3 | 1.30.0 | 2026.03.6 | 15 passed |
| 3.13.15 | 2.5.3 | 1.30.0 | 2026.03.6 | 15 passed |
| 3.14.7 | 2.5.3 | 1.30.0 | 2026.03.6 | 15 passed |

The piperidine MCA reference was exactly `503.25 kJ/mol` in every environment.
The largest cross-version difference in the fixed CBond reference output was
`5.96e-7`, below the test tolerance of `1e-6`.

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

As an additional forward-compatibility check, `python -m compileall hotpot`
completed under all six interpreters. Python 3.12 and newer report pre-existing
warnings for invalid escape sequences in unrelated modules, and every version
reports the existing `hotpot/cheminfo/search/logic.py:300` annotation warning.
These warnings do not affect the two ONNX inference packages.

The repository README still documents Python 3.9 as the requirement for the
legacy chemical kernel. Accordingly, this matrix is not a claim that the full
hotpot dependency stack and all legacy integration tests are ready for 3.14;
that migration should be tracked separately from MCA/CBond inference.
