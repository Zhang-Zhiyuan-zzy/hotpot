# Test suites

The default suite is deterministic, bounded, and independent of separately
installed executables:

```bash
python -m pytest -q --import-mode=importlib tests
```

Five explicit switches add tests whose execution contract is different from
the default suite:

| Switch | Purpose |
|---|---|
| `--run-scientific-boundary` | Governed scientific corpora and published evidence. |
| `--run-high-compute` | Long-running CPU/GPU chemistry and full-corpus recomputation. |
| `--run-external` | Official xTB, Gaussian, Zeo++, live RDL, or another separately installed program. |
| `--run-optional` | Features installed through an optional dependency group. |
| `--run-model-artifacts` | Exact inference examples that require preinstalled Hotpot model artifacts. |

Markers compose. The complete 187/182 extractant recomputation, for example,
requires `--run-scientific-boundary`, `--run-high-compute`, and
`--run-model-artifacts`. The model-artifact layer retains a bounded one-case
version of that workflow; CI enables it after installing the pinned models.
The governed corpus and stored scientific evidence require the
scientific-boundary switch.

Native C/C++ source-compilation contract tests are also high-compute. Default
tests exercise the already built native modules; enable `--run-high-compute`
to compile every standalone header/source contract.

Published README and documentation examples are read from their Markdown
source by the tests. A `Verified by` annotation must resolve to a collected
test function. Examples with external executables use the external toggle;
inference examples that need downloaded model weights use the model-artifact
toggle. The inference compatibility job installs those pinned artifacts before
enabling the toggle.

No test may write to a tracked output directory. Use `tmp_path` or an external
benchmark output selected explicitly by the caller.

## CI entry points

The coverage job runs the complete default suite on Python 3.11:

```bash
bash tests/run_coverage.sh
```

The inference compatibility job exercises the supported Python 3.10–3.14
matrix. Pass versions explicitly to narrow a local run:

```bash
bash tests/run_inference_compatibility.sh 3.11
```

Set `HOTPOT_TEST_WHEEL` to validate an installed wheel instead of the checkout.
`HOTPOT_TEST_WORKDIR` may point at any writable temporary directory; the runner
constructs its import and Matplotlib paths there and does not rely on a fixed
machine directory. The CI coverage job sets
`HOTPOT_RUN_MODEL_ARTIFACT_TESTS=1` after installing the pinned MCA and cbond
artifacts; the same variable enables that layer through `run_coverage.sh`.
