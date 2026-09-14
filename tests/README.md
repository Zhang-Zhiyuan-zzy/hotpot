# Test suite

All repository tests and test-only fixtures live under this directory. Runtime
packages under `hotpot/` must not contain test modules or test data.

- `mca/`: MCA ONNX inference tests
- `cbond/`: dynamic CBond ONNX inference tests
- `fixtures/`: test-only binary fixtures
- `cpp/`: C++ tests
- `test_cheminfo/`: core molecule, conversion, search and calculator tests
- `test_main/`, `test_plugin/`: legacy integration tests

Run the Python 3.9--3.14 inference matrix with:

```bash
bash tests/run_inference_compatibility.sh
```

The script installs the isolated dependency set in
`tests/requirements-inference.txt` and covers the Hotpot integration required
by MCA, rather than only importing the model subpackages.
