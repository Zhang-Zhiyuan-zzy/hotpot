# Scientific-boundary suite

This directory is the single pytest entry point for governed scientific
boundaries. The current reference boundary is the immutable 187-extractant
source and its 182 CBond-positive coordination complexes.

Run the governed corpus and stored-evidence checks with:

```bash
python -m pytest -q --import-mode=importlib \
  --run-scientific-boundary tests/scientific_boundary
```

Recompute the complete current 187/182 workflow only on an explicitly
allocated host:

```bash
python -m pytest -q --import-mode=importlib \
  --run-scientific-boundary --run-high-compute --run-model-artifacts \
  tests/scientific_boundary/test_extractant_workflow.py
```

The model-artifact test layer separately runs one governed case with reduced
optimization budgets to cover the same plumbing without a high-compute
allocation. CI enables that layer after installing the pinned MCA and cbond
models. It is not scientific evidence and cannot replace the full
standard-settings result. The full test evaluates all 187 inputs and requires
182 accepted complex structures plus the five governed no-CBond cases.
