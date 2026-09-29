# Stage 09: native Open Babel force fields

## Records

- [`native_openbabel_forcefields_implementation.md`](native_openbabel_forcefields_implementation.md): typed NumPy buffer boundary, C++ engine, rule framework, and packaging decisions.
- [`native_openbabel_forcefields_test_report.md`](native_openbabel_forcefields_test_report.md): regular, cross-version, wheel, and 187-molecule validation evidence.

## Related Git commits

Historical Python/SWIG rule-engine baseline retained only as Git evidence:

```text
1b904f7 0e1715a d32823d 493bc96 fab2e5a 503f2a5
```

Production typed-buffer/C++ migration:

```text
7e2341d 1e00af7 c758229 de3a5c9 082bfe5 a3de55b cb59294
a227df8 ede42bb 633f1d4 1d603f0 24f954f 16f4819 b68b10e
3ac1f68 3755636 b1bfe09 50b458d 2d42ef6
```

Legacy-label documentation cleanup: `9350ee0`.
