# Differential SMARTS audit

This optional heavy-profile runner compares Hotpot, RDKit, and Open Babel on a
small reviewed corpus. It records exact runtime/package versions, query and
target acceptance, match existence, query-order embeddings, engine-specific
unique mappings, and unique target-atom sets.

No engine, and no majority vote, is treated as the SMARTS specification.
`safe_intersection` cases are manually constrained to semantics that can be
aligned. Hotpot extensions and documented unsupported features are retained as
boundary evidence and are not promoted into the core dialect.

The report also contains a complete 118-element equivalence matrix. For every
periodic-table element it compares `[#{Z}]` with `[Symbol]` against the same
single-atom `[Symbol]` target, separately for each engine. Bare element tokens
are deliberately outside this matrix.

Run the audit and save all evidence:

```bash
python -m tests.smarts_conformance.differential.runner \
  --output /tmp/hotpot-smarts-differential.json
```

Audit mode exits successfully after recording disagreements. To use the
reviewed Tier-1 expectations as a gate, add `--strict`; project mismatches or a
manual-contract mismatch then produce a non-zero exit code. The runner is not
imported by ordinary core tests and never rewrites a golden corpus.

Atom indices are zero-based after normalization. Raw embeddings retain exact
duplicates and query-atom order; engine-unique embeddings and unordered target
atom sets are reported separately. Open Babel's one-based mappings are
converted explicitly. Target indices remain engine-local because independent
SMILES readers are not assumed to preserve shared atom numbering. Consequently
the runner compares acceptance, existence, and counts across engines, not the
embedding tuples themselves. Only raw-embedding and normalized target-set
counts participate in cross-engine classification; each engine's native
unique count is retained as diagnostic evidence because those uniquification
rules are not equivalent. RDKit is run with target
sanitization, `useChirality=False`, and both `uniquify=False` and `True`.
Open Babel is run with `single=False` and exposes both `GetMapList()` and
`GetUMapList()`. Hotpot exposes all NetworkX query mappings grouped by target
atom set.
