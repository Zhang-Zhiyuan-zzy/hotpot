# SMARTS parser and matcher benchmarks

This is an explicit quick/heavy benchmark, not a default unit test. It measures
the following stages separately:

- tokenizer-only parsing and full query compilation;
- first matching with fresh, pre-built query/target objects;
- repeated matching with the same objects (Hotpot currently has no result
  cache, so this is labelled rather than described as a cached path);
- complete embedding enumeration on increasingly long symmetric chains;
- factorial raw-embedding growth for equally sized disconnected wildcard
  queries and carbon components;
- deep branches, nested recursive SMARTS, wide logical expressions, and a
  near-match whose final atom fails;
- linear batch scans;
- threaded scans using a shared `Searcher`, followed by a correctness check
  against the linear result; and
- concurrent query compilation followed by a correctness check against serial
  compilation.

Every repeated measurement reports raw samples, median, p90, min/max, and peak
Python allocation observed by `tracemalloc`. Scaling rows include the ratio to
the previous median. There is deliberately no millisecond pass threshold:
hardware and CI load would make one brittle. Determinism, equality between
linear/concurrent searches, and equality between serial/concurrent compilation
are correctness requirements and can return non-zero.

Quick benchmark:

```bash
python -m tests.smarts_conformance.benchmarks.benchmark_smarts \
  --output /tmp/hotpot-smarts-benchmark.json
```

Explicit heavier benchmark:

```bash
python -m tests.smarts_conformance.benchmarks.benchmark_smarts \
  --profile heavy --workers 8 \
  --output /tmp/hotpot-smarts-benchmark-heavy.json
```

The JSON records Python, platform, NetworkX, CPU count, worker count, exact
measurement scopes, distributions, memory, scale trends, and result checksums.
Checksums are stable SHA-256 digests rather than process-salted Python hashes.
