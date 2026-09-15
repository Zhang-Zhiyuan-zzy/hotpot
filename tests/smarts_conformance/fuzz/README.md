# Deterministic SMARTS robustness audit

The runner has four independent streams:

1. bounded grammar-generated valid SMARTS, which must compile and match safely;
2. a named, local corruption applied to that iteration's valid seed, which
   must be rejected;
3. arbitrary text, for which acceptance is unspecified but crashes, timeouts,
   and non-determinism are failures; and
4. bounded structured stress queries covering deep branches, nested recursive
   SMARTS, mixed Boolean expressions, and multiple disconnected rings, which
   must compile and execute deterministically.

The explicit top-level seed deterministically derives a separate seed for each
stream, so changing one generator cannot silently shift the other streams.
Every input executes twice. `ValueError` and `NotImplementedError` are
recognized parser rejection outcomes. Other exceptions are reported as crashes
rather than hidden behind a catch-all fallback. Per-evaluation Unix timers
bound parser and matcher work. The grammar generator has a separate atom-count
bound from the structured stream's depth bound.

Quick audit:

```bash
python -m tests.smarts_conformance.fuzz.run_deterministic \
  --seed 20260915 \
  --output /tmp/hotpot-smarts-fuzz.json
```

Longer explicit heavy run:

```bash
python -m tests.smarts_conformance.fuzz.run_deterministic \
  --profile heavy \
  --seed 20260915 \
  --output /tmp/hotpot-smarts-fuzz-heavy.json
```

Any invalid acceptance, valid rejection, crash, timeout, or non-deterministic
outcome returns a non-zero exit code. The JSON report retains the exact seed,
valid base query, mutation name, unique-input counts, mutated input, phase
outcome, and diagnostic needed for minimization. A
confirmed finding should be reduced and moved into the stable regression
corpus; generated results are never written into that corpus automatically.
Mutation failures also retain the zero-based edit position and inserted
payload, so the exact generated case can be reconstructed without guessing.
