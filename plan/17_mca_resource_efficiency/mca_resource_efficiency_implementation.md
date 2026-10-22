# MCA Resource-Efficient Workflow Implementation Plan

## 1. Objective

Reduce MCA CPU consumption while preserving atom MCA values, ordered site
identity and type, output order, and `is_Nuc_site` semantics.

## 2. Validated decisions

- Standalone feature supply reached 27.07 molecules/s with two workers, but the
  production end-to-end comparison reached only 14.27 molecules/s with two and
  14.99 with three. Three workers are therefore the smallest allocation that
  reaches the measured 14.6--15.0 molecules/s A800 ceiling.
- One native-search site worker supplied 123.81 molecules/s on the 4,096
  molecule workload and does not constrain one GPU.
- The production GPU default is therefore one site worker plus three
  conformer/feature workers per GPU.
- CPU preprocessing uses one fused per-molecule task in the scientific order
  site detection, conformer generation, then feature extraction.
- CPU preprocessing and ONNX inference reuse the same bounded CPU set in
  phases. They do not own simultaneous disjoint pools and do not oversubscribe
  the configured total.
- `cpu_count` is the only public CPU resource control. The measured ai-14
  default is eight; explicit values remain available for other machines.

## 3. Public interface

```python
MCACpuPredictor(
    model_dir=None,
    *,
    variant: str | None = None,
    batch_size: int = 64,
    cpu_count: int | None = None,
    conformer_seed: int = 42,
    verify_model: bool = True,
    allow_charged: bool = False,
    model_source=None,
)
```

`cpu_count=None` resolves to at most eight CPUs from the current process
affinity. An explicit value must fit in that affinity. The CLI adds
`--cpu-count N`; it is CPU-only.

GPU defaults become:

```python
site_cpu_per_gpu = 1
infer_cpu_per_gpu = 3
```

## 4. CPU execution model

1. Normalize and validate a bounded input window in the coordinator.
2. Run fused preprocessing across molecules in a persistent spawn pool whose
   workers are individually pinned to the selected CPU IDs.
3. Consume worker results in input order and feed the unchanged global
   `LogicalBatchPlanner`.
4. Run CPU ONNX synchronously with the same selected CPU affinity and the same
   configured `cpu_count` as its intra-op thread count.
5. Scatter and emit results in input order.
6. Repeat without changing logical atom-row batch boundaries.

The preprocessing window contains at most 64 molecules and is reduced to the
logical batch size when that size is smaller. A one-molecule window executes
directly and does not start the process pool.

## 5. Semantic constraints

- Preserve deterministic conformer seeds.
- Preserve ordered MCA site-rule precedence and automorphic representative
  selection.
- Preserve global logical atom-row batches and padding widths.
- Preserve exact site-to-atom MCA equality and `is_Nuc_site` ownership.
- Preserve bounded generator consumption and single-predictor concurrency
  rejection.

## 6. Validation

1. Unit-test CPU resource resolution, ORT thread wiring, fused spawn workers,
   window bounds, affinity restoration, close behavior, and CLI routing.
2. Run all MCA tests and inference compatibility tests affected by the public
   signature.
3. On ai-14, compare the previous and optimized workflows on fixed simple and
   stratified workloads.
4. Record wall time, throughput, CPU allocation, and exact MCA/site/joint
   digests. The joint digest must include `is_Nuc_site` and site type.
5. Exercise the one-GPU path with the final 1/3 default only during an idle GPU
   window; do not use measurements contaminated by unrelated GPU work.

