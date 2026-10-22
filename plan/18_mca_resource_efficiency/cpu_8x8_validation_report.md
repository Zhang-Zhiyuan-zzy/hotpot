# MCA 64-CPU (8 x 8) Validation Report

## Method

- Compared the pre-optimization CPU backend at commit `60288d2` with the
  optimized workflow at `cc6dc00`.
- Started eight persistent predictor instances and pinned each instance to a
  disjoint eight-CPU set.
- Warmed every instance with eight molecules before synchronized timing.
- Each timed instance predicted the same 64-molecule stratified workload, so
  every repeat processed 512 molecules in aggregate.
- Ran three timed repeats and reported the median aggregate throughput.
- Input SHA-256:
  `40eea8b16b8a9c451b918c0e7f3c49c0abb3c099d89493f3767d47ed044dcad5`.
- Compared every returned atom value and independently compared molecule
  order, atom identity/order, ordered site indices, site type, and
  `is_Nuc_site` semantics.

The current server used CPUs 0--63. ai-14 used CPUs 64--127.

## Throughput

| Server | CPU | Old backend | Current backend | Throughput change | Wall-time change |
|---|---|---:|---:|---:|---:|
| Current server | AMD Threadripper PRO 3995WX | 17.035 mol/s | 19.210 mol/s | +12.8% | -11.3% |
| ai-14 | AMD EPYC 7763 | 17.488 mol/s | 20.439 mol/s | +16.9% | -14.4% |

Median wall time for 512 molecules fell from 30.055 to 26.653 seconds on the
current server and from 29.277 to 25.050 seconds on ai-14.

The eight instances produced one identical prediction digest within each run,
and each digest remained stable across all three repeats.

## Prediction error against the old backend

| Server | Compared atom outputs | Maximum absolute error | MAE | RMSE | Exact values | Site semantics |
|---|---:|---:|---:|---:|---:|---|
| Current server | 12,120 | 0.0 kJ/mol | 0.0 | 0.0 | 12,120 / 12,120 | identical |
| ai-14 | 12,120 | 0.0 kJ/mol | 0.0 | 0.0 | 12,120 / 12,120 | identical |

The optimized workflow therefore introduces no numerical or business-semantic
error relative to the old workflow in either environment.

## Cross-server observation

Cross-server comparison is separate from the old/new workflow comparison. The
current builds on the two hosts retained identical output structure and site
semantics, but their atom MCA values were not bit-identical:

- maximum absolute difference: 17.25 kJ/mol;
- MAE: 1.3129 kJ/mol;
- RMSE: 2.5764 kJ/mol.

Both hosts used the same FP16 model manifest digest and ONNX Runtime 1.31.0,
but their chemistry/runtime environments differ: the current server used
RDKit 2026.09.1, Python 3.11.16, and NumPy 2.4.6; ai-14 used RDKit 2026.03.6,
Python 3.13.15, and NumPy 2.5.3. MCA conformer generation depends on RDKit, so
cross-host numerical reproducibility requires pinning the complete environment,
especially RDKit, rather than only the model and Hotpot revision. This does
not affect the conclusion above: on each host, old and current Hotpot produced
exactly the same predictions under the same environment.

## Artifacts

- Reproducible runner: `benchmark_cpu_8x8.py`.
- Current-server raw files: `/tmp/mca_8x8_cc6dc00_local`.
- ai-14 raw files: `~/mca_test/results/cpu_8x8_cc6dc00`.
