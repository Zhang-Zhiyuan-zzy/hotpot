# CBond ONNX artifacts

Runtime inference uses two graphs:

- `opset21_graph.onnx` embeds the molecular graph with dynamic node and edge
  dimensions.
- `opset21_cbond_dynamic.onnx` predicts coordination-bond scores with dynamic
  ring-count and ring-size dimensions.

The dynamic head replaces 28 fixed-shape copies of the same parameters. It was
checked against every former model with a maximum absolute output difference of
`2.39e-6` and an aggregate RMSE of `6.89e-7`.

`deploy/make_dynamic_onnx.py` reproduces the dynamic graph from the historical
fixed-shape matrix and verifies that all source files carry the same weights.
The SHA-256 values and supported domain limits are recorded in `manifest.json`.
