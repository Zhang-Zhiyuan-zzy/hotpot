# Model artifacts

The private exporter writes `manifest.json` and the ONNX artifacts here during
release assembly. Runtime model lookup order is:

1. `MCAPredictor(model_dir=...)`
2. environment variable `HOTPOT_MCA_MODEL_DIR`
3. this packaged directory

Include `mecap_mca_fp16.onnx` and every adjacent `model.*.weight` file for both
CPU and GPU deployment. The standard ONNX external-data layout keeps every Git
object below GitHub's 100 MB limit without changing inference results. INT8 is
not a release artifact because validation showed an unacceptable numerical
shift. The manifest verifies both the graph and the complete weight bundle.
