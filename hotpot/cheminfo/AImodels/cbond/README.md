# Coordination-bond inference

The production runtime uses two ONNX files in `onnx/`: a dynamic molecular
graph encoder and a dynamic coordination-bond head. Model files are loaded only
when inference is first requested, and their SHA-256 hashes are checked against
`onnx/manifest.json`.

Set `HOTPOT_CBOND_DEVICE=cpu` or `HOTPOT_CBOND_DEVICE=cuda` to require a device.
The default `auto` setting selects an initialized CUDA execution provider when
available and otherwise uses CPU.

The supported domain is at most 32 ligand rings with at most 64 atoms in one
ring. Inputs beyond those limits raise an error.

## Rebuild the dynamic head

Install the export dependency, check out the historical fixed-shape release in
a separate directory, and run the converter from this branch:

```bash
python -m pip install '.[onnx-export]'
git worktree add /tmp/hotpot-cbond-fixed origin/main
python hotpot/cheminfo/AImodels/cbond/deploy/make_dynamic_onnx.py \
  --matrix-dir /tmp/hotpot-cbond-fixed/hotpot/cheminfo/AImodels/cbond/onnx \
  --source '/tmp/hotpot-cbond-fixed/hotpot/cheminfo/AImodels/cbond/onnx/opset21_cbond(2-6).onnx' \
  --output /tmp/opset21_cbond_dynamic.onnx
```

The converter first verifies that every fixed-shape source contains the same
parameter set. The released dynamic head was compared with all 28 historical
graphs and with all 10 bundled CBond fixtures; numerical results are recorded
in `onnx/manifest.json`.
