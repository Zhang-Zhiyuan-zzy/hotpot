# MeCAP MCA ONNX model card

- Property: site-resolved methyl cation affinity (MCA)
- Unit: kJ/mol
- Architecture: Uni-Mol v2 84m with a single-atom regression head
- Release precision: FP16
- Inputs: Hotpot-normalized molecules converted to RDKit-derived Uni-Mol v2
  graph features, 3D coordinates and one target atom index per inference row
- Outputs: MCA for every heavy atom, plus a subset classified by 24 ordered
  nucleophilic-site rules through Hotpot's NetworkX search backend
- Dynamic limits: 1–4096 site rows and 2–512 heavy atoms

The model was trained and evaluated primarily in the neutral-molecule domain.
Charged molecules are rejected by the high-level API by default because earlier
analysis showed poor charged-species generalization. They can be evaluated only
with the explicit `allow_charged=True` opt-in and should be treated as
out-of-domain predictions.

Release parity against the originating PyTorch checkpoint was measured on all
62 strict-neutral Mayr comparison structures using the same stored conformers:

| Variant | RMSE (kJ/mol) | Maximum absolute error (kJ/mol) |
|---|---:|---:|
| FP32 | 0.000049 | 0.000153 |
| FP16 release | 0.087672 | 0.233032 |

The INT8 dynamic-quantization candidate was rejected (maximum error about
657.73 kJ/mol). It is not present in the public manifest and must not be used as
a release model.

MCA is not the Mayr nucleophilicity parameter `N` and is not `s_N N`.
