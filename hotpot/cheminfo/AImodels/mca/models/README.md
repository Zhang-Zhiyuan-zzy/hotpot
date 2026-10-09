# MCA model pointer

`manifest.json` pins the public MCA v1 bundle in
`Zhang-Zhiyuan-zzy/hotpot-models` to an immutable Hugging Face commit and
records the SHA-256 checksums used at runtime. Hotpot downloads the bundle on
first use and caches it outside the Python package.

Resolution order:

1. `MCAPredictor(model_dir=...)`
2. `HOTPOT_MCA_MODEL_DIR`
3. a complete development bundle in this directory
4. the verified local cache
5. the pinned Hugging Face revision when `model_source="auto"`

Set `model_source="local"` or `HOTPOT_MODEL_SOURCE=local` to prohibit network
access. Run `hotpot models install mca` to populate the cache explicitly.
