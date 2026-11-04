# Molecular PyG CP311 checkpoint

Root accepts the molecular transport checkpoint on2026-11-05. Thin
`Molecule.to_pyg`/`from_pyg` now carry editable ordinary Data fields, exact H/stereo
meanings, constraints and historical observations through real batching,
DataLoader, save/load and a small training path. Superseded `to_pyg_data` and
collection prefix are removed. Deployed MCA/CBond encoders remain separate.

The independent root review found a concrete lossy discrete foreign-map case
(atomic number6.5 became6). Source-value comparison now rejects changed integer/
bool conversion, including overflow, while exact6.0 and0/1 remain valid. No
speculative finite-result guard or calculation is added.

CP311 source and actual outside-checkout pip-installed wheel each pass281tests,
with0failures/errors/skips. All185installed product origins are correct;
905installed and16support files stayed unchanged. The six baseline native payloads
are unchanged; the wheel is a repack, not a fresh native build.

The first cost campaign failed all four setups on an unsupported fixture keyword
and made0timed calls. It is retained. The corrected campaign passes4setups,
8warmups,28samples and4separate fresh-process memory probes. Complete export→Batch
→CPU→unbatch→fresh import medians are76.826ms (32×4sites),47.490ms (4×128sites),
41.291ms (512sites) and14.277ms (historical32sites). Traced peak465450–2960525B and
RSS increment2408–5372KiB are descriptive. Root accepts these usable costs; shared
host variation, native tensor allocation limits and all raw/failed observations
remain explicit. No old-export parity or leak claim follows.

[Checkpoint](molecular_checkpoint.json) indexes the reconstructible153-member
archive under`.plan20-artifacts/20261105/stage14/molecular-transport/`. It retains
scripts, exact argv/environment/input hashes, logs/JUnit, original/corrected
attempts, actual wheel and origin/immutability guards. All payloads were read
back and hashed. This is a molecular CP311 checkpoint: real GPU, other support
claims, Stage15/14D Crystal transport and overall integration remain open.
