# Stage8D inference and conformer decisions

## Entry and invariant scope

Entry is pushed a511a1306bcb51f09c25a3bc2a478e14da749246, with Stage8C's CP311
search gate accepted. Two disjoint source inventories confirm the mixed owners
and connected consumers. Current goal, mainSections6/8/9.2/Stage8D and the
Plan17/20 alignment retain authority. Preserve the existing one MCA algorithm
(usable geometry or ETKDGv3+MMFF/UFF), model/feature/site meanings, CPU/GPU
threshold/window/session behavior, CBond rules and standalone wrappers. Existing
DTO names/fields remain; D08 candidate renames are not selected. No new record,
compatibility package/selector, model route, fallback or speculative guard.

## D1 — Artifact ownership and model command

Root selects the dependency-first artifact cutover before inference controllers.
Existing ArtifactSpec/ModelSource/ModelArtifact/errors, manifest workflow,
registry and resolver move once to operations/inference/artifacts. The selected
pointer manifests and their READMEs move byte-for-byte to that package's
`data/mca/` and `data/cbond/`; they are the one physical resource owner. Registry,
MCA ModelStore, CBond runtime, packaging and CI refer to those actual resources.
Do not create ArtifactManifest, copy weights, alter source/cache precedence,
change manifest schema/content or conflate completeness with hash verification.

adapters/inference/artifacts.py owns effect leaves: filesystem/JSON reads,
SHA/file-bundle hashing, file existence/count observations and pinned hub
snapshot download. It receives explicit values, imports no operations/manifest
DTO or carrier, and does not choose source/cache policy. Named operation
functions retain their current signatures, decision order, exact errors and
only the existing named missing-local-artifact translation. Local/hub download
arguments and external-file ordering remain unchanged. Internal helpers can
split without adding a second schema/implementation or making lower code call
an operation through a callback.

The existing model CLI moves to commands/models.py with unchanged commands,
flags, outputs and lazy dispatcher behavior. Delete old artifacts sources and
pointer files in the same cutover. MCA/CBond controllers stay at their old owners
until D2; only their exact artifact imports/resource lookups change in D1.
Current class metadata names new physical owners; no historical pickle shim.
The whole Stage8D one-GPU installed smoke remains mandatory because resources
move; D1 alone makes no Stage8D/GPU/final-delivery acceptance claim.

D1 validation: focused artifact/source-precedence/cache/integrity contracts,
CLI/lazy registry and resource/package guards; actual local CPU MCA and CBond
model loading through canonical pointers; architecture ratchet, bounded source
attribution and meaningful owner checks. Required whole-stage source/installed,
conformer/CPU/fake-GPU/one-GPU and later delivery gates remain open. Root freezes
and reviews source before execution; no author may start runtime/build/model or
measurement work during authoring. Failures stay attributable, with no silent
fallback or repeated unchanged passes.

## D2 — Connected inference split

This route is selected from the actual inventories. Authoring remains held until
D1's bounded source checkpoint is reviewed and committed. Preserve all existing
public constructor arguments, DTO fields/defaults, model inputs/outputs and
trained meanings; module metadata follows the new physical owner without shims.

| Owner | Responsibility |
| --- | --- |
| operations/inference/mca/ | Existing API, batching, pipeline, CPU/GPU controllers, workers, IPC records, resource/window/OOM policy and attachment operation. |
| operations/inference/mca/results.py | The three existing public prediction records and three version1 molecular transfer records, defined once. The existing DetectedSite remains a chemistry-rule value. |
| operations/inference/mca/serialization.py | Input dispatch, H/aromatic preparation, version checks, transfer-record construction and fresh Hotpot allocation. |
| operations/inference/mca/model_store.py | Existing ModelStore, ResolvedModelArtifact, resolved_model_artifact and resolving CPU/CUDA factories; artifact/materializer lifetime covers session creation and all CUDA ready acknowledgments. |
| operations/inference/mca/site_detection.py | Compile the24queries once, prepare one target/native task, run mapped sources and exact automorphism checks. |
| rules/chemistry/mca_sites/ | Ordered rule strings, DetectedSite, scalar eligibility, first-match naming/ordering, pure1-WL partition and existing target-domain checks. |
| adapters/inference/runtime.py | Existing MCARuntime, ORT loading/options/providers/input dtype conversion, and the already-resolved CUDA factory. Receives paths/variant/manifest/device/thread values, never ModelStore or operations imports. |
| adapters/inference/all_atom_model.py | Existing version2 ONNX transform/materialization, digest/cache locking/atomic publication/temp fallback and UnsupportedAllAtomOptimization; same bytes, decisions and lifetime semantics. |
| adapters/inference/features.py | Current RDKit-to-UniMol tensor/shortest-path/padding implementation. |
| adapters/inference/system.py and device.py | CPU affinity observation/pinning/pool initializer; GPUDeviceMemory, nvidia-smi/visible-device parsing and ordered hardware observations. Batch tiers, resource decisions and OOM interpretation stay operational. |
| operations/inference/features.py plus adapters/structure/inference.py | Existing shared data_extract signatures and preparation above; read-only carrier projections/model encodings below. |
| adapters/inference/serialization.py | Prepared RDKit copying/sanitization/binary encoding/decoding; consumes external values or bytes and imports no operation DTO. |
| adapters/rdkit/conformers.py | Existing sole MCA usable-geometry or ETKDGv3+MMFF/UFF implementation, without retries or algorithm substitution. |

Normalization retains the current branch order and lock: existing own-module DTO
passes through; otherwise convert to Hotpot, prepare implicit H, prepare/copy the
RDKit feature object, collect scalar/graph values and binary representation,
then construct the same DTO. Direct RDKit input is copied without the additional
Hotpot-projection sanitize branch; other inputs obtain the prepared external
object through the conversion operation before the external leaf. Encoding
leaves return values/bytes; they never create operation DTOs or call carrier
getters. Fresh reconstruction builds the same factual record through the
structure adapter, then the operation allocates the real Hotpot class. No change
to transported valence, isotope, absent/explicit-zero partial charge, stereo or
version1 behavior. Keep the existing private function signatures where called.

Shared feature extraction keeps the current atomic-number fast path, first15
fields plus metal bit, absent/stale charge-to-zero task encoding, pair field
order and ligand-skeleton cycle-basis feature family. Operations perform the
existing strict-H projection before lower extraction. Structure leaves accept
prepared scalar arrays/field names where required and may read existing carrier
views, but never import an operation, allocate/install a carrier, or prepare H
through a callback. Reuse ATOM_FIELDS for the existing Atom field order; use the
existing pair attr_names through the operation rather than duplicating a second
schema. Preserve public extraction signatures and caller data-dictionary
updates. Update CBond and PyG consumers together.

Rules receive explicit immutable values, not Molecule/Atom/native tasks or
operation callbacks. The operation materializes native mapping results and
carrier eligibility observations, then applies the ordered pure naming rules.
Pure1-WL receives keys/node labels/edges/edge labels and only prunes impossible
comparisons. Exact native automorphism remains the final equivalence test in
the operation, retaining the first representative and isotope-sensitive result.
Do not replace it by1-WL equality or recompile/repack per rule/molecule.

The lazy MCA package retains its nine-name inference facade and adds the one
relocated calculator mca entry; its signature still requires explicit backend.
Move calculator/mca_inference.py to operations/inference/mca/attachment.py and
remove the now-superseded calculator package. The existing structure/_results
attachment stays its sole lifecycle owner: normalize before capture, infer
outside lock, transactional exact payload selection. No per-Atom mirrors,
getter inference, full molecular hashing or new result schema.

## D2 — CBond, commands and deployment consumers

CBond apply/stage/constants move once to operations/inference/cbond. The existing
public CBondRuntime constructor resolves artifacts; moving that workflow into a
lower session adapter would create an upward dependency. Retain the same single
class and constructor at operations/inference/cbond/runtime.py as the workflow
facade. Session/provider creation, execution and tensor padding move as functions
to adapters/inference/cbond.py, receiving resolved paths/manifests/device or
existing sessions/arrays. No second runtime class/DTO or adapter policy callback
is added. Preserve graph/head sessions, actual CUDA checks, exposed runtime
fields, auto-vs-explicit device behavior and the raw-argument process-local
get_cbond_runtime cache. Operations own the cache and artifact/session lifetime;
no live session enters a molecular/process transfer record.

Keep CBond score thresholds/strict comparisons/ties, donor mapping, ordered
steps, reuse of existing bonds, path versus normalized structure probabilities,
log-space path merging and max_states unchanged. The stage still reports
is_Nuc_site distinctly from model scores. Actual CBond->FF handoff is a business
gate; mocked stage tests alone do not establish it.

Move CLI implementations to commands/mca.py and commands/cbond.py, with their
actual documentation resources under commands; preserve arguments, output text,
ordinal/domain-containment behavior and lazy dispatch. Update pipeline registry,
carrier delegates, package metadata, CI, tests, validation scripts, benchmarks
and docs to the canonical owners. Delete old AImodels inference/data_extract
sources in the connected cutover; no sys.path alias or old import package.
Prediction/model companion documents follow their operation owner.

Move deployment code once to tools/model_deployment/cbond outside importable
hotpot. Update its internal imports, two example consumers and documented tool
invocation; include the repository tools in sdist reproduction inputs, never as
an importable Hotpot wheel runtime. Do not execute training/export or add weights.

## D3 — Provider-neutral conformer preparation

Existing ConformerSamplingRequest/ConformerBatch/ConformerSamplingError and
ConformerSampler move to operations/conformers/contracts.py and sampler.py.
The current sample(Molecule, request) methods perform carrier preparation and
commit, so their two existing classes remain operation provider facades in
operations/conformers/openbabel.py and rdkit.py, preserving names/constructors.
External leaves remain in the named toolkit adapters; they receive prepared
native values/options and return external batch/report values, never a carrier
callback or a concrete Hotpot class import.

OpenBabel's default and injected builder route must reuse the existing _ob_build
operation, shared seed lifecycle lock, success/error contract and one coordinate
result, with no second build algorithm. The lower OpenBabel force-field adapter
already invokes the standalone component and imports no operation. RDKit's
operation owns the existing _materialized_hydrogen_projection, calls existing
field-selective packing once, then its adapter invokes the unchanged native
component on that prepared transport and options. Preserve exactly one requested
workflow conformer, forwarded seed, accepted status, lowest ordinal/position,
coordinate mapping/finite/error behavior and original cause. No new fallback,
retry, ensemble algorithm, duplicate materialized-H helper or standalone wrapper
rewrite. The selected returned numeric buffer must be independent and immutable;
readonly storage must not rely solely on a reversible ndarray WRITEABLE flag.

Update all FF/structure/benchmark/replay/import/pickle/provenance consumers
including the two literal sampler type strings. Record actual new class module
names; do not spoof historical identities. Remove only the two unused Molecule
_model/_environ initializers, with no replacement runtime fields or restriction
on unrelated arbitrary user metadata. Stage9 still owns historical asset policy.

## Connected review and execution boundary

Root owns main records, exact legacy-import-slot/debt/budget rebinding and the
small core/structure result import/message changes. Each author receives
exclusive files; one MCA/features writer and one CBond/conformer writer may
proceed independently only after explicit release. Tests/build/model execution
waits for the integrated source review/freeze, with original failures retained.
No new Python record classes/durable molecular fields are required.

Stage8D acceptance requires the complete affected CP311 source and outside-wheel
selection, actual CPU MCA/CBond and CBond->FF, fake-GPU threshold/window/session/
serialization coverage, conformer/seed/H/stereo/immutability evidence and the
real one-GPU installed smoke on ai-14. Bind model/native/wheel/input identities,
real CUDA provider/device, at least256selected sites and repeated predictor reuse.
A new dependency environment, a D1 checkpoint or fake-GPU tests passes no real
GPU gate. Broader native/SDK/sdist/version/global obligations remain explicit.
