# MCA、CBond、xTB 与 pipeline：全签名目录

来源：当前静态源码；显式 __all__ 与直接模块定义分别列明。类型定义 owner 与公开导出路径分别保留。
只读 property、动态 owner setter、不可变值字段与可重绑定类声明值不是同一权限概念。外部继承保持可归属的边界，未声称第三方运行时属性穷尽。
[完整树](../delivery_api_tree.md) · [动态权限](dynamic_permissions.md) · [native 声明](native_contracts.md)

## 模块导航

- [`hotpot.cheminfo.operations.inference`](#hotpot-cheminfo-operations-inference)
- [`hotpot.cheminfo.operations.inference.artifacts`](#hotpot-cheminfo-operations-inference-artifacts)
- [`hotpot.cheminfo.operations.inference.artifacts.errors`](#hotpot-cheminfo-operations-inference-artifacts-errors)
- [`hotpot.cheminfo.operations.inference.artifacts.manifest`](#hotpot-cheminfo-operations-inference-artifacts-manifest)
- [`hotpot.cheminfo.operations.inference.artifacts.registry`](#hotpot-cheminfo-operations-inference-artifacts-registry)
- [`hotpot.cheminfo.operations.inference.artifacts.resolver`](#hotpot-cheminfo-operations-inference-artifacts-resolver)
- [`hotpot.cheminfo.operations.inference.cbond`](#hotpot-cheminfo-operations-inference-cbond)
- [`hotpot.cheminfo.operations.inference.cbond.apply`](#hotpot-cheminfo-operations-inference-cbond-apply)
- [`hotpot.cheminfo.operations.inference.cbond.constants`](#hotpot-cheminfo-operations-inference-cbond-constants)
- [`hotpot.cheminfo.operations.inference.cbond.runtime`](#hotpot-cheminfo-operations-inference-cbond-runtime)
- [`hotpot.cheminfo.operations.inference.cbond.stage`](#hotpot-cheminfo-operations-inference-cbond-stage)
- [`hotpot.cheminfo.operations.inference.features`](#hotpot-cheminfo-operations-inference-features)
- [`hotpot.cheminfo.operations.inference.mca`](#hotpot-cheminfo-operations-inference-mca)
- [`hotpot.cheminfo.operations.inference.mca.api`](#hotpot-cheminfo-operations-inference-mca-api)
- [`hotpot.cheminfo.operations.inference.mca.attachment`](#hotpot-cheminfo-operations-inference-mca-attachment)
- [`hotpot.cheminfo.operations.inference.mca.batching`](#hotpot-cheminfo-operations-inference-mca-batching)
- [`hotpot.cheminfo.operations.inference.mca.cpu_backend`](#hotpot-cheminfo-operations-inference-mca-cpu-backend)
- [`hotpot.cheminfo.operations.inference.mca.gpu_backend`](#hotpot-cheminfo-operations-inference-mca-gpu-backend)
- [`hotpot.cheminfo.operations.inference.mca.gpu_consumer`](#hotpot-cheminfo-operations-inference-mca-gpu-consumer)
- [`hotpot.cheminfo.operations.inference.mca.gpu_memory`](#hotpot-cheminfo-operations-inference-mca-gpu-memory)
- [`hotpot.cheminfo.operations.inference.mca.model_store`](#hotpot-cheminfo-operations-inference-mca-model-store)
- [`hotpot.cheminfo.operations.inference.mca.pipeline`](#hotpot-cheminfo-operations-inference-mca-pipeline)
- [`hotpot.cheminfo.operations.inference.mca.resources`](#hotpot-cheminfo-operations-inference-mca-resources)
- [`hotpot.cheminfo.operations.inference.mca.results`](#hotpot-cheminfo-operations-inference-mca-results)
- [`hotpot.cheminfo.operations.inference.mca.serialization`](#hotpot-cheminfo-operations-inference-mca-serialization)
- [`hotpot.cheminfo.operations.inference.mca.site_detection`](#hotpot-cheminfo-operations-inference-mca-site-detection)
- [`hotpot.cheminfo.operations.inference.mca.workers`](#hotpot-cheminfo-operations-inference-mca-workers)
- [`hotpot.pipeline`](#hotpot-pipeline)
- [`hotpot.pipeline._sample_control`](#hotpot-pipeline--sample-control)
- [`hotpot.pipeline.artifacts`](#hotpot-pipeline-artifacts)
- [`hotpot.pipeline.cli`](#hotpot-pipeline-cli)
- [`hotpot.pipeline.contracts`](#hotpot-pipeline-contracts)
- [`hotpot.pipeline.registry`](#hotpot-pipeline-registry)
- [`hotpot.pipeline.runner`](#hotpot-pipeline-runner)
- [`hotpot.pipeline.state_recovery`](#hotpot-pipeline-state-recovery)
- [`hotpot.plugins.xtb`](#hotpot-plugins-xtb)
- [`hotpot.plugins.xtb.adapter`](#hotpot-plugins-xtb-adapter)
- [`hotpot.plugins.xtb.backend`](#hotpot-plugins-xtb-backend)
- [`hotpot.plugins.xtb.capabilities`](#hotpot-plugins-xtb-capabilities)
- [`hotpot.plugins.xtb.cli`](#hotpot-plugins-xtb-cli)
- [`hotpot.plugins.xtb.contracts`](#hotpot-plugins-xtb-contracts)
- [`hotpot.plugins.xtb.runner`](#hotpot-plugins-xtb-runner)
- [`hotpot.plugins.xtb.stage`](#hotpot-plugins-xtb-stage)
- [`hotpot.plugins.xtb.state_projection`](#hotpot-plugins-xtb-state-projection)
- [`hotpot.plugins.xtb.state_resolution`](#hotpot-plugins-xtb-state-resolution)
- [`hotpot.plugins.xtb.stream`](#hotpot-plugins-xtb-stream)
- [`hotpot.plugins.xtb.workflow`](#hotpot-plugins-xtb-workflow)

<a id="hotpot-cheminfo-operations-inference"></a>
## `hotpot.cheminfo.operations.inference`

`hotpot/cheminfo/operations/inference/__init__.py`；显式 __all__: ()



<a id="hotpot-cheminfo-operations-inference-artifacts"></a>
## `hotpot.cheminfo.operations.inference.artifacts`

`hotpot/cheminfo/operations/inference/artifacts/__init__.py`；显式 __all__: ('ModelArtifact', 'ModelArtifactUnavailable', 'ModelSource', 'load_manifest', 'verify_artifact')

- 导出 `ModelArtifact` → `hotpot.cheminfo.operations.inference.artifacts.resolver.ModelArtifact`
- 导出 `ModelArtifactUnavailable` → `hotpot.cheminfo.operations.inference.artifacts.errors.ModelArtifactUnavailable`
- 导出 `ModelSource` → `hotpot.cheminfo.operations.inference.artifacts.resolver.ModelSource`
- 导出 `load_manifest` → `hotpot.cheminfo.operations.inference.artifacts.manifest.load_manifest`；`load_manifest(path: Path) -> dict`
- 导出 `verify_artifact` → `hotpot.cheminfo.operations.inference.artifacts.manifest.verify_artifact`；`verify_artifact(model_dir: Path, manifest: dict) -> None`


<a id="hotpot-cheminfo-operations-inference-artifacts-errors"></a>
## `hotpot.cheminfo.operations.inference.artifacts.errors`

`hotpot/cheminfo/operations/inference/artifacts/errors.py`；显式 __all__: 无


<a id="hotpot-cheminfo-operations-inference-artifacts-errors-modelartifactunavailable"></a>
### `ModelArtifactUnavailable`

定义 owner `hotpot.cheminfo.operations.inference.artifacts.errors.ModelArtifactUnavailable`；基类 `FileNotFoundError`。



<a id="hotpot-cheminfo-operations-inference-artifacts-manifest"></a>
## `hotpot.cheminfo.operations.inference.artifacts.manifest`

`hotpot/cheminfo/operations/inference/artifacts/manifest.py`；显式 __all__: 无


<a id="hotpot-cheminfo-operations-inference-artifacts-manifest-load-manifest"></a>
### `load_manifest`

```python
load_manifest(path: Path) -> dict
```

<a id="hotpot-cheminfo-operations-inference-artifacts-manifest-sha256"></a>
### `sha256`

```python
sha256(path: Path) -> str
```

<a id="hotpot-cheminfo-operations-inference-artifacts-manifest-bundle-sha256"></a>
### `bundle_sha256`

```python
bundle_sha256(paths: list[Path]) -> str
```

<a id="hotpot-cheminfo-operations-inference-artifacts-manifest-artifact-is-complete"></a>
### `artifact_is_complete`

```python
artifact_is_complete(model_dir: Path, manifest: dict) -> bool
```

<a id="hotpot-cheminfo-operations-inference-artifacts-manifest-verify-artifact"></a>
### `verify_artifact`

```python
verify_artifact(model_dir: Path, manifest: dict) -> None
```


<a id="hotpot-cheminfo-operations-inference-artifacts-registry"></a>
## `hotpot.cheminfo.operations.inference.artifacts.registry`

`hotpot/cheminfo/operations/inference/artifacts/registry.py`；显式 __all__: 无


<a id="hotpot-cheminfo-operations-inference-artifacts-registry-artifactspec"></a>
### `ArtifactSpec`

定义 owner `hotpot.cheminfo.operations.inference.artifacts.registry.ArtifactSpec`；基类 无。

```python
ArtifactSpec(name: str, manifest_path: Path, environment_variable: str)
```

- `name: str` [只读值字段]
- `manifest_path: Path` [只读值字段]
- `environment_variable: str` [只读值字段]

模块值/类型别名（声明存在不构成事实写许可）：

- `ARTIFACTS` = `{'mca': ArtifactSpec('mca', _ARTIFACT_DATA_DIR / 'mca' / 'manifest.json', 'HOTPOT_MCA_MODEL_DIR'), 'cbond': ArtifactSpec('cbond', _ARTIFACT_DATA_DIR / 'cbond' / 'manifest.json', 'HOTPOT_CBOND_MODEL_DIR')}`


<a id="hotpot-cheminfo-operations-inference-artifacts-resolver"></a>
## `hotpot.cheminfo.operations.inference.artifacts.resolver`

`hotpot/cheminfo/operations/inference/artifacts/resolver.py`；显式 __all__: 无


<a id="hotpot-cheminfo-operations-inference-artifacts-resolver-modelsource"></a>
### `ModelSource`

定义 owner `hotpot.cheminfo.operations.inference.artifacts.resolver.ModelSource`；基类 `str`, `Enum`。

- `AUTO` = `'auto'` [枚举常量]
- `LOCAL` = `'local'` [枚举常量]
- `HUB` = `'hub'` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-cheminfo-operations-inference-artifacts-resolver-modelartifact"></a>
### `ModelArtifact`

定义 owner `hotpot.cheminfo.operations.inference.artifacts.resolver.ModelArtifact`；基类 无。

- `__init__(self, pointer_manifest: PathLike, model_env_var: str, model_dir: Optional[PathLike]=None, source: Optional[Union[str, ModelSource]]=None)`
- `resolve(self) -> Path`
- `pointer_path` = `Path(pointer_manifest)` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `pointer` = `load_manifest(self.pointer_path)` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `packaged_dir` = `self.pointer_path.parent` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `configured_dir` = `Path(configured).expanduser() if configured else None` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `source` = `ModelSource(selected_source)` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）

模块值/类型别名（声明存在不构成事实写许可）：

- `PathLike` = `Union[str, os.PathLike]`


<a id="hotpot-cheminfo-operations-inference-cbond"></a>
## `hotpot.cheminfo.operations.inference.cbond`

`hotpot/cheminfo/operations/inference/cbond/__init__.py`；显式 __all__: ('CBondRuntime',)

- 导出 `CBondRuntime` → `hotpot.cheminfo.operations.inference.cbond.runtime.CBondRuntime`


<a id="hotpot-cheminfo-operations-inference-cbond-apply"></a>
## `hotpot.cheminfo.operations.inference.cbond.apply`

`hotpot/cheminfo/operations/inference/cbond/apply.py`；显式 __all__: 无


<a id="hotpot-cheminfo-operations-inference-cbond-apply-cbondstep"></a>
### `CBondStep`

定义 owner `hotpot.cheminfo.operations.inference.cbond.apply.CBondStep`；基类 无。

```python
CBondStep(atom_index: int, element: str, score: float, probability: float)
```

- `atom_index: int` [只读值字段]
- `element: str` [只读值字段]
- `score: float` [只读值字段]
- `probability: float` [只读值字段]

<a id="hotpot-cheminfo-operations-inference-cbond-apply-cbondpathresult"></a>
### `CBondPathResult`

定义 owner `hotpot.cheminfo.operations.inference.cbond.apply.CBondPathResult`；基类 无。

```python
CBondPathResult(molecule: Molecule, steps: tuple[CBondStep, ...], path_probability: float, donor_indices: tuple[int, ...])
```

- `molecule: Molecule` [只读值字段]
- `steps: tuple[CBondStep, ...]` [只读值字段]
- `path_probability: float` [只读值字段]
- `donor_indices: tuple[int, ...]` [只读值字段]

<a id="hotpot-cheminfo-operations-inference-cbond-apply-cbondstructureresult"></a>
### `CBondStructureResult`

定义 owner `hotpot.cheminfo.operations.inference.cbond.apply.CBondStructureResult`；基类 无。

```python
CBondStructureResult(molecule: Molecule, probability: float, steps: tuple[CBondStep, ...], donor_indices: tuple[int, ...], path_count: int, path_weight: float, log_structure_weight: float)
```

- `molecule: Molecule` [只读值字段]
- `probability: float` [只读值字段]
- `steps: tuple[CBondStep, ...]` [只读值字段]
- `donor_indices: tuple[int, ...]` [只读值字段]
- `path_count: int` [只读值字段]
- `path_weight: float` [只读值字段]
- `log_structure_weight: float` [只读值字段]

<a id="hotpot-cheminfo-operations-inference-cbond-apply-get-cbond-runtime"></a>
### `get_cbond_runtime`

```python
get_cbond_runtime(device: str=None, model_dir: str=None, model_source: str=None) -> CBondRuntime
```

<a id="hotpot-cheminfo-operations-inference-cbond-apply-extract-cbond-inputs"></a>
### `extract_cbond_inputs`

```python
extract_cbond_inputs(mol: Molecule) -> dict[str, Any]
```

<a id="hotpot-cheminfo-operations-inference-cbond-apply-get-graph-cbond-inputs"></a>
### `get_graph_cbond_inputs`

```python
get_graph_cbond_inputs(data: dict[str, Any])
```

<a id="hotpot-cheminfo-operations-inference-cbond-apply-get-cbond-inputs-model"></a>
### `get_cbond_inputs_model`

```python
get_cbond_inputs_model(data: dict[str, Any], xg)
```

<a id="hotpot-cheminfo-operations-inference-cbond-apply-pred-xg"></a>
### `pred_xg`

```python
pred_xg(mol_data: dict[str, Any], runtime: CBondRuntime=None)
```

<a id="hotpot-cheminfo-operations-inference-cbond-apply-pred-cb-value"></a>
### `pred_cb_value`

```python
pred_cb_value(xg, padded_xr, rings_mask, cbond_index, runtime: CBondRuntime=None)
```

<a id="hotpot-cheminfo-operations-inference-cbond-apply-cbond-prediction"></a>
### `cbond_prediction`

```python
cbond_prediction(mol_data: dict[str, Any], runtime: CBondRuntime=None)
```

<a id="hotpot-cheminfo-operations-inference-cbond-apply-signmod-with-offset"></a>
### `signmod_with_offset`

```python
signmod_with_offset(x, offset: float=0.0)
```

<a id="hotpot-cheminfo-operations-inference-cbond-apply-init-metal-ligand-pair"></a>
### `init_metal_ligand_pair`

```python
init_metal_ligand_pair(mol: Molecule, metal: Union[int, str, Atom])
```

<a id="hotpot-cheminfo-operations-inference-cbond-apply-auto-build-cbond"></a>
### `auto_build_cbond`

```python
auto_build_cbond(mol: Molecule, metal: Union[int, str, Atom], threshold: Optional[float]=None, greedy: bool=True, sum_prob: bool=True, runtime: CBondRuntime=None, *, first_threshold: Optional[float]=None, return_details: bool=False)
```

<a id="hotpot-cheminfo-operations-inference-cbond-apply-build-one-cbond"></a>
### `build_one_cbond`

```python
build_one_cbond(mol: Molecule, metal: Union[int, str, Atom], threshold: float=DEFAULT_FIRST_CBOND_THRESHOLD, get_all: bool=False, runtime: CBondRuntime=None)
```

<a id="hotpot-cheminfo-operations-inference-cbond-apply-build-all-possible-cbond"></a>
### `build_all_possible_cbond`

```python
build_all_possible_cbond(mol: Molecule, m: Union[int, str, Atom], threshold: Optional[float]=None, greedy: bool=True, normalize_prob: bool=True, runtime: CBondRuntime=None, *, first_threshold: Optional[float]=None, max_states: int=DEFAULT_MAX_STATES, return_details: bool=False)
```


<a id="hotpot-cheminfo-operations-inference-cbond-constants"></a>
## `hotpot.cheminfo.operations.inference.cbond.constants`

`hotpot/cheminfo/operations/inference/cbond/constants.py`；显式 __all__: 无

- 导出 `DEFAULT_CBOND_THRESHOLD` → `hotpot.cheminfo.operations.inference.cbond.constants.DEFAULT_SUBSEQUENT_CBOND_THRESHOLD`

模块值/类型别名（声明存在不构成事实写许可）：

- `DEFAULT_FIRST_CBOND_THRESHOLD` = `-0.5`
- `DEFAULT_SUBSEQUENT_CBOND_THRESHOLD` = `-0.125`
- `DEFAULT_CBOND_THRESHOLD` = `DEFAULT_SUBSEQUENT_CBOND_THRESHOLD`
- `DEFAULT_MAX_STATES` = `4096`


<a id="hotpot-cheminfo-operations-inference-cbond-runtime"></a>
## `hotpot.cheminfo.operations.inference.cbond.runtime`

`hotpot/cheminfo/operations/inference/cbond/runtime.py`；显式 __all__: 无


<a id="hotpot-cheminfo-operations-inference-cbond-runtime-cbondruntime"></a>
### `CBondRuntime`

定义 owner `hotpot.cheminfo.operations.inference.cbond.runtime.CBondRuntime`；基类 无。

- `__init__(self, model_dir=None, device: str='auto', verify: bool=True, model_source=None)`
- `embed_graph(self, x, edge_index)`
- `predict(self, xg, padded_rings, rings_mask, cbond_index)`
- `model_dir` = `ModelArtifact(pointer, 'HOTPOT_CBOND_MODEL_DIR', model_dir=model_dir, source=model_source).resolve()` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `manifest` = `load_manifest(self.model_dir / 'manifest.json')` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `requested_device` = `self._resolve_device(device)` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）


<a id="hotpot-cheminfo-operations-inference-cbond-stage"></a>
## `hotpot.cheminfo.operations.inference.cbond.stage`

`hotpot/cheminfo/operations/inference/cbond/stage.py`；显式 __all__: ('STAGE', 'get_stage')


<a id="hotpot-cheminfo-operations-inference-cbond-stage-get-stage"></a>
### `get_stage`

```python
get_stage() -> MolecularStage
```

模块值/类型别名（声明存在不构成事实写许可）：

- `STAGE` = `_CBondStage()`


<a id="hotpot-cheminfo-operations-inference-features"></a>
## `hotpot.cheminfo.operations.inference.features`

`hotpot/cheminfo/operations/inference/features.py`；显式 __all__: 无


<a id="hotpot-cheminfo-operations-inference-features-extract-potentials-cbonds"></a>
### `extract_potentials_cbonds`

```python
extract_potentials_cbonds(mol: Molecule, catoms_symbols: Optional[Iterable[str]]=None, data: dict[str, np.ndarray]=None) -> dict[str, np.ndarray]
```

<a id="hotpot-cheminfo-operations-inference-features-extract-atom-attrs"></a>
### `extract_atom_attrs`

```python
extract_atom_attrs(mol: Molecule, data: dict[str, Any]=None, atomic_number_only: bool=False) -> dict
```

<a id="hotpot-cheminfo-operations-inference-features-extract-bond-attrs"></a>
### `extract_bond_attrs`

```python
extract_bond_attrs(mol: Molecule, edge_attr_names: Iterable[str]=None, data: dict[str, Any]=None) -> dict
```

<a id="hotpot-cheminfo-operations-inference-features-extract-atom-pairs"></a>
### `extract_atom_pairs`

```python
extract_atom_pairs(mol: Molecule, data: dict[str, Any]=None) -> dict
```

<a id="hotpot-cheminfo-operations-inference-features-extract-ring-attrs"></a>
### `extract_ring_attrs`

```python
extract_ring_attrs(mol: Molecule, rings_attr_names: Iterable[str]=None, data: dict[str, Any]=None) -> dict
```


<a id="hotpot-cheminfo-operations-inference-mca"></a>
## `hotpot.cheminfo.operations.inference.mca`

`hotpot/cheminfo/operations/inference/mca/__init__.py`；显式 __all__: ('mca', 'AtomPrediction', 'MCACpuPredictor', 'MCAGpuOutOfMemoryError', 'MCAGpuPredictor', 'MoleculePrediction', 'SitePrediction', 'predict_mca', 'predict_mca_cpu', 'predict_mca_gpu')

- 导出 `mca` → `hotpot.cheminfo.operations.inference.mca.attachment.mca`；`mca(mol: Molecule, *, backend: Literal['cpu', 'gpu'], **predictor_options)`
- 导出 `AtomPrediction` → `hotpot.cheminfo.operations.inference.mca.results.AtomPrediction`
- 导出 `MCACpuPredictor` → `hotpot.cheminfo.operations.inference.mca.cpu_backend.MCACpuPredictor`
- 导出 `MCAGpuOutOfMemoryError` → `hotpot.cheminfo.operations.inference.mca.gpu_memory.MCAGpuOutOfMemoryError`
- 导出 `MCAGpuPredictor` → `hotpot.cheminfo.operations.inference.mca.gpu_backend.MCAGpuPredictor`
- 导出 `MoleculePrediction` → `hotpot.cheminfo.operations.inference.mca.results.MoleculePrediction`
- 导出 `SitePrediction` → `hotpot.cheminfo.operations.inference.mca.results.SitePrediction`
- 导出 `predict_mca` → `hotpot.cheminfo.operations.inference.mca.api.predict_mca`；`predict_mca(molecules, *, backend: Literal['cpu', 'gpu'], **kwargs)`
- 导出 `predict_mca_cpu` → `hotpot.cheminfo.operations.inference.mca.api.predict_mca_cpu`；`predict_mca_cpu(molecules, **kwargs)`
- 导出 `predict_mca_gpu` → `hotpot.cheminfo.operations.inference.mca.api.predict_mca_gpu`；`predict_mca_gpu(molecules, **kwargs)`


<a id="hotpot-cheminfo-operations-inference-mca-api"></a>
## `hotpot.cheminfo.operations.inference.mca.api`

`hotpot/cheminfo/operations/inference/mca/api.py`；显式 __all__: ('MCACpuPredictor', 'MCAGpuPredictor', 'predict_mca', 'predict_mca_cpu', 'predict_mca_gpu')

- 导出 `MCACpuPredictor` → `hotpot.cheminfo.operations.inference.mca.cpu_backend.MCACpuPredictor`
- 导出 `MCAGpuPredictor` → `hotpot.cheminfo.operations.inference.mca.gpu_backend.MCAGpuPredictor`

<a id="hotpot-cheminfo-operations-inference-mca-api-predict-mca-cpu"></a>
### `predict_mca_cpu`

```python
predict_mca_cpu(molecules, **kwargs)
```

<a id="hotpot-cheminfo-operations-inference-mca-api-predict-mca-gpu"></a>
### `predict_mca_gpu`

```python
predict_mca_gpu(molecules, **kwargs)
```

<a id="hotpot-cheminfo-operations-inference-mca-api-predict-mca"></a>
### `predict_mca`

```python
predict_mca(molecules, *, backend: Literal['cpu', 'gpu'], **kwargs)
```


<a id="hotpot-cheminfo-operations-inference-mca-attachment"></a>
## `hotpot.cheminfo.operations.inference.mca.attachment`

`hotpot/cheminfo/operations/inference/mca/attachment.py`；显式 __all__: ('mca',)


<a id="hotpot-cheminfo-operations-inference-mca-attachment-mca"></a>
### `mca`

```python
mca(mol: Molecule, *, backend: Literal['cpu', 'gpu'], **predictor_options)
```


<a id="hotpot-cheminfo-operations-inference-mca-batching"></a>
## `hotpot.cheminfo.operations.inference.mca.batching`

`hotpot/cheminfo/operations/inference/mca/batching.py`；显式 __all__: 无


<a id="hotpot-cheminfo-operations-inference-mca-batching-atomrow"></a>
### `AtomRow`

定义 owner `hotpot.cheminfo.operations.inference.mca.batching.AtomRow`；基类 无。

```python
AtomRow(molecule_index: int, atom_index: int)
```

- `molecule_index: int` [只读值字段]
- `atom_index: int` [只读值字段]

<a id="hotpot-cheminfo-operations-inference-mca-batching-logicalbatch"></a>
### `LogicalBatch`

定义 owner `hotpot.cheminfo.operations.inference.mca.batching.LogicalBatch`；基类 无。

```python
LogicalBatch(ordinal: int, rows: tuple[AtomRow, ...], pad_width: int, unique_molecule_indices: tuple[int, ...])
```

- `encoding_keys` [只读]: `tuple[EncodingKey, ...]`
- `ordinal: int` [只读值字段]
- `rows: tuple[AtomRow, ...]` [只读值字段]
- `pad_width: int` [只读值字段]
- `unique_molecule_indices: tuple[int, ...]` [只读值字段]

<a id="hotpot-cheminfo-operations-inference-mca-batching-logicalbatchplanner"></a>
### `LogicalBatchPlanner`

定义 owner `hotpot.cheminfo.operations.inference.mca.batching.LogicalBatchPlanner`；基类 无。

- `__init__(self, batch_size: int, *, first_molecule_index: int=0)`
- `push(self, molecules: Iterable[tuple[int, int]]) -> tuple[LogicalBatch, ...]`
- `finish(self) -> tuple[LogicalBatch, ...]`
- `batch_size` = `batch_size` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）

<a id="hotpot-cheminfo-operations-inference-mca-batching-scatter-batch-predictions"></a>
### `scatter_batch_predictions`

```python
scatter_batch_predictions(batch: LogicalBatch, molecule_predictions: Mapping[EncodingKey, Sequence[Prediction]]) -> tuple[Prediction, ...]
```

模块值/类型别名（声明存在不构成事实写许可）：

- `Prediction` = `TypeVar('Prediction')`
- `EncodingKey` = `tuple[int, int]`


<a id="hotpot-cheminfo-operations-inference-mca-cpu-backend"></a>
## `hotpot.cheminfo.operations.inference.mca.cpu_backend`

`hotpot/cheminfo/operations/inference/mca/cpu_backend.py`；显式 __all__: 无


<a id="hotpot-cheminfo-operations-inference-mca-cpu-backend-mcacpupredictor"></a>
### `MCACpuPredictor`

定义 owner `hotpot.cheminfo.operations.inference.mca.cpu_backend.MCACpuPredictor`；基类 无。

- `__init__(self, model_dir=None, *, variant: str | None=None, batch_size: int=64, cpu_count: int | None=None, conformer_seed: int=DEFAULT_CONFORMER_SEED, verify_model: bool=True, allow_charged: bool=False, model_source=None)`
- `predict(self, molecules)`
- `iter_predict(self, molecules) -> Iterator[MoleculePrediction]`
- `close(self)`
- `__enter__(self)`
- `__exit__(self, exc_type, exc_value, traceback)`
- `cpu_ids` = `resolve_cpu_ids(cpu_count, cpu_affinity=owner_affinity)` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `cpu_count` = `len(self.cpu_ids)` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `batch_size` = `batch_size` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `conformer_seed` = `conformer_seed` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `max_atoms` = `int(self.runtime.manifest['max_atoms'])` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `allow_charged` = `allow_charged` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `runtime` = `create_cpu_runtime(model_dir=model_dir, variant=variant, verify_model=verify_model, model_source=model_source, cpu_threads=self.cpu_count)` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）

模块值/类型别名（声明存在不构成事实写许可）：

- `CPU_PREPROCESS_WINDOW_MOLECULES` = `64`


<a id="hotpot-cheminfo-operations-inference-mca-gpu-backend"></a>
## `hotpot.cheminfo.operations.inference.mca.gpu_backend`

`hotpot/cheminfo/operations/inference/mca/gpu_backend.py`；显式 __all__: 无


<a id="hotpot-cheminfo-operations-inference-mca-gpu-backend-mcagpupredictor"></a>
### `MCAGpuPredictor`

定义 owner `hotpot.cheminfo.operations.inference.mca.gpu_backend.MCAGpuPredictor`；基类 无。

- `__init__(self, model_dir=None, *, variant: str | None=None, gpu_count: int | None=None, gpu_indices: Sequence[int] | None=None, site_cpu_per_gpu: int=DEFAULT_SITE_CPU_PER_GPU, infer_cpu_per_gpu: int=DEFAULT_INFER_CPU_PER_GPU, gpu_batch_size: int | None=None, conformer_seed: int=DEFAULT_CONFORMER_SEED, verify_model: bool=True, allow_charged: bool=False, model_source=None)`
- `predict(self, molecules)`
- `iter_predict(self, molecules) -> Iterator[MoleculePrediction]`
- `close(self) -> None`
- `__enter__(self)`
- `__exit__(self, exc_type, exc_value, traceback)`
- `gpu_indices` = `indices` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `resource_plan` = `None` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `gpu_memory_totals` = `()` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `gpu_batch_size` = `gpu_batch_size` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `conformer_seed` = `conformer_seed` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `allow_charged` = `allow_charged` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）

模块值/类型别名（声明存在不构成事实写许可）：

- `MIN_GPU_SITE_COUNT` = `256`
- `logger` = `logging.getLogger(__name__)`


<a id="hotpot-cheminfo-operations-inference-mca-gpu-consumer"></a>
## `hotpot.cheminfo.operations.inference.mca.gpu_consumer`

`hotpot/cheminfo/operations/inference/mca/gpu_consumer.py`；显式 __all__: 无


<a id="hotpot-cheminfo-operations-inference-mca-gpu-consumer-gpuconsumerconfig"></a>
### `GpuConsumerConfig`

定义 owner `hotpot.cheminfo.operations.inference.mca.gpu_consumer.GpuConsumerConfig`；基类 无。

```python
GpuConsumerConfig(session_path: str, variant: str, max_atoms: int, batch_size: int)
```

- `session_path: str` [只读值字段]
- `variant: str` [只读值字段]
- `max_atoms: int` [只读值字段]
- `batch_size: int` [只读值字段]

<a id="hotpot-cheminfo-operations-inference-mca-gpu-consumer-gpubatchrequest"></a>
### `GpuBatchRequest`

定义 owner `hotpot.cheminfo.operations.inference.mca.gpu_consumer.GpuBatchRequest`；基类 无。

```python
GpuBatchRequest(request_id: int, batch_ordinal: int, pad_width: int, molecule_indices: tuple[int, ...], features: tuple[Mapping[str, np.ndarray], ...])
```

- `request_id: int` [只读值字段]
- `batch_ordinal: int` [只读值字段]
- `pad_width: int` [只读值字段]
- `molecule_indices: tuple[int, ...]` [只读值字段]
- `features: tuple[Mapping[str, np.ndarray], ...]` [只读值字段]

<a id="hotpot-cheminfo-operations-inference-mca-gpu-consumer-gpuconsumerready"></a>
### `GpuConsumerReady`

定义 owner `hotpot.cheminfo.operations.inference.mca.gpu_consumer.GpuConsumerReady`；基类 无。

```python
GpuConsumerReady(gpu_index: int, variant: str, max_atoms: int)
```

- `gpu_index: int` [只读值字段]
- `variant: str` [只读值字段]
- `max_atoms: int` [只读值字段]

<a id="hotpot-cheminfo-operations-inference-mca-gpu-consumer-gpubatchresult"></a>
### `GpuBatchResult`

定义 owner `hotpot.cheminfo.operations.inference.mca.gpu_consumer.GpuBatchResult`；基类 无。

```python
GpuBatchResult(request_id: int, batch_ordinal: int, gpu_index: int, molecule_indices: tuple[int, ...], pad_width: int, predictions: tuple[np.ndarray, ...])
```

- `request_id: int` [只读值字段]
- `batch_ordinal: int` [只读值字段]
- `gpu_index: int` [只读值字段]
- `molecule_indices: tuple[int, ...]` [只读值字段]
- `pad_width: int` [只读值字段]
- `predictions: tuple[np.ndarray, ...]` [只读值字段]

<a id="hotpot-cheminfo-operations-inference-mca-gpu-consumer-gpuconsumerfailure"></a>
### `GpuConsumerFailure`

定义 owner `hotpot.cheminfo.operations.inference.mca.gpu_consumer.GpuConsumerFailure`；基类 无。

```python
GpuConsumerFailure(gpu_index: int, request_id: int | None, batch_ordinal: int | None, exception_type: str, message: str, traceback_text: str)
```

- `gpu_index: int` [只读值字段]
- `request_id: int | None` [只读值字段]
- `batch_ordinal: int | None` [只读值字段]
- `exception_type: str` [只读值字段]
- `message: str` [只读值字段]
- `traceback_text: str` [只读值字段]

<a id="hotpot-cheminfo-operations-inference-mca-gpu-consumer-gpu-consumer-main"></a>
### `gpu_consumer_main`

```python
gpu_consumer_main(gpu_index: int, cpu_ids: tuple[int, ...], config: GpuConsumerConfig, request_queue, result_queue) -> None
```


<a id="hotpot-cheminfo-operations-inference-mca-gpu-memory"></a>
## `hotpot.cheminfo.operations.inference.mca.gpu_memory`

`hotpot/cheminfo/operations/inference/mca/gpu_memory.py`；显式 __all__: 无


<a id="hotpot-cheminfo-operations-inference-mca-gpu-memory-automatic-gpu-batch-size"></a>
### `automatic_gpu_batch_size`

```python
automatic_gpu_batch_size(total_memory_bytes: int) -> int
```

<a id="hotpot-cheminfo-operations-inference-mca-gpu-memory-validate-gpu-batch-size"></a>
### `validate_gpu_batch_size`

```python
validate_gpu_batch_size(batch_size: int) -> int
```

<a id="hotpot-cheminfo-operations-inference-mca-gpu-memory-resolve-gpu-batch-size"></a>
### `resolve_gpu_batch_size`

```python
resolve_gpu_batch_size(total_memory_bytes: Sequence[int], requested_batch_size: int | None=None) -> int
```

<a id="hotpot-cheminfo-operations-inference-mca-gpu-memory-next-lower-gpu-batch-size"></a>
### `next_lower_gpu_batch_size`

```python
next_lower_gpu_batch_size(batch_size: int) -> int | None
```

<a id="hotpot-cheminfo-operations-inference-mca-gpu-memory-mcagpuoutofmemoryerror"></a>
### `MCAGpuOutOfMemoryError`

定义 owner `hotpot.cheminfo.operations.inference.mca.gpu_memory.MCAGpuOutOfMemoryError`；基类 `RuntimeError`。

- `__init__(self, *, gpu_index: int, batch_size: int, operation: str, total_memory_bytes: int | None=None, free_memory_bytes: int | None=None, variant: str | None=None)`
- `gpu_index` = `gpu_index` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `batch_size` = `batch_size` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `next_batch_size` = `next_lower_gpu_batch_size(batch_size)` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `operation` = `operation` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `total_memory_bytes` = `total_memory_bytes` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `free_memory_bytes` = `free_memory_bytes` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `variant` = `variant` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）

<a id="hotpot-cheminfo-operations-inference-mca-gpu-memory-is-cuda-out-of-memory"></a>
### `is_cuda_out_of_memory`

```python
is_cuda_out_of_memory(error: BaseException) -> bool
```

模块值/类型别名（声明存在不构成事实写许可）：

- `GIB` = `1024 ** 3`
- `SUPPORTED_GPU_BATCH_SIZES` = `(64, 128, 256, 512, 1024, 2048)`


<a id="hotpot-cheminfo-operations-inference-mca-model-store"></a>
## `hotpot.cheminfo.operations.inference.mca.model_store`

`hotpot/cheminfo/operations/inference/mca/model_store.py`；显式 __all__: 无


<a id="hotpot-cheminfo-operations-inference-mca-model-store-modelstore"></a>
### `ModelStore`

定义 owner `hotpot.cheminfo.operations.inference.mca.model_store.ModelStore`；基类 无。

- `__init__(self, model_dir=None, verify: bool=True, model_source=None)`
- `resolve(self, variant: str | None=None)`
- `model_dir` = `ModelArtifact(pointer, 'HOTPOT_MCA_MODEL_DIR', model_dir=model_dir, source=model_source).resolve()` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `manifest` = `load_manifest(self.model_dir / 'manifest.json')` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `verify` = `verify` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）

<a id="hotpot-cheminfo-operations-inference-mca-model-store-resolvedmodelartifact"></a>
### `ResolvedModelArtifact`

定义 owner `hotpot.cheminfo.operations.inference.mca.model_store.ResolvedModelArtifact`；基类 无。

```python
ResolvedModelArtifact(session_path: Path, variant: str, manifest: Mapping[str, object])
```

- `session_path: Path` [只读值字段]
- `variant: str` [只读值字段]
- `manifest: Mapping[str, object]` [只读值字段]

<a id="hotpot-cheminfo-operations-inference-mca-model-store-resolved-model-artifact"></a>
### `resolved_model_artifact`

```python
resolved_model_artifact(*, model_dir=None, variant: str | None=None, verify_model: bool=True, model_source=None)
```

<a id="hotpot-cheminfo-operations-inference-mca-model-store-create-cpu-runtime"></a>
### `create_cpu_runtime`

```python
create_cpu_runtime(*, model_dir=None, variant: str | None=None, verify_model: bool=True, model_source=None, cpu_threads: int | None=None) -> MCARuntime
```

<a id="hotpot-cheminfo-operations-inference-mca-model-store-create-cuda-runtime"></a>
### `create_cuda_runtime`

```python
create_cuda_runtime(*, model_dir=None, variant: str | None=None, verify_model: bool=True, model_source=None, device_id: int=0) -> MCARuntime
```


<a id="hotpot-cheminfo-operations-inference-mca-pipeline"></a>
## `hotpot.cheminfo.operations.inference.mca.pipeline`

`hotpot/cheminfo/operations/inference/mca/pipeline.py`；显式 __all__: ('FeatureDict', 'PreparedMolecule', 'SiteRecord', 'DEFAULT_CONFORMER_SEED', 'build_features', 'build_prediction', 'detect_sites', 'is_single_molecule_input', 'prepare_molecule')


<a id="hotpot-cheminfo-operations-inference-mca-pipeline-preparedmolecule"></a>
### `PreparedMolecule`

定义 owner `hotpot.cheminfo.operations.inference.mca.pipeline.PreparedMolecule`；基类 无。

```python
PreparedMolecule(molecule: MoleculeDTO, sites: tuple[SiteRecord, ...])
```

- `molecule: MoleculeDTO` [只读值字段]
- `sites: tuple[SiteRecord, ...]` [只读值字段]

<a id="hotpot-cheminfo-operations-inference-mca-pipeline-is-single-molecule-input"></a>
### `is_single_molecule_input`

```python
is_single_molecule_input(value) -> bool
```

<a id="hotpot-cheminfo-operations-inference-mca-pipeline-detect-sites"></a>
### `detect_sites`

```python
detect_sites(molecule: MoleculeDTO) -> tuple[SiteRecord, ...]
```

<a id="hotpot-cheminfo-operations-inference-mca-pipeline-prepare-molecule"></a>
### `prepare_molecule`

```python
prepare_molecule(value, position: int, allow_charged: bool) -> PreparedMolecule
```

<a id="hotpot-cheminfo-operations-inference-mca-pipeline-build-features"></a>
### `build_features`

```python
build_features(molecule: MoleculeDTO, max_atoms: int, conformer_seed: int) -> FeatureDict
```

<a id="hotpot-cheminfo-operations-inference-mca-pipeline-build-prediction"></a>
### `build_prediction`

```python
build_prediction(molecule: MoleculeDTO, sites: Sequence[SiteRecord], values: Sequence[float], model_variant: str) -> MoleculePrediction
```

模块值/类型别名（声明存在不构成事实写许可）：

- `SiteRecord` = `tuple[int, str]`
- `FeatureDict` = `dict[str, np.ndarray]`
- `DEFAULT_CONFORMER_SEED` = `42`


<a id="hotpot-cheminfo-operations-inference-mca-resources"></a>
## `hotpot.cheminfo.operations.inference.mca.resources`

`hotpot/cheminfo/operations/inference/mca/resources.py`；显式 __all__: 无


<a id="hotpot-cheminfo-operations-inference-mca-resources-insufficientcpuaffinityerror"></a>
### `InsufficientCPUAffinityError`

定义 owner `hotpot.cheminfo.operations.inference.mca.resources.InsufficientCPUAffinityError`；基类 `RuntimeError`。

- `__init__(self, *, gpu_count: int, site_cpu_per_gpu: int, infer_cpu_per_gpu: int, required_cpu_count: int, available_cpu_count: int)`
- `gpu_count` = `gpu_count` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `site_cpu_per_gpu` = `site_cpu_per_gpu` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `infer_cpu_per_gpu` = `infer_cpu_per_gpu` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `required_cpu_count` = `required_cpu_count` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `available_cpu_count` = `available_cpu_count` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）

<a id="hotpot-cheminfo-operations-inference-mca-resources-insufficientcpucounterror"></a>
### `InsufficientCPUCountError`

定义 owner `hotpot.cheminfo.operations.inference.mca.resources.InsufficientCPUCountError`；基类 `RuntimeError`。

- `__init__(self, *, cpu_count: int, available_cpu_count: int)`
- `cpu_count` = `cpu_count` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `available_cpu_count` = `available_cpu_count` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）

<a id="hotpot-cheminfo-operations-inference-mca-resources-cpuworkerassignment"></a>
### `CPUWorkerAssignment`

定义 owner `hotpot.cheminfo.operations.inference.mca.resources.CPUWorkerAssignment`；基类 无。

```python
CPUWorkerAssignment(stage: Literal['site', 'infer'], gpu_index: int, stage_worker_index: int, global_worker_index: int, cpu_id: int)
```

- `stage: Literal['site', 'infer']` [只读值字段]
- `gpu_index: int` [只读值字段]
- `stage_worker_index: int` [只读值字段]
- `global_worker_index: int` [只读值字段]
- `cpu_id: int` [只读值字段]

<a id="hotpot-cheminfo-operations-inference-mca-resources-gpucpuassignment"></a>
### `GPUCPUAssignment`

定义 owner `hotpot.cheminfo.operations.inference.mca.resources.GPUCPUAssignment`；基类 无。

```python
GPUCPUAssignment(gpu_index: int, cpu_ids: tuple[int, ...], site_cpu_ids: tuple[int, ...], infer_cpu_ids: tuple[int, ...])
```

- `gpu_index: int` [只读值字段]
- `cpu_ids: tuple[int, ...]` [只读值字段]
- `site_cpu_ids: tuple[int, ...]` [只读值字段]
- `infer_cpu_ids: tuple[int, ...]` [只读值字段]

<a id="hotpot-cheminfo-operations-inference-mca-resources-gpuresourceplan"></a>
### `GPUResourcePlan`

定义 owner `hotpot.cheminfo.operations.inference.mca.resources.GPUResourcePlan`；基类 无。

```python
GPUResourcePlan(gpu_indices: tuple[int, ...], available_cpu_ids: tuple[int, ...], site_cpu_per_gpu: int, infer_cpu_per_gpu: int, gpu_cpu_assignments: tuple[GPUCPUAssignment, ...], site_worker_assignments: tuple[CPUWorkerAssignment, ...], infer_worker_assignments: tuple[CPUWorkerAssignment, ...])
```

- `gpu_count` [只读]: `int`
- `required_cpu_count` [只读]: `int`
- `assigned_cpu_ids` [只读]: `tuple[int, ...]`
- `gpu_indices: tuple[int, ...]` [只读值字段]
- `available_cpu_ids: tuple[int, ...]` [只读值字段]
- `site_cpu_per_gpu: int` [只读值字段]
- `infer_cpu_per_gpu: int` [只读值字段]
- `gpu_cpu_assignments: tuple[GPUCPUAssignment, ...]` [只读值字段]
- `site_worker_assignments: tuple[CPUWorkerAssignment, ...]` [只读值字段]
- `infer_worker_assignments: tuple[CPUWorkerAssignment, ...]` [只读值字段]

<a id="hotpot-cheminfo-operations-inference-mca-resources-resolve-gpu-indices"></a>
### `resolve_gpu_indices`

```python
resolve_gpu_indices(*, gpu_count: int | None=None, gpu_indices: Sequence[int] | None=None) -> tuple[int, ...]
```

<a id="hotpot-cheminfo-operations-inference-mca-resources-resolve-cpu-ids"></a>
### `resolve_cpu_ids`

```python
resolve_cpu_ids(cpu_count: int | None=None, *, cpu_affinity: Sequence[int] | None=None) -> tuple[int, ...]
```

<a id="hotpot-cheminfo-operations-inference-mca-resources-validate-gpu-resource-request"></a>
### `validate_gpu_resource_request`

```python
validate_gpu_resource_request(*, gpu_count: int | None=None, gpu_indices: Sequence[int] | None=None, site_cpu_per_gpu: int=DEFAULT_SITE_CPU_PER_GPU, infer_cpu_per_gpu: int=DEFAULT_INFER_CPU_PER_GPU) -> tuple[tuple[int, ...], int, int]
```

<a id="hotpot-cheminfo-operations-inference-mca-resources-build-gpu-resource-plan"></a>
### `build_gpu_resource_plan`

```python
build_gpu_resource_plan(*, gpu_count: int | None=None, gpu_indices: Sequence[int] | None=None, site_cpu_per_gpu: int=DEFAULT_SITE_CPU_PER_GPU, infer_cpu_per_gpu: int=DEFAULT_INFER_CPU_PER_GPU, cpu_affinity: Sequence[int] | None=None) -> GPUResourcePlan
```

<a id="hotpot-cheminfo-operations-inference-mca-resources-gpu-window-atom-budget"></a>
### `gpu_window_atom_budget`

```python
gpu_window_atom_budget(gpu_batch_size: int, worker_count: int) -> int
```

模块值/类型别名（声明存在不构成事实写许可）：

- `DEFAULT_CPU_COUNT` = `8`
- `DEFAULT_SITE_CPU_PER_GPU` = `1`
- `DEFAULT_INFER_CPU_PER_GPU` = `3`


<a id="hotpot-cheminfo-operations-inference-mca-results"></a>
## `hotpot.cheminfo.operations.inference.mca.results`

`hotpot/cheminfo/operations/inference/mca/results.py`；显式 __all__: 无


<a id="hotpot-cheminfo-operations-inference-mca-results-atomprediction"></a>
### `AtomPrediction`

定义 owner `hotpot.cheminfo.operations.inference.mca.results.AtomPrediction`；基类 无。

```python
AtomPrediction(atom_index: int, element: str, mca_kj_mol: float)
```

- `to_dict(self)`
- `atom_index: int` [只读值字段]
- `element: str` [只读值字段]
- `mca_kj_mol: float` [只读值字段]

<a id="hotpot-cheminfo-operations-inference-mca-results-siteprediction"></a>
### `SitePrediction`

定义 owner `hotpot.cheminfo.operations.inference.mca.results.SitePrediction`；基类 无。

```python
SitePrediction(atom_index: int, element: str, site_type: str, mca_kj_mol: float)
```

- `to_dict(self)`
- `atom_index: int` [只读值字段]
- `element: str` [只读值字段]
- `site_type: str` [只读值字段]
- `mca_kj_mol: float` [只读值字段]

<a id="hotpot-cheminfo-operations-inference-mca-results-moleculeprediction"></a>
### `MoleculePrediction`

定义 owner `hotpot.cheminfo.operations.inference.mca.results.MoleculePrediction`；基类 无。

```python
MoleculePrediction(smiles: str, formal_charge: int, sites: tuple[SitePrediction, ...], model_variant: str, atom_predictions: tuple[AtomPrediction, ...] = ())
```

- `to_dict(self)`
- `smiles: str` [只读值字段]
- `formal_charge: int` [只读值字段]
- `sites: tuple[SitePrediction, ...]` [只读值字段]
- `model_variant: str` [只读值字段]
- `atom_predictions: tuple[AtomPrediction, ...]` = `()` [只读值字段]

<a id="hotpot-cheminfo-operations-inference-mca-results-atomdto"></a>
### `AtomDTO`

定义 owner `hotpot.cheminfo.operations.inference.mca.results.AtomDTO`；基类 无。

```python
AtomDTO(atomic_number: int, formal_charge: int, partial_charge: float | None, is_aromatic: bool, implicit_hydrogens: int, valence: int, isotope: int, coordinates: tuple[float, float, float])
```

- `atomic_number: int` [只读值字段]
- `formal_charge: int` [只读值字段]
- `partial_charge: float | None` [只读值字段]
- `is_aromatic: bool` [只读值字段]
- `implicit_hydrogens: int` [只读值字段]
- `valence: int` [只读值字段]
- `isotope: int` [只读值字段]
- `coordinates: tuple[float, float, float]` [只读值字段]

<a id="hotpot-cheminfo-operations-inference-mca-results-bonddto"></a>
### `BondDTO`

定义 owner `hotpot.cheminfo.operations.inference.mca.results.BondDTO`；基类 无。

```python
BondDTO(begin_atom_index: int, end_atom_index: int, bond_order: float, bond_kind: str, bond_direction: str | None)
```

- `begin_atom_index: int` [只读值字段]
- `end_atom_index: int` [只读值字段]
- `bond_order: float` [只读值字段]
- `bond_kind: str` [只读值字段]
- `bond_direction: str | None` [只读值字段]

<a id="hotpot-cheminfo-operations-inference-mca-results-moleculedto"></a>
### `MoleculeDTO`

定义 owner `hotpot.cheminfo.operations.inference.mca.results.MoleculeDTO`；基类 无。

```python
MoleculeDTO(version: int, smiles: str, formal_charge: int, atom_symbols: tuple[str, ...], atom_count: int, atoms: tuple[AtomDTO, ...], bonds: tuple[BondDTO, ...], rdkit_binary: bytes)
```

- `version: int` [只读值字段]
- `smiles: str` [只读值字段]
- `formal_charge: int` [只读值字段]
- `atom_symbols: tuple[str, ...]` [只读值字段]
- `atom_count: int` [只读值字段]
- `atoms: tuple[AtomDTO, ...]` [只读值字段]
- `bonds: tuple[BondDTO, ...]` [只读值字段]
- `rdkit_binary: bytes` [只读值字段]

模块值/类型别名（声明存在不构成事实写许可）：

- `MCA_MOLECULE_DTO_VERSION` = `1`


<a id="hotpot-cheminfo-operations-inference-mca-serialization"></a>
## `hotpot.cheminfo.operations.inference.mca.serialization`

`hotpot/cheminfo/operations/inference/mca/serialization.py`；显式 __all__: ('MCA_MOLECULE_DTO_VERSION', 'AtomDTO', 'BondDTO', 'MoleculeDTO', 'hotpot_mol_from_dto', 'normalize_molecule', 'rdmol_from_dto', 'validate_dto')

- 导出 `MCA_MOLECULE_DTO_VERSION` → `hotpot.cheminfo.operations.inference.mca.results.MCA_MOLECULE_DTO_VERSION`
- 导出 `AtomDTO` → `hotpot.cheminfo.operations.inference.mca.results.AtomDTO`
- 导出 `BondDTO` → `hotpot.cheminfo.operations.inference.mca.results.BondDTO`
- 导出 `MoleculeDTO` → `hotpot.cheminfo.operations.inference.mca.results.MoleculeDTO`

<a id="hotpot-cheminfo-operations-inference-mca-serialization-normalize-molecule"></a>
### `normalize_molecule`

```python
normalize_molecule(value) -> MoleculeDTO
```

<a id="hotpot-cheminfo-operations-inference-mca-serialization-validate-dto"></a>
### `validate_dto`

```python
validate_dto(molecule: MoleculeDTO, position: int, allow_charged: bool=False) -> None
```

<a id="hotpot-cheminfo-operations-inference-mca-serialization-hotpot-mol-from-dto"></a>
### `hotpot_mol_from_dto`

```python
hotpot_mol_from_dto(molecule: MoleculeDTO) -> HotpotMolecule
```

<a id="hotpot-cheminfo-operations-inference-mca-serialization-rdmol-from-dto"></a>
### `rdmol_from_dto`

```python
rdmol_from_dto(molecule: MoleculeDTO) -> Chem.Mol
```


<a id="hotpot-cheminfo-operations-inference-mca-site-detection"></a>
## `hotpot.cheminfo.operations.inference.mca.site_detection`

`hotpot/cheminfo/operations/inference/mca/site_detection.py`；显式 __all__: 无


<a id="hotpot-cheminfo-operations-inference-mca-site-detection-find-nucleophilic-sites"></a>
### `find_nucleophilic_sites`

```python
find_nucleophilic_sites(mol) -> tuple[DetectedSite, ...]
```


<a id="hotpot-cheminfo-operations-inference-mca-workers"></a>
## `hotpot.cheminfo.operations.inference.mca.workers`

`hotpot/cheminfo/operations/inference/mca/workers.py`；显式 __all__: ('CPUPreprocessResult', 'CPUPreprocessTask', 'FeatureDict', 'FeatureResult', 'FeatureTask', 'SiteDetectionResult', 'SiteDetectionTask', 'build_features_worker', 'cpu_preprocess_worker', 'detect_sites_worker')

- 导出 `FeatureDict` → `hotpot.cheminfo.operations.inference.mca.pipeline.FeatureDict`

<a id="hotpot-cheminfo-operations-inference-mca-workers-sitedetectiontask"></a>
### `SiteDetectionTask`

定义 owner `hotpot.cheminfo.operations.inference.mca.workers.SiteDetectionTask`；基类 无。

```python
SiteDetectionTask(molecule_index: int, molecule: MoleculeDTO)
```

- `molecule_index: int` [只读值字段]
- `molecule: MoleculeDTO` [只读值字段]

<a id="hotpot-cheminfo-operations-inference-mca-workers-sitedetectionresult"></a>
### `SiteDetectionResult`

定义 owner `hotpot.cheminfo.operations.inference.mca.workers.SiteDetectionResult`；基类 无。

```python
SiteDetectionResult(molecule_index: int, sites: tuple[tuple[int, str], ...])
```

- `molecule_index: int` [只读值字段]
- `sites: tuple[tuple[int, str], ...]` [只读值字段]

<a id="hotpot-cheminfo-operations-inference-mca-workers-featuretask"></a>
### `FeatureTask`

定义 owner `hotpot.cheminfo.operations.inference.mca.workers.FeatureTask`；基类 无。

```python
FeatureTask(molecule_index: int, molecule: MoleculeDTO, max_atoms: int, conformer_seed: int)
```

- `molecule_index: int` [只读值字段]
- `molecule: MoleculeDTO` [只读值字段]
- `max_atoms: int` [只读值字段]
- `conformer_seed: int` [只读值字段]

<a id="hotpot-cheminfo-operations-inference-mca-workers-featureresult"></a>
### `FeatureResult`

定义 owner `hotpot.cheminfo.operations.inference.mca.workers.FeatureResult`；基类 无。

```python
FeatureResult(molecule_index: int, feature: FeatureDict)
```

- `molecule_index: int` [只读值字段]
- `feature: FeatureDict` [只读值字段]

<a id="hotpot-cheminfo-operations-inference-mca-workers-cpupreprocesstask"></a>
### `CPUPreprocessTask`

定义 owner `hotpot.cheminfo.operations.inference.mca.workers.CPUPreprocessTask`；基类 无。

```python
CPUPreprocessTask(molecule_index: int, molecule: MoleculeDTO, sites: tuple[tuple[int, str], ...] | None, max_atoms: int, conformer_seed: int)
```

- `molecule_index: int` [只读值字段]
- `molecule: MoleculeDTO` [只读值字段]
- `sites: tuple[tuple[int, str], ...] | None` [只读值字段]
- `max_atoms: int` [只读值字段]
- `conformer_seed: int` [只读值字段]

<a id="hotpot-cheminfo-operations-inference-mca-workers-cpupreprocessresult"></a>
### `CPUPreprocessResult`

定义 owner `hotpot.cheminfo.operations.inference.mca.workers.CPUPreprocessResult`；基类 无。

```python
CPUPreprocessResult(molecule_index: int, molecule: MoleculeDTO, sites: tuple[tuple[int, str], ...], feature: FeatureDict)
```

- `molecule_index: int` [只读值字段]
- `molecule: MoleculeDTO` [只读值字段]
- `sites: tuple[tuple[int, str], ...]` [只读值字段]
- `feature: FeatureDict` [只读值字段]

<a id="hotpot-cheminfo-operations-inference-mca-workers-detect-sites-worker"></a>
### `detect_sites_worker`

```python
detect_sites_worker(task: SiteDetectionTask) -> SiteDetectionResult
```

<a id="hotpot-cheminfo-operations-inference-mca-workers-build-features-worker"></a>
### `build_features_worker`

```python
build_features_worker(task: FeatureTask) -> FeatureResult
```

<a id="hotpot-cheminfo-operations-inference-mca-workers-cpu-preprocess-worker"></a>
### `cpu_preprocess_worker`

```python
cpu_preprocess_worker(task: CPUPreprocessTask) -> CPUPreprocessResult
```


<a id="hotpot-pipeline"></a>
## `hotpot.pipeline`

`hotpot/pipeline/__init__.py`；显式 __all__: ('Artifact', 'DuplicateStageError', 'JSONScalar', 'JSONValue', 'MolecularPayload', 'MolecularRecord', 'MolecularStage', 'PipelineDefinitionError', 'PipelineExecutionError', 'PipelineRunResult', 'PipelineStatus', 'PreparedMolecularStage', 'StageContext', 'StageExecutionError', 'StageResult', 'StageSpec', 'StageStatus', 'UnknownStageError', 'artifact_from_file', 'builtin_stage_names', 'get_stage', 'register_stage', 'run_pipeline', 'stage_import_path', 'payload_sha256')

- 导出 `Artifact` → `hotpot.pipeline.contracts.Artifact`
- 导出 `DuplicateStageError` → `hotpot.pipeline.registry.DuplicateStageError`
- 导出 `JSONScalar` → `hotpot.pipeline.contracts.JSONScalar`
- 导出 `JSONValue` → `hotpot.pipeline.contracts.JSONValue`
- 导出 `MolecularPayload` → `hotpot.pipeline.contracts.MolecularPayload`
- 导出 `MolecularRecord` → `hotpot.pipeline.contracts.MolecularRecord`
- 导出 `MolecularStage` → `hotpot.pipeline.contracts.MolecularStage`
- 导出 `PipelineDefinitionError` → `hotpot.pipeline.contracts.PipelineDefinitionError`
- 导出 `PipelineExecutionError` → `hotpot.pipeline.contracts.PipelineExecutionError`
- 导出 `PipelineRunResult` → `hotpot.pipeline.contracts.PipelineRunResult`
- 导出 `PipelineStatus` → `hotpot.pipeline.contracts.PipelineStatus`
- 导出 `PreparedMolecularStage` → `hotpot.pipeline.contracts.PreparedMolecularStage`
- 导出 `StageContext` → `hotpot.pipeline.contracts.StageContext`
- 导出 `StageExecutionError` → `hotpot.pipeline.contracts.StageExecutionError`
- 导出 `StageResult` → `hotpot.pipeline.contracts.StageResult`
- 导出 `StageSpec` → `hotpot.pipeline.contracts.StageSpec`
- 导出 `StageStatus` → `hotpot.pipeline.contracts.StageStatus`
- 导出 `UnknownStageError` → `hotpot.pipeline.registry.UnknownStageError`
- 导出 `artifact_from_file` → `hotpot.pipeline.artifacts.artifact_from_file`；`artifact_from_file(stage_directory: Path, path: Path) -> Artifact`
- 导出 `builtin_stage_names` → `hotpot.pipeline.registry.builtin_stage_names`；`builtin_stage_names() -> Tuple[str, ...]`
- 导出 `get_stage` → `hotpot.pipeline.registry.get_stage`；`get_stage(name: str) -> MolecularStage`
- 导出 `register_stage` → `hotpot.pipeline.registry.register_stage`；`register_stage(name: str, import_path: str) -> None`
- 导出 `run_pipeline` → `hotpot.pipeline.runner.run_pipeline`；`run_pipeline(specs: Sequence[StageSpec], *, results_directory: Path, initial_payload: Optional[MolecularPayload]=None) -> PipelineRunResult`
- 导出 `stage_import_path` → `hotpot.pipeline.registry.stage_import_path`；`stage_import_path(name: str) -> str`
- 导出 `payload_sha256` → `hotpot.pipeline.artifacts.payload_sha256`；`payload_sha256(payload: MolecularPayload) -> str`


<a id="hotpot-pipeline--sample-control"></a>
## `hotpot.pipeline._sample_control`

`hotpot/pipeline/_sample_control.py`；显式 __all__: 无



<a id="hotpot-pipeline-artifacts"></a>
## `hotpot.pipeline.artifacts`

`hotpot/pipeline/artifacts.py`；显式 __all__: ('artifact_from_file', 'payload_sha256')


<a id="hotpot-pipeline-artifacts-artifact-from-file"></a>
### `artifact_from_file`

```python
artifact_from_file(stage_directory: Path, path: Path) -> Artifact
```

<a id="hotpot-pipeline-artifacts-payload-sha256"></a>
### `payload_sha256`

```python
payload_sha256(payload: MolecularPayload) -> str
```


<a id="hotpot-pipeline-cli"></a>
## `hotpot.pipeline.cli`

`hotpot/pipeline/cli.py`；显式 __all__: ('add_arguments', 'build_parser', 'load_cli_documentation', 'load_json_workflow', 'main', 'parse_inline_stages', 'run')


<a id="hotpot-pipeline-cli-load-cli-documentation"></a>
### `load_cli_documentation`

```python
load_cli_documentation() -> str
```

<a id="hotpot-pipeline-cli-parse-inline-stages"></a>
### `parse_inline_stages`

```python
parse_inline_stages(tokens: Sequence[str]) -> Tuple[StageSpec, ...]
```

<a id="hotpot-pipeline-cli-load-json-workflow"></a>
### `load_json_workflow`

```python
load_json_workflow(path: Path) -> Tuple[StageSpec, ...]
```

<a id="hotpot-pipeline-cli-add-arguments"></a>
### `add_arguments`

```python
add_arguments(parser: argparse.ArgumentParser) -> None
```

<a id="hotpot-pipeline-cli-run"></a>
### `run`

```python
run(args: argparse.Namespace) -> int
```

<a id="hotpot-pipeline-cli-build-parser"></a>
### `build_parser`

```python
build_parser() -> argparse.ArgumentParser
```

<a id="hotpot-pipeline-cli-main"></a>
### `main`

```python
main(argv: Optional[Sequence[str]]=None) -> int
```


<a id="hotpot-pipeline-contracts"></a>
## `hotpot.pipeline.contracts`

`hotpot/pipeline/contracts.py`；显式 __all__: ('Artifact', 'JSONScalar', 'JSONValue', 'MolecularPayload', 'MolecularRecord', 'MolecularStage', 'PipelineDefinitionError', 'PipelineExecutionError', 'PipelineRunResult', 'PipelineStatus', 'PreparedMolecularStage', 'StageContext', 'StageExecutionError', 'StageResult', 'StageSpec', 'StageStatus')


<a id="hotpot-pipeline-contracts-stagestatus"></a>
### `StageStatus`

定义 owner `hotpot.pipeline.contracts.StageStatus`；基类 `str`, `Enum`。

- `SUCCEEDED` = `'succeeded'` [枚举常量]
- `QUALITY_FAILED` = `'quality-failed'` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-pipeline-contracts-pipelinestatus"></a>
### `PipelineStatus`

定义 owner `hotpot.pipeline.contracts.PipelineStatus`；基类 `str`, `Enum`。

- `SUCCEEDED` = `'succeeded'` [枚举常量]
- `FAILED` = `'failed'` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-pipeline-contracts-pipelinedefinitionerror"></a>
### `PipelineDefinitionError`

定义 owner `hotpot.pipeline.contracts.PipelineDefinitionError`；基类 `ValueError`。


<a id="hotpot-pipeline-contracts-stageexecutionerror"></a>
### `StageExecutionError`

定义 owner `hotpot.pipeline.contracts.StageExecutionError`；基类 `RuntimeError`。


<a id="hotpot-pipeline-contracts-stagespec"></a>
### `StageSpec`

定义 owner `hotpot.pipeline.contracts.StageSpec`；基类 无。

```python
StageSpec(name: str, argv: Tuple[str, ...])
```

- `name: str` [只读值字段]
- `argv: Tuple[str, ...]` [只读值字段]

<a id="hotpot-pipeline-contracts-molecularrecord"></a>
### `MolecularRecord`

定义 owner `hotpot.pipeline.contracts.MolecularRecord`；基类 无。

```python
MolecularRecord(molecule: 'Molecule', scientific_state: Optional['ScientificStateBundle'] = None)
```

- `molecule: 'Molecule'` [只读值字段]
- `scientific_state: Optional['ScientificStateBundle']` = `None` [只读值字段]

<a id="hotpot-pipeline-contracts-molecularpayload"></a>
### `MolecularPayload`

定义 owner `hotpot.pipeline.contracts.MolecularPayload`；基类 无。

```python
MolecularPayload(records: Tuple[MolecularRecord, ...])
```

- `records: Tuple[MolecularRecord, ...]` [只读值字段]

<a id="hotpot-pipeline-contracts-stagecontext"></a>
### `StageContext`

定义 owner `hotpot.pipeline.contracts.StageContext`；基类 无。

```python
StageContext(run_directory: Path, stage_directory: Path, stage_index: int, spec: StageSpec)
```

- `run_directory: Path` [只读值字段]
- `stage_directory: Path` [只读值字段]
- `stage_index: int` [只读值字段]
- `spec: StageSpec` [只读值字段]

<a id="hotpot-pipeline-contracts-artifact"></a>
### `Artifact`

定义 owner `hotpot.pipeline.contracts.Artifact`；基类 无。

```python
Artifact(relative_path: Path, sha256: str, size_bytes: int)
```

- `relative_path: Path` [只读值字段]
- `sha256: str` [只读值字段]
- `size_bytes: int` [只读值字段]

<a id="hotpot-pipeline-contracts-stageresult"></a>
### `StageResult`

定义 owner `hotpot.pipeline.contracts.StageResult`；基类 无。

```python
StageResult(status: StageStatus, payload: MolecularPayload, artifacts: Tuple[Artifact, ...] = (), report: Mapping[str, JSONValue] = field(default_factory=dict), stderr: str = '', break_pipeline: bool = False)
```

- `status: StageStatus` [只读值字段]
- `payload: MolecularPayload` [只读值字段]
- `artifacts: Tuple[Artifact, ...]` = `()` [只读值字段]
- `report: Mapping[str, JSONValue]` = `field(default_factory=dict)` [只读值字段]
- `stderr: str` = `''` [只读值字段]
- `break_pipeline: bool` = `False` [只读值字段]

<a id="hotpot-pipeline-contracts-pipelinerunresult"></a>
### `PipelineRunResult`

定义 owner `hotpot.pipeline.contracts.PipelineRunResult`；基类 无。

```python
PipelineRunResult(status: PipelineStatus, payload: Optional[MolecularPayload], results_directory: Path, stage_results: Tuple[StageResult, ...], failed_stage_index: Optional[int] = None)
```

- `status: PipelineStatus` [只读值字段]
- `payload: Optional[MolecularPayload]` [只读值字段]
- `results_directory: Path` [只读值字段]
- `stage_results: Tuple[StageResult, ...]` [只读值字段]
- `failed_stage_index: Optional[int]` = `None` [只读值字段]

<a id="hotpot-pipeline-contracts-pipelineexecutionerror"></a>
### `PipelineExecutionError`

定义 owner `hotpot.pipeline.contracts.PipelineExecutionError`；基类 `RuntimeError`。

- `__init__(self, run_result: PipelineRunResult, stage_error: StageExecutionError) -> None`
- `run_result` = `run_result` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `stage_error` = `stage_error` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）

<a id="hotpot-pipeline-contracts-preparedmolecularstage"></a>
### `PreparedMolecularStage`

定义 owner `hotpot.pipeline.contracts.PreparedMolecularStage`；基类 `Protocol`。

- `execute(self, payload: MolecularPayload, context: StageContext) -> StageResult`

<a id="hotpot-pipeline-contracts-molecularstage"></a>
### `MolecularStage`

定义 owner `hotpot.pipeline.contracts.MolecularStage`；基类 `Protocol`。

- `prepare(self, spec: StageSpec) -> PreparedMolecularStage`

模块值/类型别名（声明存在不构成事实写许可）：

- `JSONScalar` = `Union[None, bool, int, float, str]`
- `JSONValue` = `Union[JSONScalar, Tuple['JSONValue', ...], Mapping[str, 'JSONValue']]`


<a id="hotpot-pipeline-registry"></a>
## `hotpot.pipeline.registry`

`hotpot/pipeline/registry.py`；显式 __all__: ('DuplicateStageError', 'UnknownStageError', 'builtin_stage_names', 'get_stage', 'register_stage', 'stage_import_path')


<a id="hotpot-pipeline-registry-unknownstageerror"></a>
### `UnknownStageError`

定义 owner `hotpot.pipeline.registry.UnknownStageError`；基类 `PipelineDefinitionError`。


<a id="hotpot-pipeline-registry-duplicatestageerror"></a>
### `DuplicateStageError`

定义 owner `hotpot.pipeline.registry.DuplicateStageError`；基类 `PipelineDefinitionError`。


<a id="hotpot-pipeline-registry-builtin-stage-names"></a>
### `builtin_stage_names`

```python
builtin_stage_names() -> Tuple[str, ...]
```

<a id="hotpot-pipeline-registry-register-stage"></a>
### `register_stage`

```python
register_stage(name: str, import_path: str) -> None
```

<a id="hotpot-pipeline-registry-stage-import-path"></a>
### `stage_import_path`

```python
stage_import_path(name: str) -> str
```

<a id="hotpot-pipeline-registry-get-stage"></a>
### `get_stage`

```python
get_stage(name: str) -> MolecularStage
```


<a id="hotpot-pipeline-runner"></a>
## `hotpot.pipeline.runner`

`hotpot/pipeline/runner.py`；显式 __all__: ('run_pipeline',)


<a id="hotpot-pipeline-runner-run-pipeline"></a>
### `run_pipeline`

```python
run_pipeline(specs: Sequence[StageSpec], *, results_directory: Path, initial_payload: Optional[MolecularPayload]=None) -> PipelineRunResult
```


<a id="hotpot-pipeline-state-recovery"></a>
## `hotpot.pipeline.state_recovery`

`hotpot/pipeline/state_recovery.py`；显式 __all__: ('decide_state_recovery', 'RecoveryAction', 'RecoveryPolicySet', 'RecoveryBudget', 'DEFAULT_RECOVERY_BUDGET', 'RecoveryDecision')


<a id="hotpot-pipeline-state-recovery-recoverybudget"></a>
### `RecoveryBudget`

定义 owner `hotpot.pipeline.state_recovery.RecoveryBudget`；基类 无。

```python
RecoveryBudget(max_backend_attempts: int)
```

- `max_backend_attempts: int` [只读值字段]

<a id="hotpot-pipeline-state-recovery-recoveryaction"></a>
### `RecoveryAction`

定义 owner `hotpot.pipeline.state_recovery.RecoveryAction`；基类 `Enum`。

- `STOP` = `'stop'` [枚举常量]
- `RETRY` = `'retry'` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-pipeline-state-recovery-recoverypolicyset"></a>
### `RecoveryPolicySet`

定义 owner `hotpot.pipeline.state_recovery.RecoveryPolicySet`；基类 无。

```python
RecoveryPolicySet(policy_ids: Tuple[str, ...] = ())
```

- `policy_ids: Tuple[str, ...]` = `()` [只读值字段]

<a id="hotpot-pipeline-state-recovery-recoverydecision"></a>
### `RecoveryDecision`

定义 owner `hotpot.pipeline.state_recovery.RecoveryDecision`；基类 无。

```python
RecoveryDecision(action: RecoveryAction, projection: Optional[BackendStateProjection], rationale: Tuple[DecisionReason, ...])
```

- `action: RecoveryAction` [只读值字段]
- `projection: Optional[BackendStateProjection]` [只读值字段]
- `rationale: Tuple[DecisionReason, ...]` [只读值字段]

<a id="hotpot-pipeline-state-recovery-decide-state-recovery"></a>
### `decide_state_recovery`

```python
decide_state_recovery(state: ScientificStateBundle, backend: BackendContext, *, request: BackendStateRequest=BackendStateRequest(), execution_policies: ExecutionPolicySet=ExecutionPolicySet(), execution_history: Tuple[BackendExecutionEvidence, ...], recovery_policies: RecoveryPolicySet=RecoveryPolicySet(), recovery_budget: RecoveryBudget=DEFAULT_RECOVERY_BUDGET) -> RecoveryDecision
```

模块值/类型别名（声明存在不构成事实写许可）：

- `DEFAULT_RECOVERY_BUDGET` = `RecoveryBudget(max_backend_attempts=1)`


<a id="hotpot-plugins-xtb"></a>
## `hotpot.plugins.xtb`

`hotpot/plugins/xtb/__init__.py`；显式 __all__: ('GFNXTBMethod', 'XTBApplicabilityError', 'XTBArtifact', 'XTBBackendInfo', 'XTBError', 'XTBExecutableError', 'XTBGeometry', 'XTBExecutionError', 'XTBInputError', 'XTBMethod', 'XTBParsedResult', 'XTBRequest', 'XTBResultError', 'XTBRunReport', 'XTBStreamError', 'XTBStreamMetadata', 'XTBStreamProvenance', 'XTBStreamRecord', 'XTBTask', 'commit_xtb_coordinates', 'molecule_to_xtb_geometry', 'metadata_from_report', 'parse_xtb_artifacts', 'probe_xtb_backend', 'prepare_xtb_input', 'run_xtb', 'run_gfn_xtb', 'run_gfnff', 'read_sdf_records', 'validate_element_support', 'write_sdf_records', 'XTBBackendContext')

- 导出 `GFNXTBMethod` → `hotpot.plugins.xtb.contracts.GFNXTBMethod`
- 导出 `XTBApplicabilityError` → `hotpot.plugins.xtb.contracts.XTBApplicabilityError`
- 导出 `XTBArtifact` → `hotpot.plugins.xtb.contracts.XTBArtifact`
- 导出 `XTBBackendInfo` → `hotpot.plugins.xtb.contracts.XTBBackendInfo`
- 导出 `XTBError` → `hotpot.plugins.xtb.contracts.XTBError`
- 导出 `XTBExecutableError` → `hotpot.plugins.xtb.contracts.XTBExecutableError`
- 导出 `XTBGeometry` → `hotpot.plugins.xtb.contracts.XTBGeometry`
- 导出 `XTBExecutionError` → `hotpot.plugins.xtb.contracts.XTBExecutionError`
- 导出 `XTBInputError` → `hotpot.plugins.xtb.contracts.XTBInputError`
- 导出 `XTBMethod` → `hotpot.plugins.xtb.contracts.XTBMethod`
- 导出 `XTBParsedResult` → `hotpot.plugins.xtb.contracts.XTBParsedResult`
- 导出 `XTBRequest` → `hotpot.plugins.xtb.contracts.XTBRequest`
- 导出 `XTBResultError` → `hotpot.plugins.xtb.contracts.XTBResultError`
- 导出 `XTBRunReport` → `hotpot.plugins.xtb.contracts.XTBRunReport`
- 导出 `XTBStreamError` → `hotpot.plugins.xtb.stream.XTBStreamError`
- 导出 `XTBStreamMetadata` → `hotpot.plugins.xtb.stream.XTBStreamMetadata`
- 导出 `XTBStreamProvenance` → `hotpot.plugins.xtb.stream.XTBStreamProvenance`
- 导出 `XTBStreamRecord` → `hotpot.plugins.xtb.stream.XTBStreamRecord`
- 导出 `XTBTask` → `hotpot.plugins.xtb.contracts.XTBTask`
- 导出 `commit_xtb_coordinates` → `hotpot.plugins.xtb.adapter.commit_xtb_coordinates`；`commit_xtb_coordinates(mol: Molecule, result: XTBParsedResult) -> None`
- 导出 `molecule_to_xtb_geometry` → `hotpot.plugins.xtb.adapter.molecule_to_xtb_geometry`；`molecule_to_xtb_geometry(mol: Molecule) -> XTBGeometry`
- 导出 `metadata_from_report` → `hotpot.plugins.xtb.stream.metadata_from_report`；`metadata_from_report(report: XTBRunReport, previous: Optional[XTBStreamMetadata]=None) -> XTBStreamMetadata`
- 导出 `parse_xtb_artifacts` → `hotpot.plugins.xtb.adapter.parse_xtb_artifacts`；`parse_xtb_artifacts(report: XTBRunReport, expected_geometry: XTBGeometry) -> XTBParsedResult`
- 导出 `probe_xtb_backend` → `hotpot.plugins.xtb.backend.probe_xtb_backend`；`probe_xtb_backend(executable: Optional[_ExecutablePath]=None, *, environment: Optional[Mapping[str, str]]=None, timeout_seconds: Optional[float]=30.0) -> XTBBackendInfo`
- 导出 `prepare_xtb_input` → `hotpot.plugins.xtb.adapter.prepare_xtb_input`；`prepare_xtb_input(mol: Molecule, input_path: Path) -> XTBGeometry`
- 导出 `run_xtb` → `hotpot.plugins.xtb.runner.run_xtb`；`run_xtb(request: XTBRequest) -> XTBRunReport`
- 导出 `run_gfn_xtb` → `hotpot.plugins.xtb.workflow.run_gfn_xtb`；`run_gfn_xtb(mol: Molecule, *, state: BackendStateProjection, method: GFNXTBMethod=GFNXTBMethod.GFN2_XTB, task: XTBTask=XTBTask.OPTIMIZE, executable: Optional[_ExecutablePath]=None, environment: Optional[Mapping[str, str]]=None, timeout_seconds: Optional[float]=None, work_directory: Optional[Path]=None, keep_work_directory: bool=False) -> XTBRunReport`
- 导出 `run_gfnff` → `hotpot.plugins.xtb.workflow.run_gfnff`；`run_gfnff(mol: Molecule, *, state: BackendStateProjection, task: XTBTask=XTBTask.OPTIMIZE, executable: Optional[_ExecutablePath]=None, environment: Optional[Mapping[str, str]]=None, timeout_seconds: Optional[float]=None, work_directory: Optional[Path]=None, keep_work_directory: bool=False) -> XTBRunReport`
- 导出 `read_sdf_records` → `hotpot.plugins.xtb.stream.read_sdf_records`；`read_sdf_records(text: str) -> Tuple[XTBStreamRecord, ...]`
- 导出 `validate_element_support` → `hotpot.plugins.xtb.capabilities.validate_element_support`；`validate_element_support(backend_info: XTBBackendInfo, method: XTBMethod, atomic_numbers: Iterable[int]) -> None`
- 导出 `write_sdf_records` → `hotpot.plugins.xtb.stream.write_sdf_records`；`write_sdf_records(records: Iterable[XTBStreamRecord]) -> str`
- 导出 `XTBBackendContext` → `hotpot.plugins.xtb.state_projection.XTBBackendContext`


<a id="hotpot-plugins-xtb-adapter"></a>
## `hotpot.plugins.xtb.adapter`

`hotpot/plugins/xtb/adapter.py`；显式 __all__: ('commit_xtb_coordinates', 'molecule_to_xtb_geometry', 'parse_xtb_artifacts', 'prepare_xtb_input')


<a id="hotpot-plugins-xtb-adapter-molecule-to-xtb-geometry"></a>
### `molecule_to_xtb_geometry`

```python
molecule_to_xtb_geometry(mol: Molecule) -> XTBGeometry
```

<a id="hotpot-plugins-xtb-adapter-prepare-xtb-input"></a>
### `prepare_xtb_input`

```python
prepare_xtb_input(mol: Molecule, input_path: Path) -> XTBGeometry
```

<a id="hotpot-plugins-xtb-adapter-parse-xtb-artifacts"></a>
### `parse_xtb_artifacts`

```python
parse_xtb_artifacts(report: XTBRunReport, expected_geometry: XTBGeometry) -> XTBParsedResult
```

<a id="hotpot-plugins-xtb-adapter-commit-xtb-coordinates"></a>
### `commit_xtb_coordinates`

```python
commit_xtb_coordinates(mol: Molecule, result: XTBParsedResult) -> None
```


<a id="hotpot-plugins-xtb-backend"></a>
## `hotpot.plugins.xtb.backend`

`hotpot/plugins/xtb/backend.py`；显式 __all__: ('probe_xtb_backend',)


<a id="hotpot-plugins-xtb-backend-probe-xtb-backend"></a>
### `probe_xtb_backend`

```python
probe_xtb_backend(executable: Optional[_ExecutablePath]=None, *, environment: Optional[Mapping[str, str]]=None, timeout_seconds: Optional[float]=30.0) -> XTBBackendInfo
```


<a id="hotpot-plugins-xtb-capabilities"></a>
## `hotpot.plugins.xtb.capabilities`

`hotpot/plugins/xtb/capabilities.py`；显式 __all__: ('validate_element_support',)


<a id="hotpot-plugins-xtb-capabilities-validate-element-support"></a>
### `validate_element_support`

```python
validate_element_support(backend_info: XTBBackendInfo, method: XTBMethod, atomic_numbers: Iterable[int]) -> None
```


<a id="hotpot-plugins-xtb-cli"></a>
## `hotpot.plugins.xtb.cli`

`hotpot/plugins/xtb/cli.py`；显式 __all__: ('add_arguments', 'build_parser', 'load_cli_documentation', 'main', 'run')


<a id="hotpot-plugins-xtb-cli-load-cli-documentation"></a>
### `load_cli_documentation`

```python
load_cli_documentation() -> str
```

<a id="hotpot-plugins-xtb-cli-add-arguments"></a>
### `add_arguments`

```python
add_arguments(parser: argparse.ArgumentParser) -> None
```

<a id="hotpot-plugins-xtb-cli-run"></a>
### `run`

```python
run(args: argparse.Namespace) -> int
```

<a id="hotpot-plugins-xtb-cli-build-parser"></a>
### `build_parser`

```python
build_parser() -> argparse.ArgumentParser
```

<a id="hotpot-plugins-xtb-cli-main"></a>
### `main`

```python
main(argv: Optional[Sequence[str]]=None) -> int
```


<a id="hotpot-plugins-xtb-contracts"></a>
## `hotpot.plugins.xtb.contracts`

`hotpot/plugins/xtb/contracts.py`；显式 __all__: ('GFNXTBMethod', 'XTBApplicabilityError', 'XTBArtifact', 'XTBBackendInfo', 'XTBError', 'XTBExecutableError', 'XTBExecutionError', 'XTBInputError', 'XTBGeometry', 'XTBMethod', 'XTBParsedResult', 'XTBRequest', 'XTBResultError', 'XTBRunReport', 'XTBTask')


<a id="hotpot-plugins-xtb-contracts-xtbmethod"></a>
### `XTBMethod`

定义 owner `hotpot.plugins.xtb.contracts.XTBMethod`；基类 `str`, `Enum`。

- `GFNFF` = `'gfnff'` [枚举常量]
- `GFN0_XTB` = `'gfn0'` [枚举常量]
- `GFN1_XTB` = `'gfn1'` [枚举常量]
- `GFN2_XTB` = `'gfn2'` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-plugins-xtb-contracts-gfnxtbmethod"></a>
### `GFNXTBMethod`

定义 owner `hotpot.plugins.xtb.contracts.GFNXTBMethod`；基类 `str`, `Enum`。

- `GFN0_XTB` = `'gfn0'` [枚举常量]
- `GFN1_XTB` = `'gfn1'` [枚举常量]
- `GFN2_XTB` = `'gfn2'` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-plugins-xtb-contracts-xtbtask"></a>
### `XTBTask`

定义 owner `hotpot.plugins.xtb.contracts.XTBTask`；基类 `str`, `Enum`。

- `SINGLEPOINT` = `'singlepoint'` [枚举常量]
- `OPTIMIZE` = `'optimize'` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-plugins-xtb-contracts-xtbgeometry"></a>
### `XTBGeometry`

定义 owner `hotpot.plugins.xtb.contracts.XTBGeometry`；基类 无。

```python
XTBGeometry(symbols: Tuple[str, ...], coordinates: Tuple[Tuple[float, float, float], ...])
```

- `symbols: Tuple[str, ...]` [只读值字段]
- `coordinates: Tuple[Tuple[float, float, float], ...]` [只读值字段]

<a id="hotpot-plugins-xtb-contracts-xtbparsedresult"></a>
### `XTBParsedResult`

定义 owner `hotpot.plugins.xtb.contracts.XTBParsedResult`；基类 无。

```python
XTBParsedResult(energy_hartree: float, gradient_norm: Optional[float], partial_charges: Optional[Tuple[float, ...]], optimized_geometry: Optional[XTBGeometry], atom_order_verified: bool)
```

- `energy_hartree: float` [只读值字段]
- `gradient_norm: Optional[float]` [只读值字段]
- `partial_charges: Optional[Tuple[float, ...]]` [只读值字段]
- `optimized_geometry: Optional[XTBGeometry]` [只读值字段]
- `atom_order_verified: bool` [只读值字段]

<a id="hotpot-plugins-xtb-contracts-xtbbackendinfo"></a>
### `XTBBackendInfo`

定义 owner `hotpot.plugins.xtb.contracts.XTBBackendInfo`；基类 无。

```python
XTBBackendInfo(executable: Path, version: str, revision: Optional[str], executable_sha256: str, probe_result: NativeProcessResult, gfn_xtb_max_atomic_number: int, gfnff_max_atomic_number: Optional[int])
```

- `executable: Path` [只读值字段]
- `version: str` [只读值字段]
- `revision: Optional[str]` [只读值字段]
- `executable_sha256: str` [只读值字段]
- `probe_result: NativeProcessResult` [只读值字段]
- `gfn_xtb_max_atomic_number: int` [只读值字段]
- `gfnff_max_atomic_number: Optional[int]` [只读值字段]

<a id="hotpot-plugins-xtb-contracts-xtbartifact"></a>
### `XTBArtifact`

定义 owner `hotpot.plugins.xtb.contracts.XTBArtifact`；基类 无。

```python
XTBArtifact(name: str, path: Path, sha256: str, size_bytes: int)
```

- `name: str` [只读值字段]
- `path: Path` [只读值字段]
- `sha256: str` [只读值字段]
- `size_bytes: int` [只读值字段]

<a id="hotpot-plugins-xtb-contracts-xtbrequest"></a>
### `XTBRequest`

定义 owner `hotpot.plugins.xtb.contracts.XTBRequest`；基类 无。

```python
XTBRequest(backend_info: XTBBackendInfo, method: XTBMethod, task: XTBTask, input_path: Path, work_directory: Path, charge: int, unpaired_electrons: Optional[int], environment: Mapping[str, str], timeout_seconds: Optional[float] = None)
```

- `backend_info: XTBBackendInfo` [只读值字段]
- `method: XTBMethod` [只读值字段]
- `task: XTBTask` [只读值字段]
- `input_path: Path` [只读值字段]
- `work_directory: Path` [只读值字段]
- `charge: int` [只读值字段]
- `unpaired_electrons: Optional[int]` [只读值字段]
- `environment: Mapping[str, str]` [只读值字段]
- `timeout_seconds: Optional[float]` = `None` [只读值字段]

<a id="hotpot-plugins-xtb-contracts-xtbrunreport"></a>
### `XTBRunReport`

定义 owner `hotpot.plugins.xtb.contracts.XTBRunReport`；基类 无。

```python
XTBRunReport(backend_info: XTBBackendInfo, requested_method: XTBMethod, effective_method: XTBMethod, task: XTBTask, charge: int, unpaired_electrons: Optional[int], work_directory: Path, argv: Tuple[str, ...], return_code: int, stdout: str, stderr: str, elapsed_seconds: float, process_succeeded: bool, converged: bool, artifacts: Mapping[str, XTBArtifact], provenance: ProcessProvenance, energy_hartree: Optional[float] = None, energy_unit: str = 'hartree', gradient_norm: Optional[float] = None, atom_order_verified: bool = False, coordinates_committed: bool = False, scientific_state_identity: Optional[ScientificStateIdentity] = None, facts_identity: Optional[MolecularFactsIdentity] = None, projection_convention: Optional[str] = None, workspace_retained: bool = False)
```

- `backend_info: XTBBackendInfo` [只读值字段]
- `requested_method: XTBMethod` [只读值字段]
- `effective_method: XTBMethod` [只读值字段]
- `task: XTBTask` [只读值字段]
- `charge: int` [只读值字段]
- `unpaired_electrons: Optional[int]` [只读值字段]
- `work_directory: Path` [只读值字段]
- `argv: Tuple[str, ...]` [只读值字段]
- `return_code: int` [只读值字段]
- `stdout: str` [只读值字段]
- `stderr: str` [只读值字段]
- `elapsed_seconds: float` [只读值字段]
- `process_succeeded: bool` [只读值字段]
- `converged: bool` [只读值字段]
- `artifacts: Mapping[str, XTBArtifact]` [只读值字段]
- `provenance: ProcessProvenance` [只读值字段]
- `energy_hartree: Optional[float]` = `None` [只读值字段]
- `energy_unit: str` = `'hartree'` [只读值字段]
- `gradient_norm: Optional[float]` = `None` [只读值字段]
- `atom_order_verified: bool` = `False` [只读值字段]
- `coordinates_committed: bool` = `False` [只读值字段]
- `scientific_state_identity: Optional[ScientificStateIdentity]` = `None` [只读值字段]
- `facts_identity: Optional[MolecularFactsIdentity]` = `None` [只读值字段]
- `projection_convention: Optional[str]` = `None` [只读值字段]
- `workspace_retained: bool` = `False` [只读值字段]

<a id="hotpot-plugins-xtb-contracts-xtberror"></a>
### `XTBError`

定义 owner `hotpot.plugins.xtb.contracts.XTBError`；基类 `RuntimeError`。

- `sample_failure_code: Optional[str]` = `None` [字段声明；写权限依具体合同]

<a id="hotpot-plugins-xtb-contracts-xtbexecutableerror"></a>
### `XTBExecutableError`

定义 owner `hotpot.plugins.xtb.contracts.XTBExecutableError`；基类 `XTBError`。

- `__init__(self, message: str, process_result: Optional[NativeProcessResult]=None) -> None`
- `process_result` = `process_result` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `sample_failure_code: Optional[str]` = `None` [字段声明；写权限依具体合同] ← `hotpot.plugins.xtb.contracts.XTBError`

<a id="hotpot-plugins-xtb-contracts-xtbinputerror"></a>
### `XTBInputError`

定义 owner `hotpot.plugins.xtb.contracts.XTBInputError`；基类 `XTBError`。

- `sample_failure_code: Optional[str]` = `None` [字段声明；写权限依具体合同] ← `hotpot.plugins.xtb.contracts.XTBError`

<a id="hotpot-plugins-xtb-contracts-xtbapplicabilityerror"></a>
### `XTBApplicabilityError`

定义 owner `hotpot.plugins.xtb.contracts.XTBApplicabilityError`；基类 `XTBInputError`。

- `sample_failure_code: Optional[str]` = `'XTB-ELEMENT-UNSUPPORTED'` [字段声明；写权限依具体合同]

<a id="hotpot-plugins-xtb-contracts-xtbexecutionerror"></a>
### `XTBExecutionError`

定义 owner `hotpot.plugins.xtb.contracts.XTBExecutionError`；基类 `XTBError`。

- `__init__(self, message: str, report: XTBRunReport) -> None`
- `sample_failure_code: Optional[str]` = `'XTB-NATIVE-EXECUTION'` [字段声明；写权限依具体合同]
- `report` = `report` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）

<a id="hotpot-plugins-xtb-contracts-xtbresulterror"></a>
### `XTBResultError`

定义 owner `hotpot.plugins.xtb.contracts.XTBResultError`；基类 `XTBError`。

- `__init__(self, message: str, report: XTBRunReport) -> None`
- `sample_failure_code: Optional[str]` = `'XTB-RESULT-UNAVAILABLE'` [字段声明；写权限依具体合同]
- `report` = `report` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）


<a id="hotpot-plugins-xtb-runner"></a>
## `hotpot.plugins.xtb.runner`

`hotpot/plugins/xtb/runner.py`；显式 __all__: ('run_xtb',)


<a id="hotpot-plugins-xtb-runner-run-xtb"></a>
### `run_xtb`

```python
run_xtb(request: XTBRequest) -> XTBRunReport
```


<a id="hotpot-plugins-xtb-stage"></a>
## `hotpot.plugins.xtb.stage`

`hotpot/plugins/xtb/stage.py`；显式 __all__: ('STAGE', 'get_stage')


<a id="hotpot-plugins-xtb-stage-get-stage"></a>
### `get_stage`

```python
get_stage() -> MolecularStage
```

模块值/类型别名（声明存在不构成事实写许可）：

- `STAGE` = `_XTBStage()`


<a id="hotpot-plugins-xtb-state-projection"></a>
## `hotpot.plugins.xtb.state_projection`

`hotpot/plugins/xtb/state_projection.py`；显式 __all__: ('XTBBackendContext',)


<a id="hotpot-plugins-xtb-state-projection-xtbbackendcontext"></a>
### `XTBBackendContext`

定义 owner `hotpot.plugins.xtb.state_projection.XTBBackendContext`；基类 无。

```python
XTBBackendContext(backend_info: XTBBackendInfo, method: XTBMethod)
```

- `supports_atomic_number(self, atomic_number: int) -> bool`
- `effective_nuclear_charge(self, atomic_number: int) -> int`
- `identity` [只读]: `BackendIdentity`
- `scientific_operation_requirements` [只读]: `ScientificOperationRequirements`
- `projection_convention` [只读]: `str`
- `electron_count_definition` [只读]: `str`
- `consumes_uhf` [只读]: `bool`
- `method_supported` [只读]: `bool`
- `backend_info: XTBBackendInfo` [只读值字段]
- `method: XTBMethod` [只读值字段]


<a id="hotpot-plugins-xtb-state-resolution"></a>
## `hotpot.plugins.xtb.state_resolution`

`hotpot/plugins/xtb/state_resolution.py`；显式 __all__: 无


<a id="hotpot-plugins-xtb-state-resolution-resolvedxtbstate"></a>
### `ResolvedXTBState`

定义 owner `hotpot.plugins.xtb.state_resolution.ResolvedXTBState`；基类 无。

```python
ResolvedXTBState(scientific_state: ScientificStateBundle, projection: BackendStateProjection)
```

- `scientific_state: ScientificStateBundle` [只读值字段]
- `projection: BackendStateProjection` [只读值字段]

<a id="hotpot-plugins-xtb-state-resolution-xtbstateresolutionerror"></a>
### `XTBStateResolutionError`

定义 owner `hotpot.plugins.xtb.state_resolution.XTBStateResolutionError`；基类 `XTBInputError`。

- `__init__(self, scientific_state: ScientificStateBundle, projection_result: BackendProjectionResult) -> None`
- `sample_failure_code: Optional[str]` = `'XTB-STATE-UNAVAILABLE'` [字段声明；写权限依具体合同]
- `scientific_state` = `scientific_state` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `projection_result` = `projection_result` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）

<a id="hotpot-plugins-xtb-state-resolution-resolve-xtb-state"></a>
### `resolve_xtb_state`

```python
resolve_xtb_state(mol: Molecule, method: XTBMethod, *, executable: Optional[_ExecutablePath], environment: Mapping[str, str], charge: Optional[int], uhf: Optional[int], policy: StatePolicy, carried_state: Optional[ScientificStateBundle]=None, expected_scientific_identity: Optional[ScientificStateIdentity]=None, expected_facts_identity: Optional[MolecularFactsIdentity]=None) -> ResolvedXTBState
```


<a id="hotpot-plugins-xtb-stream"></a>
## `hotpot.plugins.xtb.stream`

`hotpot/plugins/xtb/stream.py`；显式 __all__: ('XTBStreamError', 'XTBStreamMetadata', 'XTBStreamProvenance', 'XTBStreamRecord', 'metadata_from_report', 'read_sdf_records', 'write_sdf_records')


<a id="hotpot-plugins-xtb-stream-xtbstreamerror"></a>
### `XTBStreamError`

定义 owner `hotpot.plugins.xtb.stream.XTBStreamError`；基类 `ValueError`。


<a id="hotpot-plugins-xtb-stream-xtbstreamprovenance"></a>
### `XTBStreamProvenance`

定义 owner `hotpot.plugins.xtb.stream.XTBStreamProvenance`；基类 无。

```python
XTBStreamProvenance(backend_version: str, backend_revision: Optional[str], executable_sha256: str, scientific_state_identity: Optional[ScientificStateIdentity] = None, facts_identity: Optional[MolecularFactsIdentity] = None, projection_convention: Optional[str] = None)
```

- `backend_version: str` [只读值字段]
- `backend_revision: Optional[str]` [只读值字段]
- `executable_sha256: str` [只读值字段]
- `scientific_state_identity: Optional[ScientificStateIdentity]` = `None` [只读值字段]
- `facts_identity: Optional[MolecularFactsIdentity]` = `None` [只读值字段]
- `projection_convention: Optional[str]` = `None` [只读值字段]

<a id="hotpot-plugins-xtb-stream-xtbstreammetadata"></a>
### `XTBStreamMetadata`

定义 owner `hotpot.plugins.xtb.stream.XTBStreamMetadata`；基类 无。

```python
XTBStreamMetadata(total_charge: int, unpaired_electrons: Optional[int], method: XTBMethod, energy_hartree: float, provenance: XTBStreamProvenance)
```

- `total_charge: int` [只读值字段]
- `unpaired_electrons: Optional[int]` [只读值字段]
- `method: XTBMethod` [只读值字段]
- `energy_hartree: float` [只读值字段]
- `provenance: XTBStreamProvenance` [只读值字段]

<a id="hotpot-plugins-xtb-stream-xtbstreamrecord"></a>
### `XTBStreamRecord`

定义 owner `hotpot.plugins.xtb.stream.XTBStreamRecord`；基类 无。

```python
XTBStreamRecord(mol: Molecule, metadata: Optional[XTBStreamMetadata] = None)
```

- `mol: Molecule` [只读值字段]
- `metadata: Optional[XTBStreamMetadata]` = `None` [只读值字段]

<a id="hotpot-plugins-xtb-stream-metadata-from-report"></a>
### `metadata_from_report`

```python
metadata_from_report(report: XTBRunReport, previous: Optional[XTBStreamMetadata]=None) -> XTBStreamMetadata
```

<a id="hotpot-plugins-xtb-stream-read-sdf-records"></a>
### `read_sdf_records`

```python
read_sdf_records(text: str) -> Tuple[XTBStreamRecord, ...]
```

<a id="hotpot-plugins-xtb-stream-write-sdf-records"></a>
### `write_sdf_records`

```python
write_sdf_records(records: Iterable[XTBStreamRecord]) -> str
```


<a id="hotpot-plugins-xtb-workflow"></a>
## `hotpot.plugins.xtb.workflow`

`hotpot/plugins/xtb/workflow.py`；显式 __all__: ('run_gfn_xtb', 'run_gfnff')


<a id="hotpot-plugins-xtb-workflow-run-gfnff"></a>
### `run_gfnff`

```python
run_gfnff(mol: Molecule, *, state: BackendStateProjection, task: XTBTask=XTBTask.OPTIMIZE, executable: Optional[_ExecutablePath]=None, environment: Optional[Mapping[str, str]]=None, timeout_seconds: Optional[float]=None, work_directory: Optional[Path]=None, keep_work_directory: bool=False) -> XTBRunReport
```

<a id="hotpot-plugins-xtb-workflow-run-gfn-xtb"></a>
### `run_gfn_xtb`

```python
run_gfn_xtb(mol: Molecule, *, state: BackendStateProjection, method: GFNXTBMethod=GFNXTBMethod.GFN2_XTB, task: XTBTask=XTBTask.OPTIMIZE, executable: Optional[_ExecutablePath]=None, environment: Optional[Mapping[str, str]]=None, timeout_seconds: Optional[float]=None, work_directory: Optional[Path]=None, keep_work_directory: bool=False) -> XTBRunReport
```
