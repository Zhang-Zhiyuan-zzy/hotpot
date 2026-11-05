# 外部系统与运输 adapter：全签名目录

来源：当前静态源码；显式 __all__ 与直接模块定义分别列明。类型定义 owner 与公开导出路径分别保留。
只读 property、动态 owner setter、不可变值字段与可重绑定类声明值不是同一权限概念。外部继承保持可归属的边界，未声称第三方运行时属性穷尽。
[完整树](../delivery_api_tree.md) · [动态权限](dynamic_permissions.md) · [native 声明](native_contracts.md)

## 模块导航

- [`hotpot.cheminfo.adapters`](#hotpot-cheminfo-adapters)
- [`hotpot.cheminfo.adapters._pyg_crystal`](#hotpot-cheminfo-adapters--pyg-crystal)
- [`hotpot.cheminfo.adapters._pyg_molecule`](#hotpot-cheminfo-adapters--pyg-molecule)
- [`hotpot.cheminfo.adapters.calculation`](#hotpot-cheminfo-adapters-calculation)
- [`hotpot.cheminfo.adapters.calculation.contracts`](#hotpot-cheminfo-adapters-calculation-contracts)
- [`hotpot.cheminfo.adapters.cif`](#hotpot-cheminfo-adapters-cif)
- [`hotpot.cheminfo.adapters.conversion`](#hotpot-cheminfo-adapters-conversion)
- [`hotpot.cheminfo.adapters.inference`](#hotpot-cheminfo-adapters-inference)
- [`hotpot.cheminfo.adapters.inference.all_atom_model`](#hotpot-cheminfo-adapters-inference-all-atom-model)
- [`hotpot.cheminfo.adapters.inference.artifacts`](#hotpot-cheminfo-adapters-inference-artifacts)
- [`hotpot.cheminfo.adapters.inference.cbond`](#hotpot-cheminfo-adapters-inference-cbond)
- [`hotpot.cheminfo.adapters.inference.device`](#hotpot-cheminfo-adapters-inference-device)
- [`hotpot.cheminfo.adapters.inference.features`](#hotpot-cheminfo-adapters-inference-features)
- [`hotpot.cheminfo.adapters.inference.runtime`](#hotpot-cheminfo-adapters-inference-runtime)
- [`hotpot.cheminfo.adapters.inference.serialization`](#hotpot-cheminfo-adapters-inference-serialization)
- [`hotpot.cheminfo.adapters.inference.system`](#hotpot-cheminfo-adapters-inference-system)
- [`hotpot.cheminfo.adapters.io`](#hotpot-cheminfo-adapters-io)
- [`hotpot.cheminfo.adapters.io.base`](#hotpot-cheminfo-adapters-io-base)
- [`hotpot.cheminfo.adapters.io.formats`](#hotpot-cheminfo-adapters-io-formats)
- [`hotpot.cheminfo.adapters.io.formats.g16`](#hotpot-cheminfo-adapters-io-formats-g16)
- [`hotpot.cheminfo.adapters.io.formats.openbabel`](#hotpot-cheminfo-adapters-io-formats-openbabel)
- [`hotpot.cheminfo.adapters.io.formats.pubchem`](#hotpot-cheminfo-adapters-io-formats-pubchem)
- [`hotpot.cheminfo.adapters.native.forcefields`](#hotpot-cheminfo-adapters-native-forcefields)
- [`hotpot.cheminfo.adapters.native.forcefields.contracts`](#hotpot-cheminfo-adapters-native-forcefields-contracts)
- [`hotpot.cheminfo.adapters.native.forcefields.native`](#hotpot-cheminfo-adapters-native-forcefields-native)
- [`hotpot.cheminfo.adapters.native.forcefields.reports`](#hotpot-cheminfo-adapters-native-forcefields-reports)
- [`hotpot.cheminfo.adapters.native.molecule_buffers`](#hotpot-cheminfo-adapters-native-molecule-buffers)
- [`hotpot.cheminfo.adapters.native.substructure`](#hotpot-cheminfo-adapters-native-substructure)
- [`hotpot.cheminfo.adapters.native.substructure.matching`](#hotpot-cheminfo-adapters-native-substructure-matching)
- [`hotpot.cheminfo.adapters.openbabel`](#hotpot-cheminfo-adapters-openbabel)
- [`hotpot.cheminfo.adapters.openbabel._runtime`](#hotpot-cheminfo-adapters-openbabel--runtime)
- [`hotpot.cheminfo.adapters.openbabel.conversion`](#hotpot-cheminfo-adapters-openbabel-conversion)
- [`hotpot.cheminfo.adapters.openbabel.crystal`](#hotpot-cheminfo-adapters-openbabel-crystal)
- [`hotpot.cheminfo.adapters.openbabel.forcefields`](#hotpot-cheminfo-adapters-openbabel-forcefields)
- [`hotpot.cheminfo.adapters.openbabel.perception`](#hotpot-cheminfo-adapters-openbabel-perception)
- [`hotpot.cheminfo.adapters.openbabel.stereo_transport`](#hotpot-cheminfo-adapters-openbabel-stereo-transport)
- [`hotpot.cheminfo.adapters.pyg`](#hotpot-cheminfo-adapters-pyg)
- [`hotpot.cheminfo.adapters.rdkit`](#hotpot-cheminfo-adapters-rdkit)
- [`hotpot.cheminfo.adapters.rdkit.conformers`](#hotpot-cheminfo-adapters-rdkit-conformers)
- [`hotpot.cheminfo.adapters.rdkit.conversion`](#hotpot-cheminfo-adapters-rdkit-conversion)
- [`hotpot.cheminfo.adapters.rdkit.sampling`](#hotpot-cheminfo-adapters-rdkit-sampling)
- [`hotpot.cheminfo.adapters.rdkit.stereo_transport`](#hotpot-cheminfo-adapters-rdkit-stereo-transport)
- [`hotpot.cheminfo.adapters.structure`](#hotpot-cheminfo-adapters-structure)
- [`hotpot.cheminfo.adapters.structure.crystal`](#hotpot-cheminfo-adapters-structure-crystal)
- [`hotpot.cheminfo.adapters.structure.forcefields`](#hotpot-cheminfo-adapters-structure-forcefields)
- [`hotpot.cheminfo.adapters.structure.geometry`](#hotpot-cheminfo-adapters-structure-geometry)
- [`hotpot.cheminfo.adapters.structure.inference`](#hotpot-cheminfo-adapters-structure-inference)
- [`hotpot.cheminfo.adapters.structure.scientific_state`](#hotpot-cheminfo-adapters-structure-scientific-state)
- [`hotpot.cheminfo.adapters.structure.stereochemistry`](#hotpot-cheminfo-adapters-structure-stereochemistry)
- [`hotpot.cheminfo.adapters.structure.substructure`](#hotpot-cheminfo-adapters-structure-substructure)
- [`hotpot.cheminfo.adapters.structure.substructure_query`](#hotpot-cheminfo-adapters-structure-substructure-query)
- [`hotpot.cheminfo.adapters.thermo`](#hotpot-cheminfo-adapters-thermo)
- [`hotpot.cheminfo.adapters.thermo.chemical`](#hotpot-cheminfo-adapters-thermo-chemical)
- [`hotpot.cheminfo.adapters.visualization`](#hotpot-cheminfo-adapters-visualization)
- [`hotpot.cheminfo.adapters.visualization.draw`](#hotpot-cheminfo-adapters-visualization-draw)

<a id="hotpot-cheminfo-adapters"></a>
## `hotpot.cheminfo.adapters`

`hotpot/cheminfo/adapters/__init__.py`；显式 __all__: ()



<a id="hotpot-cheminfo-adapters--pyg-crystal"></a>
## `hotpot.cheminfo.adapters._pyg_crystal`

`hotpot/cheminfo/adapters/_pyg_crystal.py`；显式 __all__: 无


<a id="hotpot-cheminfo-adapters--pyg-crystal-encode-cell"></a>
### `encode_cell`

```python
encode_cell(cell, atom_count, bond_count)
```

<a id="hotpot-cheminfo-adapters--pyg-crystal-decode-cell"></a>
### `decode_cell`

```python
decode_cell(data, key_rows, bond_keys)
```

<a id="hotpot-cheminfo-adapters--pyg-crystal-foreign-cell"></a>
### `foreign_cell`

```python
foreign_cell(selected, atom_count, bond_count)
```

模块值/类型别名（声明存在不构成事实写许可）：

- `CELL_FIELDS` = `('cell_matrix', 'cell_present', 'bond_image', 'bond_image_present', 'occupancy', 'occupancy_present', 'space_group_R', 'space_group_t', 'hotpot_crystal')`


<a id="hotpot-cheminfo-adapters--pyg-molecule"></a>
## `hotpot.cheminfo.adapters._pyg_molecule`

`hotpot/cheminfo/adapters/_pyg_molecule.py`；显式 __all__: 无


<a id="hotpot-cheminfo-adapters--pyg-molecule-capture-details"></a>
### `capture_details`

```python
capture_details(record, metadata, *, properties, atom_uids, source_frames, generated, selected_frame, partial_charge, mca, mca_join)
```

<a id="hotpot-cheminfo-adapters--pyg-molecule-encode"></a>
### `encode`

```python
encode(record, details, available)
```

<a id="hotpot-cheminfo-adapters--pyg-molecule-decode"></a>
### `decode`

```python
decode(data, field_map)
```

模块值/类型别名（声明存在不构成事实写许可）：

- `SCHEMA` = `'hotpot.molecule/2'`
- `KINDS` = `('unknown', 'zero', 'single', 'double', 'triple', 'aromatic', 'dative')`
- `ATOM_COLUMNS` = `('atomic_number', 'formal_charge', 'implicit_hydrogens', 'is_aromatic', 'id', 'x', 'y', 'z')`
- `NODE_FIELDS` = `('z', 'formal_charge', 'total_h', 'aromatic', 'atom_id')`
- `EDGE_FIELDS` = `('bond_key', 'bond_order', 'bond_kind', 'bond_direction', 'bond_aromatic', 'bond_id', 'bond_constraint')`
- `INDEX_WIDTHS` = `{'edge_index': 2, 'angle_index': 3, 'torsion_index': 4, 'wiberg_index': 2}`


<a id="hotpot-cheminfo-adapters-calculation"></a>
## `hotpot.cheminfo.adapters.calculation`

`hotpot/cheminfo/adapters/calculation/__init__.py`；显式 __all__: 无



<a id="hotpot-cheminfo-adapters-calculation-contracts"></a>
## `hotpot.cheminfo.adapters.calculation.contracts`

`hotpot/cheminfo/adapters/calculation/contracts.py`；显式 __all__: ('BackendIdentity', 'BackendContext')


<a id="hotpot-cheminfo-adapters-calculation-contracts-backendidentity"></a>
### `BackendIdentity`

定义 owner `hotpot.cheminfo.adapters.calculation.contracts.BackendIdentity`；基类 无。

```python
BackendIdentity(digest: str)
```

- `digest: str` [只读值字段]

<a id="hotpot-cheminfo-adapters-calculation-contracts-backendcontext"></a>
### `BackendContext`

定义 owner `hotpot.cheminfo.adapters.calculation.contracts.BackendContext`；基类 `Protocol`。

- `supports_atomic_number(self, atomic_number: int) -> bool`
- `effective_nuclear_charge(self, atomic_number: int) -> int`
- `identity` [只读]: `BackendIdentity`
- `scientific_operation_requirements` [只读]: `ScientificOperationRequirements`
- `projection_convention` [只读]: `str`
- `consumes_uhf` [只读]: `bool`
- `method_supported` [只读]: `bool`


<a id="hotpot-cheminfo-adapters-cif"></a>
## `hotpot.cheminfo.adapters.cif`

`hotpot/cheminfo/adapters/cif.py`；显式 __all__: 无



<a id="hotpot-cheminfo-adapters-conversion"></a>
## `hotpot.cheminfo.adapters.conversion`

`hotpot/cheminfo/adapters/conversion.py`；显式 __all__: 无



<a id="hotpot-cheminfo-adapters-inference"></a>
## `hotpot.cheminfo.adapters.inference`

`hotpot/cheminfo/adapters/inference/__init__.py`；显式 __all__: ()



<a id="hotpot-cheminfo-adapters-inference-all-atom-model"></a>
## `hotpot.cheminfo.adapters.inference.all_atom_model`

`hotpot/cheminfo/adapters/inference/all_atom_model.py`；显式 __all__: 无


<a id="hotpot-cheminfo-adapters-inference-all-atom-model-unsupportedallatomoptimization"></a>
### `UnsupportedAllAtomOptimization`

定义 owner `hotpot.cheminfo.adapters.inference.all_atom_model.UnsupportedAllAtomOptimization`；基类 `RuntimeError`。


<a id="hotpot-cheminfo-adapters-inference-all-atom-model-materialize-all-atom-model"></a>
### `materialize_all_atom_model`

```python
materialize_all_atom_model(source: Path, external_digest: str | None=None)
```


<a id="hotpot-cheminfo-adapters-inference-artifacts"></a>
## `hotpot.cheminfo.adapters.inference.artifacts`

`hotpot/cheminfo/adapters/inference/artifacts.py`；显式 __all__: 无


<a id="hotpot-cheminfo-adapters-inference-artifacts-read-json"></a>
### `read_json`

```python
read_json(path: Path) -> dict
```

<a id="hotpot-cheminfo-adapters-inference-artifacts-sha256"></a>
### `sha256`

```python
sha256(path: Path) -> str
```

<a id="hotpot-cheminfo-adapters-inference-artifacts-bundle-sha256"></a>
### `bundle_sha256`

```python
bundle_sha256(paths: list[Path]) -> str
```

<a id="hotpot-cheminfo-adapters-inference-artifacts-is-file"></a>
### `is_file`

```python
is_file(path: Path) -> bool
```

<a id="hotpot-cheminfo-adapters-inference-artifacts-external-files"></a>
### `external_files`

```python
external_files(directory: Path, pattern: str) -> tuple[tuple[Path, ...], int]
```

<a id="hotpot-cheminfo-adapters-inference-artifacts-download-snapshot"></a>
### `download_snapshot`

```python
download_snapshot(*, repo_id: str, revision: str, allow_patterns: list[str], cache_dir: Path, local_dir: Path, local_files_only: bool)
```


<a id="hotpot-cheminfo-adapters-inference-cbond"></a>
## `hotpot.cheminfo.adapters.inference.cbond`

`hotpot/cheminfo/adapters/inference/cbond.py`；显式 __all__: 无


<a id="hotpot-cheminfo-adapters-inference-cbond-padding-rings"></a>
### `padding_rings`

```python
padding_rings(xg, rings_node_index, rings_node_nums)
```

<a id="hotpot-cheminfo-adapters-inference-cbond-resolve-device"></a>
### `resolve_device`

```python
resolve_device(device: str) -> str
```

<a id="hotpot-cheminfo-adapters-inference-cbond-create-sessions"></a>
### `create_sessions`

```python
create_sessions(graph_path, cbond_path, requested_device, device)
```

<a id="hotpot-cheminfo-adapters-inference-cbond-embed-graph"></a>
### `embed_graph`

```python
embed_graph(session, x, edge_index)
```

<a id="hotpot-cheminfo-adapters-inference-cbond-predict"></a>
### `predict`

```python
predict(session, xg, padded_rings, rings_mask, cbond_index)
```

模块值/类型别名（声明存在不构成事实写许可）：

- `MAX_RINGS_NUMS` = `32`
- `MAX_RINGS_SIZE` = `64`


<a id="hotpot-cheminfo-adapters-inference-device"></a>
## `hotpot.cheminfo.adapters.inference.device`

`hotpot/cheminfo/adapters/inference/device.py`；显式 __all__: 无


<a id="hotpot-cheminfo-adapters-inference-device-gpudevicememory"></a>
### `GPUDeviceMemory`

定义 owner `hotpot.cheminfo.adapters.inference.device.GPUDeviceMemory`；基类 无。

```python
GPUDeviceMemory(gpu_index: int, total_bytes: int, free_bytes: int | None = None)
```

- `gpu_index: int` [只读值字段]
- `total_bytes: int` [只读值字段]
- `free_bytes: int | None` = `None` [只读值字段]

<a id="hotpot-cheminfo-adapters-inference-device-parse-nvidia-smi-memory"></a>
### `parse_nvidia_smi_memory`

```python
parse_nvidia_smi_memory(output: str, gpu_index: int) -> GPUDeviceMemory
```

<a id="hotpot-cheminfo-adapters-inference-device-gpu-memory-bytes"></a>
### `gpu_memory_bytes`

```python
gpu_memory_bytes(gpu_index: int) -> GPUDeviceMemory
```

<a id="hotpot-cheminfo-adapters-inference-device-gpu-total-memory-bytes"></a>
### `gpu_total_memory_bytes`

```python
gpu_total_memory_bytes(gpu_index: int) -> int
```

<a id="hotpot-cheminfo-adapters-inference-device-gpu-total-memories-bytes"></a>
### `gpu_total_memories_bytes`

```python
gpu_total_memories_bytes(gpu_indices: Sequence[int], memory_query: Callable[[int], int] | None=None) -> tuple[int, ...]
```

模块值/类型别名（声明存在不构成事实写许可）：

- `MIB` = `1024 ** 2`


<a id="hotpot-cheminfo-adapters-inference-features"></a>
## `hotpot.cheminfo.adapters.inference.features`

`hotpot/cheminfo/adapters/inference/features.py`；显式 __all__: 无


<a id="hotpot-cheminfo-adapters-inference-features-mol-to-unimolv2"></a>
### `mol_to_unimolv2`

```python
mol_to_unimolv2(mol: Chem.Mol, max_atoms: int=512)
```

<a id="hotpot-cheminfo-adapters-inference-features-collate-molecule-rows"></a>
### `collate_molecule_rows`

```python
collate_molecule_rows(features, molecule_indices, max_atoms=None)
```

模块值/类型别名（声明存在不构成事实写许可）：

- `ALLOWABLE` = `<长模块值；见 hotpot/cheminfo/adapters/inference/features.py:12>`


<a id="hotpot-cheminfo-adapters-inference-runtime"></a>
## `hotpot.cheminfo.adapters.inference.runtime`

`hotpot/cheminfo/adapters/inference/runtime.py`；显式 __all__: 无


<a id="hotpot-cheminfo-adapters-inference-runtime-cuda-provider-available"></a>
### `cuda_provider_available`

```python
cuda_provider_available() -> bool
```

<a id="hotpot-cheminfo-adapters-inference-runtime-mcaruntime"></a>
### `MCARuntime`

定义 owner `hotpot.cheminfo.adapters.inference.runtime.MCARuntime`；基类 无。

- `__init__(self, session_path, *, backend: str, variant: str, manifest: Mapping[str, object], cuda_device_id: int | None=None, cpu_threads: int | None=None)`
- `predict(self, arrays, batch_size: int)`
- `close(self)`
- `variant` = `variant` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `manifest` = `manifest` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `session` = `ort.InferenceSession(str(session_path), sess_options=options, providers=providers)` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `input_dtypes` = `{input_.name: ORT_DTYPES[input_.type] for input_ in self.session.get_inputs()}` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）

<a id="hotpot-cheminfo-adapters-inference-runtime-create-cuda-runtime-from-artifact"></a>
### `create_cuda_runtime_from_artifact`

```python
create_cuda_runtime_from_artifact(*, session_path, variant: str, max_atoms: int, device_id: int) -> MCARuntime
```

模块值/类型别名（声明存在不构成事实写许可）：

- `ORT_DTYPES` = `{'tensor(float)': np.float32, 'tensor(float16)': np.float16, 'tensor(int32)': np.int32, 'tensor(int64)': np.int64}`


<a id="hotpot-cheminfo-adapters-inference-serialization"></a>
## `hotpot.cheminfo.adapters.inference.serialization`

`hotpot/cheminfo/adapters/inference/serialization.py`；显式 __all__: 无


<a id="hotpot-cheminfo-adapters-inference-serialization-copy-feature-molecule"></a>
### `copy_feature_molecule`

```python
copy_feature_molecule(molecule, *, sanitize: bool)
```

<a id="hotpot-cheminfo-adapters-inference-serialization-encode-molecule"></a>
### `encode_molecule`

```python
encode_molecule(molecule) -> bytes
```

<a id="hotpot-cheminfo-adapters-inference-serialization-decode-molecule"></a>
### `decode_molecule`

```python
decode_molecule(binary: bytes)
```


<a id="hotpot-cheminfo-adapters-inference-system"></a>
## `hotpot.cheminfo.adapters.inference.system`

`hotpot/cheminfo/adapters/inference/system.py`；显式 __all__: 无


<a id="hotpot-cheminfo-adapters-inference-system-current-cpu-affinity"></a>
### `current_cpu_affinity`

```python
current_cpu_affinity() -> tuple[int, ...]
```

<a id="hotpot-cheminfo-adapters-inference-system-pin-current-process"></a>
### `pin_current_process`

```python
pin_current_process(cpu_ids: int | Sequence[int]) -> None
```

<a id="hotpot-cheminfo-adapters-inference-system-pin-worker-from-queue"></a>
### `pin_worker_from_queue`

```python
pin_worker_from_queue(cpu_queue) -> None
```


<a id="hotpot-cheminfo-adapters-io"></a>
## `hotpot.cheminfo.adapters.io`

`hotpot/cheminfo/adapters/io/__init__.py`；显式 __all__: ()



<a id="hotpot-cheminfo-adapters-io-base"></a>
## `hotpot.cheminfo.adapters.io.base`

`hotpot/cheminfo/adapters/io/base.py`；显式 __all__: 无


<a id="hotpot-cheminfo-adapters-io-base-iobase"></a>
### `IoBase`

定义 owner `hotpot.cheminfo.adapters.io.base.IoBase`；基类 无。

- `__init__(self, src, fmt=None, **kwargs)`
- `src` = `src` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `src_type` = `self._src_checks(src)` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `fmt` = `self._determine_fmt(src, fmt)` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `ob_opt` = `kwargs.pop('ob_opt', {})` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `kwargs` = `kwargs` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）


<a id="hotpot-cheminfo-adapters-io-formats"></a>
## `hotpot.cheminfo.adapters.io.formats`

`hotpot/cheminfo/adapters/io/formats/__init__.py`；显式 __all__: ()



<a id="hotpot-cheminfo-adapters-io-formats-g16"></a>
## `hotpot.cheminfo.adapters.io.formats.g16`

`hotpot/cheminfo/adapters/io/formats/g16.py`；显式 __all__: 无


<a id="hotpot-cheminfo-adapters-io-formats-g16-from-gjf"></a>
### `from_gjf`

```python
from_gjf(script: str) -> 'Molecule'
```


<a id="hotpot-cheminfo-adapters-io-formats-openbabel"></a>
## `hotpot.cheminfo.adapters.io.formats.openbabel`

`hotpot/cheminfo/adapters/io/formats/openbabel.py`；显式 __all__: 无


<a id="hotpot-cheminfo-adapters-io-formats-openbabel-get-ob-conversion"></a>
### `get_ob_conversion`

```python
get_ob_conversion(fmt='smi', **kwargs)
```

<a id="hotpot-cheminfo-adapters-io-formats-openbabel-write-obmol-to-string"></a>
### `write_obmol_to_string`

```python
write_obmol_to_string(obmol: 'ob.OBMol', fmt='smi', **kwargs)
```


<a id="hotpot-cheminfo-adapters-io-formats-pubchem"></a>
## `hotpot.cheminfo.adapters.io.formats.pubchem`

`hotpot/cheminfo/adapters/io/formats/pubchem.py`；显式 __all__: 无



<a id="hotpot-cheminfo-adapters-native-forcefields"></a>
## `hotpot.cheminfo.adapters.native.forcefields`

`hotpot/cheminfo/adapters/native/forcefields/__init__.py`；显式 __all__: 无



<a id="hotpot-cheminfo-adapters-native-forcefields-contracts"></a>
## `hotpot.cheminfo.adapters.native.forcefields.contracts`

`hotpot/cheminfo/adapters/native/forcefields/contracts.py`；显式 __all__: ('ComplexSessionInput',)


<a id="hotpot-cheminfo-adapters-native-forcefields-contracts-complexsessioninput"></a>
### `ComplexSessionInput`

定义 owner `hotpot.cheminfo.adapters.native.forcefields.contracts.ComplexSessionInput`；基类 无。

```python
ComplexSessionInput(schema_version: int, atomic_numbers: NDArray[np.int32], formal_charges: NDArray[np.int32], partial_charges: NDArray[np.float64], coordinates: NDArray[np.float64], atom_aromatic: NDArray[np.uint8], ligand_bond_indices: NDArray[np.int32], ligand_bond_orders: NDArray[np.float64], ligand_bond_kinds: NDArray[np.uint8], ligand_bond_aromatic: NDArray[np.uint8], metal_indices: NDArray[np.int32], intended_coordination_bonds: NDArray[np.int32], intended_coordination_orders: NDArray[np.float64], intended_coordination_kinds: NDArray[np.uint8], unit_cell: Optional[NDArray[np.float64]])
```

- `schema_version: int` [只读值字段]
- `atomic_numbers: NDArray[np.int32]` [只读值字段]
- `formal_charges: NDArray[np.int32]` [只读值字段]
- `partial_charges: NDArray[np.float64]` [只读值字段]
- `coordinates: NDArray[np.float64]` [只读值字段]
- `atom_aromatic: NDArray[np.uint8]` [只读值字段]
- `ligand_bond_indices: NDArray[np.int32]` [只读值字段]
- `ligand_bond_orders: NDArray[np.float64]` [只读值字段]
- `ligand_bond_kinds: NDArray[np.uint8]` [只读值字段]
- `ligand_bond_aromatic: NDArray[np.uint8]` [只读值字段]
- `metal_indices: NDArray[np.int32]` [只读值字段]
- `intended_coordination_bonds: NDArray[np.int32]` [只读值字段]
- `intended_coordination_orders: NDArray[np.float64]` [只读值字段]
- `intended_coordination_kinds: NDArray[np.uint8]` [只读值字段]
- `unit_cell: Optional[NDArray[np.float64]]` [只读值字段]


<a id="hotpot-cheminfo-adapters-native-forcefields-native"></a>
## `hotpot.cheminfo.adapters.native.forcefields.native`

`hotpot/cheminfo/adapters/native/forcefields/native.py`；显式 __all__: ('ComplexOptimizationOptions', 'ConvergenceLevel', 'DEFAULT_CONVERGENCE_LEVEL', 'CoordinationStageOptions', 'FrameDetail', 'MetalPlacementOptions', 'RingScreeningOptions', 'StructureSnapshot', 'assess_metal_position', 'create_coordination_session', 'create_optimization_session', 'optimize_complex', 'place_metal', 'place_metals', 'restore_coordination', 'run_complex_workflow', 'run_complex_workflow_from_input', 'set_coordination_active_mask', 'set_ligand_bond_active_mask', 'snapshot_structure', 'update_structure_coordinates')

- 导出 `ConvergenceLevel` → `hotpot.cheminfo.obWrappers.contracts.ConvergenceLevel`
- 导出 `DEFAULT_CONVERGENCE_LEVEL` → `hotpot.cheminfo.obWrappers.contracts.DEFAULT_CONVERGENCE_LEVEL`

<a id="hotpot-cheminfo-adapters-native-forcefields-native-framedetail"></a>
### `FrameDetail`

定义 owner `hotpot.cheminfo.adapters.native.forcefields.native.FrameDetail`；基类 `str`, `Enum`。

- `NONE` = `'none'` [枚举常量]
- `OPTIMIZATION` = `'optimization'` [枚举常量]
- `ALL_ATTEMPTS` = `'all_attempts'` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-cheminfo-adapters-native-forcefields-native-metalplacementoptions"></a>
### `MetalPlacementOptions`

定义 owner `hotpot.cheminfo.adapters.native.forcefields.native.MetalPlacementOptions`；基类 无。

```python
MetalPlacementOptions(maximum_candidate_count: int = 96, fibonacci_direction_count: int = 32, sphere_intersection_count: int = 24, least_squares_iteration_count: int = 16, maximum_actionable_ring_size: int = _BOND_RING_MAX_SIZE, coordination_distance_scale: float = 1.0, coordination_distance_ratio_minimum: float = 0.7, coordination_distance_ratio_maximum: float = 1.5, absolute_center_clearance_angstrom: float = 0.5, center_covalent_radius_scale: float = 0.55, minimum_path_atom_clearance: float = 0.35, minimum_path_bond_clearance: float = 0.5, broad_phase_skin_angstrom: float = 0.25, duplicate_tolerance_angstrom: float = 1e-08, retain_candidate_evidence: bool = False, geometry_settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS)
```

- `maximum_candidate_count: int` = `96` [只读值字段]
- `fibonacci_direction_count: int` = `32` [只读值字段]
- `sphere_intersection_count: int` = `24` [只读值字段]
- `least_squares_iteration_count: int` = `16` [只读值字段]
- `maximum_actionable_ring_size: int` = `_BOND_RING_MAX_SIZE` [只读值字段]
- `coordination_distance_scale: float` = `1.0` [只读值字段]
- `coordination_distance_ratio_minimum: float` = `0.7` [只读值字段]
- `coordination_distance_ratio_maximum: float` = `1.5` [只读值字段]
- `absolute_center_clearance_angstrom: float` = `0.5` [只读值字段]
- `center_covalent_radius_scale: float` = `0.55` [只读值字段]
- `minimum_path_atom_clearance: float` = `0.35` [只读值字段]
- `minimum_path_bond_clearance: float` = `0.5` [只读值字段]
- `broad_phase_skin_angstrom: float` = `0.25` [只读值字段]
- `duplicate_tolerance_angstrom: float` = `1e-08` [只读值字段]
- `retain_candidate_evidence: bool` = `False` [只读值字段]
- `geometry_settings: GeometrySettings` = `DEFAULT_GEOMETRY_SETTINGS` [只读值字段]

<a id="hotpot-cheminfo-adapters-native-forcefields-native-ringscreeningoptions"></a>
### `RingScreeningOptions`

定义 owner `hotpot.cheminfo.adapters.native.forcefields.native.RingScreeningOptions`；基类 无。

```python
RingScreeningOptions(maximum_actionable_ring_size: int = _BOND_RING_MAX_SIZE, maximum_relevant_cycle_count: int = _MAXIMUM_RELEVANT_CYCLE_COUNT, geometry_settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS)
```

- `maximum_actionable_ring_size: int` = `_BOND_RING_MAX_SIZE` [只读值字段]
- `maximum_relevant_cycle_count: int` = `_MAXIMUM_RELEVANT_CYCLE_COUNT` [只读值字段]
- `geometry_settings: GeometrySettings` = `DEFAULT_GEOMETRY_SETTINGS` [只读值字段]

<a id="hotpot-cheminfo-adapters-native-forcefields-native-coordinationstageoptions"></a>
### `CoordinationStageOptions`

定义 owner `hotpot.cheminfo.adapters.native.forcefields.native.CoordinationStageOptions`；基类 无。

```python
CoordinationStageOptions(forcefield: str = 'UFF', attempt_limit: int = 20, relaxation_steps: int = 100, perturb_sigma: float = 0.5, trajectory_start: TrajectoryStart = TrajectoryStart.COORDINATION_RESTORATION, frame_detail: FrameDetail = FrameDetail.NONE, torsion_singularity_threshold: float = TORSION_SINGULARITY_THRESHOLD, torsion_repair_angle_radians: float = TORSION_REPAIR_ANGLE_RADIANS, placement: MetalPlacementOptions = field(default_factory=MetalPlacementOptions))
```

- `forcefield: str` = `'UFF'` [只读值字段]
- `attempt_limit: int` = `20` [只读值字段]
- `relaxation_steps: int` = `100` [只读值字段]
- `perturb_sigma: float` = `0.5` [只读值字段]
- `trajectory_start: TrajectoryStart` = `TrajectoryStart.COORDINATION_RESTORATION` [只读值字段]
- `frame_detail: FrameDetail` = `FrameDetail.NONE` [只读值字段]
- `torsion_singularity_threshold: float` = `TORSION_SINGULARITY_THRESHOLD` [只读值字段]
- `torsion_repair_angle_radians: float` = `TORSION_REPAIR_ANGLE_RADIANS` [只读值字段]
- `placement: MetalPlacementOptions` = `field(default_factory=MetalPlacementOptions)` [只读值字段]

<a id="hotpot-cheminfo-adapters-native-forcefields-native-complexoptimizationoptions"></a>
### `ComplexOptimizationOptions`

定义 owner `hotpot.cheminfo.adapters.native.forcefields.native.ComplexOptimizationOptions`；基类 无。

```python
ComplexOptimizationOptions(forcefield: str = 'UFF', algorithm: str = 'conjugate', epochs: int = 100, steps_per_epoch: int = 100, untangling_attempt_limit: int = 30, perturb_interval: Optional[int] = None, perturb_sigma: float = 0.5, trajectory_start: TrajectoryStart = TrajectoryStart.COMPLEX_UNTANGLING, frame_detail: FrameDetail = FrameDetail.NONE, retain_epoch_history: bool = False, increasing_vdw: bool = False, vdw_cutoff_start: float = 0.0, vdw_cutoff_end: float = 12.5, energy_tolerance: float = 1e-06, stopping: Optional[OptimizationStoppingCriteria] = None, torsion_singularity_threshold: float = TORSION_SINGULARITY_THRESHOLD, torsion_repair_angle_radians: float = TORSION_REPAIR_ANGLE_RADIANS, ring_screening: RingScreeningOptions = field(default_factory=RingScreeningOptions), convergence_level: ConvergenceLevel = DEFAULT_CONVERGENCE_LEVEL)
```

- `forcefield: str` = `'UFF'` [只读值字段]
- `algorithm: str` = `'conjugate'` [只读值字段]
- `epochs: int` = `100` [只读值字段]
- `steps_per_epoch: int` = `100` [只读值字段]
- `untangling_attempt_limit: int` = `30` [只读值字段]
- `perturb_interval: Optional[int]` = `None` [只读值字段]
- `perturb_sigma: float` = `0.5` [只读值字段]
- `trajectory_start: TrajectoryStart` = `TrajectoryStart.COMPLEX_UNTANGLING` [只读值字段]
- `frame_detail: FrameDetail` = `FrameDetail.NONE` [只读值字段]
- `retain_epoch_history: bool` = `False` [只读值字段]
- `increasing_vdw: bool` = `False` [只读值字段]
- `vdw_cutoff_start: float` = `0.0` [只读值字段]
- `vdw_cutoff_end: float` = `12.5` [只读值字段]
- `energy_tolerance: float` = `1e-06` [只读值字段]
- `stopping: Optional[OptimizationStoppingCriteria]` = `None` [只读值字段]
- `torsion_singularity_threshold: float` = `TORSION_SINGULARITY_THRESHOLD` [只读值字段]
- `torsion_repair_angle_radians: float` = `TORSION_REPAIR_ANGLE_RADIANS` [只读值字段]
- `ring_screening: RingScreeningOptions` = `field(default_factory=RingScreeningOptions)` [只读值字段]
- `convergence_level: ConvergenceLevel` = `DEFAULT_CONVERGENCE_LEVEL` [只读值字段]

<a id="hotpot-cheminfo-adapters-native-forcefields-native-structuresnapshot"></a>
### `StructureSnapshot`

定义 owner `hotpot.cheminfo.adapters.native.forcefields.native.StructureSnapshot`；基类 无。

```python
StructureSnapshot(coordinates: NDArray[np.float64], active_ligand_bond_mask: NDArray[np.uint8], active_coordination_mask: NDArray[np.uint8], component_ids: NDArray[np.int32], ligand_bond_count: int, active_bond_count: int, coordinate_revision: int, topology_revision: int)
```

- `coordinates: NDArray[np.float64]` [只读值字段]
- `active_ligand_bond_mask: NDArray[np.uint8]` [只读值字段]
- `active_coordination_mask: NDArray[np.uint8]` [只读值字段]
- `component_ids: NDArray[np.int32]` [只读值字段]
- `ligand_bond_count: int` [只读值字段]
- `active_bond_count: int` [只读值字段]
- `coordinate_revision: int` [只读值字段]
- `topology_revision: int` [只读值字段]

<a id="hotpot-cheminfo-adapters-native-forcefields-native-create-coordination-session"></a>
### `create_coordination_session`

```python
create_coordination_session(source: Union['Molecule', ComplexSessionInput]) -> '_ob_native.StructureSession'
```

<a id="hotpot-cheminfo-adapters-native-forcefields-native-create-optimization-session"></a>
### `create_optimization_session`

```python
create_optimization_session(source: Union['Molecule', ComplexSessionInput]) -> '_ob_native.StructureSession'
```

<a id="hotpot-cheminfo-adapters-native-forcefields-native-snapshot-structure"></a>
### `snapshot_structure`

```python
snapshot_structure(session: '_ob_native.StructureSession') -> StructureSnapshot
```

<a id="hotpot-cheminfo-adapters-native-forcefields-native-update-structure-coordinates"></a>
### `update_structure_coordinates`

```python
update_structure_coordinates(session: '_ob_native.StructureSession', coordinates: NDArray[np.float64]) -> None
```

<a id="hotpot-cheminfo-adapters-native-forcefields-native-set-coordination-active-mask"></a>
### `set_coordination_active_mask`

```python
set_coordination_active_mask(session: '_ob_native.StructureSession', active_mask: NDArray[np.uint8]) -> None
```

<a id="hotpot-cheminfo-adapters-native-forcefields-native-set-ligand-bond-active-mask"></a>
### `set_ligand_bond_active_mask`

```python
set_ligand_bond_active_mask(session: '_ob_native.StructureSession', active_mask: NDArray[np.uint8]) -> None
```

<a id="hotpot-cheminfo-adapters-native-forcefields-native-assess-metal-position"></a>
### `assess_metal_position`

```python
assess_metal_position(session: '_ob_native.StructureSession', metal_index: int, coordinates: tuple[float, float, float], *, options: MetalPlacementOptions=_DEFAULT_METAL_PLACEMENT_OPTIONS) -> '_ob_native.PlacementCandidateEvidence'
```

<a id="hotpot-cheminfo-adapters-native-forcefields-native-place-metal"></a>
### `place_metal`

```python
place_metal(session: '_ob_native.StructureSession', metal_index: int, *, options: MetalPlacementOptions=_DEFAULT_METAL_PLACEMENT_OPTIONS) -> '_ob_native.MetalPlacementResult'
```

<a id="hotpot-cheminfo-adapters-native-forcefields-native-place-metals"></a>
### `place_metals`

```python
place_metals(session: '_ob_native.StructureSession', *, options: MetalPlacementOptions=_DEFAULT_METAL_PLACEMENT_OPTIONS) -> '_ob_native.MetalPlacementReport'
```

<a id="hotpot-cheminfo-adapters-native-forcefields-native-restore-coordination"></a>
### `restore_coordination`

```python
restore_coordination(session: '_ob_native.StructureSession', perturbation_offsets: NDArray[np.float64], *, options: CoordinationStageOptions=CoordinationStageOptions()) -> NativeCoordinationStageResult
```

<a id="hotpot-cheminfo-adapters-native-forcefields-native-optimize-complex"></a>
### `optimize_complex`

```python
optimize_complex(session: '_ob_native.StructureSession', untangling_offsets: NDArray[np.float64], optimization_offsets: NDArray[np.float64], *, options: ComplexOptimizationOptions=ComplexOptimizationOptions()) -> NativeComplexOptimizationResult
```

<a id="hotpot-cheminfo-adapters-native-forcefields-native-run-complex-workflow"></a>
### `run_complex_workflow`

```python
run_complex_workflow(session: '_ob_native.StructureSession', coordination_offsets: NDArray[np.float64], untangling_offsets: NDArray[np.float64], optimization_offsets: NDArray[np.float64], *, coordination_options: CoordinationStageOptions=CoordinationStageOptions(), optimization_options: ComplexOptimizationOptions=ComplexOptimizationOptions()) -> NativeComplexWorkflowResult
```

<a id="hotpot-cheminfo-adapters-native-forcefields-native-run-complex-workflow-from-input"></a>
### `run_complex_workflow_from_input`

```python
run_complex_workflow_from_input(source: Union['Molecule', ComplexSessionInput], coordination_offsets: NDArray[np.float64], untangling_offsets: NDArray[np.float64], optimization_offsets: NDArray[np.float64], *, coordination_options: CoordinationStageOptions=CoordinationStageOptions(), optimization_options: ComplexOptimizationOptions=ComplexOptimizationOptions()) -> NativeComplexWorkflowResult
```


<a id="hotpot-cheminfo-adapters-native-forcefields-reports"></a>
## `hotpot.cheminfo.adapters.native.forcefields.reports`

`hotpot/cheminfo/adapters/native/forcefields/reports.py`；显式 __all__: ('ComplexOptimizationResult', 'ComplexWorkflowResult', 'CoordinationStageResult', 'NativeBondRingFinding', 'NativeRingCheckpointReport', 'NativeRingGraphScope', 'NativeStageStatus', 'NativeTopologyRevision', 'NativeTrajectoryBatch', 'unpack_native_topology_revisions')


<a id="hotpot-cheminfo-adapters-native-forcefields-reports-nativestagestatus"></a>
### `NativeStageStatus`

定义 owner `hotpot.cheminfo.adapters.native.forcefields.reports.NativeStageStatus`；基类 `str`, `Enum`。

- `COMPLETED` = `'completed'` [枚举常量]
- `PARTIAL` = `'partial'` [枚举常量]
- `FAILED` = `'failed'` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-cheminfo-adapters-native-forcefields-reports-nativeringgraphscope"></a>
### `NativeRingGraphScope`

定义 owner `hotpot.cheminfo.adapters.native.forcefields.reports.NativeRingGraphScope`；基类 `str`, `Enum`。

- `LIGAND_SKELETON` = `'ligand_skeleton'` [枚举常量]
- `FULL_GRAPH` = `'full_graph'` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-cheminfo-adapters-native-forcefields-reports-nativebondringfinding"></a>
### `NativeBondRingFinding`

定义 owner `hotpot.cheminfo.adapters.native.forcefields.reports.NativeBondRingFinding`；基类 无。

```python
NativeBondRingFinding(ring_index: int, ring_atom_indices: Tuple[int, ...], bond_key: Tuple[int, int], state: PiercingState, indeterminacy_causes: Tuple[SegmentCycleIndeterminacy, ...], aabb_separated: bool, surface_complete: bool)
```

- `ring_index: int` [只读值字段]
- `ring_atom_indices: Tuple[int, ...]` [只读值字段]
- `bond_key: Tuple[int, int]` [只读值字段]
- `state: PiercingState` [只读值字段]
- `indeterminacy_causes: Tuple[SegmentCycleIndeterminacy, ...]` [只读值字段]
- `aabb_separated: bool` [只读值字段]
- `surface_complete: bool` [只读值字段]

<a id="hotpot-cheminfo-adapters-native-forcefields-reports-nativeringcheckpointreport"></a>
### `NativeRingCheckpointReport`

定义 owner `hotpot.cheminfo.adapters.native.forcefields.reports.NativeRingCheckpointReport`；基类 无。

```python
NativeRingCheckpointReport(state: PiercingState, scope: NativeRingGraphScope, maximum_actionable_ring_size: int, maximum_relevant_cycle_count: int, relevant_cycle_count: int, selected_ring_count: int, excluded_ring_count: int, active_bond_count: int, candidate_pair_count: int, aabb_separated_pair_count: int, exact_pair_count: int, piercing_pair_count: int, does_not_pierce_pair_count: int, undetermined_pair_count: int, scan_complete: bool, actionable_findings: Tuple[NativeBondRingFinding, ...])
```

- `state: PiercingState` [只读值字段]
- `scope: NativeRingGraphScope` [只读值字段]
- `maximum_actionable_ring_size: int` [只读值字段]
- `maximum_relevant_cycle_count: int` [只读值字段]
- `relevant_cycle_count: int` [只读值字段]
- `selected_ring_count: int` [只读值字段]
- `excluded_ring_count: int` [只读值字段]
- `active_bond_count: int` [只读值字段]
- `candidate_pair_count: int` [只读值字段]
- `aabb_separated_pair_count: int` [只读值字段]
- `exact_pair_count: int` [只读值字段]
- `piercing_pair_count: int` [只读值字段]
- `does_not_pierce_pair_count: int` [只读值字段]
- `undetermined_pair_count: int` [只读值字段]
- `scan_complete: bool` [只读值字段]
- `actionable_findings: Tuple[NativeBondRingFinding, ...]` [只读值字段]

<a id="hotpot-cheminfo-adapters-native-forcefields-reports-nativetopologyrevision"></a>
### `NativeTopologyRevision`

定义 owner `hotpot.cheminfo.adapters.native.forcefields.reports.NativeTopologyRevision`；基类 无。

```python
NativeTopologyRevision(active_ligand_bond_mask: NDArray[np.uint8], active_coordination_bond_mask: NDArray[np.uint8])
```

- `active_ligand_bond_mask: NDArray[np.uint8]` [只读值字段]
- `active_coordination_bond_mask: NDArray[np.uint8]` [只读值字段]

<a id="hotpot-cheminfo-adapters-native-forcefields-reports-nativetrajectorybatch"></a>
### `NativeTrajectoryBatch`

定义 owner `hotpot.cheminfo.adapters.native.forcefields.reports.NativeTrajectoryBatch`；基类 无。

```python
NativeTrajectoryBatch(coordinates: NDArray[np.float64], start: TrajectoryStart, stages: Tuple[TrajectoryStage, ...], events: Tuple[TrajectoryEvent, ...], component_indices: Tuple[Optional[int], ...], attempts: Tuple[Optional[int], ...], steps: Tuple[Optional[int], ...], energies_kj_mol: NDArray[np.float64], evidence: Tuple[Optional[FrameEvidence], ...], topology_revisions: Tuple[NativeTopologyRevision, ...], frame_topology_revisions: NDArray[np.int32], selected_frame_index: Optional[int], terminal_frame_index: Optional[int])
```

- `frame_count` [只读]: `int`
- `coordinates: NDArray[np.float64]` [只读值字段]
- `start: TrajectoryStart` [只读值字段]
- `stages: Tuple[TrajectoryStage, ...]` [只读值字段]
- `events: Tuple[TrajectoryEvent, ...]` [只读值字段]
- `component_indices: Tuple[Optional[int], ...]` [只读值字段]
- `attempts: Tuple[Optional[int], ...]` [只读值字段]
- `steps: Tuple[Optional[int], ...]` [只读值字段]
- `energies_kj_mol: NDArray[np.float64]` [只读值字段]
- `evidence: Tuple[Optional[FrameEvidence], ...]` [只读值字段]
- `topology_revisions: Tuple[NativeTopologyRevision, ...]` [只读值字段]
- `frame_topology_revisions: NDArray[np.int32]` [只读值字段]
- `selected_frame_index: Optional[int]` [只读值字段]
- `terminal_frame_index: Optional[int]` [只读值字段]

<a id="hotpot-cheminfo-adapters-native-forcefields-reports-coordinationstageresult"></a>
### `CoordinationStageResult`

定义 owner `hotpot.cheminfo.adapters.native.forcefields.reports.CoordinationStageResult`；基类 无。

```python
CoordinationStageResult(status: NativeStageStatus, selected_coordinates: NDArray[np.float64], terminal_coordinates: NDArray[np.float64], final_active_coordination_mask: NDArray[np.uint8], attempt_limit: int, attempts_completed: int, metal_relocation_attempt_count: int, relocated_metal_indices: Tuple[int, ...], infeasible_metal_indices: Tuple[int, ...], forced_bond_keys: Tuple[Tuple[int, int], ...], rejected_piercing_trial_count: int, undetermined_trial_count: int, excluded_ring_observation_count: int, warning_codes: Tuple[str, ...], trajectory: NativeTrajectoryBatch, bond_count: int, placement_report: '_ob_native.MetalPlacementReport', elapsed_seconds: float)
```

- `status: NativeStageStatus` [只读值字段]
- `selected_coordinates: NDArray[np.float64]` [只读值字段]
- `terminal_coordinates: NDArray[np.float64]` [只读值字段]
- `final_active_coordination_mask: NDArray[np.uint8]` [只读值字段]
- `attempt_limit: int` [只读值字段]
- `attempts_completed: int` [只读值字段]
- `metal_relocation_attempt_count: int` [只读值字段]
- `relocated_metal_indices: Tuple[int, ...]` [只读值字段]
- `infeasible_metal_indices: Tuple[int, ...]` [只读值字段]
- `forced_bond_keys: Tuple[Tuple[int, int], ...]` [只读值字段]
- `rejected_piercing_trial_count: int` [只读值字段]
- `undetermined_trial_count: int` [只读值字段]
- `excluded_ring_observation_count: int` [只读值字段]
- `warning_codes: Tuple[str, ...]` [只读值字段]
- `trajectory: NativeTrajectoryBatch` [只读值字段]
- `bond_count: int` [只读值字段]
- `placement_report: '_ob_native.MetalPlacementReport'` [只读值字段]
- `elapsed_seconds: float` [只读值字段]

<a id="hotpot-cheminfo-adapters-native-forcefields-reports-complexoptimizationresult"></a>
### `ComplexOptimizationResult`

定义 owner `hotpot.cheminfo.adapters.native.forcefields.reports.ComplexOptimizationResult`；基类 无。

```python
ComplexOptimizationResult(status: NativeStageStatus, selected_coordinates: NDArray[np.float64], terminal_coordinates: NDArray[np.float64], final_active_coordination_mask: NDArray[np.uint8], untangling_attempt_limit: int, untangling_attempts_completed: int, initial_piercing_count: int, final_piercing_count: int, minimum_piercing_count: int, untangling_resolved: bool, selected_frame_index: int, best_epoch: int, final_energy_kj_mol: float, best_energy_kj_mol: float, rms_gradient_kj_mol_angstrom: float, max_gradient_kj_mol_angstrom: float, energy_changes: Tuple[float, ...], max_displacements: Tuple[float, ...], epoch_energies: Tuple[float, ...], exploded: bool, converged: bool, terminal_converged: bool, epochs_completed: int, steps_submitted: int, initialization_steps: int, selected_segment_epochs_completed: int, backend_energy_unit: str, termination_reason: str, warning_codes: Tuple[str, ...], trajectory: NativeTrajectoryBatch, final_checkpoint: NativeRingCheckpointReport, elapsed_seconds: float)
```

- `status: NativeStageStatus` [只读值字段]
- `selected_coordinates: NDArray[np.float64]` [只读值字段]
- `terminal_coordinates: NDArray[np.float64]` [只读值字段]
- `final_active_coordination_mask: NDArray[np.uint8]` [只读值字段]
- `untangling_attempt_limit: int` [只读值字段]
- `untangling_attempts_completed: int` [只读值字段]
- `initial_piercing_count: int` [只读值字段]
- `final_piercing_count: int` [只读值字段]
- `minimum_piercing_count: int` [只读值字段]
- `untangling_resolved: bool` [只读值字段]
- `selected_frame_index: int` [只读值字段]
- `best_epoch: int` [只读值字段]
- `final_energy_kj_mol: float` [只读值字段]
- `best_energy_kj_mol: float` [只读值字段]
- `rms_gradient_kj_mol_angstrom: float` [只读值字段]
- `max_gradient_kj_mol_angstrom: float` [只读值字段]
- `energy_changes: Tuple[float, ...]` [只读值字段]
- `max_displacements: Tuple[float, ...]` [只读值字段]
- `epoch_energies: Tuple[float, ...]` [只读值字段]
- `exploded: bool` [只读值字段]
- `converged: bool` [只读值字段]
- `terminal_converged: bool` [只读值字段]
- `epochs_completed: int` [只读值字段]
- `steps_submitted: int` [只读值字段]
- `initialization_steps: int` [只读值字段]
- `selected_segment_epochs_completed: int` [只读值字段]
- `backend_energy_unit: str` [只读值字段]
- `termination_reason: str` [只读值字段]
- `warning_codes: Tuple[str, ...]` [只读值字段]
- `trajectory: NativeTrajectoryBatch` [只读值字段]
- `final_checkpoint: NativeRingCheckpointReport` [只读值字段]
- `elapsed_seconds: float` [只读值字段]

<a id="hotpot-cheminfo-adapters-native-forcefields-reports-complexworkflowresult"></a>
### `ComplexWorkflowResult`

定义 owner `hotpot.cheminfo.adapters.native.forcefields.reports.ComplexWorkflowResult`；基类 无。

```python
ComplexWorkflowResult(coordination: CoordinationStageResult, optimization: ComplexOptimizationResult, selected_coordinates: NDArray[np.float64], terminal_coordinates: NDArray[np.float64], final_active_coordination_mask: NDArray[np.uint8], warning_codes: Tuple[str, ...], trajectory: NativeTrajectoryBatch)
```

- `coordination: CoordinationStageResult` [只读值字段]
- `optimization: ComplexOptimizationResult` [只读值字段]
- `selected_coordinates: NDArray[np.float64]` [只读值字段]
- `terminal_coordinates: NDArray[np.float64]` [只读值字段]
- `final_active_coordination_mask: NDArray[np.uint8]` [只读值字段]
- `warning_codes: Tuple[str, ...]` [只读值字段]
- `trajectory: NativeTrajectoryBatch` [只读值字段]

<a id="hotpot-cheminfo-adapters-native-forcefields-reports-unpack-native-topology-revisions"></a>
### `unpack_native_topology_revisions`

```python
unpack_native_topology_revisions(session_input: ComplexSessionInput, revisions: Sequence['NativeTopologyRevision']) -> Tuple[Tuple['BondTopology', ...], ...]
```


<a id="hotpot-cheminfo-adapters-native-molecule-buffers"></a>
## `hotpot.cheminfo.adapters.native.molecule_buffers`

`hotpot/cheminfo/adapters/native/molecule_buffers.py`；显式 __all__: 无


<a id="hotpot-cheminfo-adapters-native-molecule-buffers-nativebondkindcode"></a>
### `NativeBondKindCode`

定义 owner `hotpot.cheminfo.adapters.native.molecule_buffers.NativeBondKindCode`；基类 `IntEnum`。

- `SINGLE` = `1` [枚举常量]
- `DOUBLE` = `2` [枚举常量]
- `TRIPLE` = `3` [枚举常量]
- `AROMATIC` = `4` [枚举常量]
- `ZERO` = `5` [枚举常量]
- `DATIVE` = `6` [枚举常量]
- `UNKNOWN` = `7` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-cheminfo-adapters-native-molecule-buffers-nativemoleculefield"></a>
### `NativeMoleculeField`

定义 owner `hotpot.cheminfo.adapters.native.molecule_buffers.NativeMoleculeField`；基类 `Enum`。

- `ATOMIC_NUMBERS` = `'atomic_numbers'` [枚举常量]
- `FORMAL_CHARGES` = `'formal_charges'` [枚举常量]
- `PARTIAL_CHARGES` = `'partial_charges'` [枚举常量]
- `COORDINATES` = `'coordinates'` [枚举常量]
- `ATOM_AROMATIC` = `'atom_aromatic'` [枚举常量]
- `BOND_INDICES` = `'bond_indices'` [枚举常量]
- `BOND_ORDERS` = `'bond_orders'` [枚举常量]
- `BOND_KINDS` = `'bond_kinds'` [枚举常量]
- `BOND_AROMATIC` = `'bond_aromatic'` [枚举常量]
- `UNIT_CELL` = `'unit_cell'` [枚举常量]
- `STEREO_RECORDS` = `'stereo_records'` [枚举常量]
- `ATOM_TRANSPORT` = `'atom_transport'` [枚举常量]
- `BOND_TRANSPORT` = `'bond_transport'` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-cheminfo-adapters-native-molecule-buffers-nativestereoanchorkindcode"></a>
### `NativeStereoAnchorKindCode`

定义 owner `hotpot.cheminfo.adapters.native.molecule_buffers.NativeStereoAnchorKindCode`；基类 `IntEnum`。

- `ATOM` = `1` [枚举常量]
- `BOND` = `2` [枚举常量]
- `AXIS` = `3` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-cheminfo-adapters-native-molecule-buffers-nativestereocarrierkindcode"></a>
### `NativeStereoCarrierKindCode`

定义 owner `hotpot.cheminfo.adapters.native.molecule_buffers.NativeStereoCarrierKindCode`；基类 `IntEnum`。

- `ATOM` = `1` [枚举常量]
- `IMPLICIT_H` = `2` [枚举常量]
- `LONE_PAIR` = `3` [枚举常量]
- `VACANCY` = `4` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-cheminfo-adapters-native-molecule-buffers-nativestereogroupkindcode"></a>
### `NativeStereoGroupKindCode`

定义 owner `hotpot.cheminfo.adapters.native.molecule_buffers.NativeStereoGroupKindCode`；基类 `IntEnum`。

- `NONE` = `0` [枚举常量]
- `ABS` = `1` [枚举常量]
- `AND` = `2` [枚举常量]
- `OR` = `3` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-cheminfo-adapters-native-molecule-buffers-nativeatomtransportfieldcode"></a>
### `NativeAtomTransportFieldCode`

定义 owner `hotpot.cheminfo.adapters.native.molecule_buffers.NativeAtomTransportFieldCode`；基类 `IntFlag`。

- `ISOTOPE` = `1 << 0` [枚举常量]
- `RADICAL_ELECTRONS` = `1 << 1` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-cheminfo-adapters-native-molecule-buffers-nativehydrogentransportfieldcode"></a>
### `NativeHydrogenTransportFieldCode`

定义 owner `hotpot.cheminfo.adapters.native.molecule_buffers.NativeHydrogenTransportFieldCode`；基类 `IntFlag`。

- `IMPLICIT` = `1 << 0` [枚举常量]
- `BRACKET_EXPLICIT` = `1 << 1` [枚举常量]
- `NO_IMPLICIT` = `1 << 2` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-cheminfo-adapters-native-molecule-buffers-nativeutf8table"></a>
### `NativeUtf8Table`

定义 owner `hotpot.cheminfo.adapters.native.molecule_buffers.NativeUtf8Table`；基类 无。

```python
NativeUtf8Table(data: NDArray[np.uint8], offsets: NDArray[np.uint64], lengths: NDArray[np.uint64], present: NDArray[np.uint8])
```

- `data: NDArray[np.uint8]` [只读值字段]
- `offsets: NDArray[np.uint64]` [只读值字段]
- `lengths: NDArray[np.uint64]` [只读值字段]
- `present: NDArray[np.uint8]` [只读值字段]

<a id="hotpot-cheminfo-adapters-native-molecule-buffers-nativestereorecordbuffers"></a>
### `NativeStereoRecordBuffers`

定义 owner `hotpot.cheminfo.adapters.native.molecule_buffers.NativeStereoRecordBuffers`；基类 无。

```python
NativeStereoRecordBuffers(schema_version: int, anchor_kinds: NDArray[np.uint32], anchor_offsets: NDArray[np.uint64], anchor_counts: NDArray[np.uint64], anchor_indices: NDArray[np.uint64], carrier_offsets: NDArray[np.uint64], carrier_counts: NDArray[np.uint64], carrier_kinds: NDArray[np.uint32], carrier_has_atoms: NDArray[np.uint32], carrier_atoms: NDArray[np.uint64], group_kinds: NDArray[np.uint32], group_ids: NDArray[np.int64], group_present: NDArray[np.uint32], group_scopes: NativeUtf8Table, descriptors: NativeUtf8Table, source_namespaces: NativeUtf8Table, source_codes: NativeUtf8Table)
```

- `schema_version: int` [只读值字段]
- `anchor_kinds: NDArray[np.uint32]` [只读值字段]
- `anchor_offsets: NDArray[np.uint64]` [只读值字段]
- `anchor_counts: NDArray[np.uint64]` [只读值字段]
- `anchor_indices: NDArray[np.uint64]` [只读值字段]
- `carrier_offsets: NDArray[np.uint64]` [只读值字段]
- `carrier_counts: NDArray[np.uint64]` [只读值字段]
- `carrier_kinds: NDArray[np.uint32]` [只读值字段]
- `carrier_has_atoms: NDArray[np.uint32]` [只读值字段]
- `carrier_atoms: NDArray[np.uint64]` [只读值字段]
- `group_kinds: NDArray[np.uint32]` [只读值字段]
- `group_ids: NDArray[np.int64]` [只读值字段]
- `group_present: NDArray[np.uint32]` [只读值字段]
- `group_scopes: NativeUtf8Table` [只读值字段]
- `descriptors: NativeUtf8Table` [只读值字段]
- `source_namespaces: NativeUtf8Table` [只读值字段]
- `source_codes: NativeUtf8Table` [只读值字段]

<a id="hotpot-cheminfo-adapters-native-molecule-buffers-nativeatomtransportbuffers"></a>
### `NativeAtomTransportBuffers`

定义 owner `hotpot.cheminfo.adapters.native.molecule_buffers.NativeAtomTransportBuffers`；基类 无。

```python
NativeAtomTransportBuffers(schema_version: int, atoms: NDArray[np.uint64], presence_masks: NDArray[np.uint32], isotopes: NDArray[np.int32], radical_electrons: NDArray[np.int32], has_current_hydrogens: NDArray[np.uint32], current_hydrogen_presence_masks: NDArray[np.uint32], current_implicit_hydrogens: NDArray[np.int32], current_bracket_explicit_hydrogens: NDArray[np.int32], current_no_implicit: NDArray[np.uint32], has_source_hydrogens: NDArray[np.uint32], source_hydrogen_presence_masks: NDArray[np.uint32], source_implicit_hydrogens: NDArray[np.int32], source_bracket_explicit_hydrogens: NDArray[np.int32], source_no_implicit: NDArray[np.uint32], source_namespaces: NativeUtf8Table)
```

- `schema_version: int` [只读值字段]
- `atoms: NDArray[np.uint64]` [只读值字段]
- `presence_masks: NDArray[np.uint32]` [只读值字段]
- `isotopes: NDArray[np.int32]` [只读值字段]
- `radical_electrons: NDArray[np.int32]` [只读值字段]
- `has_current_hydrogens: NDArray[np.uint32]` [只读值字段]
- `current_hydrogen_presence_masks: NDArray[np.uint32]` [只读值字段]
- `current_implicit_hydrogens: NDArray[np.int32]` [只读值字段]
- `current_bracket_explicit_hydrogens: NDArray[np.int32]` [只读值字段]
- `current_no_implicit: NDArray[np.uint32]` [只读值字段]
- `has_source_hydrogens: NDArray[np.uint32]` [只读值字段]
- `source_hydrogen_presence_masks: NDArray[np.uint32]` [只读值字段]
- `source_implicit_hydrogens: NDArray[np.int32]` [只读值字段]
- `source_bracket_explicit_hydrogens: NDArray[np.int32]` [只读值字段]
- `source_no_implicit: NDArray[np.uint32]` [只读值字段]
- `source_namespaces: NativeUtf8Table` [只读值字段]

<a id="hotpot-cheminfo-adapters-native-molecule-buffers-nativebondtransportbuffers"></a>
### `NativeBondTransportBuffers`

定义 owner `hotpot.cheminfo.adapters.native.molecule_buffers.NativeBondTransportBuffers`；基类 无。

```python
NativeBondTransportBuffers(schema_version: int, atom_indices: NDArray[np.uint64], display_directions: NativeUtf8Table)
```

- `schema_version: int` [只读值字段]
- `atom_indices: NDArray[np.uint64]` [只读值字段]
- `display_directions: NativeUtf8Table` [只读值字段]

<a id="hotpot-cheminfo-adapters-native-molecule-buffers-nativemoleculebuffers"></a>
### `NativeMoleculeBuffers`

定义 owner `hotpot.cheminfo.adapters.native.molecule_buffers.NativeMoleculeBuffers`；基类 无。

```python
NativeMoleculeBuffers(schema_version: int, atom_count: int, bond_count: int, atomic_numbers: NDArray[np.int32] | None = None, formal_charges: NDArray[np.int32] | None = None, partial_charges: NDArray[np.float64] | None = None, coordinates: NDArray[np.float64] | None = None, atom_aromatic: NDArray[np.uint8] | None = None, bond_indices: NDArray[np.int32] | None = None, bond_orders: NDArray[np.float64] | None = None, bond_kinds: NDArray[np.uint8] | None = None, bond_aromatic: NDArray[np.uint8] | None = None, unit_cell: NDArray[np.float64] | None = None, stereo_records: NativeStereoRecordBuffers | None = None, atom_transport: NativeAtomTransportBuffers | None = None, bond_transport: NativeBondTransportBuffers | None = None)
```

- `schema_version: int` [只读值字段]
- `atom_count: int` [只读值字段]
- `bond_count: int` [只读值字段]
- `atomic_numbers: NDArray[np.int32] | None` = `None` [只读值字段]
- `formal_charges: NDArray[np.int32] | None` = `None` [只读值字段]
- `partial_charges: NDArray[np.float64] | None` = `None` [只读值字段]
- `coordinates: NDArray[np.float64] | None` = `None` [只读值字段]
- `atom_aromatic: NDArray[np.uint8] | None` = `None` [只读值字段]
- `bond_indices: NDArray[np.int32] | None` = `None` [只读值字段]
- `bond_orders: NDArray[np.float64] | None` = `None` [只读值字段]
- `bond_kinds: NDArray[np.uint8] | None` = `None` [只读值字段]
- `bond_aromatic: NDArray[np.uint8] | None` = `None` [只读值字段]
- `unit_cell: NDArray[np.float64] | None` = `None` [只读值字段]
- `stereo_records: NativeStereoRecordBuffers | None` = `None` [只读值字段]
- `atom_transport: NativeAtomTransportBuffers | None` = `None` [只读值字段]
- `bond_transport: NativeBondTransportBuffers | None` = `None` [只读值字段]

模块值/类型别名（声明存在不构成事实写许可）：

- `MOLECULE_BUFFER_SCHEMA_VERSION` = `1`


<a id="hotpot-cheminfo-adapters-native-substructure"></a>
## `hotpot.cheminfo.adapters.native.substructure`

`hotpot/cheminfo/adapters/native/substructure/__init__.py`；显式 __all__: 无



<a id="hotpot-cheminfo-adapters-native-substructure-matching"></a>
## `hotpot.cheminfo.adapters.native.substructure.matching`

`hotpot/cheminfo/adapters/native/substructure/matching.py`；显式 __all__: ()



<a id="hotpot-cheminfo-adapters-openbabel"></a>
## `hotpot.cheminfo.adapters.openbabel`

`hotpot/cheminfo/adapters/openbabel/__init__.py`；显式 __all__: ()



<a id="hotpot-cheminfo-adapters-openbabel--runtime"></a>
## `hotpot.cheminfo.adapters.openbabel._runtime`

`hotpot/cheminfo/adapters/openbabel/_runtime.py`；显式 __all__: ()



<a id="hotpot-cheminfo-adapters-openbabel-conversion"></a>
## `hotpot.cheminfo.adapters.openbabel.conversion`

`hotpot/cheminfo/adapters/openbabel/conversion.py`；显式 __all__: 无


<a id="hotpot-cheminfo-adapters-openbabel-conversion-extract-obatom-coordinate"></a>
### `extract_obatom_coordinate`

```python
extract_obatom_coordinate(obatom: 'ob.OBMol')
```

<a id="hotpot-cheminfo-adapters-openbabel-conversion-extract-obmol-coordinates"></a>
### `extract_obmol_coordinates`

```python
extract_obmol_coordinates(obmol: 'ob.OBMol') -> np.ndarray
```

<a id="hotpot-cheminfo-adapters-openbabel-conversion-set-obmol-coordinates"></a>
### `set_obmol_coordinates`

```python
set_obmol_coordinates(obmol: 'ob.OBMol', coords)
```

<a id="hotpot-cheminfo-adapters-openbabel-conversion-to-arrays"></a>
### `to_arrays`

```python
to_arrays(obmol)
```


<a id="hotpot-cheminfo-adapters-openbabel-crystal"></a>
## `hotpot.cheminfo.adapters.openbabel.crystal`

`hotpot/cheminfo/adapters/openbabel/crystal.py`；显式 __all__: 无


<a id="hotpot-cheminfo-adapters-openbabel-crystal-to-obcell"></a>
### `to_obcell`

```python
to_obcell(crystal) -> 'ob.OBUnitCell'
```


<a id="hotpot-cheminfo-adapters-openbabel-forcefields"></a>
## `hotpot.cheminfo.adapters.openbabel.forcefields`

`hotpot/cheminfo/adapters/openbabel/forcefields.py`；显式 __all__: ('build', 'single_optimize', 'optimize')

- 导出 `build` → `hotpot.cheminfo.obWrappers.builder.build`；`build(mol: 'Molecule', *, stereo_warnings: Optional[bool]=None) -> BuildReport`
- 导出 `single_optimize` → `hotpot.cheminfo.obWrappers.operation.single_optimize`；`single_optimize(mol: 'Molecule', forcefield: str, steps: int, *, singularity_threshold: float=TORSION_SINGULARITY_THRESHOLD, repair_angle_radians: float=TORSION_REPAIR_ANGLE_RADIANS) -> SingleOptimizationReport`
- 导出 `optimize` → `hotpot.cheminfo.obWrappers.forcefield.optimize`；`optimize(mol: 'Molecule', forcefield: str, *, algorithm: str='conjugate', epochs: int=100, steps_per_epoch: int=5, perturb_interval: Optional[int]=None, perturbation_offsets: Optional[np.ndarray]=None, retain_frames: bool=False, retain_epoch_history: bool=False, increasing_vdw: bool=False, vdw_cutoff_start: float=1.0, vdw_cutoff_end: float=10.0, energy_tolerance: float=1e-06, convergence_level: ConvergenceLevel=DEFAULT_CONVERGENCE_LEVEL, stopping_window: Optional[int]=None, maximum_energy_change_kj_mol: float=0.0001, maximum_atom_displacement_angstrom: float=0.0001, maximum_rms_gradient_kj_mol_angstrom: float=1.0, maximum_gradient_kj_mol_angstrom: float=5.0, singularity_threshold: float=TORSION_SINGULARITY_THRESHOLD, repair_angle_radians: float=TORSION_REPAIR_ANGLE_RADIANS) -> OptimizationReport`


<a id="hotpot-cheminfo-adapters-openbabel-perception"></a>
## `hotpot.cheminfo.adapters.openbabel.perception`

`hotpot/cheminfo/adapters/openbabel/perception.py`；显式 __all__: 无



<a id="hotpot-cheminfo-adapters-openbabel-stereo-transport"></a>
## `hotpot.cheminfo.adapters.openbabel.stereo_transport`

`hotpot/cheminfo/adapters/openbabel/stereo_transport.py`；显式 __all__: 无



<a id="hotpot-cheminfo-adapters-pyg"></a>
## `hotpot.cheminfo.adapters.pyg`

`hotpot/cheminfo/adapters/pyg.py`；显式 __all__: ()



<a id="hotpot-cheminfo-adapters-rdkit"></a>
## `hotpot.cheminfo.adapters.rdkit`

`hotpot/cheminfo/adapters/rdkit/__init__.py`；显式 __all__: ()



<a id="hotpot-cheminfo-adapters-rdkit-conformers"></a>
## `hotpot.cheminfo.adapters.rdkit.conformers`

`hotpot/cheminfo/adapters/rdkit/conformers.py`；显式 __all__: 无


<a id="hotpot-cheminfo-adapters-rdkit-conformers-ensure-3d-conformer"></a>
### `ensure_3d_conformer`

```python
ensure_3d_conformer(mol: Chem.Mol, seed: int=42) -> Chem.Mol
```


<a id="hotpot-cheminfo-adapters-rdkit-conversion"></a>
## `hotpot.cheminfo.adapters.rdkit.conversion`

`hotpot/cheminfo/adapters/rdkit/conversion.py`；显式 __all__: 无



<a id="hotpot-cheminfo-adapters-rdkit-sampling"></a>
## `hotpot.cheminfo.adapters.rdkit.sampling`

`hotpot/cheminfo/adapters/rdkit/sampling.py`；显式 __all__: 无


<a id="hotpot-cheminfo-adapters-rdkit-sampling-generate"></a>
### `generate`

```python
generate(transport, *, options, constraints=None)
```


<a id="hotpot-cheminfo-adapters-rdkit-stereo-transport"></a>
## `hotpot.cheminfo.adapters.rdkit.stereo_transport`

`hotpot/cheminfo/adapters/rdkit/stereo_transport.py`；显式 __all__: 无



<a id="hotpot-cheminfo-adapters-structure"></a>
## `hotpot.cheminfo.adapters.structure`

`hotpot/cheminfo/adapters/structure/__init__.py`；显式 __all__: 无



<a id="hotpot-cheminfo-adapters-structure-crystal"></a>
## `hotpot.cheminfo.adapters.structure.crystal`

`hotpot/cheminfo/adapters/structure/crystal.py`；显式 __all__: ()



<a id="hotpot-cheminfo-adapters-structure-forcefields"></a>
## `hotpot.cheminfo.adapters.structure.forcefields`

`hotpot/cheminfo/adapters/structure/forcefields.py`；显式 __all__: ('capture_topology', 'collect_coordination_environments')


<a id="hotpot-cheminfo-adapters-structure-forcefields-capture-topology"></a>
### `capture_topology`

```python
capture_topology(mol: 'Molecule', *, allow_added_hydrogens: bool=True) -> TopologyReference
```

<a id="hotpot-cheminfo-adapters-structure-forcefields-capture-atom-pair-acceptance-values"></a>
### `capture_atom_pair_acceptance_values`

```python
capture_atom_pair_acceptance_values(distances, *, capture_radii: bool, overlap_tolerance: float) -> Tuple[Tuple[Tuple[int, int], bool, float, Optional[float]], ...]
```

<a id="hotpot-cheminfo-adapters-structure-forcefields-capture-bond-acceptance-values"></a>
### `capture_bond_acceptance_values`

```python
capture_bond_acceptance_values(mol: 'Molecule', atoms: Sequence['Atom'], *, capture_radii: bool) -> Tuple[Tuple[int, Tuple[int, int], Tuple[int, int], Optional[float], Optional[bool]], ...]
```

<a id="hotpot-cheminfo-adapters-structure-forcefields-capture-coordination-acceptance-values"></a>
### `capture_coordination_acceptance_values`

```python
capture_coordination_acceptance_values(mol: 'Molecule', atoms: Sequence['Atom']) -> Tuple[Tuple[int, ...], Tuple[Tuple[int, int], ...], Tuple[Tuple[int, int], ...]]
```

<a id="hotpot-cheminfo-adapters-structure-forcefields-capture-acceptance-bond-positions"></a>
### `capture_acceptance_bond_positions`

```python
capture_acceptance_bond_positions(mol: 'Molecule') -> Tuple[Tuple[Tuple[int, int], int], ...]
```

<a id="hotpot-cheminfo-adapters-structure-forcefields-pack-complex-session-input"></a>
### `pack_complex_session_input`

```python
pack_complex_session_input(mol: 'Molecule') -> ComplexSessionInput
```

<a id="hotpot-cheminfo-adapters-structure-forcefields-collect-coordination-environments"></a>
### `collect_coordination_environments`

```python
collect_coordination_environments(mol: 'Molecule') -> Tuple[CoordinationEnvironment, ...]
```


<a id="hotpot-cheminfo-adapters-structure-geometry"></a>
## `hotpot.cheminfo.adapters.structure.geometry`

`hotpot/cheminfo/adapters/structure/geometry.py`；显式 __all__: ('PairScope', 'RingScope', 'AtomGeometry', 'AtomPairTarget', 'BondGeometry', 'RingGeometry', 'BondRingTarget', 'AtomPairDistance', 'BondRingFinding', 'RingEdgeDistance', 'BondRingScanReport', 'BondRingScreeningReport', 'BondRingScreeningPlan', 'BondRingFrameWorkspace', 'point_from_atom', 'segment_from_bond', 'cycle_from_ring', 'iter_atom_geometries', 'iter_atom_pair_targets', 'iter_ring_geometries', 'iter_bond_ring_targets', 'measure_atom_pair_distances', 'determine_bond_ring_relation', 'iter_bond_ring_findings', 'scan_bond_ring_relations', 'prepare_bond_ring_screening_plan', 'prepare_bond_ring_frame', 'screen_bond_ring_workspace', 'screen_segments_against_ring_workspace', 'screen_bonds_against_rings', 'screen_bond_ring_relations', 'determine_bond_ring_piercing_state')


<a id="hotpot-cheminfo-adapters-structure-geometry-atomgeometry"></a>
### `AtomGeometry`

定义 owner `hotpot.cheminfo.adapters.structure.geometry.AtomGeometry`；基类 `Generic[AtomSourceT]`。

```python
AtomGeometry(atom: AtomSourceT, point: Point, key: int)
```

- `atom: AtomSourceT` [只读值字段]
- `point: Point` [只读值字段]
- `key: int` [只读值字段]

<a id="hotpot-cheminfo-adapters-structure-geometry-atompairtarget"></a>
### `AtomPairTarget`

定义 owner `hotpot.cheminfo.adapters.structure.geometry.AtomPairTarget`；基类 `Generic[AtomSourceT]`。

```python
AtomPairTarget(first: AtomGeometry[AtomSourceT], second: AtomGeometry[AtomSourceT], bonded: bool)
```

- `first: AtomGeometry[AtomSourceT]` [只读值字段]
- `second: AtomGeometry[AtomSourceT]` [只读值字段]
- `bonded: bool` [只读值字段]

<a id="hotpot-cheminfo-adapters-structure-geometry-bondgeometry"></a>
### `BondGeometry`

定义 owner `hotpot.cheminfo.adapters.structure.geometry.BondGeometry`；基类 `Generic[BondSourceT]`。

```python
BondGeometry(bond: BondSourceT, segment: Segment, key: Tuple[int, int])
```

- `bond: BondSourceT` [只读值字段]
- `segment: Segment` [只读值字段]
- `key: Tuple[int, int]` [只读值字段]

<a id="hotpot-cheminfo-adapters-structure-geometry-ringgeometry"></a>
### `RingGeometry`

定义 owner `hotpot.cheminfo.adapters.structure.geometry.RingGeometry`；基类 `Generic[RingSourceT]`。

```python
RingGeometry(ring: RingSourceT, cycle: Cycle, key: Tuple[int, ...])
```

- `ring: RingSourceT` [只读值字段]
- `cycle: Cycle` [只读值字段]
- `key: Tuple[int, ...]` [只读值字段]

<a id="hotpot-cheminfo-adapters-structure-geometry-bondringtarget"></a>
### `BondRingTarget`

定义 owner `hotpot.cheminfo.adapters.structure.geometry.BondRingTarget`；基类 `Generic[RingSourceT, BondSourceT]`。

```python
BondRingTarget(ring: RingGeometry[RingSourceT], bond: BondGeometry[BondSourceT])
```

- `ring: RingGeometry[RingSourceT]` [只读值字段]
- `bond: BondGeometry[BondSourceT]` [只读值字段]

<a id="hotpot-cheminfo-adapters-structure-geometry-atompairdistance"></a>
### `AtomPairDistance`

定义 owner `hotpot.cheminfo.adapters.structure.geometry.AtomPairDistance`；基类 `Generic[AtomSourceT]`。

```python
AtomPairDistance(target: AtomPairTarget[AtomSourceT], measurement: PointPairDistance)
```

- `target: AtomPairTarget[AtomSourceT]` [只读值字段]
- `measurement: PointPairDistance` [只读值字段]

<a id="hotpot-cheminfo-adapters-structure-geometry-bondringfinding"></a>
### `BondRingFinding`

定义 owner `hotpot.cheminfo.adapters.structure.geometry.BondRingFinding`；基类 `Generic[RingSourceT, BondSourceT]`。

```python
BondRingFinding(target: BondRingTarget[RingSourceT, BondSourceT], relation: SegmentCycleRelation)
```

- `target: BondRingTarget[RingSourceT, BondSourceT]` [只读值字段]
- `relation: SegmentCycleRelation` [只读值字段]

<a id="hotpot-cheminfo-adapters-structure-geometry-ringedgedistance"></a>
### `RingEdgeDistance`

定义 owner `hotpot.cheminfo.adapters.structure.geometry.RingEdgeDistance`；基类 `Generic[BondSourceT]`。

```python
RingEdgeDistance(bond: BondSourceT, measurement: ClosestCycleEdge)
```

- `bond: BondSourceT` [只读值字段]
- `measurement: ClosestCycleEdge` [只读值字段]

<a id="hotpot-cheminfo-adapters-structure-geometry-bondringscanreport"></a>
### `BondRingScanReport`

定义 owner `hotpot.cheminfo.adapters.structure.geometry.BondRingScanReport`；基类 `Generic[RingSourceT, BondSourceT]`。

```python
BondRingScanReport(findings: Tuple[BondRingFinding[RingSourceT, BondSourceT], ...], ring_scope: RingScope, max_ring_size: int, selected_ring_count: int, excluded_ring_count: int, piercing_pair_count: int, does_not_pierce_pair_count: int, undetermined_pair_count: int)
```

- `candidate_pair_count` [只读]: `int`
- `scan_complete` [只读]: `bool`
- `piercings` [只读]: `Tuple[BondRingFinding[RingSourceT, BondSourceT], ...]`
- `undetermined` [只读]: `Tuple[BondRingFinding[RingSourceT, BondSourceT], ...]`
- `findings: Tuple[BondRingFinding[RingSourceT, BondSourceT], ...]` [只读值字段]
- `ring_scope: RingScope` [只读值字段]
- `max_ring_size: int` [只读值字段]
- `selected_ring_count: int` [只读值字段]
- `excluded_ring_count: int` [只读值字段]
- `piercing_pair_count: int` [只读值字段]
- `does_not_pierce_pair_count: int` [只读值字段]
- `undetermined_pair_count: int` [只读值字段]

<a id="hotpot-cheminfo-adapters-structure-geometry-bondringscreeningreport"></a>
### `BondRingScreeningReport`

定义 owner `hotpot.cheminfo.adapters.structure.geometry.BondRingScreeningReport`；基类 `Generic[RingSourceT, BondSourceT]`。

```python
BondRingScreeningReport(actionable_findings: Tuple[BondRingFinding[RingSourceT, BondSourceT], ...], ring_scope: RingScope, max_ring_size: int, selected_ring_count: int, excluded_ring_count: int, candidate_pair_count: int, aabb_separated_pair_count: int, exact_pair_count: int, piercing_pair_count: int, does_not_pierce_pair_count: int, undetermined_pair_count: int, scan_complete: bool)
```

- `state` [只读]: `PiercingState`
- `piercings` [只读]: `Tuple[BondRingFinding[RingSourceT, BondSourceT], ...]`
- `undetermined` [只读]: `Tuple[BondRingFinding[RingSourceT, BondSourceT], ...]`
- `actionable_findings: Tuple[BondRingFinding[RingSourceT, BondSourceT], ...]` [只读值字段]
- `ring_scope: RingScope` [只读值字段]
- `max_ring_size: int` [只读值字段]
- `selected_ring_count: int` [只读值字段]
- `excluded_ring_count: int` [只读值字段]
- `candidate_pair_count: int` [只读值字段]
- `aabb_separated_pair_count: int` [只读值字段]
- `exact_pair_count: int` [只读值字段]
- `piercing_pair_count: int` [只读值字段]
- `does_not_pierce_pair_count: int` [只读值字段]
- `undetermined_pair_count: int` [只读值字段]
- `scan_complete: bool` [只读值字段]

<a id="hotpot-cheminfo-adapters-structure-geometry-bondringscreeningplan"></a>
### `BondRingScreeningPlan`

定义 owner `hotpot.cheminfo.adapters.structure.geometry.BondRingScreeningPlan`；基类 `Generic[RingSourceT, BondSourceT]`。

```python
BondRingScreeningPlan(ring_scope: RingScope, max_ring_size: int, settings: GeometrySettings, rings: Tuple[RingSourceT, ...], bonds: Tuple[BondSourceT, ...], ring_atom_keys: Tuple[RingKey, ...], ring_edge_keys: Tuple[FrozenSet[BondKey], ...], bond_keys: Tuple[BondKey, ...], candidate_bond_keys_by_ring: Tuple[Tuple[BondKey, ...], ...], excluded_ring_count: int)
```

- `ring_scope: RingScope` [只读值字段]
- `max_ring_size: int` [只读值字段]
- `settings: GeometrySettings` [只读值字段]
- `rings: Tuple[RingSourceT, ...]` [只读值字段]
- `bonds: Tuple[BondSourceT, ...]` [只读值字段]
- `ring_atom_keys: Tuple[RingKey, ...]` [只读值字段]
- `ring_edge_keys: Tuple[FrozenSet[BondKey], ...]` [只读值字段]
- `bond_keys: Tuple[BondKey, ...]` [只读值字段]
- `candidate_bond_keys_by_ring: Tuple[Tuple[BondKey, ...], ...]` [只读值字段]
- `excluded_ring_count: int` [只读值字段]

<a id="hotpot-cheminfo-adapters-structure-geometry-bondringframeworkspace"></a>
### `BondRingFrameWorkspace`

定义 owner `hotpot.cheminfo.adapters.structure.geometry.BondRingFrameWorkspace`；基类 `Generic[RingSourceT, BondSourceT]`。

```python
BondRingFrameWorkspace(plan: BondRingScreeningPlan[RingSourceT, BondSourceT], coordinate_keys: Tuple[int, ...], coordinates: np.ndarray, rings: Tuple[_PreparedRingFrame[RingSourceT], ...], bonds: Tuple[BondGeometry[BondSourceT], ...])
```

- `plan: BondRingScreeningPlan[RingSourceT, BondSourceT]` [只读值字段]
- `coordinate_keys: Tuple[int, ...]` [只读值字段]
- `coordinates: np.ndarray` [只读值字段]
- `rings: Tuple[_PreparedRingFrame[RingSourceT], ...]` [只读值字段]
- `bonds: Tuple[BondGeometry[BondSourceT], ...]` [只读值字段]

<a id="hotpot-cheminfo-adapters-structure-geometry-point-from-atom"></a>
### `point_from_atom`

```python
point_from_atom(atom: AtomSourceT) -> Point
```

<a id="hotpot-cheminfo-adapters-structure-geometry-segment-from-bond"></a>
### `segment_from_bond`

```python
segment_from_bond(bond: BondSourceT) -> Segment
```

<a id="hotpot-cheminfo-adapters-structure-geometry-cycle-from-ring"></a>
### `cycle_from_ring`

```python
cycle_from_ring(ring: RingSourceT) -> Cycle
```

<a id="hotpot-cheminfo-adapters-structure-geometry-iter-atom-geometries"></a>
### `iter_atom_geometries`

```python
iter_atom_geometries(structure: _StructureLike[AtomSourceT, BondSourceT]) -> Iterator[AtomGeometry[AtomSourceT]]
```

<a id="hotpot-cheminfo-adapters-structure-geometry-iter-atom-pair-targets"></a>
### `iter_atom_pair_targets`

```python
iter_atom_pair_targets(structure: _StructureLike[AtomSourceT, BondSourceT], pair_scope: PairScope) -> Iterator[AtomPairTarget[AtomSourceT]]
```

<a id="hotpot-cheminfo-adapters-structure-geometry-iter-ring-geometries"></a>
### `iter_ring_geometries`

```python
iter_ring_geometries(mol: _MoleculeLike[AtomSourceT, BondSourceT, RingSourceT], *, ring_scope: RingScope, max_ring_size: int) -> Iterator[RingGeometry[RingSourceT]]
```

<a id="hotpot-cheminfo-adapters-structure-geometry-iter-bond-ring-targets"></a>
### `iter_bond_ring_targets`

```python
iter_bond_ring_targets(mol: _MoleculeLike[AtomSourceT, BondSourceT, RingSourceT], *, ring_scope: RingScope, max_ring_size: int) -> Iterator[BondRingTarget[RingSourceT, BondSourceT]]
```

<a id="hotpot-cheminfo-adapters-structure-geometry-measure-atom-pair-distances"></a>
### `measure_atom_pair_distances`

```python
measure_atom_pair_distances(structure: _StructureLike[AtomSourceT, BondSourceT], pair_scope: PairScope) -> Tuple[AtomPairDistance[AtomSourceT], ...]
```

<a id="hotpot-cheminfo-adapters-structure-geometry-determine-bond-ring-relation"></a>
### `determine_bond_ring_relation`

```python
determine_bond_ring_relation(ring: RingSourceT, bond: BondSourceT, *, settings: GeometrySettings=DEFAULT_GEOMETRY_SETTINGS) -> BondRingFinding[RingSourceT, BondSourceT]
```

<a id="hotpot-cheminfo-adapters-structure-geometry-iter-bond-ring-findings"></a>
### `iter_bond_ring_findings`

```python
iter_bond_ring_findings(mol: _MoleculeLike[AtomSourceT, BondSourceT, RingSourceT], *, ring_scope: RingScope, max_ring_size: int, settings: GeometrySettings=DEFAULT_GEOMETRY_SETTINGS) -> Iterator[BondRingFinding[RingSourceT, BondSourceT]]
```

<a id="hotpot-cheminfo-adapters-structure-geometry-scan-bond-ring-relations"></a>
### `scan_bond_ring_relations`

```python
scan_bond_ring_relations(mol: _MoleculeLike[AtomSourceT, BondSourceT, RingSourceT], *, ring_scope: RingScope, max_ring_size: int, settings: GeometrySettings=DEFAULT_GEOMETRY_SETTINGS) -> BondRingScanReport[RingSourceT, BondSourceT]
```

<a id="hotpot-cheminfo-adapters-structure-geometry-prepare-bond-ring-screening-plan"></a>
### `prepare_bond_ring_screening_plan`

```python
prepare_bond_ring_screening_plan(mol: _MoleculeLike[AtomSourceT, BondSourceT, RingSourceT], *, ring_scope: RingScope, max_ring_size: int, bonds: Optional[Iterable[BondSourceT]]=None, settings: GeometrySettings=DEFAULT_GEOMETRY_SETTINGS) -> BondRingScreeningPlan[RingSourceT, BondSourceT]
```

<a id="hotpot-cheminfo-adapters-structure-geometry-prepare-bond-ring-frame"></a>
### `prepare_bond_ring_frame`

```python
prepare_bond_ring_frame(plan: BondRingScreeningPlan[RingSourceT, BondSourceT]) -> BondRingFrameWorkspace[RingSourceT, BondSourceT]
```

<a id="hotpot-cheminfo-adapters-structure-geometry-screen-segments-against-ring-workspace"></a>
### `screen_segments_against_ring_workspace`

```python
screen_segments_against_ring_workspace(segments: Iterable[BondGeometry[BondSourceT]], workspace: BondRingFrameWorkspace[RingSourceT, BondSourceT], *, bond_keys: Optional[Iterable[BondKey]]=None, ring_keys: Optional[Iterable[RingKey]]=None, stop_after_confirmed: bool=False) -> BondRingScreeningReport[RingSourceT, BondSourceT]
```

<a id="hotpot-cheminfo-adapters-structure-geometry-screen-bond-ring-workspace"></a>
### `screen_bond_ring_workspace`

```python
screen_bond_ring_workspace(workspace: BondRingFrameWorkspace[RingSourceT, BondSourceT], *, bond_keys: Optional[Iterable[BondKey]]=None, ring_keys: Optional[Iterable[RingKey]]=None, stop_after_confirmed: bool=False) -> BondRingScreeningReport[RingSourceT, BondSourceT]
```

<a id="hotpot-cheminfo-adapters-structure-geometry-screen-bonds-against-rings"></a>
### `screen_bonds_against_rings`

```python
screen_bonds_against_rings(mol: _MoleculeLike[AtomSourceT, BondSourceT, RingSourceT], bonds: Iterable[BondSourceT], *, ring_scope: RingScope, max_ring_size: int, settings: GeometrySettings=DEFAULT_GEOMETRY_SETTINGS) -> BondRingScreeningReport[RingSourceT, BondSourceT]
```

<a id="hotpot-cheminfo-adapters-structure-geometry-screen-bond-ring-relations"></a>
### `screen_bond_ring_relations`

```python
screen_bond_ring_relations(mol: _MoleculeLike[AtomSourceT, BondSourceT, RingSourceT], *, ring_scope: RingScope, max_ring_size: int, settings: GeometrySettings=DEFAULT_GEOMETRY_SETTINGS) -> BondRingScreeningReport[RingSourceT, BondSourceT]
```

<a id="hotpot-cheminfo-adapters-structure-geometry-determine-bond-ring-piercing-state"></a>
### `determine_bond_ring_piercing_state`

```python
determine_bond_ring_piercing_state(mol: _MoleculeLike[AtomSourceT, BondSourceT, RingSourceT], *, ring_scope: RingScope, max_ring_size: int, settings: GeometrySettings=DEFAULT_GEOMETRY_SETTINGS) -> PiercingState
```

模块值/类型别名（声明存在不构成事实写许可）：

- `PairScope` = `Literal['all', 'bonded', 'nonbonded']`
- `RingScope` = `Literal['full_graph', 'ligand_skeleton']`
- `BondKey` = `Tuple[int, int]`
- `RingKey` = `Tuple[int, ...]`
- `AtomSourceT` = `TypeVar('AtomSourceT', bound=_AtomLike)`
- `BondSourceT` = `TypeVar('BondSourceT', bound=_BondLike[_AtomLike])`
- `RingSourceT` = `TypeVar('RingSourceT', bound=_RingLike[_AtomLike])`


<a id="hotpot-cheminfo-adapters-structure-inference"></a>
## `hotpot.cheminfo.adapters.structure.inference`

`hotpot/cheminfo/adapters/structure/inference.py`；显式 __all__: 无


<a id="hotpot-cheminfo-adapters-structure-inference-mca-molecule-record"></a>
### `mca_molecule_record`

```python
mca_molecule_record(atoms, bonds, formal_charge)
```

<a id="hotpot-cheminfo-adapters-structure-inference-direct-edge-to-indirect"></a>
### `direct_edge_to_indirect`

```python
direct_edge_to_indirect(attr_or_index: np.ndarray) -> np.ndarray
```

<a id="hotpot-cheminfo-adapters-structure-inference-extract-potentials-cbonds"></a>
### `extract_potentials_cbonds`

```python
extract_potentials_cbonds(mol, catoms_symbols: Optional[Iterable[str]]=None, data: dict[str, np.ndarray]=None) -> dict[str, np.ndarray]
```

<a id="hotpot-cheminfo-adapters-structure-inference-extract-atom-attrs"></a>
### `extract_atom_attrs`

```python
extract_atom_attrs(mol, data: dict[str, Any]=None, atomic_number_only: bool=False, *, projected=None, fields=()) -> dict
```

<a id="hotpot-cheminfo-adapters-structure-inference-extract-bond-attrs"></a>
### `extract_bond_attrs`

```python
extract_bond_attrs(mol, edge_attr_names: Iterable[str]=None, data: dict[str, Any]=None) -> dict
```

<a id="hotpot-cheminfo-adapters-structure-inference-extract-atom-pairs"></a>
### `extract_atom_pairs`

```python
extract_atom_pairs(atom_pairs, pair_attr_names, data: dict[str, Any]=None) -> dict
```

<a id="hotpot-cheminfo-adapters-structure-inference-extract-ring-attrs"></a>
### `extract_ring_attrs`

```python
extract_ring_attrs(rings, rings_attr_names: Iterable[str]=None, data: dict[str, Any]=None) -> dict
```


<a id="hotpot-cheminfo-adapters-structure-scientific-state"></a>
## `hotpot.cheminfo.adapters.structure.scientific_state`

`hotpot/cheminfo/adapters/structure/scientific_state.py`；显式 __all__: ()



<a id="hotpot-cheminfo-adapters-structure-stereochemistry"></a>
## `hotpot.cheminfo.adapters.structure.stereochemistry`

`hotpot/cheminfo/adapters/structure/stereochemistry.py`；显式 __all__: ()



<a id="hotpot-cheminfo-adapters-structure-substructure"></a>
## `hotpot.cheminfo.adapters.structure.substructure`

`hotpot/cheminfo/adapters/structure/substructure.py`；显式 __all__: 无



<a id="hotpot-cheminfo-adapters-structure-substructure-query"></a>
## `hotpot.cheminfo.adapters.structure.substructure_query`

`hotpot/cheminfo/adapters/structure/substructure_query.py`；显式 __all__: ()



<a id="hotpot-cheminfo-adapters-thermo"></a>
## `hotpot.cheminfo.adapters.thermo`

`hotpot/cheminfo/adapters/thermo/__init__.py`；显式 __all__: ()



<a id="hotpot-cheminfo-adapters-thermo-chemical"></a>
## `hotpot.cheminfo.adapters.thermo.chemical`

`hotpot/cheminfo/adapters/thermo/chemical.py`；显式 __all__: 无



<a id="hotpot-cheminfo-adapters-visualization"></a>
## `hotpot.cheminfo.adapters.visualization`

`hotpot/cheminfo/adapters/visualization/__init__.py`；显式 __all__: ()



<a id="hotpot-cheminfo-adapters-visualization-draw"></a>
## `hotpot.cheminfo.adapters.visualization.draw`

`hotpot/cheminfo/adapters/visualization/draw.py`；显式 __all__: ('compute_atom_bond_colors', 'create_svg_colorbar', 'merge_colorbar_to_svg')


<a id="hotpot-cheminfo-adapters-visualization-draw-choose-best-colnum"></a>
### `choose_best_colnum`

```python
choose_best_colnum(n_mols, min_col=4, max_col=7, default_col=5)
```

<a id="hotpot-cheminfo-adapters-visualization-draw-compute-atom-bond-colors"></a>
### `compute_atom_bond_colors`

```python
compute_atom_bond_colors(mol: 'Chem.Mol', atom_values: np.ndarray, cmap_name: str='coolwarm', vmin: Optional[float]=None, vmax: Optional[float]=None, threshold: float=0.75) -> tuple[dict[int, RGBColor], dict[int, RGBColor], 'mcolors.Normalize', 'cm.ScalarMappable']
```

<a id="hotpot-cheminfo-adapters-visualization-draw-save-or-convert-svg"></a>
### `save_or_convert_svg`

```python
save_or_convert_svg(svg_text: str, save_path: str=None, output_format: str=None) -> str
```

<a id="hotpot-cheminfo-adapters-visualization-draw-create-svg-colorbar"></a>
### `create_svg_colorbar`

```python
create_svg_colorbar(name: str='coolwarm', norm: 'mcolors.Normalize'=None, orientation: Literal['vertical', 'horizontal']='vertical')
```

<a id="hotpot-cheminfo-adapters-visualization-draw-merge-colorbar-to-svg"></a>
### `merge_colorbar_to_svg`

```python
merge_colorbar_to_svg(main_svg, colorbar_svg, fig_size, orientation)
```

模块值/类型别名（声明存在不构成事实写许可）：

- `RGBColor` = `tuple[float, float, float]`
- `IndexColorType` = `Union[dict[int, RGBColor], list[int], tuple[int, ...]]`
- `DrawOutputFormat` = `Literal['svg', 'png', 'pdf', 'ps', 'eps', 'jpeg', 'jpg', 'tiff', 'bmp', 'webp']`
- `atom_palette` = `{7: (0, 0, 0.5), 8: (0.5, 0, 0), 1: (0, 0.5, 0.5)}`
