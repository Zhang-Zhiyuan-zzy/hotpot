# 保留的独立 OB/RD native wrapper：全签名目录

来源：当前静态源码；显式 __all__ 与直接模块定义分别列明。类型定义 owner 与公开导出路径分别保留。
只读 property、动态 owner setter、不可变值字段与可重绑定类声明值不是同一权限概念。外部继承保持可归属的边界，未声称第三方运行时属性穷尽。
[完整树](../delivery_api_tree.md) · [动态权限](dynamic_permissions.md) · [native 声明](native_contracts.md)

## 模块导航

- [`hotpot.cheminfo.obWrappers`](#hotpot-cheminfo-obwrappers)
- [`hotpot.cheminfo.obWrappers.builder`](#hotpot-cheminfo-obwrappers-builder)
- [`hotpot.cheminfo.obWrappers.checks`](#hotpot-cheminfo-obwrappers-checks)
- [`hotpot.cheminfo.obWrappers.contracts`](#hotpot-cheminfo-obwrappers-contracts)
- [`hotpot.cheminfo.obWrappers.forcefield`](#hotpot-cheminfo-obwrappers-forcefield)
- [`hotpot.cheminfo.obWrappers.native`](#hotpot-cheminfo-obwrappers-native)
- [`hotpot.cheminfo.obWrappers.operation`](#hotpot-cheminfo-obwrappers-operation)
- [`hotpot.cheminfo.obWrappers.packing`](#hotpot-cheminfo-obwrappers-packing)
- [`hotpot.cheminfo.obWrappers.registry`](#hotpot-cheminfo-obwrappers-registry)
- [`hotpot.cheminfo.obWrappers.reports`](#hotpot-cheminfo-obwrappers-reports)
- [`hotpot.cheminfo.obWrappers.settings`](#hotpot-cheminfo-obwrappers-settings)
- [`hotpot.cheminfo.rdWrappers`](#hotpot-cheminfo-rdwrappers)
- [`hotpot.cheminfo.rdWrappers.contracts`](#hotpot-cheminfo-rdwrappers-contracts)
- [`hotpot.cheminfo.rdWrappers.native`](#hotpot-cheminfo-rdwrappers-native)
- [`hotpot.cheminfo.rdWrappers.sampler`](#hotpot-cheminfo-rdwrappers-sampler)

<a id="hotpot-cheminfo-obwrappers"></a>
## `hotpot.cheminfo.obWrappers`

`hotpot/cheminfo/obWrappers/__init__.py`；显式 __all__: ('BondKindCode', 'BuildReport', 'CoordinateChange', 'ConvergenceLevel', 'DEFAULT_CONVERGENCE_LEVEL', 'HybridizationChange', 'OptimizationFrame', 'OptimizationCheckReport', 'OptimizationFailure', 'OptimizationReport', 'RuleApplication', 'RuleDescriptor', 'RuleExecutionReport', 'RuleStage', 'SingleOptimizationReport', 'available_rules', 'build', 'check_optimization_state', 'inspect_rules', 'optimize', 'single_optimize')

- 导出 `BondKindCode` → `hotpot.cheminfo.obWrappers.contracts.BondKindCode`
- 导出 `BuildReport` → `hotpot.cheminfo.obWrappers.contracts.BuildReport`
- 导出 `CoordinateChange` → `hotpot.cheminfo.obWrappers.contracts.CoordinateChange`
- 导出 `ConvergenceLevel` → `hotpot.cheminfo.obWrappers.contracts.ConvergenceLevel`
- 导出 `DEFAULT_CONVERGENCE_LEVEL` → `hotpot.cheminfo.obWrappers.contracts.DEFAULT_CONVERGENCE_LEVEL`
- 导出 `HybridizationChange` → `hotpot.cheminfo.obWrappers.contracts.HybridizationChange`
- 导出 `OptimizationFrame` → `hotpot.cheminfo.obWrappers.contracts.OptimizationFrame`
- 导出 `OptimizationCheckReport` → `hotpot.cheminfo.obWrappers.contracts.OptimizationCheckReport`
- 导出 `OptimizationFailure` → `hotpot.cheminfo.obWrappers.contracts.OptimizationFailure`
- 导出 `OptimizationReport` → `hotpot.cheminfo.obWrappers.contracts.OptimizationReport`
- 导出 `RuleApplication` → `hotpot.cheminfo.obWrappers.contracts.RuleApplication`
- 导出 `RuleDescriptor` → `hotpot.cheminfo.obWrappers.contracts.RuleDescriptor`
- 导出 `RuleExecutionReport` → `hotpot.cheminfo.obWrappers.contracts.RuleExecutionReport`
- 导出 `RuleStage` → `hotpot.cheminfo.obWrappers.contracts.RuleStage`
- 导出 `SingleOptimizationReport` → `hotpot.cheminfo.obWrappers.contracts.SingleOptimizationReport`
- 导出 `available_rules` → `hotpot.cheminfo.obWrappers.registry.available_rules`；`available_rules(stage: Optional[RuleStage]=None) -> tuple[RuleDescriptor, ...]`
- 导出 `build` → `hotpot.cheminfo.obWrappers.builder.build`；`build(mol: 'Molecule', *, stereo_warnings: Optional[bool]=None) -> BuildReport`
- 导出 `check_optimization_state` → `hotpot.cheminfo.obWrappers.checks.check_optimization_state`；`check_optimization_state(mol: 'Molecule', forcefield: str, *, previous_coordinates: Optional[ArrayLike]=None, previous_energy_kj_mol: Optional[float]=None, singularity_threshold: float=TORSION_SINGULARITY_THRESHOLD, repair_angle_radians: float=TORSION_REPAIR_ANGLE_RADIANS) -> OptimizationCheckReport`
- 导出 `inspect_rules` → `hotpot.cheminfo.obWrappers.registry.inspect_rules`；`inspect_rules(mol: 'Molecule', stage: RuleStage, *, singularity_threshold: float=TORSION_SINGULARITY_THRESHOLD, repair_angle_radians: float=TORSION_REPAIR_ANGLE_RADIANS) -> RuleExecutionReport`
- 导出 `optimize` → `hotpot.cheminfo.obWrappers.forcefield.optimize`；`optimize(mol: 'Molecule', forcefield: str, *, algorithm: str='conjugate', epochs: int=100, steps_per_epoch: int=5, perturb_interval: Optional[int]=None, perturbation_offsets: Optional[np.ndarray]=None, retain_frames: bool=False, retain_epoch_history: bool=False, increasing_vdw: bool=False, vdw_cutoff_start: float=1.0, vdw_cutoff_end: float=10.0, energy_tolerance: float=1e-06, convergence_level: ConvergenceLevel=DEFAULT_CONVERGENCE_LEVEL, stopping_window: Optional[int]=None, maximum_energy_change_kj_mol: float=0.0001, maximum_atom_displacement_angstrom: float=0.0001, maximum_rms_gradient_kj_mol_angstrom: float=1.0, maximum_gradient_kj_mol_angstrom: float=5.0, singularity_threshold: float=TORSION_SINGULARITY_THRESHOLD, repair_angle_radians: float=TORSION_REPAIR_ANGLE_RADIANS) -> OptimizationReport`
- 导出 `single_optimize` → `hotpot.cheminfo.obWrappers.operation.single_optimize`；`single_optimize(mol: 'Molecule', forcefield: str, steps: int, *, singularity_threshold: float=TORSION_SINGULARITY_THRESHOLD, repair_angle_radians: float=TORSION_REPAIR_ANGLE_RADIANS) -> SingleOptimizationReport`


<a id="hotpot-cheminfo-obwrappers-builder"></a>
## `hotpot.cheminfo.obWrappers.builder`

`hotpot/cheminfo/obWrappers/builder.py`；显式 __all__: ('build',)


<a id="hotpot-cheminfo-obwrappers-builder-build"></a>
### `build`

```python
build(mol: 'Molecule', *, stereo_warnings: Optional[bool]=None) -> BuildReport
```


<a id="hotpot-cheminfo-obwrappers-checks"></a>
## `hotpot.cheminfo.obWrappers.checks`

`hotpot/cheminfo/obWrappers/checks.py`；显式 __all__: ('check_optimization_state',)


<a id="hotpot-cheminfo-obwrappers-checks-check-optimization-state"></a>
### `check_optimization_state`

```python
check_optimization_state(mol: 'Molecule', forcefield: str, *, previous_coordinates: Optional[ArrayLike]=None, previous_energy_kj_mol: Optional[float]=None, singularity_threshold: float=TORSION_SINGULARITY_THRESHOLD, repair_angle_radians: float=TORSION_REPAIR_ANGLE_RADIANS) -> OptimizationCheckReport
```


<a id="hotpot-cheminfo-obwrappers-contracts"></a>
## `hotpot.cheminfo.obWrappers.contracts`

`hotpot/cheminfo/obWrappers/contracts.py`；显式 __all__: ('BuildReport', 'BondKindCode', 'CoordinateChange', 'ConvergenceLevel', 'DEFAULT_CONVERGENCE_LEVEL', 'HybridizationChange', 'OptimizationFrame', 'OptimizationCheckReport', 'OptimizationFailure', 'OptimizationReport', 'RuleApplication', 'RuleDescriptor', 'RuleExecutionReport', 'RuleStage', 'SingleOptimizationReport')


<a id="hotpot-cheminfo-obwrappers-contracts-bondkindcode"></a>
### `BondKindCode`

定义 owner `hotpot.cheminfo.obWrappers.contracts.BondKindCode`；基类 `IntEnum`。

- `SINGLE` = `1` [枚举常量]
- `DOUBLE` = `2` [枚举常量]
- `TRIPLE` = `3` [枚举常量]
- `AROMATIC` = `4` [枚举常量]
- `ZERO` = `5` [枚举常量]
- `DATIVE` = `6` [枚举常量]
- `UNKNOWN` = `7` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-cheminfo-obwrappers-contracts-rulestage"></a>
### `RuleStage`

定义 owner `hotpot.cheminfo.obWrappers.contracts.RuleStage`；基类 `Enum`。

- `PRE_BUILD` = `'pre_build'` [枚举常量]
- `PRE_FORCEFIELD_SETUP` = `'pre_forcefield_setup'` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-cheminfo-obwrappers-contracts-optimizationfailure"></a>
### `OptimizationFailure`

定义 owner `hotpot.cheminfo.obWrappers.contracts.OptimizationFailure`；基类 `IntEnum`。

- `NONE` = `0` [枚举常量]
- `NONFINITE_COORDINATES` = `1` [枚举常量]
- `NONFINITE_ENERGY` = `2` [枚举常量]
- `NONFINITE_GRADIENTS` = `3` [枚举常量]
- `EXPLOSION_DETECTED` = `4` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-cheminfo-obwrappers-contracts-convergencelevel"></a>
### `ConvergenceLevel`

定义 owner `hotpot.cheminfo.obWrappers.contracts.ConvergenceLevel`；基类 `IntEnum`。

- `OPENBABEL` = `0` [枚举常量]
- `FAST` = `1` [枚举常量]
- `BALANCED` = `2` [枚举常量]
- `STRICT` = `3` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-cheminfo-obwrappers-contracts-ruledescriptor"></a>
### `RuleDescriptor`

定义 owner `hotpot.cheminfo.obWrappers.contracts.RuleDescriptor`；基类 无。

```python
RuleDescriptor(rule_id: str, version: str, stage: RuleStage, priority: int)
```

- `rule_id: str` [只读值字段]
- `version: str` [只读值字段]
- `stage: RuleStage` [只读值字段]
- `priority: int` [只读值字段]

<a id="hotpot-cheminfo-obwrappers-contracts-hybridizationchange"></a>
### `HybridizationChange`

定义 owner `hotpot.cheminfo.obWrappers.contracts.HybridizationChange`；基类 无。

```python
HybridizationChange(atom_index: int, before: int, after: int)
```

- `atom_index: int` [只读值字段]
- `before: int` [只读值字段]
- `after: int` [只读值字段]

<a id="hotpot-cheminfo-obwrappers-contracts-coordinatechange"></a>
### `CoordinateChange`

定义 owner `hotpot.cheminfo.obWrappers.contracts.CoordinateChange`；基类 无。

```python
CoordinateChange(atom_index: int, before: Tuple[float, float, float], after: Tuple[float, float, float])
```

- `atom_index: int` [只读值字段]
- `before: Tuple[float, float, float]` [只读值字段]
- `after: Tuple[float, float, float]` [只读值字段]

<a id="hotpot-cheminfo-obwrappers-contracts-ruleapplication"></a>
### `RuleApplication`

定义 owner `hotpot.cheminfo.obWrappers.contracts.RuleApplication`；基类 无。

```python
RuleApplication(descriptor: RuleDescriptor, atom_indices: Tuple[int, ...], metric_before: Optional[float], hybridization_changes: Tuple[HybridizationChange, ...], coordinate_changes: Tuple[CoordinateChange, ...])
```

- `descriptor: RuleDescriptor` [只读值字段]
- `atom_indices: Tuple[int, ...]` [只读值字段]
- `metric_before: Optional[float]` [只读值字段]
- `hybridization_changes: Tuple[HybridizationChange, ...]` [只读值字段]
- `coordinate_changes: Tuple[CoordinateChange, ...]` [只读值字段]

<a id="hotpot-cheminfo-obwrappers-contracts-ruleexecutionreport"></a>
### `RuleExecutionReport`

定义 owner `hotpot.cheminfo.obWrappers.contracts.RuleExecutionReport`；基类 无。

```python
RuleExecutionReport(stage: RuleStage, applications: Tuple[RuleApplication, ...] = ())
```

- `applied` [只读]: `bool`
- `stage: RuleStage` [只读值字段]
- `applications: Tuple[RuleApplication, ...]` = `()` [只读值字段]

<a id="hotpot-cheminfo-obwrappers-contracts-buildreport"></a>
### `BuildReport`

定义 owner `hotpot.cheminfo.obWrappers.contracts.BuildReport`；基类 无。

```python
BuildReport(succeeded: bool, rules: RuleExecutionReport)
```

- `succeeded: bool` [只读值字段]
- `rules: RuleExecutionReport` [只读值字段]

<a id="hotpot-cheminfo-obwrappers-contracts-singleoptimizationreport"></a>
### `SingleOptimizationReport`

定义 owner `hotpot.cheminfo.obWrappers.contracts.SingleOptimizationReport`；基类 无。

```python
SingleOptimizationReport(coordinates: NDArray[np.float64], energy: float, energy_unit: str, backend_energy_unit: str, exploded: bool, rules: RuleExecutionReport)
```

- `coordinates: NDArray[np.float64]` [只读值字段]
- `energy: float` [只读值字段]
- `energy_unit: str` [只读值字段]
- `backend_energy_unit: str` [只读值字段]
- `exploded: bool` [只读值字段]
- `rules: RuleExecutionReport` [只读值字段]

<a id="hotpot-cheminfo-obwrappers-contracts-optimizationcheckreport"></a>
### `OptimizationCheckReport`

定义 owner `hotpot.cheminfo.obWrappers.contracts.OptimizationCheckReport`；基类 无。

```python
OptimizationCheckReport(evaluated_coordinates: NDArray[np.float64], energy: float, energy_unit: str, backend_energy_unit: str, rms_gradient: float, max_gradient: float, gradient_unit: str, energy_change: Optional[float], max_displacement: Optional[float], finite_coordinates: bool, exploded: bool, failure: OptimizationFailure, rules: RuleExecutionReport)
```

- `usable` [只读]: `bool`
- `evaluated_coordinates: NDArray[np.float64]` [只读值字段]
- `energy: float` [只读值字段]
- `energy_unit: str` [只读值字段]
- `backend_energy_unit: str` [只读值字段]
- `rms_gradient: float` [只读值字段]
- `max_gradient: float` [只读值字段]
- `gradient_unit: str` [只读值字段]
- `energy_change: Optional[float]` [只读值字段]
- `max_displacement: Optional[float]` [只读值字段]
- `finite_coordinates: bool` [只读值字段]
- `exploded: bool` [只读值字段]
- `failure: OptimizationFailure` [只读值字段]
- `rules: RuleExecutionReport` [只读值字段]

<a id="hotpot-cheminfo-obwrappers-contracts-optimizationframe"></a>
### `OptimizationFrame`

定义 owner `hotpot.cheminfo.obWrappers.contracts.OptimizationFrame`；基类 无。

```python
OptimizationFrame(coordinates: NDArray[np.float64], energy: float, rms_gradient: float, max_gradient: float, exploded: bool, converged: bool, epoch_index: int, segment_epochs_completed: int, segment_index: int, energy_change: Optional[float], max_displacement: Optional[float])
```

- `coordinates: NDArray[np.float64]` [只读值字段]
- `energy: float` [只读值字段]
- `rms_gradient: float` [只读值字段]
- `max_gradient: float` [只读值字段]
- `exploded: bool` [只读值字段]
- `converged: bool` [只读值字段]
- `epoch_index: int` [只读值字段]
- `segment_epochs_completed: int` [只读值字段]
- `segment_index: int` [只读值字段]
- `energy_change: Optional[float]` [只读值字段]
- `max_displacement: Optional[float]` [只读值字段]

<a id="hotpot-cheminfo-obwrappers-contracts-optimizationreport"></a>
### `OptimizationReport`

定义 owner `hotpot.cheminfo.obWrappers.contracts.OptimizationReport`；基类 无。

```python
OptimizationReport(coordinates: NDArray[np.float64], terminal_coordinates: NDArray[np.float64], frames: Tuple[OptimizationFrame, ...], selected_frame_index: int, best_epoch: int, final_energy: float, best_energy: float, rms_gradient: float, max_gradient: float, exploded: bool, converged: bool, epochs_completed: int, steps_submitted: int, initialization_steps: int, selected_segment_epochs_completed: int, energy_unit: str, backend_energy_unit: str, termination_reason: str, terminal_converged: bool, energy_changes: Tuple[float, ...], max_displacements: Tuple[float, ...], epoch_energies: Tuple[float, ...], rules: RuleExecutionReport, convergence_level: ConvergenceLevel = DEFAULT_CONVERGENCE_LEVEL)
```

- `coordinates: NDArray[np.float64]` [只读值字段]
- `terminal_coordinates: NDArray[np.float64]` [只读值字段]
- `frames: Tuple[OptimizationFrame, ...]` [只读值字段]
- `selected_frame_index: int` [只读值字段]
- `best_epoch: int` [只读值字段]
- `final_energy: float` [只读值字段]
- `best_energy: float` [只读值字段]
- `rms_gradient: float` [只读值字段]
- `max_gradient: float` [只读值字段]
- `exploded: bool` [只读值字段]
- `converged: bool` [只读值字段]
- `epochs_completed: int` [只读值字段]
- `steps_submitted: int` [只读值字段]
- `initialization_steps: int` [只读值字段]
- `selected_segment_epochs_completed: int` [只读值字段]
- `energy_unit: str` [只读值字段]
- `backend_energy_unit: str` [只读值字段]
- `termination_reason: str` [只读值字段]
- `terminal_converged: bool` [只读值字段]
- `energy_changes: Tuple[float, ...]` [只读值字段]
- `max_displacements: Tuple[float, ...]` [只读值字段]
- `epoch_energies: Tuple[float, ...]` [只读值字段]
- `rules: RuleExecutionReport` [只读值字段]
- `convergence_level: ConvergenceLevel` = `DEFAULT_CONVERGENCE_LEVEL` [只读值字段]

模块值/类型别名（声明存在不构成事实写许可）：

- `DEFAULT_CONVERGENCE_LEVEL` = `ConvergenceLevel.FAST`


<a id="hotpot-cheminfo-obwrappers-forcefield"></a>
## `hotpot.cheminfo.obWrappers.forcefield`

`hotpot/cheminfo/obWrappers/forcefield.py`；显式 __all__: ('optimize', 'single_optimize')

- 导出 `single_optimize` → `hotpot.cheminfo.obWrappers.operation.single_optimize`；`single_optimize(mol: 'Molecule', forcefield: str, steps: int, *, singularity_threshold: float=TORSION_SINGULARITY_THRESHOLD, repair_angle_radians: float=TORSION_REPAIR_ANGLE_RADIANS) -> SingleOptimizationReport`

<a id="hotpot-cheminfo-obwrappers-forcefield-optimize"></a>
### `optimize`

```python
optimize(mol: 'Molecule', forcefield: str, *, algorithm: str='conjugate', epochs: int=100, steps_per_epoch: int=5, perturb_interval: Optional[int]=None, perturbation_offsets: Optional[np.ndarray]=None, retain_frames: bool=False, retain_epoch_history: bool=False, increasing_vdw: bool=False, vdw_cutoff_start: float=1.0, vdw_cutoff_end: float=10.0, energy_tolerance: float=1e-06, convergence_level: ConvergenceLevel=DEFAULT_CONVERGENCE_LEVEL, stopping_window: Optional[int]=None, maximum_energy_change_kj_mol: float=0.0001, maximum_atom_displacement_angstrom: float=0.0001, maximum_rms_gradient_kj_mol_angstrom: float=1.0, maximum_gradient_kj_mol_angstrom: float=5.0, singularity_threshold: float=TORSION_SINGULARITY_THRESHOLD, repair_angle_radians: float=TORSION_REPAIR_ANGLE_RADIANS) -> OptimizationReport
```


<a id="hotpot-cheminfo-obwrappers-native"></a>
## `hotpot.cheminfo.obWrappers.native`

`hotpot/cheminfo/obWrappers/native.py`；显式 __all__: ()



<a id="hotpot-cheminfo-obwrappers-operation"></a>
## `hotpot.cheminfo.obWrappers.operation`

`hotpot/cheminfo/obWrappers/operation.py`；显式 __all__: ('single_optimize',)


<a id="hotpot-cheminfo-obwrappers-operation-single-optimize"></a>
### `single_optimize`

```python
single_optimize(mol: 'Molecule', forcefield: str, steps: int, *, singularity_threshold: float=TORSION_SINGULARITY_THRESHOLD, repair_angle_radians: float=TORSION_REPAIR_ANGLE_RADIANS) -> SingleOptimizationReport
```


<a id="hotpot-cheminfo-obwrappers-packing"></a>
## `hotpot.cheminfo.obWrappers.packing`

`hotpot/cheminfo/obWrappers/packing.py`；显式 __all__: ()



<a id="hotpot-cheminfo-obwrappers-registry"></a>
## `hotpot.cheminfo.obWrappers.registry`

`hotpot/cheminfo/obWrappers/registry.py`；显式 __all__: ('available_rules', 'inspect_rules')


<a id="hotpot-cheminfo-obwrappers-registry-available-rules"></a>
### `available_rules`

```python
available_rules(stage: Optional[RuleStage]=None) -> tuple[RuleDescriptor, ...]
```

<a id="hotpot-cheminfo-obwrappers-registry-inspect-rules"></a>
### `inspect_rules`

```python
inspect_rules(mol: 'Molecule', stage: RuleStage, *, singularity_threshold: float=TORSION_SINGULARITY_THRESHOLD, repair_angle_radians: float=TORSION_REPAIR_ANGLE_RADIANS) -> RuleExecutionReport
```


<a id="hotpot-cheminfo-obwrappers-reports"></a>
## `hotpot.cheminfo.obWrappers.reports`

`hotpot/cheminfo/obWrappers/reports.py`；显式 __all__: ()



<a id="hotpot-cheminfo-obwrappers-settings"></a>
## `hotpot.cheminfo.obWrappers.settings`

`hotpot/cheminfo/obWrappers/settings.py`；显式 __all__: ('TORSION_REPAIR_ANGLE_RADIANS', 'TORSION_SINGULARITY_THRESHOLD')


模块值/类型别名（声明存在不构成事实写许可）：

- `TORSION_SINGULARITY_THRESHOLD` = `1e-06`
- `TORSION_REPAIR_ANGLE_RADIANS` = `0.001`


<a id="hotpot-cheminfo-rdwrappers"></a>
## `hotpot.cheminfo.rdWrappers`

`hotpot/cheminfo/rdWrappers/__init__.py`；显式 __all__: ('ConformerBatch', 'ConformerRecord', 'CoordinationEmbeddingConstraints', 'ETKDGOptions', 'ETKDGSampler', 'GenerationStatus', 'RuntimeInfo', 'generate_conformers', 'runtime_info')

- 导出 `ConformerBatch` → `hotpot.cheminfo.rdWrappers.contracts.ConformerBatch`
- 导出 `ConformerRecord` → `hotpot.cheminfo.rdWrappers.contracts.ConformerRecord`
- 导出 `CoordinationEmbeddingConstraints` → `hotpot.cheminfo.rdWrappers.contracts.CoordinationEmbeddingConstraints`
- 导出 `ETKDGOptions` → `hotpot.cheminfo.rdWrappers.contracts.ETKDGOptions`
- 导出 `ETKDGSampler` → `hotpot.cheminfo.rdWrappers.sampler.ETKDGSampler`
- 导出 `GenerationStatus` → `hotpot.cheminfo.rdWrappers.contracts.GenerationStatus`
- 导出 `RuntimeInfo` → `hotpot.cheminfo.rdWrappers.contracts.RuntimeInfo`
- 导出 `generate_conformers` → `hotpot.cheminfo.rdWrappers.sampler.generate_conformers`；`generate_conformers(mol: Molecule, *, options: ETKDGOptions | None=None, constraints: CoordinationEmbeddingConstraints | None=None) -> ConformerBatch`
- 导出 `runtime_info` → `hotpot.cheminfo.rdWrappers.sampler.runtime_info`；`runtime_info() -> RuntimeInfo`


<a id="hotpot-cheminfo-rdwrappers-contracts"></a>
## `hotpot.cheminfo.rdWrappers.contracts`

`hotpot/cheminfo/rdWrappers/contracts.py`；显式 __all__: ('ConformerBatch', 'ConformerRecord', 'CoordinationEmbeddingConstraints', 'ETKDGOptions', 'GenerationStatus', 'RuntimeInfo')


<a id="hotpot-cheminfo-rdwrappers-contracts-generationstatus"></a>
### `GenerationStatus`

定义 owner `hotpot.cheminfo.rdWrappers.contracts.GenerationStatus`；基类 `IntEnum`。

- `SUCCESS` = `0` [枚举常量]
- `PARTIAL` = `1` [枚举常量]
- `UNSUPPORTED` = `2` [枚举常量]
- `FAILED` = `3` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-cheminfo-rdwrappers-contracts-etkdgoptions"></a>
### `ETKDGOptions`

定义 owner `hotpot.cheminfo.rdWrappers.contracts.ETKDGOptions`；基类 无。

```python
ETKDGOptions(conformer_count: int = 1, seed: int | None = None, prune_rms_threshold: float | None = None, use_random_coordinates: bool = False)
```

- `conformer_count: int` = `1` [只读值字段]
- `seed: int | None` = `None` [只读值字段]
- `prune_rms_threshold: float | None` = `None` [只读值字段]
- `use_random_coordinates: bool` = `False` [只读值字段]

<a id="hotpot-cheminfo-rdwrappers-contracts-coordinationembeddingconstraints"></a>
### `CoordinationEmbeddingConstraints`

定义 owner `hotpot.cheminfo.rdWrappers.contracts.CoordinationEmbeddingConstraints`；基类 无。

```python
CoordinationEmbeddingConstraints(coordination_pairs: NDArray[np.int32], metal_donor_bounds: NDArray[np.float64], donor_pair_indices: NDArray[np.int32] | None = None, donor_pair_bounds: NDArray[np.float64] | None = None)
```

- `coordination_pairs: NDArray[np.int32]` [只读值字段]
- `metal_donor_bounds: NDArray[np.float64]` [只读值字段]
- `donor_pair_indices: NDArray[np.int32] | None` = `None` [只读值字段]
- `donor_pair_bounds: NDArray[np.float64] | None` = `None` [只读值字段]

<a id="hotpot-cheminfo-rdwrappers-contracts-conformerrecord"></a>
### `ConformerRecord`

定义 owner `hotpot.cheminfo.rdWrappers.contracts.ConformerRecord`；基类 无。

```python
ConformerRecord(ordinal: int, engine_conformer_id: int)
```

- `ordinal: int` [只读值字段]
- `engine_conformer_id: int` [只读值字段]

<a id="hotpot-cheminfo-rdwrappers-contracts-conformerbatch"></a>
### `ConformerBatch`

定义 owner `hotpot.cheminfo.rdWrappers.contracts.ConformerBatch`；基类 无。

```python
ConformerBatch(status: GenerationStatus, coordinates: NDArray[np.float64], records: tuple[ConformerRecord, ...], engine: str, engine_version: str, diagnostic: str | None)
```

- `status: GenerationStatus` [只读值字段]
- `coordinates: NDArray[np.float64]` [只读值字段]
- `records: tuple[ConformerRecord, ...]` [只读值字段]
- `engine: str` [只读值字段]
- `engine_version: str` [只读值字段]
- `diagnostic: str | None` [只读值字段]

<a id="hotpot-cheminfo-rdwrappers-contracts-runtimeinfo"></a>
### `RuntimeInfo`

定义 owner `hotpot.cheminfo.rdWrappers.contracts.RuntimeInfo`；基类 无。

```python
RuntimeInfo(hotpot_version: str, rdkit_version: str, native: bool)
```

- `hotpot_version: str` [只读值字段]
- `rdkit_version: str` [只读值字段]
- `native: bool` [只读值字段]


<a id="hotpot-cheminfo-rdwrappers-native"></a>
## `hotpot.cheminfo.rdWrappers.native`

`hotpot/cheminfo/rdWrappers/native.py`；显式 __all__: 无


<a id="hotpot-cheminfo-rdwrappers-native-generate"></a>
### `generate`

```python
generate(mol: Molecule, options: ETKDGOptions, constraints: CoordinationEmbeddingConstraints | None) -> ConformerBatch
```

<a id="hotpot-cheminfo-rdwrappers-native-runtime-info"></a>
### `runtime_info`

```python
runtime_info() -> RuntimeInfo
```


<a id="hotpot-cheminfo-rdwrappers-sampler"></a>
## `hotpot.cheminfo.rdWrappers.sampler`

`hotpot/cheminfo/rdWrappers/sampler.py`；显式 __all__: 无


<a id="hotpot-cheminfo-rdwrappers-sampler-etkdgsampler"></a>
### `ETKDGSampler`

定义 owner `hotpot.cheminfo.rdWrappers.sampler.ETKDGSampler`；基类 无。

- `__init__(self, options: ETKDGOptions | None=None)`
- `generate(self, mol: Molecule, *, constraints: CoordinationEmbeddingConstraints | None=None) -> ConformerBatch`
- `options` = `options if options is not None else ETKDGOptions()` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）

<a id="hotpot-cheminfo-rdwrappers-sampler-generate-conformers"></a>
### `generate_conformers`

```python
generate_conformers(mol: Molecule, *, options: ETKDGOptions | None=None, constraints: CoordinationEmbeddingConstraints | None=None) -> ConformerBatch
```

<a id="hotpot-cheminfo-rdwrappers-sampler-runtime-info"></a>
### `runtime_info`

```python
runtime_info() -> RuntimeInfo
```
