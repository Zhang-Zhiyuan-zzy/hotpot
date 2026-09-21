# 附件 A001：`forcefields.py` 与 `geometry.py` 类型和命名审查

> 实施状态（2026-09-21）：已完成。当前代码位于 `forcefields/` 与 `geometry/` package；
> 两者已无 `Any`，callable annotation 已补齐。异构但稳定的 force-field 诊断值使用
> `ForceFieldDiagnosticValue`，新增 bond 事务数据使用 `_BondAttributePayload`；仅原样
> 保存/恢复的 conformer state 与 bond source metadata 继续使用 `object`。

## 1. 审查范围

父记录：[`plan/forcefields_incremental_review.md`](../forcefields_incremental_review.md)

依据 `skills/development.md` 第 4.3 节的新规则，本附件只审查：

1. 能否移除 `Any`，以及应该替换成什么类型；
2. 化学对象的类、实例、形式参数和局部变量是否保留了化学语义；
3. 缺失类型标注是否会让代码通过“省略 annotation”绕开新规则。

本附件不批准、不实施任何函数体、控制流、默认值或化学业务逻辑变更。

审查对象：

- `hotpot/cheminfo/forcefields.py`
- `hotpot/cheminfo/geometry.py`

## 2. 总结

| 文件 | 含 `Any` 的参数位置 | 含 `Any` 的返回位置 | 含 `Any` 的字段/局部标注 | 合计 |
|---|---:|---:|---:|---:|
| `forcefields.py` | 56 | 4 | 23 | 83 |
| `geometry.py` | 41 | 5 | 4 | 50 |
| **合计** | **97** | **9** | **27** | **133** |

结论：当前未发现一个能够证明必须保留 `Any` 的位置。它们可以分为五类：

1. 已知 Hotpot 化学对象，应直接使用 `Molecule`、`Atom`、`Bond`、`Ring` 等；
2. 已知 Open Babel 或 multiprocessing 对象，应使用后端提供的具体类型；
3. 已知 geometry report/check，应使用现有 dataclass；
4. 异构但结构稳定的数据，应新增 `TypedDict`、dataclass、类型别名或 `Protocol`；
5. 只被原样保存和恢复的 opaque 状态，应使用 `object`，而不是允许任意操作的 `Any`。

此外，`forcefields.py` 有 13 个 callable、`geometry.py` 有 42 个 callable 至少缺少
一个参数或返回标注。它们不一定直接违反“禁止 `Any`”，但后续不能通过删除 annotation
来规避精确类型要求。

## 3. 共同的类型基础设施

### T-001：使用延迟类型导入解决 Core 循环依赖

`core.py` 在 `Molecule`、`Atom`、`Bond` 和 `Ring` 定义前导入 `forcefields` 与
`geometry`，因此两个模块不能在运行时反向导入这些类。后续整改应使用：

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Atom, AtomPair, Bond, Molecule, Ring
```

两个文件已经启用 `from __future__ import annotations`，类型不会在函数定义时求值。
不得以循环导入为理由继续保留 `Any`，也不应为了 annotation 改变运行时 import 顺序。

### T-002：建立少量有明确所有权的共享类型

不应在 `forcefields.py` 和 `geometry.py` 各自复制同一组松散 union。建议由
`geometry.py` 定义 geometry gate 自身拥有的类型：

- `GeometryDiagnosticValue`：`None/bool/int/float/str`、这些值的递归 tuple，以及
  topology signature；
- `GeometryQualityThresholdOverrides`：与 `GeometryQualityThresholds` 字段对应的
  `TypedDict(total=False)`；
- `ForceFieldQualityReportProtocol` 和候选阶段的 typed mapping；
- `CoordinationGeometryMetrics`：当前金属索引、配位数、donor 索引、距离和角度结构。

`forcefields.py` 直接引用 `geo.GeometryCheck`、`geo.GeometryQualityReport`、
`geo.TopologyReference` 和上述协议，不得反向让 geometry import forcefields。

## 4. `forcefields.py` 必改清单

### FF-T001：所有分子输入从 `Any` 收紧为 `Molecule`

以下接口的 `mol: Any` 都应改成 `mol: Molecule`：

- 后端和准备 helper：`_require_explicit_complex()`、`_make_constraints()`、
  `_single_ob_optimization()`、`_ob_build()`、`_ob_optimize()`、
  `_recalculate_neutral_donor_valence()`、`_capture_workflow_topology()`；
- 工作副本/事务 helper：`_hydrogenated_working_copy()`、`_structure_worker_proxy()`、
  `_copy_molecule_metadata()`、`_prepare_working_copy_commit()`、
  `_snapshot_molecule_for_commit()`、
  `_restore_failed_commit()`、`_commit_working_copy()`；
- optimizer：`_OpenBabelOptimizer._setup()`、`_observe_frame()`、`optimize()`；
- ligand/worker/workflow：`_build_ligand_proxies()`、`_run_complexes_build()`、
  `_run_seeded_ob_build()`、`_seeded_ob_build_coordinates()`、
  `_build_complex_working()`、`_run_optimizer_on_working()`、
  `_complexes_build_impl()`；
- 全部公开函数：`perturb()`、`collect_coordination_environments()`、
  `prepare_coordination_geometry()`、`build3d()`、`optimize()`、
  `build_complex3d()`、`optimize_complex()`、`complexes_build()`、
  `build_and_optimize()`、`auto_optimize()`。

相关返回类型同时收紧：

| 位置 | 当前 | 目标 |
|---|---|---|
| `_hydrogenated_working_copy()` | `Any` | `Molecule` |
| `_structure_worker_proxy()` | `Any` | `Molecule` |
| `_build_complex_working()` | `Tuple[Any, ComplexBuildDiagnostics]` | `Tuple[Molecule, ComplexBuildDiagnostics]` |
| `_capture_workflow_topology()` | `Any` | `geo.TopologyReference` |

### FF-T002：原子、键和事务快照使用真实 Core 类型

| 位置 | 当前问题 | 目标类型 |
|---|---|---|
| `_atom_commit_signature(atom)` | `atom: Any` | `atom: Atom` |
| `_bond_commit_signature(bond, ...)` | `bond: Any` | `bond: Bond` |
| `_MoleculeCommitSnapshot.atoms` | `Tuple[Any, ...]` | `Tuple[Atom, ...]` |
| `_MoleculeCommitSnapshot.bonds` | `Tuple[Any, ...]` | `Tuple[Bond, ...]` |
| `atom_state` | 四个 `Any` 混合对象 | `Tuple[Atom, np.ndarray, List[Atom], List[Bond]]` 的 tuple |
| `graph` | `Any` | `nx.Graph` |
| `row_to_index` | `Any` | `Optional[Mapping[int, int]]` |
| `angles` / `torsions` / `rings` | `Any` | `List[Angle]` / `List[Torsion]` / `List[Ring]` |
| `ligand_rings` | `Any` | `Optional[List[Ring]]` |
| `obmol` | `Any` | `Optional[ob.OBMol]` |
| `atom_pair_items` | `Tuple[Tuple[Any, Any], ...]` | 以 `AtomPair` 和其 key 的真实类型表示 |

`ligand_rings_signature` 应按 Core 当前生成的两层 tuple 定义独立别名，不能继续使用
`Any`。这些字段属于私有事务快照，精确标注不会扩大公共 API。

### FF-T003：geometry report 类型全部显式化

| 位置 | 目标类型 |
|---|---|
| `ForceFieldRunReport.quality_report` | `Optional[geo.GeometryQualityReport]` |
| `ForceFieldRunReport.epoch_quality_reports` | `Tuple[geo.GeometryQualityReport, ...]` |
| `Build3DReport.quality_report` | `geo.GeometryQualityReport` |
| `CandidateRejection.quality_failures` | `Tuple[geo.GeometryCheck, ...]` |
| `ForceFieldWorkflowReport.quality_report` | `geo.GeometryQualityReport` |
| `_ObservedFrame.quality_report` | `geo.GeometryQualityReport` |
| `GeometryQualityError.report` | `Optional[geo.GeometryQualityReport]` |
| `_format_geometry_checks(checks)` | `Tuple[geo.GeometryCheck, ...]` 或更一般的 `Sequence` |
| `_format_geometry_rejection(report)` | `geo.GeometryQualityReport` |
| optimizer 的 `topology_reference` 参数 | `geo.TopologyReference` |
| optimizer 和公开工作流的 `quality_thresholds` | `geo.GeometryQualityThresholdOverrides` 或完整 threshold input alias |

`ForceFieldWorkflowReport.build: Any` 不应简单改为 `object`。推荐将基类定义为以
build report 为类型参数的 generic，或显式使用
`Union[Build3DReport, ComplexBuildDiagnostics]`；两个子类继续收紧为各自具体类型。
当前 forcefields 将 `quality_thresholds` 写成 `Mapping[str, float]`，但 geometry 的
两个 bond-ratio 阈值实际是 float tuple；这不是 `Any` 问题，却是已经存在的错误窄标注，
应与新的 threshold contract 一并修正。

### FF-T004：Open Babel、IPC、异常工厂和 callable 类型

| 位置 | 当前 | 建议 |
|---|---|---|
| `_OpenBabelOptimizer._setup(..., obmol)` | `Any` | `ob.OBMol` |
| `_gradients(obmol)` / `_observe_frame(..., obmol)` | `Any` | `ob.OBMol` |
| `_iter_obmol_atoms()` 返回 | 缺失 | `Iterator[ob.OBAtom]` |
| worker 的 `connection` | `Any` | `multiprocessing.connection.Connection` |
| receive/send connection | `Any` | `Connection` |
| `worker_error_type` | `Any` | 对两种 worker exception class 的 `Type[...]`/factory protocol |
| `timeout_error_type` | `Any` | 对两种 timeout exception class 的 `Type[...]`/factory protocol |
| 两个锁装饰器的 `function` | 无标注 | `ParamSpec` + `TypeVar` + `Callable` |
| `_optimizer_methods()` | 无返回标注 | 两个 backend callable 的 tuple |
| `_initialize_with_budget(initialize)` | 无标注 | `Callable[[int, float], bool]` |

异常构造器和 `_OpenBabelOptimizer.__init__()` 的缺失返回标注应补为 `None`。

### FF-T005：异构 payload 不得继续使用 `Any`

- `_WorkingCopyCommit.added_bonds`：为 `Bond.attr_dict` 建立 `BondAttributePayload`
  `TypedDict`，或者先在 Core 提供稳定的 bond serialization dataclass。
- `_WorkingCopyCommit.conformer_state` 和 `_MoleculeCommitSnapshot.conformer_state`：该
  代码只保存并恢复值，不对值执行操作，最低限度应使用 `Mapping[str, object]`；更好的
  方案是在 `Conformers` 中提供有类型的 snapshot dataclass。
- `BuildWorkerResult.conformers`：建立明确的 conformer payload；如果最终确认该字段没有
  生产消费者，应在另一项逻辑审议中删除，而不是用 `Any` 掩盖 dormant 字段。
- `_translate_legacy_complex_build_options()` 与 `complexes_build(**options)`：定义
  `ComplexBuildOptions` `TypedDict` 或完整显式 canonical signature。兼容 wrapper 可使用
  受限的 option-value union；不能继续用 `Mapping[str, Any]` / `**options: Any`。

### FF-N001：分子对象名称必须携带生命周期角色

以下是机械重命名，不应改动行为：

| 当前名称 | 实际对象 | 建议名称 |
|---|---|---|
| `_copy_molecule_metadata(source, target)` | 两个 `Molecule` | `source_mol`, `target_mol` |
| `_hydrogenated_working_copy()` 内的 `working` | 补氢工作副本 | `working_mol` |
| `_structure_worker_proxy()` 内的 `proxy` | worker 用分子副本 | `worker_mol` 或 `proxy_mol` |
| commit helper 的 `mol, working` | 原始分子和工作副本 | `original_mol, working_mol` |
| atom identity loop 的 `atom, source` | 原始/工作副本原子 | `original_atom, working_atom` |
| `_build_ligand_proxies()` 内的 `clone` | 代理构筑分子副本 | `clone_mol` |
| 同函数内的 `component` | 一个分子组件 | `component_mol`；若已确认全为配体则用 `ligand_mol` |
| `worker_proxy` | 传入子进程的 `Molecule` | `worker_mol` 或 `worker_proxy_mol` |
| `_build_complex_working()` 及公开流程内的 `working` | 事务工作副本 | `working_mol` |
| `_run_optimizer_on_working(working)` | 被原地优化的工作分子 | `working_mol` |

`mol`、`atom`、`bond`、`obmol` 本身符合简洁化学命名规则，不应为了“更长”而无条件
改名。只有同一作用域同时出现原始分子、副本或 worker 分子时才添加角色前缀。

## 5. `geometry.py` 必改清单

### GEO-T001：化学对象直接使用 Core 类型或真实的最小协议

| 当前 `Any` 用途 | 应替换为 |
|---|---|
| topology、ring-scope、quality gate 中的 `mol` 参数 | `Molecule` |
| `_AtomPairTable.atoms`、`_atom_coordinates(atoms)`、`_atom_index(atom)` | `Tuple[Atom, ...]`、`Sequence[Atom]`、`Atom` |
| `_rings_for_scope()` 返回及所有 `ring` 参数 | `Sequence[Ring]`、`Ring` |
| `_bond_key()`、`_bond_kind()`、`_topology_bond_signature()` 的 `bond` | `Bond` |
| `bond_intersects_ring()` | `ring: Ring, bond: Bond` |
| intersection tuple | 新别名 `BondRingIntersection = Tuple[Ring, Bond]` |
| `closest_ring_edge_to_bond()` 返回 | `Bond` |
| `closest_ring_opening_edge()` 返回 | `Optional[Bond]` |
| 两个内部 `edge_distance(edge)` | `edge_bond: Bond` |
| `_bond_position_data()` | `Iterator[Tuple[int, Bond, int, int]]` |

这组改动应通过 `TYPE_CHECKING` 引用 Core 类；不得把运行时 Core import 引入 geometry。

原子对距离接口不能盲目标注为 `Molecule`：`Molecule.is_disorder` 和
`Ring.is_disorder` 都会调用 `has_too_close_atoms()`。因此 `_pair_table()`、
`find/has_overlapping_atoms()` 和 `find/has_too_close_atoms()` 存在真实的结构化多态，
应使用只包含 `atoms` 与 `bonds` 的最小 `Protocol`（例如 `AtomBondGeometry`），参数名用
`structure`。这是本次审查中唯一已经有第二种生产对象、值得使用 `Protocol` 的位置。

### GEO-T002：质量门控数据建立真实类型

- `GeometryCheck.measured` 和 `threshold`：使用受限递归
  `GeometryDiagnosticValue`，覆盖当前 scalar、tuple、`AtomTopologySignature`、
  `BondTopologySignature` 和 `None`；不要直接换成 `object` 后结束整改。
- `GeometryQualityReport.metrics`：定义 `GeometryMetricValue`，并把
  coordination-environment 字典提升为 `CoordinationGeometryMetrics` `TypedDict` 或
  immutable dataclass。
- `GeometryQualityThresholdOverrides`：替代三个公开/私有入口的
  `Mapping[str, Any]`，字段与 `GeometryQualityThresholds` 一一对应。
- `forcefield_report`：定义 geometry-owned `ForceFieldQualityReportProtocol` 与候选阶段
  typed mapping。这样 `ForceFieldRunReport` 可以结构化满足协议，同时避免循环 import。
- `_report_value(report, name) -> Any`：当前“动态字段名 + 动态返回”是 `Any` 扩散源。
  后续应先规范化为 typed view，或按字段使用 overload/专用读取函数；不能只把返回值
  改成宽泛 union 后继续无检查地 `float()`、`bool()`。

### GEO-T003：补齐缺失 annotation

以下区域需要补齐参数、iterator 和返回类型，避免以“无 annotation”规避新规则：

- `_iter_overlap_issues()`、`_iter_too_close_issues()`、
  `_iter_bond_ring_intersections()`、`_bond_position_data()` 的 iterator 返回；
- `Point`、`to_point()`、`Line`、`Plane`、`points_on_same_plane()`、`CyclePlanes` 的
  参数、property 和方法返回类型；
- `points_on_same_plane()` 当前真实返回是 `Optional[bool]`，不得误标为 `bool`；
- `get_line_relationship()` 当前注解为 `str`，实际返回 `LinesRelationship`；
  `calculate_line_distance()` 和 `Line.distance_to_line()` 应返回
  `Tuple[LinesRelationship, float]`，不能继续使用 `(str, float)` 这种非类型注解；
- `Plane.line_intersect_point()` 应标注 `Optional[np.ndarray]`，不能使用缺少类型参数的
  bare `Optional`；
- 数组输入应统一使用项目已有 `ArrayLike` 或清晰的 `Sequence[float]`/`np.ndarray`，
  不应把所有数学输入写成 `object`。

### GEO-N001：替换丢失化学含义的对象名

| 当前名称 | 实际对象 | 建议名称 |
|---|---|---|
| `_pair_table(obj)` | `Molecule` 或 `Ring` 的 atom/bond view | `_pair_table(structure)`，类型为最小 `Protocol` |
| `_topology_checks(..., reference)` | topology snapshot | `topology_reference` |
| `_report_value(report, ...)` | forcefield report | `forcefield_report`；若按计划拆分可删除该通用 helper |
| `_forcefield_checks(report, ...)` | forcefield report | `forcefield_report` |
| `bond_positions` 中的 `candidate` | `Bond` | `candidate_bond` |
| closest-edge helper 中的 `edge` | ring `Bond` | `ring_bond` |

`mol`、`atom`、`bond`、`ring`、`coordinates`、`points`、`line` 和 `plane` 都已具有明确
的化学或几何含义，应保留。数学公式内的 `p1/p2`、`v1/v2` 也是局部标准记号，不属于
需要机械扩写的化学对象命名问题。

## 6. 类名审查结论

当前两个模块的类名整体符合新规则，无需仅为命名进行修改：

- `ForceFieldRunReport`、`ComplexBuildDiagnostics`、`CoordinationEnvironment`、
  `_OpenBabelOptimizer` 等能够表达领域用途；
- `AtomPairGeometryIssue`、`GeometryQualityReport`、`TopologyReference` 等能够表达
  geometry 领域语义；
- `Point`、`Line`、`Plane`、`CyclePlanes` 属于数学几何对象，名称虽非化学对象，
  但并不含混。

是否删除 dormant class、调整 class hierarchy 或重构旧几何对象不属于本附件范围。

## 7. 建议实施顺序

1. **类型基础提交**：增加 `TYPE_CHECKING` import、类型别名、`TypedDict`、`Protocol`
   和 generic，不改函数体。
2. **geometry 类型提交**：先收紧 Core 化学对象、intersection、threshold、metric 和
   forcefield-report contract；forcefields 将依赖这些类型。
3. **forcefields 类型提交**：收紧 Molecule/Core/Open Babel/IPC/report/snapshot 类型。
4. **命名提交**：只做 `working_mol`、`clone_mol`、`component_mol`、`ring_bond` 等
   机械重命名，和 annotation 改动分开，便于审查。
5. **缺失 annotation 提交**：补齐 decorator、iterator 和旧几何对象签名。
6. **静态与运行校验**：增加或启用覆盖 Python 3.9–3.14 的类型检查；运行现有
   geometry/forcefield 测试，并用 AST/control-flow diff 确认没有业务逻辑变化。

## 8. 实施边界

- 不得为了让类型检查通过而增加 `cast(Any, ...)`、`# type: ignore` 或无条件分支。
- 不得把所有 `Any` 机械替换为 `object`；只有完全 opaque、仅存取不操作的状态适用
  `object`。
- 不得为绕过 Core 循环导入复制 `Molecule` protocol；已知对象直接使用延迟
  `Molecule` annotation。
- 类型和命名整理不得改变补氢、working-copy、提交、质量门控、worker 或力场策略。
- 如果精确类型暴露出当前数据结构本身不稳定，应另列逻辑问题，不能在本次类型提交中
  顺手改变业务实现。
