# 附件 A006：默认语义命名与冗余封装审查

> 实施状态（2026-09-21）：已完成。本文列出的命名、无策略 wrapper 与 report 伪泛化均已
> 收束；`excluded_ring_count` 后续已实现为真实计数，因此保留用于大环覆盖 warning。

## 1. 结论

本次问题成立。`Molecule.rings` 和 `Molecule.rings_for_scope()` 已经把 Relevant
Cycles 定义为业务层默认的 `rings`，因此 force-field 业务代码再使用
`relevant_ring` 属于重复编码默认值。

本轮已经将名称恢复为：

```text
_select_ring_opening_edge
ring_memberships
candidate_ring
```

该函数仍调用 `mol.rings_for_scope(ring_scope)`；只是名称不再重复底层算法。测试继续明确
禁止开环流程读取旧 `cycle_basis_rings`，因此简化名称没有弱化行为契约。

审查还确认了：

- 业务层存在 2 组同类冗余算法限定名；
- 存在 5 个确定或高置信的无价值转发层；
- geometry report 中存在 3 组“当前实现永远得到同一值”的伪泛化字段；
- `geometry.Cycle` 没有此问题，它是纯几何名，且与环识别算法完全独立。

除 `_select_ring_opening_edge()` 的纠正外，本附件只记录后续整改项，不在本轮混入接口
删除或报告结构变更。

## 2. 命名边界

| 层级 | 默认表达 | 需要保留限定词的情形 |
|---|---|---|
| 化学业务层 | `ring`、`rings`、`ring_memberships` | 显式选择非默认环族，例如 `cycle_basis_rings` |
| 图算法层 | `relevant_cycles()` | Relevant Cycles 是算法正式名称，不能简化为含混的 `cycles()` |
| 几何层 | `Cycle` | 这是数学对象名称，不应改成 `Ring` 或 `RelevantCycle` |
| 生命周期 | `mol` | 同一作用域存在原分子与副本时，使用 `working_mol`、`clone_mol` |
| 后端 | 通用业务名 | 实际限定 Open Babel 实现时保留 `_ob_*` |
| 测试和文档 | 描述被验证的性质 | 可明确写 Relevant Cycles 或 cycle basis，以说明证据范围 |

判定原则：一个限定词必须区分当前同层级中真实存在的后端、算法、化学语义、来源或
生命周期；仅强调默认实现时就是噪声。

## 3. 已纠正项

| 位置 | 原名称 | 当前名称 | 判断 |
|---|---|---|---|
| `forcefields.py` 开环选边 | `_select_relevant_ring_opening_edge` | `_select_ring_opening_edge` | `ring` 已代表默认环，不需要重复 `relevant` |
| 同一函数局部映射 | `relevant_ring_memberships` | `ring_memberships` | 映射只存在一种环成员关系 |
| 同一函数循环变量 | `relevant_ring` | `candidate_ring` | `candidate` 表达循环角色；算法限定是多余的 |

需要保留的行为说明是：FF 穿环检测默认受 `max_ring_size=16` 限制，但选边时必须查询当前 scope
下的全部 rings，以免把同时属于更大环的共享边错判为可开边。这个约束应写在 docstring
和测试里，不应塞进函数或变量名。

## 4. 仍存在的冗余命名

### NAME-001：Core 默认 rings 的内部实现名重复算法

| 当前名称 | 建议名称 | 影响范围 |
|---|---|---|
| `Molecule._materialize_relevant_rings()` | `_materialize_rings()` | `rings`、`rings_for_scope()`、`ligand_rings` 的内部调用 |
| `Molecule._relevant_cycle_indices_cache` | `_ring_indices_cache` | Core 初始化、失效逻辑、force-field 事务快照及缓存测试 |
| `_MoleculeCommitSnapshot.relevant_cycle_indices_cache` | `ring_indices_cache` | 仅随 Core 缓存机械同步 |

这些名称位于化学对象业务层；真正的算法调用
`graph_algorithms.relevant_cycles(...)` 应继续保留正式算法名。`_cycle_basis_rings` 和
`cycle_basis_rings` 是非默认旧环族，也必须保留限定词。

建议作为一个纯机械重命名提交实施，不改变缓存 key、Ring identity、枚举上限或失效时机。

### NAME-002：geometry 化学适配记录使用模糊 `source`

| 当前字段/变量 | 建议名称 | 理由 |
|---|---|---|
| `AtomGeometry.source` | `atom` | 类型已经是化学原子，直接表达对象 |
| `BondGeometry.source` | `bond` | 类型已经是化学键 |
| `RingGeometry.source` | `ring` | 类型已经是化学环 |
| `RingEdgeDistance.source_bond` | `bond` | 所属 dataclass 已提供上下文 |
| `target_source` | `targets` | 实际值是 `BondRingTarget` iterator，不是 source object |

这不是默认值重复，而是违反既定的化学对象命名规则。字段属于已公开 geometry API，需在
独立提交中同步 Core、forcefields、测试和双语 README。

### NAME-003：生命周期名称不完整，但限定词本身必要

| 当前名称 | 建议名称 | 判断 |
|---|---|---|
| `_build_complex_working()` | `_prepare_complex_working_mol()` | `working` 不能删除；应补足对象名并准确表达 preparation |
| `_run_complexes_build()` | `_build_ligand_proxies_worker()` | 实际是 child-process target，不是完整公开 workflow |
| `_run_seeded_ob_build()` | `_seeded_ob_build_worker()` | 与上一 worker target 统一表达进程角色 |
| `_run_optimizer_on_working()` | `_optimize_working_mol()` | helper 有价值，但当前名称把化学对象缩成了形容词 |
| `_structure_worker_proxy()` | `_make_worker_mol()` | 实际返回为 worker 重编号的 `Molecule`，不是抽象 proxy 类型 |
| 局部变量 `worker_proxy` | `worker_mol` | 值是具体 `Molecule` |

`working_mol`、`clone_mol`、`component_mol`、`requested_forcefield` 和
`effective_forcefield` 都在区分真实生命周期或策略，不属于冗余命名。

## 5. 确定或高置信的冗余封装

| 编号 | 位置 | 现状 | 建议 |
|---|---|---|---|
| WRAP-001 | `forcefields._ob_optimize()` | 零调用；只返回 `_single_ob_optimization(...).energy` | 删除；已在 FF-Q003 单独确认 |
| WRAP-002 | `forcefields._capture_workflow_topology()` | 五个调用点；逐参数转发公开 `capture_topology()`，无策略或适配 | 调用点直接使用 `capture_topology()` |
| WRAP-003 | `forcefields._format_geometry_rejection()` | 一个调用点；只把 `report.failures` 交给 `_format_geometry_checks()` | 内联后删除 |
| WRAP-004 | `Molecule.cycle_basis_rings_for_scope()` | 无生产消费者；测试仅以 monkeypatch 证明 FF 不调用它 | 删除；保留两个明确的旧属性及共用私有 materializer |
| WRAP-005 | `geometry.object._cycle_edges()` | 仅被 `Cycle.edges` 调用一次且没有独立契约 | 可直接内联到 property |

以下两项不能直接归入普通小清理：

- `_iter_obmol_atoms()` 当前只是 `ob.OBMolAtomIter()` 的一行包装。若 Python 3.9 / Open
  Babel 3.1 分模块后两侧没有不同实现，应删除；若确有版本差异，应迁入版本 utils，而不是
  留在主业务文件充当测试 seam。
- `_complexes_build_impl()`、`_translate_legacy_complex_build_options()` 和
  `complexes_build(**options)` 构成旧参数兼容双层入口。删除方案已经在 A002 批准，但它
  涉及完整公共签名，应作为兼容层清理节点整体实施。

## 6. geometry report 中的伪泛化

这些项目不是名称本身错误，而是数据结构宣称了当前并不存在的变化维度：

| 项目 | 当前事实 | 后续选择 |
|---|---|---|
| `_selected_rings()` 第二返回值 | 已改为真实的被排除环数 | 保留，用于报告大于 `max_ring_size` 的未扫描环 |
| `excluded_ring_count` | 已可准确计算 | 保留；FF 将非零值记录为结构化 warning，不把它当作确认互穿 |
| `candidate_pair_count` / `evaluated_pair_count` | dense scan 中都等于 `len(findings)` | 已删除重复的 `evaluated_pair_count`；`candidate_pair_count` 改为只读派生属性 |
| `RingFamily` | 协议无法验证第三方 `rings_for_scope()` 的算法 | 已从 report 和公开 API 删除，避免硬编码自证 |

这一组已作为独立 schema 提交同步修改 `BondRingScanReport`、force-field metrics、测试和
双语 README。若未来需要记录科研溯源，必须由 ring provider 显式提供可验证的 provenance，
不能恢复未经接口证明的硬编码 `ring_family`。

## 7. 已审查并应保留

| 接口或名称 | 保留理由 |
|---|---|
| `graph.relevant_cycles()`、`RelevantCycleLimitExceeded` | Relevant Cycles 的正式图算法名称及其专属失败类型 |
| `cycle_basis_rings`、`ligand_cycle_basis_rings` | 明确标识非默认旧算法，限定词不可省略 |
| `geometry.Cycle`、`SegmentCycleRelation`、`CycleSurfaceModel` | 纯数学几何概念，不包含环感知来源 |
| `point_from_atom()`、`segment_from_bond()`、`cycle_from_ring()` | 化学对象到纯几何对象的必要边界，虽短但不是无意义转发 |
| `measure_planarity()`、`determine_segment_cycle_relation()` | 隐藏数值实现并提供稳定的事实判定 API |
| `iter_segment_cycle_relations()` | 对同一环复用预处理的批量路径，有真实性能语义 |
| `determine_bond_ring_piercing_state()` | early-exit 查询，与 dense report 有真实复杂度差异 |
| `scan_bond_ring_relations()` | 返回完整可审计证据，与 early-exit 查询职责不同 |
| `_run_optimizer_on_working()` 的职责 | 集中三个流程共同的 optimizer 构造和参数传递；应保留该边界，但按 NAME-003 改名 |
| `_structure_worker_proxy()` 的职责及 IPC helper | 承担进程边界、ID 重写、协议验证或锁；应保留职责，但按 NAME-003 改名 |
| `_make_constraints()` | 已明确保留的约束扩展接口，即使当前返回空约束也不是历史兼容层 |
| `_find_forcefield_prototype()` | 隔离 nullable native plugin lookup，并提供稳定测试 seam |

`Point.from_coordinates()` 与 `Point(...)` 当前等价，但前者是已公开且可读的 named
constructor；单纯少一行代码不足以证明应删除，暂不列入整改。

## 8. 推荐实施顺序

1. 已完成：恢复 `_select_ring_opening_edge()`，保持行为测试。
2. 纯内部命名：实施 NAME-001 与 NAME-003，每组单独提交。
3. geometry 公开字段命名：实施 NAME-002，同步文档和所有消费者。
4. 无调用/无策略 wrapper：依次处理 WRAP-001 至 WRAP-005。
5. 单独审议并实施 geometry report schema 收缩。
6. 按 A002 删除 legacy complex-build 参数翻译层。

每一步都只处理一种问题，不借“简化命名”改变 ring perception、穿环数学判定、开环规则、
优化策略或失败处理。
