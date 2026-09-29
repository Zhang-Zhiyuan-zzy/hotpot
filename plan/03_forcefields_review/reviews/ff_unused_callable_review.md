# FF-Q003：无调用且非预留接口专项审查

> 实施状态（2026-09-21）：`_ob_optimize()`、`BuildWorkerResult.conformers` 和拆包后确认
> 无版本差异的一行 `_iter_obmol_atoms()` wrapper 已删除。`Point` 已成为 geometry package
> 的基础公开对象，按用户后续决定保留。

## 1. 判定规则

“源码中没有普通函数调用”不等于“无调用”。本审查同时检查：

- 模块内直接名称引用；
- 其他 Hotpot 模块、测试和文档中的引用；
- decorator、callback 和 `multiprocessing.Process(target=...)` 等间接引用；
- `__all__` 中的公共 API；
- 明确标注为 future hook / reserved interface 的预留入口；
- Git 历史中函数是否由旧公开兼容入口遗留。

只有同时满足以下条件才列为确定删除项：

1. 没有生产调用；
2. 没有测试直接验证其独立业务契约；
3. 不是公共业务入口；
4. 不是已批准的预留实施接口；
5. 不作为 decorator、worker target 或其他 callback 被间接调用。

## 2. 审查结论

### D001：`_ob_optimize()`——确定删除

**位置**：`hotpot/cheminfo/forcefields.py:489-491`

```python
def _ob_optimize(mol: Any, ff: str = "UFF", steps: int = 100) -> float:
    return _single_ob_optimization(mol, ff, steps).energy
```

证据：

- 全仓库没有 `_ob_optimize(...)` 调用；唯一匹配是函数定义本身；
- 不在 `forcefields.__all__`；
- 没有单独业务能力，仅丢弃 `_single_ob_optimization()` 返回值中的 `energy_unit` 和
  `exploded`，保留 energy；
- Git 历史显示它原本是公开 `ob_optimize()`，docstring 为 “Compatibility primitive”；
  提交 `707bc9b` 将其私有化，但没有随旧 API 一并删除；
- 当前公开 `optimize()` 走 `_OpenBabelOptimizer`，配体候选则直接调用
  `_single_ob_optimization()`，两条生产路径都不需要它。

整改：直接删除函数，不重命名、不提供 alias、不迁移调用。保留
`assert not hasattr(ff, "ob_optimize")` 这一公共旧入口负向测试；无需为私有名称不存在建立
长期公共契约。

### D002：没有发现第二个满足全部删除条件的 forcefield 函数

AST 名称引用统计中，除 `_ob_optimize()` 外，每个私有顶层函数至少有一个真实模块内引用。
对引用数较低、容易被误判的函数复核如下：

| 函数 | 实际用途 | 判断 |
|---|---|---|
| `_find_forcefield_prototype()` | `_get_forcefield()` 的唯一 plugin lookup；同时是精确测试 seam | 在用 |
| `_iter_obmol_atoms()` | optimizer 的梯度采集 | 在用 |
| `_format_geometry_rejection()` | 配体候选质量拒绝信息 | 在用 |
| `_resolve_organic_forcefield()` | 公开 `optimize()` 的后端选择 | 在用 |
| `_recalculate_neutral_donor_valence()` | 络合物 working copy 补氢前处理 | 在用 |
| `_prepare_working_copy_commit()` | 原子化提交前校验和 payload 构造 | 在用 |
| `_snapshot_molecule_for_commit()` | 提交失败回滚 | 在用 |
| `_restore_failed_commit()` | `_commit_working_copy()` 异常路径 | 在用 |
| `_build_ligand_proxies()` | complex worker 中执行真正的代理构筑 | 在用 |
| `_run_complexes_build()` | multiprocessing target | 间接在用 |
| `_run_seeded_ob_build()` | seeded ordinary build 的 multiprocessing target | 间接在用 |
| `_seeded_ob_build_coordinates()` | `build3d(seed=...)` | 在用 |
| `_translate_legacy_complex_build_options()` | 只服务旧参数兼容层 | 当前有调用，但随 C001 整组删除 |

`_ob_build()` 也不是删除项。虽然它曾是公开 compatibility primitive，但现在承担三个内部
生产职责：普通无 seed 3D 构筑、seed worker 内部构筑，以及配体候选构筑。

## 3. 不能按“模块内零调用”删除的公共函数

以下函数可能在 `forcefields.py` 自身没有调用点，但它们是当前公开 API 或由 Core 调用：

| 函数 | 外部引用/职责 |
|---|---|
| `perturb()` | `__all__` 中的独立坐标微扰工具，文档公开 |
| `collect_coordination_environments()` | 当前公开配位环境分析，已有专项测试 |
| `prepare_coordination_geometry()` | 已批准保留、尚未实施的配位数结构处理 hook |
| `build_complex3d()` | 当前公开的“只构筑络合物”组合入口 |
| `build_and_optimize()` | `Molecule.build3d()` 的标准 façade 后端 |
| `auto_optimize()` | `Molecule.optimize()` 的标准 façade 后端 |

公共 API 是否仍有必要属于接口设计问题，不能只按仓库内引用数做 dead-code 删除。

## 4. 非函数残留

以下内容不属于本次“无调用函数”结论，但应登记到后续清理：

### `BuildWorkerResult.conformers`

`forcefields.py:139` 的 `conformers` 字段没有生产写入或读取，唯一直接使用是测试
`tests/test_cheminfo/test_complexes_build.py:670-672` 验证它可以存值。代码没有说明这是
后续预留字段，因此目前更像 generalized worker envelope 重构留下的闲置 schema。

建议在 worker protocol 专项整改中删除该字段和“只验证可存储”的测试；不能用这项测试
反向证明字段有业务用途。

### `geometry.Point`

本段结论已被后续 geometry package 重构推翻：`Point` 现在由关系计算、转换层、测试和
双语 API 文档共同使用，是必须保留的基础几何值对象，不再属于清理候选。

相比之下，`Line`、`Plane`、`CyclePlanes` 和 `to_point()` 仍有 Core 或彼此之间的调用，
不能因代码风格较旧而直接删除。

### 预留的配位几何类型

`CoordinationGeometryCandidate`、`CoordinationGeometryResult` 当前没有被构造，但与明确保留的
`prepare_coordination_geometry()` hook 构成预留契约。根据用户要求“不是预留实施接口”才
删除，这两类暂不列为 dead code；待实施该 hook 时一起校验字段设计。

## 5. 候选实施节点（待批准）

首个纯代码清理节点可只执行：

1. 删除 `_ob_optimize()`；
2. 删除 `geometry.Point`；
3. 删除 `BuildWorkerResult.conformers` 及其无业务意义的存储测试；
4. 运行 forcefield、geometry、complex build 测试；
5. 单独提交，不夹带命名、类型、算法或模块迁移。

第 2、3 项虽不是函数，仍符合“无使用且无预留目的”的同一清理原则；若希望 FF-Q003 严格
只处理函数，可把它们拆成后续独立提交。
