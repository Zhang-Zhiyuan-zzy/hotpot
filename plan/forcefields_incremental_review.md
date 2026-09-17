# `forcefields.py` 逐项问题审议记录

## 记录规则

- 按发现顺序记录问题，不预先要求问题属于同一主题。
- 每项区分：当前事实、问题判断、候选整改；候选整改不等于已经批准实施。
- 本阶段只分析和记录，不修改 forcefield 或 geometry 业务逻辑。
- 最终在全部问题审议后，从本记录汇总形成整改计划。
- 本记录的后续附件统一存放在 `plan/reviews/`，并登记到下方附件索引。

## 附件索引

- [A001：`forcefields.py` 与 `geometry.py` 类型和命名审查](reviews/ff_geo_typing_naming_review.md)
- [A002：`forcefields.py` 与 `geometry.py` 兼容性代码审查](reviews/ff_geo_compatibility_review.md)
- [A003：Python 3.9 / Open Babel 3.1 force-field 模块隔离方案](reviews/forcefields_python39_module_split.md)
- [A004：环几何有效性与键—环关系统一评审](reviews/ring_validity_and_intersection_brainstorm.md)
- [A005：FF-Q003 无调用且非预留接口专项审查](reviews/ff_unused_callable_review.md)

## FF-Q001：`working`、`mol` 和 `Any` 分别表示什么？

### 用户问题

`_run_optimizer_on_working()` 的参数为什么叫 `working`，类型却标为 `Any`；传给
`optimizer.optimize()` 后形参又叫 `mol`？`working` 到底是不是 `mol`，还能否输入
其他对象？

### 结论

`working` **是 Hotpot `Molecule` 对象**，但它不是调用方持有的原始分子，而是该
分子的事务性工作副本。`working` 描述的是对象在工作流中的角色和生命周期，不是
一种独立类型。

调用：

```python
optimizer.optimize(working, ...)
```

进入 `_OpenBabelOptimizer.optimize(self, mol, ...)` 后，Python 只是把同一个对象绑定到
局部形参名 `mol`；此时 `mol is working` 为真。这里没有转换，也没有生成另一个对象。
但该 `mol` 仍然不是最外层公开 API 收到的原始分子。

### 实际对象来源

生产代码共有三个 `_run_optimizer_on_working()` 调用点：

```text
_complexes_build_impl
└─ _build_complex_working
   └─ _hydrogenated_working_copy
      └─ working = copy(original_mol)

optimize
└─ _hydrogenated_working_copy
   └─ working = copy(original_mol)

optimize_complex
└─ _hydrogenated_working_copy
   └─ working = copy(original_mol)
```

`Molecule.__copy__()` 委托 `Molecule.copy()`，后者明确创建新的 `Molecule()`。因此三个
生产调用点最终传入的都是 Hotpot `Molecule` 工作副本。

工作副本承担以下事务语义：

1. 在副本上补氢、构筑、优化和执行几何质量门控；
2. 中间失败时不让调用方原始分子处于半修改状态；
3. 全部成功后，才由 `_commit_working_copy(original_mol, working_mol)` 原子化回写。

### 是否支持 `Molecule` 以外的输入？

正常生产路径中没有第二种受支持的类型：

- 不是 Open Babel `OBMol`；`OBMol` 是在优化器内部通过 `mol2obmol()` 临时创建的；
- 不是 RDKit `Mol`；该入口没有 RDKit 转换；
- 也不是通用 molecular graph 接口。

Python 运行时理论上可以传入一个完整模拟 Hotpot 接口的 duck-typed 对象。单元测试也
在 monkeypatch 掉转换和质量门控后使用 `_OptimizerMolecule` 测试替身。但真实流程需要
`atoms`、`bonds`、`crystal`、可写 `coordinates`、构象管理方法，以及提交阶段使用的
`_atoms`、`_bonds`、`_graph`、`_conformers`、`_atom_pairs` 等私有状态。这样的替身
实际上必须重现大部分 `Molecule` 契约，因而不构成正式扩展点。

### 问题判断

| 项目 | 判断 |
|---|---|
| `working` 与 `mol` 是否为不同对象 | 否；在该调用边界是同一个工作副本 |
| `working` 是否为不同类型 | 否；实际仍是 Hotpot `Molecule` |
| 当前事务逻辑是否错误 | 否 |
| `Any` 是否准确 | 否；它错误暗示该入口具有任意输入多态性 |
| 是否属于化学业务逻辑问题 | 否；属于类型契约和命名清晰度问题 |
| 严重程度 | 低到中：不影响当前运行，但削弱静态检查并容易造成错误理解 |

### 候选整改（尚未实施）

1. 通过 `TYPE_CHECKING` 导入 `Molecule`，避免 `core.py` 与 `forcefields.py` 的运行时
   循环导入。
2. 将 `_hydrogenated_working_copy()`、`_build_complex_working()`、
   `_run_optimizer_on_working()` 和 `_OpenBabelOptimizer.optimize()` 的分子参数/返回值
   从 `Any` 收紧为 `Molecule`。
3. 将事务层变量统一命名为 `working_mol`；优化器内部也可采用同名，以持续说明它会被
   原地修改。若希望优化器保持通用底层语义，则可命名为 `target_mol`，但需要明确其
   原地修改契约。
4. 当前不建议为假想的其他后端创建 `Protocol`。只有在出现真实的第二种生产实现后，
   才应根据最小稳定协议抽象类型。

### 用户决策

- 已确定项目级规则：除非确实无法表达边界类型，否则禁止使用 `Any` 作为 annotation。
- 已确定命名规则：类、实例和形式参数优先表达化学对象；变量使用 `mol`、`atom`、
  `cbond` 等简洁领域名称；存在副本或生命周期角色时使用 `clone_mol`、
  `working_mol` 等复合名称。
- 以上规则已经写入 `skills/development.md`；本项后续整改应据此收紧类型并统一命名。

## FF-Q002：当前还保留了哪些兼容性代码？

### 用户问题

`complexes_build()` 附近存在旧参数名翻译。项目目前不承担历史包袱，因此不仅要检查显式
的名称映射，也要识别版本分支、对象形态回退和旧行为保留等其他兼容性设计。

### 结论

完整证据和逐项建议见附件 A002。主要结论如下：

1. `_LEGACY_COMPLEX_BUILD_OPTIONS`、`_translate_legacy_complex_build_options()` 和
   `complexes_build(**options)` 是明确的旧 API 兼容层，应删除并收束为一个具有当前显式
   签名的 `complexes_build()`。
2. `_ob_optimize()` 是公开兼容 primitive 私有化后遗留的无调用函数，应删除；
   `_ob_build()` 仍有真实生产调用，不属于可删除残留。
3. `_atom_index()`、`_rings_for_scope()`、`_bond_kind()` 和 `_report_value()` 都允许当前
   Hotpot 契约之外的对象/数据形态，属于需要收束的结构兼容。
4. Open Babel 3.1 RNG 路径确属版本兼容，但它仍被 Python 3.9 支持矩阵需要；不能只删
   代码而不同时改变 `pyproject.toml`、requirements、CI 和文档中的支持政策。
5. 非平面环的 center-fan 判定明确以“保留历史语义”为目标，但会影响候选结构的几何质量
   门控。它必须先定义新的几何语义，再修改实现和测试，不能作为纯接口清理直接删除。

### 用户决策

- 已确定：旧 API 不保留兼容别名、参数翻译或静默兜底；后续实现直接采用当前名称和显式
  接口。
- 已确定：保留 Python 3.9/Open Babel 3.1，但与 3.10+ 主实现分文件隔离。
- 非平面环和键—环关系的统一候选语义集中记录于 FF-Q004 / 附件 A004，待实现前最终审定。

### 用户补充决策：Python 3.9 隔离而非删除

- 暂不放弃 Python 3.9 和 Open Babel 3.1。
- 将 `forcefields.py` 改为 `forcefields/` package；`__init__.py` 是唯一版本选择点。
- Python 3.10+ 使用 `ff.py` + `utils.py`；Python 3.9 使用 `ff39.py`，并按需连接
  `utils.py` + `utils39.py`。
- `ff.py` 与 `ff39.py` 的公开接口和签名必须完全一致。
- 具体模块边界、迁移顺序及测试围栏见附件 A003。

### 非平面环问题补充结论

当前 center-fan 语义存在已经动态复现的显著缺陷：对一个凹六边形，仅向一个顶点施加
`5e-8 Å` 的 z 方向扰动，就会使位于凹口外部的 probe 从“不穿环”变为“穿环”。这是平面
point-in-polygon 与非平面 center-fan 两套覆盖区域不同造成的阈值不连续；环结构本身并未
发生几何上可分辨的改变。

统一方案不再选择单个 center-fan 或单个 Earcut 面作为最终事实，而是先进行环自身纯几何
分级，再枚举最多 8 元环的全部合法三角剖分并对键—环关系取共识。详细状态、算法、接口和
测试矩阵见附件 A004。

## FF-Q003：无调用且不是预留接口的函数

### 用户问题

以 `_ob_optimize()` 为代表，审查 forcefield/geometry 中既无调用、也没有预留实施目的的
函数，避免将旧兼容残留仅改成私有名称后长期保留。

### 结论

- `_ob_optimize()` 是 forcefields 中唯一已确认同时满足“私有、零调用、非预留”的函数，
  应直接删除。它是旧公开 compatibility primitive 私有化后的残留。
- 其他低引用私有函数均能追踪到真实生产调用、decorator 或 multiprocessing target，不能
  仅凭文本调用次数少而删除。
- `geometry.Point` 不是函数，但同样无任何使用和预留职责，是确定的 dead-code 候选。
- `BuildWorkerResult.conformers` 是无生产读写的闲置字段；它属于 worker schema 清理，不应
  用现有“可以存值”的测试伪装成业务需求。
- `prepare_coordination_geometry()` 及其两个结果类型属于用户明确要求保留的未来实施接口，
  不在删除范围。

完整引用核查见附件 A005。

## FF-Q004：如何统一判定环几何有效性和键—环关系？

### 用户问题

合并审议两个原问题：一是非平面环是否能够得到严格、成熟的穿环判断；二是是否应先判断
环自身的折叠、近接触、自交/打结状态，再判断外部键与环的关系。

### 结论

- 当前 center-fan 存在已动态复现的高严重度误报，并把非法/不可判输入静默转换为 `False`；
  必须删除该语义。
- 统一计算分为环边界状态、环面族和键—环关系三层；讨论只采用纯几何量，不调用力场或
  复杂能量。可选 `O(n²)` geometry penalty 仅用于排序。
- 环状态分为 `VALID/SUSPECT/INVALID/UNRESOLVED`，环面状态独立分为
  `DEFINED/UNSTABLE/UNDEFINED`；两者不能相互冒充。
- 键—环关系分为 `CLEAR/PIERCES/TOUCHES/AMBIGUOUS_SURFACE/UNDEFINED_RING`；未知和
  擦边不能压缩成 `False`。
- 非平面闭合边界没有唯一内部曲面。对当前 `n <= 8` 的环，优先枚举最多 132 个合法
  vertex-only 三角剖分并取关系共识，避免用某一条任意对角线裁决。
- `mapbox-earcut` 已验证能够修复现有凹环反例，但只能生成单个三角面，降级为辅助或测试
  oracle；形式 exact predicates 真正成为硬需求时才评估 CGAL。

当前事实、两组状态分级、具体接口、迁移步骤和测试矩阵统一见附件 A004；原 FF-Q005 不再
作为独立问题保留。
