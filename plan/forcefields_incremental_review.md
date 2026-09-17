# `forcefields.py` 逐项问题审议记录

## 记录规则

- 按发现顺序记录问题，不预先要求问题属于同一主题。
- 每项区分：当前事实、问题判断、候选整改；候选整改不等于已经批准实施。
- 本阶段只分析和记录，不修改 forcefield 或 geometry 业务逻辑。
- 最终在全部问题审议后，从本记录汇总形成整改计划。
- 本记录的后续附件统一存放在 `plan/reviews/`，并登记到下方附件索引。

## 附件索引

- [A001：`forcefields.py` 与 `geometry.py` 类型和命名审查](reviews/ff_geo_typing_naming_review.md)

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
