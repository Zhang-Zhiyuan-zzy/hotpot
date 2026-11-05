# 动态事实投影、写权限与生命周期

本附件补充 AST 类体无法直接列出的有限动态 API。来源均为当前 `2c2ef889abb6adcceba043b606da2e718dbea109` 源码；没有导入 Hotpot、调用模型、引擎或测试。其余方法、字段及继承族见 [完整 API 树及主题签名](../delivery_api_tree.md)。

## Atom：19 列数值投影

源码 `structure/_fields.py:ATOM_SCHEMA/ATOM_SCALAR_READONLY`、`structure/atom.py:_numeric_property`、`structure/_native/facts.cpp:atom_setattr/install_dispatch`。数值 property 的 fset 缺省不能用来断言只读：当前标量写操作由 native `Atom.__setattr__(self, key, value)` 分派到事实提交边界。

| 顺序 | 精确属性 | 投影类型 | 标量权限 | 含义/边界 |
|---|---|---|---|---|
| 0 | `atomic_number` | `int` | 读/写 | 元素事实，走 eager setter |
| 1 | `n` | `int` | 只读 | 电子组态投影 |
| 2 | `s` | `int` | 只读 | 电子组态投影 |
| 3 | `p` | `int` | 只读 | 电子组态投影 |
| 4 | `d` | `int` | 只读 | 电子组态投影 |
| 5 | `f` | `int` | 只读 | 电子组态投影 |
| 6 | `g` | `int` | 只读 | 电子组态投影 |
| 7 | `formal_charge` | `int` | 读/写 | 原子形式电荷事实 |
| 8 | `partial_charge` | `float | None` | 只读 | 当前完整部分电荷结果投影 |
| 9 | `is_aromatic` | `bool` | 只读 | 已准备的芳香表达投影 |
| 10 | `x` | `float` | 读/写 | Cartesian 坐标 |
| 11 | `y` | `float` | 读/写 | Cartesian 坐标 |
| 12 | `z` | `float` | 读/写 | Cartesian 坐标 |
| 13 | `valence` | `int` | 只读 | 化合价规则计算投影 |
| 14 | `implicit_hydrogens` | `int` | 读/写 | 显式声明与规则准备的来源保持区分 |
| 15 | `id` | `int` | 读/写 | 公开来源编号，不是私有 UID |
| 16 | `x_constraint` | `bool` | 读/写 | x 轴约束事实 |
| 17 | `y_constraint` | `bool` | 读/写 | y 轴约束事实 |
| 18 | `z_constraint` | `bool` | 读/写 | z 轴约束事实 |

其他事实/别名权限：

- `symbol`: 读/写，映射到 `atomic_number`。
- `coordinates`: 读/写，映射到 `x/y/z`，一次事实提交。
- `constraint`: 读/写，读为三个轴约束都为真；写入三个轴约束。
- `isotope`: 当前读取由 stereo 事实投影，写入由 native sparse 分派提交；缺失不等于任意质量数。Python 类体仅 getter，实际写权限须结合 native 源码判定。
- `frac_valence`: native 分派保留的可写稀疏实例属性，仅赋值后存在；不是19列数值字段，不能据此扩张构造器或批量字段集合。
- `electron_configuration`, `mca`, `idx`, `mol`, `molecule`, `attrs`, `attrs_enumerator`, `attr_dict`, `rings` 等为只读投影。是否能够对返回的容器执行变更，不等于变更权威事实。

`Atom(mol: Molecule=None, **kwargs)` 当前接受的声明键：`atomic_number`, `symbol`, `formal_charge`, `implicit_hydrogens`, `coordinates`, `x`, `y`, `z`, `id`, `constraint`, `x_constraint`, `y_constraint`, `z_constraint`。完整签名以声明目录为准。

## Molecule：批量事实入口与帧观察

```python
Molecule.updata_atoms(self, *, atoms=None, **columns)
```

当前精确拼写是 `updata_atoms`。`atoms=None` 表示全部，或给属于本对象的 Atom 句柄选择。已审批量可写键为 `atomic_number`, `symbol`, `formal_charge`, `implicit_hydrogens`, `id`, `coordinates`, `x`, `y`, `z`。约束、结果、索引、owner 等不自动变成批量可写列。

`Molecule.atom_matrix` 是只读 property，按上述19列顺序返回独立的 `float64` 数组；对返回数组的修改不构成事实提交。当前没有 `atom_attr_matrix` 并行 property。

`charge` 与 `spin_mult` 具有声明 setter。`coordinates` 有完整坐标 setter。下列7个属性在 `structure/molecule.py:_frame_observation_property` 安装：

| 属性 | 权限 | owner |
|---|---|---|
| `energy` | 读/写 | 选中帧观察/完整赋值 |
| `partial_charge` | 读/写 | 选中帧/完整部分电荷赋值 |
| `gibbs` | 读/写 | 选中帧观察 |
| `force` | 读/写 | 选中帧观察 |
| `zero_point` | 读/写 | 选中帧观察 |
| `thermo` | 读/写 | 选中帧观察 |
| `capacity` | 读/写 | 选中帧观察 |

帧观察与原子级 `Atom.partial_charge` 不共享写权限。已返回的不可变结果内容保留；owner 的当前性另行检查，不能将失效 attachment 继续投影为当前观察。`Molecule.properties`、可选 `identifier` 为现有实例数据，未扩张成任意新的事实 schema。

## Bond、角度与扭转约束

`Bond.bond_order`, `Bond.constraint`, `Bond.id` 由 `MolBlock.__getattr__/__setattr__` 数值族提供，均可读写。`bond_kind`, `bond_direction`, `bond_source`, `bond_source_metadata`, `is_aromatic` 为只读 property；正常 order 更新经过完整事实提交。

`Angle.constraint`、`Torsion.constraint` 由 `views/geometry.py:_constraint_property` 在类体之后安装，为读/写 bool。它们检查视图来源并调用 `_commit_constraints`，不保留独立可写约束真相。`Molecule.clear_constraints()`、`constraint_bonds_angles(exclude_metal_bonds=True)` 是集合操作。`structure.constraints._ConstraintSet` 是私有 owner，没有新顶层 Constraint 导出。

## Crystal 与对称性

`Crystal.a`, `b`, `c`, `alpha`, `beta`, `gamma` 由 `structure/crystal.py:_parameter` 动态安装，均读/写，提交晶胞更新。`matrix`, `volume`, `space_group`, `symm_units`, `site_relations`, `bond_images`, `representation`, `image_map`, `molecule` 是只读 property，修改通过命名操作 `set_cell`, `set_space_group`, `set_site_relations` 完成。

`SpaceGroup` 不可变，`SpaceGroup(number)` 使用随包表；另有 `from_operations(operators, *, reference_number=None, to_reference=None)`。`SymmUnit` 为公开抽象族，实际实例统一为私有 `_AffineSymmUnit(R, t, context)`，通过 `SymmUnit.from_affine`, `from_xyz`, `identity` 与 `SpaceGroup.symm_units/lift` 创建。`R`, `t`, `context`, `xyz`, `kind` 是只读。

实际 `kind` 家族是 `identity`, `translation`, `inversion`, `rotation`, `screw`, `reflection`, `glide`, `rotoinversion`。不虚构同名公开子类。对称运算协议为 `@`/`compose`, `~`/`inverse`, `**`/`power`，另有 `operate`, `reduce_to_cell`、相等/哈希、复制和序列化方法；`SpaceGroup` 支持 `in`/`contains`。详细精确签名见声明目录。

## 视图、集合与协议继承

`AtomSeq` 的 `atomN`, `bondN`, `aNidx` 是1开始的有限成员访问族，N 必须为实际成员位置；它们不是字面属性名。成员、索引与几何的来源约束在对应操作检查。本地继承已经展开到 Bond、Angle、Torsion、Ring、JointRing。

`AtomPairs` 继承 `collections.abc.Mapping`，除了本地 `__len__`, `__iter__`, `__getitem__`, `getdefault`，还保留 `get`, `items`, `keys`, `values` 等标准映射族。`Conformers` 是 `len`/索引/迭代集合，保留 `index_attr/add/clear/coordinates/energy/gibbs/partial_charge/force`。其 `add` 的注解在模块结尾覆盖；机器目录同时保留原声明与注解覆盖来源。

`MolBundle` 的实际公开实例面是 `list_mol` 与 `to_pyg_dataset`，加构造器；它没有自行声明 `__iter__`, `__len__` 或 `__getitem__`，本目录不把理想化集合协议添加给它。其他类型的运算、集合、复制和序列化 magic 方法都随各类完整成员列出。

## 转换、stereo 与 native schema 边界

通用图入口是 `Molecule.to_pyg`, `Molecule.from_pyg`, `Crystal.to_pyg`, `Crystal.from_pyg`；当前 `hotpot.molecule/2` 详见 `operations/pyg.md`。集合输出为 `to_pyg_dataset`。部署模型保持 `operations.inference.features`/各自 MCA、CBond 输入接口，通用图不冒充模型特征。

`hotpot.cheminfo.stereo` 显式导出23个名称，包括全部18个运输/结果类型与5个命名操作。实际 schema 字段、Enum 常量与不可变 dataclass 字段逐个列于完整目录，定义 owner 为 `structure.stereo`，既有公开身份为 `hotpot.cheminfo.stereo`。Level-1 payload 与 Level-2 assign/validate/align 的职责保持区分。

原生 `.pyi` 的类字段并不单独证明写权限。本目录结合当前 pybind binding 的 `def_readonly/def_readwrite/def_property[_readonly]` 记录可确定的 native 字段权限；只靠 stub 无法确定者保留“not specified by stub”。原生边界不是自动批准用户 API 的依据。
