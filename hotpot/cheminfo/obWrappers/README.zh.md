# Open Babel 包装规则

## 简介

`hotpot.cheminfo.obWrappers` 是围绕少数 Open Babel 操作建立的小型、可审计规则层。它只处理已经确认的后端病态行为，不替换 Open Babel，也不改变 Hotpot 已公开的力场接口。

本包严格分离三项职责：

1. C++17 规则规划器读取不可变的原子、键和坐标快照，生成显式修改计划；
2. Python bridge 将计划应用到 Open Babel SWIG 对象；
3. 原生 `OBBuilder` 或 `OBForceField` 执行实际的化学操作。

如果没有规则命中，bridge 只调用一次原生 Open Babel 操作，不修改分子。每次规则应用都会报告稳定的规则 ID、版本、阶段、优先级、目标原子、度量和具体修改。

本包目前处理：

- Open Babel `OBBuilder` 在一个严格限定的中性四配位 P(V) 环境下可复现的构型缺陷；
- UFF 优化前的数值预检：拓扑上不应线性、但几何上严格或近似共线的中心可能导致 proper-torsion 梯度非有限。

本包**不会**修改 UFF 能量或梯度公式，不实现新力场，不判断结构的化学合理性，也不负责金属重定位。特别是，基准 case 109 中的 Eu--N 塌缩不属于本包范围，应由络合物配位工作流处理。

## 设计边界

### 为什么采用 C++ 规划器和 Python SWIG bridge？

Open Babel 的 Python 对象由 SWIG 生成。若将 `OBMol.this` 中未公开的指针传给另一个 pybind11 模块，会把 Hotpot 耦合到 SWIG 内部实现、对象所有权和具体 Open Babel C++ ABI；Open Babel 3.1 与 3.2 的动态库 ABI 也不同。

因此，原生 `_ob_rules` 扩展不依赖 Open Babel C++。它接收普通值快照，返回普通修改记录。只有 Python bridge 读取或修改 `OBMol`。该设计能够：

- 避免在 SWIG 和 pybind11 之间共享 C++ 对象指针；
- 避免重复加载另一份 Open Babel 动态库；
- 让同一套原生规则支持不同 Open Babel 版本；
- 对未命中分子保留原生 Open Babel 路径；
- 让每次例外修改均可观察、可测试。

### 为什么不继承、fork 或直接修改 Open Babel？

| 方案 | 决定 | 原因 |
|---|---|---|
| 当前的组合包装 | 采用 | 局部、可逆、带版本，而且不改变公开力场 API |
| 继承 `OBBuilder`/`OBForceField` | 不采用 | 所需内部操作不是稳定扩展点，会依赖 protected 实现和 ABI 行为 |
| 长期维护 Open Babel fork | 本包不采用 | 可以直接改 UFF 导数，但升级、二进制分发和验证成本很高 |
| 向 Open Babel 上游提交补丁 | 可并行推进 | 适合最终修复奇异导数，但不能立即保护已经发布的 Open Babel 版本 |

共线规则只避免已知奇异输入并校验后端状态，不能称为修正了 UFF 公式。

## 包结构

```text
hotpot/cheminfo/obWrappers/
├── __init__.py                 # Python 公开导出
├── contracts.py                # 不可变公开报告和值对象
├── builder.py                  # 带规则的 OBBuilder 委托
├── forcefield.py               # UFF 预检及有限状态检查
├── registry.py                 # 只读公开注册表视图
├── settings.py                 # 规则数值阈值
├── snapshot.py                 # SWIG 快照、计划翻译和修改应用
├── native.py                   # 延迟导入编译扩展
├── _ob_rules.pyi               # 原生模块类型接口
└── _native/
    ├── bindings.cpp            # pybind11 绑定
    ├── rules.hpp               # 原生值类型和规则契约
    ├── registry.hpp            # 注册表与规划器声明
    ├── registry.cpp            # 校验、排序和执行
    ├── phosphorus_builder.cpp  # 中性四配位 P(V) 构筑规则
    └── degenerate_torsion.cpp  # 确定性共线修复计划
```

## 调用流程

### 坐标构筑

```text
forcefields.backend._ob_build(mol)
  -> mol2obmol(mol)
  -> obWrappers.build(obmol, builder=OBBuilder())
       -> 提取原子与键快照
       -> _ob_rules.plan_build(...)
       -> 未命中：只调用一次 OBBuilder.Build(obmol)
       -> 命中：
            应用临时杂化修改
            只调用一次 OBBuilder.Build(obmol)
            在 finally 中恢复杂化状态
  -> 将坐标写回 Hotpot Molecule
```

### 力场设置与优化

```text
forcefields.backend._setup_forcefield_backend(...)
  -> obWrappers.prepare_optimization(obmol, effective_forcefield)
       -> 非 UFF：严格不操作
       -> UFF：
            提取原子、键与坐标快照
            _ob_rules.plan_optimization(...)
            应用确定性坐标修改
  -> OBForceField.Setup(obmol, constraints)
  -> 若有规则应用：
       validate_forcefield_state(backend, obmol)
       拒绝非有限能量或梯度
  -> 执行原生最速下降或共轭梯度优化
```

短程候选优化与按 epoch 运行的状态化优化器共享同一个 setup 函数。因此，该预检会覆盖配体 warmup、环开结、配位键恢复、普通优化和络合物优化。

## 注册契约

原生规则定义为：

```cpp
using RuleCondition = bool (*)(
    const MoleculeSnapshot&,
    const RuleParameters&
);

using RuleAction = void (*)(
    MoleculeSnapshot&,
    const RuleParameters&,
    const RuleDescriptor&,
    RulePlan&
);

struct RuleDefinition {
    RuleDescriptor descriptor;
    RuleCondition condition;
    RuleAction action;
};
```

condition 只判断是否执行 action。action 只修改规划器内部的快照，并向 `RulePlan` 添加显式修改记录；它不能调用 Open Babel、选择工作流分支、判定化学结构是否可接受或吞掉失败。

规则在原生模块初始化时通过 `RuleRegistrar` 注册，并按照以下顺序确定性执行：

```text
(stage, priority, rule_id, version)
```

同一计划内，后续规则可看到前序规则对规划快照的修改。重复的 `(stage, rule_id)` 会被拒绝。一个计划最多生成 256 次规则应用。Python 目前只提供注册表读取接口。

### 添加编译期规则

1. 选择一个已有 `RuleStage`；只有确实出现新的生命周期边界时，才同时扩展 C++ 和 Python 枚举。
2. 在 `_native/` 中添加一个专注于单一问题的 C++ 源文件，分别实现纯 condition 和 action。
3. 为规则指定稳定的语义 `rule_id`、版本、阶段和优先级。
4. 使用翻译单元局部的 `const RuleRegistrar` 完成注册。
5. 将新源文件加入仓库根目录 `setup.py` 的 `_ob_rules` 扩展。
6. 添加原生边界测试：阳性命中、每一项明确排除、确定性及规则顺序。
7. 添加 bridge 测试：临时修改能够恢复，未命中输入只调用一次原生 Open Babel。
8. 添加真实力场集成测试和适用基准，再考虑默认启用。

不应注册宽泛的兜底规则。每条新规则必须对应一个已有证据的后端问题，并具有狭窄的纳入条件和明确排除条件。

## 内置规则 1：四配位 P(V) 构筑

描述符：

```text
rule_id  = tetracoordinate_pentavalent_phosphorus_build
version  = 1.0.0
stage    = pre_build
priority = 100
```

对于原子 $p$，以 $Z_p$ 表示原子序数，$q_p$ 表示形式电荷，$h_p$ 表示 Open Babel 杂化状态，$d_p$ 表示图上的度，$o_b$ 表示相邻键的整数键级。只有同时满足以下条件才会命中：

$$
Z_p=15,\quad q_p=0,\quad h_p=5,\quad d_p=4,
$$

$$
\sum_{b\sim p}o_b=5,
$$

并且四根相邻键严格由以下部分组成：

- 一根 P 与 O 或 S 之间的非芳香双键；
- 三根非芳香单键。

action 生成以下临时修改：

$$
h_p:5\rightarrow3.
$$

bridge 只在 `OBBuilder.Build()` 运行期间应用该修改，并在 `finally` 中恢复 $h_p=5$。形式电荷、键级、芳香性标志和拓扑均不改变。

明确排除：

- 三配位磷；
- 带正电的 phosphonium；
- 五配位 phosphorane；
- P$^+$--O$^-$ 等电荷分离表示；
- P=C 等不与 O/S 相连的双键；
- 存在芳香相邻键；
- Open Babel 杂化状态不是 5。

这是构筑器修正规则，并不声称所有命中 P(V) 中心在优化后都必须严格为四面体。

## 内置规则 2：非线性扭转的退化几何预检

描述符：

```text
rule_id  = degenerate_nonlinear_torsion
version  = 1.0.0
stage    = pre_forcefield_setup
priority = 100
```

只有 `prepare_optimization(..., forcefield)` 接收到不区分大小写的 `"UFF"` 时才会规划本规则。

对相邻原子 $i-j-k$ 定义：

$$
\mathbf{u}=\mathbf{x}_i-\mathbf{x}_j,\qquad
\mathbf{v}=\mathbf{x}_k-\mathbf{x}_j,
$$

$$
s(i,j,k)=
\frac{\lVert\mathbf{u}\times\mathbf{v}\rVert}
     {\lVert\mathbf{u}\rVert\lVert\mathbf{v}\rVert}
=|\sin\theta_{ijk}|.
$$

候选必须满足：

- 中心 $j$ 不是金属；
- Open Babel 杂化状态不小于 2；
- 图上的度为 3 或 4；
- 两个向量长度均大于 $10^{-12}$；
- $s(i,j,k)\le\epsilon_s$，默认 $\epsilon_s=10^{-6}$；
- $i$ 或 $k$ 至少有一个不同于 $j$ 的邻接原子，使该角能够参与真正的四原子扭转；
- 删除与 $j$ 相连的对应键后，至少一侧能够成为可分离支链。

优先选择更小的可分离支链；大小相同时选择原子索引更小的根。候选按照 $(s,i,k,\text{支链根})$ 排序，以保证确定性。

选定支链围绕中心 $j$ 刚性旋转，默认 $\alpha=10^{-3}$ rad。旋转轴由与支链方向最不平行的笛卡尔基向量确定，并使用 Rodrigues 公式：

$$
\mathbf{r}'=\mathbf{r}\cos\alpha
+(\hat{\mathbf{a}}\times\mathbf{r})\sin\alpha
+\hat{\mathbf{a}}(\hat{\mathbf{a}}\cdot\mathbf{r})(1-\cos\alpha).
$$

因为整个可分离支链围绕中心进行同一次刚性旋转，其内部几何和键长保持不变。规划器会在一个中心上重复规划，直到不存在候选。

明确排除：

- 所有非 UFF 力场；
- 金属中心；
- 杂化状态小于 2 的中心；
- 度不是 3 或 4 的中心；
- 几何上不退化的角；
- 相邻向量长度为零；
- 不能参与真正 proper torsion 的角；
- 删除中心键后仍能从支链返回中心、因而无法分离的成环侧。

本规则不修改或钳制 UFF 的扭转导数。它只将严格奇异的起始几何移动一个很小且确定的角度；随后 bridge 检查 Open Babel 是否给出有限能量和有限梯度。

## Python 公开 API

### 函数

| 函数 | 作用 |
|---|---|
| `available_rules(stage: Optional[RuleStage] = None) -> tuple[RuleDescriptor, ...]` | 按确定性执行顺序返回编译入模块的规则，可按阶段筛选 |
| `build(obmol: ob.OBMol, *, builder: Optional[ob.OBBuilder] = None, stereo_warnings: Optional[bool] = None) -> BuildReport` | 规划临时构筑修改，只调用一次原生 `OBBuilder`，之后恢复临时杂化信息 |
| `prepare_optimization(obmol: ob.OBMol, forcefield: str, *, singularity_threshold: float = 1.0e-6, repair_angle_radians: float = 1.0e-3) -> OptimizationPreparationReport` | 在 `Setup()` 前应用确定性的 UFF 坐标预检修改；其他力场严格不操作 |
| `validate_forcefield_state(backend: ob.OBForceField, obmol: ob.OBMol) -> ForceFieldStateReport` | 计算带梯度能量，报告能量和梯度是否有限 |

### 公开值类型

所有报告类都是 frozen dataclass。

| 类型 | 字段及含义 |
|---|---|
| `RuleStage` | `PRE_BUILD`、`PRE_FORCEFIELD_SETUP` |
| `RuleDescriptor` | `rule_id`、`version`、`stage`、`priority` |
| `HybridizationChange` | 从零开始的 `atom_index`、`before`、`after` |
| `CoordinateChange` | 从零开始的 `atom_index`，以及以 Å 为单位的三分量 `before`、`after` 坐标 |
| `RuleApplication` | 描述符、目标原子索引、可选度量、杂化修改、坐标修改 |
| `RuleExecutionReport` | 阶段和有序应用；存在应用时 `.applied` 为真 |
| `BuildReport` | `succeeded` 与规则执行报告 |
| `OptimizationPreparationReport` | 力场名和规则报告；`.applied` 与规则报告一致 |
| `ForceFieldStateReport` | 后端原生单位的能量、有限性标志、从零开始的非有限梯度原子索引；能量与梯度均有限时 `.passed` 为真 |

### 稳定示例

列出编译注册表：

```python
from hotpot.cheminfo.obWrappers import available_rules

print([
    (rule.rule_id, rule.version, rule.stage.value, rule.priority)
    for rule in available_rules()
])
```

输出：

```text
[('tetracoordinate_pentavalent_phosphorus_build', '1.0.0', 'pre_build', 100), ('degenerate_nonlinear_torsion', '1.0.0', 'pre_forcefield_setup', 100)]
```

该顺序由 [`test_native_rules.py`](../../../tests/test_cheminfo/obWrappers/test_native_rules.py) 校验。

委托一个未命中的分子：

```python
from hotpot.cheminfo.obWrappers import build
from openbabel import openbabel as ob

conversion = ob.OBConversion()
conversion.SetInFormat("smi")
mol = ob.OBMol()
conversion.ReadString(mol, "CCO")
report = build(mol)
print(report.succeeded, report.rules.applied, len(report.rules.applications))
```

输出：

```text
True False 0
```

只调用一次原生构筑器的行为由 [`test_bridge.py`](../../../tests/test_cheminfo/obWrappers/test_bridge.py) 校验。

构筑一个匹配的 P(V) 分子，并验证临时杂化没有残留：

```python
from hotpot.cheminfo.obWrappers import build
from openbabel import openbabel as ob

conversion = ob.OBConversion()
conversion.SetInFormat("smi")
mol = ob.OBMol()
conversion.ReadString(
    mol,
    "CCOP(=O)(OCC)c1ccc2ccc3ccc(P(=O)(OCC)OCC)nc3c2n1",
)
before = [a.GetHyb() for a in ob.OBMolAtomIter(mol) if a.GetAtomicNum() == 15]
report = build(mol)
after = [a.GetHyb() for a in ob.OBMolAtomIter(mol) if a.GetAtomicNum() == 15]
print(report.succeeded)
print([item.descriptor.rule_id for item in report.rules.applications])
print(before, after)
```

输出：

```text
True
['tetracoordinate_pentavalent_phosphorus_build', 'tetracoordinate_pentavalent_phosphorus_build']
[5, 5] [5, 5]
```

拓扑、杂化恢复和不存在 P 中心线性几何的行为由 [`test_bridge.py`](../../../tests/test_cheminfo/obWrappers/test_bridge.py) 校验。

构筑后验证力场状态：

```python
mol.AddHydrogens()
build(mol)
backend = ob.OBForceField.FindForceField("UFF")
print(backend.Setup(mol))
state = validate_forcefield_state(backend, mol)
print(
    state.finite_energy,
    state.finite_gradients,
    state.nonfinite_gradient_atom_indices,
    state.passed,
)
```

输出：

```text
True
True True () True
```

[`test_bridge.py`](../../../tests/test_cheminfo/obWrappers/test_bridge.py) 对萃取剂数据中的全部含 P 样本校验了同一不变量。

## 原生 `_ob_rules` API

`_ob_rules` 是实施模块，不是稳定的最终用户 API；其类型接口记录在 `_ob_rules.pyi`。

| 原生符号 | 作用 |
|---|---|
| `AtomSnapshot(atomic_number, formal_charge, hybridization, is_metal)` | 不可变原子输入 |
| `BondSnapshot(begin, end, order, aromatic)` | 从零开始的不可变键输入 |
| `RuleStage` | 原生生命周期阶段枚举 |
| `RuleDescriptor` | 已注册规则的身份 |
| `HybridizationChange` | 规划的杂化修改 |
| `CoordinateChange` | 规划的坐标修改 |
| `RuleApplication` | 一次规则应用及其证据 |
| `RulePlan` | 某阶段的有序应用 |
| `available_rules(stage=None)` | 读取编译注册表 |
| `plan_build(atoms, bonds)` | 校验图快照并执行 PRE_BUILD 规则 |
| `plan_optimization(atoms, bonds, coordinates, singularity_threshold, repair_angle_radians)` | 校验坐标快照并执行 PRE_FORCEFIELD_SETUP 规则 |
| `RuleApplicationLimitExceeded` | 单个计划要求超过 256 次应用 |

原生规划器会校验：正原子序数、非负杂化状态、有效且非自连的键索引、非负键级、坐标数量匹配以及坐标有限。优化参数必须满足：

$$
0\le\epsilon_s<1,\qquad 0<\alpha<\pi,\qquad
|\sin\alpha|>\epsilon_s.
$$

## forcefields 接入点

生产代码仅有两个窄接入点：

- `forcefields/backend.py::_ob_build()` 在已有构筑器锁和力场锁内调用 `obWrappers.build()`；
- `forcefields/backend.py::_setup_forcefield_backend()` 在每次原生 `Setup()` 前调用 `prepare_optimization()`；若有规则应用，则在 Setup 后调用 `validate_forcefield_state()`，并将仍然非有限的状态转换为结构化 setup 错误。

`_single_ob_optimization()` 与状态化 `_OpenBabelOptimizer` 共享该 setup helper，所有公开力场函数都没有新增 wrapper 专用参数。

## 失败语义

- 无法导入 `_ob_rules` 时抛出 `ImportError`，提示安装兼容 wheel 或重新构建 Hotpot。
- 原生快照或数值设置无效时抛出 `ValueError`。
- 应用次数超过 256 时抛出 `RuleApplicationLimitExceeded`。
- 原生 builder 返回 `False` 时得到 `BuildReport(succeeded=False)`；forcefields 接入层将其转换为 `ForceFieldError`。
- 即使原生构筑抛出异常，临时杂化也会恢复。
- `prepare_optimization()` 会原位修改命中的坐标，独立调用时不承诺自动回滚；生产工作流仍在 working molecule 上运行，并保留既有的事务式提交行为。
- 原生 `Setup()` 失败仍为 `ForceFieldSetupError(stage="setup")`。
- 规则应用后状态仍非有限时为 `ForceFieldSetupError(stage="preflight-validation")`。
- 没有规则命中不是错误，也不产生警告。

## 并发与兼容性

C++ 规划器不持有 Open Babel 对象，只处理快照副本。模块初始化完成后，注册表仅被读取；原生规划期间 pybind11 会释放 GIL。

Open Babel 操作仍受其进程级状态限制。通过 `hotpot.cheminfo.forcefields` 发起的调用保留原有构筑器锁和力场锁。直接调用 `obWrappers.build()` 的用户不得并发修改同一个 `OBMol` 或 `OBBuilder`。

本包支持 Hotpot 的 Python 3.9--3.14 范围：

- Python 3.9 使用项目的 Open Babel 3.1 兼容 worker；
- Python 3.10--3.14 使用当前 Open Babel 3.2 路径；
- `_ob_rules` 只链接 Python/pybind11 和 C++ 标准库，不链接 `libopenbabel`，因此同一份规则源码可服务两种 Open Babel ABI；
- 扩展仍然依赖 CPython ABI，必须为每个受支持解释器分别构建。

## 构建

采用 editable 安装，让 setuptools 构建两个 pybind11 扩展：

```bash
$ python -m pip install -e .
```

开发阶段显式原位重建：

```bash
$ python setup.py build_ext --inplace --force
```

构建分发包：

```bash
$ python -m build
```

`_ob_rules` 在仓库根目录 `setup.py` 中声明，需要 C++17 编译器。wheel 必须包含与目标 CPython ABI 匹配的扩展。

## 测试证据与限制

专项验证命令：

```bash
$ python -m pytest -q -p no:cacheprovider \
    tests/test_cheminfo/obWrappers \
    tests/test_cheminfo/test_forcefield_integration.py
```

2026-09-28 在 Python 3.11、Open Babel 3.2.1 上的实际结果：

```text
36 passed in 24.71s
```

测试覆盖原生纳入/排除边界、计划确定性、记录只读、注册顺序、只调用一次原生构筑、临时元数据恢复、`molecules/extractant/extractants.smi` 中全部含 P 结构、UFF 有限梯度、非 UFF 严格不操作、公开力场集成、多进程和线程串行化。

此外，使用 16 个 worker、与 wrapper 前基线完全相同的随机种子和力场设置，重新运行了全部 187 个 Eu--萃取剂工作流：

| 结果 | 修复前基线 | 注册规则后 |
|---|---:|---:|
| 质量门控通过 | 167 / 187 | 175 / 187 |
| CBond 完成后质量失败 | 11 / 187 | 3 / 187 |
| 进入力场前 CBond 失败 | 9 / 187 | 9 / 187 |
| 非有限梯度案例 | 9 / 187 | 0 / 187 |
| 16 workers 墙钟时间 | 128.431 s | 138.753 s |

8 个原失败结构变为通过，没有原本通过的结构变为失败；9 个与 P(V) 有关的非有限梯度失败全部消失。其中 case 54 已得到有限 UFF 结果，但因某原子与 Eu 距离过近仍未通过质量门控，所以不计入 8 个新增通过案例。Case 61 和 case 109 保留原有的几何失败。力场累计时间增加 150.097 s（7.70%）；在前后均通过且不含 P 的 161 个案例中，单例耗时增量中位数为 0.708 s。这是当前可移植 SWIG 快照与规则规划边界的实测成本，不应把它描述成零开销抽象。

当前限制：

- 这是定向兼容层，不是通用分子修复引擎；
- 注册表可在编译期扩展，但 Python 端只读；
- 当前只有 PRE_BUILD 和 PRE_FORCEFIELD_SETUP 两个阶段；
- 只有预检规则实际应用后，Setup helper 才执行额外的有限状态校验；
- Open Babel 没有通过本接口暴露实际启用的 UFF torsion term 列表，因此扭转规则以图路径作为保守代理；
- 没有修改任何 UFF 公式或参数；
- 金属放置和配位塌缩（包括基准 case 109）仍由力场配位工作流负责。
