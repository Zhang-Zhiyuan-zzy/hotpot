# Open Babel 原生力场包装层

英文主文档：[README.md](README.md)

## 1. 目标与边界

`hotpot.cheminfo.obWrappers` 是 Hotpot 力场子系统调用 Open Babel 时使用
的直接原生边界。公开函数接受 Hotpot `Molecule`；调用者不需要、也不能
向这些接口传入 Open Babel `OBMol`。

Python 先将分子复制为具有明确 dtype、C 连续布局的 NumPy 数组；pybind11
扩展读取这些 buffer，在 C++ 内部构造临时 `OBMol`，应用已注册的后端修正
规则，并在 C++ 内完成 `OBBuilder` 或完整的力场 epoch 循环。返回 Python
的只有值对象：坐标、数值诊断和不可变规则报告。

本包当前提供：

- 基于原生 `OBBuilder` 的三维坐标构筑；
- 原生最速下降和共轭梯度优化；
- 可选的微扰分段、逐步增大的范德华截断、稳定性停止条件和受控轨迹记录；
- 只读的、针对明确 Open Babel 缺陷的规则注册表；
- 每次规则应用的可审计证据。

优化器按职责拆分，且不会在 epoch 循环中增加 Python 回调：

| 层次 | Python 入口 | 原生 C++ 入口 | 职责 |
|---|---|---|---|
| 优化操作 | `single_optimize(...)` | `optimization_operation.hpp` 中的 `OptimizationOperation` | 力场建立和直接优化操作 |
| 状态检查 | `check_optimization_state(...)` | `optimization_checks.hpp` 中的函数 | 能量、梯度、位移、有限性、爆炸和收敛事实 |
| 流程控制 | `optimize(...)` | `run_optimization_controller(...)` | epoch 编排、重启、微扰、选帧和终止 |

流程控制、优化操作和状态检查都在一次原生调用内完成，因此拆分没有增加
逐 epoch 的 Python/C++ 往返。注册的修正规则仍在原有的构筑前和力场建立前
节点执行；该包仍是 Open Babel 外围的可扩展缺陷修正层，而不是替代力场。

本包不实现新力场，不判断结构在化学上是否可接受，不处理环—键互穿、
配位键恢复或金属重定位。这些科学与流程策略属于上层
`hotpot.cheminfo.forcefields` 以及 geometry/chemistry 模块。

## 2. 应用代码应该调用哪一层？

一般 Hotpot 业务应调用高层力场包：

```python
from hotpot import read_mol
from hotpot.cheminfo import forcefields as ff


def main():
    mol = read_mol("CCO", "smi")
    report = ff.build_and_optimize(
        mol,
        forcefield="UFF",
        epochs=20,
        steps_per_epoch=25,
        seed=2026,
    )
    print(report.quality_report.passed)


if __name__ == "__main__":
    main()
```

高层 API 负责加氢、有机分子/络合物流程选择、环—键开结、配位键恢复、
几何验收、轨迹持久化、带随机种子的 worker 执行、警告以及对输入分子的
事务性更新。

只有在需要一项轻量 Open Babel 操作，或需要检查原生规则证据时，才直接
调用 `obWrappers`：

```python
from hotpot import read_mol
from hotpot.cheminfo.obWrappers import build, optimize

mol = read_mol("CCO", "smi")
build_report = build(mol)
optimization = optimize(
    mol,
    "UFF",
    epochs=20,
    steps_per_epoch=25,
    retain_frames=True,
)

print(build_report.succeeded)
print(optimization.best_energy, optimization.energy_unit)
print(optimization.termination_reason)
```

两项操作都会更新 `mol.coordinates`。`optimize()` 写入可用帧中能量最低的
选中帧；`terminal_coordinates` 则保留最后观测帧，供失败诊断使用。

## 3. 公开 API 总览

所有受支持的公开名称均由 `hotpot.cheminfo.obWrappers` 导出。编译扩展
`_ob_native` 和 buffer 打包器属于私有实现。

| API | 作用 | 返回值 |
|---|---|---|
| `build(mol, *, stereo_warnings=None)` | 通过原生 `OBBuilder` 构筑三维坐标，并应用构筑前规则 | `BuildReport` |
| `single_optimize(mol, forcefield, steps, *, ...)` | 执行一次独立的原生最速下降操作 | `SingleOptimizationReport` |
| `check_optimization_state(mol, forcefield, *, ...)` | 只读测量当前状态，不修改分子 | `OptimizationCheckReport` |
| `optimize(mol, forcefield, *, ...)` | 运行原生、按 epoch 组织的 Open Babel 优化 | `OptimizationReport` |
| `available_rules(stage=None)` | 按确定顺序列出编译期规则 | `tuple[RuleDescriptor, ...]` |
| `inspect_rules(mol, stage, *, ...)` | 不构筑、不优化，仅报告将命中的规则 | `RuleExecutionReport` |

### 3.1 `build`

```python
build(
    mol: Molecule,
    *,
    stereo_warnings: bool | None = None,
) -> BuildReport
```

`mol` 必须是 Hotpot `Molecule`。`stereo_warnings=None` 保留 Open Babel
默认行为；布尔值会传给对应的 `OBBuilder.Build` 重载。只有构筑器报告成功
时，坐标才会提交到 `mol`。

```python
from hotpot import read_mol
from hotpot.cheminfo.obWrappers import build

mol = read_mol("OP(=O)(O)O", "smi")
report = build(mol)
print(report.succeeded)
print([item.descriptor.rule_id for item in report.rules.applications])
```

### 3.2 `optimize`

```python
optimize(
    mol: Molecule,
    forcefield: str,
    *,
    algorithm: str = "conjugate",
    epochs: int = 100,
    steps_per_epoch: int = 5,
    perturb_interval: int | None = None,
    perturbation_offsets: numpy.ndarray | None = None,
    retain_frames: bool = False,
    retain_epoch_history: bool = False,
    increasing_vdw: bool = False,
    vdw_cutoff_start: float = 1.0,
    vdw_cutoff_end: float = 10.0,
    energy_tolerance: float = 1.0e-6,
    convergence_level: ConvergenceLevel = ConvergenceLevel.FAST,
    stopping_window: int | None = None,
    maximum_energy_change_kj_mol: float = 1.0e-4,
    maximum_atom_displacement_angstrom: float = 1.0e-4,
    maximum_rms_gradient_kj_mol_angstrom: float = 1.0,
    maximum_gradient_kj_mol_angstrom: float = 5.0,
    singularity_threshold: float = 1.0e-6,
    repair_angle_radians: float = 1.0e-3,
) -> OptimizationReport
```

重要参数：

| 参数 | 含义 |
|---|---|
| `forcefield` | Open Babel 力场插件名，例如可用环境中的 `"UFF"`、`"MMFF94"`、`"MMFF94s"` 或 `"GAFF"` |
| `algorithm` | 只能是 `"conjugate"` 或 `"steepest"` |
| `epochs`、`steps_per_epoch` | 优化预算上限；每个实际执行的 epoch 提交一个步长块 |
| `perturb_interval` | 每隔指定 epoch 加入微扰并启动新的优化 segment |
| `perturbation_offsets` | 连续数组，形状为 `(K, N, 3)`；`N` 是原子数，`K = (epochs - 1) // perturb_interval` |
| `retain_frames` | 将每个实际执行 epoch 作为 `OptimizationFrame` 返回 |
| `retain_epoch_history` | 独立于坐标帧保留标量能量历史 |
| `increasing_vdw` | 使用线性增大的范德华截断反复建立力场 segment |
| `convergence_level` | Open Babel 报告停止后所需的收敛证据；实测默认值为 `FAST` |
| `stopping_window` | 启用稳定窗口停止；`None` 表示不采用该额外停止规则 |
| `singularity_threshold`、`repair_angle_radians` | 退化扭转规则的数值参数，不是化学验收标准 |

请求微扰时，调用者必须显式提供 offsets。该薄包装层不会自行生成随机
微扰；高层 forcefields 工作流负责生成和记录微扰。

```python
import numpy as np

from hotpot import read_mol
from hotpot.cheminfo.obWrappers import build, optimize

mol = read_mol("CCCC", "smi")
build(mol)

epochs = 12
interval = 4
offsets = np.zeros(((epochs - 1) // interval, len(mol.atoms), 3))
report = optimize(
    mol,
    "UFF",
    algorithm="conjugate",
    epochs=epochs,
    steps_per_epoch=10,
    perturb_interval=interval,
    perturbation_offsets=offsets,
    retain_frames=True,
    retain_epoch_history=True,
    stopping_window=4,
)
print(report.epochs_completed, report.best_energy)
```

#### 收敛档位

所有档位都会执行硬性数值失败检查，包括非有限坐标、能量或梯度，以及
Open Babel 的结构爆炸检测。

| 数值 | 名称 | Open Babel 发出停止信号后所需的附加证据 |
|---:|---|---|
| 0 | `OPENBABEL` | 不附加梯度阈值；数值状态可用即可接受 Open Babel 停止信号 |
| 1 | `FAST`（默认） | RMS 梯度 <= 3、最大梯度 <= 10 kJ/(mol Å) |
| 2 | `BALANCED` | RMS 梯度 <= 1、最大梯度 <= 5 kJ/(mol Å) |
| 3 | `STRICT` | 最大梯度 <= 0.1 后端能量单位/Å；逐字保持此前的停止行为 |

基于相同三维起点的 187 个配体、181 个络合物配对基准选择了 `FAST`：相对
`STRICT`，配体累计优化时间减少 52.4%，络合物减少 38.0%。配体几何门控
保持 185/187；络合物由 166/181 变为 165/181（逐样本有 2 个退化、1 个
改善）。这对应“轻微可靠性妥协、显著缩短运行时间”的默认策略。需要完全
复现旧停止行为时应显式使用 `STRICT`。可运行基准及口径见
`tests/benchmarks/obwrapper_convergence/README.md`。

### 3.3 `single_optimize`

```python
single_optimize(
    mol: Molecule,
    forcefield: str,
    steps: int,
    *,
    singularity_threshold: float = 1.0e-6,
    repair_angle_radians: float = 1.0e-3,
) -> SingleOptimizationReport
```

该操作执行一个原生最速下降步块，应用已注册的力场建立前规则，并更新
`mol.coordinates`；它不运行 epoch 控制器或化学几何验收。

```python
from hotpot import read_mol
from hotpot.cheminfo.obWrappers import build, single_optimize

mol = read_mol("CCO", "smi")
build(mol)
report = single_optimize(mol, "UFF", 100)
print(report.energy, report.exploded)
```

### 3.4 `check_optimization_state`

```python
check_optimization_state(
    mol: Molecule,
    forcefield: str,
    *,
    previous_coordinates: numpy.ndarray | None = None,
    previous_energy_kj_mol: float | None = None,
    singularity_threshold: float = 1.0e-6,
    repair_angle_radians: float = 1.0e-3,
) -> OptimizationCheckReport
```

该只读操作返回能量、RMS/最大梯度、可选的能量/位移变化、坐标有限性、
爆炸状态和结构化数值失败分类。

```python
from hotpot.cheminfo.obWrappers import check_optimization_state

state = check_optimization_state(mol, "UFF")
print(state.energy, state.rms_gradient, state.failure.name)
```

### 3.5 `available_rules`

```python
available_rules(
    stage: RuleStage | None = None,
) -> tuple[RuleDescriptor, ...]
```

结果按 `(stage, priority, rule_id, version)` 排序。指定 stage 只进行过滤，
不改变排序规则。

```python
from hotpot.cheminfo.obWrappers import RuleStage, available_rules

for rule in available_rules(RuleStage.PRE_BUILD):
    print(rule.rule_id, rule.version, rule.priority)
```

### 3.6 `inspect_rules`

```python
inspect_rules(
    mol: Molecule,
    stage: RuleStage,
    *,
    singularity_threshold: float = 1.0e-6,
    repair_angle_radians: float = 1.0e-3,
) -> RuleExecutionReport
```

这是只读诊断操作：它构造相同的临时原生表示并执行规则规划，但不会运行
`OBBuilder`、不会建立力场，也不会修改 `mol`。

```python
from hotpot import read_mol
from hotpot.cheminfo.obWrappers import RuleStage, inspect_rules

mol = read_mol("OP(=O)(O)O", "smi")
report = inspect_rules(mol, RuleStage.PRE_BUILD)
print(report.applied)
for application in report.applications:
    print(application.descriptor.rule_id, application.atom_indices)
```

## 4. 报告与值契约

公开报告均为 frozen dataclass。坐标仍是 NumPy 数组，但报告属性与规则记录
不可重新赋值。

| 类型 | 含义 |
|---|---|
| `BuildReport` | 构筑成功标志和构筑前规则证据 |
| `OptimizationReport` | 选中/末帧坐标、帧、能量/梯度事实、预算、终止事实和全部 setup 规则证据 |
| `OptimizationFrame` | 一次实际执行 epoch 后记录的数值事实 |
| `OptimizationCheckReport` | 只读能量、梯度、位移、有限性、爆炸和失败事实 |
| `OptimizationFailure` | 结构化的数值状态分类 |
| `ConvergenceLevel` | 从 `OPENBABEL=0` 到 `STRICT=3` 的整数策略档位 |
| `RuleExecutionReport` | 某生命周期阶段的有序规则应用；非空时 `.applied` 为真 |
| `RuleApplication` | 一次规则应用的目标、度量、杂化变化和坐标变化 |
| `RuleDescriptor` | 稳定规则 ID、语义版本、阶段和优先级 |
| `HybridizationChange` | 原子索引以及变化前后的杂化值 |
| `CoordinateChange` | 原子索引以及变化前后的笛卡尔坐标 |
| `RuleStage` | `PRE_BUILD` 或 `PRE_FORCEFIELD_SETUP` |
| `BondKindCode` | 私有 typed-buffer 边界使用的稳定整数编码 |
| `SingleOptimizationReport` | Hotpot 内部短程最速下降路径使用的契约 |

`OptimizationReport` 暴露的能量单位统一为 `kJ/mol`，同时用
`backend_energy_unit` 保留 Open Babel 的原始单位字符串。主要字段包括：

- `coordinates`：可用帧中能量最低的选中帧，同时写入
  `mol.coordinates`；
- `terminal_coordinates`：最后观测帧，数值失败时仍会保留；
- `frames`：仅在 `retain_frames=True` 时保存 epoch 坐标帧；
- `selected_frame_index` 与 `best_epoch`：选中帧对应的实际执行 epoch；
- `final_energy` 与 `best_energy`：末帧能量与选中帧能量；
- `termination_reason`：`converged`、`stability_reached`、
  `budget_exhausted` 或显式数值失败原因；
- `terminal_converged`：Open Babel 是否在最后观测帧报告收敛；
- `rules`：所有原生优化 segment 的构筑力场前规则证据合集。

即使没有保留坐标帧，`selected_frame_index` 仍表示实际执行 epoch 的索引；
不要用它索引一个空的 `frames` 元组。

## 5. 原生边界与执行模型

```text
Hotpot Molecule
    |
    | Python：复制为有类型、C 连续的 NumPy 数组
    v
MoleculeData 值 buffer
    |
    | pybind11：校验并读取 buffer
    v
临时 C++ 值 -> 临时 Open Babel OBMol
    |
    | 注册规则 + OBBuilder / 力场 setup / epoch 循环
    v
C++ 结果值
    |
    | Python：不可变报告与选中坐标
    v
Hotpot Molecule
```

buffer schema 包括：

| Buffer | dtype 与形状 |
|---|---|
| 原子序数 | `int32[N]` |
| 形式电荷 | `int32[N]` |
| 部分电荷 | `float64[N]` |
| 坐标 | `float64[N, 3]` |
| 原子芳香性标志 | `uint8[N]` |
| 键端点原子行号 | `int32[M, 2]` |
| 键级 | `float64[M]` |
| 键类型编码 | `uint8[M]` |
| 键芳香性标志 | `uint8[M]` |
| 可选晶胞 | `float64[6]` |

本版本明确不设计 `HpMol`，也不提供其他用于取代 `hotpot.Molecule` 的
C++ 对象。`MoleculeData` 只是瞬态传输对象，不是公开化学对象模型。
SWIG 对象或 `OBMol` 指针不会跨越 pybind11 边界，力场每个 epoch 也不会
重新进入 Python。

Open Babel 使用进程级全局插件和力场状态，因此本包通过内部递归互斥锁串行
执行同一进程中的原生操作。大量独立分子仍应使用进程级并行 worker。

## 6. 内置规则

### 6.1 四配位 P(V) 构筑保护

```text
rule_id  = tetracoordinate_pentavalent_phosphorus_build
version  = 1.0.0
stage    = PRE_BUILD
priority = 100
```

规则只匹配中性四配位 P(V)：一根非芳香 P=O 或 P=S 键，加三根非芳香
单键。调用 `OBBuilder` 时，它临时将目标中心的杂化值从 5 改为 3，随后
恢复原生分子状态。公开报告记录受影响原子和临时变化；形式电荷、拓扑和
Hotpot 键级不会被改写。

### 6.2 退化非线性扭转保护

```text
rule_id  = degenerate_nonlinear_torsion
version  = 1.0.0
stage    = PRE_FORCEFIELD_SETUP
priority = 100
```

对于相邻原子 $i-j-k$，规则计算

$$
s(i,j,k)=
\frac{\lVert(\mathbf{x}_i-\mathbf{x}_j)\times
(\mathbf{x}_k-\mathbf{x}_j)\rVert}
{\lVert\mathbf{x}_i-\mathbf{x}_j\rVert
 \lVert\mathbf{x}_k-\mathbf{x}_j\rVert}
=|\sin\theta_{ijk}|.
$$

若参与 proper torsion 的拓扑严格或近似共线，默认
$s\leq10^{-6}$，规则会在力场 setup 前将一个可分离支链旋转
`1.0e-3` rad，以避开已知的 UFF 扭转梯度非有限输入，并显式记录该变化。
它不修改 UFF 能量公式，也不是通用几何修复算法。

## 7. 扩展原生包装框架

框架将扩展点隔离开来，使未来能够针对新的、与力场直接相关的后端病态
行为增加处理，而不重新引入 Python/SWIG 控制循环：

1. 在 `_native/` 下新增一个职责单一的 condition/action 翻译单元；
2. 分配稳定的 `RuleDescriptor`，并用 `RuleRegistrar` 注册
   `RuleDefinition`；
3. 将源文件加入 `_ob_native` 扩展构建；
4. 补充阳性、排除项、执行顺序和边界测试；
5. 在默认启用前，用真实构筑/力场流程及相关 benchmark 证明其作用。

规则按 `(stage, priority, rule_id, version)` 确定性执行。后续规则能够看到
前序规则对私有原生快照的改动。重复 `(stage, rule_id)` 注册以及单次计划
超过 256 个 application 都会被拒绝。Python 只开放注册表检查，不开放
运行期规则注册。

condition/action 必须保持狭窄：它们可以为已经证实的 Open Babel 输入病态
状态做准备并报告修改，但不能决定结构在化学上是否可接受、选择全局力场
工作流、吞掉后端失败或成为无条件兜底。

## 8. 可复现性与局限

- `build()` 有意不提供 seed。Open Babel 3.2 的 builder RNG 状态无法在
  同一进程内可靠重置，因此不能假设重复直接调用会逐位一致。需要可复现
  worker 执行时，请调用带 `seed=` 的高层 forcefields API。
- seed 不保证跨 Open Babel 版本、编译器、CPU 或平台得到相同浮点坐标。
- `optimize()` 选择已观测可用帧中的最低能帧；收敛与化学验收是不同事实。
  接受结构前应查看高层几何验收报告。
- 数值失败会保留末帧坐标并返回显式 `termination_reason`，不会静默伪装成
  成功。
- 该边界拒绝 Hotpot dative bond，因为 Open Babel 无法无损表达其语义。
  配位体系需要的临时拓扑变换由高层络合物工作流负责。
- 原生扩展必须匹配编译时 Open Babel ABI；应安装相容 wheel，或在目标环境
  中重新构建 Hotpot。
- 本包装层只处理两个已有证据的后端缺陷，不能据此声称 UFF 普遍适用于
  配位络合物。

## 9. 包结构

```text
hotpot/cheminfo/obWrappers/
├── __init__.py                 # 受支持的公开导出
├── builder.py                  # Hotpot Molecule 构筑 facade
├── operation.py                # 独立优化操作 facade
├── checks.py                   # 独立只读检查 facade
├── forcefield.py               # 完整优化流程控制 facade
├── registry.py                 # 只读规则检查
├── contracts.py                # 不可变公开报告
├── reports.py                  # 原生报告到公开报告的转换
├── packing.py                  # 私有 typed NumPy buffer schema
├── native.py                   # 延迟加载原生模块并传递 buffer
├── settings.py                 # 规则数值默认值
├── _ob_native.pyi              # 私有扩展类型接口
└── _native/
    ├── molecule_data.*         # 瞬态 C++ 值 buffer
    ├── openbabel_adapter.*     # 值 buffer <-> 临时 OBMol
    ├── native_engine.*         # 运行时 facade 与事务边界
    ├── optimization_operation.* # 力场优化操作原语
    ├── optimization_checks.*    # 数值测量与判定
    ├── optimization_controller.* # epoch 级流程控制
    ├── rules.hpp               # 原生规则/报告契约
    ├── registry.*              # 确定性编译期注册表
    ├── phosphorus_builder.cpp  # P(V) 构筑保护
    ├── degenerate_torsion.cpp  # 扭转奇异性保护
    └── native_bindings.cpp     # pybind11 模块
```
